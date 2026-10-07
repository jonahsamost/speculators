"""Load released DeepSeek-V4-Flash-DSpark draft weights into our clean-room model.

The release stores the draft under the ``mtp.*`` namespace (3 stages), plus the
shared ``embed`` / ``head`` it ties to the frozen target. Every weight is fp8
(experts fp4) with a companion ``.scale`` tensor; loading into our bf16 model
dequantizes ``weight * scale`` (the ``.scale`` keys are consumed there, not
mapped to a parameter).

This module provides:

* :func:`map_released_key` — released key -> our parameter key (or ``None`` when
  the key is a ``.scale`` sidecar, a base ``layers.*`` layer, or a base-model-only
  tensor our draft doesn't carry).
* :func:`expected_draft_keys` — the parameter keys our :class:`DSparkDraftModel`
  exposes, built analytically from the config (no torch needed) for verification.
* :func:`verify_mapping` — check the release↔ours key bijection from a safetensors
  index (structural "can we load it" check; no download, no dequant).
* :func:`load_released_draft` — the real loader (maps + dequantizes into the model).

Naming already lines up (we adopted the official ``wq_a`` / ``attn_sink`` /
``experts.i.w{1,2,3}`` names); the only renames are ``ffn.gate`` -> ``ffn.router``,
``hc_attn_*`` -> ``attn_hc.*``, ``hc_ffn_*`` -> ``ffn_hc.*``, and the stage-0/2
extras (``main_proj`` / ``norm`` / ``markov_head`` / ``confidence_head``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .config import DSparkDraftConfig

# Which mtp stage owns the "extra" (non-per-layer) parts.
_STAGE0 = 0  # main_proj / main_norm
# stage last (n_draft_layers - 1) owns: norm, markov_head, confidence_head

_HC_SITE = {"hc_attn": "attn_hc", "hc_ffn": "ffn_hc"}


def map_released_key(key: str, n_draft_layers: int = 3) -> str | None:
    """Map a released checkpoint key to our parameter key, or ``None`` to skip.

    ``None`` is returned for: ``.scale`` sidecars (folded into dequant), base
    ``layers.*`` decoder layers (target-only), and the base model's own
    ``norm`` / ``hc_head_*``. V4.1 has no standalone terminal HC head.
    """
    if key.endswith(".scale"):
        return None
    if key.startswith("layers."):
        return None  # base 43-layer target — not part of the draft
    # ---- shared with the frozen target ----
    if key == "embed.weight":
        return "embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    # base-model-only tensors (the draft's equivalents come from mtp.{last}.*)
    if key in ("norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
        return None

    m = re.match(r"^mtp\.(\d+)\.(.*)$", key)
    if not m:
        return None
    stage, rest = int(m.group(1)), m.group(2)

    # ---- stage-0 extras (target-hidden conditioning) -> model level ----
    if rest == "main_proj.weight":
        return "fc.weight"
    if rest == "main_norm.weight":
        return "hidden_norm.weight"

    # ---- last-stage extras (output head) -> model level ----
    if rest == "norm.weight":
        return "norm.weight"
    if rest == "markov_head.embed.weight":
        return "markov_head.markov_w1.weight"
    if rest == "markov_head.head.weight":
        return "markov_head.markov_w2.weight"
    if rest == "confidence_head.proj.weight":
        return "confidence_head.proj.weight"
    hh = re.match(r"^hc_head_(fn|base|scale)$", rest)
    if hh:
        return None

    # ---- per-layer block parts ----
    hc = re.match(r"^hc_(attn|ffn)_(fn|base|scale)$", rest)
    if hc:
        return f"layers.{stage}.{_HC_SITE['hc_' + hc.group(1)]}.{hc.group(2)}"
    if rest.startswith("ffn.gate."):
        if rest.endswith(".bias_vl"):
            return None
        return f"layers.{stage}.ffn.router.{rest[len('ffn.gate.') :]}"
    if rest.startswith(
        ("attn.", "attn_norm.", "ffn_norm.", "ffn.experts.", "ffn.shared_experts.")
    ):
        return f"layers.{stage}.{rest}"
    return None


def expected_draft_keys(cfg: DSparkDraftConfig) -> set[str]:
    """The parameter keys our DSparkDraftModel exposes (built from config)."""
    keys: set[str] = {
        "embed_tokens.weight",
        "lm_head.weight",
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
    }
    for n in range(cfg.n_draft_layers):
        p = f"layers.{n}."
        keys |= {
            p + "attn.wq_a.weight",
            p + "attn.q_norm.weight",
            p + "attn.wq_b.weight",
            p + "attn.wkv.weight",
            p + "attn.kv_norm.weight",
            p + "attn.wo_a.weight",
            p + "attn.wo_b.weight",
            p + "attn.attn_sink",
            p + "attn_norm.weight",
            p + "ffn_norm.weight",
            p + "ffn.router.weight",
            p + "ffn.router.bias",
            p + "ffn.shared_experts.w1.weight",
            p + "ffn.shared_experts.w2.weight",
            p + "ffn.shared_experts.w3.weight",
        }
        for site in ("attn_hc", "ffn_hc"):
            keys |= {p + f"{site}.fn", p + f"{site}.base", p + f"{site}.scale"}
        for e in range(cfg.n_routed_experts):
            keys |= {
                p + f"ffn.experts.{e}.w1.weight",
                p + f"ffn.experts.{e}.w2.weight",
                p + f"ffn.experts.{e}.w3.weight",
            }
    return keys


def expected_draft_shapes(cfg: DSparkDraftConfig) -> dict[str, list[int]]:
    """Expected parameter shapes (as ``nn.Linear`` weight ``[out, in]`` etc.).

    Analytic (no torch) so it can be diffed against a released safetensors
    header. Matches the module definitions in :mod:`.backbone` / :mod:`.draft`.
    """
    H, V = cfg.hidden_size, cfg.vocab_size
    hd, nh = cfg.head_dim, cfg.num_heads
    qlr, olr, og = cfg.q_lora_rank, cfg.o_lora_rank, cfg.o_groups
    mi, ne, mr = cfg.moe_inter_dim, cfg.n_routed_experts, cfg.markov_rank
    hc = cfg.hc_mult
    mix = (2 + hc) * hc
    s: dict[str, list[int]] = {
        "embed_tokens.weight": [V, H],
        "lm_head.weight": [V, H],
        "fc.weight": [H, H * cfg.num_target_layers],
        "hidden_norm.weight": [H],
        "norm.weight": [H],
        "markov_head.markov_w1.weight": [V, mr],
        "markov_head.markov_w2.weight": [V, mr],
        "confidence_head.proj.weight": [1, H + mr],
    }
    for n in range(cfg.n_draft_layers):
        p = f"layers.{n}."
        s |= {
            p + "attn.wq_a.weight": [qlr, H],
            p + "attn.q_norm.weight": [qlr],
            p + "attn.wq_b.weight": [nh * hd, qlr],
            p + "attn.wkv.weight": [hd, H],
            p + "attn.kv_norm.weight": [hd],
            p + "attn.wo_a.weight": [og * olr, nh * hd // og],
            p + "attn.wo_b.weight": [H, og * olr],
            p + "attn.attn_sink": [nh],
            p + "attn_norm.weight": [H],
            p + "ffn_norm.weight": [H],
            p + "ffn.router.weight": [ne, H],
            p + "ffn.router.bias": [ne],
        }
        for w, out in (("w1", mi), ("w2", H), ("w3", mi)):
            in_ = H if w != "w2" else mi
            s[p + f"ffn.shared_experts.{w}.weight"] = [out, in_]
            for e in range(ne):
                s[p + f"ffn.experts.{e}.{w}.weight"] = [out, in_]
        for site in ("attn_hc", "ffn_hc"):
            s[p + f"{site}.fn"] = [mix, hc * H]
            s[p + f"{site}.base"] = [mix]
            s[p + f"{site}.scale"] = [3]
    return s


# Released quant dtypes: attn/shared linears are fp8 (1 byte/value, unpacked);
# experts are fp4 packed 2-per-byte (stored as I8), so their last dim is halved.
_FP4_DTYPES = {"I8", "U8", "F4_E2M1", "F4", "FP4"}


def verify_shapes(released: dict, cfg: DSparkDraftConfig) -> dict:
    """Check released tensor shapes against ours (fp4 experts unpacked ×2).

    ``released`` maps checkpoint key -> ``{"shape": [...], "dtype": "..."}``
    (e.g. parsed from safetensors headers). Skips ``.scale`` sidecars. Returns
    a report with ``ok`` and any ``mismatches``.
    """
    exp = expected_draft_shapes(cfg)
    checked = 0
    mismatches: list[tuple] = []
    for rk, info in released.items():
        if rk.endswith(".scale"):
            continue
        tgt = map_released_key(rk, cfg.n_draft_layers)
        if tgt is None or tgt not in exp:
            continue
        shape = list(info["shape"])
        if info.get("dtype") in _FP4_DTYPES and len(shape) == 2:
            shape = [shape[0], shape[1] * 2]  # unpack fp4 nibble packing
        checked += 1
        if shape != list(exp[tgt]):
            mismatches.append((rk, tgt, info["shape"], info.get("dtype"), exp[tgt]))
    return {"ok": not mismatches, "checked": checked, "mismatches": mismatches}


def verify_mapping(released_keys, cfg: DSparkDraftConfig) -> dict:
    """Check the release↔ours key bijection (no torch, no download).

    Returns a report dict with ``ok`` and the offending sets. ``released_keys``
    is any iterable of checkpoint tensor names (e.g. an index's weight_map keys).
    """
    expected = expected_draft_keys(cfg)
    mapped: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for k in released_keys:
        tgt = map_released_key(k, cfg.n_draft_layers)
        if tgt is None:
            continue
        if tgt in mapped:
            collisions.append((tgt, mapped[tgt], k))
        mapped[tgt] = k
    mapped_targets = set(mapped)
    unfilled = expected - mapped_targets  # our params with no release source
    unexpected = mapped_targets - expected  # release keys mapping to nothing we have
    return {
        "ok": not unfilled and not unexpected and not collisions,
        "num_expected": len(expected),
        "num_mapped": len(mapped_targets),
        "unfilled": sorted(unfilled),
        "unexpected": sorted(unexpected),
        "collisions": collisions,
    }


def _e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    """Decode exact powers-of-two stored as float8-e8m0 bytes."""
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def _encode_e8m0_scales(
    minimum_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Encode power-of-two scales which are at least ``minimum_scale``.

    Rounding upward prevents a finite source value from overflowing the FP8 or
    MXFP4 payload. E8M0 is stored as the IEEE-754 exponent byte, so the encoded
    value is simply ``exponent + 127``.
    """
    safe = torch.where(
        minimum_scale > 0, minimum_scale.float(), torch.ones_like(minimum_scale.float())
    )
    # Byte 0 decodes to IEEE zero in the serving kernels, so the smallest
    # non-zero scale is exponent byte 1 (2**-126). Byte 255 is reserved.
    exponent = torch.ceil(torch.log2(safe)).to(torch.int32).clamp(-126, 127)
    encoded = (exponent + 127).to(torch.uint8)
    return encoded.view(dtype)


def quantize_released_weight(
    weight: torch.Tensor,
    template_weight: torch.Tensor,
    template_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP32 weights into a released V4.1 tensor layout.

    The released tensors are deliberately used as the schema: their dtype and
    scale geometry decide whether this is 32x32 FP8 or row-wise 32-value
    packed MXFP4. This makes export fail loudly if a future release changes its
    representation instead of silently producing a checkpoint vLLM misreads.
    """
    source = weight.float()
    if template_weight.dtype == torch.float8_e4m3fn:
        if source.ndim != 2 or template_scale.ndim != 2:
            raise ValueError("FP8 DSpark weights and scales must be 2-D")
        rows, cols = source.shape
        scale_rows, scale_cols = template_scale.shape
        if rows % scale_rows or cols % scale_cols:
            raise ValueError(
                f"cannot infer exact FP8 tiles for weight {tuple(source.shape)} "
                f"and scale {tuple(template_scale.shape)}"
            )
        row_block, col_block = rows // scale_rows, cols // scale_cols
        if (row_block, col_block) != (32, 32):
            raise ValueError(
                f"expected released FP8 32x32 tiles, got {row_block}x{col_block}"
            )
        blocks = source.view(scale_rows, row_block, scale_cols, col_block)
        amax = blocks.abs().amax(dim=(1, 3))
        scales = _encode_e8m0_scales(
            amax / torch.finfo(template_weight.dtype).max, template_scale.dtype
        )
        expanded = (
            _e8m0_scales(scales).repeat_interleave(32, 0).repeat_interleave(32, 1)
        )
        quantized = (source / expanded).to(template_weight.dtype)
        return quantized, scales

    if template_weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"unsupported released DSpark dtype: {template_weight.dtype}")
    if source.ndim != 2 or template_scale.ndim != 2:
        raise ValueError("MXFP4 DSpark weights and scales must be 2-D")
    rows, cols = source.shape
    if cols % 32 or tuple(template_scale.shape) != (rows, cols // 32):
        raise ValueError(
            f"expected row-wise MXFP4 groups of 32 for weight {tuple(source.shape)}, "
            f"got scale {tuple(template_scale.shape)}"
        )
    groups = source.view(rows, cols // 32, 32)
    amax = groups.abs().amax(dim=-1)
    scales = _encode_e8m0_scales(amax / 6.0, template_scale.dtype)
    normalized = groups / _e8m0_scales(scales).unsqueeze(-1)
    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=normalized.device,
    )
    codes = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1).to(torch.uint8)
    codes = codes.view(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.view(template_weight.dtype), scales


def dequantize_released_weight(
    weight: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize a released V4.1 FP8 or packed MXFP4 matrix on CPU.

    FP8 dense/shared-expert matrices use one scale per 32x32 tile. Routed
    expert matrices use packed E2M1 values (two nibbles per byte) with one
    scale per row and 32 unpacked columns.
    """
    decoded_scale = _e8m0_scales(scale)
    if weight.dtype == torch.float8_e4m3fn:
        if weight.ndim != 2 or decoded_scale.ndim != 2:
            raise ValueError("FP8 DSpark weights require 2-D weight and scale tensors")
        expanded = decoded_scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
        if tuple(expanded.shape) != tuple(weight.shape):
            raise ValueError(
                f"FP8 scale shape {tuple(scale.shape)} does not tile weight "
                f"shape {tuple(weight.shape)} by 32x32"
            )
        return (weight.float() * expanded).to(dtype)

    if weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"unsupported quantized DSpark dtype: {weight.dtype}")
    packed = weight.view(torch.uint8)
    unpacked = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = (packed >> 4) & 0x0F
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    values = table[unpacked.long()]
    expanded = decoded_scale.repeat_interleave(32, dim=-1)
    if tuple(expanded.shape) != tuple(values.shape):
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale.shape)} does not tile unpacked "
            f"weight shape {tuple(values.shape)} by 32 columns"
        )
    return (values * expanded).to(dtype)


def load_released_state_dict(
    checkpoint_dir: str | Path,
    cfg: DSparkDraftConfig,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load and dequantize only the native ``mtp.*`` draft tensors.

    Routed experts are assembled into the stacked layout used by
    :class:`GroupedExperts`. Frozen verifier-owned embedding/head/norm weights
    are intentionally omitted and are reconstructed from the verifier when the
    converted checkpoint is loaded for training.
    """
    root = Path(checkpoint_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing checkpoint index: {index_path}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    native_keys = {
        key
        for key in weight_map
        if key.startswith("mtp.") and not key.endswith(".bias_vl")
    }
    shards: dict[str, list[str]] = {}
    for key in native_keys:
        shards.setdefault(weight_map[key], []).append(key)

    native: dict[str, torch.Tensor] = {}
    for shard, keys in shards.items():
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                native[key] = handle.get_tensor(key)

    converted: dict[str, torch.Tensor] = {}
    experts: dict[tuple[int, str], dict[int, torch.Tensor]] = {}
    expert_re = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
    for key, tensor in native.items():
        if key.endswith(".scale"):
            continue
        target = map_released_key(key, cfg.n_draft_layers)
        match = expert_re.match(key)
        if match:
            stage, expert, projection = (
                int(match.group(1)),
                int(match.group(2)),
                match.group(3),
            )
            scale = native.get(key.removesuffix(".weight") + ".scale")
            if scale is None:
                raise ValueError(f"missing quantization scale for {key}")
            value = dequantize_released_weight(tensor, scale, dtype=dtype)
            experts.setdefault((stage, projection), {})[expert] = value
            continue
        if target is None:
            continue
        scale = (
            native.get(key.removesuffix(".weight") + ".scale")
            if key.endswith(".weight")
            else None
        )
        if scale is not None:
            tensor = dequantize_released_weight(tensor, scale, dtype=dtype)
        elif tensor.is_floating_point():
            tensor = tensor.to(
                dtype if tensor.dtype != torch.float32 else torch.float32
            )
        converted[target] = tensor

    for (stage, projection), by_expert in experts.items():
        expected = set(range(cfg.n_routed_experts))
        if set(by_expert) != expected:
            missing = sorted(expected - set(by_expert))
            raise ValueError(
                f"stage {stage} {projection} is missing routed experts: {missing[:8]}"
            )
        converted[f"layers.{stage}.ffn.experts.{projection}"] = torch.stack(
            [by_expert[index] for index in range(cfg.n_routed_experts)]
        )

    return converted
