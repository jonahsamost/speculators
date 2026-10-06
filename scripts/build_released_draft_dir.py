#!/usr/bin/env python3
"""Assemble a STANDALONE vllm-ascend draft dir from the RELEASED fp8 DSpark model.

The released `DeepSeek-V4-Flash-DSpark` ships the draft BUNDLED into the 48-shard fp8 model:
the draft is the `mtp.*` tensors (last 3 shards, fp8 e4m3 block-quant + `.scale`), and it SHARES
the target's top-level `embed.weight` (shard 1) + `head.weight` (shard 45) — it has no embed/head
of its own. vllm's draft `load_weights` wants exactly {mtp.* , embed.weight , head.weight}.

This builds that standalone dir WITHOUT needing shards 1/45: it takes `mtp.*` (fp8, verbatim — no
dequant, pure byte copy) from the released model, and BORROWS `embed.weight` + `head.weight` from
our already-serving converted bf16 draft dir (identical base model / vocab, embed is unquantized).
config.json = our proven-good bf16 draft config + the released fp8 `quantization_config` grafted in
(the ONLY change needed so vllm loads the mtp linears as fp8; embed/head stay bf16, no scale => skipped).

  # bf16 target (current stack — the new vllm-ascend build makes the draft bf16): DEQUANT fp8->bf16
  python scripts/build_released_draft_dir.py --dequant-bf16 \
      --released   /path/DeepSeek-V4-Flash-DSpark     `# has shards w/ mtp.* + config.json + index` \
      --our-draft  /share/.../ckpt_faithful_ep1_vllm  `# our converted bf16 draft (embed/head source)` \
      --out        /share/.../released_draft_bf16_standalone

  # fp8 target (legacy): copy mtp fp8 verbatim + graft fp8 quant_config (drop --dequant-bf16)

Serve it on our bf16 harness (isolates: known-good draft, our exact target + aux path):
  DRAFT=<out> MODEL=/share/.../DeepSeek-V4-Flash-bf16 NUM_SPEC=5 bash serve_dsv4_bf16_dualnode.sh head
  (put <out> on /share so BOTH nodes see it, like the converted draft.)

Runs on CPU. --dequant-bf16 does fp8->bf16 arithmetic (block-fp8 w*scale, 128-block; the proven
DeepSeek/Ascend math). Needs torch (>=2.1 for float8_e4m3fn) + safetensors (>=0.4).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch


# --- proven DeepSeek/Ascend fp8->bf16 dequant (verbatim from convert_model_flash.py, which is
#     adapted from ModelZoo DeepSeek-V2 fp8_cast_bf16.py). This is the SAME math that produced
#     DeepSeek-V4-Flash-bf16, so the dequant'd draft matches the bf16 target byte-for-byte scheme.
def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128,
                   is_mx: bool = False) -> torch.Tensor:
    """Dequant block-quant `weight` [M,N] by `scale` (block_size x block_size blocks) -> default dtype."""
    M, N = weight.shape
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)
    if is_mx:
        scale_expanded = scale.repeat_interleave(block_size, dim=1)
    else:
        scale_m, scale_n = scale.shape
        assert scale_m == (M + block_size - 1) // block_size, "Mismatch in scale rows vs weight rows."
        assert scale_n == (N + block_size - 1) // block_size, "Mismatch in scale cols vs weight cols."
        scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:M, :N]
    return (weight * scale_expanded).to(torch.get_default_dtype())


def unpack_mxfloat4_to_fp32(packed_tensor: torch.Tensor) -> torch.Tensor:
    """Unpack a uint8 tensor holding two MX-fp4 (e2m1) values per byte into fp32 (last dim doubled)."""
    e2m1_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32, device=packed_tensor.device)
    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F
    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)
    fp32_tensor = e2m1_values[unpacked.long()]
    new_shape = list(packed_tensor.shape)
    new_shape[-1] = new_shape[-1] * 2
    return fp32_tensor.view(*new_shape)


def _load_index(d: Path) -> dict:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    # single-file fallback: map every key to the one file
    from safetensors import safe_open  # noqa: PLC0415

    single = d / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise SystemExit(f"!! no model.safetensors[.index.json] in {d}")


def _tensor(d: Path, wm: dict, key: str):
    """Load a single tensor by key using the dir's weight_map (verbatim, no cast)."""
    from safetensors.torch import load_file  # noqa: PLC0415

    if key not in wm:
        raise SystemExit(f"!! '{key}' not found in {d} (keys sample: {list(wm)[:3]})")
    sd = load_file(str(d / wm[key]))  # small: only the shard holding `key`
    return sd[key]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--released", required=True, help="released fp8 DSpark model dir (mtp.* source)")
    ap.add_argument("--our-draft", required=True, dest="ours",
                    help="our converted bf16 draft dir (embed.weight + head.weight source)")
    ap.add_argument("--out", required=True, help="output standalone draft dir")
    ap.add_argument("--ignore-embed-head", action=argparse.BooleanOptionalAction, default=True,
                    help="add embed/head to the fp8 quant_config's `ignored_layers` so vllm does NOT "
                         "try to fp8-quantize the borrowed bf16 embed/head (they have no .scale). "
                         "Default ON (pre-empts the fp8 load hiccup); --no-ignore-embed-head to disable.")
    ap.add_argument("--dequant-bf16", action=argparse.BooleanOptionalAction, default=False,
                    help="DEQUANT the released fp8 mtp.* to bf16 (block-fp8: w*scale, 128-block) instead "
                         "of copying fp8 verbatim, and DROP quantization_config. Use when the target is "
                         "bf16 — the new vllm-ascend build binds draft precision to the target's "
                         "(bf16 target => draft built bf16, so it needs bf16 mtp weights, no .scale). "
                         "Default OFF (fp8 verbatim, for an fp8 target).")
    args = ap.parse_args()

    from safetensors.torch import save_file  # noqa: PLC0415

    rel, ours, out = Path(args.released), Path(args.ours), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rel_wm = _load_index(rel)
    ours_wm = _load_index(ours)

    # --- 1) collect the fp8 mtp.* tensors (verbatim), loading each source shard ONCE ---
    mtp_keys = sorted(k for k in rel_wm if k.startswith("mtp."))
    if not mtp_keys:
        raise SystemExit(f"!! no mtp.* keys in {rel}/index — is this the released DSpark model?")
    shards = sorted({rel_wm[k] for k in mtp_keys})
    print(f">>> released: {len(mtp_keys)} mtp.* tensors across {len(shards)} shard(s): {shards}")
    from safetensors.torch import load_file  # noqa: PLC0415

    raw: dict = {}
    for sh in shards:
        sd = load_file(str(rel / sh))
        for k, v in sd.items():
            if k.startswith("mtp."):
                raw[k] = v
    n_scale = sum(k.endswith(".scale") for k in raw)
    print(f"    loaded {len(raw)} mtp.* ({n_scale} .scale => fp8 block-quant)")

    merged: dict = {}
    if args.dequant_bf16:
        # DEQUANT fp8 block-quant -> bf16 with the proven DeepSeek math (weight_dequant above).
        # bf16 target => the new vllm-ascend build makes the draft bf16, so it must get bf16 mtp
        # weights and NO .scale. Free each source tensor after use to keep peak RAM ~one copy.
        torch.set_default_dtype(torch.bfloat16)
        n_deq = n_pass = 0
        stats = []
        for k in [w for w in raw if not w.endswith(".scale")]:
            v = raw[k]
            if v.element_size() == 1:  # fp8 (or int8/fp4-packed) block-quant weight
                sk = k.replace(".weight", ".scale")
                s = raw.get(sk)
                if s is None:
                    raise SystemExit(f"!! fp8 weight '{k}' has no scale '{sk}' in released mtp shards")
                if v.dtype == torch.int8:                      # MX fp4, block 32
                    w = unpack_mxfloat4_to_fp32(v.view(torch.uint8))
                    w = weight_dequant(w, s, block_size=32, is_mx=True)
                else:                                          # fp8 e4m3, block 128
                    w = weight_dequant(v, s, block_size=128)
                merged[k] = w.to(torch.bfloat16)
                raw.pop(sk, None)                              # free the scale
                n_deq += 1
                if len(stats) < 4:
                    stats.append((k, float(merged[k].abs().max()), tuple(merged[k].shape)))
            else:
                merged[k] = v                                  # bf16 norm/bias — verbatim
                n_pass += 1
            raw.pop(k, None)                                   # free the source weight
        print(f"    dequant: {n_deq} fp8->bf16, {n_pass} passthrough (bf16 draft, no .scale)")
        for name, amax, shp in stats:                          # magnitude sanity gate
            flag = "" if 1e-4 < amax < 1e4 else "  !! OUT-OF-RANGE — dequant may be inverted/wrong"
            print(f"      {name}: |max|={amax:.4g} shape={shp}{flag}")
    else:
        merged = dict(raw)  # verbatim fp8 (+ .scale) — original behavior (fp8 target)
        print(f"    kept {len(merged)} mtp.* verbatim (fp8 + .scale)")

    # --- 2) borrow embed.weight + head.weight from our bf16 draft ---
    for k in ("embed.weight", "head.weight"):
        merged[k] = _tensor(ours, ours_wm, k)
        print(f"    borrowed {k}: {tuple(merged[k].shape)} {merged[k].dtype}  (from our bf16 draft)")

    # sanity: layer count + head-level placement should mirror the released reference
    lids = sorted({int(m.group(1)) for k in merged for m in [re.match(r"mtp\.(\d+)\.", k)] if m})
    assert "mtp.0.main_proj.weight" in merged, "main_proj must sit on mtp.0"
    last = lids[-1]
    assert f"mtp.{last}.markov_head.markov_w1.weight" in merged, f"markov_head must sit on mtp.{last}"
    print(f">>> mtp layers {lids}; main_proj@0, heads@{last}  (matches released layout)")

    # --- 3) write weights (single file; ~13GB fp8 + ~2GB bf16 embed/head is well within limits) ---
    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f">>> wrote {out/'model.safetensors'} ({len(merged)} tensors)")

    # --- 4) config.json = our proven-good draft config (+ fp8 quant_config only when NOT dequant) ---
    ours_cfg = json.loads((ours / "config.json").read_text())
    if args.dequant_bf16:
        ours_cfg.pop("quantization_config", None)  # weights are bf16 now — no quant config at all
        (out / "config.json").write_text(json.dumps(ours_cfg, indent=2))
        print(">>> wrote config.json (base=our bf16 draft, NO quantization_config — pure bf16 draft)")
    else:
        rel_cfg = json.loads((rel / "config.json").read_text())
        qcfg = rel_cfg.get("quantization_config")
        if qcfg is None:
            raise SystemExit(f"!! released config has no quantization_config — is {rel} the fp8 release?")
        qcfg = dict(qcfg)  # copy — don't mutate rel_cfg
        if args.ignore_embed_head:
            ig = list(qcfg.get("ignored_layers", []))
            for name in ("embed", "head"):  # borrowed bf16, no .scale => must stay unquantized
                if name not in ig:
                    ig.append(name)
            qcfg["ignored_layers"] = ig
        ours_cfg["quantization_config"] = qcfg
        (out / "config.json").write_text(json.dumps(ours_cfg, indent=2))
        print(f">>> wrote config.json (base=our bf16 draft, grafted fp8 quant: {qcfg.get('quant_method')}"
              f"{', ignored_layers=' + str(qcfg['ignored_layers']) if args.ignore_embed_head else ''})")

    # copy small companions if present (harmless; draft_model_config mostly ignores tokenizer)
    for name in ("generation_config.json",):
        src = ours / name
        if src.exists():
            shutil.copy2(src, out / name)

    # --- verify embed/head geometry vs config ---
    V, H = ours_cfg.get("vocab_size"), ours_cfg.get("hidden_size")
    for k in ("embed.weight", "head.weight"):
        assert tuple(merged[k].shape) == (V, H), f"{k} {tuple(merged[k].shape)} != ({V},{H})"
    print(f">>> embed/head geometry OK vs config (vocab={V}, hidden={H})")
    kind = "bf16 (dequant'd)" if args.dequant_bf16 else "fp8 (verbatim)"
    print(f"\n✓ standalone released {kind} draft dir ready:", out)
    print("  next:  DRAFT=%s MODEL=/share/.../DeepSeek-V4-Flash-bf16 NUM_SPEC=5 \\" % out)
    print("         bash examples/ascend_npu_dflash/serve_dsv4_bf16_dualnode.sh head")


if __name__ == "__main__":
    main()

