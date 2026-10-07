#!/usr/bin/env python3
"""Convert a trained DSV4-DSpark draft checkpoint into the ``mtp.*`` layout that
vLLM-Ascend's DSpark serve loads.

WHY
---
Our trainer (``scripts/train.py`` + ``save_pretrained``) writes a consolidated
``model.safetensors`` in the clean-room ``layers.{n}.*`` namespace, with the routed
experts STACKED (``layers.{n}.ffn.experts.w{1,2,3}`` shaped ``[E, out, in]`` — the
``GroupedExperts`` module). vLLM-Ascend's DSpark draft (``vllm_ascend/models/
deepseek_v4_dspark.py`` on the ``dspark-dsv4`` fork) loads the RELEASED ``mtp.*``
layout with PER-EXPERT keys ``mtp.{n}.ffn.experts.{e}.w{1,2,3}.weight`` and remaps
them internally (``_remap_dspark_name``: ``.attn.``->``.self_attn.``, ``.ffn.``->
``.mlp.``, ``.w1.``->``.gate_proj.`` ...).

So this conversion is exactly the INVERSE of
``speculators/src/speculators/models/dsv4_dspark/weights.py::map_released_key``
(released ``mtp.*`` -> ours ``layers.*``), PLUS unstacking the ``GroupedExperts``
weights back to per-expert tensors. This script emits a bf16 interchange checkpoint;
use the serving-export script to restore the released fp8/fp4 representation required
by the DeepSeek V4.1 vLLM runtime.

Key map (ours -> released ``mtp.*``):

  embed_tokens.weight                 -> embed.weight
  lm_head.weight                      -> head.weight
  main_proj.weight                    -> mtp.0.main_proj.weight
  main_norm.weight                    -> mtp.0.main_norm.weight
  norm.weight                         -> mtp.{last}.norm.weight
  markov_head.*                       -> mtp.{last}.markov_head.*
  confidence_head.proj.weight         -> mtp.{last}.confidence_head.proj.weight
  layers.{n}.attn.*                   -> mtp.{n}.attn.*            (incl attn_sink)
  layers.{n}.attn_norm.weight         -> mtp.{n}.attn_norm.weight
  layers.{n}.ffn_norm.weight          -> mtp.{n}.ffn_norm.weight
  layers.{n}.attn_hc.{fn,base,scale}  -> mtp.{n}.hc_attn_{fn,base,scale}
  layers.{n}.ffn_hc.{fn,base,scale}   -> mtp.{n}.hc_ffn_{fn,base,scale}
  layers.{n}.ffn.router.{weight,bias} -> mtp.{n}.ffn.gate.{weight,bias}
  layers.{n}.ffn.shared_experts.w{k}.weight -> mtp.{n}.ffn.shared_experts.w{k}.weight
  layers.{n}.ffn.experts.w{k}  [E,..] -> mtp.{n}.ffn.experts.{e}.w{k}.weight   (UNSTACK)
  (per-expert ffn.experts.{e}.w{k}.weight, non-EP ckpt, passes through unchanged)
  SKIP: verifier_lm_head.*, verifier_norm.*, *freqs_cis* (target-only / buffers)

Usage
-----
    # 1) DRY RUN first — confirm the input key layout matches the map above:
    python scripts/convert_dspark_to_vllm.py --in <ckpt_dir> --inspect

    # 2) Real convert:
    python scripts/convert_dspark_to_vllm.py --in <ckpt_dir> --out <out_dir> \
        [--config-from <released_or_target/config.json>]

``<ckpt_dir>`` is a trainer epoch dir (``.../ckpt_.../<epoch>/``) holding
``model.safetensors`` (or a sharded set + ``model.safetensors.index.json``) and
``config.json``. Needs ``torch`` + ``safetensors`` (present in the training env).

NB on config.json: the WEIGHTS are fully specified; the config is not. This copies a
config into ``<out_dir>`` but you MUST make it vLLM-Ascend-DSpark-loadable — verify
``architectures`` + ``dspark_block_size`` + ``num_nextn_predict_layers`` against the
RELEASED draft's config.json (you'll have it from the step-1 released-draft eval).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_SKIP = ("verifier_lm_head", "verifier_norm")


def _map(key: str, last: int):
    """Return the released base key(s) for one of our keys, or None to skip.

    For stacked experts the caller expands over the leading (expert) dim; here we
    return the per-expert TEMPLATE key with ``{e}`` to be filled in.
    """
    if key.startswith(_SKIP) or "freqs_cis" in key:
        return None
    if key == "embed_tokens.weight":
        return ["embed.weight"]
    if key == "lm_head.weight":
        return ["head.weight"]
    # target-hidden conditioning: our DFlash/DSpark backbone names these `fc`/`hidden_norm`
    # (dflash/core.py:93,98 — "fc(=main_proj role)"), NOT main_proj/main_norm. The released
    # serve REQUIRES `mtp.0.main_proj`+`main_norm` (deepseek_v4_dspark.py load_weights required set),
    # so map both spellings. fc = Linear(len(target_layer_ids)*H -> H, bias=False) => [H, 3H],
    # shape-identical to serve main_proj [H, H*num_target_layers]; hidden_norm[H] = main_norm[H].
    if key in ("fc.weight", "main_proj.weight"):
        return ["mtp.0.main_proj.weight"]
    if key in ("hidden_norm.weight", "main_norm.weight"):
        return ["mtp.0.main_norm.weight"]
    if key == "norm.weight":
        return [f"mtp.{last}.norm.weight"]
    if key == "markov_head.markov_w1.weight":
        return [f"mtp.{last}.markov_head.embed.weight"]
    if key == "markov_head.markov_w2.weight":
        return [f"mtp.{last}.markov_head.head.weight"]
    if key == "confidence_head.proj.weight":
        return [f"mtp.{last}.confidence_head.proj.weight"]
    # Compatibility with checkpoints produced before the V4.1 mHC correction.
    # vLLM collapses with the last FFN's pre-mix and has no terminal HC module.
    if re.fullmatch(r"hc_head\.hc_(fn|base|scale)", key):
        return None

    lm = re.fullmatch(r"layers\.(\d+)\.(.*)", key)
    if not lm:
        return None
    n, rest = int(lm.group(1)), lm.group(2)

    hc = re.fullmatch(r"(attn_hc|ffn_hc)\.(fn|base|scale)", rest)
    if hc:
        site = "hc_attn" if hc.group(1) == "attn_hc" else "hc_ffn"
        return [f"mtp.{n}.{site}_{hc.group(2)}"]
    if rest.startswith("ffn.router."):
        return [f"mtp.{n}.ffn.gate.{rest[len('ffn.router.'):]}"]
    if re.fullmatch(r"ffn\.experts\.w[123]", rest):  # STACKED [E, out, in] -> expand
        wn = rest.split(".")[-1]
        return [f"mtp.{n}.ffn.experts.{{e}}.{wn}.weight"]  # template; {e} filled by caller
    # shared_experts / per-expert experts / attn.* / attn_norm / ffn_norm -> direct
    return [f"mtp.{n}.{rest}"]


def convert(state_dict: dict, n_layers: int):
    """ours state_dict -> released mtp.* state_dict. Returns (out, skipped, n_unstacked)."""
    last = n_layers - 1
    out: dict = {}
    skipped: list[str] = []
    n_unstacked = 0
    for k, v in state_dict.items():
        tgt = _map(k, last)
        if tgt is None:
            skipped.append(k)
            continue
        base = tgt[0]
        if "{e}" in base:  # stacked experts -> one tensor per expert
            n_unstacked += 1
            for e in range(v.shape[0]):
                out[base.format(e=e)] = v[e].contiguous().clone()
        else:
            out[base] = v
    return out, skipped, n_unstacked


def _load_state_dict(in_dir: Path):
    """Load our trainer ckpt (single or sharded safetensors) as {key: tensor}."""
    from safetensors.torch import load_file  # noqa: PLC0415

    idx = in_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        sd: dict = {}
        for shard in sorted(set(wm.values())):
            sd.update(load_file(str(in_dir / shard)))
        return sd
    single = in_dir / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    raise SystemExit(f"!! no model.safetensors[.index.json] in {in_dir}")


def _n_layers(state_dict: dict) -> int:
    ns = {int(m.group(1)) for k in state_dict for m in [re.match(r"layers\.(\d+)\.", k)] if m}
    if not ns:
        raise SystemExit("!! no layers.{n}.* keys found — is this a dsv4_dspark ckpt?")
    return max(ns) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="trainer ckpt dir (has model.safetensors + config.json)")
    ap.add_argument("--out", help="output dir for the mtp.* checkpoint (required unless --inspect)")
    ap.add_argument("--config-from", help="config.json to copy into --out (released/target draft config)")
    ap.add_argument("--inspect", action="store_true", help="dry run: load + show the key mapping, write nothing")
    args = ap.parse_args()

    in_dir = Path(args.inp)
    sd = _load_state_dict(in_dir)
    n_layers = _n_layers(sd)
    out, skipped, n_unstacked = convert(sd, n_layers)

    print(f">>> input: {len(sd)} tensors, {n_layers} draft layers  ({in_dir})")
    print(f">>> output: {len(out)} tensors  ({n_unstacked} stacked-expert params unstacked, "
          f"{len(skipped)} skipped)")
    if skipped:
        print(f"    skipped (target-only/buffers): {sorted(skipped)[:8]}{' ...' if len(skipped) > 8 else ''}")
    # a few sample mappings for eyeballing
    print("    sample mtp.* keys:")
    for k in list(out)[:6] + [k for k in out if ".experts.0." in k][:1]:
        print(f"      {k}  {tuple(out[k].shape)}")

    if args.inspect:
        print("\n(--inspect: nothing written. Re-run with --out <dir> to convert.)")
        return
    if not args.out:
        raise SystemExit("!! --out is required (or use --inspect)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file  # noqa: PLC0415

    save_file(out, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"\n>>> wrote {out_dir / 'model.safetensors'}")

    # config.json — copied, then patched to be vLLM-Ascend-DSpark spec-decode-loadable.
    src_cfg = Path(args.config_from) if args.config_from else (in_dir / "config.json")
    if src_cfg.exists():
        cfg = json.loads(src_cfg.read_text())
        # Multimodal/wrapper checkpoints such as DeepSeek-V4.1 keep the actual
        # language-model configuration under ``text_config``. Preserve the
        # wrapper for AutoConfig, but validate and patch the nested payload that
        # vLLM ultimately passes to the DSpark model.
        model_cfg = cfg.get("text_config", cfg)
        if not isinstance(model_cfg, dict):
            raise SystemExit(
                f"!! config from {src_cfg} has a non-object text_config"
            )
        # ★ serve reads the aux target layers from `eagle_aux_hidden_state_layer_ids` (EAGLE3 path in
        # model_runner). The released draft config leaves it None → the serve falls back to
        # get_eagle3_default_aux_hidden_state_layers() = 4 layers → target emits 4*H while our draft's
        # main_proj wants 3*H (dspark_target_layer_ids) → dim mismatch at the first draft proposal.
        # Pin it to dspark_target_layer_ids so the target captures exactly the layers the draft trained on.
        tids = model_cfg.get("dspark_target_layer_ids")
        if tids and not model_cfg.get("eagle_aux_hidden_state_layer_ids"):
            model_cfg["eagle_aux_hidden_state_layer_ids"] = tids
            print(f">>> patched config.json: eagle_aux_hidden_state_layer_ids = {tids} (from dspark_target_layer_ids)")
        # ★ GUARD the serve-critical fields. The vLLM-Ascend DSpark serve reads these dspark_*/
        # sliding_window names; if any is ABSENT it silently falls back to a WRONG default and caps
        # (or crashes) acceptance — NOT a loud error at serve time. Known silent fallbacks:
        #   dspark_noise_token_id absent  -> mask slots filled with token 0 (proposer: `... or 0`)
        #                                    -> wrong draft-attn inputs on slots 1..g-1 -> capped tail
        #   dspark_target_layer_ids/eagle_aux absent -> eagle3's 4 default aux layers -> 4H vs 3H
        #                                    main_proj -> dim-mismatch crash at first proposal
        #   sliding_window absent          -> hf_config.sliding_window AttributeError at serve
        # Our OWN (plain-name) config lacks these, so this fires loudly if you forget --config-from.
        _required = ["dspark_noise_token_id", "dspark_target_layer_ids", "dspark_block_size",
                     "dspark_markov_rank", "sliding_window", "eagle_aux_hidden_state_layer_ids"]
        _missing = [k for k in _required if model_cfg.get(k) in (None, "", [])]
        if _missing:
            raise SystemExit(
                f"!! config from {src_cfg} is MISSING serve-critical field(s): {_missing}\n"
                f"   The vLLM-Ascend DSpark serve would silently use wrong defaults (mask-slot token 0,\n"
                f"   or eagle3's 4 aux layers -> 3H/4H main_proj crash). Pass --config-from <the RELEASED\n"
                f"   draft config.json> (it carries all dspark_* fields), not our plain-name training config."
            )
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f">>> wrote config.json (from {src_cfg})")
        print("    serve-critical fields: "
              + "  ".join(f"{k}={model_cfg.get(k)}" for k in
                          ("dspark_noise_token_id", "dspark_block_size", "dspark_target_layer_ids",
                           "sliding_window", "dspark_markov_rank")))
    else:
        print(f"!! no config.json at {src_cfg} — provide one in {out_dir} before serving")

    # Make the output world-READABLE so a *different* account can serve it (the serve process
    # often runs as a different user than the one that converted). umask on the writer is
    # commonly 077 → files land 600 → the serve user gets FileNotFoundError (ENOENT, not EACCES)
    # on model.safetensors. Files: 0644, the dir itself: 0755 (traverse+list). NB: mode travels
    # WITH the file on `mv`; only the DESTINATION's parent dirs still need o+x to be reachable.
    try:
        out_dir.chmod(0o755)
        for _f in out_dir.iterdir():
            if _f.is_file():
                _f.chmod(0o644)
        print(f">>> chmod: {out_dir} 0755, files 0644 (readable by the serve account)")
    except OSError as _e:  # some shared FS reject chmod — non-fatal, user can chmod by hand
        print(f"!! could not chmod {out_dir} ({_e}); if the serve account can't read it, run: "
              f"chmod -R a+rX {out_dir}")

    print("⚠  VERIFY config.json before loading: `architectures`, `dspark_block_size`, "
          "`num_nextn_predict_layers` must match the RELEASED draft's config (from step-1 eval).")
    print("⚠  Run --inspect on the ACTUAL ckpt first to confirm the expert layout "
          "(stacked `ffn.experts.w1` vs per-expert) matches the map before trusting the output.", file=sys.stderr)


if __name__ == "__main__":
    main()
