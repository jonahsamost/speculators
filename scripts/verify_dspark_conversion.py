#!/usr/bin/env python3
"""UT-1 (offline, CPU): INDEPENDENTLY verify the layers.* -> mtp.* conversion is bit-exact.

`convert_dspark_to_vllm.py` is a pure rename + expert-unstack (NO numerical transform), so every
mtp.* tensor MUST be byte-identical to its source layers.* tensor. This script re-derives the
expected source for each mtp.* key WITHOUT importing the converter's mapping (so a bug in the
converter can't hide behind its own logic), and asserts equality. Catches: wrong expert-unstack
order/slice, accidental transpose, wrong rename, dropped/extra tensors.

  python scripts/verify_dspark_conversion.py --in <orig ckpt dir> --out <converted mtp dir>

Runs on CPU; needs only torch + safetensors. `--in` = the trainer ckpt (layers.*), `--out` = the
convert_dspark_to_vllm.py output (mtp.*). Exit code 0 = all match.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _load(d: Path) -> dict:
    from safetensors.torch import load_file  # noqa: PLC0415

    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        sd: dict = {}
        for shard in sorted(set(wm.values())):
            sd.update(load_file(str(d / shard)))
        return sd
    single = d / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    raise SystemExit(f"!! no model.safetensors[.index.json] in {d}")


def _eq(a, b) -> bool:
    import torch  # noqa: PLC0415

    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def expected_source_key(mtp_key: str, last: int):
    """Given an mtp.* key, INDEPENDENTLY return (layers_key, expert_idx_or_None) it must come from,
    re-deriving the inverse of the intended mapping from scratch (NOT via the converter)."""
    # model-level extras
    if mtp_key == "embed.weight":
        return ("embed_tokens.weight", None)
    if mtp_key == "head.weight":
        return ("lm_head.weight", None)
    if mtp_key == "mtp.0.main_proj.weight":
        return ("fc.weight", None)
    if mtp_key == "mtp.0.main_norm.weight":
        return ("hidden_norm.weight", None)
    if mtp_key == f"mtp.{last}.norm.weight":
        return ("norm.weight", None)
    m = re.fullmatch(rf"mtp\.{last}\.markov_head\.(.*)", mtp_key)
    if m:
        return (f"markov_head.{m.group(1)}", None)
    m = re.fullmatch(rf"mtp\.{last}\.hc_head_(fn|base|scale)", mtp_key)
    if m:
        return (f"hc_head.hc_{m.group(1)}", None)
    m = re.fullmatch(rf"mtp\.{last}\.confidence_head\.(.*)", mtp_key)
    if m:
        return (f"confidence_head.{m.group(1)}", None)  # serve ignores it, but it should still round-trip
    # per-layer
    lm = re.fullmatch(r"mtp\.(\d+)\.(.*)", mtp_key)
    if not lm:
        return (None, None)
    n, rest = int(lm.group(1)), lm.group(2)
    # hc: mtp.n.hc_{attn,ffn}_{fn,base,scale} <- layers.n.{attn_hc,ffn_hc}.{fn,base,scale}
    hc = re.fullmatch(r"hc_(attn|ffn)_(fn|base|scale)", rest)
    if hc:
        site = "attn_hc" if hc.group(1) == "attn" else "ffn_hc"
        return (f"layers.{n}.{site}.{hc.group(2)}", None)
    # router: mtp.n.ffn.gate.{weight,bias} <- layers.n.ffn.router.{weight,bias}
    g = re.fullmatch(r"ffn\.gate\.(weight|bias)", rest)
    if g:
        return (f"layers.{n}.ffn.router.{g.group(1)}", None)
    # unstacked routed experts: mtp.n.ffn.experts.{e}.w{k}.weight <- layers.n.ffn.experts.w{k}[e]
    ex = re.fullmatch(r"ffn\.experts\.(\d+)\.w([123])\.weight", rest)
    if ex:
        return (f"layers.{n}.ffn.experts.w{ex.group(2)}", int(ex.group(1)))
    # everything else (attn.*, attn_norm, ffn_norm, shared_experts.*) is a direct rename
    return (f"layers.{n}.{rest}", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="original trainer ckpt dir (layers.*)")
    ap.add_argument("--out", dest="out", required=True, help="converted mtp.* dir")
    args = ap.parse_args()

    src = _load(Path(args.inp))
    dst = _load(Path(args.out))
    n_layers = max((int(m.group(1)) for k in src for m in [re.match(r"layers\.(\d+)\.", k)] if m), default=-1) + 1
    last = n_layers - 1
    print(f">>> in: {len(src)} tensors ({n_layers} draft layers)   out: {len(dst)} tensors")

    ok, mismatch, missing_src, unmapped = 0, [], [], []
    for mkey, mten in dst.items():
        skey, e = expected_source_key(mkey, last)
        if skey is None:
            unmapped.append(mkey)
            continue
        if skey not in src:
            missing_src.append((mkey, skey))
            continue
        sten = src[skey] if e is None else src[skey][e]
        if _eq(mten, sten):
            ok += 1
        else:
            mismatch.append((mkey, skey, e, tuple(mten.shape), tuple(sten.shape),
                             str(mten.dtype), str(sten.dtype)))

    print(f">>> matched bit-exact: {ok}/{len(dst)}")
    if unmapped:
        print(f"!! {len(unmapped)} out keys this checker couldn't map (extend the checker): {unmapped[:6]}")
    if missing_src:
        print(f"!! {len(missing_src)} out keys whose EXPECTED source is absent in --in:")
        for mk, sk in missing_src[:12]:
            print(f"     {mk}  <-  {sk}  (MISSING in input)")
    if mismatch:
        print(f"!! {len(mismatch)} MISMATCHES (shape/dtype/values differ — a real conversion bug):")
        for mk, sk, e, msh, ssh, mdt, sdt in mismatch[:20]:
            tag = f"[e={e}]" if e is not None else ""
            print(f"     {mk} {tag}\n        out {msh} {mdt}  !=  {sk} {ssh} {sdt}")
    # also flag input tensors that SHOULD have been converted but have no output (dropped)
    covered = set()
    for mkey in dst:
        skey, _ = expected_source_key(mkey, last)
        if skey:
            covered.add(skey)
    # confidence_head.proj.bias: LEGACY checkpoints only. ConfidenceHead is now bias-free
    # (the released mtp.* layout has no such key); kept so pre-fix checkpoints still verify.
    droppable = [k for k in src if not k.startswith(("verifier_", "freqs")) and k not in covered
                 and not re.fullmatch(r"(confidence_head\.proj\.bias)", k)]
    if droppable:
        print(f"!! {len(droppable)} input tensors NOT represented in the output (silently dropped?): {droppable[:12]}")

    bad = bool(mismatch or missing_src or unmapped or droppable)
    print("\n" + ("!! CONVERSION HAS ISSUES — see above." if bad
                  else "✓ conversion is bit-exact: every mtp.* tensor == its layers.* source. "
                       "Conversion is NOT the bug."))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

