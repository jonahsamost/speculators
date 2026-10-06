"""Graft B+C: torch.compile'd grouped-GEMM MoE experts (opt-in ``DSPARK_COMPILE=1``) — **SEED TECH**.

Kills the per-shape grouped-GEMM recompile — measured **42% of wall-clock** once ``MAX_ANCHORS`` solved
the HS-bound (see ``docs/deployment/ascend-npu-dsv4-dspark-compile-recompile.md``) — by compiling the
experts forward ONCE with a symbolic token dim (``maybe_mark_dynamic``) → **one AscendC kernel for every
routed-token count** (no recompile). Validated bit-exact (fwd+bwd, ``unique_graphs=1``) → ~**1.74×**.

★ **DEFAULT OFF and MUST STAY OFF on the main (torch 2.10) stack.** It requires a matched stack our
training/serve env does NOT have:
    torch 2.12.0+cpu · torch_npu 2.12.0rc1 · inductor_npu_ext (gitcode Ascend/torchair @3c9418c2) · triton-ascend.
Enabling it now would desync training vs the 2.10 serve (invalidating rolled data). This is a seed
capability for a future FULL-stack (train+serve) upgrade. Everything here is lazy-imported so importing
this module on the 2.10 stack is a no-op until :func:`enable` is called.

Graft B = torchtitan-npu ``_run_experts_grouped_mm`` (``torch._grouped_mm`` + fused ``w13`` + ``npu_swiglu``).
Graft C = ``torch.compile(backend="inductor")`` + ``maybe_mark_dynamic(x, 0)`` + the NPU compile shims.
"""

from __future__ import annotations

import os

import torch

_ENABLED = False
_COMPILED = None


def _install_compile_shims() -> None:
    """torch 2.12's Triton/TMA probes call get_device_capability()/...properties().major → None on NPU →
    '>=' TypeError during compile. inductor_npu_ext codegens AscendC (not Triton), so benign no-Triton
    values are safe. Also set capture_scalar_outputs for the data-dependent ``offs`` (torchtitan-npu does)."""

    def _safe_cap(_f):
        def inner(*a, **k):
            try:
                c = _f(*a, **k)
            except Exception:  # noqa: BLE001
                c = None
            return c if c is not None else (0, 0)

        return inner

    for _name in ("cuda", "npu"):
        mod = getattr(torch, _name, None)
        gc = getattr(mod, "get_device_capability", None) if mod is not None else None
        if gc is not None:
            mod.get_device_capability = _safe_cap(gc)
    torch._dynamo.config.capture_scalar_outputs = True
    # ★ RECOMPILE-THRASH FIX (the 54%-wall-clock spikes that survived maybe_mark_dynamic).
    # maybe_mark_dynamic does NOT collapse the grouped-GEMM to a single kernel on the NPU AscendC
    # path — after DSPARK_MOE_BUCKET quantization there are still >8 distinct routed-token shapes
    # (seq-len varies per rollout sample). dynamo's DEFAULT cache_size_limit is 8: once exceeded it
    # LRU-EVICTS a compiled shape, and when that shape recurs it triggers a fresh ~21s AscendC
    # rebuild -> perpetual thrash that never amortizes (726 recompiles over a 7774-step run = 54%).
    # Raise the limits so every bucket shape compiles ONCE and stays resident (kernels are also
    # disk-cached under $TMPDIR/.npu_kernels_$USER, so the ceiling is cheap). Confirm the cause with
    # TORCH_LOGS=recompiles -> "hit config.cache_size_limit (8)". Override via DSPARK_COMPILE_CACHE_LIMIT.
    _lim = int(os.environ.get("DSPARK_COMPILE_CACHE_LIMIT", "1024"))
    torch._dynamo.config.cache_size_limit = _lim
    torch._dynamo.config.accumulated_cache_size_limit = _lim * 2


def _register_grouped_mm_npu() -> None:
    """aten::_grouped_mm → npu_grouped_matmul (verbatim from torchtitan-npu ops/_grouped_mm.py)."""
    import torch_npu  # noqa: PLC0415

    try:

        @torch.library.impl("aten::_grouped_mm", "PrivateUse1")
        def _(self, mat2, offs, bias=None, out_dtype=None):
            split_along_k = self.ndim == 2 and mat2.ndim == 2
            return torch_npu.npu_grouped_matmul(
                [self],
                [mat2],
                group_list=offs.to(dtype=torch.int64),
                group_list_type=0,
                split_item=2,
                group_type=(2 if split_along_k else 0),
                bias=[bias] if bias is not None else None,
                output_dtype=out_dtype,
            )[0]
    except RuntimeError:
        pass  # already registered (idempotent)


def _expert_activation(h, swiglu_limit, scores):
    """The elementwise SwiGLU bridge between the two grouped-GEMMs — the ONLY part we compile
    (torchtitan-npu pattern). It's elementwise so it tolerates a dynamic token dim; the two
    grouped-GEMMs stay EAGER because compiling THEM makes inductor-NPU codegen a per-shape AscendC
    kernel (the ~26s "recompile" spikes). ``h[N, 2*inter]`` = gate|up from the first grouped-GEMM."""
    import torch_npu  # noqa: PLC0415

    if swiglu_limit > 0:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
    h = torch_npu.npu_swiglu(h, dim=-1)  # [N, inter]
    return h * scores.reshape(-1, 1).to(h.dtype)


def enable() -> None:
    """Register the NPU ``aten::_grouped_mm`` -> ``npu_grouped_matmul`` impl. FULLY EAGER now — NOTHING
    is torch.compile'd (see run()). Call ONCE at startup with ``DSPARK_COMPILE=1`` on the torch-2.12
    stack (needs ``torch._grouped_mm``). No-op / unsafe on the 2.10 main stack."""
    global _ENABLED
    if _ENABLED:
        return
    _install_compile_shims()  # harmless dynamo / get_device_capability shims
    _register_grouped_mm_npu()  # torch._grouped_mm -> npu_grouped_matmul on NPU (needed for the eager GMMs)
    _ENABLED = True


def run(w1, w3, w2, x, counts, swiglu_limit, scores):
    """Experts forward used by ``moe_grouped_gemm._fused_permute_dispatch_npu`` when enabled.
    ★ FULLY EAGER: eager grouped-GEMMs + eager SwiGLU, NOTHING torch.compile'd. On this inductor_npu_ext
    stack, compiling ANY variable-token op (even the elementwise swiglu bridge) makes inductor-NPU
    codegen a per-shape AscendC kernel = the ~26s recompile spikes (CONFIRMED: jit_compile=False took,
    yet the compiled-swiglu variant still recompiled 61%). So compile nothing; with jit_compile=False the
    eager ``torch._grouped_mm`` uses the shape-generic aclnn kernel -> ZERO per-shape rebuild. Uses the
    SAME ``torch._grouped_mm`` as the old compiled path (correct numerics), NOT the broken
    ``swiglu_grouped``/``_NpuGroupedMatmul`` eager path. ``w13 = cat([w1, w3])`` [E, 2*inter, dim]."""
    w13 = torch.cat([w1, w3], dim=1)  # [E, 2*inter, dim]
    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
    h = torch._grouped_mm(
        x.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offs
    )  # [N, 2*inter] EAGER
    h = _expert_activation(
        h, swiglu_limit, scores
    )  # [N, inter] EAGER swiglu (NOT compiled)
    return torch._grouped_mm(
        h.bfloat16(), w2.bfloat16().transpose(-2, -1), offs=offs
    ).float()  # [N, dim] EAGER
