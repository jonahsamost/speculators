"""Fused grouped-GEMM implementation of the DSV4 DSpark MoE dispatch (opt-in NPU).

Replaces the 256-way per-expert eager loop (`moe._moe_dispatch_torch`) with ONE grouped
matmul per projection (gate / up / down), sorted by expert. This is the same math but:
  * variable per-expert token counts go into a single `npu_grouped_matmul` (group_list) that
    handles them WITHOUT a per-shape kernel recompile -> kills the forward recompile spikes;
  * 256 small GEMMs collapse to 3 grouped GEMMs -> big throughput win.

Registers itself under `("moe_dispatch", "npu")` (import this module from the NPU bridge and
`set_active_backend("npu")`). CPU-runnable via a pure-torch grouped fallback, so the routing /
permute / swiglu logic unit-tests without an NPU (test_moe_grouped_gemm.py).

⚠️ Distributed note: grouped-GEMM needs each expert's FULL weight locally (it stacks them).
That conflicts with per-expert FSDP2 sharding (weights are gathered only inside each expert's
forward hook). So this is a clean win for single-card / expert-replicated / **EP** (each rank
owns whole experts) layouts; the per-expert-FSDP faithful run needs EP (Track B) or a
gather-before-group. The MATH here is layout-independent and is what the unit test pins down.
"""

from __future__ import annotations

import os

import torch

from .kernels import register_kernel
from .moe import GroupedExperts, swiglu_grouped

_MOE_OP = "moe_dispatch"

# Bucket the routed-token count so the grouped-GEMM sees only a handful of distinct row-counts.
# npu_grouped_matmul recompiles per unique #rows; under EP the per-rank routed-token count varies
# every step with routing imbalance -> a recompile spike each time a new count appears (observed
# fwd_ms jumping 300ms -> 16s). Rounding the count up to a DSPARK_MOE_BUCKET multiple collapses it
# to few shapes (recompiles amortize after the first few). Padding is a few % of the TOKEN dim
# (NOT per-expert capacity), so the memory/compute waste is small. 0/1 disables bucketing.
_MOE_BUCKET = int(os.environ.get("DSPARK_MOE_BUCKET", "512"))
# DSPARK_PROFILE_MOE=1 -> sync+time each MoE sub-op (permute / experts-GEMM / unpermute) and print any sub-op
# whose time exceeds DSPARK_PROFILE_MOE_MS (default 2000ms; set 0 to print EVERY step). Pins WHICH op does the
# per-shape AscendC build (the fwd "recompile" spike). Diagnostic only.
_MOE_PROF = os.environ.get("DSPARK_PROFILE_MOE") == "1"
_MOE_PROF_MS = float(os.environ.get("DSPARK_PROFILE_MOE_MS", "2000"))
# Fused Ascend routing ops (npu_moe_token_permute/unpermute AND npu_moe_token_unpermute_grad) FAIL
# on a 0-row input ("input shape has 0", error 561002). Under EP a rank can receive 0 tokens in a
# step (all top-k picks miss its 32 experts) -> the local grouped path gets [0, dim] and the unpermute
# backward crashes. Pad an empty step to a non-zero floor so the ops run and the experts stay in the
# autograd graph (dummy tokens carry zero router weight -> zero grad, which is the correct grad for a
# rank that processed no tokens). Non-empty steps are unaffected.
_MOE_EMPTY_MIN = 16


def _bucket_count(n: int) -> int:
    if n == 0:
        return _MOE_BUCKET if _MOE_BUCKET > 1 else _MOE_EMPTY_MIN
    if _MOE_BUCKET <= 1:
        return n
    return ((n + _MOE_BUCKET - 1) // _MOE_BUCKET) * _MOE_BUCKET


def _grouped_matmul_torch(
    x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Reference grouped matmul: x[T, K] blocks (by `counts`) @ weight[E, K, M] -> [T, M].

    Autograd-native (split + matmul + cat); used on CPU and as the parity oracle.
    """
    outs = []
    off = 0
    for e in range(weight.shape[0]):
        n = int(counts[e])
        outs.append(x[off : off + n] @ weight[e])
        off += n
    return torch.cat(outs, dim=0) if outs else x.new_zeros((0, weight.shape[-1]))


class _NpuGroupedMatmul(torch.autograd.Function):
    """torch_npu.npu_grouped_matmul with an explicit backward (mirrors MindSpeed GMMFunction).

    x[T, K], weight[E, K, M], counts[E] (raw per-group token counts) -> [T, M].
    """

    @staticmethod
    def forward(ctx, x, weight, counts):
        import torch_npu

        group_list = torch.cumsum(counts, dim=0).to(torch.int64)
        out = torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]
        ctx.save_for_backward(x, weight, group_list)
        return out

    @staticmethod
    def backward(ctx, grad):
        import torch_npu

        x, weight, group_list = ctx.saved_tensors
        # dx = grad[T, M] @ weight[E, K, M]^T  (grouped) ; dw = x^T @ grad (grouped)
        dx = torch_npu.npu_grouped_matmul(
            [grad],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
        )[0]
        dw = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=2,
            group_list_type=0,
        )[0]
        return dx, dw.reshape(weight.shape), None


def _grouped_matmul(x, weight, counts):
    if x.device.type == "npu":
        return _NpuGroupedMatmul.apply(x, weight, counts)
    return _grouped_matmul_torch(x, weight, counts)


def _fused_permute_dispatch_npu(
    x: torch.Tensor,
    indices: torch.Tensor,
    w_flat: torch.Tensor,
    experts: GroupedExperts,
    n_experts: int,
) -> torch.Tensor:
    """NPU fused dispatch: ``npu_moe_token_permute`` -> grouped-GEMM SwiGLU -> ``npu_moe_token_unpermute``.

    Same math as the ``argsort`` + gather + ``index_add`` path, but the sort/scatter run on the
    fused Ascend routing ops instead of an int64 ``argsort`` (which falls back to AiCPU). Mirrors
    torchtitan-npu ``_run_local_experts``:
      * ``x[N, dim]``   -- the (UN-replicated) input tokens;
      * ``indices[N, k]`` -- per-token expert ids (k = topk for the routed path, or 1 for the
        already-flattened EP per-unit path);
      * ``w_flat[N*k]`` -- the matching router weights (row-major over (token, slot)).
    Returns the UNPERMUTED per-unit output ``[N*k, dim]`` in original (row-major) order -- the
    caller sums the k slots (routed topk) or uses it as-is (k == 1). The grouped GEMM is still
    the validated ``_grouped_matmul`` (Graft A swaps only the permute; Graft B swaps the GEMM).

    The token count ``N`` is BUCKETED (:func:`_bucket_count`) so the whole permute/grouped/unpermute
    chain sees a recompile-stable shape: pad rows are zero tokens routed to expert 0 with zero
    router weight (zero contribution) and are sliced off after unpermute -- mathematically identical
    to the un-bucketed path (the CPU parity reference has no padding), just fewer distinct shapes.
    """
    import torch.nn.functional as F  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415  (NPU-only; every caller gates on device.type == "npu")

    w1, w3, w2 = experts.local_weights()
    n, dim = x.shape
    k = indices.shape[1]
    nb = _bucket_count(n)
    if nb > n:
        pad = nb - n
        x = F.pad(x, (0, 0, 0, pad))  # [nb, dim] zero tokens
        indices = F.pad(indices, (0, 0, 0, pad))  # [nb, k] -> expert 0
        w_flat = F.pad(w_flat.reshape(n, k), (0, 0, 0, pad)).reshape(
            -1
        )  # [nb*k] zero weight

    # Ascend routing ops take int32 expert ids (ids < n_experts fit); ids are non-differentiable.
    if _MOE_PROF:
        import time as _time  # noqa: PLC0415

        torch.npu.synchronize()
        _t0 = _time.perf_counter()
    idx = indices.to(torch.int32)
    routed_input, sorted_idx = torch_npu.npu_moe_token_permute(x, idx)  # [nb*k, dim]
    routed_scores, _ = torch_npu.npu_moe_token_permute(
        w_flat.reshape(-1, 1), idx.reshape(-1, 1)
    )  # [nb*k, 1]
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts)  # [E], sum = nb*k
    if _MOE_PROF:
        torch.npu.synchronize()
        _t1 = _time.perf_counter()
    from . import (
        moe_compile,
    )

    if moe_compile._ENABLED:
        # Compiled (shape-generic, no per-shape recompile) fused-w13 experts — kills the 42% recompile.
        out = moe_compile.run(
            w1,
            w3,
            w2,
            routed_input,
            counts,
            experts.swiglu_limit,
            routed_scores.reshape(-1),
        )  # [nb*k, dim]
    else:
        # ★ bf16 experts (was .float()=fp32). Rationale: (1) train/serve CONSISTENCY — the serve
        # runs the draft's MoE in bf16, so fp32-train/bf16-serve is a needless numerical gap (we've
        # been bitten by train/serve convention mismatches before); (2) SPEED — fp32 grouped-GEMM
        # over 256 experts was the bulk of the COMPILE=0 steady cost (~2.2s vs ~1.6s); (3) CORRECTNESS
        # is preserved by AMP option-A: fp32 MASTER weights + optimizer keep the cross-step update
        # accumulation precise, while the matmul (bf16 in, fp32 ACCUMULATE on-device) is where bf16
        # belongs. Mirrors the validated compile path (moe_compile._experts_grouped_mm uses bf16).
        # NOTE: the CPU fp32 oracle (_grouped_matmul_torch) is unchanged — any eager-vs-oracle parity
        # test must use a bf16 tolerance, not bit-exact.
        out = swiglu_grouped(
            routed_input.bfloat16(),
            w1.bfloat16(),
            w3.bfloat16(),
            w2.bfloat16(),
            counts,
            routed_scores.reshape(-1).bfloat16(),
            experts.swiglu_limit,
            _grouped_matmul,
        )  # [nb*k, dim]
    if _MOE_PROF:
        torch.npu.synchronize()
        _t2 = _time.perf_counter()
    unpermuted = torch_npu.npu_moe_token_unpermute(
        out.to(routed_input.dtype), sorted_idx, None
    )
    if _MOE_PROF:
        torch.npu.synchronize()
        _t3 = _time.perf_counter()
        _pm, _gm, _um = (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, (_t3 - _t2) * 1000
        if (
            max(_pm, _gm, _um) > _MOE_PROF_MS
        ):  # spike (or EVERY step if DSPARK_PROFILE_MOE_MS=0)
            try:
                _r = torch.distributed.get_rank()
            except Exception:  # noqa: BLE001
                _r = 0
            print(
                f">>> [MOE-PROF r{_r}] n={n} nb={nb} k={k}  permute+bincount={_pm:.0f}ms  "
                f"experts_gemm={_gm:.0f}ms  unpermute={_um:.0f}ms",
                flush=True,
            )
    if nb > n:
        unpermuted = unpermuted.view(nb, k, dim)[:n].reshape(
            n * k, dim
        )  # drop pad tokens
    return unpermuted


def moe_dispatch_grouped(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: GroupedExperts,
    n_routed_experts: int,
) -> torch.Tensor:
    """Grouped-GEMM equivalent of moe._moe_dispatch_torch. Returns y[T, dim] (fp32).

    ``experts`` holds stacked weights (``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]``);
    no per-forward restacking. Reads ``.to_local()`` when they are Shard(0) DTensors.
    """
    T, dim = x.shape
    device = x.device
    topk = indices.shape[1]

    if device.type == "npu":
        # Fused Ascend routing (no int64-argsort AiCPU fallback); same math as the CPU path below.
        unpermuted = _fused_permute_dispatch_npu(
            x, indices, weights.reshape(-1), experts, n_routed_experts
        )  # [T*topk, dim]
        return unpermuted.view(T, topk, dim).float().sum(dim=1)  # sum the topk slots

    w1, w3, w2 = experts.local_weights()

    # flatten routing: each token -> topk (token, expert, weight); sort by expert
    tok_ids = torch.arange(T, device=device).repeat_interleave(topk)  # [T*topk]
    exp_ids = indices.reshape(-1)  # [T*topk]
    w_flat = weights.reshape(-1).float()  # [T*topk]
    order = torch.argsort(exp_ids, stable=True)
    tok_ids, exp_ids, w_flat = tok_ids[order], exp_ids[order], w_flat[order]
    counts = torch.bincount(exp_ids, minlength=n_routed_experts)  # [E]

    xg = x[tok_ids].float()  # [T*topk, dim]
    out = swiglu_grouped(
        xg,
        w1.float(),
        w3.float(),
        w2.float(),
        counts,
        w_flat,
        experts.swiglu_limit,
        _grouped_matmul,
    )  # [T*topk, dim]

    y = torch.zeros(T, dim, dtype=torch.float32, device=device)
    y.index_add_(0, tok_ids, out)  # sum a token's topk contributions
    return y


register_kernel(_MOE_OP, "npu", moe_dispatch_grouped)


def enable() -> None:
    """Activate the fused grouped-GEMM MoE process-wide (call once at NPU training startup).

    Importing this module already registers the impl under ("moe_dispatch", "npu"); this just
    flips the active backend to "npu" (ops without an npu impl fall back to torch, so it's
    safe to switch globally). Usage in the trainer/model init on NPU:
        from speculators.models.dsv4_dspark.backbone import moe_grouped_gemm
        moe_grouped_gemm.enable()
    """
    from .kernels import set_active_backend

    set_active_backend("npu")
