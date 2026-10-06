"""Mixture-of-experts FFN for the DSV4 DSpark draft backbone.

``256`` routed experts (SwiGLU with a magnitude clamp) + ``1`` shared expert,
top-``k`` routing with ``sqrtsoftplus`` scoring. The draft layers are all
score-routed (hash routing only applies to the target's first few layers, which
the draft does not include), so no token-id / hash path is needed here.

Routed experts are held as **stacked weights** (:class:`GroupedExperts`:
``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]``) rather than a ``ModuleList`` of
per-expert Linears. This is the layout expert-parallelism and the grouped-matmul
kernel both want (one ``Shard(0)`` DTensor per projection under EP; a single
grouped GEMM instead of a 256-way loop). The heavy dispatch is routed through
:mod:`.kernels` under the op ``moe_dispatch``: the pure-torch reference here loops
over experts (correct, CPU-parity), and the NPU bridge registers a fused
grouped-GEMM (``moe_grouped_gemm``) / expert-parallel all-to-all (``moe_ep``) under
the same op key.
"""

from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from .kernels import get_kernel, torch_kernel

_MOE_OP = "moe_dispatch"


class Router(nn.Module):
    """Score-based top-k router (``sqrtsoftplus``).

    ``bias`` shifts scores for *selection* (top-k) only; the combine
    ``weights`` are gathered from the pre-bias scores, sum-normalized, then
    scaled by ``route_scale`` (this is the non-softmax path).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.topk = cfg.n_activated_experts
        self.route_scale = cfg.route_scale
        self.score_func = cfg.score_func
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))
        # Aux-loss-free load-balancing bias (DeepSeek noaux_tc): shifts the top-k SELECTION only (never
        # the combine weights), nudged by a load-balance RULE at train time — NOT by backprop. A
        # persistent BUFFER (not a Parameter) so FSDP leaves it REPLICATED (not Shard(0)) -> it stays
        # identical across ranks and is updatable in-place with the all-reduced global load.
        self.register_buffer("bias", torch.zeros(cfg.n_routed_experts), persistent=True)
        self.n_routed_experts = cfg.n_routed_experts
        # DSPARK_LOG_EXPERT_LOAD=1 -> stash per-expert selection counts each fwd (diagnostic; core.py).
        self._log_load = os.environ.get("DSPARK_LOG_EXPERT_LOAD") == "1"
        # DSPARK_MOE_BALANCE=1 -> the noaux_tc bias update (update_load_balance_bias, called per step from
        # _backbone_forward): b_i += rate * sign(avg_load - load_i). OFF by default = the historical frozen
        # bias (zero balancing) that let the router COLLAPSE to a few experts. rate via DSPARK_MOE_BALANCE_RATE.
        self._balance = os.environ.get("DSPARK_MOE_BALANCE") == "1"
        self._balance_rate = float(os.environ.get("DSPARK_MOE_BALANCE_RATE", "1e-3"))
        self._sel_counts: torch.Tensor | None = None
        self._step_load: torch.Tensor | None = None

    def _score(self, scores: torch.Tensor) -> torch.Tensor:
        if self.score_func == "softmax":
            return scores.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return scores.sigmoid()
        return F.softplus(scores).sqrt()  # sqrtsoftplus (DSV4 default)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._score(F.linear(x.float(), self.weight.float()))
        selection = scores + self.bias.float()
        indices = selection.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        if self._log_load or (self._balance and self.training):
            _cnt = (
                torch.bincount(indices.reshape(-1), minlength=self.n_routed_experts)
                .detach()
                .float()
            )
            if (
                self._log_load
            ):  # this rank's per-expert selection histogram (diagnostic)
                self._sel_counts = _cnt
            if (
                self._balance and self.training
            ):  # stash for the per-step bias update (SET, so a recompute
                self._step_load = (
                    _cnt  # forward just re-sets the same value -> no double count)
                )
        return weights, indices

    @torch.no_grad()
    def update_load_balance_bias(self) -> None:
        """DeepSeek noaux_tc load balancing: nudge the (non-grad) selection ``bias`` toward uniform expert
        load WITHOUT an aux loss. ``b_i += rate * sign(avg_load - load_i)`` (under-loaded up, over-loaded
        down). Called ONCE per step from ``_backbone_forward``. Under EP the router runs on each rank's
        DP-shard of tokens, so all-reduce the counts to the GLOBAL load — the bias is REPLICATED and must
        update identically on every rank. No-op when balancing is off or no load was stashed."""
        if not self._balance or self._step_load is None:
            return
        load = self._step_load
        import torch.distributed as dist  # noqa: PLC0415

        if dist.is_initialized():
            dist.all_reduce(
                load, op=dist.ReduceOp.SUM
            )  # -> global per-expert load (same on all ranks)
        delta = self._balance_rate * torch.sign(load.mean() - load)
        delta = (
            delta - delta.mean()
        )  # zero-mean the step (torchtitan-canonical): keeps the bias CENTERED
        self.bias.add_(
            delta.to(self.bias.dtype)
        )  # so it can't drift as a whole over a long run
        self._step_load = None


class Expert(nn.Module):
    """SwiGLU expert with an optional magnitude clamp (fp32 activations).

    Used for the single **shared** expert (a plain dense FFN). The routed experts
    live in :class:`GroupedExperts` as stacked weights.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, inter_dim, bias=False)  # up
        self.w2 = nn.Linear(inter_dim, dim, bias=False)  # down
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weight is not None:
            h = weight * h
        return self.w2(h.to(dtype))


def swiglu_grouped(
    xg: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    counts: torch.Tensor,
    w_flat: torch.Tensor,
    swiglu_limit: float,
    matmul,
) -> torch.Tensor:
    """SwiGLU over token groups, one group per (local) expert. Shared by the eager
    torch reference and the NPU grouped-GEMM / EP kernels (they differ only in
    ``matmul``: a per-group loop vs a fused ``npu_grouped_matmul``).

    ``xg[N, dim]`` are routed tokens already sorted by expert; ``counts[E]`` the
    per-expert token count; ``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]`` the
    stacked expert weights; ``w_flat[N]`` the per-token router weight (applied
    pre-down, matching :class:`Expert`). ``matmul(x, W, counts)`` does the grouped
    ``x @ W`` with ``W[E, in, out]``.
    """
    gate = matmul(xg, w1.transpose(1, 2), counts)  # [N, inter]
    up = matmul(xg, w3.transpose(1, 2), counts)
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    h = F.silu(gate) * up
    h = w_flat[:, None] * h
    return matmul(h, w2.transpose(1, 2), counts)  # [N, dim]


class GroupedExperts(nn.Module):
    """Routed experts as stacked weights (``w1/w3 [E, inter, dim]``, ``w2 [E, dim, inter]``).

    ``E`` is the number of experts held **locally**: all ``n_routed_experts`` without
    EP, or ``n_routed_experts // ep_size`` under expert-parallelism (each rank owns a
    disjoint slice, seeded per-rank so the shards don't init identically). Under EP the
    stacked weights are wrapped as ``Shard(0)`` DTensors at setup so the optimizer /
    clip / checkpoint see uniform DTensors; the forward reads ``.to_local()``.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_local: int,
        swiglu_limit: float,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_local_experts = n_local
        self.swiglu_limit = swiglu_limit
        # Weight layout [out, in] per expert (like nn.Linear.weight), stacked over experts,
        # so a checkpoint round-trips to/from per-expert ``experts.{i}.w{1,2,3}.weight`` by a
        # plain stack/index.
        self.w1 = nn.Parameter(torch.empty(n_local, inter_dim, dim))  # gate
        self.w3 = nn.Parameter(torch.empty(n_local, inter_dim, dim))  # up
        self.w2 = nn.Parameter(torch.empty(n_local, dim, inter_dim))  # down
        # nn.Linear default init is kaiming_uniform_(a=sqrt(5)) -> U(-1/sqrt(fan_in)); apply
        # it per-expert (fan_in = dim for w1/w3, inter for w2 -- same for every expert, so a
        # single fill matches). Per-rank seed under EP so disjoint shards differ.
        b1, b2 = 1.0 / math.sqrt(dim), 1.0 / math.sqrt(inter_dim)
        ctx = (
            torch.random.fork_rng(devices=[])
            if seed is not None
            else contextlib.nullcontext()
        )
        with ctx, torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            self.w1.uniform_(-b1, b1)
            self.w3.uniform_(-b1, b1)
            self.w2.uniform_(-b2, b2)

    def local_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(w1, w3, w2) as plain local tensors (``.to_local()`` when Shard(0) DTensors)."""

        def loc(p):
            return p.to_local() if hasattr(p, "to_local") else p

        return loc(self.w1), loc(self.w3), loc(self.w2)


@torch_kernel(_MOE_OP)
def _moe_dispatch_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    experts: GroupedExperts,
    n_routed_experts: int,
) -> torch.Tensor:
    """Route ``x [tokens, dim]`` to its top-k experts and combine (fp32 accum).

    Pure-torch parity reference (CPU / any accelerator): loops over ALL experts in a
    fixed order, feeding each only its routed tokens. ``experts`` holds the full stacked
    routed-expert weights (non-EP); the accelerator bridge registers a fused grouped
    GEMM (``moe_grouped_gemm``) and an expert-parallel all-to-all (``moe_ep``) under the
    same op for throughput / EP.
    """
    w1, w3, w2 = experts.local_weights()
    lim = experts.swiglu_limit
    y = torch.zeros_like(x, dtype=torch.float32)
    for e in range(n_routed_experts):
        tok, slot = torch.where(indices == e)
        xe = x[tok].float()
        gate = xe @ w1[e].t().float()
        up = xe @ w3[e].t().float()
        if lim > 0:
            up = torch.clamp(up, min=-lim, max=lim)
            gate = torch.clamp(gate, max=lim)
        h = F.silu(gate) * up
        h = weights[tok, slot, None].float() * h
        y[tok] += h @ w2[e].t().float()
    return y


class MoE(nn.Module):
    """Routed + shared mixture of experts."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.dim = cfg.hidden_size
        self.n_routed_experts = (
            cfg.n_routed_experts
        )  # GLOBAL count (router + dispatch use it)
        self.router = Router(cfg)
        # Expert-parallel: build only THIS rank's slice of stacked experts
        # (n_routed_experts // EP), seeded per-rank. self.experts.num_local_experts is the
        # local count; ep_expert_offset maps local idx -> global expert id. Non-EP builds
        # all n_routed_experts.
        from . import (
            moe_ep,
        )

        ep = moe_ep._EP
        if ep is not None and ep.size > 1:
            n_local = cfg.n_routed_experts // ep.size
            self.ep_expert_offset = ep.rank * n_local
            seed = 0xE9E9 + ep.rank
        else:
            n_local = cfg.n_routed_experts
            self.ep_expert_offset = 0
            seed = None
        self.experts = GroupedExperts(
            cfg.hidden_size, cfg.moe_inter_dim, n_local, cfg.swiglu_limit, seed=seed
        )
        # The released config carries exactly one shared expert; the checkpoint key is
        # ``ffn.shared_experts.{w1,w2,w3}`` (a single expert), so this is a single module.
        if cfg.n_shared_experts != 1:
            raise ValueError(
                "only n_shared_experts == 1 is supported (matches the release)."
            )
        self.shared_experts = Expert(
            cfg.hidden_size, cfg.moe_inter_dim, cfg.swiglu_limit
        )

    def forward(self, x: torch.Tensor, backend: str | None = None) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.router(x)
        y = get_kernel(_MOE_OP, backend)(
            x, weights, indices, self.experts, self.n_routed_experts
        )
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)
