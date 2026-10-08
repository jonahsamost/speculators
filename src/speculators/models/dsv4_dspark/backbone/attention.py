"""Multi-head latent attention with a per-head learnable sink (DSV4 draft).

Two layers here:

* :func:`sink_block_attention` — the numerical core, dispatched through
  :mod:`.kernels` so an NPU bridge can swap a fused impl in. The torch
  reference is a dense, non-causal sink-softmax over ``[context + block]`` KV,
  bit-exact to the vLLM-Ascend gold reference (validated fwd+bwd on NPU in
  ``examples/ascend_npu_dflash/dspark_attn_ref_bench.py``). Vanilla SDPA cannot
  express the sink (it is an extra term in the softmax denominator), so this
  eager einsum form is the sink-correct path.

* :class:`LatentAttention` — the MLA projection stack (low-rank query
  ``wq_a→q_norm→wq_b`` + per-head RMS, single shared-KV ``wkv→kv_norm``,
  grouped low-rank output ``wo_a→wo_b``) plus the learnable ``sink``. Its
  training forward assembles per-anchor-block queries/keys/values and calls
  :func:`sink_block_attention`; it does not use a KV cache (teacher-forced).

Weight-name note: the released checkpoint stores these as ``wq_a`` / ``wq_b`` /
``wkv`` / ``wo_a`` / ``wo_b`` / ``attn_sink``; the loader maps those onto the
descriptive attribute names here.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from .kernels import get_kernel, torch_kernel
from .norm import RMSNorm, UnweightedRMSNorm
from .rotary import apply_rotary_emb, freqs_cis_from_positions

_SINK_OP = "sink_block_attention"
_SHARED_KV_RANK = 3


def _finite_diag(stage: str, tensor: torch.Tensor) -> None:
    if os.environ.get("DSPARK_FINITE_DIAG") != "1":
        return
    bad = ~torch.isfinite(tensor)
    if not bool(bad.any().item()):
        return
    coordinates = torch.nonzero(bad, as_tuple=False)[:32].tolist()
    raise FloatingPointError(
        f"Non-finite DSV4 DSpark attention substage {stage}: "
        f"count={int(bad.sum())}, first_coordinates={coordinates}"
    )


@torch_kernel(_SINK_OP)
def _sink_block_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense non-causal sink-softmax attention (fp32 accumulation).

    Shapes: ``q [N, Sq, H, D]``, ``k/v [N, Sk, D]`` (a SINGLE shared KV head — the
    DSV4 draft's MLA has one KV head shared across the H query heads), ``sink [H]``,
    optional ``attn_bias [N, Sq, Sk]`` (or broadcastable) additive mask applied to
    the logits *before* the sink term. Returns ``[N, Sq, H, D]``.

    Shared-KV einsums (``nkd``, not ``nkhd``) contract the single KV head against
    every query head directly — mathematically identical to broadcasting K/V to H
    heads first, but WITHOUT materializing the H×-larger ``[N, Sk, H, D]`` tensor
    (−~2.1 GB, ~20× faster fwd; matches the vLLM-Ascend #12005 shared-KV op).

    The sink is a synthetic key whose logit is the per-head ``sink`` scalar; it
    contributes to the softmax denominator but nothing to the value sum:
    ``p_j = exp(s_j) / (Σ_j exp(s_j) + exp(sink))``.
    """
    if k.dim() != _SHARED_KV_RANK:
        raise ValueError(
            "shared-KV sink attention expects k/v [N, Sk, D] "
            f"(single head); got {tuple(k.shape)}"
        )
    s = torch.einsum("nqhd,nkd->nqhk", q.float(), k.float()) * scale
    _finite_diag("scores_before_bias", s)
    if attn_bias is not None:
        s = s + attn_bias.float().unsqueeze(2)  # [N, Sq, 1, Sk] broadcast over heads
    # The mask intentionally contributes -inf; only NaNs and +inf are invalid.
    if os.environ.get("DSPARK_FINITE_DIAG") == "1":
        invalid_scores = torch.isnan(s) | torch.isposinf(s)
        if bool(invalid_scores.any().item()):
            coordinates = torch.nonzero(invalid_scores, as_tuple=False)[:32].tolist()
            raise FloatingPointError(
                "Invalid DSV4 DSpark attention scores after bias: "
                f"count={int(invalid_scores.sum())}, first_coordinates={coordinates}"
            )
    sink_h = sink.float().view(1, 1, -1, 1)
    # Treat the sink as one extra key with a zero value.  torch.softmax uses a
    # numerically hardened backward; differentiating through our former manual
    # ``exp(logit - row_max)`` implementation produced non-finite gradients in
    # the first DSV4 layer at real training scale, despite finite forward values.
    sink_logits = sink_h.expand(*s.shape[:-1], 1)
    p = torch.softmax(torch.cat([s, sink_logits], dim=-1), dim=-1)[..., :-1]
    _finite_diag("softmax_probabilities", p)
    output = torch.einsum("nqhk,nkd->nqhd", p, v.float())
    _finite_diag("value_aggregation_fp32", output)
    output = output.to(q.dtype)
    _finite_diag("value_aggregation_compute_dtype", output)
    return output


def sink_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Dispatch sink attention to the active backend (torch by default)."""
    return get_kernel(_SINK_OP, backend)(q, k, v, sink, scale, attn_bias)


class LatentAttention(nn.Module):
    """MLA + per-head sink for one draft layer (teacher-forced, no KV cache).

    ``forward`` takes the block hidden states (queries) and the context+block
    hidden states (keys/values source) already assembled by the caller, plus
    the absolute positions for each. RoPE values are built inside each layer
    so checkpointed layers never share their materialized rotation storage.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.eps = cfg.rms_norm_eps
        self.scale = cfg.head_dim**-0.5

        self.wq_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.num_heads * cfg.head_dim, bias=False)
        self.q_head_norm = UnweightedRMSNorm(cfg.rms_norm_eps)  # per-head RMS on q
        self.wkv = nn.Linear(
            cfg.hidden_size, cfg.head_dim, bias=False
        )  # single shared KV head
        self.kv_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.wo_a = nn.Linear(
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
            cfg.o_groups * cfg.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size, bias=False
        )
        self.attn_sink = nn.Parameter(torch.zeros(cfg.num_heads))

    def project_q(self, block_x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Project a block to multi-head queries and apply trailing-slice RoPE."""
        rd = self.rope_head_dim
        q = self.q_norm(self.wq_a(block_x))
        q = self.wq_b(q).unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_head_norm(q)
        rope = apply_rotary_emb(q[..., -rd:], freqs_cis)
        return torch.cat([q[..., :-rd], rope], dim=-1)

    def project_kv(self, kv_x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Project input to one shared KV head and apply trailing-slice RoPE."""
        rd = self.rope_head_dim
        kv = self.wkv(kv_x)
        _finite_diag("project_kv_linear", kv)
        kv = self.kv_norm(kv)
        _finite_diag("project_kv_norm", kv)
        _finite_diag("project_kv_rope_frequencies", freqs_cis)
        rope = apply_rotary_emb(kv[..., -rd:], freqs_cis)
        _finite_diag("project_kv_rope", rope)
        return torch.cat([kv[..., :-rd], rope], dim=-1)

    def forward(
        self,
        block_x: torch.Tensor,
        context_x: torch.Tensor,
        block_positions: torch.Tensor,
        context_positions: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block-gamma draft attention (teacher-forced, no KV cache).

        ``block_x [N, gamma, dim]`` are the draft block hidden states (queries +
        block keys/values); ``context_x [N, W, dim]`` is the target-hidden
        context window (``main_x``) providing the sliding-window keys/values.
        Each block query attends densely (non-causal) to ``[context | block]``
        with the per-head sink. Returns ``[N, gamma, dim]``.
        """
        # Keep the complete latent-attention path in fp32.  The surrounding
        # trainer autocasts the draft backbone to bf16, but rare real anchors
        # can overflow a bf16 Q/K/V or grouped output projection even though
        # the mathematically accumulated result is representable.  Scores were
        # already fp32; extending fp32 to the projections removes that partial
        # precision boundary.  Cast only the final residual update back to the
        # caller's compute dtype.
        output_dtype = block_x.dtype
        with torch.autocast(device_type=block_x.device.type, enabled=False):
            block_freqs = freqs_cis_from_positions(
                block_positions, self.rope_head_dim, self.cfg.rope_theta
            )
            context_freqs = freqs_cis_from_positions(
                context_positions, self.rope_head_dim, self.cfg.rope_theta
            )
            _finite_diag("block_rope_frequencies", block_freqs)
            _finite_diag("context_rope_frequencies", context_freqs)
            block_x_fp32 = block_x.float()
            context_x_fp32 = context_x.float()
            q = self.project_q(block_x_fp32, block_freqs)  # [N, gamma, H, D]
            _finite_diag("project_q", q)
            kv_ctx = self.project_kv(context_x_fp32, context_freqs)  # [N, W, D]
            _finite_diag("project_kv_context", kv_ctx)
            kv_blk = self.project_kv(block_x_fp32, block_freqs)  # [N, gamma, D]
            _finite_diag("project_kv_block", kv_blk)
            kv = torch.cat(
                [kv_ctx, kv_blk], dim=1
            )  # [N, W+gamma, D] (single shared KV head)
            # Shared-KV: pass the single KV head directly (NO .expand to H heads).
            # The nkd einsums broadcast it over query heads without materializing
            # the H×-larger [N, Sk, H, D] tensor (matches the serving implementation).
            o = sink_block_attention(q, kv, kv, self.attn_sink, self.scale, attn_bias)
            _finite_diag("sink_attention_output", o)
            output = self.combine_output(o, block_freqs)
            _finite_diag("combined_output_fp32", output)
        output = output.to(output_dtype)
        _finite_diag("combined_output_compute_dtype", output)
        return output

    def combine_output(
        self, o: torch.Tensor, q_freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """Apply the grouped low-rank output projection.

        Input is ``[N, Sq, H, head_dim]`` and output is ``[N, Sq, dim]``. First
        de-rotate the RoPE slice, then apply the grouped ``wo_a`` and ``wo_b``.
        """
        rd = self.rope_head_dim
        derot = apply_rotary_emb(o[..., -rd:], q_freqs_cis, inverse=True)
        _finite_diag("output_derotation", derot)
        o = torch.cat([o[..., :-rd], derot], dim=-1)
        n, sq = o.shape[0], o.shape[1]
        o = o.reshape(n, sq, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.cfg.o_lora_rank, -1)
        o = torch.einsum("nsgd,grd->nsgr", o, wo_a)
        _finite_diag("output_projection_a", o)
        o = self.wo_b(o.flatten(2))
        _finite_diag("output_projection_b", o)
        return o
