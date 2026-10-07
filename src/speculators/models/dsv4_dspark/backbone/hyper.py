"""Manifold-constrained hyper-connections (mHC) for the DSV4 DSpark backbone.

Instead of a single residual stream, each block keeps ``hc_mult`` copies. At a
sublayer site (attention / MoE) a :class:`HyperConnection`:

  * collapses the ``hc_mult`` streams to one input for the sublayer (``pre``
    weights), and
  * returns per-stream placement weights ``post`` and a Sinkhorn-projected
    doubly-stochastic stream-mixing matrix ``comb`` used by :func:`place` to
    fold the sublayer output back into the multi-stream residual.

DeepSeek V4.1 reuses the pre-mix computed at the final block's FFN boundary to
collapse the residual streams before the shared norm + lm_head.  It does not
have a separate learned terminal hyper-connection head.

The math follows the reference exactly: ``pre = σ(·)+ε``, ``post = 2·σ(·)``
(no ε), ``comb = softmax(·)+ε`` then Sinkhorn-Knopp (one column normalization,
then ``iters-1`` row/column passes). The Sinkhorn is the mHC-specific NPU
insertion point — dispatched via :mod:`.kernels` under ``mhc_hyper_connection``;
the torch reference is below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .kernels import get_kernel, torch_kernel
from .norm import UnweightedRMSNorm

_HC_OP = "mhc_hyper_connection"


@torch_kernel(_HC_OP)
def _hyper_connection_torch(
    module: HyperConnection, streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference mHC forward, including the carried ``pre`` weights.

    ``post [B, S, hc]``, ``comb [B, S, hc, hc]`` (doubly-stochastic),
    ``collapsed [B, S, D]``. Reads parameters off ``module`` so a bridge can
    share the signature.
    """
    hc = module.hc_mult
    eps = module.hc_eps
    flat = module.input_norm(streams.flatten(start_dim=2).float())
    mix = F.linear(flat, module.fn.float())
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = module.base.float().split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = module.scale.float().unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    # Sinkhorn-Knopp toward doubly-stochastic: start with one column pass
    # (the softmax already made rows sum ~1), then iters-1 row/column passes.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    return post, comb, collapsed, pre


def place(
    out: torch.Tensor,
    residual_streams: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Fold a sublayer output back into the multi-stream residual.

    ``out [B, S, D]``, ``residual_streams [B, S, hc, D]``, ``post [B, S, hc]``,
    ``comb [B, S, hc, hc]`` -> new streams ``[B, S, hc, D]``:
    ``new[j] = post[j]·out + Σ_m comb[m, j]·residual[m]`` (matches the reference
    ``sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)``).
    """
    placed = post.unsqueeze(-1) * out.unsqueeze(-2)
    mixed = torch.sum(comb.unsqueeze(-1) * residual_streams.unsqueeze(-2), dim=2)
    return (placed + mixed).type_as(out)


class HyperConnection(nn.Module):
    """One mHC site (attention or FFN)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.input_norm = UnweightedRMSNorm(cfg.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * cfg.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))  # index 0=pre, 1=post, 2=comb

    def forward(
        self,
        streams: torch.Tensor,
        backend: str | None = None,
        *,
        return_pre: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        result = get_kernel(_HC_OP, backend)(self, streams)
        if len(result) == 4:
            post, comb, collapsed, pre = result
        else:
            # Accelerator bridges written against the original three-tensor
            # contract remain valid. Only the final FFN needs this fallback.
            post, comb, collapsed = result
            pre = self.pre_weights(streams) if return_pre else None
        if return_pre:
            assert pre is not None
            return post, comb, collapsed, pre
        return post, comb, collapsed

    def pre_weights(self, streams: torch.Tensor) -> torch.Tensor:
        """Return the collapse weights produced at this mHC boundary.

        vLLM carries these weights from one sublayer boundary to the next and
        reuses the final FFN boundary's weights for the V4.1 output collapse.
        The training reference computes them explicitly so the same tensor is
        available after the final FFN has been placed into the streams.
        """
        flat = self.input_norm(streams.flatten(start_dim=2).float())
        pre_logits = F.linear(flat, self.fn[: self.hc_mult].float())
        return (
            torch.sigmoid(
                pre_logits * self.scale[0].float() + self.base[: self.hc_mult].float()
            )
            + self.hc_eps
        )


def collapse_streams(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Collapse V4.1 residual streams with a carried FFN pre-mix."""
    return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
