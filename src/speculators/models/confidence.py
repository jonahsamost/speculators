"""Shared acceptance-confidence heads and loss helpers."""

import torch
from torch import nn
from torch.nn import functional

__all__ = [
    "ConfidenceHead",
    "masked_confidence_loss",
    "sparse_distribution_overlap",
]


class ConfidenceHead(nn.Module):
    """Project per-position draft features to one acceptance logit."""

    def __init__(self, input_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def masked_confidence_loss(
    confidence_logits: torch.Tensor,
    acceptance_targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-target BCE averaged over valid positions.

    ``acceptance_targets`` must be detached by the caller. This keeps the target
    definition explicit at the algorithm boundary and prevents this helper from
    silently changing which draft tensors receive gradients.
    """

    # BCE's CUDA bfloat16 backward can produce NaNs for otherwise finite logits
    # and soft targets.  Keep this small scalar-head objective in fp32; the cast
    # backward still delivers gradients to a lower-precision producer safely.
    logits_fp32 = confidence_logits.float()
    targets_fp32 = acceptance_targets.detach().float().clamp(0.0, 1.0)
    mask = loss_mask.float()
    elementwise = functional.binary_cross_entropy_with_logits(
        logits_fp32,
        targets_fp32,
        reduction="none",
    )
    if weights is not None:
        elementwise = elementwise * weights.to(elementwise.dtype)
    return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)


def sparse_distribution_overlap(
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    verifier_logits: torch.Tensor,
) -> torch.Tensor:
    """Return ``sum_v min(q_v, p_v)`` for a sparse candidate distribution.

    ``candidate_logits`` define the normalized draft distribution ``q`` over
    ``candidate_ids``. The verifier distribution ``p`` remains normalized over
    its full vocabulary; only its mass at the draft-supported IDs contributes.
    """

    draft_prob = candidate_logits.float().softmax(dim=-1)
    verifier_log_normalizer = torch.logsumexp(verifier_logits.float(), dim=-1)
    verifier_candidate_prob = torch.exp(
        verifier_logits.float().gather(-1, candidate_ids.long())
        - verifier_log_normalizer.unsqueeze(-1)
    )
    return torch.minimum(draft_prob, verifier_candidate_prob).sum(dim=-1)
