"""Unit tests for shared acceptance-confidence primitives."""

import torch

from speculators.models.confidence import (
    ConfidenceHead,
    masked_confidence_loss,
    sparse_distribution_overlap,
)


def test_confidence_head_projects_each_position() -> None:
    head = ConfidenceHead(7)

    assert head(torch.randn(2, 3, 7)).shape == (2, 3)


def test_sparse_overlap_uses_only_runtime_candidates() -> None:
    verifier_logits = torch.tensor([[[8.0, 7.0, -8.0, -8.0]]])
    candidate_ids = torch.tensor([[[2, 3]]])
    candidate_logits = torch.tensor([[[0.0, 0.0]]])

    overlap = sparse_distribution_overlap(
        candidate_ids, candidate_logits, verifier_logits
    )

    assert float(overlap.item()) < 1e-5


def test_masked_confidence_loss_masks_gradients() -> None:
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])

    loss = masked_confidence_loss(logits, targets, mask)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0] != 0
    assert logits.grad[0, 1] == 0


def test_masked_confidence_loss_computes_in_fp32() -> None:
    logits = torch.tensor([[20.0, -20.0]], dtype=torch.bfloat16, requires_grad=True)
    # Targets are analytical probabilities, so tolerate small numerical drift
    # at the boundary by clamping them to their documented domain.
    targets = torch.tensor([[1.01, -0.01]], dtype=torch.bfloat16)
    mask = torch.ones_like(targets)

    loss = masked_confidence_loss(logits, targets, mask)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
