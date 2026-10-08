import pytest
import torch

from speculators.train.trainer import _clip_grad_norm_or_raise


def test_gradient_clipping_uses_overflow_safe_norm():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 1e30)

    norm = _clip_grad_norm_or_raise(model, 1.0)

    assert norm == pytest.approx(2**0.5 * 1e30, rel=1e-6)
    assert torch.linalg.vector_norm(model.weight.grad).item() == pytest.approx(
        1.0, rel=1e-6
    )


def test_gradient_clipping_leaves_small_gradients_unchanged():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[0.3, 0.4]])

    norm = _clip_grad_norm_or_raise(model, 1.0)

    assert norm == pytest.approx(0.5)
    assert torch.equal(model.weight.grad, torch.tensor([[0.3, 0.4]]))


def test_gradient_clipping_rejects_nonfinite_elements():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("inf"), 0.0]])

    with pytest.raises(FloatingPointError, match="weight"):
        _clip_grad_norm_or_raise(model, 1.0)
