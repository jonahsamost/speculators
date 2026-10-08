from types import SimpleNamespace

import pytest
from torch import nn

from speculators.train.optimizers import (
    build_optimizers,
    split_named_params_for_muon,
)


class _ModelWithConfidenceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.confidence_head = nn.Linear(4, 1)


def _config(**overrides):
    values = {
        "optimizer": "adamw",
        "lr": 1e-5,
        "weight_decay": 0.01,
        "confidence_head_lr": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_confidence_head_can_use_a_separate_adamw_learning_rate() -> None:
    model = _ModelWithConfidenceHead()
    (optimizer,) = build_optimizers(
        model,
        _config(confidence_head_lr=1e-4),
    )

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-5, 1e-4]
    )
    assert {id(param) for param in optimizer.param_groups[1]["params"]} == {
        id(param) for param in model.confidence_head.parameters()
    }


def test_confidence_head_lr_requires_a_confidence_head() -> None:
    with pytest.raises(ValueError, match="no trainable confidence_head"):
        build_optimizers(nn.Linear(4, 4), _config(confidence_head_lr=1e-4))


def test_confidence_head_parameters_do_not_use_muon() -> None:
    model = _ModelWithConfidenceHead()

    muon, adamw = split_named_params_for_muon(model)

    assert not any("confidence_head" in name for name, _ in muon)
    assert {name for name, _ in adamw if "confidence_head" in name} == {
        "confidence_head.weight",
        "confidence_head.bias",
    }
