"""Unit tests for MTP checkpoint stitching helpers."""

import torch

from speculators.cli.stitch import _filter_auxiliary_keys


def test_stitch_excludes_non_native_confidence_parameters() -> None:
    weights = {
        "mtp_layers.0.input_proj.weight": torch.ones(1),
        "confidence_head.proj.weight": torch.ones(1),
        "confidence_head.proj.bias": torch.ones(1),
        "confidence_step_embeddings.weight": torch.ones(1),
    }

    assert set(_filter_auxiliary_keys(weights)) == {"mtp_layers.0.input_proj.weight"}
