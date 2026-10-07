from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file


_SCRIPT = (
    Path(__file__).parents[3] / "scripts" / "quantize_dspark_for_vllm.py"
)
_SPEC = importlib.util.spec_from_file_location("quantize_dspark_for_vllm", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_weight_map_reads_single_safetensors_file(tmp_path: Path) -> None:
    save_file(
        {
            "mtp.0.main_norm.weight": torch.ones(4),
            "mtp.0.main_proj.weight": torch.ones(4, 4),
        },
        tmp_path / "model.safetensors",
    )

    assert _MODULE._weight_map(tmp_path) == {
        "mtp.0.main_norm.weight": "model.safetensors",
        "mtp.0.main_proj.weight": "model.safetensors",
    }
