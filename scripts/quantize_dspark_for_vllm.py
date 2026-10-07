#!/usr/bin/env python3
"""Requantize a converted BF16 DSpark checkpoint for DeepSeek V4.1 vLLM.

The input must already use the released ``mtp.*`` namespace (produce it with
``convert_dspark_to_vllm.py``). The released DeepSeek-V4.1-Flash-DSpark model
is used as a schema, not as a weight source: each trained matrix is encoded
with the corresponding released tensor's dtype and scale geometry. Dense and
shared-expert matrices become 32x32 FP8 E4M3; routed experts become packed
MXFP4 with an E8M0 scale per row and 32 values.

Example::

    python scripts/quantize_dspark_for_vllm.py \
      --in runs/my-run/checkpoint-vllm-bf16 \
      --reference /home/ubuntu/models/DeepSeek-V4.1-Flash \
      --out runs/my-run/checkpoint-vllm-quantized
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from speculators.models.dsv4_dspark.weights import quantize_released_weight


def _weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = root / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return dict.fromkeys(handle.keys(), single.name)
    raise SystemExit(f"no model.safetensors[.index.json] in {root}")


def _load_state(root: Path, weight_map: dict[str, str]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        state.update(load_file(str(root / shard), device="cpu"))
    return state


def _model_config(config: dict) -> dict:
    nested = config.get("text_config", config)
    if not isinstance(nested, dict):
        raise SystemExit("config.json has a non-object text_config")
    return nested


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_root = Path(args.input)
    reference_root = Path(args.reference)
    output_root = Path(args.out)
    output_file = output_root / "model.safetensors"
    if output_file.exists():
        raise SystemExit(f"refusing to overwrite existing {output_file}")

    input_map = _weight_map(input_root)
    reference_map = _weight_map(reference_root)
    state = _load_state(input_root, input_map)
    if not state or any(key.endswith(".scale") for key in state):
        raise SystemExit("--in must be the converted, unquantized BF16 checkpoint")

    missing_reference = sorted(set(state) - set(reference_map))
    if missing_reference:
        raise SystemExit(
            "trained tensors have no released schema counterpart: "
            + ", ".join(missing_reference[:12])
        )

    by_reference_shard: dict[str, list[str]] = {}
    for key in state:
        by_reference_shard.setdefault(reference_map[key], []).append(key)

    output: dict[str, torch.Tensor] = {}
    quantized_counts = {"fp8": 0, "mxfp4": 0, "passthrough": 0}
    for shard, keys in sorted(by_reference_shard.items()):
        with safe_open(
            str(reference_root / shard), framework="pt", device="cpu"
        ) as handle:
            for key in keys:
                source = state.pop(key)
                template = handle.get_tensor(key)
                if tuple(source.shape) != tuple(
                    template.shape
                ) and template.dtype not in (
                    torch.int8,
                    torch.uint8,
                ):
                    raise SystemExit(
                        f"shape mismatch for {key}: trained {tuple(source.shape)}, "
                        f"released {tuple(template.shape)}"
                    )
                scale_key = key.removesuffix(".weight") + ".scale"
                if key.endswith(".weight") and scale_key in reference_map:
                    scale_shard = reference_map[scale_key]
                    if scale_shard == shard:
                        template_scale = handle.get_tensor(scale_key)
                    else:
                        with safe_open(
                            str(reference_root / scale_shard),
                            framework="pt",
                            device="cpu",
                        ) as scale_handle:
                            template_scale = scale_handle.get_tensor(scale_key)
                    quantized, scale = quantize_released_weight(
                        source, template, template_scale
                    )
                    output[key] = quantized.contiguous()
                    output[scale_key] = scale.contiguous()
                    kind = "fp8" if template.dtype == torch.float8_e4m3fn else "mxfp4"
                    quantized_counts[kind] += 1
                else:
                    if tuple(source.shape) != tuple(template.shape):
                        raise SystemExit(
                            f"shape mismatch for {key}: trained {tuple(source.shape)}, "
                            f"released {tuple(template.shape)}"
                        )
                    output[key] = source.to(template.dtype).contiguous()
                    quantized_counts["passthrough"] += 1

    input_config = json.loads((input_root / "config.json").read_text())
    reference_config = json.loads((reference_root / "config.json").read_text())
    quantization_config = reference_config.get("quantization_config")
    quantization_location = input_config
    if quantization_config is None:
        quantization_config = _model_config(reference_config).get("quantization_config")
        quantization_location = _model_config(input_config)
    if not isinstance(quantization_config, dict):
        raise SystemExit("released reference config has no quantization_config")
    quantization_location["quantization_config"] = quantization_config

    output_root.mkdir(parents=True, exist_ok=True)
    save_file(output, str(output_file), metadata={"format": "pt"})
    (output_root / "config.json").write_text(json.dumps(input_config, indent=2))
    for companion in ("generation_config.json",):
        source = input_root / companion
        if source.exists():
            shutil.copy2(source, output_root / companion)

    manifest = {
        "source": str(input_root.resolve()),
        "reference_schema": str(reference_root.resolve()),
        "tensor_count": len(output),
        **quantized_counts,
    }
    (output_root / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"wrote vLLM DSpark checkpoint to {output_root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
