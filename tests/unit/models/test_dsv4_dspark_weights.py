import torch

from speculators.models.dsv4_dspark.weights import (
    dequantize_released_weight,
    map_released_key,
)


def test_v41_native_key_mapping() -> None:
    assert map_released_key("mtp.0.main_proj.weight") == "fc.weight"
    assert map_released_key("mtp.0.main_norm.weight") == "hidden_norm.weight"
    assert (
        map_released_key("mtp.2.markov_head.embed.weight")
        == "markov_head.markov_w1.weight"
    )
    assert (
        map_released_key("mtp.2.markov_head.head.weight")
        == "markov_head.markov_w2.weight"
    )
    assert map_released_key("mtp.1.ffn.gate.weight") == "layers.1.ffn.router.weight"
    assert map_released_key("mtp.1.ffn.gate.bias_vl") is None


def test_mxfp4_dequantizes_packed_nibbles() -> None:
    # low nibble=+1, high nibble=-1; one E8M0 scale block of 2**1.
    packed = torch.full((1, 16), 0xA2, dtype=torch.uint8).view(torch.int8)
    scale = torch.tensor([[128]], dtype=torch.uint8)

    got = dequantize_released_weight(packed, scale, dtype=torch.float32)

    assert got.shape == (1, 32)
    assert torch.equal(got[0, 0::2], torch.full((16,), 2.0))
    assert torch.equal(got[0, 1::2], torch.full((16,), -2.0))


def test_fp8_dequantizes_32_by_32_tiles() -> None:
    weight = torch.ones((32, 64), dtype=torch.float8_e4m3fn)
    # First tile scale 1, second tile scale 2.
    scale = torch.tensor([[127, 128]], dtype=torch.uint8)

    got = dequantize_released_weight(weight, scale, dtype=torch.float32)

    assert torch.equal(got[:, :32], torch.ones((32, 32)))
    assert torch.equal(got[:, 32:], torch.full((32, 32), 2.0))
