"""Config-contract tests for the native DeepSeek V4.1 DSpark converter."""

from unittest.mock import patch

import pytest
from transformers import Qwen3Config

from speculators.config import VerifierConfig
from speculators.convert.dsv4_dspark.converter import DSV4DSparkConverter
from speculators.models.dsv4_dspark import DSV4DSparkConfig


def _released_v41_text_config() -> dict:
    """Architecture fields from deepseek-ai/DeepSeek-V4.1-Flash config.json."""
    return {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 5120,
        "rms_norm_eps": 1e-20,
        "num_hidden_layers": 40,
        "compress_ratios": [0] * 2 + [2] * 38 + [0] * 3,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "sliding_window": 128,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        "n_shared_experts": 1,
        "moe_intermediate_size": 2304,
        "scoring_func": "sqrtsoftplus",
        "hidden_act": "silu",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }


def _shape_carrier() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=129280,
        hidden_size=5120,
        intermediate_size=2304,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=80,
        rms_norm_eps=1e-20,
    )


def _build_config(source: dict | None = None) -> DSV4DSparkConfig:
    verifier = VerifierConfig(
        name_or_path="deepseek-ai/DeepSeek-V4.1-Flash",
        architectures=["DeepseekV41ForCausalLM"],
    )
    with patch.object(VerifierConfig, "from_pretrained", return_value=verifier):
        return DSV4DSparkConverter()._build_config(
            source or _released_v41_text_config(),
            _shape_carrier(),
            "deepseek-ai/DeepSeek-V4.1-Flash",
        )


def test_build_config_matches_released_v41_dspark() -> None:
    config = _build_config()

    tl = config.transformer_layer_config
    assert tl.num_hidden_layers == 3
    assert tl.layer_types == ["sliding_attention"] * 3
    assert tl.sliding_window == config.window_size == 128
    assert tl.use_sliding_window is True
    assert config.sliding_window_non_causal is True
    assert tl.hidden_size == 5120
    assert config.block_size == 5
    assert config.aux_hidden_state_layer_ids == [37, 38, 39]
    assert config.markov_rank == 256
    assert config.q_lora_rank == 1280
    assert config.n_routed_experts == 128
    assert config.n_activated_experts == 3
    assert config.moe_inter_dim == 2304
    assert config.enable_confidence_head is True
    assert config.confidence_head_with_markov is True
    assert config.confidence_head_bias is False


def test_incomplete_released_config_is_rejected() -> None:
    source = _released_v41_text_config()
    del source["dspark_target_layer_ids"]
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        _build_config(source)


def test_non_sliding_native_draft_layout_is_rejected() -> None:
    source = _released_v41_text_config()
    source["compress_ratios"][-1] = 2
    with pytest.raises(ValueError, match="compress_ratio=0"):
        _build_config(source)


def test_native_config_rejects_full_attention_metadata() -> None:
    config = _build_config()
    dumped = config.model_dump()
    dumped["transformer_layer_config"]["layer_types"] = ["full_attention"] * 3

    with pytest.raises(ValueError, match="native DSV4 DSpark requires"):
        DSV4DSparkConfig.model_validate(dumped)
