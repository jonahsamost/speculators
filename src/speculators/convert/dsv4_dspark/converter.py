"""Convert a fused DeepSeek V4.1 ``mtp.*`` DSpark head for fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.utils import ensure_checkpoint_is_local, load_checkpoint_config
from speculators.models.dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel
from speculators.models.dsv4_dspark.weights import load_released_state_dict
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = ["DSV4DSparkConverter"]


class DSV4DSparkConverter:
    """Extract and dequantize the verifier's already-trained native drafter.

    The fused serving checkpoint stores dense draft matrices as block-scaled
    FP8 and routed experts as packed MXFP4. Training uses BF16 parameters, so
    conversion is deliberately an offline, one-time operation.
    """

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        validate: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source = load_checkpoint_config(local_path)
        text = source.get("text_config", source)

        verifier_config = AutoConfig.from_pretrained(base_model).get_text_config()
        config = self._build_config(text, verifier_config, base_model)
        logger.info(
            "Native DSpark contract: layers={}, block={}, window={}, aux_layers={}, "
            "hidden={}, experts={}x top-{}",
            config.transformer_layer_config.num_hidden_layers,
            config.block_size,
            config.window_size,
            config.aux_hidden_state_layer_ids,
            config.transformer_layer_config.hidden_size,
            config.n_routed_experts,
            config.n_activated_experts,
        )

        logger.info("Dequantizing native DeepSeek V4.1 DSpark weights for training")
        state = load_released_state_dict(local_path, config.backbone_config())
        model = DSV4DSparkDraftModel(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected converted DSpark keys: {unexpected}")

        allowed_missing = {
            "embed_tokens.weight",
            "lm_head.weight",
            "verifier_lm_head.weight",
            "verifier_norm.weight",
        }
        critical = sorted(set(missing) - allowed_missing)
        if critical:
            raise ValueError(
                f"native DSpark conversion left trainable weights missing: {critical}"
            )

        # Training is deliberately uniform BF16. The vLLM export step restores
        # the released checkpoint's per-tensor inference dtypes and quantization.
        model.to(dtype=torch.bfloat16)
        model.save_pretrained(str(output_path))
        logger.success("Saved trainable native DSpark checkpoint to {}", output_path)
        if validate:
            DSV4DSparkDraftModel.from_pretrained(
                str(output_path), verifier=base_model, local_files_only=True
            )

    def _build_config(
        self,
        text: dict,
        verifier_config: PretrainedConfig,
        base_model: str,
    ) -> DSV4DSparkConfig:
        """Build the trainable draft config from the fused serving config.

        The fused checkpoint is the architecture authority.  In particular,
        ``sliding_window`` configures every native DSpark layer in vLLM because
        those layers occupy the checkpoint's trailing ``compress_ratios == 0``
        positions.  Keep both the generic Speculators layer metadata and the
        native backbone field synchronized so training cannot silently fall
        back to full attention.
        """
        self._verify_source(text)
        draft_layers = int(text["num_nextn_predict_layers"])
        verifier_config.num_hidden_layers = draft_layers
        window_size = int(text["sliding_window"])
        # This nested config is a shape carrier for the native draft blocks,
        # not a description of the verifier's heterogeneous attention stack.
        # DFlash scaffolding dispatches its masks through these three fields.
        verifier_config.layer_types = ["sliding_attention"] * draft_layers
        verifier_config.sliding_window = window_size
        verifier_config.use_sliding_window = True
        verifier_config.hidden_size = int(text["hidden_size"])
        verifier_config.vocab_size = int(text["vocab_size"])
        verifier_config.rms_norm_eps = float(text["rms_norm_eps"])

        block_size = int(text["dspark_block_size"])
        return DSV4DSparkConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=int(text["vocab_size"]),
            block_size=block_size,
            aux_hidden_state_layer_ids=list(text["dspark_target_layer_ids"]),
            mask_token_id=int(text["dspark_noise_token_id"]),
            sliding_window_non_causal=True,
            sample_from_anchor=True,
            markov_rank=int(text["dspark_markov_rank"]),
            markov_head_type="vanilla",
            enable_confidence_head=True,
            confidence_head_with_markov=True,
            confidence_head_bias=False,
            num_heads=int(text["num_attention_heads"]),
            head_dim=int(text["head_dim"]),
            rope_head_dim=int(text["qk_rope_head_dim"]),
            q_lora_rank=int(text["q_lora_rank"]),
            o_lora_rank=int(text["o_lora_rank"]),
            o_groups=int(text["o_groups"]),
            window_size=window_size,
            rope_theta=float(text["rope_theta"]),
            rope_factor=float(text["rope_scaling"]["factor"]),
            original_seq_len=int(
                text["rope_scaling"]["original_max_position_embeddings"]
            ),
            beta_fast=float(text["rope_scaling"]["beta_fast"]),
            beta_slow=float(text["rope_scaling"]["beta_slow"]),
            n_routed_experts=int(text["dspark_n_routed_experts"]),
            n_shared_experts=int(text["n_shared_experts"]),
            n_activated_experts=int(text["dspark_num_experts_per_tok"]),
            moe_inter_dim=int(text["moe_intermediate_size"]),
            score_func=str(text["scoring_func"]),
            route_scale=float(text["routed_scaling_factor"]),
            swiglu_limit=float(text["swiglu_limit"]),
            hc_mult=int(text["hc_mult"]),
            hc_sinkhorn_iters=int(text["hc_sinkhorn_iters"]),
            hc_eps=float(text["hc_eps"]),
            speculators_config=SpeculatorsConfig(
                algorithm="dsv4_dspark",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=block_size)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(base_model),
            ),
        )

    @staticmethod
    def _verify_source(text: dict) -> None:
        required = {
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_nextn_predict_layers",
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
            "dspark_markov_rank",
            "dspark_n_routed_experts",
            "dspark_num_experts_per_tok",
            "sliding_window",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "rope_theta",
            "rope_scaling",
            "n_shared_experts",
            "moe_intermediate_size",
            "scoring_func",
            "routed_scaling_factor",
            "swiglu_limit",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "rms_norm_eps",
            "compress_ratios",
            "hidden_act",
            "topk_method",
            "norm_topk_prob",
        }
        missing = sorted(required - text.keys())
        if missing:
            raise ValueError(
                "checkpoint is not a fused DeepSeek V4 DSpark model; missing "
                f"config fields: {missing}"
            )
        rope_required = {
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
        }
        rope_missing = sorted(rope_required - text["rope_scaling"].keys())
        if rope_missing:
            raise ValueError(
                "checkpoint has incomplete DSpark rope_scaling; missing fields: "
                f"{rope_missing}"
            )

        draft_layers = int(text["num_nextn_predict_layers"])
        target_layers = int(text["num_hidden_layers"])
        compress_ratios = list(text["compress_ratios"])
        draft_ratios = compress_ratios[target_layers : target_layers + draft_layers]
        if len(draft_ratios) != draft_layers or any(draft_ratios):
            raise ValueError(
                "native DSpark draft layers must be the trailing sliding-window-only "
                "layers (compress_ratio=0); got "
                f"compress_ratios[{target_layers}:{target_layers + draft_layers}]="
                f"{draft_ratios}"
            )
        target_layer_ids = list(text["dspark_target_layer_ids"])
        if not target_layer_ids or any(
            layer_id < 0 or layer_id >= target_layers for layer_id in target_layer_ids
        ):
            raise ValueError(
                "dspark_target_layer_ids must identify verifier layers in "
                f"[0, {target_layers}); got {target_layer_ids}"
            )
        for positive_field in (
            "sliding_window",
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_n_routed_experts",
            "dspark_num_experts_per_tok",
        ):
            if int(text[positive_field]) <= 0:
                raise ValueError(f"{positive_field} must be positive")
        supported_values = {
            "num_key_value_heads": 1,
            "hidden_act": "silu",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
        }
        unsupported = {
            key: (text[key], expected)
            for key, expected in supported_values.items()
            if text[key] != expected
        }
        if unsupported:
            raise ValueError(
                "unsupported native DSpark architecture values "
                f"(got, expected): {unsupported}"
            )
