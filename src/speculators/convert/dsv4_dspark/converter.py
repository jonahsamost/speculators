"""Convert a fused DeepSeek V4.1 ``mtp.*`` DSpark head for fine-tuning."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig

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
        self._verify_source(text)

        verifier_config = AutoConfig.from_pretrained(base_model).get_text_config()
        draft_layers = int(text["num_nextn_predict_layers"])
        verifier_config.num_hidden_layers = draft_layers
        # Transformers 5 validates this derived Qwen metadata on reload. The
        # adapter describes the 40-layer verifier, but this nested config is
        # only a shape carrier for the three native draft blocks.
        if hasattr(verifier_config, "layer_types"):
            verifier_config.layer_types = ["full_attention"] * draft_layers
        verifier_config.hidden_size = int(text["hidden_size"])
        verifier_config.vocab_size = int(text["vocab_size"])
        verifier_config.rms_norm_eps = float(text["rms_norm_eps"])

        block_size = int(text["dspark_block_size"])
        config = DSV4DSparkConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=int(text["vocab_size"]),
            block_size=block_size,
            aux_hidden_state_layer_ids=list(text["dspark_target_layer_ids"]),
            mask_token_id=int(text["dspark_noise_token_id"]),
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
            o_groups=8,
            window_size=int(text["sliding_window"]),
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
            score_func="sqrtsoftplus",
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

        logger.info("Dequantizing native DeepSeek V4.1 DSpark weights to BF16")
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
            "hc_head.hc_fn",
            "hc_head.hc_base",
            "hc_head.hc_scale",
        }
        critical = sorted(set(missing) - allowed_missing)
        if critical:
            raise ValueError(
                f"native DSpark conversion left trainable weights missing: {critical}"
            )

        # V4.1 does not serialize a separate final mHC collapse. Start it as a
        # neutral mean over the four residual streams; it remains trainable.
        with torch.no_grad():
            model.hc_head.hc_fn.zero_()
            target = 1.0 / config.hc_mult - config.hc_eps
            model.hc_head.hc_base.fill_(math.log(target / (1.0 - target)))
            model.hc_head.hc_scale.zero_()
        logger.warning(
            "The source checkpoint omits the final mHC collapse; initialized it "
            "to a neutral stream mean. Validate converted-vs-native logits before "
            "a production fine-tune."
        )

        model.to(dtype=torch.bfloat16)
        model.save_pretrained(str(output_path))
        logger.success("Saved trainable native DSpark checkpoint to {}", output_path)
        if validate:
            DSV4DSparkDraftModel.from_pretrained(
                str(output_path), verifier=base_model, local_files_only=True
            )

    @staticmethod
    def _verify_source(text: dict) -> None:
        required = {
            "hidden_size",
            "vocab_size",
            "num_nextn_predict_layers",
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
            "dspark_markov_rank",
            "dspark_n_routed_experts",
            "dspark_num_experts_per_tok",
        }
        missing = sorted(required - text.keys())
        if missing:
            raise ValueError(
                "checkpoint is not a fused DeepSeek V4 DSpark model; missing "
                f"config fields: {missing}"
            )
