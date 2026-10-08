"""DSV4 DSpark trains each anchor as a window-local attention problem."""

from types import SimpleNamespace

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import eager
from speculators.models.dsv4_dspark.backbone.attention import LatentAttention
from speculators.models.dsv4_dspark.backbone.rotary import (
    freqs_cis_from_positions,
    precompute_freqs_cis,
)
from speculators.models.dsv4_dspark.core import (
    DSV4DSparkConfig,
    DSV4DSparkDraftModel,
)


def _real_freqs(length: int, dim: int) -> torch.Tensor:
    return torch.view_as_real(
        precompute_freqs_cis(dim, length, 0, 10_000.0, 1.0, 32.0, 1.0)
    )


def test_anchor_local_context_respects_window_and_document_boundaries():
    # Two packed documents: [0..6] and [7..11]. The second anchor has only
    # three preceding tokens in its own document, so the first local slot is
    # padding even though position 6 exists in the packed tensor.
    document_ids = torch.tensor([0] * 7 + [1] * 5)
    indices, valid = DSV4DSparkDraftModel._anchor_local_context(
        anchor_positions=torch.tensor([5, 10, 0]),
        anchor_valid=torch.tensor([True, True, False]),
        document_ids=document_ids,
        window_size=4,
    )

    assert torch.equal(indices[0], torch.tensor([1, 2, 3, 4]))
    assert bool(valid[0].all())
    assert torch.equal(indices[1], torch.tensor([6, 7, 8, 9]))
    assert torch.equal(valid[1], torch.tensor([False, True, True, True]))
    assert not bool(valid[2].any())


def test_anchor_local_attention_matches_full_sequence_mask():
    """Compact [A,B,W+B] attention equals the former sequence-sized mask."""
    torch.manual_seed(7)
    cfg = SimpleNamespace(
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=12,
        o_groups=2,
        o_lora_rank=4,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
    )
    attention = LatentAttention(cfg)

    total_seq_len, window, block = 12, 4, 3
    anchors = torch.tensor([5, 10])
    anchor_valid = torch.ones(2, dtype=torch.bool)
    document_ids = torch.tensor([0] * 7 + [1] * 5)
    # Position IDs reset at the packed-document boundary, as they do in the
    # training collator.
    position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4])
    block_positions = position_ids[anchors, None] + torch.arange(block)

    context = torch.randn(1, total_seq_len, cfg.hidden_size)
    block_x = torch.randn(anchors.numel(), block, cfg.hidden_size)
    indices, context_valid = DSV4DSparkDraftModel._anchor_local_context(
        anchors, anchor_valid, document_ids, window
    )
    local_context = context[0, indices]
    local_bias = DSV4DSparkDraftModel._anchor_local_attention_bias(
        context_valid, block, context.dtype
    )
    compact = attention(
        block_x,
        local_context,
        block_positions,
        position_ids[indices],
        local_bias,
    )

    # Reference the previous representation: one flattened query sequence,
    # the complete packed base sequence, and every synthetic block. Everything
    # outside an anchor's local context and own block is masked to -inf.
    num_anchors = anchors.numel()
    full_bias = torch.full(
        (1, num_anchors * block, total_seq_len + num_anchors * block),
        float("-inf"),
    )
    for anchor_index in range(num_anchors):
        query_slice = slice(anchor_index * block, (anchor_index + 1) * block)
        full_bias[
            0, query_slice, indices[anchor_index, context_valid[anchor_index]]
        ] = 0
        own_block = slice(
            total_seq_len + anchor_index * block,
            total_seq_len + (anchor_index + 1) * block,
        )
        full_bias[0, query_slice, own_block] = 0

    reference = attention(
        block_x.reshape(1, num_anchors * block, cfg.hidden_size),
        context,
        block_positions.reshape(1, -1),
        position_ids.unsqueeze(0),
        full_bias,
    ).reshape(num_anchors, block, cfg.hidden_size)

    assert torch.allclose(compact, reference, atol=2e-6, rtol=2e-5)


def test_position_rope_is_fresh_and_matches_precomputed_values():
    """One layer cannot corrupt the materialized RoPE used by another."""
    positions = torch.tensor([[0, 1, 7], [7, 8, 15]])
    expected = _real_freqs(16, 8)[positions]

    first = freqs_cis_from_positions(positions, 8, 10_000.0)
    second = freqs_cis_from_positions(positions, 8, 10_000.0)

    assert torch.allclose(first, expected)
    assert torch.allclose(second, expected)
    assert first.data_ptr() != second.data_ptr()
    first[0, 0, 0] = float("nan")
    assert torch.isfinite(second).all()
    # The repeated absolute position must produce the same rotation in both
    # overlapping windows without sharing storage.
    assert torch.equal(second[0, 2], second[1, 0])


def test_full_dspark_training_forward_keeps_head_contracts():
    """Local windows still feed token, Markov, confidence, and loss heads."""
    torch.manual_seed(11)
    hidden_size, vocab_size = 16, 32
    transformer_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        sliding_window=4,
        use_sliding_window=True,
        layer_types=["sliding_attention"],
        _attn_implementation="eager",
    )
    config = DSV4DSparkConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=vocab_size,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=vocab_size - 1,
        markov_rank=4,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        num_heads=2,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=12,
        o_lora_rank=4,
        o_groups=2,
        window_size=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=16,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    model = DSV4DSparkDraftModel(config).train()
    for parameter in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
    ):
        torch.nn.init.normal_(parameter)

    seq_len, max_anchors = 12, 2
    _, loss, metrics = model(
        hidden_states=torch.randn(1, seq_len, 3 * hidden_size),
        input_ids=torch.randint(0, vocab_size - 1, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len, dtype=torch.bool),
        verifier_last_hidden_states=torch.randn(1, seq_len, hidden_size),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        position_ids=torch.arange(seq_len).unsqueeze(0),
        max_anchors=max_anchors,
        loss_config={"kl_div": (eager.kl_div_loss, 1.0)},
        tv_loss_fn=eager.tv_loss,
    )

    assert torch.isfinite(loss)
    assert "confidence_loss_sum" in metrics
    loss.backward()
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()
    assert model.confidence_head is not None
    assert model.confidence_head.proj.weight.grad is not None
    assert torch.isfinite(model.confidence_head.proj.weight.grad).all()
