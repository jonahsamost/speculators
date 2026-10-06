"""Parity test for the shared-KV sink attention (drop the per-head ``.expand``).

The DSV4 draft MLA has ONE shared KV head. It used to broadcast it to all H query
heads via ``kv.unsqueeze(2).expand(-1,-1,H,-1)`` before a ``nkhd`` einsum, which
materialized an H×-larger ``[N,Sk,H,D]`` tensor. The optimization contracts the single
KV head directly (``nkd`` einsums) — mathematically identical, no H× materialization
(−~2.1 GB, ~20× fwd; matches the vLLM-Ascend #12005 shared-KV op). This pins that
identity so the optimization can't silently drift from the old expand semantics.
"""

import pytest
import torch

from speculators.models.dsv4_dspark.backbone.attention import (
    _sink_block_attention_torch,
)


def _expand_reference(q, kv, sink, scale, attn_bias):
    """The OLD path: broadcast the single KV head to H heads, then per-head einsums."""
    h = q.shape[2]
    k = kv.unsqueeze(2).expand(-1, -1, h, -1)  # [N, Sk, H, D]
    s = torch.einsum("nqhd,nkhd->nqhk", q.float(), k.float()) * scale
    if attn_bias is not None:
        s = s + attn_bias.float().unsqueeze(2)
    sink_h = sink.float().view(1, 1, -1, 1)
    row_max = torch.maximum(s.max(dim=-1, keepdim=True).values, sink_h)
    e = torch.exp(s - row_max)
    denom = e.sum(dim=-1, keepdim=True) + torch.exp(sink_h - row_max)
    p = e / denom
    return torch.einsum("nqhk,nkhd->nqhd", p, k.float()).to(q.dtype)


class TestSharedKVSinkAttention:
    def test_matches_per_head_expand(self):
        torch.manual_seed(0)
        # Shapes mirror a real draft step: window 128 + block 5 = 133 keys, H heads.
        n, sq, sk, h, d = 2, 5, 133, 4, 16
        q = torch.randn(n, sq, h, d)
        kv = torch.randn(n, sk, d)  # single shared KV head [N, Sk, D]
        sink = torch.randn(h)
        bias = torch.randn(n, sq, sk)
        scale = d**-0.5

        got = _sink_block_attention_torch(q, kv, kv, sink, scale, bias)
        ref = _expand_reference(q, kv, sink, scale, bias)

        assert got.shape == (n, sq, h, d)
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)

    def test_no_bias_path_matches(self):
        torch.manual_seed(1)
        n, sq, sk, h, d = 1, 3, 20, 8, 32
        q = torch.randn(n, sq, h, d)
        kv = torch.randn(n, sk, d)
        sink = torch.randn(h)
        scale = d**-0.5
        got = _sink_block_attention_torch(q, kv, kv, sink, scale, None)
        ref = _expand_reference(q, kv, sink, scale, None)
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)

    def test_rejects_pre_expanded_4d_kv(self):
        # Guard: a stale caller passing an already-expanded [N,Sk,H,D] KV must fail
        # loudly, rather than silently producing incorrect results.
        q = torch.randn(1, 2, 4, 8)
        kv4 = torch.randn(1, 3, 4, 8)
        with pytest.raises(ValueError, match="single head"):
            _sink_block_attention_torch(q, kv4, kv4, torch.zeros(4), 0.1)

    def test_all_masked_rows_have_finite_backward_through_sink(self):
        """The synthetic sink keeps even fully masked attention rows valid."""
        q = torch.randn(1, 3, 4, 8, requires_grad=True)
        kv = torch.randn(1, 5, 8, requires_grad=True)
        sink = torch.randn(4, requires_grad=True)
        bias = torch.full((1, 3, 5), float("-inf"))

        output = _sink_block_attention_torch(q, kv, kv, sink, 8**-0.5, bias)
        output.square().sum().backward()

        assert torch.isfinite(output).all()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()
        assert kv.grad is not None
        assert torch.isfinite(kv.grad).all()
        assert sink.grad is not None
        assert torch.isfinite(sink.grad).all()
