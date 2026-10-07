from types import SimpleNamespace

import torch

from speculators.models.dsv4_dspark.backbone.hyper import (
    HyperConnection,
    collapse_streams,
)


def test_v41_final_collapse_reuses_ffn_pre_mix() -> None:
    cfg = SimpleNamespace(
        hc_mult=4,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        hidden_size=8,
    )
    connection = HyperConnection(cfg)
    torch.manual_seed(7)
    torch.nn.init.normal_(connection.fn, std=0.02)
    torch.nn.init.normal_(connection.base, std=0.02)
    torch.nn.init.normal_(connection.scale, std=0.02)
    streams = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)

    pre = connection.pre_weights(streams)
    collapsed = collapse_streams(streams, pre)

    flat = connection.input_norm(streams.flatten(start_dim=2).float())
    expected_pre = (
        torch.sigmoid(
            torch.nn.functional.linear(flat, connection.fn[:4].float())
            * connection.scale[0].float()
            + connection.base[:4].float()
        )
        + connection.hc_eps
    )
    expected = (expected_pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)

    torch.testing.assert_close(pre, expected_pre)
    torch.testing.assert_close(collapsed, expected)
