"""One mHC-wrapped DSV4 draft decoder block.

Residual flow keeps ``hc_mult`` streams; each sublayer (latent attention, then
MoE) is wrapped by a :class:`~.hyper.HyperConnection`:

    residual = streams
    post, comb, x = attn_hc(streams);  x = attn_norm(x);  x = attn(x, …)
    streams = place(x, residual, post, comb)
    residual = streams
    post, comb, x = ffn_hc(streams);   x = ffn_norm(x);   x = ffn(x)
    streams = place(x, residual, post, comb)

The attention is the block-gamma draft attention (:class:`~.attention.LatentAttention`),
which additionally consumes the target-hidden context ``main_x`` and the rope
frequencies for the block and context positions.
"""

from __future__ import annotations

import os as _os
import time as _time

import torch
from torch import nn

from .attention import LatentAttention
from .hyper import HyperConnection, place
from .moe import MoE
from .norm import RMSNorm

# DSPARK_PROFILE_FWD=1 -> sync+time each block sub-op (mHC = Sinkhorn hyper-connection, MLA = latent
# attention, MoE) and print any > DSPARK_PROFILE_FWD_MS (default 2000ms). Complements DSPARK_PROFILE_MOE
# (which splits the MoE internals). Pins whether a fwd spike is MLA / mHC-Sinkhorn / MoE vs. HS-fetch
# (read fetch_ms/align_ms for that). Diagnostic only; syncs serialize the pipe so it SLOWS the run —
# off (default) = zero cost. Lower DSPARK_PROFILE_FWD_MS (e.g. 0) to print every sub-op every step.
_FWD_PROF = _os.environ.get("DSPARK_PROFILE_FWD") == "1"
_FWD_PROF_MS = float(_os.environ.get("DSPARK_PROFILE_FWD_MS", "2000"))

# DSPARK_SATDUMP intra-layer capture: _backbone_forward sets this to a list during its
# (one-shot) satdump window; each block then appends its per-sub-stage tensors. None elsewhere.
_SAT_SUB = None


def _finite_diag(stage: str, tensor: torch.Tensor) -> None:
    """Fail at the first bad decoder substage when diagnostics are enabled."""
    if _os.environ.get("DSPARK_FINITE_DIAG") != "1":
        return
    bad = ~torch.isfinite(tensor)
    if not bool(bad.any().item()):
        return
    coordinates = torch.nonzero(bad, as_tuple=False)[:32].tolist()
    raise FloatingPointError(
        f"Non-finite DSV4 DSpark decoder substage {stage}: "
        f"count={int(bad.sum())}, first_coordinates={coordinates}"
    )


def _prof(tag, fn):
    if not _FWD_PROF:
        return fn()
    torch.npu.synchronize()
    _t0 = _time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    _dt = (_time.perf_counter() - _t0) * 1000.0
    if _dt > _FWD_PROF_MS:
        print(f"[FWD_PROF] {tag}: {_dt:.0f} ms", flush=True)
    return out


class MhcDecoderBlock(nn.Module):
    """Latent-attention + MoE block with two-site hyper-connections."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn = LatentAttention(cfg)
        self.ffn = MoE(cfg)
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn_hc = HyperConnection(cfg)
        self.ffn_hc = HyperConnection(cfg)

    def forward(
        self,
        streams: torch.Tensor,
        context_x: torch.Tensor,
        block_positions: torch.Tensor,
        context_positions: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated streams and this block's FFN pre-mix.

        ``context_x [N, W, dim]`` is the shared target-hidden context (``main_x``).
        """
        _sub = {} if _SAT_SUB is not None else None
        residual = streams
        post, comb, x = _prof("mHC.attn", lambda: self.attn_hc(streams))
        _finite_diag("mhc_attn_post", post)
        _finite_diag("mhc_attn_comb", comb)
        _finite_diag("mhc_attn_input", x)
        if _sub is not None:
            _sub["hc_pre_attn"] = x.detach().float().cpu()
        x = self.attn_norm(x)
        _finite_diag("attn_norm", x)
        if _sub is not None:
            _sub["attn_norm"] = x.detach().float().cpu()
        x = _prof(
            "MLA.attn",
            lambda: self.attn(
                x, context_x, block_positions, context_positions, attn_bias
            ),
        )
        _finite_diag("attention_output", x)
        if _sub is not None:
            _sub["attn_out"] = x.detach().float().cpu()
        streams = place(x, residual, post, comb)
        _finite_diag("post_attention_streams", streams)
        if _sub is not None:
            _sub["post_attn"] = streams.detach().float().cpu()

        residual = streams
        post, comb, x, ffn_pre = _prof(
            "mHC.ffn", lambda: self.ffn_hc(streams, return_pre=True)
        )
        _finite_diag("mhc_ffn_post", post)
        _finite_diag("mhc_ffn_comb", comb)
        _finite_diag("mhc_ffn_input", x)
        _finite_diag("mhc_ffn_pre", ffn_pre)
        if _sub is not None:
            _sub["hc_pre_ffn"] = x.detach().float().cpu()
        x = self.ffn_norm(x)
        _finite_diag("ffn_norm", x)
        if _sub is not None:
            _sub["ffn_norm"] = x.detach().float().cpu()
        x = _prof("MoE.ffn", lambda: self.ffn(x))
        _finite_diag("moe_output", x)
        if _sub is not None:
            _sub["moe_out"] = x.detach().float().cpu()
        streams = place(x, residual, post, comb)
        _finite_diag("post_moe_streams", streams)
        if _sub is not None:
            _sub["layer_out"] = streams.detach().float().cpu()
            _SAT_SUB.append(_sub)
        return streams, ffn_pre
