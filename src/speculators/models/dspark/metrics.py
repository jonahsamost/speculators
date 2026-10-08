"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).

Two acceptance lengths are reported, both counting the verifier's bonus token:
``accept_len`` from that analytical rate (acceptance under rejection sampling)
and ``eal`` from greedy argmax matches (acceptance at temperature 0, and the
value comparable across the DFlash family).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    tv_loss,
)
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

__all__ = [
    "compute_metrics",
]

_EPS = 1e-8


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    valid = loss_mask.bool()
    valid_vocab = valid.unsqueeze(-1)
    invalid_logits = valid_vocab & ~torch.isfinite(logits)
    invalid_targets = valid_vocab & ~torch.isfinite(targets)
    if invalid_logits.any().item() or invalid_targets.any().item():
        raise FloatingPointError(
            "Non-finite DSpark distributions on supervised positions: "
            f"logits={int(invalid_logits.sum().item())}, "
            f"targets={int(invalid_targets.sum().item())}"
        )

    # Anchor selection pads to a static ``max_anchors`` shape.  Loss functions
    # such as TV and CE must not see non-finite values from those masked rows:
    # masking their scalar losses afterwards is too late because softmax
    # backward has already traversed them. ``where`` also cuts their gradient
    # path while preserving the exact objective on every supervised row.
    safe_logits = torch.where(valid_vocab, logits, torch.zeros_like(logits))
    safe_targets = torch.where(valid_vocab, targets, torch.zeros_like(targets))
    loss, term_losses = compound_loss(
        safe_logits,
        safe_targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
    )

    # Analytical per-position acceptance rate = distributional overlap
    # = 1 - TV; the fused kernel avoids the two full-vocab fp32 softmaxes.
    with torch.no_grad():
        accept_rate = (
            1.0 - tv_loss_fn(safe_logits, safe_targets)
        ).float().clamp(0.0, 1.0)
        # Per-block cumulative acceptance product over the draft slots (slot 0
        # is the anchor), shared by the accept-length and calibration metrics.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        # Keep confidence BCE in fp32. CUDA's bfloat16 BCE backward can return
        # NaNs even when its logits, soft targets, and forward loss are finite.
        confidence_logits_fp32 = confidence_logits.float()
        c_star = accept_rate.detach()

        # ``select_anchors`` pads each batch to ``max_anchors``.  Those padded
        # slots are deliberately excluded by ``loss_mask``, but their draft
        # activations are still materialized.  Do not feed a non-finite padded
        # activation into BCE and try to remove it afterwards: NaN * 0 remains
        # NaN, and BCE backward can poison every upstream parameter.  Conversely,
        # a non-finite supervised slot is a real model/data error and must fail
        # loudly rather than being silently sanitized.
        invalid_logits = valid & ~torch.isfinite(confidence_logits_fp32)
        invalid_targets = valid & ~torch.isfinite(c_star)
        if invalid_logits.any().item() or invalid_targets.any().item():
            raise FloatingPointError(
                "Non-finite DSpark confidence values on supervised positions: "
                f"logits={int(invalid_logits.sum().item())}, "
                f"targets={int(invalid_targets.sum().item())}"
            )
        safe_confidence_logits = torch.where(
            valid, confidence_logits_fp32, torch.zeros_like(confidence_logits_fp32)
        )
        safe_c_star = torch.where(valid, c_star, torch.zeros_like(c_star))
        bce = binary_cross_entropy_with_logits(
            safe_confidence_logits, safe_c_star, reduction="none"
        )  # [1, T]
        # D-PACE expects token CE, so keep the confidence BCE on fixed decay.
        confidence_decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, confidence_decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = safe_confidence_logits.sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted bonus.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy
    pred_ids = torch.argmax(safe_logits, dim=-1)
    target_ids = torch.argmax(safe_targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    # Greedy counterpart to accept_len, on the same per-block/bonus-token
    # convention, so DFlash-family runs compare on one number.
    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(num_blocks, block_size)[:, start_pos:],
        draft_mask.to(torch.bool),
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total

    return loss, metrics
