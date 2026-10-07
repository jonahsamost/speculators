"""Configuration for the DSV4 DSpark faithful draft.

Self-contained, backend-agnostic dataclass config for the DeepSeek-V4-Flash
DSpark *draft* (the semi-autoregressive drafter trained on cached target
hidden states). Values are the released
``deepseek-ai/DeepSeek-V4.1-Flash/config.json`` (read 2026-10-07).
Conversion still reads every value from the checkpoint rather than relying on
these defaults.

The draft reuses the target's decoder-layer *shape* (MLA + sink + 128-expert
MoE + hyper-connections) but is a small stack (``n_draft_layers`` = 3) with
extra DSpark parts (``main_proj`` conditioning, Markov + confidence heads).
This config carries both the backbone shape and the DSpark-method knobs; it is
a plain dataclass so the modules unit-test on a CPU box with only torch. The
speculators ``SpeculatorModelConfig`` wrapper is added at trainer-integration
time and delegates to this.

No dependency on ``alloy`` / ``transformers`` / any accelerator package: the
whole draft is meant to be self-contained and GPU/CPU-runnable, with NPU
kernels inserted opt-in via :mod:`.backbone.kernels`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class DSparkDraftConfig:
    """Shape + method config for the DSV4 DSpark draft.

    Attribute names are deliberately mechanism-descriptive (not vendor
    class names) so the model definition reads as a standalone port. The
    checkpoint weight-key mapping (``mtp.N.attn.wq_a`` etc.) lives in the
    weight loader, not here.
    """

    # ---- vocabulary / hidden ------------------------------------------------
    vocab_size: int = 129280
    hidden_size: int = 5120
    rms_norm_eps: float = 1e-20

    # ---- draft stack --------------------------------------------------------
    n_draft_layers: int = 3  # official n_mtp_layers
    # DSpark samples from the anchor position, so every slot is predictive and
    # block_size equals the number of draft tokens. This maps directly to the
    # released checkpoint's dspark_block_size.
    block_size: int = 5
    noise_token_id: int = 128799  # fills draft_input_ids[:, 1:] (the gamma mask slots)
    target_layer_ids: tuple[int, ...] = (37, 38, 39)  # verifier layers -> main_proj
    markov_rank: int = 256

    # ---- multi-head latent attention (MLA) ----------------------------------
    num_heads: int = 64
    head_dim: int = 512  # per-head q/k/v width (nope | rope)
    rope_head_dim: int = 64  # trailing rope slice of head_dim
    q_lora_rank: int = 1280  # low-rank query bottleneck (wq_a -> wq_b)
    o_lora_rank: int = 1024  # grouped output bottleneck (wo_a -> wo_b)
    o_groups: int = 8  # head groups for the grouped output projection
    window_size: int = 128  # sliding-window context the draft attends to
    # RoPE (yarn); draft uses the "main" theta only (no compress path).
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    # ---- mixture of experts -------------------------------------------------
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    n_activated_experts: int = 3  # top-k
    moe_inter_dim: int = 2304
    score_func: str = "sqrtsoftplus"  # router scoring
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # ---- hyper-connections (mHC) -------------------------------------------
    hc_mult: int = 4  # number of residual-stream copies
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ---- loss (DSpark distribution term + confidence BCE + block decay) -----
    # L = Σ_k w_k·[ce_alpha·CE + l1_alpha·L1 + conf_alpha·BCE_conf], w_k=exp(-(k-1)/γ).
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    confidence_alpha: float = 1.0
    decay_gamma: float = 4.0

    # ---- training-time weight dtype (bf16; the released ckpt is fp4/fp8) ----
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if isinstance(self.target_layer_ids, list):
            self.target_layer_ids = tuple(self.target_layer_ids)
        if self.head_dim <= self.rope_head_dim:
            raise ValueError("head_dim must exceed rope_head_dim (nope|rope split).")
        if self.num_heads * self.head_dim % self.o_groups != 0:
            raise ValueError("o_groups must divide num_heads*head_dim.")

    @property
    def num_target_layers(self) -> int:
        return len(self.target_layer_ids)

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    def small(self, **overrides) -> DSparkDraftConfig:
        """A tiny variant for fast CPU parity/unit tests (few seconds).

        Keeps every structural feature (MLA + sink + MoE + mHC + heads) but
        shrinks widths/counts. Any field can be overridden via kwargs.
        """
        base = dict(
            vocab_size=256,
            hidden_size=128,
            n_draft_layers=3,
            block_size=5,
            noise_token_id=255,
            target_layer_ids=(0, 1, 2),
            markov_rank=32,
            num_heads=4,
            head_dim=32,
            rope_head_dim=8,
            q_lora_rank=64,
            o_lora_rank=64,
            o_groups=2,
            window_size=16,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
            moe_inter_dim=128,
            hc_mult=2,
            hc_sinkhorn_iters=2,
        )
        base.update(overrides)
        return replace(self, **base)
