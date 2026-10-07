"""DSV4 DSpark draft = upstream dense-line DSpark method + our sparse backbone.

We subclass upstream's :class:`~speculators.models.dspark.core.DSparkDraftModel`
and override ONLY the decoder stack inside ``_backbone_forward``: the anchor
sampling, target-distribution computation, Markov + confidence heads, compound
loss, registration, ``from_training_args`` and the data contract are all
inherited verbatim. In place of the Qwen3 DFlash decoder layers we run our
clean-room DSV4 stack — multi-head latent attention + per-head sink + 256-expert
MoE + hyper-connections — with our interleaved DSV4 RoPE.

The block-attention contract is identical to DFlash's: the draft block queries
(``noise_embedding [1, TB, H]``) attend to ``[target-hidden context | block]``
under the additive ``attention_mask`` (block-diagonal + sliding window). Our
existing ``MhcDecoderBlock`` forward already implements exactly this shape
(``block_x``, ``context_x``, per-position freqs, ``attn_bias``); this module
only wires the RoPE positions and mask, and manages the mHC streams across the
stack (expand once, collapse with the final FFN boundary's pre-mix at the end).
"""

from __future__ import annotations

import os
from typing import ClassVar, Literal

import torch
from pydantic import Field
from torch import nn

from speculators import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel

# Absolute (not relative) imports: transformers' custom_object_save parses the
# config module's RELATIVE imports to bundle them for trust_remote_code, and
# mishandles two-level ones (``from .backbone.block`` -> looks for the file
# ``backbone.block.py``). Absolute imports are not parsed, so save_pretrained works.
from speculators.models.dsv4_dspark.backbone.block import MhcDecoderBlock
from speculators.models.dsv4_dspark.backbone.hyper import collapse_streams
from speculators.models.dsv4_dspark.backbone.rotary import precompute_freqs_cis
from speculators.models.dsv4_dspark.config import DSparkDraftConfig

__all__ = ["DSV4DSparkConfig", "DSV4DSparkDraftModel"]


@SpeculatorModelConfig.register("dsv4_dspark")
class DSV4DSparkConfig(DSparkSpeculatorConfig):
    """Dense-line DSpark config + the DSV4 sparse-backbone hyperparameters.

    ``transformer_layer_config`` still carries the shared shape the SpeculatorModel
    machinery needs (hidden_size, vocab_size, num_hidden_layers = draft depth,
    rms_norm_eps); the fields below configure our MLA + MoE + mHC backbone.
    """

    speculators_model_type: Literal["dsv4_dspark"] = "dsv4_dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(default_factory=lambda: ["DSV4DSparkDraftModel"])
    confidence_head_bias: bool = False

    # multi-head latent attention
    num_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    # mixture of experts
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 2048
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def backbone_config(self) -> DSparkDraftConfig:
        """Build the plain backbone dataclass our modules consume."""
        tl = self.transformer_layer_config
        return DSparkDraftConfig(
            vocab_size=tl.vocab_size,
            hidden_size=tl.hidden_size,
            rms_norm_eps=tl.rms_norm_eps,
            n_draft_layers=tl.num_hidden_layers,
            block_size=self.block_size,
            noise_token_id=self.mask_token_id or 0,
            target_layer_ids=tuple(self.aux_hidden_state_layer_ids or (0, 1, 2)),
            markov_rank=self.markov_rank,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            o_groups=self.o_groups,
            window_size=self.window_size,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            original_seq_len=self.original_seq_len,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_activated_experts,
            moe_inter_dim=self.moe_inter_dim,
            score_func=self.score_func,
            route_scale=self.route_scale,
            swiglu_limit=self.swiglu_limit,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
        )


@SpeculatorModel.register("dsv4_dspark")
class DSV4DSparkDraftModel(DSparkDraftModel):
    """DSpark method (inherited) over the DSV4-native sparse backbone."""

    config_class: ClassVar[type[DSV4DSparkConfig]] = DSV4DSparkConfig  # type: ignore[assignment,misc]
    _no_split_modules: ClassVar[list[str]] = ["MhcDecoderBlock"]  # type: ignore[assignment]

    def __init__(self, config: DSV4DSparkConfig) -> None:
        # Force the additive (eager) float mask BEFORE super().__init__ reads it to
        # pick the mask builder — our sink attention consumes it as an additive bias.
        config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
        # DFlash/DSpark __init__ builds fc(=main_proj role)/hidden_norm/norm/embed/
        # lm_head/verifier + markov/confidence + a stack of Qwen3 layers we discard.
        super().__init__(config=config)
        bb = config.backbone_config()
        self.backbone_cfg = bb

        # Activation checkpointing (recompute): DSPARK_RECOMPUTE=1 recomputes each draft layer's
        # forward in backward instead of storing its activations -> frees per-layer activation so
        # MAX_ANCHORS can scale past the memory wall (the anchor lever that rebalances an HS-bound
        # run by slowing the step to the serve's HS-production rate). See _backbone_forward.
        self.grad_checkpoint = bool(int(os.environ.get("DSPARK_RECOMPUTE", "0")))

        # Swap the decoder stack for our DSV4-native blocks. DeepSeek V4.1 has
        # no standalone terminal mHC head; the final layer's FFN pre-mix is
        # reused to collapse the residual streams.
        self.layers = nn.ModuleList(
            MhcDecoderBlock(bb) for _ in range(bb.n_draft_layers)
        )

        # Our interleaved DSV4 RoPE cache, indexed by absolute position (YaRN off on the sliding
        # path -> original_seq_len=0). Stored as a REAL [seqlen, rope//2, 2] tensor (view_as_real
        # of the complex freqs): NPU aclnn can't index/mul complex64, and casting a complex buffer
        # to bf16 (trainer AMP) drops the imag part -> scale-only, no rotation. Real cos/sin is
        # NPU-index-safe AND survives the bf16 cast losslessly (proper rotation). See apply_rotary_emb.
        self._rope_dim = bb.rope_head_dim
        # Hugging Face constructs large checkpoints inside a default ``meta``
        # device context. RoPE is a derived, non-persistent buffer, so build its
        # initial value explicitly on CPU; ``_rope_at`` reconstructs it after
        # ``to_empty`` when the loaded model reaches its real device.
        with torch.device("cpu"):
            initial_freqs = torch.view_as_real(
                precompute_freqs_cis(
                    bb.rope_head_dim,
                    bb.original_seq_len or 1,
                    0,
                    bb.rope_theta,
                    bb.rope_factor,
                    bb.beta_fast,
                    bb.beta_slow,
                )
            ).contiguous()
        self.register_buffer(
            "freqs_cis",
            initial_freqs,
            persistent=False,
        )
        self._init_backbone_params()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> DSV4DSparkDraftModel:
        """Build a DSV4 DSpark model from CLI args (DSV4 backbone fields default to
        the released config; the DSpark method fields mirror upstream)."""
        # DSV4 backbone hyperparameters default to the released config; any passed
        # on the CLI (e.g. --n-routed-experts for a reduced first run) override.
        _BACKBONE = (
            "num_heads",
            "head_dim",
            "rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "window_size",
            "n_routed_experts",
            "n_shared_experts",
            "n_activated_experts",
            "moe_inter_dim",
            "hc_mult",
            "hc_sinkhorn_iters",
        )
        overrides = {k: kwargs[k] for k in _BACKBONE if kwargs.get(k) is not None}
        config = DSV4DSparkConfig(
            **cls._build_base_config_kwargs("dsv4_dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", True),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", True),
            # DSV4-Flash: the released mtp.* layout has NO confidence_head.proj.bias.
            confidence_head_bias=False,
            **overrides,
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        parts = {
            p
            for p, flag in (
                ("attn", "init_attn_from_target"),
                ("hc", "init_hc_from_target"),
                ("norm", "init_norm_from_target"),
                ("moe", "init_moe_from_target"),
            )
            if kwargs.get(flag)
        }
        if kwargs.get("init_layer_from_target"):
            parts |= {"attn", "hc", "norm", "moe"}
        if parts:
            model.load_verifier_layer(
                parts,
                skip_router=bool(kwargs.get("init_moe_no_router")),
                zero_bias=bool(kwargs.get("init_moe_zero_bias")),
            )
        return model

    def load_verifier_weights(self) -> None:
        """Load the shared frozen embed + lm_head + verifier norm from the DSV4
        verifier, which uses the OFFICIAL DeepSeek key names (``embed.weight`` /
        ``head.weight`` / ``norm.weight``) rather than HF's (``embed_tokens`` /
        ``lm_head`` / ``model.norm``). Same masking/freezing as the base."""
        import warnings  # noqa: PLC0415

        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        # Meta build (non-rank0 under --init-on-meta): params carry no data, so
        # skip file loading entirely; real weights arrive via broadcast_from_rank0.
        # Still freeze here (requires_grad_() works on meta) -- the trainable-param
        # set must match rank0's or FSDP2's grad reduce_scatter hangs (mirrors #776).
        if self.embed_tokens.weight.is_meta:
            self.embed_tokens.weight.requires_grad_(False)
            self.lm_head.weight.requires_grad_(False)
            self.verifier_lm_head.weight.requires_grad_(False)
            if hasattr(self, "verifier_norm"):
                self.verifier_norm.weight.requires_grad_(False)
            return

        sc = getattr(getattr(self, "config", None), "speculators_config", None)
        if sc is None or sc.verifier.name_or_path is None:
            return
        w = load_model_layers(
            ["embed.weight", "head.weight", "norm.weight"], sc.verifier.name_or_path
        )
        embed_w = w["embed.weight"]
        lm_head_w = w.get("head.weight", embed_w)

        if self.embed_tokens.weight.isnan().any():
            self.embed_tokens.load_state_dict({"weight": embed_w})
        if self.use_draft_vocab:
            if self.t2d is None or not torch.any(self.t2d).item():
                raise ValueError(
                    "t2d not set; call load_vocab_mappings before load_verifier_weights."
                )
            lm_head_w = lm_head_w[
                self.t2d.to(device=lm_head_w.device, dtype=torch.bool), :
            ]
        if self.lm_head.weight.isnan().any():
            self.lm_head.load_state_dict(
                {"weight": lm_head_w.detach().clone()}, strict=False
            )
        self.verifier_lm_head.load_state_dict(
            {"weight": lm_head_w.detach().clone()}, strict=False
        )
        if hasattr(self, "verifier_norm"):
            if "norm.weight" not in w:
                warnings.warn(
                    f"no norm.weight in {sc.verifier.name_or_path}; using default.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.verifier_norm.load_state_dict({"weight": w["norm.weight"]})
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self.verifier_lm_head.weight.requires_grad_(False)
        if hasattr(self, "verifier_norm"):
            self.verifier_norm.weight.requires_grad_(False)

    def load_verifier_moe(self) -> None:
        """Back-compat shim: warm-start only the MoE. See :meth:`load_verifier_layer`."""
        self.load_verifier_layer({"moe"})

    def load_verifier_layer(
        self, parts, *, skip_router: bool = False, zero_bias: bool = False
    ) -> None:
        """Warm-start the requested PARTS of each draft layer from the matching verifier
        layer: draft layer ``n`` <- verifier layer ``target_layer_ids[n]`` (the layers
        ``main_proj`` conditions on). A draft layer is architecturally a DSV4 target layer,
        so this is a strong TRAINABLE init (not frozen), unlike the shared embed/lm_head.

        ``parts`` is any subset of:
          * ``"attn"`` — the MLA **core** projections (wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/
            wo_b/attn_sink). The verifier's sparse-attn ``compressor``/``indexer`` submodules
            are skipped — the draft does dense sliding-window sink attention (no such parts).
          * ``"hc"``   — the two hyper-connections (``attn_hc`` <- ``hc_attn_*``,
            ``ffn_hc`` <- ``hc_ffn_*``; the sole flat->dotted rename).
          * ``"norm"`` — the two RMSNorms (``attn_norm``, ``ffn_norm``).
          * ``"moe"``  — routed experts (EP-local slice) + router (<- ``ffn.gate``) + shared.
            ``skip_router`` leaves the router at a fresh random init (experts still warm-
            started); ``zero_bias`` warm-starts ``gate.weight`` but not the aux-free balance
            bias (starts at 0 for ``DSPARK_MOE_BALANCE`` to rebuild). See the router-copy site.

        EP-aware for experts (the stacked ``GroupedExperts`` weights are still plain per-rank
        tensors at build time — before the ``Shard(0)`` wrap — so each rank copies only its
        ``[ep_expert_offset : +n_local]`` slice); the rest are replicated (full copy per rank).
        No-op on meta params (non-rank0 under ``--init-on-meta``; broadcast fills). Requires
        the draft dims to match the verifier's (the faithful config); raises on any missing
        key / shape mismatch. Verifier uses official DeepSeek names, so copies are 1:1."""
        import logging  # noqa: PLC0415

        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        parts = set(parts)
        if not parts:
            return
        # meta build (non-rank0 under --init-on-meta): weights arrive via broadcast.
        if self.layers[0].ffn.router.weight.is_meta:
            return
        sc = getattr(getattr(self, "config", None), "speculators_config", None)
        if sc is None or sc.verifier.name_or_path is None:
            return
        path = sc.verifier.name_or_path
        bb = self.backbone_cfg
        tids = bb.target_layer_ids
        if len(tids) < bb.n_draft_layers:
            raise ValueError(
                f"init-from-target needs >= n_draft_layers ({bb.n_draft_layers}) "
                f"target_layer_ids, got {tids}."
            )

        # replicated (non-EP) parts: (verifier-key suffix, draft attribute path). For attn
        # and norm the two are identical; hc renames the flat hc_{attn,ffn}_* -> attn_hc/ffn_hc.
        PART_MAP = {
            "attn": [
                ("attn.wq_a.weight", "attn.wq_a.weight"),
                ("attn.q_norm.weight", "attn.q_norm.weight"),
                ("attn.wq_b.weight", "attn.wq_b.weight"),
                ("attn.wkv.weight", "attn.wkv.weight"),
                ("attn.kv_norm.weight", "attn.kv_norm.weight"),
                ("attn.wo_a.weight", "attn.wo_a.weight"),
                ("attn.wo_b.weight", "attn.wo_b.weight"),
                ("attn.attn_sink", "attn.attn_sink"),
            ],
            "hc": [
                ("hc_attn_fn", "attn_hc.fn"),
                ("hc_attn_base", "attn_hc.base"),
                ("hc_attn_scale", "attn_hc.scale"),
                ("hc_ffn_fn", "ffn_hc.fn"),
                ("hc_ffn_base", "ffn_hc.base"),
                ("hc_ffn_scale", "ffn_hc.scale"),
            ],
            "norm": [
                ("attn_norm.weight", "attn_norm.weight"),
                ("ffn_norm.weight", "ffn_norm.weight"),
            ],
        }

        def _need(dct: dict, k: str):
            if k not in dct:
                raise KeyError(f"init-from-target: '{k}' not found in verifier {path}")
            return dct[k]

        def _copy(param, src) -> None:
            src = src.to(device=param.device, dtype=param.dtype)
            if tuple(src.shape) != tuple(param.shape):
                raise ValueError(
                    f"init-from-target: source shape {tuple(src.shape)} != draft "
                    f"{tuple(param.shape)} -- needs the faithful config matching the verifier."
                )
            param.data.copy_(src)

        def _resolve(module, dotted: str):
            obj = module
            for a in dotted.split("."):
                obj = getattr(obj, a)
            return obj

        for n in range(bb.n_draft_layers):
            blk = self.layers[n]
            pre = f"layers.{tids[n]}."

            simple = [
                (pre + vk, da)
                for part in ("attn", "hc", "norm")
                if part in parts
                for vk, da in PART_MAP[part]
            ]

            moe_keys: list[str] = []
            eids: list[int] = []
            if "moe" in parts:
                ffn = blk.ffn
                off = int(getattr(ffn, "ep_expert_offset", 0))
                eids = [off + i for i in range(ffn.experts.w1.shape[0])]
                moe_keys = [
                    pre + "ffn.gate.weight",
                    pre + "ffn.gate.bias",
                    pre + "ffn.shared_experts.w1.weight",
                    pre + "ffn.shared_experts.w2.weight",
                    pre + "ffn.shared_experts.w3.weight",
                ]
                for e in eids:
                    moe_keys += [
                        f"{pre}ffn.experts.{e}.w1.weight",
                        f"{pre}ffn.experts.{e}.w2.weight",
                        f"{pre}ffn.experts.{e}.w3.weight",
                    ]

            w = load_model_layers([vk for vk, _ in simple] + moe_keys, path)

            for vk, da in simple:
                _copy(_resolve(blk, da), _need(w, vk))

            if "moe" in parts:
                # router (gate): weight trainable; bias is the aux-free balance bias.
                if skip_router:
                    # Fresh router (--init-moe-no-router): do NOT import the verifier's
                    # routing. Its gate.weight was tuned to spread the verifier's DIVERSE
                    # 61-layer hidden distribution; on the draft's NARROWER hidden states
                    # (3 layers + injected target-HS) that same W projects everything onto
                    # a few experts -> fast collapse. A fresh small-normal init routes
                    # ~uniformly at step 0 (matches _init_backbone_params std); let
                    # DSPARK_MOE_BALANCE + training find the draft's own routing. bias
                    # stays at its 0 init. torch.empty at build would be raw garbage here
                    # (skipped by _init_backbone_params if finite), so re-init explicitly.
                    nn.init.normal_(ffn.router.weight, std=0.02)
                else:
                    _copy(ffn.router.weight, _need(w, pre + "ffn.gate.weight"))
                    # zero_bias (--init-moe-zero-bias): keep the semantic gate.weight but
                    # leave the balance bias at its 0 init so DSPARK_MOE_BALANCE rebuilds
                    # it for the DRAFT distribution, instead of inheriting the verifier's
                    # mis-calibrated (historically frozen) balance solution.
                    if not zero_bias:
                        _copy(ffn.router.bias, _need(w, pre + "ffn.gate.bias"))
                for wn in ("w1", "w2", "w3"):
                    _copy(
                        getattr(ffn.shared_experts, wn).weight,
                        _need(w, f"{pre}ffn.shared_experts.{wn}.weight"),
                    )
                # routed experts: stacked [n_local, out, in]; local slot i <- global expert off+i
                for wn in ("w1", "w2", "w3"):
                    stacked = getattr(ffn.experts, wn)
                    slot_shape = tuple(stacked.shape[1:])
                    for i, e in enumerate(eids):
                        src = _need(w, f"{pre}ffn.experts.{e}.{wn}.weight").to(
                            device=stacked.device, dtype=stacked.dtype
                        )
                        if tuple(src.shape) != slot_shape:
                            raise ValueError(
                                f"init-from-target: expert {e} {wn} {tuple(src.shape)} != "
                                f"draft slot {slot_shape} (needs faithful config)."
                            )
                        stacked.data[i].copy_(src)

        router_note = ""
        if "moe" in parts:
            if skip_router:
                router_note = " [router: FRESH random init, not warm-started]"
            elif zero_bias:
                router_note = " [router: gate.weight warm-started, balance bias zeroed]"
        logging.getLogger(__name__).info(
            "init-from-target: warm-started parts %s of %d draft layers from verifier layers %s.%s",
            sorted(parts),
            bb.n_draft_layers,
            list(tids[: bb.n_draft_layers]),
            router_note,
        )

    def _init_backbone_params(self) -> None:
        """Initialize the freshly-built backbone params (post_init ran on the old
        Qwen3 layers). Uninitialized ``torch.empty`` params (mHC fn) would NaN."""
        # Meta build (non-rank0 under --init-on-meta): no data to init; the random
        # init done on rank 0 is broadcast to every rank in the FSDP setup.
        if any(p.is_meta for p in self.parameters()):
            return
        std = 0.02
        for m in self.layers:
            for name, p in m.named_parameters():
                if p.dim() >= 2 and (
                    ".fn" in name or "weight" in name or "hc_fn" in name
                ):
                    if (
                        torch.isnan(p).any()
                        or not p.abs().sum().isfinite()
                        or p.abs().sum() == 0
                    ):
                        nn.init.normal_(p, std=std)

    def fsdp_wrap_plan(self) -> list[nn.Module]:
        """Fine-grained FSDP2 units for :func:`apply_fully_sharded` (children first).

        ``ffn.experts`` (a :class:`GroupedExperts` with stacked ``[E, ...]`` weights) is
        one FSDP unit; ``attn`` (MLA) gets its own group; ``block`` then manages only the
        small remainder (norms, hyper-connections, router). The root ``fully_shard(model)``
        (added by the caller) covers embed/head/heads.

        Under EP the routed experts are NOT FSDP-wrapped (see fsdp_ignored_params): they
        are rank-local Shard(0) slices, dispatched via all-to-all, not FSDP all-gather.
        The non-EP path shards the stacked experts over the DP mesh (fine for the small
        reduced config; the 256-expert faithful run uses EP, not non-EP per-layer FSDP).
        """
        ep = self._ep_active()
        plan: list[nn.Module] = []
        for block in self.layers:
            if not ep:
                plan.append(block.ffn.experts)  # GroupedExperts: one FSDP unit
            plan.append(block.ffn.shared_experts)
            plan.append(block.attn)
            plan.append(block)  # remainder: norms, attn_hc/ffn_hc, router
        return plan

    def _ep_active(self) -> bool:
        """True when expert-parallelism is configured (routed experts are partitioned)."""
        # ABSOLUTE import (see the module-header note): transformers' custom_object_save
        # statically parses this file's RELATIVE imports to bundle them for trust_remote_code
        # and mishandles ``from .backbone ...`` (looks for backbone.py) -> save_pretrained
        # crashes at checkpoint time. Keep backbone imports absolute.
        from speculators.models.dsv4_dspark.backbone import moe_ep  # noqa: PLC0415

        return moe_ep._EP is not None and moe_ep._EP.size > 1

    def fsdp_ignored_params(self) -> set:
        """Params FSDP must leave alone. Under EP the routed experts are rank-local Shard(0)
        slices (each rank owns a disjoint set) -> exclude from FSDP sharding so grouped-GEMM
        sees the local slice and tokens are moved by all-to-all instead of all-gather."""
        if not self._ep_active():
            return set()
        params: set = set()
        for block in self.layers:
            params.update(block.ffn.experts.parameters())  # stacked w1/w2/w3
        return params

    def ep_local_param_keys(self) -> set:
        """state_dict names of EP rank-local params (routed experts) to drop from the
        rank0 broadcast -- each rank keeps its own per-rank-initialized slice."""
        if not self._ep_active():
            return set()
        ignored_ids = {id(p) for p in self.fsdp_ignored_params()}
        return {name for name, p in self.named_parameters() if id(p) in ignored_ids}

    def _rebuild_freqs(self, seqlen: int) -> None:
        self.freqs_cis = (
            torch.view_as_real(
                precompute_freqs_cis(
                    self._rope_dim,
                    seqlen,
                    0,
                    self.backbone_cfg.rope_theta,
                    self.backbone_cfg.rope_factor,
                    self.backbone_cfg.beta_fast,
                    self.backbone_cfg.beta_slow,
                )
            )
            .contiguous()
            .to(self.freqs_cis.device)
        )

    def _rope_at(self, positions: torch.Tensor) -> torch.Tensor:
        """Interleaved rotary cache at the given absolute positions -> REAL [len, rope_dim//2, 2]
        (cos in [...,0], sin in [...,1]). Indexing the REAL view is NPU-safe (aclnn rejects
        complex64); the buffer holds view_as_real(freqs). See apply_rotary_emb."""
        # from_pretrained's meta-init zeroes persistent=False FLOATING buffers via to_empty()
        # (it doesn't re-run __init__ and freqs_cis isn't in the state_dict) -> rebuild once.
        # (Training builds via from_training_args, no meta, so this is a one-shot no-op there.)
        if not getattr(self, "_freqs_ok", False):
            self._freqs_ok = True
            if not bool(self.freqs_cis.any()):
                self._rebuild_freqs(self.freqs_cis.shape[0])
        if positions.numel() and int(positions.max()) >= self.freqs_cis.shape[0]:
            self._rebuild_freqs(int(positions.max()) + 1)
        return self.freqs_cis.to(positions.device)[positions]

    def _backbone_forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """DFlash scaffolding (copied) with the decoder stack swapped for our
        DSV4 sparse stack + DSV4 RoPE. Returns the same 5-tuple DSpark consumes."""
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size), self.mask_token_id, dtype=torch.long, device=device
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)  # [1, TB, H]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)  # [1, T, H]  (main_x context)

        from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks

        block_positions = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )  # [TB]
        ctx_positions = position_ids[0]  # [T]

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            # `verifier_last_hidden_states` is ALREADY the verifier's FINAL POST-NORM hidden (the HS
            # dumper writes `self.norm(...)` output; data.py:507 feeds it verbatim). The verifier's
            # TRUE next-token logits are `lm_head(post_norm_hidden)` — single norm (DEFAULT, F1,
            # reproduction-faithful = the real verifier the serve accepts against).
            #
            # ★ DSPARK_TEACHER_DOUBLE_NORM=1 re-applies `verifier_norm` → norm(norm(h)) = the accidental
            # "double-norm bug", made a DELIBERATE knob. 2026-07-25 eval: it BEAT single-norm by ~0.2
            # accept_len, entirely in the tail. It is NOT a uniform sharpening (that's --kd-temperature):
            # it's a per-dimension `~w²` RESHAPING of the teacher hidden (flips the teacher argmax ~16%
            # vs the real verifier — T1). Why the reshaping helps the tail is not understood; empirical.
            # Off by default (single-norm); set =1 to reproduce/A-B the double-norm win.
            _teacher_h = verifier_last_hidden_states
            if os.environ.get("DSPARK_TEACHER_DOUBLE_NORM") == "1":
                _teacher_h = self.verifier_norm(_teacher_h)
            verifier_logits = self.verifier_lm_head(_teacher_h)
            if not self.config.sample_from_anchor:
                # False: shift right by 1 so slot j predicts the token AT position j
                # (slot 0 = the given anchor). True (DSpark, matches the vllm-ascend
                # serve): NO shift — slot k predicts the NEXT token (position k+1).
                verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        # DSV4 RoPE freqs at the ctx and block absolute positions.
        ctx_freqs = self._rope_at(ctx_positions)
        block_freqs = self._rope_at(block_positions)

        # mHC streams across the stack; each block attends noise -> [ctx | block].
        hc = self.backbone_cfg.hc_mult
        streams = noise_embedding.unsqueeze(2).repeat(1, 1, hc, 1)  # [1, TB, hc, H]

        # ── DSPARK_SATDUMP=1: one-shot per-stage capture mirroring the serve model.forward dump
        #    (deepseek_v4_dspark.py), so the two can be bisected layer-by-layer to the FIRST
        #    diverging stage. run_train_block triggers this on its first block.
        _sat = os.environ.get("DSPARK_SATDUMP") == "1" and not getattr(
            self, "_satdumped", False
        )
        _rec = (
            {
                "embed": noise_embedding.detach().float().cpu(),
                "streams_in": streams.detach().float().cpu(),
                "layers": [],
            }
            if _sat
            else None
        )
        if _sat:
            from speculators.models.dsv4_dspark.backbone import (
                block as _blkmod,
            )

            _blkmod._SAT_SUB = []  # arm per-sub-stage capture for the draft layers

        # ── MoE noaux_tc LOAD BALANCING (DSPARK_MOE_BALANCE=1): apply the bias nudge from the PREVIOUS
        #    step's stashed load, BEFORE this step's layers run — so the selection bias stays CONSTANT
        #    through this forward AND its activation-checkpoint recompute in backward. (Updating it AFTER
        #    the layers made the AC recompute re-route with a changed bias → CheckpointError shape-mismatch,
        #    which fires hard with a fresh/uniform router.) One-step lag is negligible; first step
        #    _step_load is None → no-op; the all-reduce inside makes the update identical across ranks.
        if self.training and getattr(self.layers[0].ffn.router, "_balance", False):
            if not getattr(self, "_balance_logged", False):
                self._balance_logged = True
                import torch.distributed as _d  # noqa: PLC0415

                if not _d.is_initialized() or _d.get_rank() == 0:
                    _r0 = self.layers[0].ffn.router
                    print(
                        f"[MOE-BALANCE] noaux_tc ON: rate={_r0._balance_rate} across "
                        f"{len(self.layers)} routers (bias updated PRE-layer from prev-step load; AC-safe)",
                        flush=True,
                    )
            for _layer in self.layers:
                _layer.ffn.router.update_load_balance_bias()

        final_ffn_pre = None
        for layer_idx, layer in enumerate(self.layers):
            attn_bias = (
                sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask
            )
            layer_args = (
                streams,
                fc_output,
                block_freqs,
                ctx_freqs,
                self._mask_to_bias(attn_bias),
            )
            if self.grad_checkpoint and self.training:
                # Recompute this layer (attn + EP-MoE all-to-all + mHC) in backward to free its
                # activation. use_reentrant=False so the collective inside the MoE recomputes
                # correctly under the saved-tensor hooks (standard AC+EP; the a2a re-runs in bwd).
                from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

                streams, final_ffn_pre = checkpoint(
                    layer, *layer_args, use_reentrant=False
                )
            else:
                streams, final_ffn_pre = layer(*layer_args)
            if _sat:
                _rec["layers"].append(streams.detach().float().cpu())

        # ── MoE LOAD-BALANCE diagnostic (DSPARK_LOG_EXPERT_LOAD=1): confirm/deny expert collapse
        #    (a few of the 256 experts hogging the routing). rank0, throttled ~1/20 fwd. Zero cost off.
        if getattr(self.layers[0].ffn.router, "_log_load", False):
            self._eload_ctr = getattr(self, "_eload_ctr", 0) + 1
            import torch.distributed as _dist  # noqa: PLC0415

            _rk = _dist.get_rank() if _dist.is_initialized() else 0
            if _rk == 0 and self._eload_ctr % 20 == 1:
                for _n in range(len(self.layers)):
                    _c = getattr(self.layers[_n].ffn.router, "_sel_counts", None)
                    if _c is None:
                        continue
                    E = _c.shape[0]
                    _cf = _c.float()
                    _tot = _cf.sum().clamp(min=1)
                    _p = _cf / _tot
                    _used = int((_c > 0).sum())
                    _tk = min(16, E)
                    _tv, _ti = _cf.topk(_tk)
                    _top = float(_tv.sum() / _tot)
                    _ent = float(
                        -(_p[_p > 0] * _p[_p > 0].log()).sum()
                        / torch.log(torch.tensor(float(E)))
                    )
                    _hot = (
                        _ti.tolist()
                    )  # the hottest expert IDs — compare across INIT/trained/datasets:
                    #   same IDs everywhere = router collapse; data-dependent IDs = data drives it.
                    print(
                        f"[MOE-LOAD L{_n}] used={_used}/{E} dead={E - _used} "
                        f"top{_tk}={_top:.2f} entropy={_ent:.3f}  hot={_hot}  "
                        f"(entropy 1.0=uniform / low=collapsed)",
                        flush=True,
                    )

        if _sat:
            self._satdumped = True
            assert final_ffn_pre is not None
            _rec["final_ffn_pre"] = final_ffn_pre.detach().float().cpu()
            _rec["head_hidden"] = (
                collapse_streams(streams, final_ffn_pre).detach().float().cpu()
            )
            _rec["substages"] = _blkmod._SAT_SUB
            _blkmod._SAT_SUB = None
            _sdir = os.environ.get(
                "DSPARK_SATDUMP_DIR", os.path.expanduser("~/dspark_sat")
            )
            os.makedirs(_sdir, exist_ok=True)
            torch.save(_rec, os.path.join(_sdir, "train_sat.pt"))
            print(
                f">>> [DSPARK_SATDUMP] train: embed + {len(_rec['layers'])} layers "
                f"({len(_rec['substages'])} sub-staged) + final ffn pre-mix "
                f"→ {_sdir}/train_sat.pt",
                flush=True,
            )

        assert final_ffn_pre is not None
        hidden = self.norm(collapse_streams(streams, final_ffn_pre))  # [1, TB, H]
        logits = self.lm_head(hidden)

        # ── DSPARK_TRAIN_PARITY_DUMP=1: one-shot, rank0 — write the FIRST anchor's block in the
        #    EXACT serve-dump format (dsv4_dspark_forward_parity_v2.py's PIECE 1 schema) so the SAME
        #    parity script can replay it. This VALIDATES run_train_block: fed this train-produced aux,
        #    run_train_block MUST reproduce these base_logits (≈0). If it does, the harness faithfully
        #    reproduces training → any gap vs the SERVE dump is a REAL train/serve inconsistency. If it
        #    does NOT, run_train_block's single-block rebuild is the bug (mask/anchor/context layout).
        if os.environ.get("DSPARK_TRAIN_PARITY_DUMP") == "1" and not getattr(
            self, "_train_parity_dumped", False
        ):
            import torch.distributed as _pd  # noqa: PLC0415

            if not _pd.is_initialized() or _pd.get_rank() == 0:
                self._train_parity_dumped = True
                _ap = anchor_positions.reshape(-1)
                if int(_ap.numel()) > 0:
                    _a0 = int(_ap[0])
                    _bs = self.block_size
                    _dp = (int(position_ids[0, _a0]) + torch.arange(_bs)).long()
                    _dir = os.environ.get(
                        "DSPARK_TRAIN_PARITY_DIR",
                        os.path.expanduser("~/dspark_train_parity"),
                    )
                    os.makedirs(_dir, exist_ok=True)
                    torch.save(
                        {
                            "aux": hidden_states[0, :_a0].detach().float().cpu(),
                            "ctx_positions": position_ids[0, :_a0].detach().cpu(),
                            "anchor_token": int(input_ids[0, _a0]),
                            "draft_positions": _dp.cpu(),
                            "serve_base_logits": logits[0, 0:_bs]
                            .detach()
                            .float()
                            .cpu(),  # TRAIN base logits
                            "drafted": logits[0, 0:_bs].argmax(-1).detach().cpu(),
                            "block_size": int(_bs),
                            "dspark_noise_token_id": int(self.mask_token_id),
                        },
                        os.path.join(_dir, "serve_block_0.pt"),
                    )
                    print(
                        f">>> [DSPARK_TRAIN_PARITY_DUMP] wrote serve_block_0.pt (a0={_a0}, "
                        f"ctx_len={_a0}, block={_bs}) to {_dir}",
                        flush=True,
                    )

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        if not self.config.sample_from_anchor:
            # False: slot 0 is the given anchor (not predicted) -> mask its loss. True
            # (DSpark): slot 0 IS a trained prediction (from the anchor's hidden) -> keep.
            aligned_loss_mask[:, :: self.block_size] = 0

        # DSPARK_DIAGNOSE=1: one-shot dump of the per-slot TARGET alignment — is
        # argmax(targets[slot k]) the TRUE next token input_ids[anchor+k+1]? Decides
        # "slot-0 target misaligned (data)" vs "target fine, draft can't learn (model)".
        if os.environ.get("DSPARK_DIAGNOSE") == "1" and not getattr(
            self, "_diag_done", False
        ):
            self._diag_done = True
            with torch.no_grad():
                bs = self.block_size
                tgt_arg = targets.view(-1, bs, targets.shape[-1]).argmax(
                    -1
                )  # [nb, bs] draft-vocab
                d2t = getattr(self, "d2t", None)
                ap = anchor_positions.reshape(-1)
                seqlen = input_ids.shape[1]
                print(
                    f"\n### DSPARK_DIAGNOSE sample_from_anchor={self.config.sample_from_anchor} "
                    f"block={bs} use_draft_vocab={getattr(self, 'use_draft_vocab', None)} ###",
                    flush=True,
                )
                for j in range(min(6, int(ap.numel()))):
                    a = int(ap[j])
                    true_next = [
                        int(input_ids[0, a + 1 + k]) if a + 1 + k < seqlen else -1
                        for k in range(bs)
                    ]
                    tgt = [int(tgt_arg[j, k]) for k in range(bs)]
                    tgt_v = [int(d2t[t]) for t in tgt] if d2t is not None else tgt
                    aligned = [tgt_v[k] == true_next[k] for k in range(bs)]
                    print(
                        f"a={a} tok[a]={int(input_ids[0, a])} | true_next(a+1..a+{bs})={true_next} "
                        f"| target_argmax(->verif)={tgt_v} | aligned={aligned}",
                        flush=True,
                    )
        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @staticmethod
    def _mask_to_bias(mask: torch.Tensor | None) -> torch.Tensor | None:
        """Reshape the DFlash eager float mask to our sink attn_bias [1, TB, Sk]."""
        if mask is None:
            return None
        # eager float mask is [1, 1, TB, Sk] (or [1, TB, Sk]); collapse the head dim.
        while mask.dim() > 3:
            mask = mask.squeeze(1)
        return mask
