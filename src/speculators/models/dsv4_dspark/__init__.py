"""Faithful DeepSeek-V4-Flash DSpark draft — the SPARSE line (Plan 甲, Track A).

A clean-room, backend-agnostic reproduction of DeepSeek's DSpark draft *backbone*
for DeepSeek-V4-Flash: multi-head latent attention + per-head attention sink +
256-expert MoE + hyper-connections, with NPU kernels inserted opt-in via
:mod:`.backbone.kernels`.

The DSpark *method* (anchor-block sampling, Markov + confidence heads, compound
loss, the SpeculatorModel wrapper + trainer/data contract) is reused verbatim
from upstream's dense-line ``speculators.models.dspark``; this package provides
only the DSV4-native backbone and the thin subclass that swaps it in.
"""

from __future__ import annotations

from .config import DSparkDraftConfig
from .core import DSV4DSparkConfig, DSV4DSparkDraftModel

__all__ = ["DSV4DSparkConfig", "DSV4DSparkDraftModel", "DSparkDraftConfig"]
