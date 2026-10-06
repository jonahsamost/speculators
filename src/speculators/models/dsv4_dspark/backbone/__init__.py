"""Backend-agnostic DSV4 decoder backbone for the DSpark draft.

Fresh, self-contained torch port of the DeepSeek-V4-Flash decoder shape
(multi-head latent attention + per-head attention sink + 256-expert MoE +
hyper-connections), with accelerator kernels inserted opt-in via
:mod:`.kernels`. No dependency on any private repo or accelerator package.
"""

from __future__ import annotations

from .kernels import (
    get_kernel,
    register_kernel,
    set_active_backend,
    torch_kernel,
)
from .norm import RMSNorm, UnweightedRMSNorm

__all__ = [
    "RMSNorm",
    "UnweightedRMSNorm",
    "get_kernel",
    "register_kernel",
    "set_active_backend",
    "torch_kernel",
]
