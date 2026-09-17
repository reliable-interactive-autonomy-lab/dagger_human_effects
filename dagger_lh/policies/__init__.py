"""Policy implementations and construction helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from ..config import PolicyConfig
from .base import BasePolicy, ObsNormalizer, resolve_device

_REGISTRY = {}


def _registry():
    if not _REGISTRY:
        from .act import ACTPolicy
        from .bc_rnn import BCRNNPolicy
        _REGISTRY.update(bc_rnn=BCRNNPolicy, act=ACTPolicy)
    return _REGISTRY


def available_algos() -> Tuple[str, ...]:
    return tuple(sorted(_registry()))


def build_policy(
    obs_shapes: Dict[str, Tuple[int, ...]],
    ac_dim: int,
    cfg: PolicyConfig,
    low_dim_keys: Iterable[str],
    image_keys: Iterable[str],
    device: Optional[torch.device] = None,
) -> BasePolicy:
    reg = _registry()
    if cfg.algo not in reg:
        raise ValueError(f"unknown algo '{cfg.algo}'; choose from {available_algos()}")
    policy = reg[cfg.algo](obs_shapes, ac_dim, cfg, list(low_dim_keys), list(image_keys))
    if device is not None:
        policy = policy.to(device)
    return policy


def load_policy(path: str, device: Optional[torch.device] = None,
                cfg_override: Optional[PolicyConfig] = None) -> Tuple[BasePolicy, Dict]:
    """Rebuild a policy from a checkpoint written by `BasePolicy.save`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = cfg_override or ckpt["cfg"]
    policy = build_policy(
        ckpt["obs_shapes"], ckpt["ac_dim"], cfg,
        ckpt["low_dim_keys"], ckpt["image_keys"],
    )
    # A checkpoint saved after LoRA injection carries adapter keys; rebuild them.
    if any("lora_" in k for k in ckpt["state_dict"]):
        from ..lora import inject_lora
        inject_lora(
            policy,
            rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout,
            target_substrings=cfg.lora.target_substrings, adapt_rnn=cfg.lora.adapt_rnn,
        )
    sd = dict(ckpt["state_dict"])
    # Checkpoints written before `action_noise_std` existed lack that buffer.
    # Default it to zero rather than failing: an old checkpoint is simply a
    # policy with no baked-in rollout noise.
    for key, ref in policy.state_dict().items():
        if key.endswith("action_noise_std") and key not in sd:
            sd[key] = torch.zeros_like(ref)
    policy.load_state_dict(sd)
    if device is not None:
        policy = policy.to(device)
    return policy, ckpt.get("extra", {})


__all__ = [
    "BasePolicy", "ObsNormalizer", "resolve_device",
    "build_policy", "load_policy", "available_algos",
]
