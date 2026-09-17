"""Common policy interface shared by the BC-RNN and ACT implementations."""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import PolicyConfig


def resolve_device(spec: str = "auto") -> torch.device:
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ObsNormalizer(nn.Module):
    """Per-key mean/std normalisation for low-dim observations.

    Registered as buffers so the stats travel with the checkpoint and are never
    treated as trainable parameters by the LoRA machinery.
    """

    def __init__(self, shapes: Dict[str, Tuple[int, ...]], low_dim_keys):
        super().__init__()
        self.low_dim_keys = list(low_dim_keys)
        self._safe = {k: k.replace("-", "__") for k in self.low_dim_keys}
        for k in self.low_dim_keys:
            dim = int(np.prod(shapes[k]))
            self.register_buffer(f"mean_{self._safe[k]}", torch.zeros(dim))
            self.register_buffer(f"std_{self._safe[k]}", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1))

    @torch.no_grad()
    def fit(self, stats: Dict[str, Dict[str, np.ndarray]]) -> None:
        for k in self.low_dim_keys:
            if k not in stats:
                continue
            m = torch.as_tensor(np.asarray(stats[k]["mean"]).ravel(), dtype=torch.float32)
            s = torch.as_tensor(np.asarray(stats[k]["std"]).ravel(), dtype=torch.float32)
            s = torch.clamp(s, min=1e-3)
            getattr(self, f"mean_{self._safe[k]}").copy_(m.to(self.fitted.device))
            getattr(self, f"std_{self._safe[k]}").copy_(s.to(self.fitted.device))
        self.fitted.fill_(1.0)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if float(self.fitted.item()) < 0.5:
            return obs
        out = dict(obs)
        for k in self.low_dim_keys:
            if k in out:
                m = getattr(self, f"mean_{self._safe[k]}")
                s = getattr(self, f"std_{self._safe[k]}")
                out[k] = (out[k] - m) / s
        return out


class BasePolicy(nn.Module, abc.ABC):
    """A trainable policy that can also be rolled out one step at a time."""

    #: number of consecutive timesteps a training batch must provide
    seq_length: int = 1

    def __init__(self, obs_shapes: Dict[str, Tuple[int, ...]], ac_dim: int,
                 cfg: PolicyConfig, low_dim_keys, image_keys):
            super().__init__()
            self.obs_shapes = {k: tuple(v) for k, v in obs_shapes.items()}
            self.ac_dim = int(ac_dim)
            self.cfg = cfg
            self.low_dim_keys = list(low_dim_keys)
            self.image_keys = list(image_keys)
            self.normalizer = ObsNormalizer(self.obs_shapes, self.low_dim_keys)
            # Calibrated exploration noise applied to every action at rollout.
            # Stored as a buffer so it travels with the checkpoint: a ladder rung
            # saved at 30% competence *is* a 30% policy when loaded, with no
            # external bookkeeping to forget. See scripts/calibrate_ladder.py.
            self.register_buffer("action_noise_std", torch.zeros(1))

    # ------------------------------------------------------------------ training
    @abc.abstractmethod
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return (scalar loss, dict of scalar metrics for logging)."""

    # ------------------------------------------------------------------ rollout
    @abc.abstractmethod
    def reset_rollout(self) -> None:
        """Clear any recurrent / action-chunk state between episodes."""

    @abc.abstractmethod
    def _act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Single-step action for a single (non-batched) observation."""

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Policy action with any baked-in rollout noise applied."""
        action = self._act(obs)
        sigma = float(self.action_noise_std.item())
        if sigma > 0.0:
            # torch's generator, so seeding a rollout makes this reproducible.
            noise = torch.randn(action.shape, dtype=torch.float64).numpy() * sigma
            action = np.clip(np.asarray(action, dtype=np.float64) + noise, -1.0, 1.0)
        return action

    def set_action_noise(self, sigma: float) -> None:
        with torch.no_grad():
            self.action_noise_std.fill_(float(sigma))

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _to_tensor_obs(self, obs: Dict[str, np.ndarray], add_time: bool = False
                       ) -> Dict[str, torch.Tensor]:
        """numpy observation -> batched float tensors on the policy's device."""
        dev = self.device
        out: Dict[str, torch.Tensor] = {}
        for k in self.low_dim_keys:
            t = torch.as_tensor(np.asarray(obs[k], dtype=np.float32), device=dev).reshape(1, -1)
            out[k] = t.unsqueeze(1) if add_time else t
        for k in self.image_keys:
            arr = np.asarray(obs[k])
            if arr.ndim == 3 and arr.shape[-1] == 3:  # HWC uint8 -> CHW float
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.as_tensor(arr.astype(np.float32) / 255.0, device=dev).unsqueeze(0)
            out[k] = t.unsqueeze(1) if add_time else t
        return out

    @staticmethod
    def prepare_batch_obs(batch_obs: Dict[str, torch.Tensor], image_keys,
                          device: torch.device) -> Dict[str, torch.Tensor]:
        """Move a dataset batch to `device`, converting images to float CHW in [0,1]."""
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch_obs.items():
            t = v.to(device, non_blocking=True)
            if k in image_keys:
                t = t.float() / 255.0
                if t.shape[-1] == 3:  # (..., H, W, C) -> (..., C, H, W)
                    t = t.permute(*range(t.ndim - 3), t.ndim - 1, t.ndim - 3, t.ndim - 2)
            else:
                t = t.float()
            out[k] = t
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "obs_shapes": self.obs_shapes,
            "ac_dim": self.ac_dim,
            "low_dim_keys": self.low_dim_keys,
            "image_keys": self.image_keys,
            "cfg": self.cfg,
            "extra": extra or {},
        }, path)
