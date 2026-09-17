"""ACT (Action Chunking Transformer) backed by LeRobot's reference implementation.

Rather than reimplementing ACT, this wraps `lerobot.policies.act.modeling_act.ACT`
-- the maintained reference port of Zhao et al. 2023 -- so architecture, the VAE
objective and temporal ensembling all match the published model. Defaults come
straight from `ACTConfig` (dim_model 512, 8 heads, FFN 3200, 4 encoder layers,
1 decoder layer, latent 32, kl_weight 10, ImageNet-pretrained ResNet18).

This wrapper supplies only what the DAgger loop needs on top:
  * robosuite observation dicts -> LeRobot's flat batch keys
  * the `BasePolicy` train/rollout interface
  * LoRA-compatible module structure (LeRobot's ACT uses `nn.MultiheadAttention`,
    whose packed `in_proj_weight` cannot take an adapter, so `patch_attention_for_lora`
    rewrites those blocks into explicit q/k/v/out `nn.Linear`s with identical maths)

Observation routing for robosuite: `robot0_*` keys are concatenated into
`observation.state`, remaining low-dim keys (e.g. `object`) into
`observation.environment_state`, and each camera into `observation.images`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PolicyConfig
from .base import BasePolicy

# LeRobot batch keys
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGES = "observation.images"
ACTION = "action"


def _split_low_dim(low_dim_keys: List[str]) -> Tuple[List[str], List[str]]:
    """robosuite proprioception -> robot state; everything else -> env state."""
    robot = [k for k in low_dim_keys if k.startswith("robot")]
    env = [k for k in low_dim_keys if not k.startswith("robot")]
    return robot, env


class _ExplicitMHA(nn.Module):
    """Drop-in replacement for `nn.MultiheadAttention` with separate projections.

    LeRobot's ACT uses `nn.MultiheadAttention`, which packs q/k/v into a single
    `in_proj_weight` Parameter and reads `out_proj.weight` directly -- neither can
    carry a LoRA adapter. This module is mathematically identical (weights are
    copied across, including the packed in-projection split) but exposes four
    ordinary `nn.Linear`s that `inject_lora` can wrap.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = embed_dim // mha.num_heads
        self.dropout = mha.dropout
        bias = mha.in_proj_bias is not None
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim,
                                  bias=mha.out_proj.bias is not None)
        with torch.no_grad():
            w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
            self.q_proj.weight.copy_(w_q)
            self.k_proj.weight.copy_(w_k)
            self.v_proj.weight.copy_(w_v)
            if bias:
                b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.copy_(b_q)
                self.k_proj.bias.copy_(b_k)
                self.v_proj.bias.copy_(b_v)
            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # ACT runs attention with (seq, batch, dim) tensors.
        S, B, _ = x.shape
        return x.view(S, B * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, **kwargs):
        S, B, _ = query.shape
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        mask = None
        if key_padding_mask is not None:
            # (B, S_k) bool -> broadcastable additive mask over (B*H, S_q, S_k)
            m = key_padding_mask.view(B, 1, 1, -1).expand(B, self.num_heads, 1, -1)
            m = m.reshape(B * self.num_heads, 1, -1)
            mask = torch.zeros_like(m, dtype=q.dtype).masked_fill(m, float("-inf"))
        if attn_mask is not None:
            am = attn_mask if attn_mask.dtype != torch.bool else \
                torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float("-inf"))
            mask = am if mask is None else mask + am

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        o = o.transpose(0, 1).contiguous().view(S, B, self.num_heads * self.head_dim)
        return self.out_proj(o), None


def patch_attention_for_lora(module: nn.Module) -> int:
    """Recursively swap `nn.MultiheadAttention` for `_ExplicitMHA`. Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, _ExplicitMHA(child))
            n += 1
        else:
            n += patch_attention_for_lora(child)
    return n


class ACTPolicy(BasePolicy):
    seq_length = 1  # one observation in, a chunk of actions out

    def __init__(self, obs_shapes: Dict[str, Tuple[int, ...]], ac_dim: int,
                 cfg: PolicyConfig, low_dim_keys, image_keys):
        super().__init__(obs_shapes, ac_dim, cfg, low_dim_keys, image_keys)
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACT, ACTTemporalEnsembler

        self.robot_keys, self.env_keys = _split_low_dim(self.low_dim_keys)
        self.state_dim = sum(int(np.prod(self.obs_shapes[k])) for k in self.robot_keys)
        self.env_dim = sum(int(np.prod(self.obs_shapes[k])) for k in self.env_keys)
        self.chunk_size = int(cfg.chunk_size)

        input_features: Dict[str, Any] = {}
        if self.state_dim:
            input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.state_dim,))
        if self.env_dim:
            input_features[OBS_ENV_STATE] = PolicyFeature(
                type=FeatureType.ENV, shape=(self.env_dim,))
        self.image_feature_keys: List[str] = []
        for k in self.image_keys:
            fk = f"observation.images.{k}"
            self.image_feature_keys.append(fk)
            input_features[fk] = PolicyFeature(
                type=FeatureType.VISUAL, shape=tuple(self.obs_shapes[k]))

        if not input_features:
            raise ValueError("ACT needs at least one observation feature")

        self.act_config = ACTConfig(
            n_obs_steps=1,
            chunk_size=self.chunk_size,
            # Temporal ensembling requires re-querying every step, so LeRobot
            # constrains n_action_steps to 1 in that mode.
            n_action_steps=(1 if cfg.temporal_agg else self.chunk_size),
            input_features=input_features,
            output_features={ACTION: PolicyFeature(
                type=FeatureType.ACTION, shape=(self.ac_dim,))},
            normalization_mapping={
                # Observations are normalised by this package's ObsNormalizer and
                # robosuite actions already live in [-1, 1], so LeRobot's own
                # normalisation layers are bypassed.
                "STATE": NormalizationMode.IDENTITY,
                "ENV": NormalizationMode.IDENTITY,
                "VISUAL": NormalizationMode.IDENTITY,
                "ACTION": NormalizationMode.IDENTITY,
            },
            dim_model=int(cfg.act_hidden_dim),
            n_heads=int(cfg.act_n_heads),
            n_encoder_layers=int(cfg.act_enc_layers),
            n_decoder_layers=int(cfg.act_dec_layers),
            latent_dim=int(cfg.act_latent_dim),
            kl_weight=float(cfg.act_kl_weight),
            dropout=float(cfg.act_dropout),
            use_vae=True,
            temporal_ensemble_coeff=(float(cfg.temporal_agg_k)
                                     if cfg.temporal_agg else None),
            vision_backbone="resnet18",
            pretrained_backbone_weights=(
                "ResNet18_Weights.IMAGENET1K_V1" if image_keys else None),
        )
        self.model = ACT(self.act_config)
        # Make every attention block LoRA-adaptable (mathematically a no-op).
        self._n_patched_attn = patch_attention_for_lora(self.model)

        self._ensembler_cls = ACTTemporalEnsembler
        self._ensembler = None
        self._queue: List[np.ndarray] = []

    # ------------------------------------------------------------------ batching
    def _concat(self, obs: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        return torch.cat([obs[k] for k in keys], dim=-1)

    def _to_lerobot_batch(self, obs: Dict[str, torch.Tensor],
                          actions: Optional[torch.Tensor] = None,
                          pad: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        if self.robot_keys:
            batch[OBS_STATE] = self._concat(obs, self.robot_keys)
        if self.env_keys:
            batch[OBS_ENV_STATE] = self._concat(obs, self.env_keys)
        for fk, k in zip(self.image_feature_keys, self.image_keys):
            batch[fk] = obs[k]
        if self.image_feature_keys:
            batch[OBS_IMAGES] = [batch[fk] for fk in self.image_feature_keys]
        if actions is not None:
            batch[ACTION] = actions
            # LeRobot marks *padded* steps as True; our dataset marks valid as 1.
            batch["action_is_pad"] = (pad < 0.5) if pad is not None else torch.zeros(
                actions.shape[:2], dtype=torch.bool, device=actions.device)
        return batch

    def _squeeze_time(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Drop the length-1 time axis the sequence dataset adds (ACT is single-obs)."""
        out = {}
        for k, v in obs.items():
            if k in self.image_keys:
                out[k] = v[:, 0] if v.ndim == 5 else v      # (B,T,C,H,W) -> (B,C,H,W)
            else:
                out[k] = v[:, 0] if v.ndim == 3 else v      # (B,T,D)     -> (B,D)
        return out

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device
        obs = self.prepare_batch_obs(batch["obs"], self.image_keys, dev)
        obs = self._squeeze_time(self.normalizer(obs))
        actions = batch["actions"].to(dev).float()
        pad = batch.get("action_pad_mask")
        pad = pad.to(dev).float() if pad is not None else None
        lb = self._to_lerobot_batch(obs, actions, pad)

        actions_hat, (mu, log_sigma_x2) = self.model(lb)
        # Reference ACT objective (lerobot ACTPolicy.forward): masked L1 + KL.
        valid = ~lb["action_is_pad"]
        l1 = (F.l1_loss(actions, actions_hat, reduction="none")
              * valid.unsqueeze(-1)).mean()
        metrics = {"l1": float(l1.item())}
        if self.act_config.use_vae and mu is not None:
            mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
                        ).sum(-1).mean()
            loss = l1 + mean_kld * self.act_config.kl_weight
            metrics["kl"] = float(mean_kld.item())
        else:
            loss = l1
        metrics["loss"] = float(loss.item())
        return loss, metrics

    # ------------------------------------------------------------------ rollout
    def reset_rollout(self) -> None:
        self._queue = []
        if self.act_config.temporal_ensemble_coeff is not None:
            self._ensembler = self._ensembler_cls(
                self.act_config.temporal_ensemble_coeff, self.chunk_size)
            self._ensembler.reset()

    @torch.no_grad()
    def _chunk(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        obs_t = self.normalizer(self._to_tensor_obs(obs, add_time=False))
        lb = self._to_lerobot_batch(obs_t)
        return self.model(lb)[0]  # (1, chunk, ac_dim)

    @torch.no_grad()
    def _act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            if self._ensembler is not None:
                chunk = self._chunk(obs)
                action = self._ensembler.update(chunk)[0]
                return np.clip(action.cpu().numpy(), -1.0, 1.0)
            if not self._queue:
                chunk = self._chunk(obs)[0].cpu().numpy()
                self._queue = [chunk[i] for i in range(chunk.shape[0])]
            return np.clip(self._queue.pop(0), -1.0, 1.0)
        finally:
            if was_training:
                self.train()
