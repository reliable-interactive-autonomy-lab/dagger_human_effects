"""BC-RNN backed by robomimic's reference `BC_RNN` algorithm.

The network, the GMM action head, the observation encoders and the training step
all come from robomimic itself: the policy is built through
`robomimic.algo.algo_factory("bc", config, ...)` using a config assembled from
robomimic's own `config_factory("bc")` and then set to the published BC-RNN
hyperparameters for low-dim robosuite tasks (see
`robomimic/scripts/generate_paper_configs.py::modify_bc_rnn_config_for_dataset`):

    train.seq_length      = 10        algo.rnn.horizon      = 10
    algo.rnn.enabled      = True      algo.rnn.hidden_dim   = 400  (1000 for image)
    algo.gmm.enabled      = True      algo.actor_layer_dims = ()   -- no MLP layers
    optim lr              = 1e-4

This wrapper adds the `BasePolicy` interface, observation normalisation and LoRA
compatibility; loss computation is delegated to robomimic's `BC_RNN`.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..config import PolicyConfig
from .base import BasePolicy


def init_obs_utils(low_dim_keys, image_keys) -> None:
    """robomimic tracks observation modalities in module-level global state."""
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs={
        "obs": {"low_dim": list(low_dim_keys), "rgb": list(image_keys)}
    })


def build_bc_rnn_config(cfg: PolicyConfig, low_dim_keys, image_keys):
    """robomimic BCConfig set to the published BC-RNN hyperparameters."""
    import robomimic.models.obs_core  # noqa: F401  (registers VisualCore etc.)
    from robomimic.config import config_factory

    config = config_factory("bc")
    has_images = len(list(image_keys)) > 0

    with config.values_unlocked():
        config.observation.modalities.obs.low_dim = list(low_dim_keys)
        config.observation.modalities.obs.rgb = list(image_keys)
        config.observation.modalities.goal.low_dim = []
        config.observation.modalities.goal.rgb = []

        # --- paper BC-RNN settings ---
        config.train.seq_length = int(cfg.seq_length)
        config.algo.rnn.enabled = True
        config.algo.rnn.horizon = int(cfg.seq_length)
        config.algo.rnn.hidden_dim = int(cfg.rnn_hidden_dim)
        config.algo.rnn.rnn_type = "LSTM"
        config.algo.rnn.num_layers = int(cfg.rnn_num_layers)
        config.algo.rnn.open_loop = False
        config.algo.rnn.kwargs.bidirectional = False
        config.algo.gmm.enabled = bool(cfg.use_gmm)
        config.algo.gmm.num_modes = int(cfg.num_modes)
        config.algo.gmm.min_std = float(cfg.min_std)
        config.algo.gmm.std_activation = cfg.std_activation
        config.algo.gmm.low_noise_eval = bool(cfg.low_noise_eval)
        config.algo.actor_layer_dims = tuple(cfg.mlp_layer_dims)  # () in the paper

        # Paper visual encoder: ResNet18 + spatial-softmax keypoints, 64-d feature.
        config.observation.encoder.rgb.core_class = "VisualCore"
        config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
        config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
        config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
        config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
        config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
        config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
        config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
        config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
        config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0
        if has_images and int(cfg.rnn_hidden_dim) == 400:
            # The paper uses RNN dim 1000 for image observations.
            config.algo.rnn.hidden_dim = 1000
    return config


class BCRNNPolicy(BasePolicy):
    """Wraps robomimic's `BC_RNN` algo object."""

    def __init__(self, obs_shapes: Dict[str, Tuple[int, ...]], ac_dim: int,
                 cfg: PolicyConfig, low_dim_keys, image_keys):
        super().__init__(obs_shapes, ac_dim, cfg, low_dim_keys, image_keys)
        import robomimic.algo as algo_module

        init_obs_utils(low_dim_keys, image_keys)
        self.seq_length = int(cfg.seq_length)
        self.rm_config = build_bc_rnn_config(cfg, low_dim_keys, image_keys)

        obs_key_shapes = OrderedDict(
            (k, list(self.obs_shapes[k])) for k in self._ordered_keys()
        )
        # robomimic's algo_factory builds the reference BC_RNN, including its
        # RNNGMMActorNetwork and observation encoders.
        self.algo = algo_module.algo_factory(
            algo_name="bc",
            config=self.rm_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=self.ac_dim,
            device=torch.device("cpu"),  # moved by BasePolicy.to()
        )
        # Register robomimic's nets so .to(), .parameters() and state_dict() work.
        self.nets = self.algo.nets
        self._rnn_state = None

    def _ordered_keys(self) -> List[str]:
        """Stable key order -- robomimic concatenates encoder outputs in dict order."""
        return list(self.low_dim_keys) + list(self.image_keys)

    @property
    def net(self):
        """The underlying `RNNGMMActorNetwork` (LoRA injection target)."""
        return self.algo.nets["policy"]

    def to(self, *args, **kwargs):  # type: ignore[override]
        out = super().to(*args, **kwargs)
        # Keep robomimic's own device bookkeeping in sync.
        try:
            self.algo.device = next(self.nets.parameters()).device
        except StopIteration:
            pass
        return out

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = self.device
        obs = self.prepare_batch_obs(batch["obs"], self.image_keys, dev)
        obs = self.normalizer(obs)
        actions = batch["actions"].to(dev).float()

        # robomimic's BC_RNN loss path: forward the policy, then _compute_losses.
        rm_batch = {"obs": obs, "goal_obs": None, "actions": actions}
        predictions = self.algo._forward_training(rm_batch)
        losses = self.algo._compute_losses(predictions, rm_batch)

        loss = losses["action_loss"]
        pad = batch.get("action_pad_mask")
        if pad is not None and "log_probs" in predictions:
            # Exclude timesteps that only exist because the window ran past the
            # end of the episode. robomimic's own loader drops these instead.
            w = pad.to(dev).float()
            lp = predictions["log_probs"]
            if lp.shape == w.shape:
                loss = -(lp * w).sum() / w.sum().clamp(min=1e-6)
        metrics = {k: float(v.item()) for k, v in losses.items()
                   if isinstance(v, torch.Tensor) and v.numel() == 1}
        if "log_probs" in losses:
            metrics["nll"] = -float(losses["log_probs"].item())
        metrics.setdefault("nll", float(loss.item()))
        return loss, metrics

    # ------------------------------------------------------------------ rollout
    def reset_rollout(self) -> None:
        self._rnn_state = None
        self.algo.reset()

    @torch.no_grad()
    def _act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            obs_t = self.normalizer(self._to_tensor_obs(obs, add_time=False))
            if self._rnn_state is None:
                self._rnn_state = self.net.get_rnn_init_state(
                    batch_size=1, device=self.device)
            action, self._rnn_state = self.net.forward_step(
                obs_dict=obs_t, rnn_state=self._rnn_state)
            return np.clip(action[0].detach().cpu().numpy(), -1.0, 1.0)
        finally:
            if was_training:
                self.train()
