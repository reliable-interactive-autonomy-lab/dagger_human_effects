"""Configuration dataclasses for the DAgger / learned-helplessness pipeline.

Configs are plain dataclasses so they are introspectable and serialisable; a YAML
file may override any leaf via dotted keys or nested mappings.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# The low-dim observation set used across robomimic's published robosuite
# experiments (generate_paper_configs.py): end-effector pose, gripper fingers and
# task object state. Note the key is `object`, not robosuite's `object-state` --
# robomimic's dataset extraction renames it, so we match robomimic to stay
# compatible with its official datasets and checkpoints.
DEFAULT_LOW_DIM_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


@dataclass
class EnvConfig:
    env_name: str = "Lift"
    robots: str = "Panda"
    controller: Optional[str] = None  # None -> robosuite BASIC composite controller
    control_freq: int = 20
    horizon: int = 400
    reward_shaping: bool = False
    # Observation configuration
    low_dim_keys: List[str] = field(default_factory=lambda: list(DEFAULT_LOW_DIM_KEYS))
    image_keys: List[str] = field(default_factory=list)  # e.g. ["agentview_image"]
    camera_names: List[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    camera_height: int = 84
    camera_width: int = 84
    # Rename raw robosuite observation keys to robomimic's dataset naming.
    obs_key_aliases: Dict[str, str] = field(
        default_factory=lambda: {"object": "object-state"})
    # Reproduce robosuite <=1.4.1 object-state sign conventions, which is what
    # robomimic's published datasets contain. Required for pretraining on those
    # datasets; see dagger_lh/envs.py::OBJ_EEF_REL_SUFFIXES.
    legacy_object_state: bool = True
    # Rendering for the human operator (separate, higher-res than policy input)
    render_camera: str = "agentview"
    render_height: int = 512
    render_width: int = 512
    seed: Optional[int] = None

    @property
    def use_image_obs(self) -> bool:
        return len(self.image_keys) > 0


@dataclass
class TeleopConfig:
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    # Smoothing on the human command (first-order lag); 1.0 = no smoothing.
    smoothing: float = 0.6
    # Hold-to-intervene (True) vs press-to-toggle (False)
    hold_to_intervene: bool = False
    window_scale: float = 1.0
    show_second_view: bool = True
    fps_cap: int = 20


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # Which module name fragments receive adapters. Empty -> all eligible modules.
    target_substrings: List[str] = field(default_factory=list)
    adapt_rnn: bool = True          # low-rank deltas on LSTM weight matrices
    train_biases: bool = True       # also unfreeze biases (BitFit-style)
    train_output_head: bool = False  # fully unfreeze the final action head


@dataclass
class PolicyConfig:
    """Policy architecture.

    Defaults reproduce the published reference configurations:
      * bc_rnn -- robomimic's BC-RNN paper config for low-dim robosuite tasks
        (seq_length 10, LSTM hidden 400, GMM head with 5 modes, and *no* MLP
        layers between the RNN and the output head; lr 1e-4).
        See robomimic/scripts/generate_paper_configs.py.
      * act -- LeRobot's `ACTConfig` defaults (dim_model 512, 8 heads, FFN 3200,
        4 encoder layers, 1 decoder layer, latent 32, kl_weight 10, dropout 0.1).
        `chunk_size` is the one deliberate change: the reference 100 is tuned for
        50 Hz ALOHA, so 20 (~1 s at robosuite's 20 Hz control rate) is used here.
    """
    algo: str = "bc_rnn"  # "bc_rnn" | "act"

    # --- bc_rnn: robomimic BC-RNN paper config (low_dim) ---
    mlp_layer_dims: List[int] = field(default_factory=list)  # paper: no MLP layers
    rnn_hidden_dim: int = 400
    rnn_num_layers: int = 2
    num_modes: int = 5
    min_std: float = 0.0001
    std_activation: str = "softplus"
    low_noise_eval: bool = True
    seq_length: int = 10
    use_gmm: bool = True

    # --- act: LeRobot ACTConfig defaults ---
    chunk_size: int = 20
    act_hidden_dim: int = 512
    act_n_heads: int = 8
    act_enc_layers: int = 4
    act_dec_layers: int = 1
    act_latent_dim: int = 32
    act_kl_weight: float = 10.0
    act_dropout: float = 0.1
    temporal_agg: bool = True      # reference ACT temporal ensembling
    temporal_agg_k: float = 0.01   # reference coefficient

    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class TrainConfig:
    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"
    lr: float = 1e-3               # LoRA params tolerate a higher LR than full FT
    base_lr: float = 1e-4          # full (pre)training; robomimic BC-RNN paper value
    weight_decay: float = 0.0
    batch_size: int = 32
    # On-the-fly finetuning between rounds
    finetune_steps: int = 300
    finetune_max_seconds: float = 25.0
    # Floor on how long the "updating policy" screen is shown. Set this above the
    # slowest real update so every condition -- including the sham update -- takes
    # the same wall-clock time; otherwise the delay itself leaks the condition.
    finetune_min_seconds: float = 0.0
    grad_clip: float = 10.0
    # Full pretraining
    pretrain_epochs: int = 100
    pretrain_steps_per_epoch: int = 100
    # Replay balancing: probability of sampling from the newest round's data
    new_data_prob: float = 0.5
    num_workers: int = 0
    seed: int = 0


@dataclass
class StudyConfig:
    """Learned-helplessness study parameters.

    The manipulation is the *contingency* between the operator's corrective effort
    and the policy's improvement. See dagger_lh/study.py for the condition logic.
    """
    participant_id: str = "p000"
    # Condition applied during phase 1
    condition: str = "responsive"
    # Two-phase (induction -> restored control) escape test. This is the classic
    # learned-helplessness paradigm: if helplessness was induced, participants
    # fail to exploit controllability once it is restored.
    two_phase: bool = False
    phase1_rounds: int = 6
    phase2_rounds: int = 6
    phase2_condition: str = "responsive"
    # Single-phase length (used when two_phase is False)
    num_rounds: int = 10
    rollouts_per_round: int = 1
    # Scripted-success (`yoked`) knobs. Requires a ladder built by
    # scripts/train_policy_ladder.py; without one, `yoked` cannot make the
    # policy's competence follow a curve and is refused at startup.
    ladder_path: Optional[str] = None
    yoked_schedule: List[float] = field(default_factory=list)
    # Sham/noncontingent knobs
    sham_success_cap: float = 0.0     # forced ceiling on reported success rate
    degrade_noise_std: float = 0.02   # per-round action noise added under "degraded"
    degrade_growth: float = 0.015     # noise added per round under "degraded"
    # Surveys
    survey_enabled: bool = True
    survey_every_n_rounds: int = 1
    # Where the operator is told the policy stands (perceived-progress feedback)
    show_progress_feedback: bool = True
    seed: int = 0


@dataclass
class PathsConfig:
    root: str = "data"
    demos: str = "data/demos"
    checkpoints: str = "data/checkpoints"
    sessions: str = "data/sessions"
    logs: str = "data/logs"


@dataclass
class Config:
    mode: str = "dagger"  # "dagger" | "demo"
    env: EnvConfig = field(default_factory=EnvConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    base_checkpoint: Optional[str] = None
    demo_dataset: Optional[str] = None
    record_video: bool = True

    # ---------------- (de)serialisation ----------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return _build(cls, d or {})

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        if path is None:
            return cls()
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def override(self, dotted: Dict[str, Any]) -> "Config":
        """Apply `{"policy.algo": "act"}`-style overrides, returning a new Config."""
        d = self.to_dict()
        for key, val in dotted.items():
            if val is None:
                continue
            node = d
            parts = key.split(".")
            for p in parts[:-1]:
                if p not in node or not isinstance(node[p], dict):
                    raise KeyError(f"unknown config section '{p}' in '{key}'")
                node = node[p]
            if parts[-1] not in node:
                raise KeyError(f"unknown config key '{key}'")
            node[parts[-1]] = val
        return Config.from_dict(d)


def _build(cls, d: Dict[str, Any]):
    """Recursively instantiate nested dataclasses from a plain dict."""
    if not dataclasses.is_dataclass(cls):
        return d
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    for name, f in fields.items():
        if name not in d:
            continue
        val = d[name]
        if dataclasses.is_dataclass(f.type) and isinstance(val, dict):
            kwargs[name] = _build(f.type, val)
        elif isinstance(val, dict) and hasattr(f.type, "__dataclass_fields__"):
            kwargs[name] = _build(f.type, val)
        else:
            kwargs[name] = copy.deepcopy(val)
    # Nested dataclass fields declared via `field(default_factory=...)` whose type is
    # a string annotation (from __future__ annotations) need manual handling.
    for name in ("env", "teleop", "policy", "train", "study", "paths", "lora"):
        if name in kwargs and isinstance(kwargs[name], dict):
            target = _NESTED.get(name)
            if target is not None:
                kwargs[name] = _build(target, kwargs[name])
    return cls(**kwargs)


_NESTED = {
    "env": EnvConfig,
    "teleop": TeleopConfig,
    "policy": PolicyConfig,
    "train": TrainConfig,
    "study": StudyConfig,
    "paths": PathsConfig,
    "lora": LoRAConfig,
}
