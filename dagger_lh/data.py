"""Demonstration storage and sequence sampling.

Files are written in **robomimic's HDF5 layout**, so anything collected here can be
fed straight to `robomimic/scripts/train.py`, `dataset_states_to_obs.py`, etc. Two
extra per-timestep arrays are added that robomimic ignores but the study needs:

  ``actor``        0 = policy acted, 1 = human acted
  ``intervention`` 1 while the human held control authority

The loader below is purpose-built rather than reusing `robomimic.utils.dataset.
SequenceDataset` because DAgger needs things that class does not offer: in-memory
aggregation across rounds, action-chunk targets with padding masks for ACT, and
round-recency resampling.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# ======================================================================== writing

@dataclass
class EpisodeBuffer:
    """Accumulates one episode in memory before it is committed to disk."""
    obs: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[int] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    actor: List[int] = field(default_factory=list)
    intervention: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    model_xml: str = ""

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray, reward: float,
            done: bool, state: Optional[np.ndarray] = None,
            actor: int = 1, intervention: int = 1) -> None:
        for k, v in obs.items():
            self.obs.setdefault(k, []).append(np.asarray(v))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(int(done))
        if state is not None:
            self.states.append(np.asarray(state, dtype=np.float64))
        self.actor.append(int(actor))
        self.intervention.append(int(intervention))

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def num_human_steps(self) -> int:
        return int(np.sum(self.actor)) if self.actor else 0


class DemoWriter:
    """Appends episodes to a robomimic-format HDF5 file."""

    def __init__(self, path: str, env_meta: Dict[str, Any]):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.env_meta = env_meta
        self._n = 0
        with h5py.File(self.path, "a") as f:
            grp = f.require_group("data")
            grp.attrs["env_args"] = json.dumps(env_meta)
            existing = [k for k in grp.keys() if k.startswith("demo_")]
            self._n = len(existing)
            grp.attrs["total"] = int(grp.attrs.get("total", 0))

    @property
    def num_demos(self) -> int:
        return self._n

    def write_episode(self, ep: EpisodeBuffer) -> Optional[str]:
        if len(ep) == 0:
            return None
        name = f"demo_{self._n}"
        with h5py.File(self.path, "a") as f:
            data = f.require_group("data")
            g = data.create_group(name)
            T = len(ep)
            g.attrs["num_samples"] = T
            if ep.model_xml:
                g.attrs["model_file"] = ep.model_xml
            for k, v in ep.meta.items():
                g.attrs[k] = v if isinstance(v, (int, float, bool, str)) else json.dumps(v)

            obs_g = g.create_group("obs")
            nobs_g = g.create_group("next_obs")
            for k, seq in ep.obs.items():
                arr = np.stack(seq)
                obs_g.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
                # next_obs is obs shifted by one, last frame repeated (robomimic
                # does the same for terminal transitions).
                nxt = np.concatenate([arr[1:], arr[-1:]], axis=0)
                nobs_g.create_dataset(k, data=nxt, compression="gzip", compression_opts=4)

            g.create_dataset("actions", data=np.stack(ep.actions).astype(np.float32))
            g.create_dataset("rewards", data=np.asarray(ep.rewards, dtype=np.float32))
            g.create_dataset("dones", data=np.asarray(ep.dones, dtype=np.int64))
            g.create_dataset("actor", data=np.asarray(ep.actor, dtype=np.uint8))
            g.create_dataset("intervention", data=np.asarray(ep.intervention, dtype=np.uint8))
            if ep.states:
                g.create_dataset("states", data=np.stack(ep.states))

            data.attrs["total"] = int(data.attrs.get("total", 0)) + T
        self._n += 1
        return name

    def write_filter_key(self, key: str, demo_names: Sequence[str]) -> None:
        """robomimic 'mask/<key>' filter keys, e.g. one per DAgger round."""
        with h5py.File(self.path, "a") as f:
            mask = f.require_group("mask")
            if key in mask:
                del mask[key]
            mask.create_dataset(key, data=np.array(
                [n.encode("utf-8") for n in demo_names]))


# ======================================================================== loading

class Episode:
    __slots__ = ("obs", "actions", "actor", "intervention", "rewards",
                 "meta", "length", "name")

    def __init__(self, name: str, obs: Dict[str, np.ndarray], actions: np.ndarray,
                 actor: np.ndarray, intervention: np.ndarray, rewards: np.ndarray,
                 meta: Dict[str, Any]):
        self.name = name
        self.obs = obs
        self.actions = actions
        self.actor = actor
        self.intervention = intervention
        self.rewards = rewards
        self.meta = meta
        self.length = int(actions.shape[0])


def load_episodes(path: str, obs_keys: Iterable[str],
                  demo_names: Optional[Sequence[str]] = None) -> List[Episode]:
    """Read episodes from a robomimic-format HDF5 into memory."""
    obs_keys = list(obs_keys)
    out: List[Episode] = []
    if not os.path.exists(path):
        return out
    with h5py.File(path, "r") as f:
        data = f["data"]
        names = list(demo_names) if demo_names is not None else sorted(
            (k for k in data.keys() if k.startswith("demo_")),
            key=lambda s: int(s.split("_")[1]),
        )
        for name in names:
            if name not in data:
                continue
            g = data[name]
            obs = {k: np.asarray(g["obs"][k]) for k in obs_keys if k in g["obs"]}
            missing = [k for k in obs_keys if k not in obs]
            if missing:
                raise KeyError(f"{path}:{name} lacks observation keys {missing}")
            T = int(g.attrs.get("num_samples", g["actions"].shape[0]))
            actor = (np.asarray(g["actor"]) if "actor" in g
                     else np.ones(T, dtype=np.uint8))
            interv = (np.asarray(g["intervention"]) if "intervention" in g
                      else np.ones(T, dtype=np.uint8))
            out.append(Episode(
                name=name, obs=obs,
                actions=np.asarray(g["actions"], dtype=np.float32),
                actor=actor, intervention=interv,
                rewards=np.asarray(g.get("rewards", np.zeros(T)), dtype=np.float32),
                meta=dict(g.attrs),
            ))
    return out


class SequenceDataset(Dataset):
    """Samples (obs window, action window) pairs from a list of episodes.

    Two alignment modes, because the two policy families need different things and
    getting this wrong is silent -- the loss converges either way:

    ``aligned`` (BC-RNN, robomimic's convention)
        obs[t : t+L] paired index-for-index with actions[t : t+L]. The recurrent
        policy consumes one observation per step and emits the action for *that*
        step, so the two windows must start at the same index.

    ``chunked`` (ACT)
        obs[t-n+1 : t+1] (a history ending at t, n = 1 by default) paired with
        actions[t : t+chunk]. ACT conditions on the current observation and
        predicts a chunk of *future* actions.

    Windows running past the end of an episode repeat the final element, with
    `action_pad_mask` marking the real steps so padding contributes no loss.
    """

    ALIGNED = "aligned"
    CHUNKED = "chunked"

    def __init__(
        self,
        episodes: List[Episode],
        obs_keys: Iterable[str],
        image_keys: Iterable[str] = (),
        obs_seq_len: int = 10,
        action_seq_len: int = 10,
        human_only: bool = True,
        pad_action: bool = True,
        mode: str = "aligned",
    ):
        if mode not in (self.ALIGNED, self.CHUNKED):
            raise ValueError(f"mode must be 'aligned' or 'chunked', got {mode!r}")
        if mode == self.ALIGNED and obs_seq_len != action_seq_len:
            raise ValueError(
                "aligned mode pairs observations and actions index-for-index, so "
                f"obs_seq_len ({obs_seq_len}) must equal action_seq_len "
                f"({action_seq_len})")
        self.mode = mode
        self.episodes = list(episodes)
        self.obs_keys = list(obs_keys)
        self.image_keys = set(image_keys)
        self.obs_seq_len = int(obs_seq_len)
        self.action_seq_len = int(action_seq_len)
        self.human_only = bool(human_only)
        self.pad_action = bool(pad_action)
        self.index: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        self.index = []
        for ei, ep in enumerate(self.episodes):
            for t in range(ep.length):
                # HG-DAgger: only states the human actually labelled are targets.
                if self.human_only and not ep.actor[t]:
                    continue
                if not self.pad_action and t + self.action_seq_len > ep.length:
                    continue
                self.index.append((ei, t))

    def __len__(self) -> int:
        return len(self.index)

    def round_of(self, i: int) -> int:
        ei, _ = self.index[i]
        return int(self.episodes[ei].meta.get("round", 0))

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ei, t = self.index[i]
        ep = self.episodes[ei]

        # --- action window always starts at t, right-padded at episode end ---
        a_idx = np.arange(t, t + self.action_seq_len)
        valid = (a_idx < ep.length).astype(np.float32)
        a_idx = np.clip(a_idx, 0, ep.length - 1)
        actions = torch.from_numpy(np.ascontiguousarray(ep.actions[a_idx]))

        # --- observation window ---
        if self.mode == self.ALIGNED:
            # Same span as the actions: obs[t+k] is the input for actions[t+k].
            o_idx = a_idx
        else:
            # History ending at t, left-padded at the start of the episode.
            o_idx = np.clip(np.arange(t - self.obs_seq_len + 1, t + 1),
                            0, ep.length - 1)
        obs = {}
        for k in self.obs_keys:
            obs[k] = torch.from_numpy(np.ascontiguousarray(ep.obs[k][o_idx]))

        return {
            "obs": obs,
            "actions": actions,
            "action_pad_mask": torch.from_numpy(valid),
            "episode_index": ei,
            "round": self.round_of(i),
        }

    # ------------------------------------------------------------------ sampling
    def recency_sampler(self, new_data_prob: float = 0.5,
                        num_samples: Optional[int] = None) -> WeightedRandomSampler:
        """Oversample the newest round so fresh corrections take effect quickly.

        `new_data_prob` is the total probability mass given to the latest round;
        the remainder is spread uniformly over all earlier data. With a single
        round present this degenerates to uniform sampling.
        """
        rounds = np.array([self.round_of(i) for i in range(len(self))])
        if len(rounds) == 0:
            raise ValueError("cannot build a sampler over an empty dataset")
        newest = rounds.max()
        is_new = rounds == newest
        n_new, n_old = int(is_new.sum()), int((~is_new).sum())
        w = np.empty(len(rounds), dtype=np.float64)
        p = float(np.clip(new_data_prob, 0.0, 1.0))
        if n_old == 0:
            w[:] = 1.0
        else:
            w[is_new] = p / max(n_new, 1)
            w[~is_new] = (1.0 - p) / n_old
        return WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=int(num_samples or len(self)),
            replacement=True,
        )


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs = {k: torch.stack([b["obs"][k] for b in batch]).float()
           for k in batch[0]["obs"]}
    return {
        "obs": obs,
        "actions": torch.stack([b["actions"] for b in batch]).float(),
        "action_pad_mask": torch.stack([b["action_pad_mask"] for b in batch]).float(),
        "round": torch.tensor([b["round"] for b in batch]),
    }


def compute_obs_stats(episodes: List[Episode], low_dim_keys: Iterable[str]
                      ) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-key mean/std over all timesteps, for `ObsNormalizer.fit`."""
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for k in low_dim_keys:
        chunks = [ep.obs[k].reshape(ep.length, -1) for ep in episodes if k in ep.obs]
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0).astype(np.float64)
        stats[k] = {
            "mean": arr.mean(axis=0),
            "std": np.maximum(arr.std(axis=0), 1e-3),
        }
    return stats


def dataset_summary(episodes: List[Episode]) -> Dict[str, Any]:
    if not episodes:
        return {"num_episodes": 0, "num_steps": 0, "num_human_steps": 0}
    total = sum(ep.length for ep in episodes)
    human = sum(int(ep.actor.sum()) for ep in episodes)
    succ = [bool(ep.meta.get("success", False)) for ep in episodes]
    return {
        "num_episodes": len(episodes),
        "num_steps": int(total),
        "num_human_steps": int(human),
        "human_fraction": float(human) / max(total, 1),
        "success_rate": float(np.mean(succ)) if succ else 0.0,
        "rounds": sorted({int(ep.meta.get("round", 0)) for ep in episodes}),
    }
