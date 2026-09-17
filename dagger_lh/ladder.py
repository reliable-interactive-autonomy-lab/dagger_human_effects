"""A ladder of policy checkpoints at known success rates.

Built by `scripts/train_policy_ladder.py`, which snapshots one training run and
verifies each rung on a held-out set of initial states. Two uses:

1. **Choosing a starting competence.** Pick the rung whose success rate leaves the
   right amount of headroom for a study (see the base-policy discussion in the
   README -- an expert policy makes even the contingent condition non-contingent).

2. **Driving a scripted success trajectory.** The `yoked` condition needs the
   policy's competence to follow a predetermined curve that is independent of
   what the operator does. Swapping in a rung per round is what makes that
   literally true; before this existed, `yoked` behaved identically to
   `noncontingent`.

A rung is a checkpoint plus an optional `action_noise_std`. The working method on
Lift (`scripts/calibrate_ladder.py`) produces rungs that all share one checkpoint
and differ only in calibrated rollout noise, so competence is the only thing that
varies across the ladder; `scripts/build_ladder_noise.py` and
`scripts/train_policy_ladder.py` instead produce a distinct checkpoint per rung.
Both shapes load through the same interface.

Ladder checkpoints are saved *before* LoRA injection, so their state dicts do not
match a policy that has adapters attached. `load_rung` therefore builds a
separate, adapter-free policy for rollouts. That is the right structure anyway:
under a scripted schedule the trainable policy is still updated and rolled back so
the wall-clock cost is unchanged, while the behaviour the operator sees comes from
the ladder.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch


class PolicyLadder:
    def __init__(self, manifest_path: str):
        self.path = manifest_path
        with open(manifest_path) as f:
            self.manifest: Dict[str, Any] = json.load(f)
        self.root = os.path.dirname(os.path.abspath(manifest_path))
        rungs = self.manifest.get("rungs") or []
        if not rungs:
            raise ValueError(f"{manifest_path} lists no rungs")
        self.rungs: List[Dict[str, Any]] = sorted(
            rungs, key=lambda r: float(r["verified_success"]))
        self._cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ lookup
    @property
    def task(self) -> str:
        return str(self.manifest.get("task", ""))

    @property
    def algo(self) -> str:
        return str(self.manifest.get("algo", ""))

    def available(self) -> List[float]:
        return [float(r["verified_success"]) for r in self.rungs]

    def nearest(self, target: float) -> Dict[str, Any]:
        """The rung whose *verified* success rate is closest to `target`."""
        return min(self.rungs,
                   key=lambda r: abs(float(r["verified_success"]) - float(target)))

    def checkpoint_for(self, target: float) -> str:
        return os.path.join(self.root, self.nearest(target)["checkpoint"])

    def describe(self) -> str:
        pairs = ", ".join(f"{r['verified_success']:.0%}" for r in self.rungs)
        method = self.manifest.get("method", "?")
        return (f"ladder '{os.path.basename(self.path)}': {self.algo} on "
                f"{self.task}, {len(self.rungs)} rungs [{pairs}] via {method}")

    def noise_for(self, target: float) -> float:
        """Rollout action noise the nearest rung needs to hit `target`."""
        return float(self.nearest(target).get("action_noise_std", 0.0) or 0.0)

    # ------------------------------------------------------------------ loading
    def load_rung(self, target: float, device, policy_cfg,
                  env_obs_shapes: Optional[Dict[str, Any]] = None,
                  low_dim_keys: Optional[Sequence[str]] = None,
                  image_keys: Optional[Sequence[str]] = None):
        """Load (and cache) the adapter-free policy nearest to `target`."""
        from .policies import load_policy

        rung = self.nearest(target)
        path = os.path.join(self.root, rung["checkpoint"])
        if path in self._cache:
            return self._cache[path], rung
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ladder rung missing: {path} (referenced by {self.path})")
        policy, _ = load_policy(path, device, policy_cfg)
        policy.eval()
        self._cache[path] = policy
        return policy, rung

    def validate_against(self, obs_shapes: Dict[str, Any], ac_dim: int,
                         algo: str) -> None:
        """Fail early if the ladder does not match the live environment.

        A ladder trained on a different task or observation set would load and
        then behave nonsensically, which is much harder to notice mid-study than
        an error at startup.
        """
        if self.algo and algo and self.algo != algo:
            raise ValueError(
                f"ladder was trained with algo '{self.algo}' but this session "
                f"uses '{algo}'")
        for rung in self.rungs[:1]:
            path = os.path.join(self.root, rung["checkpoint"])
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if int(blob["ac_dim"]) != int(ac_dim):
                raise ValueError(
                    f"ladder action dim {blob['ac_dim']} != env action dim {ac_dim}")
            got = {k: tuple(v) for k, v in blob["obs_shapes"].items()}
            want = {k: tuple(v) for k, v in obs_shapes.items()}
            if got != want:
                raise ValueError(
                    f"ladder observation shapes {got} != env {want}")


def resolve_schedule(schedule: Sequence[float], n_rounds: int) -> List[float]:
    """Stretch or trim a success schedule to exactly `n_rounds` entries.

    A schedule shorter than the session repeats its last value, so a partially
    specified curve (e.g. `[0.1, 0.1, 0.1]`) holds flat rather than silently
    wrapping around to the start.
    """
    vals = [float(v) for v in schedule]
    if not vals:
        raise ValueError("empty schedule")
    if len(vals) >= n_rounds:
        return vals[:n_rounds]
    return vals + [vals[-1]] * (n_rounds - len(vals))
