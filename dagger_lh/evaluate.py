"""Headless policy rollouts for measuring success rate.

Evaluation here is **paired by default**: `sample_init_states` draws a fixed set
of initial simulator states once, and every policy is then evaluated from that
same set. Without this, success rate is dominated by initial object placement --
measured on Lift, a policy whose weights were bitwise identical scored anywhere
from 0.15 to 0.70 across 20-episode runs with different seeds. Paired evaluation
removes that variance from any before/after comparison, which matters because the
whole study rests on detecting within-round improvements.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import EnvConfig
from .envs import RobosuiteEnv, make_env


def sample_init_states(env: RobosuiteEnv, n: int, seed: int = 0) -> List[np.ndarray]:
    """Draw `n` reproducible initial simulator states for paired evaluation."""
    np.random.seed(seed)
    states = []
    for _ in range(int(n)):
        env.reset()
        states.append(env.get_sim_state())
    return states


def rollout(env: RobosuiteEnv, policy, horizon: Optional[int] = None,
            render_frames: Optional[List[np.ndarray]] = None,
            action_noise_std: float = 0.0,
            rng: Optional[np.random.Generator] = None,
            init_state: Optional[np.ndarray] = None,
            torch_seed: Optional[int] = None) -> Dict[str, Any]:
    """Run one closed-loop episode; returns success and length.

    `torch_seed` makes the episode reproducible. Both reference policies are
    stochastic even at evaluation time -- robomimic's `low_noise_eval` still
    *samples* from the GMM (with std ~1e-4) and ACT's VAE decode is seeded -- and
    a policy near its competence boundary flips between success and failure on
    perturbations that small. Seeding keeps the reference behaviour intact while
    making measurements repeatable.
    """
    if torch_seed is not None:
        torch.manual_seed(int(torch_seed))
    obs = env.reset_to_sim_state(init_state) if init_state is not None else env.reset()
    policy.reset_rollout()
    H = int(horizon or env.cfg.horizon)
    rng = rng or np.random.default_rng()
    success = False
    for t in range(H):
        action = np.asarray(policy.act(obs), dtype=np.float64)
        if action_noise_std > 0:
            action = np.clip(action + rng.normal(0, action_noise_std, action.shape), -1, 1)
        obs, _, done, info = env.step(action)
        if render_frames is not None:
            render_frames.append(env.render_operator(height=256, width=256))
        if info.get("success"):
            success = True
            break
        if done:
            break
    return {"success": success, "steps": t + 1}


def evaluate_policy(policy, env_cfg: EnvConfig, n_episodes: int = 20,
                    horizon: Optional[int] = None, seed: int = 0,
                    env: Optional[RobosuiteEnv] = None,
                    progress: bool = False,
                    init_states: Optional[List[np.ndarray]] = None,
                    action_noise_std: float = 0.0) -> Dict[str, Any]:
    """Success rate over `n_episodes` rollouts.

    Pass `init_states` (from `sample_init_states`) to evaluate every policy from
    the same starting configurations; strongly preferred for any comparison.
    """
    own_env = env is None
    if own_env:
        cfg = EnvConfig(**{**env_cfg.__dict__})
        cfg.seed = seed
        env = make_env(cfg)
    try:
        np.random.seed(seed)
        if init_states is not None:
            n_episodes = min(int(n_episodes), len(init_states))
        results = []
        for i in range(int(n_episodes)):
            r = rollout(env, policy, horizon=horizon,
                        init_state=init_states[i] if init_states else None,
                        torch_seed=seed * 100_003 + i,
                        action_noise_std=action_noise_std,
                        rng=np.random.default_rng(seed * 7919 + i))
            results.append(r)
            if progress:
                sr = np.mean([x["success"] for x in results])
                print(f"    eval {i + 1}/{n_episodes}  running success {sr:.2f}",
                      flush=True)
        succ = [r["success"] for r in results]
        lens = [r["steps"] for r in results]
        p = float(np.mean(succ))
        n = max(len(results), 1)
        return {
            "n_episodes": len(results),
            "success_rate": p,
            # Binomial standard error, so a reported delta can be judged.
            "success_se": float(np.sqrt(max(p * (1 - p), 0.0) / n)),
            "successes": int(np.sum(succ)),
            "paired": init_states is not None,
            "action_noise_std": float(action_noise_std),
            "mean_steps": float(np.mean(lens)),
            "mean_steps_success": float(
                np.mean([l for l, s in zip(lens, succ) if s])) if any(succ) else float("nan"),
        }
    finally:
        if own_env:
            env.close()
