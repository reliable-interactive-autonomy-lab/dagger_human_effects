#!/usr/bin/env python
"""Build a policy ladder by training on demonstrations corrupted to varying degrees.

Why this method rather than picking early checkpoints
-----------------------------------------------------
Two more obvious approaches were tried on Lift and both failed:

* **Snapshot selection** (`scripts/train_policy_ladder.py`). BC-RNN's success rate
  is not a stable function of training step -- measured on a *fixed* set of
  initial states it swung 0.90 -> 0.37 -> 0.57 -> 0.47 over consecutive
  200-step intervals, far beyond sampling error. Picking the snapshot that
  happens to measure 20% therefore selects a fluke: three different targets
  (10/20/30%) all verified at 0.34-0.38 on held-out states, and two of them were
  the same snapshot.

* **Weight interpolation** between the random init and the trained policy. The
  curve is a step, not a ramp: success stayed at 0.00 from alpha 0.0 to 0.8, hit
  0.05 at 0.9 and 0.95 at 1.0. Nothing to bisect.

Corruption level is a better control variable because it is a property of the
*training data*, not of where training happened to stop. Each rung is trained to
convergence, so it is stably mediocre rather than transiently bad, and the
mapping from noise to competence is smooth and monotone enough to bisect.

    python scripts/build_ladder_noise.py \
        --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
        --out-dir data/checkpoints/ladder \
        --targets 0 10 20 30 40 50 60 70 80 90 100
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dagger_lh.config import Config
from dagger_lh.data import Episode, dataset_summary, load_episodes
from dagger_lh.envs import make_env, silence_robosuite_logs
from dagger_lh.evaluate import evaluate_policy, sample_init_states
from dagger_lh.policies import build_policy, resolve_device
from dagger_lh.policies import load_policy as _load_policy
from dagger_lh.trainer import PolicyTrainer


def corrupt(episodes: List[Episode], sigma: float, seed: int,
            kind: str = "action", heavy_sigma: float = 0.35) -> List[Episode]:
    """Copy `episodes` with Gaussian noise added to the demonstrated actions.

    Degrading the *actions* (rather than dropping demos or adding observation
    noise) produces a policy that is imprecise in the way a mediocre teleoperator
    is imprecise: it reaches for the object but grasps sloppily. That is the right
    kind of failure for a study where participants judge whether the robot is
    getting better -- a policy that fails randomly reads as broken instead.
    """
    if sigma <= 0:
        return episodes          # includes the -1.0 sentinel: train on clean data
    rng = np.random.default_rng(seed)

    if kind == "mix":
        # `sigma` is instead the *fraction* of demonstrations replaced by heavily
        # corrupted ones. A mixture of clean and bad demonstrations tends to
        # degrade competence more gradually than adding the same noise to every
        # demonstration, which on this task turns out to be a cliff: the policy
        # learns a blend of good and bad behaviour rather than a uniformly
        # imprecise one.
        frac = float(np.clip(sigma, 0.0, 1.0))
        n_bad = int(round(frac * len(episodes)))
        idx = set(rng.permutation(len(episodes))[:n_bad].tolist())
        out = []
        for i, ep in enumerate(episodes):
            if i not in idx:
                out.append(ep)
                continue
            acts = ep.actions.astype(np.float64).copy()
            noise = rng.normal(0.0, heavy_sigma, acts.shape)
            noise[:, 6:] = 0.0
            acts = np.clip(acts + noise, -1.0, 1.0).astype(np.float32)
            out.append(Episode(ep.name, ep.obs, acts, ep.actor, ep.intervention,
                               ep.rewards, dict(ep.meta)))
        return out

    out: List[Episode] = []
    for ep in episodes:
        acts = ep.actions.astype(np.float64).copy()
        noise = rng.normal(0.0, sigma, acts.shape)
        if kind == "action_pose":
            # Leave the gripper channel alone: noising it makes the demos
            # incoherent (open/close flapping) rather than merely imprecise.
            noise[:, 6:] = 0.0
        acts = np.clip(acts + noise, -1.0, 1.0).astype(np.float32)
        obs = ep.obs
        if kind == "obs":
            obs = {k: (v + rng.normal(0, sigma, v.shape)).astype(v.dtype)
                   if v.dtype.kind == "f" else v for k, v in ep.obs.items()}
            acts = ep.actions
        out.append(Episode(ep.name, obs, acts, ep.actor, ep.intervention,
                           ep.rewards, dict(ep.meta)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="data/checkpoints/ladder")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument("--noise-levels", type=float, nargs="+", default=None,
                    help="corruption sigmas to sweep (default: a spread from 0 "
                         "to 1.0 chosen to bracket the whole range)")
    ap.add_argument("--noise-kind", default="action_pose",
                    choices=["action", "action_pose", "obs", "mix"],
                    help="'action_pose' adds Gaussian noise to the demonstrated "
                         "pose deltas; 'mix' instead treats the level as the "
                         "FRACTION of demonstrations replaced by heavily "
                         "corrupted ones, which degrades more gradually")
    ap.add_argument("--heavy-sigma", type=float, default=0.35,
                    help="noise applied to the corrupted subset when "
                         "--noise-kind mix")
    ap.add_argument("--train-steps", type=int, default=4000,
                    help="gradient steps per noise level; enough to converge")
    ap.add_argument("--algo", default=None, choices=["bc_rnn", "act"])
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--select-episodes", type=int, default=40)
    ap.add_argument("--verify-episodes", type=int, default=100)
    ap.add_argument("--tolerance", type=float, default=0.06)
    ap.add_argument("--bisect-iters", type=int, default=4,
                    help="extra sigmas tried per missed band (0 to disable)")
    ap.add_argument("--stability-check", action="store_true",
                    help="re-measure each rung to confirm it is stably mediocre "
                         "rather than transiently bad")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-candidates", nargs="*", default=[],
                    help="existing checkpoints to consider alongside the swept "
                         "levels. Useful for the top of the ladder: a clean-data "
                         "policy trained for longer than --train-steps reaches "
                         "success rates the sweep itself will not.")
    ap.add_argument("--extra-train-steps", type=int, default=None,
                    help="additional gradient steps to train the sigma=0 policy "
                         "for, as a separate candidate, to reach the top rungs")
    ap.add_argument("--keep-pool", action="store_true")
    args = ap.parse_args()

    silence_robosuite_logs()
    cfg = Config.load(args.config)
    over = {"policy.algo": args.algo, "train.base_lr": args.lr,
            "train.device": args.device, "env.horizon": args.horizon}
    cfg = cfg.override({k: v for k, v in over.items() if v is not None})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = sorted({round(float(t) / 100.0, 4) for t in args.targets})
    levels = (list(args.noise_levels) if args.noise_levels
              else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75, 1.00])
    os.makedirs(args.out_dir, exist_ok=True)
    pool_dir = os.path.join(args.out_dir, "_pool")
    os.makedirs(pool_dir, exist_ok=True)

    obs_keys = list(cfg.env.low_dim_keys) + list(cfg.env.image_keys)
    clean = load_episodes(args.dataset, obs_keys)
    if not clean:
        print(f"no episodes in {args.dataset}", file=sys.stderr)
        return 1
    print(f"dataset: {json.dumps(dataset_summary(clean))}")

    obs_shapes = {k: (int(clean[0].obs[k].reshape(clean[0].length, -1).shape[1]),)
                  for k in cfg.env.low_dim_keys}
    for k in cfg.env.image_keys:
        obs_shapes[k] = (3, cfg.env.camera_height, cfg.env.camera_width)
    ac_dim = int(clean[0].actions.shape[-1])

    device = resolve_device(cfg.train.device)
    env = make_env(cfg.env)
    sel_states = sample_init_states(env, args.select_episodes, seed=args.seed + 11)
    ver_states = sample_init_states(env, args.verify_episodes, seed=args.seed + 9901)
    print(f"device {device} | horizon {cfg.env.horizon} | "
          f"{args.train_steps} steps per level")
    print(f"select on {len(sel_states)} states, verify on {len(ver_states)} disjoint")
    print(f"noise kind '{args.noise_kind}', levels {levels}")
    print(f"targets: {[f'{t:.0%}' for t in targets]}\n")

    t0 = time.time()
    rungs: List[Dict[str, Any]] = []

    def train_at(sigma: float) -> Dict[str, Any]:
        """Train a fresh policy on data corrupted at `sigma` and score it."""
        torch.manual_seed(args.seed + 1)
        policy = build_policy(obs_shapes, ac_dim, cfg.policy,
                              cfg.env.low_dim_keys, cfg.env.image_keys, device)
        trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)
        eps = corrupt(clean, sigma, seed=args.seed + 7, kind=args.noise_kind,
                      heavy_sigma=args.heavy_sigma)
        # Normalisation comes from the corrupted data the policy actually sees.
        trainer.fit_normalizer(eps)
        loader = trainer.make_loader(eps, recency=False)
        opt = trainer._ensure_optimizer(cfg.train.base_lr)
        it = iter(loader)
        policy.train()
        for _ in range(args.train_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            trainer._step(batch, opt)
        policy.eval()
        ev = evaluate_policy(policy, cfg.env, n_episodes=len(sel_states),
                             seed=args.seed + 1, env=env, init_states=sel_states)
        path = os.path.join(pool_dir, f"sigma_{sigma:.4f}.pth")
        policy.save(path, extra={"noise_sigma": sigma,
                                 "noise_kind": args.noise_kind,
                                 "train_steps": args.train_steps,
                                 "select_eval": ev})
        rec = {"sigma": sigma, "select": ev["success_rate"], "path": path,
               "mean_steps": ev["mean_steps"]}
        bar = "#" * int(round(rec["select"] * 30))
        print(f"  sigma {sigma:5.3f} -> {rec['select']:5.2f}  {bar:<30} "
              f"{time.time() - t0:5.0f}s", flush=True)
        rungs.append(rec)
        del policy, trainer
        return rec

    print("=== noise sweep ===")
    for sigma in levels:
        train_at(sigma)

    # The top of the ladder often needs more training than the sweep budget:
    # clean data at --train-steps may still be short of its ceiling, leaving the
    # 90-100% bands unreachable no matter how little noise is added.
    if args.extra_train_steps:
        print(f"\n=== extra clean-data run ({args.extra_train_steps} steps) ===")
        saved = args.train_steps
        args.train_steps = int(args.extra_train_steps)
        rec = train_at(-1.0)          # sentinel sigma: clean data, longer training
        rec["sigma"] = 0.0
        rec["long_run"] = True
        args.train_steps = saved

    for path in args.extra_candidates:
        if not os.path.exists(path):
            print(f"  extra candidate missing, skipping: {path}")
            continue
        pol, _ = _load_policy(path, device, cfg.policy)
        pol.eval()
        ev = evaluate_policy(pol, cfg.env, n_episodes=len(sel_states),
                             seed=args.seed + 1, env=env, init_states=sel_states)
        rungs.append({"sigma": float("nan"), "select": ev["success_rate"],
                      "path": path, "mean_steps": ev["mean_steps"],
                      "external": True})
        print(f"  extra {os.path.basename(path):34s} -> {ev['success_rate']:5.2f}")
        del pol

    # ---------------- fill missed bands by bisecting sigma ----------------
    def best_for(t: float) -> Dict[str, Any]:
        return min(rungs, key=lambda r: abs(r["select"] - t))

    if args.bisect_iters:
        for _ in range(args.bisect_iters):
            gaps = [t for t in targets
                    if abs(best_for(t)["select"] - t) > args.tolerance]
            if not gaps:
                break
            # One new sigma per pass, for the worst-served band: each costs a
            # full training run, so spend them where they help most.
            t = max(gaps, key=lambda t: abs(best_for(t)["select"] - t))
            above = [r for r in rungs if r["select"] > t]   # less noise
            below = [r for r in rungs if r["select"] <= t]  # more noise
            if not above or not below:
                print(f"  {t:.0%}: not bracketed by any sigma pair; skipping")
                break
            lo = min(above, key=lambda r: r["select"])   # smallest sigma above t
            hi = max(below, key=lambda r: r["select"])   # largest sigma below t
            new_sigma = 0.5 * (lo["sigma"] + hi["sigma"])
            if any(abs(r["sigma"] - new_sigma) < 1e-4 for r in rungs):
                print(f"  {t:.0%}: sigma already sampled at {new_sigma:.4f}; stopping")
                break
            print(f"\n=== filling {t:.0%}: sigma between {hi['sigma']:.3f} "
                  f"({hi['select']:.2f}) and {lo['sigma']:.3f} ({lo['select']:.2f}) ===")
            train_at(new_sigma)

    # ---------------- select, verify, publish ----------------
    print(f"\n=== verification on {len(ver_states)} held-out initial states ===")
    from dagger_lh.policies import load_policy

    verified: Dict[str, Dict[str, Any]] = {}
    manifest = {
        "method": "noise_sweep", "noise_kind": args.noise_kind,
        "heavy_sigma": args.heavy_sigma,
        "task": cfg.env.env_name, "algo": cfg.policy.algo,
        "dataset": args.dataset, "horizon": cfg.env.horizon,
        "train_steps_per_level": args.train_steps,
        "select_episodes": len(sel_states), "verify_episodes": len(ver_states),
        "seed": args.seed, "created": time.time(),
        "note": ("each rung is trained to convergence on demonstrations corrupted "
                 "at its own sigma; selection and verification use disjoint "
                 "initial-state sets, so verified_success is the unbiased number"),
        "rungs": [],
    }
    print(f"  {'target':>7} {'sigma':>7} {'select':>7} {'verified':>9}  file")
    for t in targets:
        rec = best_for(t)
        if rec["path"] not in verified:
            pol, _ = load_policy(rec["path"], device, cfg.policy)
            pol.eval()
            verified[rec["path"]] = evaluate_policy(
                pol, cfg.env, n_episodes=len(ver_states), seed=args.seed + 2,
                env=env, init_states=ver_states)
            if args.stability_check:
                again = evaluate_policy(
                    pol, cfg.env, n_episodes=len(ver_states), seed=args.seed + 3,
                    env=env, init_states=ver_states)
                verified[rec["path"]]["stability_delta"] = abs(
                    again["success_rate"] - verified[rec["path"]]["success_rate"])
        ev = verified[rec["path"]]
        name = f"{cfg.policy.algo}_{cfg.env.env_name}_p{int(round(t * 100)):03d}.pth"
        dest = os.path.join(args.out_dir, name)
        shutil.copyfile(rec["path"], dest)
        blob = torch.load(dest, map_location="cpu", weights_only=False)
        blob["extra"].update({"target_success": t,
                              "verified_success": ev["success_rate"],
                              "verify_se": ev["success_se"]})
        torch.save(blob, dest)
        manifest["rungs"].append({
            "target": t, "noise_sigma": rec["sigma"],
            "select_success": rec["select"],
            "verified_success": ev["success_rate"],
            "verified_se": ev["success_se"],
            "stability_delta": ev.get("stability_delta"),
            "source": ("external" if rec.get("external")
                       else ("clean_long_run" if rec.get("long_run") else "sweep")),
            "mean_steps": ev["mean_steps"], "grad_steps": args.train_steps,
            "checkpoint": name,
        })
        flag = "" if abs(ev["success_rate"] - t) <= args.tolerance * 2 else "  <-- off target"
        print(f"  {t:6.0%} {rec['sigma']:7.3f} {rec['select']:7.2f} "
              f"{ev['success_rate']:9.2f}  {name}{flag}")

    with open(os.path.join(args.out_dir, "ladder.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "curve.csv"), "w") as f:
        f.write("noise_sigma,select_success,mean_steps\n")
        for r in sorted(rungs, key=lambda x: x["sigma"]):
            f.write(f"{r['sigma']:.6g},{r['select']:.6g},{r['mean_steps']:.6g}\n")

    env.close()
    if not args.keep_pool:
        shutil.rmtree(pool_dir, ignore_errors=True)

    devs = [abs(r["verified_success"] - r["target"]) for r in manifest["rungs"]]
    n_distinct = len({r["noise_sigma"] for r in manifest["rungs"]})
    print(f"\nmanifest -> {os.path.join(args.out_dir, 'ladder.json')}")
    print(f"largest deviation from target: {max(devs):.2f}  mean {np.mean(devs):.3f}")
    print(f"distinct policies: {n_distinct} for {len(targets)} targets")
    if n_distinct < len(targets):
        print(f"WARNING  {len(targets) - n_distinct} target(s) reuse another "
              f"rung's policy. Add intermediate --noise-levels around the "
              f"crowded bands and re-run.")
    print(f"total time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
