#!/usr/bin/env python
"""Train a ladder of policies at controlled success rates (0%, 10%, ... 100%).

Produces one checkpoint per target success rate by snapshotting a single training
run frequently and picking the snapshot closest to each target. Taking them all
from one trajectory is deliberate: architecture, observation normalisation and
training data are then identical across the ladder, and *only* competence varies.

Two measurement details matter, because success rate is noisy enough to make a
naive version of this produce a ladder that does not reproduce:

  * **Paired, seeded evaluation.** Every snapshot is scored from the same fixed
    set of initial simulator states with a per-episode torch seed. Without this,
    a policy with unchanged weights scored anywhere from 0.15 to 0.70 on Lift.

  * **Selection and verification use disjoint state sets.** Picking the snapshot
    whose measured score is closest to a target is a selection on a noisy
    quantity, so the selected score is optimistically biased. The reported number
    comes from a second, held-out set of initial states the selection never saw.

    python scripts/train_policy_ladder.py \
        --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
        --out-dir data/checkpoints/ladder --targets 0 10 20 30 40 50 60 70 80 90 100
"""
from __future__ import annotations

import argparse
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
from dagger_lh.data import dataset_summary, load_episodes
from dagger_lh.envs import make_env, silence_robosuite_logs
from dagger_lh.evaluate import evaluate_policy, sample_init_states
from dagger_lh.policies import build_policy, resolve_device
from dagger_lh.trainer import PolicyTrainer


def interpolate_state(lo: Dict[str, Any], hi: Dict[str, Any],
                      alpha: float) -> Dict[str, Any]:
    """Linear blend of two state dicts: (1-alpha)*lo + alpha*hi.

    Valid here because both endpoints come from the same training trajectory --
    checkpoints along one path are typically linearly connected, so the blend
    traces a smooth competence curve rather than a broken model. Non-float
    tensors (and the observation-normaliser buffers, identical across snapshots
    of one run) are taken from `lo` unchanged.
    """
    out = {}
    for k, v in lo.items():
        w = hi.get(k)
        if (w is not None and torch.is_tensor(v) and torch.is_tensor(w)
                and v.dtype.is_floating_point and v.shape == w.shape):
            out[k] = (1.0 - alpha) * v + alpha * w
        else:
            out[k] = v
    return out


def fmt_row(step: int, sel: float, n: int, secs: float) -> str:
    bar = "#" * int(round(sel * 30))
    return (f"  step {step:6d}  sel {sel:5.2f} ({int(round(sel * n)):3d}/{n:3d})  "
            f"{bar:<30}  {secs:5.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="data/checkpoints/ladder")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    help="target success rates in percent")
    ap.add_argument("--algo", default=None, choices=["bc_rnn", "act"])
    ap.add_argument("--horizon", type=int, default=None,
                    help="rollout horizon; defaults to the config's env.horizon")
    ap.add_argument("--max-steps", type=int, default=14000,
                    help="gradient steps before giving up on the remaining targets")
    ap.add_argument("--eval-every", type=int, default=200,
                    help="gradient steps between snapshots")
    ap.add_argument("--select-episodes", type=int, default=30,
                    help="paired rollouts per snapshot during training")
    ap.add_argument("--verify-episodes", type=int, default=100,
                    help="paired rollouts on the held-out states for the final table")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="acceptable |measured - target| before a gap is refined")
    ap.add_argument("--refine-passes", type=int, default=2,
                    help="extra fine-grained passes to fill missed bands")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit-demos", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-low", default=None,
                    help="skip training and build the whole ladder by "
                         "interpolating between this checkpoint and --anchor-high. "
                         "More reliable than snapshot selection when the learning "
                         "curve is unstable -- see the README.")
    ap.add_argument("--anchor-high", default=None)
    ap.add_argument("--interpolate-gaps", action="store_true",
                    help="fill any band the training curve jumped over by "
                         "bisecting a weight interpolation between the "
                         "bracketing snapshots. Gives precise control when the "
                         "learning curve is too steep to sample finely.")
    ap.add_argument("--interp-iters", type=int, default=7,
                    help="bisection steps per interpolated rung")
    ap.add_argument("--keep-pool", action="store_true",
                    help="keep every snapshot, not just the selected ladder")
    args = ap.parse_args()

    silence_robosuite_logs()
    cfg = Config.load(args.config)
    over = {"policy.algo": args.algo, "train.base_lr": args.lr,
            "train.device": args.device, "env.horizon": args.horizon}
    cfg = cfg.override({k: v for k, v in over.items() if v is not None})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = sorted({round(float(t) / 100.0, 4) for t in args.targets})
    os.makedirs(args.out_dir, exist_ok=True)
    pool_dir = os.path.join(args.out_dir, "_pool")
    os.makedirs(pool_dir, exist_ok=True)

    # ---------------- data + policy ----------------
    obs_keys = list(cfg.env.low_dim_keys) + list(cfg.env.image_keys)
    episodes = load_episodes(args.dataset, obs_keys)
    if args.limit_demos:
        episodes = episodes[:args.limit_demos]
    if not episodes:
        print(f"no episodes in {args.dataset}", file=sys.stderr)
        return 1
    print(f"dataset: {json.dumps(dataset_summary(episodes))}")

    obs_shapes = {k: (int(episodes[0].obs[k].reshape(episodes[0].length, -1).shape[1]),)
                  for k in cfg.env.low_dim_keys}
    for k in cfg.env.image_keys:
        obs_shapes[k] = (3, cfg.env.camera_height, cfg.env.camera_width)
    ac_dim = int(episodes[0].actions.shape[-1])

    device = resolve_device(cfg.train.device)
    policy = build_policy(obs_shapes, ac_dim, cfg.policy,
                          cfg.env.low_dim_keys, cfg.env.image_keys, device)
    trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)
    trainer.fit_normalizer(episodes)
    loader = trainer.make_loader(episodes, recency=False)
    if loader is None:
        print("no trainable samples", file=sys.stderr)
        return 1

    env = make_env(cfg.env)
    # Two disjoint sets of starting configurations: one drives selection during
    # training, the other is never seen until the final verification pass.
    sel_states = sample_init_states(env, args.select_episodes, seed=args.seed + 11)
    ver_states = sample_init_states(env, args.verify_episodes, seed=args.seed + 9901)
    print(f"device {device} | horizon {cfg.env.horizon} | "
          f"select on {len(sel_states)} states, verify on {len(ver_states)} disjoint states")
    print(f"targets: {[f'{t:.0%}' for t in targets]}\n")

    snapshots: List[Dict[str, Any]] = []
    t_start = time.time()
    anchor_mode = bool(args.anchor_low and args.anchor_high)

    def snapshot(step: int) -> Dict[str, Any]:
        ev = evaluate_policy(policy, cfg.env, n_episodes=len(sel_states),
                             seed=args.seed + 1, env=env, init_states=sel_states)
        path = os.path.join(pool_dir, f"step_{step:06d}.pth")
        policy.save(path, extra={"grad_steps": step, "select_eval": ev})
        rec = {"step": step, "select": ev["success_rate"], "path": path,
               "mean_steps": ev["mean_steps"]}
        snapshots.append(rec)
        print(fmt_row(step, ev["success_rate"], len(sel_states), time.time() - t_start),
              flush=True)
        return rec

    # ---------------- anchor interpolation (no training) ----------------
    if anchor_mode:
        print("=== anchor interpolation ===")
        print(f"  low  {args.anchor_low}")
        print(f"  high {args.anchor_high}")
        sd_lo = torch.load(args.anchor_low, map_location="cpu",
                           weights_only=False)["state_dict"]
        sd_hi = torch.load(args.anchor_high, map_location="cpu",
                           weights_only=False)["state_dict"]

        def score(state) -> float:
            policy.load_state_dict(state)
            policy.to(device)
            policy.eval()
            return evaluate_policy(policy, cfg.env, n_episodes=len(sel_states),
                                   seed=args.seed + 1, env=env,
                                   init_states=sel_states)["success_rate"]

        for label, state, alpha in (("low", sd_lo, 0.0), ("high", sd_hi, 1.0)):
            sr = score(state)
            path = os.path.join(pool_dir, f"anchor_{label}.pth")
            policy.save(path, extra={"anchor": label, "alpha": alpha,
                                     "select_eval": {"success_rate": sr}})
            snapshots.append({"step": int(alpha * 1000), "select": sr,
                              "path": path, "mean_steps": float("nan"),
                              "interpolated": alpha not in (0.0, 1.0),
                              "alpha": alpha})
            print(f"  anchor {label:4s} (alpha {alpha:.0f}) -> {sr:.2f}")

        for t in targets:
            if abs(min(snapshots, key=lambda s: abs(s['select'] - t))["select"]
                   - t) <= args.tolerance:
                continue
            a_lo, a_hi = 0.0, 1.0
            best = None
            print(f"  {t:.0%}: bisecting")
            for _ in range(args.interp_iters):
                a = 0.5 * (a_lo + a_hi)
                blend = interpolate_state(sd_lo, sd_hi, a)
                sr = score(blend)
                print(f"    alpha {a:.4f} -> {sr:.2f}")
                if best is None or abs(sr - t) < abs(best[1] - t):
                    best = (a, sr, {k: (v.clone() if torch.is_tensor(v) else v)
                                    for k, v in blend.items()})
                if abs(sr - t) <= args.tolerance:
                    break
                if sr < t:
                    a_lo = a
                else:
                    a_hi = a
            a, sr, blend = best
            path = os.path.join(pool_dir, f"interp_{int(round(t * 100)):03d}.pth")
            policy.load_state_dict(blend)
            policy.to(device)
            policy.save(path, extra={"interpolated": True, "alpha": a,
                                     "select_eval": {"success_rate": sr}})
            snapshots.append({"step": int(round(a * 1000)), "select": sr,
                              "path": path, "mean_steps": float("nan"),
                              "interpolated": True, "alpha": a,
                              "between": [0, 1]})
            print(f"    kept alpha {a:.4f} at {sr:.2f}")
        # Skip the sweep and refinement; jump to select + verify.
        args.refine_passes = 0
        args.interpolate_gaps = False

    # ---------------- main sweep ----------------
    if not anchor_mode:
     print("=== sweep ===")
     snapshot(0)                                     # random init -> the 0% rung
     opt = trainer._ensure_optimizer(cfg.train.base_lr)
     it = iter(loader)
     step = 0
     policy.train()
     while step < args.max_steps:
        for _ in range(args.eval_every):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            trainer._step(batch, opt)
            step += 1
        policy.eval()
        rec = snapshot(step)
        policy.train()
        # Stop once the top target is comfortably reached and held.
        if rec["select"] >= max(targets) - 1e-9 and max(targets) >= 1.0:
            if len([s for s in snapshots if s["select"] >= 0.99]) >= 2:
                print("  top of the ladder reached; stopping the sweep")
                break
     policy.eval()

    # ---------------- refinement ----------------
    def best_for(t: float) -> Dict[str, Any]:
        return min(snapshots, key=lambda s: abs(s["select"] - t))

    for p in range(args.refine_passes):
        gaps = [t for t in targets if abs(best_for(t)["select"] - t) > args.tolerance]
        if not gaps:
            break
        print(f"\n=== refinement pass {p + 1}: bands still missed "
              f"{[f'{g:.0%}' for g in gaps]} ===")
        for t in gaps:
            below = [s for s in snapshots if s["select"] <= t]
            above = [s for s in snapshots if s["select"] > t]
            if not below or not above:
                print(f"  {t:.0%}: no bracketing snapshots; skipping")
                continue
            lo = max(below, key=lambda s: (s["select"], -s["step"]))
            hi = min(above, key=lambda s: (s["select"], s["step"]))
            span = max(hi["step"] - lo["step"], 1)
            fine = max(10, span // 6)
            print(f"  {t:.0%}: resuming from step {lo['step']} "
                  f"({lo['select']:.2f}) toward {hi['step']} ({hi['select']:.2f}), "
                  f"every {fine} steps")
            blob = torch.load(lo["path"], map_location="cpu", weights_only=False)
            policy.load_state_dict(blob["state_dict"])
            policy.to(device)
            trainer.reset_optimizer()
            # A smaller learning rate stretches the transition so the bands are
            # resolvable rather than jumped over.
            opt = trainer._ensure_optimizer(cfg.train.base_lr * 0.4)
            it = iter(loader)
            cur = lo["step"]
            policy.train()
            while cur < hi["step"] + span:
                for _ in range(fine):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader)
                        batch = next(it)
                    trainer._step(batch, opt)
                    cur += 1
                policy.eval()
                rec = snapshot(cur)
                policy.train()
                if abs(rec["select"] - t) <= args.tolerance:
                    break
            policy.eval()

    # ---------------- interpolation fill ----------------
    # A steep learning curve can jump a whole band between snapshots, so two
    # targets end up selecting the same snapshot and the ladder silently has
    # fewer levels than it claims. Bisecting a blend of the bracketing snapshots
    # hits the band directly.
    if args.interpolate_gaps:
        gaps = [t for t in targets
                if abs(best_for(t)["select"] - t) > args.tolerance]
        if gaps:
            print(f"\n=== interpolation fill for {[f'{g:.0%}' for g in gaps]} ===")
        for t in gaps:
            below = [s for s in snapshots if s["select"] <= t]
            above = [s for s in snapshots if s["select"] > t]
            if not below or not above:
                print(f"  {t:.0%}: not bracketed by any pair; skipping")
                continue
            lo = max(below, key=lambda s: (s["select"], -s["step"]))
            hi = min(above, key=lambda s: (s["select"], s["step"]))
            sd_lo = torch.load(lo["path"], map_location="cpu",
                               weights_only=False)["state_dict"]
            sd_hi = torch.load(hi["path"], map_location="cpu",
                               weights_only=False)["state_dict"]
            a_lo, a_hi = 0.0, 1.0
            best: Optional[Tuple[float, float, Dict[str, Any]]] = None
            print(f"  {t:.0%}: bisecting between step {lo['step']} "
                  f"({lo['select']:.2f}) and {hi['step']} ({hi['select']:.2f})")
            for i in range(args.interp_iters):
                a = 0.5 * (a_lo + a_hi)
                blend = interpolate_state(sd_lo, sd_hi, a)
                policy.load_state_dict(blend)
                policy.to(device)
                policy.eval()
                ev = evaluate_policy(policy, cfg.env, n_episodes=len(sel_states),
                                     seed=args.seed + 1, env=env,
                                     init_states=sel_states)
                sr = ev["success_rate"]
                print(f"    alpha {a:.4f} -> {sr:.2f}")
                if best is None or abs(sr - t) < abs(best[1] - t):
                    best = (a, sr, {k: v.clone() if torch.is_tensor(v) else v
                                    for k, v in blend.items()})
                if abs(sr - t) <= args.tolerance:
                    break
                if sr < t:
                    a_lo = a
                else:
                    a_hi = a
            if best is not None:
                a, sr, blend = best
                path = os.path.join(pool_dir,
                                    f"interp_{int(round(t * 100)):03d}.pth")
                policy.load_state_dict(blend)
                policy.to(device)
                policy.save(path, extra={"interpolated": True, "alpha": a,
                                         "between": [lo["step"], hi["step"]],
                                         "select_eval": {"success_rate": sr}})
                snapshots.append({
                    "step": -(int(round(t * 100)) + 1),   # negative = synthetic
                    "select": sr, "path": path,
                    "mean_steps": float("nan"), "interpolated": True,
                    "alpha": a, "between": [lo["step"], hi["step"]],
                })
                print(f"    kept alpha {a:.4f} at {sr:.2f}")

    # ---------------- select + verify ----------------
    print(f"\n=== verification on {len(ver_states)} held-out initial states ===")
    chosen: Dict[float, Dict[str, Any]] = {}
    for t in targets:
        chosen[t] = best_for(t)

    manifest = {
        "method": "anchor_interpolation" if anchor_mode else "snapshot_selection",
        "task": cfg.env.env_name, "algo": cfg.policy.algo,
        "dataset": args.dataset, "horizon": cfg.env.horizon,
        "select_episodes": len(sel_states), "verify_episodes": len(ver_states),
        "seed": args.seed, "created": time.time(),
        "note": ("selection and verification use disjoint initial-state sets; "
                 "verified_success is the unbiased number"),
        "rungs": [],
    }
    # Verify each *distinct* snapshot once and reuse the number. Two targets can
    # legitimately land on the same snapshot; measuring it twice would report two
    # different success rates for identical weights (float nondeterminism on
    # accelerators is enough to move a marginal policy), which reads as a
    # contradiction in the manifest.
    verified: Dict[str, Dict[str, Any]] = {}
    print(f"  {'target':>7} {'select':>7} {'verified':>9} {'steps':>7}  file")
    for t in targets:
        rec = chosen[t]
        if rec["path"] not in verified:
            blob = torch.load(rec["path"], map_location="cpu", weights_only=False)
            policy.load_state_dict(blob["state_dict"])
            policy.to(device)
            policy.eval()
            verified[rec["path"]] = evaluate_policy(
                policy, cfg.env, n_episodes=len(ver_states),
                seed=args.seed + 2, env=env, init_states=ver_states)
        ev = verified[rec["path"]]
        blob = torch.load(rec["path"], map_location="cpu", weights_only=False)
        policy.load_state_dict(blob["state_dict"])
        policy.to(device)
        policy.eval()
        name = f"{cfg.policy.algo}_{cfg.env.env_name}_p{int(round(t * 100)):03d}.pth"
        dest = os.path.join(args.out_dir, name)
        policy.save(dest, extra={
            "grad_steps": rec["step"], "target_success": t,
            "select_success": rec["select"], "verified_success": ev["success_rate"],
            "verify_se": ev["success_se"], "dataset": args.dataset,
        })
        manifest["rungs"].append({
            "target": t, "select_success": rec["select"],
            "verified_success": ev["success_rate"], "verified_se": ev["success_se"],
            "mean_steps": ev["mean_steps"], "grad_steps": rec["step"],
            "interpolated": bool(rec.get("interpolated")),
            "alpha": rec.get("alpha"), "between": rec.get("between"),
            "checkpoint": os.path.relpath(dest, args.out_dir),
        })
        flag = "" if abs(ev["success_rate"] - t) <= args.tolerance * 2 else "  <-- off target"
        print(f"  {t:6.0%} {rec['select']:7.2f} {ev['success_rate']:9.2f} "
              f"{rec['step']:7d}  {name}{flag}")

    mpath = os.path.join(args.out_dir, "ladder.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "curve.csv"), "w") as f:
        f.write("grad_steps,select_success,mean_steps\n")
        for s in sorted(snapshots, key=lambda x: x["step"]):
            f.write(f"{s['step']},{s['select']:.6g},{s['mean_steps']:.6g}\n")

    env.close()
    if not args.keep_pool:
        shutil.rmtree(pool_dir, ignore_errors=True)
    else:
        print(f"\nsnapshot pool kept in {pool_dir}")

    worst = max(abs(r["verified_success"] - r["target"]) for r in manifest["rungs"])
    n_distinct = len({chosen[t]["path"] for t in targets})
    n_interp = sum(1 for t in targets if chosen[t].get("interpolated"))
    print(f"\nmanifest -> {mpath}")
    print(f"curve    -> {os.path.join(args.out_dir, 'curve.csv')}")
    print(f"largest deviation from target: {worst:.2f}")
    print(f"distinct policies: {n_distinct} for {len(targets)} targets"
          + (f" ({n_interp} interpolated)" if n_interp else ""))
    if n_distinct < len(targets):
        # Saying "11 policies" when several are byte-identical would be wrong,
        # and any manipulation built on the duplicated rungs would be a no-op.
        dupes = len(targets) - n_distinct
        print(f"WARNING  {dupes} target(s) reused another rung's snapshot, so the "
              f"ladder has {n_distinct} genuinely different policies, not "
              f"{len(targets)}. Re-run with a smaller --eval-every and/or a "
              f"lower --lr so the transition is sampled more finely.")
    print(f"total time: {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
