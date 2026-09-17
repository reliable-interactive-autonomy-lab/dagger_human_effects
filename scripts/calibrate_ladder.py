#!/usr/bin/env python
"""Build a policy ladder by calibrating rollout-time action noise.

Each rung is one base policy plus a calibrated noise level, found by bisection,
that makes it succeed at the target rate. Because every rung shares the same
weights, competence is the *only* thing that differs across the ladder.

Why this and not the other three methods
----------------------------------------
All were built and measured on Lift. Only this one produced a usable ladder.

* **Earlier checkpoints of one run** — success rate is not a stable function of
  training step: consecutive 200-step snapshots scored 0.90 → 0.37 → 0.57 → 0.47
  on a *fixed* evaluation set. Targets of 10/20/30% all verified at 0.34-0.38.
* **Weight interpolation** init → trained — a step, not a ramp: 0.00 until
  alpha 0.8, then 0.05 at 0.9 and 0.95 at 1.0.
* **Training on corrupted demonstrations** — non-monotone. Action noise on demos
  acts as regularisation at moderate levels (sigma 0.09 → 0.25 but sigma 0.20 →
  0.57), and even replacing *every* demonstration with a heavily corrupted one
  still left the policy at 0.40. Lift tolerates sloppy demonstrations too well to
  serve as a competence knob.

Rollout noise is monotone by construction over its useful range and needs no
retraining, so bisection converges reliably:

    noise  0.00  0.05  0.10  0.15  0.20  0.25  0.30
    succ   1.00  0.93  0.70  0.40  0.17  0.10  0.00

    python scripts/calibrate_ladder.py \
        --checkpoint data/checkpoints/bc_rnn_Lift_base.pth \
        --out-dir data/checkpoints/ladder \
        --targets 0 10 20 30 40 50 60 70 80 90 100
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
from dagger_lh.envs import make_env, silence_robosuite_logs
from dagger_lh.evaluate import evaluate_policy, sample_init_states
from dagger_lh.policies import load_policy, resolve_device


def isotonic_decreasing(xs: List[float], ys: List[float]) -> List[float]:
    """Pool-adjacent-violators fit of a non-increasing curve (no extra deps).

    Averages out the measurement noise in individual evaluations instead of
    trusting whichever one happened to land nearest a target.
    """
    order = np.argsort(xs)
    y = [float(ys[i]) for i in order]
    w = [1.0] * len(y)
    # Merge adjacent blocks that violate monotonicity, replacing them by their
    # weighted mean, until the sequence is non-increasing.
    i = 0
    while i < len(y) - 1:
        if y[i] < y[i + 1] - 1e-12:
            tot = w[i] + w[i + 1]
            y[i] = (y[i] * w[i] + y[i + 1] * w[i + 1]) / tot
            w[i] = tot
            del y[i + 1], w[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out, k = [], 0
    for val, weight in zip(y, w):
        out.extend([val] * int(round(weight)))
    return out


def invert_curve(points: Dict[float, float], target: float,
                 noise_max: float) -> float:
    """Noise level whose fitted success rate equals `target`.

    Fits a monotone curve through every measurement collected so far and reads
    off the crossing, rather than returning the single measured point closest to
    the target -- selecting on a noisy measurement is biased, pooling is not.
    """
    xs = sorted(points)
    ys_raw = [points[x] for x in xs]
    ys = isotonic_decreasing(xs, ys_raw)
    if len(ys) != len(xs):          # defensive: PAVA must preserve length
        ys = ys_raw
    if target >= ys[0]:
        return float(xs[0])
    if target <= ys[-1]:
        return float(xs[-1])
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= target >= y1:
            if abs(y0 - y1) < 1e-9:
                return float(0.5 * (x0 + x1))
            f = (y0 - target) / (y0 - y1)
            return float(x0 + f * (x1 - x0))
    return float(min(xs, key=lambda x: abs(points[x] - target)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="a competent base policy; its clean success rate is the "
                         "ceiling of the ladder")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="data/checkpoints/ladder")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--noise-max", type=float, default=0.45,
                    help="upper end of the bisection bracket; must drive the "
                         "policy to its floor")
    ap.add_argument("--select-episodes", type=int, default=30)
    ap.add_argument("--verify-episodes", type=int, default=60)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--no-fit-curve", dest="fit_curve", action="store_false",
                    help="pick each rung's noise from the single closest "
                         "measurement instead of a monotone fit through all of "
                         "them. The fit is on by default because selecting on a "
                         "noisy measurement biases the rung.")
    ap.add_argument("--refine-from", default=None,
                    help="an existing ladder.json to improve rather than rebuild. "
                         "Re-bisects only --refine-targets, inside the bracket "
                         "implied by the neighbouring rungs, so a few off-target "
                         "bands can be fixed without redoing the whole ladder.")
    ap.add_argument("--refine-targets", type=float, nargs="*", default=None,
                    help="targets (percent) to re-bisect; default: every rung "
                         "further than --tolerance from its target")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    silence_robosuite_logs()
    cfg = Config.load(args.config).override({"env.horizon": args.horizon})
    if args.device:
        cfg = cfg.override({"train.device": args.device})
    device = resolve_device(cfg.train.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = sorted({round(float(t) / 100.0, 4) for t in args.targets})
    os.makedirs(args.out_dir, exist_ok=True)

    policy, extra = load_policy(args.checkpoint, device, cfg.policy)
    policy.eval()
    policy.set_action_noise(0.0)   # calibration passes noise in explicitly
    env = make_env(cfg.env)
    sel = sample_init_states(env, args.select_episodes, seed=args.seed + 11)
    ver = sample_init_states(env, args.verify_episodes, seed=args.seed + 9901)
    print(f"base policy {args.checkpoint}")
    print(f"device {device} | horizon {cfg.env.horizon}")
    print(f"select on {len(sel)} states, verify on {len(ver)} disjoint states")
    print(f"targets: {[f'{t:.0%}' for t in targets]}\n")

    t0 = time.time()
    cache: Dict[float, float] = {}

    def score(noise: float) -> float:
        key = round(noise, 5)
        if key not in cache:
            ev = evaluate_policy(policy, cfg.env, n_episodes=len(sel),
                                 seed=args.seed + 1, env=env, init_states=sel,
                                 action_noise_std=key)
            cache[key] = ev["success_rate"]
        return cache[key]

    ceiling = score(0.0)
    floor = score(args.noise_max)
    print(f"bracket: noise 0.000 -> {ceiling:.2f} | "
          f"noise {args.noise_max:.3f} -> {floor:.2f}")
    if ceiling < max(targets) - args.tolerance:
        print(f"WARNING  the base policy tops out at {ceiling:.2f}; targets above "
              f"that cannot be reached. Train a stronger base policy first.",
              file=sys.stderr)
    if floor > min(targets) + args.tolerance:
        print(f"WARNING  noise {args.noise_max} still leaves {floor:.2f}; raise "
              f"--noise-max.", file=sys.stderr)

    # When refining, start from the existing rungs and only redo the bad ones.
    prior: Dict[float, Dict[str, Any]] = {}
    to_do = list(targets)
    if args.refine_from:
        with open(args.refine_from) as f:
            prior_manifest = json.load(f)
        for r in prior_manifest.get("rungs", []):
            prior[round(float(r["target"]), 4)] = r
        if args.refine_targets is not None:
            to_do = sorted({round(float(t) / 100.0, 4) for t in args.refine_targets})
        else:
            to_do = [t for t in targets
                     if t in prior
                     and abs(prior[t]["verified_success"] - t) > args.tolerance]
        print(f"refining {[f'{t:.0%}' for t in to_do]} "
              f"(keeping {len(prior) - len(to_do)} existing rungs)")

    rungs: List[Dict[str, Any]] = []
    print(f"\n{'target':>7}  bisection")
    for t in targets:
        if t not in to_do and t in prior:
            rungs.append({"target": t,
                          "action_noise_std": prior[t]["action_noise_std"],
                          "select_success": prior[t]["select_success"]})
            print(f"  {t:5.0%}  (kept: noise {prior[t]['action_noise_std']:.4f})")
            continue
        # Success falls monotonically with noise, so the bracket is inverted:
        # lo_noise is the *high*-success end. When refining, narrow the bracket
        # to the neighbouring rungs' noise levels -- monotonicity guarantees the
        # answer lies between them, and a tight bracket converges in a few
        # iterations instead of eight.
        lo, hi = 0.0, float(args.noise_max)
        if prior:
            higher = [p["action_noise_std"] for tt, p in prior.items() if tt > t]
            lower = [p["action_noise_std"] for tt, p in prior.items() if tt < t]
            if higher:
                lo = max(0.0, min(higher) * 0.9)
            if lower:
                hi = min(float(args.noise_max), max(lower) * 1.1)
            if hi <= lo:
                lo, hi = 0.0, float(args.noise_max)
        best: Optional[Tuple[float, float]] = None
        trace = []
        for _ in range(args.iters):
            for cand in (lo, hi):
                sr = score(cand)
                if best is None or abs(sr - t) < abs(best[1] - t):
                    best = (cand, sr)
            mid = 0.5 * (lo + hi)
            sr = score(mid)
            trace.append(f"{mid:.3f}->{sr:.2f}")
            if best is None or abs(sr - t) < abs(best[1] - t):
                best = (mid, sr)
            if abs(sr - t) <= args.tolerance:
                break
            if sr > t:          # too good, add noise
                lo = mid
            else:               # too bad, remove noise
                hi = mid
        noise, sel_sr = best
        rungs.append({"target": t, "action_noise_std": round(noise, 5),
                      "select_success": sel_sr})
        print(f"  {t:5.0%}  {' '.join(trace[:6])}  => noise {noise:.4f} "
              f"({sel_sr:.2f})")

    if args.fit_curve and len(cache) >= 4:
        # Re-derive every noise level from a monotone fit through all the
        # measurements taken during bisection. Each individual evaluation is
        # noisy; the fit uses all of them, so it is a better estimate of the
        # noise level that actually produces each target than the single
        # closest-measuring point.
        print(f"\n=== monotone fit over {len(cache)} measurements ===")
        for r in rungs:
            fitted = invert_curve(cache, r["target"], args.noise_max)
            if abs(fitted - r["action_noise_std"]) > 1e-4:
                print(f"  {r['target']:5.0%}  noise {r['action_noise_std']:.4f} "
                      f"-> {fitted:.4f} (from fit)")
            r["action_noise_std"] = round(fitted, 5)

    # ---------------- verification on held-out states ----------------
    print(f"\n=== verification on {len(ver)} held-out initial states ===")
    print(f"  {'target':>7} {'noise':>7} {'select':>7} {'verified':>9} {'+/-':>6}")
    base_name = f"{cfg.policy.algo}_{cfg.env.env_name}_ladder_base.pth"
    shutil.copyfile(args.checkpoint, os.path.join(args.out_dir, base_name))

    manifest = {
        "method": "rollout_noise_calibration",
        "noise_from_monotone_fit": bool(args.fit_curve),
        "task": cfg.env.env_name, "algo": cfg.policy.algo,
        "base_checkpoint": base_name, "noise_baked": True,
        "horizon": cfg.env.horizon,
        "select_episodes": len(sel), "verify_episodes": len(ver),
        "seed": args.seed, "created": time.time(),
        "note": ("every rung is the same policy at a different rollout action-noise "
                 "level, so competence is the only thing that varies; selection "
                 "and verification use disjoint initial-state sets"),
        "rungs": [],
    }
    for r in rungs:
        # Write a self-contained checkpoint per level: the calibrated noise is
        # baked into the policy as a buffer, so loading p030.pth gives a 30%
        # policy with no external bookkeeping. Verification then measures the
        # saved file exactly as a study would use it, rather than measuring the
        # base policy with noise passed in separately.
        pct = int(round(r["target"] * 100))
        name = f"{cfg.policy.algo}_{cfg.env.env_name}_p{pct:03d}.pth"
        dest = os.path.join(args.out_dir, name)
        policy.set_action_noise(r["action_noise_std"])
        policy.save(dest, extra={
            "target_success": r["target"],
            "action_noise_std": r["action_noise_std"],
            "select_success": r["select_success"],
            "base_checkpoint": os.path.basename(args.checkpoint),
            "method": "rollout_noise_calibration",
        })
        rung_policy, _ = load_policy(dest, device, cfg.policy)
        rung_policy.eval()
        ev = evaluate_policy(rung_policy, cfg.env, n_episodes=len(ver),
                             seed=args.seed + 2, env=env, init_states=ver)
        blob = torch.load(dest, map_location="cpu", weights_only=False)
        blob["extra"].update({"verified_success": ev["success_rate"],
                              "verify_se": ev["success_se"]})
        torch.save(blob, dest)
        del rung_policy
        r_out = {
            "target": r["target"], "action_noise_std": r["action_noise_std"],
            "select_success": r["select_success"],
            "verified_success": ev["success_rate"],
            "verified_se": ev["success_se"], "mean_steps": ev["mean_steps"],
            "checkpoint": name,
        }
        manifest["rungs"].append(r_out)
        flag = "" if abs(ev["success_rate"] - r["target"]) <= args.tolerance * 2 \
            else "  <-- off target"
        print(f"  {r['target']:6.0%} {r['action_noise_std']:7.4f} "
              f"{r['select_success']:7.2f} {ev['success_rate']:9.2f} "
              f"{ev['success_se']:6.2f}{flag}")

    with open(os.path.join(args.out_dir, "ladder.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "curve.csv"), "w") as f:
        f.write("action_noise_std,select_success\n")
        for n, sr in sorted(cache.items()):
            f.write(f"{n:.6g},{sr:.6g}\n")

    env.close()
    devs = [abs(r["verified_success"] - r["target"]) for r in manifest["rungs"]]
    noises = [r["action_noise_std"] for r in manifest["rungs"]]
    print(f"\nmanifest -> {os.path.join(args.out_dir, 'ladder.json')}")
    print(f"largest deviation from target: {max(devs):.2f}  mean {np.mean(devs):.3f}")
    print(f"distinct noise levels: {len(set(noises))} for {len(targets)} targets")
    ver_sorted = [r["verified_success"] for r in
                  sorted(manifest["rungs"], key=lambda x: -x["action_noise_std"])]
    if any(b < a - 0.08 for a, b in zip(ver_sorted, ver_sorted[1:])):
        print("WARNING  verified success is not monotone in noise; widen "
              "--verify-episodes or narrow the target spacing")
    print(f"total time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
