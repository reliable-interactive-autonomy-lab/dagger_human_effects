#!/usr/bin/env python
"""Inspect a policy ladder: print the rung table and optionally re-verify it.

    python scripts/show_ladder.py --ladder data/checkpoints/ladder/ladder.json
    python scripts/show_ladder.py --ladder ... --plot ladder.png
    python scripts/show_ladder.py --ladder ... --reverify 100

`--reverify` re-measures every rung from a *fresh* set of initial states. Worth
doing once before a study: it is the only way to catch a ladder whose numbers
were fitted to one particular state sample, and it confirms the checkpoints still
behave the same way under the environment you will actually run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dagger_lh.config import Config
from dagger_lh.envs import silence_robosuite_logs
from dagger_lh.ladder import PolicyLadder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--reverify", type=int, default=0,
                    help="rollouts per rung from fresh initial states (0 = skip)")
    ap.add_argument("--seed", type=int, default=4242,
                    help="a seed the ladder was NOT built with")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    silence_robosuite_logs()
    lad = PolicyLadder(args.ladder)
    m = lad.manifest
    print(lad.describe())
    print(f"task {m.get('task')} | algo {m.get('algo')} | horizon {m.get('horizon')}")
    print(f"built from {m.get('dataset')}")
    print(f"select/verify episodes: {m.get('select_episodes')}/{m.get('verify_episodes')}")

    print(f"\n  {'target':>7} {'noise':>7} {'select':>7} {'verified':>9} "
          f"{'+/-':>6} {'steps':>7}  checkpoint")
    for r in lad.rungs:
        tag = ""
        if r.get("interpolated"):
            tag = f"  [interp a={r.get('alpha', 0):.3f}]"
        elif r.get("noise_sigma") is not None:
            tag = f"  [train sigma={r['noise_sigma']:.3f}]"
        print(f"  {r['target']:6.0%} {r.get('action_noise_std', 0) or 0:7.4f} "
              f"{r['select_success']:7.2f} {r['verified_success']:9.2f} "
              f"{r.get('verified_se', 0):6.2f} {r.get('mean_steps', 0):7.0f}  "
              f"{r['checkpoint']}{tag}")

    devs = [abs(r["verified_success"] - r["target"]) for r in lad.rungs]
    print(f"\nmax deviation from target: {max(devs):.2f}  "
          f"mean {float(np.mean(devs)):.3f}")
    # A ladder is only useful if the rungs are ordered and separated.
    ver = [r["verified_success"] for r in lad.rungs]
    if any(b < a for a, b in zip(ver, ver[1:])):
        print("WARNING  rungs are not monotonic in verified success")
    gaps = [b - a for a, b in zip(ver, ver[1:])]
    if gaps and min(gaps) <= 0:
        print("WARNING  two rungs share the same success rate; the ladder has "
              "fewer usable levels than it appears to")
    # The decisive check is which training snapshot each rung came from, not the
    # file name: the builder writes a separate file per target, so two rungs can
    # be byte-identical policies under different names. Duplicated snapshots mean
    # fewer real levels than the table suggests, and a manipulation built on the
    # duplicates would be a no-op.
    from collections import Counter
    # What makes two rungs genuinely different depends on how the ladder was
    # built: distinct weights, or the same weights at a distinct noise level.
    ident = [(r["checkpoint"], round(float(r.get("action_noise_std") or 0.0), 5),
              r.get("grad_steps"), r.get("noise_sigma")) for r in lad.rungs]
    n_distinct = len(set(ident))
    print(f"distinct policy settings: {n_distinct} / {len(ident)} rungs")
    if n_distinct < len(ident):
        for key, n in Counter(ident).items():
            if n > 1:
                shared = [f"{r['target']:.0%}" for r, k in zip(lad.rungs, ident)
                          if k == key]
                print(f"  WARNING  targets {', '.join(shared)} are the same "
                      f"policy at the same setting -- a manipulation built on "
                      f"them would be a no-op")
        print("  Sample the control variable more finely around the crowded "
              "bands and rebuild.")

    if args.reverify:
        import torch
        from dagger_lh.envs import make_env
        from dagger_lh.evaluate import evaluate_policy, sample_init_states
        from dagger_lh.policies import load_policy, resolve_device

        cfg = Config.load(args.config)
        if m.get("horizon"):
            cfg = cfg.override({"env.horizon": int(m["horizon"])})
        if args.device:
            cfg = cfg.override({"train.device": args.device})
        device = resolve_device(cfg.train.device)
        env = make_env(cfg.env)
        states = sample_init_states(env, args.reverify, seed=args.seed)
        print(f"\nre-verifying on {len(states)} fresh states (seed {args.seed}):")
        print(f"  {'target':>7} {'stored':>7} {'fresh':>7} {'delta':>7}")
        rows = []
        try:
            for r in lad.rungs:
                path = os.path.join(lad.root, r["checkpoint"])
                pol, _ = load_policy(path, device, cfg.policy)
                ev = evaluate_policy(
                    pol, cfg.env, n_episodes=len(states), seed=args.seed + 1,
                    env=env, init_states=states,
                    action_noise_std=float(r.get("action_noise_std") or 0.0))
                d = ev["success_rate"] - r["verified_success"]
                rows.append((r["target"], r["verified_success"], ev["success_rate"]))
                print(f"  {r['target']:6.0%} {r['verified_success']:7.2f} "
                      f"{ev['success_rate']:7.2f} {d:+7.2f}")
        finally:
            env.close()
        drift = max(abs(c - b) for _, b, c in rows)
        print(f"\nlargest drift between stored and fresh measurement: {drift:.2f}")
        if drift > 0.15:
            print("  That is large. The stored numbers were probably fitted to "
                  "their verification sample; prefer the fresh ones, or rebuild "
                  "the ladder with more --verify-episodes.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curve_path = os.path.join(lad.root, "curve.csv")
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        if os.path.exists(curve_path):
            with open(curve_path) as f:
                rows = list(csv.DictReader(f))
            key = ("action_noise_std" if "action_noise_std" in rows[0]
                   else ("noise_sigma" if "noise_sigma" in rows[0] else "grad_steps"))
            xs = [float(r[key]) for r in rows]
            ys = [float(r["select_success"]) for r in rows]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], lw=1.2,
                    color="#9aa0b5", label="training curve (selection set)")
        # x-axis is whichever control variable this ladder actually used
        def xval(r):
            if r.get("action_noise_std") is not None:
                return float(r["action_noise_std"])
            if r.get("noise_sigma") is not None:
                return float(r["noise_sigma"])
            return float(r.get("grad_steps", 0))
        ax.scatter([xval(r) for r in lad.rungs],
                   [r["verified_success"] for r in lad.rungs],
                   zorder=3, s=46, color="#2a6df5",
                   label="ladder rungs (held-out verification)")
        for r in lad.rungs:
            ax.annotate(f"{r['verified_success']:.0%}",
                        (xval(r), r["verified_success"]),
                        textcoords="offset points", xytext=(6, -3), fontsize=8)
        for t in sorted({r["target"] for r in lad.rungs}):
            ax.axhline(t, color="#e4e6ee", lw=0.7, zorder=0)
        method = m.get("method", "")
        ax.set_xlabel({"rollout_noise_calibration": "rollout action noise",
                       "noise_sweep": "training-data corruption sigma"}
                      .get(method, "gradient steps"))
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.04, 1.04)
        ax.set_title(f"Policy ladder — {m.get('algo')} on {m.get('task')}")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
