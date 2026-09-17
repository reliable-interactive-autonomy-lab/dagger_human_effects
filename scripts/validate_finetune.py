#!/usr/bin/env python
"""Measure whether on-the-fly LoRA finetuning actually improves the policy.

This is the load-bearing assumption of the whole study: in the `responsive`
condition, a few hundred LoRA steps on a handful of fresh demonstrations must
produce an improvement the operator can *see* within one round. If it does not,
the contingent arm is indistinguishable from the non-contingent one and the
experiment measures nothing.

Runs the same budget the live session uses, and reports the sham arm alongside as
a control -- the sham update must leave success rate statistically unchanged.

    python scripts/validate_finetune.py \
        --checkpoint data/checkpoints/bc_rnn_Lift_study.pth \
        --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
        --rounds 4 --demos-per-round 5 --eval-episodes 20
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dagger_lh.config import Config
from dagger_lh.data import load_episodes
from dagger_lh.envs import make_env, silence_robosuite_logs
from dagger_lh.evaluate import evaluate_policy, sample_init_states
from dagger_lh.lora import lora_state_dict
from dagger_lh.policies import load_policy, resolve_device
from dagger_lh.trainer import PolicyTrainer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True,
                    help="source of the 'corrections' fed in each round")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--demos-per-round", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--finetune-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sham", action="store_true",
                    help="also run the non-contingent arm as a control")
    ap.add_argument("--skip-demos", type=int, default=0,
                    help="skip the first N demos, so the 'corrections' are data "
                         "the base policy was not pretrained on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    silence_robosuite_logs()
    cfg = Config.load(args.config)
    if args.finetune_steps:
        cfg = cfg.override({"train.finetune_steps": args.finetune_steps})
    if args.device:
        cfg = cfg.override({"train.device": args.device})
    device = resolve_device(cfg.train.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_keys = list(cfg.env.low_dim_keys) + list(cfg.env.image_keys)
    pool = load_episodes(args.dataset, obs_keys)
    if args.skip_demos:
        pool = pool[int(args.skip_demos):]
        print(f"using {len(pool)} held-out demos as corrections "
              f"(skipped first {args.skip_demos})")
    if len(pool) < args.rounds * args.demos_per_round:
        print("warning: dataset smaller than requested rounds x demos",
              file=sys.stderr)

    env = make_env(cfg.env)
    # One fixed set of starting configurations, shared by every evaluation, so
    # before/after differences are not swamped by object-placement variance.
    init_states = sample_init_states(env, args.eval_episodes, seed=args.seed)
    print(f"paired evaluation on {len(init_states)} fixed initial states")
    results = {}
    arms = ["responsive"] + (["noncontingent"] if args.sham else [])
    try:
        for arm in arms:
            print(f"\n{'=' * 62}\narm: {arm}\n{'=' * 62}")
            policy, extra = load_policy(args.checkpoint, device, cfg.policy)
            trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)
            print(trainer.enable_lora())
            # The checkpoint already carries normalisation stats from pretraining.
            # Refitting them mid-session would change the policy outside the sham
            # snapshot's scope, so it must not happen between rounds.

            base = evaluate_policy(policy, cfg.env, n_episodes=args.eval_episodes,
                                   seed=args.seed, env=env,
                                   init_states=init_states)
            print(f"round 0 (base)        success={base['success_rate']:.2f}"
                  f" +/-{base['success_se']:.2f}  steps={base['mean_steps']:.0f}")
            traj = [base["success_rate"]]
            aggregated = []
            times = []
            for r in range(args.rounds):
                lo = r * args.demos_per_round
                fresh = pool[lo:lo + args.demos_per_round]
                for ep in fresh:
                    ep.meta = dict(ep.meta)
                    ep.meta["round"] = r          # newest round gets oversampled
                aggregated.extend(fresh)
                t0 = time.time()
                rep = trainer.finetune(aggregated,
                                       commit=(arm == "responsive"))
                dt = time.time() - t0
                times.append(dt)
                ev = evaluate_policy(policy, cfg.env,
                                     n_episodes=args.eval_episodes,
                                     seed=args.seed, env=env,
                                     init_states=init_states)
                traj.append(ev["success_rate"])
                print(f"round {r + 1}  +{len(fresh)} demos  "
                      f"{rep.steps} steps in {dt:4.1f}s  "
                      f"loss {rep.loss_start:8.3f} -> {rep.loss_end:8.3f}  "
                      f"success={ev['success_rate']:.2f} +/-{ev['success_se']:.2f}")
            results[arm] = {
                "trajectory": traj,
                "base": traj[0],
                "final": traj[-1],
                "delta": traj[-1] - traj[0],
                "mean_update_seconds": float(np.mean(times)),
                "max_update_seconds": float(np.max(times)),
            }
    finally:
        env.close()

    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    for arm, r in results.items():
        arrow = " -> ".join(f"{v:.2f}" for v in r["trajectory"])
        print(f"{arm:15s} {arrow}   delta={r['delta']:+.2f}  "
              f"update {r['mean_update_seconds']:.1f}s avg / "
              f"{r['max_update_seconds']:.1f}s max")

    ok = True
    if "responsive" in results:
        d = results["responsive"]["delta"]
        good = d > 0.05
        ok &= good
        print(f"\n{'PASS' if good else 'FAIL'}  contingent finetuning improves the "
              f"policy (delta={d:+.2f}, need >+0.05)")
        budget = cfg.train.finetune_max_seconds
        fast = results["responsive"]["max_update_seconds"] <= budget + 2
        ok &= fast
        print(f"{'PASS' if fast else 'FAIL'}  updates fit the between-round budget "
              f"({results['responsive']['max_update_seconds']:.1f}s "
              f"vs {budget:.0f}s)")
    if "noncontingent" in results:
        d = abs(results["noncontingent"]["delta"])
        # Sham leaves weights identical, so any movement is rollout stochasticity.
        good = d <= 0.15
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  sham finetuning does not improve the "
              f"policy (|delta|={d:.2f}, rollout noise only)")
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
