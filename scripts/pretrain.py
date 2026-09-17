#!/usr/bin/env python
"""Pretrain a base policy on a demonstration dataset.

The base policy must already be competent before the DAgger study starts: the
whole point is that a participant's corrections produce a *visible* change within
a few hundred LoRA steps, which only works if the policy is near the decision
boundary rather than random.

Default data source is robomimic's official dataset for the task, so the base
policy reproduces published BC-RNN / ACT numbers rather than something ad hoc:

    python scripts/pretrain.py --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
        --algo bc_rnn --epochs 100 --eval-episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dagger_lh.config import Config
from dagger_lh.data import dataset_summary, load_episodes
from dagger_lh.envs import silence_robosuite_logs
from dagger_lh.evaluate import evaluate_policy
from dagger_lh.policies import build_policy, resolve_device
from dagger_lh.trainer import PolicyTrainer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config to start from")
    ap.add_argument("--dataset", required=True, help="robomimic-format HDF5")
    ap.add_argument("--algo", default=None, choices=["bc_rnn", "act"])
    ap.add_argument("--env", default=None, help="override env name (e.g. Lift)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="checkpoint path")
    ap.add_argument("--eval-episodes", type=int, default=0,
                    help="rollout episodes for success rate (0 = skip)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="also evaluate every N epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-success", type=float, default=None,
                    help="stop once rollout success reaches this (e.g. 0.4). "
                         "Produces a deliberately mediocre base policy, which is "
                         "what a DAgger study needs -- see --help notes.")
    ap.add_argument("--limit-demos", type=int, default=None,
                    help="train on only the first N demos (leaves the rest as "
                         "held-out data for validate_finetune.py)")
    ap.add_argument("--eval-slice", type=int, default=25,
                    help="epochs between checks when using --target-success")
    args = ap.parse_args()

    if args.target_success is not None and not args.eval_episodes:
        args.eval_episodes = 20

    silence_robosuite_logs()
    cfg = Config.load(args.config)
    overrides = {
        "policy.algo": args.algo,
        "env.env_name": args.env,
        "train.pretrain_epochs": args.epochs,
        "train.pretrain_steps_per_epoch": args.steps_per_epoch,
        "train.base_lr": args.lr,
        "train.batch_size": args.batch_size,
        "train.device": args.device,
        "train.seed": args.seed,
    }
    cfg = cfg.override({k: v for k, v in overrides.items() if v is not None})

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_keys = list(cfg.env.low_dim_keys) + list(cfg.env.image_keys)
    print(f"loading {args.dataset}")
    episodes = load_episodes(args.dataset, obs_keys)
    if not episodes:
        print(f"error: no episodes found in {args.dataset}", file=sys.stderr)
        return 1
    if args.limit_demos:
        episodes = episodes[:int(args.limit_demos)]
        print(f"limited to first {len(episodes)} demos "
              f"(remainder held out for finetuning validation)")
    print("dataset:", json.dumps(dataset_summary(episodes), indent=None))

    obs_shapes = {}
    for k in cfg.env.low_dim_keys:
        obs_shapes[k] = (int(episodes[0].obs[k].reshape(episodes[0].length, -1).shape[1]),)
    for k in cfg.env.image_keys:
        obs_shapes[k] = (3, cfg.env.camera_height, cfg.env.camera_width)
    ac_dim = int(episodes[0].actions.shape[-1])
    print("obs shapes:", obs_shapes, "| action dim:", ac_dim)

    device = resolve_device(cfg.train.device)
    print("device:", device)
    policy = build_policy(obs_shapes, ac_dim, cfg.policy,
                          cfg.env.low_dim_keys, cfg.env.image_keys, device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {cfg.policy.algo}  ({n_params:,} parameters)")

    trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)

    out = args.out or os.path.join(
        cfg.paths.checkpoints,
        f"{cfg.policy.algo}_{cfg.env.env_name}_base.pth")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    n_epochs = int(args.epochs or cfg.train.pretrain_epochs)

    if args.target_success is not None:
        # Train in short slices and stop as soon as the policy is good enough --
        # but no better. A base policy that already succeeds every time leaves the
        # operator's corrections nothing to improve, which would make even the
        # contingent condition feel non-contingent.
        print(f"training until success_rate >= {args.target_success:.2f} "
              f"(checking every {args.eval_slice} epochs, max {n_epochs})")
        done, ev, best = 0, {}, None
        while done < n_epochs:
            chunk = min(args.eval_slice, n_epochs - done)
            trainer.pretrain(episodes, epochs=chunk,
                             steps_per_epoch=args.steps_per_epoch, logger=None)
            done += chunk
            ev = evaluate_policy(policy, cfg.env, n_episodes=args.eval_episodes,
                                 seed=args.seed)
            sr = ev["success_rate"]
            print(f"  epoch {done:4d}  success_rate={sr:.2f}  "
                  f"mean_steps={ev['mean_steps']:.0f}")
            policy.save(out, extra={"pretrain_epochs": done, "eval": ev,
                                    "dataset": args.dataset,
                                    "target_success": args.target_success})
            if sr >= args.target_success:
                print(f"reached target at epoch {done}; stopping so the policy "
                      f"keeps room to improve during the study")
                break
        else:
            print(f"warning: never reached {args.target_success:.2f} within "
                  f"{n_epochs} epochs (final {ev.get('success_rate', 0):.2f})",
                  file=sys.stderr)
    elif args.eval_every and args.eval_episodes:
        # Train in slices so intermediate success rates can be reported.
        done = 0
        while done < n_epochs:
            chunk = min(args.eval_every, n_epochs - done)
            print(f"--- epochs {done + 1}..{done + chunk} ---")
            trainer.pretrain(episodes, epochs=chunk,
                             steps_per_epoch=args.steps_per_epoch)
            done += chunk
            ev = evaluate_policy(policy, cfg.env, n_episodes=args.eval_episodes,
                                 seed=args.seed)
            print(f"  [epoch {done}] success_rate={ev['success_rate']:.2f} "
                  f"mean_steps={ev['mean_steps']:.0f}")
            policy.save(out, extra={"pretrain_epochs": done, "eval": ev,
                                    "dataset": args.dataset})
    else:
        t0 = time.time()
        report = trainer.pretrain(episodes, epochs=n_epochs,
                                  steps_per_epoch=args.steps_per_epoch)
        print(f"pretrain done: {report.steps} steps in {report.seconds:.0f}s  "
              f"loss {report.loss_start:.4f} -> {report.loss_end:.4f}")
        ev = {}
        if args.eval_episodes:
            print(f"evaluating {args.eval_episodes} episodes...")
            ev = evaluate_policy(policy, cfg.env, n_episodes=args.eval_episodes,
                                 seed=args.seed, progress=True)
            print(f"success_rate={ev['success_rate']:.2f} "
                  f"mean_steps={ev['mean_steps']:.0f}")
        policy.save(out, extra={"pretrain_epochs": n_epochs, "eval": ev,
                                "dataset": args.dataset,
                                "seconds": time.time() - t0})

    cfg.base_checkpoint = out
    cfg.save(os.path.splitext(out)[0] + "_config.yaml")
    print(f"saved checkpoint -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
