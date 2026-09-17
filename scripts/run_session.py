#!/usr/bin/env python
"""Run an interactive session: demonstration collection or a DAgger study.

    # collect seed demonstrations by keyboard teleoperation
    python scripts/run_session.py --mode demo --demos 10

    # DAgger with real corrections (control arm)
    python scripts/run_session.py --mode dagger --condition responsive \
        --checkpoint data/checkpoints/bc_rnn_Lift_base.pth --rounds 10

    # learned-helplessness induction, then restored control (escape test)
    python scripts/run_session.py --mode dagger --two-phase \
        --condition noncontingent --phase2-condition responsive \
        --phase1-rounds 6 --phase2-rounds 6 \
        --checkpoint data/checkpoints/bc_rnn_Lift_base.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dagger_lh.config import Config
from dagger_lh.data import load_episodes
from dagger_lh.envs import silence_robosuite_logs
from dagger_lh.policies import build_policy, load_policy, resolve_device
from dagger_lh.runner import SessionRunner
from dagger_lh.study import Condition
from dagger_lh.trainer import PolicyTrainer


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config file")
    ap.add_argument("--mode", choices=["demo", "dagger"], default=None)
    ap.add_argument("--participant", default=None, help="participant id")

    g = ap.add_argument_group("task")
    g.add_argument("--env", default=None, help="robosuite env (Lift, PickPlaceCan, ...)")
    g.add_argument("--robot", default=None)
    g.add_argument("--horizon", type=int, default=None)
    g.add_argument("--images", action="store_true",
                   help="use camera observations instead of low-dim only")

    g = ap.add_argument_group("policy")
    g.add_argument("--algo", choices=["bc_rnn", "act"], default=None)
    g.add_argument("--checkpoint", default=None, help="base policy checkpoint")
    g.add_argument("--device", default=None)
    g.add_argument("--lora-rank", type=int, default=None)
    g.add_argument("--no-lora", action="store_true",
                   help="finetune all weights instead of LoRA adapters")

    g = ap.add_argument_group("study")
    g.add_argument("--condition", choices=list(Condition.ALL), default=None)
    g.add_argument("--two-phase", action="store_true",
                   help="induction phase then restored-control escape test")
    g.add_argument("--phase2-condition", choices=list(Condition.ALL), default=None)
    g.add_argument("--phase1-rounds", type=int, default=None)
    g.add_argument("--phase2-rounds", type=int, default=None)
    g.add_argument("--rounds", type=int, default=None, help="single-phase round count")
    g.add_argument("--demos", type=int, default=None, help="demo-mode target count")
    g.add_argument("--rollouts-per-round", type=int, default=None)
    g.add_argument("--ladder", default=None,
                   help="ladder.json from scripts/train_policy_ladder.py; "
                        "required by the 'yoked' (scripted success) condition")
    g.add_argument("--yoked-schedule", type=float, nargs="+", default=None,
                   help="success rate per round for 'yoked', e.g. "
                        "0.1 0.1 0.2 0.2 0.3 0.3")
    g.add_argument("--no-survey", action="store_true")
    g.add_argument("--seed", type=int, default=None)

    g = ap.add_argument_group("training")
    g.add_argument("--finetune-steps", type=int, default=None)
    g.add_argument("--finetune-max-seconds", type=float, default=None)
    g.add_argument("--finetune-min-seconds", type=float, default=None)

    g = ap.add_argument_group("teleop")
    g.add_argument("--hold-to-intervene", action="store_true",
                   help="hold TAB/SHIFT to drive instead of toggling")
    g.add_argument("--pos-sensitivity", type=float, default=None)
    g.add_argument("--rot-sensitivity", type=float, default=None)
    g.add_argument("--window-scale", type=float, default=None)

    g.add_argument("--dataset", default=None,
                   help="append to / seed from this demo HDF5")
    g.add_argument("--seed-dataset", default=None,
                   help="prior demos to include in the DAgger training pool")
    return ap


def resolve_config(args) -> Config:
    cfg = Config.load(args.config)
    dotted = {
        "mode": args.mode,
        "env.env_name": args.env,
        "env.robots": args.robot,
        "env.horizon": args.horizon,
        "env.seed": args.seed,
        "policy.algo": args.algo,
        "policy.lora.rank": args.lora_rank,
        "train.device": args.device,
        "train.finetune_steps": args.finetune_steps,
        "train.finetune_max_seconds": args.finetune_max_seconds,
        "train.finetune_min_seconds": args.finetune_min_seconds,
        "train.seed": args.seed,
        "study.participant_id": args.participant,
        "study.condition": args.condition,
        "study.phase2_condition": args.phase2_condition,
        "study.phase1_rounds": args.phase1_rounds,
        "study.phase2_rounds": args.phase2_rounds,
        "study.num_rounds": args.rounds if args.rounds is not None else args.demos,
        "study.rollouts_per_round": args.rollouts_per_round,
        "study.seed": args.seed,
        "study.ladder_path": args.ladder,
        "study.yoked_schedule": args.yoked_schedule,
        "teleop.pos_sensitivity": args.pos_sensitivity,
        "teleop.rot_sensitivity": args.rot_sensitivity,
        "teleop.window_scale": args.window_scale,
        "demo_dataset": args.dataset,
    }
    cfg = cfg.override({k: v for k, v in dotted.items() if v is not None})
    if args.two_phase:
        cfg = cfg.override({"study.two_phase": True})
    if args.no_survey:
        cfg = cfg.override({"study.survey_enabled": False})
    if args.hold_to_intervene:
        cfg = cfg.override({"teleop.hold_to_intervene": True})
    if args.no_lora:
        cfg = cfg.override({"policy.lora.enabled": False})
    if args.images:
        cfg = cfg.override({"env.image_keys": ["agentview_image",
                                               "robot0_eye_in_hand_image"]})
    if args.checkpoint:
        cfg.base_checkpoint = args.checkpoint
    conds = {cfg.study.condition}
    if cfg.study.two_phase:
        conds.add(cfg.study.phase2_condition)
    if "yoked" in conds and not cfg.study.ladder_path:
        raise SystemExit(
            "the 'yoked' condition needs --ladder: a scripted success curve "
            "requires checkpoints of known competence. Build one with\n"
            "  python scripts/train_policy_ladder.py --dataset <demos.hdf5> "
            "--out-dir data/checkpoints/ladder")
    return cfg


def main() -> int:
    args = build_parser().parse_args()
    silence_robosuite_logs()
    cfg = resolve_config(args)

    if cfg.study.seed is not None:
        torch.manual_seed(cfg.study.seed)
        np.random.seed(cfg.study.seed)

    print(f"mode={cfg.mode}  env={cfg.env.env_name}  algo={cfg.policy.algo}  "
          f"participant={cfg.study.participant_id}")
    if cfg.mode == "dagger":
        print(f"condition={cfg.study.condition}"
              + (f" -> phase2={cfg.study.phase2_condition}" if cfg.study.two_phase else ""))

    runner = SessionRunner(cfg)
    try:
        if cfg.mode == "demo":
            summary = runner.run_demo_mode(num_demos=args.demos)
            print(json.dumps(summary, indent=2))
            return 0

        # ---------------- dagger ----------------
        device = resolve_device(cfg.train.device)
        obs_shapes = runner.env.obs_shapes()
        ac_dim = runner.env.action_dim

        if cfg.base_checkpoint and os.path.exists(cfg.base_checkpoint):
            print(f"loading base policy {cfg.base_checkpoint}")
            policy, extra = load_policy(cfg.base_checkpoint, device, cfg.policy)
            if extra.get("eval"):
                print("  base policy eval:", extra["eval"])
        else:
            if cfg.base_checkpoint:
                print(f"warning: checkpoint {cfg.base_checkpoint} not found; "
                      "starting from a randomly initialised policy",
                      file=sys.stderr)
            policy = build_policy(obs_shapes, ac_dim, cfg.policy,
                                  cfg.env.low_dim_keys, cfg.env.image_keys, device)

        trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)

        seed_path = args.seed_dataset
        if seed_path and os.path.exists(seed_path):
            seeded = load_episodes(seed_path, runner.obs_keys)
            runner.episodes.extend(seeded)
            print(f"seeded training pool with {len(seeded)} prior episodes")
            trainer.fit_normalizer(runner.episodes)

        if cfg.policy.lora.enabled:
            print(trainer.enable_lora())
        else:
            print(f"full finetuning: {trainer.param_summary()}")

        summary = runner.run_dagger_mode(trainer)
        print(json.dumps(summary, indent=2, default=str))
        final = os.path.join(runner.session_dir, "policy_final.pth")
        policy.save(final, extra={"summary": summary})
        print(f"final policy -> {final}")
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
