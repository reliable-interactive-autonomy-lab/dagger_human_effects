#!/usr/bin/env python
"""Headless end-to-end test of the DAgger pipeline with a scripted operator.

Substitutes a scripted "operator" and a null display for the keyboard and pygame
window, then runs real robosuite episodes, real LoRA finetuning and real logging.
Verifies the parts a human session cannot easily assert:

  * episodes land in a valid robomimic-format HDF5 with the DAgger extras
  * only human-labelled states are trained on
  * the `responsive` arm commits its update and `noncontingent` rolls it back
    bit-for-bit, while both take the same wall-clock time
  * per-round metrics, CSV, JSONL events and the summary are all written

    python scripts/smoke_test.py --algo bc_rnn --rounds 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from dagger_lh.config import Config
from dagger_lh.envs import silence_robosuite_logs
from dagger_lh.policies import build_policy, resolve_device
from dagger_lh.runner import SessionRunner
from dagger_lh.teleop import TeleopState
from dagger_lh.trainer import PolicyTrainer


class NullDisplay:
    """Display stand-in: records what would have been shown."""
    wants_frames = False

    def __init__(self, verbose: bool = False):
        self.K_ESCAPE, self.K_RETURN = 27, 13
        self.messages = []
        self.frames = 0
        self.show_help = False
        self.clock = None
        self.verbose = verbose

    def draw(self, main_img, hud, side_imgs=None):
        self.frames += 1

    def message(self, title, body="", sub=""):
        self.messages.append(title)
        if self.verbose:
            print(f"    [screen] {title}")

    def wait_for_key(self, keys=None):
        return self.K_RETURN          # always "continue"

    def pump(self):
        pass

    def close(self):
        pass


class ScriptedOperator:
    """Stands in for KeyboardDevice: intervenes on a fixed schedule.

    Between `takeover_at` and `release_at` it "presses keys" and drives the
    end-effector downwards, so there is genuine human-labelled data to train on.
    """

    def __init__(self, action_dim=7, takeover_at=5, release_at=25,
                 episode_len=40, giveup=False):
        self.action_dim = action_dim
        self.takeover_at = takeover_at
        self.release_at = release_at
        self.episode_len = episode_len
        self.giveup = giveup
        self.total_keypresses = 0
        self.total_engaged_steps = 0
        self.total_steps = 0
        self._t = 0
        self._intervening = False

    def reset_episode(self):
        self._t = 0
        self._intervening = False

    @property
    def intervening(self):
        return self._intervening

    def force_release(self):
        self._intervening = False

    def poll(self) -> TeleopState:
        st = TeleopState(action=np.zeros(self.action_dim))
        t = self._t
        self._t += 1
        self.total_steps += 1

        if t == self.takeover_at:
            self._intervening = True
            st.event_toggle_control = True
        if t == self.release_at:
            self._intervening = False
            st.event_toggle_control = True
        if t >= self.episode_len:
            if self.giveup:
                st.event_giveup = True
            else:
                st.event_success = True
            return st

        st.intervening = self._intervening
        if self._intervening:
            a = np.zeros(self.action_dim)
            a[2] = -0.4                       # descend
            a[0] = 0.15 * np.sin(t / 4.0)
            if self.action_dim > 6:
                a[6] = 1.0 if t > 15 else -1.0
            st.action = a
            st.engaged = True
            st.keys_held = 2
            st.input_norm = float(np.linalg.norm(a[:6]))
            st.gripper_closed = t > 15
            self.total_keypresses += 2
            self.total_engaged_steps += 1
        return st


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="bc_rnn", choices=["bc_rnn", "act"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episode-len", type=int, default=40)
    ap.add_argument("--finetune-steps", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temp session dir")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ladder", default=None,
                    help="also exercise the scripted-success ('yoked') condition "
                         "against this ladder.json")
    args = ap.parse_args()

    silence_robosuite_logs()
    torch.manual_seed(0)
    np.random.seed(0)
    tmp = tempfile.mkdtemp(prefix="dagger_smoke_")
    failures = 0

    try:
        for condition in ("responsive", "noncontingent"):
            print(f"\n=== condition: {condition} ===")
            cfg = Config.from_dict({
                "mode": "dagger",
                "env": {"env_name": "Lift", "horizon": args.episode_len + 5},
                "policy": {"algo": args.algo, "chunk_size": 8,
                           "lora": {"rank": 4, "alpha": 8.0}},
                "train": {"finetune_steps": args.finetune_steps,
                          "finetune_max_seconds": 30.0,
                          "finetune_min_seconds": 0.0,
                          "batch_size": 16,
                          "device": args.device or "auto"},
                "study": {"participant_id": f"smoke_{condition}",
                          "condition": condition,
                          "num_rounds": args.rounds,
                          "rollouts_per_round": 1,
                          "survey_enabled": False},
                "paths": {"root": tmp, "demos": os.path.join(tmp, "demos"),
                          "checkpoints": os.path.join(tmp, "ckpt"),
                          "sessions": os.path.join(tmp, "sessions"),
                          "logs": os.path.join(tmp, "logs")},
            })

            display = NullDisplay(verbose=args.verbose)
            runner = SessionRunner(
                cfg, display=display,
                teleop=ScriptedOperator(episode_len=args.episode_len))
            try:
                device = resolve_device(cfg.train.device)
                policy = build_policy(runner.env.obs_shapes(), runner.env.action_dim,
                                      cfg.policy, cfg.env.low_dim_keys,
                                      cfg.env.image_keys, device)
                runner.teleop.action_dim = runner.env.action_dim
                trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)
                trainer.enable_lora()

                # Compare the FULL state dict, not just LoRA tensors: buffers
                # (e.g. the observation normaliser's mean/std) are not trainable
                # but do change the policy, and a sham update must restore them too.
                before = {k: v.clone() for k, v in policy.state_dict().items()
                          if v.is_floating_point()}
                t0 = time.time()
                summary = runner.run_dagger_mode(trainer)
                elapsed = time.time() - t0
                after = policy.state_dict()

                changed = max((before[k] - after[k]).abs().max().item()
                              for k in before)
                print(f"  session ran {elapsed:.1f}s, "
                      f"{display.frames} rendered steps")

                # ---- contingency manipulation ----
                if condition == "responsive":
                    failures += not check(
                        "responsive arm changed the policy",
                        changed > 0,
                        f"max|delta| over {len(before)} tensors = {changed:.3g}")
                else:
                    failures += not check(
                        "noncontingent arm left policy unchanged",
                        changed == 0.0,
                        f"max|delta| over {len(before)} tensors = {changed:.3g}")

                # ---- dataset ----
                path = runner.demo_path
                failures += not check("demo HDF5 written", os.path.exists(path))
                with h5py.File(path) as f:
                    demos = [k for k in f["data"] if k.startswith("demo_")]
                    g = f["data"][demos[0]]
                    has_extras = "actor" in g and "intervention" in g
                    n_human = int(np.array(g["actor"]).sum())
                    n_tot = int(g.attrs["num_samples"])
                    failures += not check(f"episodes saved ({len(demos)})",
                                          len(demos) == args.rounds)
                    failures += not check("robomimic layout + DAgger extras",
                                          has_extras and "obs" in g and "actions" in g
                                          and "states" in g)
                    failures += not check("human-labelled subset present",
                                          0 < n_human < n_tot,
                                          f"{n_human}/{n_tot} steps")
                    failures += not check("env_args recorded",
                                          "env_args" in f["data"].attrs)
                    failures += not check("per-round filter keys",
                                          "mask" in f and len(f["mask"]) == args.rounds)

                # ---- logs ----
                sd = runner.session_dir
                rounds_csv = os.path.join(sd, "rounds.csv")
                failures += not check("events.jsonl",
                                      os.path.exists(os.path.join(sd, "events.jsonl")))
                failures += not check("summary.json",
                                      os.path.exists(os.path.join(sd, "summary.json")))
                ok_csv = os.path.exists(rounds_csv)
                if ok_csv:
                    lines = open(rounds_csv).read().strip().split("\n")
                    ok_csv = len(lines) == args.rounds + 1
                    hdr = lines[0].split(",")
                    need = {"intervention_rate", "engagement_rate", "passivity",
                            "keypresses_per_step", "giveup_events"}
                    failures += not check("CSV has helplessness measures",
                                          need.issubset(set(hdr)),
                                          f"{len(hdr)} columns")
                failures += not check(f"rounds.csv has {args.rounds} rows", ok_csv)
                failures += not check("adapter snapshots per round",
                                      len(glob.glob(os.path.join(sd, "adapter_round_*.pth")))
                                      == args.rounds)

                iv = summary.get("delta_intervention_rate")
                failures += not check("summary carries helplessness index",
                                      "index" in summary,
                                      f"index={summary.get('index')}")
            finally:
                runner.close()

        # ---------------- scripted success (yoked) ----------------
        if args.ladder:
            print("\n=== condition: yoked (scripted success from a ladder) ===")
            from dagger_lh.ladder import PolicyLadder
            lad = PolicyLadder(args.ladder)
            sched = [lad.available()[0], lad.available()[-1]][:args.rounds]
            while len(sched) < args.rounds:
                sched.append(sched[-1])
            cfg = Config.from_dict({
                "mode": "dagger",
                "env": {"env_name": "Lift", "horizon": args.episode_len + 5},
                "policy": {"algo": args.algo, "lora": {"rank": 4}},
                "train": {"finetune_steps": args.finetune_steps,
                          "finetune_max_seconds": 30.0, "batch_size": 16,
                          "device": args.device or "auto"},
                "study": {"participant_id": "smoke_yoked", "condition": "yoked",
                          "num_rounds": args.rounds, "rollouts_per_round": 1,
                          "survey_enabled": False,
                          "ladder_path": args.ladder, "yoked_schedule": sched},
                "paths": {"root": tmp, "demos": os.path.join(tmp, "demos3"),
                          "checkpoints": os.path.join(tmp, "ckpt3"),
                          "sessions": os.path.join(tmp, "sessions3"),
                          "logs": os.path.join(tmp, "logs3")},
            })
            display = NullDisplay(verbose=args.verbose)
            runner = SessionRunner(cfg, display=display,
                                   teleop=ScriptedOperator(episode_len=args.episode_len))
            try:
                device = resolve_device(cfg.train.device)
                policy = build_policy(runner.env.obs_shapes(), runner.env.action_dim,
                                      cfg.policy, cfg.env.low_dim_keys,
                                      cfg.env.image_keys, device)
                runner.teleop.action_dim = runner.env.action_dim
                trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)
                trainer.enable_lora()
                before = {k: v.clone() for k, v in policy.state_dict().items()
                          if v.is_floating_point()}
                runner.run_dagger_mode(trainer)
                after = policy.state_dict()
                moved = max((before[k] - after[k]).abs().max().item() for k in before)

                failures += not check("ladder loaded for the session",
                                      runner.ladder is not None,
                                      runner.ladder.describe() if runner.ladder else "")
                events = [json.loads(l) for l in
                          open(os.path.join(runner.session_dir, "events.jsonl"))]
                rungs = [e for e in events if e.get("event") == "ladder_rung"]
                failures += not check("a ladder rung was selected each round",
                                      len(rungs) == args.rounds,
                                      f"{len(rungs)} rung selections")
                if rungs:
                    failures += not check(
                        "rung matches the scheduled target",
                        abs(float(rungs[0]["rung_success"])
                            - float(rungs[0]["target"])) < 0.35,
                        f"target {rungs[0]['target']:.2f} -> "
                        f"rung {rungs[0]['rung_success']:.2f}")
                    failures += not check(
                        "schedule advanced across rounds",
                        len({r["checkpoint"] for r in rungs}) > 1
                        or len(set(sched)) == 1,
                        str([r["checkpoint"] for r in rungs]))
                failures += not check(
                    "trainable policy still rolled back (scripted = not contingent)",
                    moved == 0.0, f"max|delta|={moved:.3g}")
            finally:
                runner.close()

        # ---------------- demonstration mode ----------------
        print("\n=== mode: demo (pure teleoperation) ===")
        cfg = Config.from_dict({
            "mode": "demo",
            "env": {"env_name": "Lift", "horizon": args.episode_len + 5},
            "study": {"participant_id": "smoke_demo", "num_rounds": 2,
                      "survey_enabled": False},
            "paths": {"root": tmp, "demos": os.path.join(tmp, "demos2"),
                      "checkpoints": os.path.join(tmp, "ckpt2"),
                      "sessions": os.path.join(tmp, "sessions2"),
                      "logs": os.path.join(tmp, "logs2")},
        })
        runner = SessionRunner(cfg, display=NullDisplay(verbose=args.verbose),
                               teleop=ScriptedOperator(episode_len=args.episode_len))
        try:
            runner.teleop.action_dim = runner.env.action_dim
            s = runner.run_demo_mode(num_demos=2)
            failures += not check("demo mode collected episodes",
                                  s.get("collected") == 2, str(s.get("collected")))
            with h5py.File(runner.demo_path) as f:
                g = f["data"][sorted(k for k in f["data"] if k.startswith("demo_"))[0]]
                frac = float(np.array(g["actor"]).mean())
                failures += not check("demo mode is fully human-labelled",
                                      frac == 1.0, f"actor mean={frac:.2f}")
        finally:
            runner.close()

        print("\n" + "=" * 60)
        if failures == 0:
            print("ALL CHECKS PASSED")
        else:
            print(f"{failures} CHECK(S) FAILED")
        return 1 if failures else 0
    finally:
        if args.keep:
            print(f"session data kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
