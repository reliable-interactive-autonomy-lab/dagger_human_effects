#!/usr/bin/env python
"""Generate synthetic participant sessions with a known helplessness effect.

Two uses:
  1. exercise `analyze.py` end to end without recruiting anyone;
  2. power analysis -- pick `--n-per-condition` by seeing what effect size the
     analysis can actually detect before running real participants.

The generative model is intentionally simple and explicit: each simulated
operator starts at a baseline effort level and, under a non-contingent
condition, decays toward zero at `--decay` per round (the helplessness effect),
with per-participant and per-round noise. Under `responsive` effort is flat.
This is an assumption about behaviour, not a prediction -- it exists so the
analysis code has data with a *known* ground truth to recover.

    python scripts/simulate_study.py --out data/sim_sessions --n-per-condition 12
    python scripts/analyze.py --sessions data/sim_sessions --plot sim.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dagger_lh.config import Config
from dagger_lh.study import RoundMetrics, SessionLogger, StudyController


def simulate_participant(pid: str, condition: str, n_rounds: int,
                         rng: np.random.Generator, decay: float,
                         two_phase: bool = False, phase1: int = 0,
                         phase2_condition: str = "responsive",
                         out_root: str = "data/sim_sessions") -> str:
    cfg = Config.from_dict({
        "mode": "dagger",
        "study": {
            "participant_id": pid, "condition": condition,
            "num_rounds": n_rounds, "two_phase": two_phase,
            "phase1_rounds": phase1, "phase2_rounds": n_rounds - phase1,
            "phase2_condition": phase2_condition,
        },
    })
    ctrl = StudyController(cfg.study)
    sdir = os.path.join(out_root, f"{pid}_{condition}")
    logger = SessionLogger(sdir, cfg)

    # per-participant traits
    base_interv = float(np.clip(rng.normal(0.45, 0.10), 0.10, 0.85))
    base_engage = float(np.clip(base_interv * rng.normal(1.05, 0.08), 0.05, 0.95))
    base_keys = float(np.clip(rng.normal(1.8, 0.4), 0.4, 3.5))
    trait_decay = float(np.clip(rng.normal(decay, decay * 0.35), 0.0, 0.6))

    for r in range(n_rounds):
        plan = ctrl.plan_round(r)
        contingent = plan.condition == "responsive"
        # Effort decays only while the operator lacks control; once control is
        # restored, a *helpless* operator recovers only partially.
        if two_phase and plan.phase == 2:
            elapsed_noncontingent = phase1
            recovery = 0.35  # partial recovery = the helplessness signature
            local = r - phase1
            decay_factor = ((1 - trait_decay) ** elapsed_noncontingent)
            decay_factor *= (1 + recovery * (local + 1) / max(n_rounds - phase1, 1))
            decay_factor = min(decay_factor, 1.0)
        elif contingent:
            decay_factor = 1.0
        else:
            decay_factor = (1 - trait_decay) ** r

        steps = int(rng.integers(150, 260))
        interv = float(np.clip(base_interv * decay_factor
                               + rng.normal(0, 0.035), 0.0, 1.0))
        engage = float(np.clip(base_engage * decay_factor
                               + rng.normal(0, 0.035), 0.0, 1.0))
        human_steps = int(steps * interv)
        engaged_steps = int(steps * engage)
        idle = max(0, human_steps - engaged_steps)
        keys = int(steps * base_keys * decay_factor * rng.normal(1.0, 0.12))
        # Latency to intervene grows as the operator disengages.
        lat = float(np.clip(1.5 / max(decay_factor, 0.05)
                            * rng.normal(1.0, 0.15), 0.3, 60.0))
        # Success improves only when corrections actually land.
        if contingent:
            succ_p = float(np.clip(0.15 + 0.09 * r + rng.normal(0, 0.06), 0, 1))
        else:
            succ_p = float(np.clip(0.12 + rng.normal(0, 0.05), 0, 1))
        n_roll = int(cfg.study.rollouts_per_round)
        n_succ = int(rng.random(n_roll).__lt__(succ_p).sum())
        giveups = int(rng.random() < (0.30 * (1 - decay_factor)))

        m = RoundMetrics(
            round_index=r, phase=plan.phase, condition=plan.condition,
            contingent=contingent, n_rollouts=n_roll, n_success=n_succ,
            steps_total=steps, human_steps=human_steps,
            engaged_steps=engaged_steps, keypresses=max(keys, 0),
            n_interventions=max(1, int(rng.integers(1, 6) * decay_factor + 0.5)),
            first_intervention_step=int(lat * 20),
            time_to_first_intervention=lat, idle_steps=idle,
            giveup_events=giveups, episode_seconds=steps / 20.0,
        )
        # Self-report tracks perceived contingency.
        scale = decay_factor if not contingent else 1.0
        m.survey = {
            "control": int(np.clip(round(1 + 6 * scale + rng.normal(0, 0.6)), 1, 7)),
            "effectiveness": int(np.clip(round(1 + 6 * scale + rng.normal(0, 0.7)), 1, 7)),
            "expect_success": int(np.clip(round(1 + 6 * scale + rng.normal(0, 0.8)), 1, 7)),
            "frustration": int(np.clip(round(7 - 5 * scale + rng.normal(0, 0.8)), 1, 7)),
            "persistence": int(np.clip(round(1 + 6 * scale + rng.normal(0, 0.7)), 1, 7)),
            "effort": int(np.clip(round(1 + 6 * decay_factor + rng.normal(0, 0.6)), 1, 7)),
        }
        m.train_report = {"steps": 300, "committed": contingent,
                          "seconds": 8.0, "num_samples": 200 + 60 * r}
        ctrl.record(m)
        logger.log_round(m)

    summary = ctrl.summary()
    summary["simulated"] = True
    logger.write_summary(summary)
    return sdir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sim_sessions")
    ap.add_argument("--n-per-condition", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--decay", type=float, default=0.18,
                    help="per-round effort decay under non-contingency")
    ap.add_argument("--conditions", nargs="+",
                    default=["responsive", "noncontingent"])
    ap.add_argument("--two-phase", action="store_true",
                    help="also simulate an escape-test group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    made = 0
    for cond in args.conditions:
        for i in range(args.n_per_condition):
            simulate_participant(f"sim{i:03d}", cond, args.rounds, rng,
                                 args.decay, out_root=args.out)
            made += 1
    if args.two_phase:
        for i in range(args.n_per_condition):
            simulate_participant(f"esc{i:03d}", "noncontingent", args.rounds, rng,
                                 args.decay, two_phase=True,
                                 phase1=args.rounds // 2,
                                 phase2_condition="responsive", out_root=args.out)
            made += 1
    print(f"simulated {made} sessions -> {args.out}")
    print(f"now run: python scripts/analyze.py --sessions {args.out} --plot sim.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
