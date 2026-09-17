"""Learned-helplessness study apparatus: conditions, measures, logging, surveys.

Theory of operation
-------------------
Learned helplessness (Seligman & Maier) is induced by *non-contingency*: outcomes
stop depending on what the subject does. Applied to interactive imitation learning,
the subject is a teleoperator teaching a policy via DAgger, and the manipulated
variable is whether their corrections actually improve the policy.

Conditions (`Condition`):
  responsive     corrections drive a real LoRA update; the policy improves.
  noncontingent  the update runs (same wall-clock, same compute) and is then
                 discarded. Effort and outcome are decoupled -- the induction arm.
  degraded       as above, plus injected action noise that grows each round, so
                 the policy visibly gets *worse* however hard the operator works.
  yoked          the policy follows a prerecorded improvement schedule that is
                 independent of this operator's corrections.

`two_phase` runs the classic escape test: induce under `condition` for
`phase1_rounds`, then restore controllability (`phase2_condition`, normally
`responsive`) for `phase2_rounds`. Helplessness shows up as failure to exploit
the restored contingency -- effort and intervention stay suppressed relative to a
control group that had `responsive` throughout.

Dependent measures (`RoundMetrics`) are the standard helplessness indicators
translated to teleoperation: intervention rate, latency to first intervention,
keypress effort, passive/idle fraction, explicit give-up events, and self-reported
perceived control. `helplessness_index` aggregates them.

Nothing here reveals the condition to the operator: the HUD text comes from
`progress_feedback`, which is deliberately identical in structure across arms.
"""
from __future__ import annotations

import json
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, StudyConfig


class Condition:
    RESPONSIVE = "responsive"
    NONCONTINGENT = "noncontingent"
    DEGRADED = "degraded"
    YOKED = "yoked"
    ALL = (RESPONSIVE, NONCONTINGENT, DEGRADED, YOKED)


def validate_condition(name: str) -> str:
    if name not in Condition.ALL:
        raise ValueError(f"unknown condition '{name}'; choose from {Condition.ALL}")
    return name


# ============================================================== per-round measures

@dataclass
class RoundMetrics:
    """Behavioural measures for one DAgger round (one or more rollouts)."""
    round_index: int = 0
    phase: int = 1
    condition: str = Condition.RESPONSIVE
    contingent: bool = True

    # --- task outcome ---
    n_rollouts: int = 0
    n_success: int = 0
    steps_total: int = 0

    # --- effort / engagement (the helplessness signal) ---
    human_steps: int = 0              # steps where the human held authority
    engaged_steps: int = 0            # steps where the human actually pressed keys
    keypresses: int = 0
    n_interventions: int = 0          # discrete take-over episodes
    first_intervention_step: Optional[int] = None
    time_to_first_intervention: Optional[float] = None
    idle_steps: int = 0               # in control but pressing nothing
    giveup_events: int = 0            # explicit "X" presses
    discard_events: int = 0
    episode_seconds: float = 0.0

    # --- self report ---
    survey: Dict[str, Any] = field(default_factory=dict)

    # --- training bookkeeping ---
    train_report: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    @property
    def success_rate(self) -> float:
        return self.n_success / max(self.n_rollouts, 1)

    @property
    def intervention_rate(self) -> float:
        """Fraction of environment steps the human took over."""
        return self.human_steps / max(self.steps_total, 1)

    @property
    def engagement_rate(self) -> float:
        """Fraction of steps with actual key input."""
        return self.engaged_steps / max(self.steps_total, 1)

    @property
    def passivity(self) -> float:
        """Fraction of *held-authority* steps spent doing nothing."""
        return self.idle_steps / max(self.human_steps, 1)

    @property
    def keypresses_per_step(self) -> float:
        return self.keypresses / max(self.steps_total, 1)

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("survey", None)
        row.pop("train_report", None)
        row.update(
            success_rate=self.success_rate,
            intervention_rate=self.intervention_rate,
            engagement_rate=self.engagement_rate,
            passivity=self.passivity,
            keypresses_per_step=self.keypresses_per_step,
        )
        for k, v in (self.survey or {}).items():
            row[f"survey_{k}"] = v
        for k, v in (self.train_report or {}).items():
            row[f"train_{k}"] = v
        return row


def helplessness_index(rounds: Sequence[RoundMetrics],
                       baseline_rounds: int = 2) -> Dict[str, float]:
    """Effort decline from the start of a session to the end.

    `baseline_rounds` sizes the early and late comparison windows; it shrinks
    automatically on short sessions so the windows never overlap.

    Raw deltas are reported per measure, but `index` is deliberately a mean of
    *relative* declines in the three effort measures (intervention rate,
    engagement rate, keypresses per step) rather than a mean of the raw deltas --
    those are in incommensurable units, so an unnormalised average would let
    keypresses-per-step swamp two rates bounded in [0, 1].

    index is in [-1, 1]:  1 = effort ceased entirely, 0 = effort held steady,
    negative = the operator worked *harder* by the end. Latency, passivity and
    give-ups rise with helplessness and are reported alongside rather than folded
    in, because they can dissociate from effort (an operator may keep taking
    control while doing nothing once there).
    """
    if len(rounds) < 2:
        return {}
    w = max(1, min(int(baseline_rounds), len(rounds) // 2))
    early = rounds[:w]
    late = rounds[-w:]

    def mean(rs, f):
        vals = [f(r) for r in rs]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def rel_decline(f) -> float:
        e, l = mean(early, f), mean(late, f)
        if not (np.isfinite(e) and np.isfinite(l)) or e <= 1e-9:
            return float("nan")
        return float(np.clip((e - l) / e, -1.0, 1.0))

    f_interv = lambda r: r.intervention_rate      # noqa: E731
    f_engage = lambda r: r.engagement_rate        # noqa: E731
    f_keys = lambda r: r.keypresses_per_step      # noqa: E731
    f_passive = lambda r: r.passivity             # noqa: E731
    f_lat = lambda r: r.time_to_first_intervention  # noqa: E731

    declines = [rel_decline(f) for f in (f_interv, f_engage, f_keys)]
    declines = [d for d in declines if np.isfinite(d)]

    out = {
        "delta_intervention_rate": mean(late, f_interv) - mean(early, f_interv),
        "delta_engagement_rate": mean(late, f_engage) - mean(early, f_engage),
        "delta_keypresses_per_step": mean(late, f_keys) - mean(early, f_keys),
        "delta_passivity": mean(late, f_passive) - mean(early, f_passive),
        "delta_latency_to_intervene": mean(late, f_lat) - mean(early, f_lat),
        "rel_decline_intervention": rel_decline(f_interv),
        "rel_decline_engagement": rel_decline(f_engage),
        "rel_decline_keypresses": rel_decline(f_keys),
        "total_giveups": float(sum(r.giveup_events for r in rounds)),
        "baseline_window": float(w),
        "index": float(np.mean(declines)) if declines else float("nan"),
    }
    return out


# ============================================================== condition control

@dataclass
class RoundPlan:
    """What the runner should do for one round, given the condition schedule."""
    round_index: int
    phase: int
    condition: str
    commit_update: bool       # False -> sham update (runs, then rolled back)
    action_noise_std: float   # injected policy-action noise
    reason: str = ""
    #: Under `yoked`, the success rate the policy should exhibit this round. The
    #: runner swaps in the nearest ladder rung, so competence follows a curve
    #: that is independent of the operator's corrections.
    ladder_target: Optional[float] = None


class StudyController:
    """Owns the condition schedule and the operator-facing progress feedback."""

    def __init__(self, cfg: StudyConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        validate_condition(cfg.condition)
        if cfg.two_phase:
            validate_condition(cfg.phase2_condition)
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.rounds: List[RoundMetrics] = []

    # ------------------------------------------------------------------ schedule
    @property
    def total_rounds(self) -> int:
        if self.cfg.two_phase:
            return int(self.cfg.phase1_rounds) + int(self.cfg.phase2_rounds)
        return int(self.cfg.num_rounds)

    def phase_of(self, round_index: int) -> int:
        if not self.cfg.two_phase:
            return 1
        return 1 if round_index < int(self.cfg.phase1_rounds) else 2

    def condition_of(self, round_index: int) -> str:
        phase = self.phase_of(round_index)
        return self.cfg.condition if phase == 1 else self.cfg.phase2_condition

    def plan_round(self, round_index: int) -> RoundPlan:
        phase = self.phase_of(round_index)
        cond = self.condition_of(round_index)
        # Rounds elapsed *within* the current condition, so noise restarts when
        # the escape phase begins.
        local = round_index - (0 if phase == 1 else int(self.cfg.phase1_rounds))

        if cond == Condition.RESPONSIVE:
            return RoundPlan(round_index, phase, cond, True, 0.0,
                             "corrections applied via LoRA update")
        if cond == Condition.NONCONTINGENT:
            return RoundPlan(round_index, phase, cond, False, 0.0,
                             "update computed then discarded (non-contingent)")
        if cond == Condition.DEGRADED:
            noise = float(self.cfg.degrade_noise_std) + local * float(self.cfg.degrade_growth)
            return RoundPlan(round_index, phase, cond, False, noise,
                             f"update discarded + action noise std={noise:.3f}")
        if cond == Condition.YOKED:
            sched = self.yoked_schedule()
            target = sched[min(round_index, len(sched) - 1)]
            return RoundPlan(round_index, phase, cond, False, 0.0,
                             f"scripted competence {target:.0%} from the ladder",
                             ladder_target=target)
        raise ValueError(cond)

    def yoked_schedule(self) -> List[float]:
        """Success rate per round for the scripted condition.

        Defaults to a flat curve at `sham_success_cap` -- i.e. a policy that never
        improves, matching what a non-contingent participant experiences but
        produced by construction rather than by discarding updates.
        """
        from .ladder import resolve_schedule
        sched = list(self.cfg.yoked_schedule)
        if not sched:
            sched = [float(self.cfg.sham_success_cap)]
        return resolve_schedule(sched, self.total_rounds)

    def is_contingent(self, round_index: int) -> bool:
        return self.condition_of(round_index) == Condition.RESPONSIVE

    # ------------------------------------------------------------------ feedback
    def progress_feedback(self, round_index: int,
                          measured_success: float,
                          train_report: Optional[Dict[str, Any]] = None) -> str:
        """Text shown to the operator between rounds.

        Structurally identical across arms so the *format* cannot cue condition.
        Under non-contingent arms the reported figure is capped, matching what the
        operator actually observes (a policy that is not improving).
        """
        if not self.cfg.show_progress_feedback:
            return ""
        cond = self.condition_of(round_index)
        pct = 100.0 * measured_success
        if cond != Condition.RESPONSIVE:
            pct = min(pct, 100.0 * float(self.cfg.sham_success_cap))
        n = len(self.rounds)
        prev = (100.0 * self.rounds[-1].success_rate) if n else 0.0
        arrow = "=" if abs(pct - prev) < 1e-6 else ("up" if pct > prev else "down")
        return (f"round {round_index + 1}/{self.total_rounds}\n"
                f"success this round: {pct:.0f}%\n"
                f"change vs last: {arrow}")

    # ------------------------------------------------------------------ recording
    def record(self, metrics: RoundMetrics) -> None:
        self.rounds.append(metrics)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "participant_id": self.cfg.participant_id,
            "condition": self.cfg.condition,
            "two_phase": self.cfg.two_phase,
            "phase2_condition": self.cfg.phase2_condition if self.cfg.two_phase else None,
            "n_rounds": len(self.rounds),
        }
        out.update(helplessness_index(self.rounds))
        if self.cfg.two_phase:
            p1 = [r for r in self.rounds if r.phase == 1]
            p2 = [r for r in self.rounds if r.phase == 2]
            for tag, rs in (("phase1", p1), ("phase2", p2)):
                if rs:
                    out[f"{tag}_intervention_rate"] = float(
                        np.mean([r.intervention_rate for r in rs]))
                    out[f"{tag}_engagement_rate"] = float(
                        np.mean([r.engagement_rate for r in rs]))
                    out[f"{tag}_success_rate"] = float(
                        np.mean([r.success_rate for r in rs]))
                    out[f"{tag}_giveups"] = int(sum(r.giveup_events for r in rs))
            if p1 and p2:
                # The escape test: does effort recover once control is restored?
                out["escape_recovery_intervention"] = (
                    out["phase2_intervention_rate"] - out["phase1_intervention_rate"])
                out["escape_recovery_engagement"] = (
                    out["phase2_engagement_rate"] - out["phase1_engagement_rate"])
        return out


# ============================================================== event logging

class SessionLogger:
    """Append-only JSONL event log plus a per-round CSV.

    JSONL keeps the full fidelity (every step of every episode if asked); the CSV
    is the analysis-ready per-round table.
    """

    def __init__(self, session_dir: str, config: Config,
                 session_id: Optional[str] = None):
        os.makedirs(session_dir, exist_ok=True)
        self.dir = session_dir
        self.session_id = session_id or f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.events_path = os.path.join(session_dir, "events.jsonl")
        self.rounds_path = os.path.join(session_dir, "rounds.csv")
        self.steps_path = os.path.join(session_dir, "steps.jsonl")
        self._round_header: Optional[List[str]] = None
        config.save(os.path.join(session_dir, "config.yaml"))
        self.log("session_start", {
            "session_id": self.session_id,
            "participant_id": config.study.participant_id,
            "condition": config.study.condition,
            "two_phase": config.study.two_phase,
            "mode": config.mode,
            "algo": config.policy.algo,
            "env": config.env.env_name,
            "platform": platform.platform(),
            "python": platform.python_version(),
        })

    # ------------------------------------------------------------------ writing
    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec = {"t": time.time(), "session_id": self.session_id, "event": event}
        rec.update(payload or {})
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def log_step(self, payload: Dict[str, Any]) -> None:
        """High-frequency per-step trace (kept in a separate file)."""
        with open(self.steps_path, "a") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")

    def log_round(self, metrics: RoundMetrics) -> None:
        row = metrics.as_row()
        self.log("round_complete", row)
        write_header = self._round_header is None and not os.path.exists(self.rounds_path)
        if self._round_header is None:
            self._round_header = list(row.keys())
        with open(self.rounds_path, "a") as f:
            if write_header:
                f.write(",".join(self._round_header) + "\n")
            f.write(",".join(_csv_cell(row.get(k)) for k in self._round_header) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> str:
        path = os.path.join(self.dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        self.log("session_end", summary)
        return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v)
    return f'"{s}"' if ("," in s or '"' in s) else s


# ============================================================== surveys

@dataclass
class SurveyItem:
    key: str
    prompt: str
    low: str
    high: str
    scale: int = 7


# Items chosen to separate the constructs helplessness theory distinguishes:
# contingency belief (control), outcome expectancy (expect_success), affect
# (frustration), and behavioural intention (persistence).
DEFAULT_SURVEY: List[SurveyItem] = [
    SurveyItem("control", "How much control did you have over the robot's learning?",
               "no control", "complete control"),
    SurveyItem("effectiveness", "How effective were your corrections?",
               "not at all", "extremely"),
    SurveyItem("expect_success", "How likely is the robot to succeed next round?",
               "very unlikely", "very likely"),
    SurveyItem("frustration", "How frustrated do you feel?",
               "not at all", "extremely"),
    SurveyItem("persistence", "How willing are you to keep teaching?",
               "not at all", "very willing"),
    SurveyItem("effort", "How much effort did you just put in?",
               "none", "maximum"),
]


class SurveyRunner:
    """Collects Likert responses through the pygame display."""

    def __init__(self, display, items: Optional[List[SurveyItem]] = None):
        self.display = display
        self.items = items or DEFAULT_SURVEY

    def run(self, header: str = "") -> Dict[str, int]:
        import pygame
        answers: Dict[str, int] = {}
        for item in self.items:
            keys = [getattr(pygame, f"K_{i}") for i in range(1, item.scale + 1)]
            body = (f"{item.prompt}\n\n"
                    f"1 = {item.low}     {item.scale} = {item.high}\n")
            while True:
                self.display.message(
                    header or "Quick check-in", body,
                    sub=f"press 1-{item.scale}")
                k = self.display.wait_for_key(keys + [pygame.K_ESCAPE])
                if k == pygame.K_ESCAPE:
                    return answers
                if k in keys:
                    answers[item.key] = keys.index(k) + 1
                    break
        return answers
