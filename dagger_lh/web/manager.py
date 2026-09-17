"""Session lifecycle and randomised condition assignment.

One `multiprocessing.Process` per participant (see `dagger_lh.web.__init__` for
why threads are not an option). The manager owns spawning, capacity limits,
reconnection and reaping; the FastAPI layer only relays bytes.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import Config
from ..study import Condition, validate_condition
from .worker import run_participant


@dataclass
class Arm:
    """One cell of the design: a condition, optionally the two-phase variant."""
    name: str
    condition: str
    two_phase: bool = False
    phase2_condition: str = Condition.RESPONSIVE
    weight: float = 1.0

    def apply(self, cfg: Config) -> Config:
        over = {
            "study.condition": self.condition,
            "study.two_phase": self.two_phase,
        }
        if self.two_phase:
            over["study.phase2_condition"] = self.phase2_condition
        return cfg.override(over)


#: Default design: contingent control, non-contingent induction, and the
#: two-phase escape test. The escape arm is the one that distinguishes learned
#: helplessness from rational disengagement, so it is in by default.
DEFAULT_ARMS: List[Arm] = [
    Arm("responsive", Condition.RESPONSIVE),
    Arm("noncontingent", Condition.NONCONTINGENT),
    Arm("escape", Condition.NONCONTINGENT, two_phase=True,
        phase2_condition=Condition.RESPONSIVE),
]


class Assigner:
    """Balanced random assignment that survives restarts.

    Counts completed and in-flight assignments per arm and hands out whichever is
    furthest behind, breaking ties at random. Persisted to disk so a server
    restart mid-study does not silently unbalance the design.
    """

    def __init__(self, arms: List[Arm], state_path: str, seed: int = 0):
        self.arms = arms
        self.state_path = state_path
        self.rng = secrets.SystemRandom() if seed == 0 else __import__(
            "random").Random(seed)
        self.counts: Dict[str, int] = {a.name: 0 for a in arms}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    saved = json.load(f)
                for k, v in (saved.get("counts") or {}).items():
                    if k in self.counts:
                        self.counts[k] = int(v)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)) or ".",
                    exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"counts": self.counts, "updated": time.time()}, f, indent=2)
        os.replace(tmp, self.state_path)  # atomic, so a crash cannot truncate it

    def assign(self, forced: Optional[str] = None) -> Arm:
        with self._lock:
            if forced:
                for a in self.arms:
                    if a.name == forced:
                        self.counts[a.name] += 1
                        self._save()
                        return a
                raise KeyError(f"unknown arm '{forced}'")
            # weight-adjusted deficit, so unequal allocation ratios work too
            deficits = [(self.counts[a.name] / max(a.weight, 1e-9), a)
                        for a in self.arms]
            lowest = min(d for d, _ in deficits)
            candidates = [a for d, a in deficits if d == lowest]
            arm = self.rng.choice(candidates)
            self.counts[arm.name] += 1
            self._save()
            return arm

    def release(self, arm_name: str) -> None:
        """Give an assignment back (participant declined consent / screened out)."""
        with self._lock:
            if self.counts.get(arm_name, 0) > 0:
                self.counts[arm_name] -= 1
                self._save()

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.counts)


@dataclass
class Session:
    sid: str
    participant_id: str
    arm: Arm
    proc: Any = None
    in_q: Any = None
    out_q: Any = None
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    connected: bool = False
    ever_connected: bool = False
    finished: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)
    session_dir: Optional[str] = None
    recruitment: Dict[str, Any] = field(default_factory=dict)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def public(self) -> Dict[str, Any]:
        return {
            "sid": self.sid,
            "participant_id": self.participant_id,
            # Arm is deliberately omitted from anything the client can read.
            "created": self.created,
            "age_seconds": round(time.time() - self.created, 1),
            "connected": self.connected,
            "alive": self.alive(),
            "finished": self.finished,
            "last_event": self.events[-1]["event"] if self.events else None,
        }


class SessionManager:
    def __init__(self, cfg: Config, arms: Optional[List[Arm]] = None,
                 max_sessions: int = 8, options: Optional[Dict[str, Any]] = None,
                 state_dir: str = "data/web"):
        self.cfg = cfg
        self.arms = arms or DEFAULT_ARMS
        for a in self.arms:
            validate_condition(a.condition)
            if a.two_phase:
                validate_condition(a.phase2_condition)
        self.max_sessions = int(max_sessions)
        self.options = dict(options or {})
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.assigner = Assigner(self.arms,
                                 os.path.join(state_dir, "assignment.json"))
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")  # required: fork + GL/torch is unsafe

    # ------------------------------------------------------------------ capacity
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for s in self.sessions.values()
                       if s.alive() and not s.finished)

    def has_capacity(self) -> bool:
        return self.active_count() < self.max_sessions

    # ------------------------------------------------------------------ lifecycle
    def create(self, participant_id: Optional[str] = None,
               forced_arm: Optional[str] = None,
               recruitment: Optional[Dict[str, Any]] = None) -> Session:
        if not self.has_capacity():
            raise RuntimeError("server at capacity")
        self.reap()
        sid = secrets.token_urlsafe(18)
        pid = participant_id or f"p_{secrets.token_hex(5)}"
        arm = self.assigner.assign(forced_arm)
        cfg = arm.apply(self.cfg).override({"study.participant_id": pid})

        in_q = self._ctx.Queue(maxsize=256)
        out_q = self._ctx.Queue(maxsize=256)
        opts = dict(self.options)
        opts["recruitment"] = dict(recruitment or {})
        proc = self._ctx.Process(
            target=run_participant,
            args=(cfg.to_dict(), in_q, out_q, opts),
            daemon=True,
            name=f"participant-{pid}",
        )
        proc.start()
        sess = Session(sid=sid, participant_id=pid, arm=arm, proc=proc,
                       in_q=in_q, out_q=out_q,
                       recruitment=dict(recruitment or {}))
        with self._lock:
            self.sessions[sid] = sess
        return sess

    def get(self, sid: str) -> Optional[Session]:
        with self._lock:
            return self.sessions.get(sid)

    def note_event(self, sess: Session, payload: Dict[str, Any]) -> None:
        ev = payload.get("event", "?")
        sess.events.append({"event": ev, "t": time.time(),
                            "data": payload.get("data")})
        if ev == "session_dir":
            sess.session_dir = (payload.get("data") or {}).get("dir")
        if ev in ("finished", "worker_exit", "error"):
            sess.finished = True
        if ev in ("consent_declined", "screened_out"):
            # Hand the assignment back so the arm is not left short.
            self.assigner.release(sess.arm.name)

    def stop(self, sid: str) -> bool:
        sess = self.get(sid)
        if sess is None:
            return False
        try:
            sess.in_q.put_nowait(("disconnect", {}))
        except (queue.Full, ValueError, OSError):
            pass
        if sess.alive():
            sess.proc.terminate()
            sess.proc.join(timeout=5)
        sess.finished = True
        return True

    def reap(self, max_age: float = 4 * 3600, idle_grace: float = 600.0) -> int:
        """Clear out dead, finished, or abandoned sessions."""
        now = time.time()
        removed = 0
        with self._lock:
            for sid, s in list(self.sessions.items()):
                stale = (not s.alive()) or (now - s.created > max_age)
                abandoned = (not s.connected) and (now - s.last_seen > idle_grace)
                if stale or abandoned:
                    if s.alive():
                        try:
                            s.proc.terminate()
                            s.proc.join(timeout=3)
                        except Exception:
                            pass
                    self.sessions.pop(sid, None)
                    removed += 1
        return removed

    def shutdown(self) -> None:
        for sid in list(self.sessions):
            self.stop(sid)

    def status(self) -> Dict[str, Any]:
        self.reap()
        with self._lock:
            sess = [s.public() for s in self.sessions.values()]
        return {
            "active": self.active_count(),
            "capacity": self.max_sessions,
            "assignment_counts": self.assigner.snapshot(),
            "arms": [{"name": a.name, "condition": a.condition,
                      "two_phase": a.two_phase} for a in self.arms],
            "sessions": sess,
        }
