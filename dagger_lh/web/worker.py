"""The per-participant worker process.

Runs one participant's entire session: robosuite, the policy, LoRA finetuning,
condition scheduling and logging. Communicates with the web server through two
multiprocessing queues and never touches the network itself.

The study logic is *not* reimplemented here. `RemoteDisplay`, `RemoteTeleop` and
`RemoteSurvey` implement the same interfaces the local pygame front-end does, and
are injected into the same `SessionRunner` the desktop version uses. The online
and in-lab studies therefore run identical code paths for conditions, sham
updates, metrics and logging -- they cannot drift apart, and `scripts/analyze.py`
reads both without changes.
"""
from __future__ import annotations

import os
import queue
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from ..config import Config
from ..study import DEFAULT_SURVEY, SurveyItem
from ..teleop import BaseTeleop, KEYMAP_HELP
from . import protocol as P

# Queue depth for outbound video. Small on purpose: if the client cannot keep up
# we want to drop stale frames rather than build a latency backlog.
FRAME_QUEUE_MAX = 3


class Disconnected(Exception):
    """The participant's socket went away and did not come back in time."""


class RemoteDisplay:
    """Stands in for `OperatorDisplay`, emitting frames/screens over a queue."""

    wants_frames = True

    def __init__(self, out_q, in_q, cfg: Config, jpeg_quality: int = 70,
                 frame_size: int = 384, response_timeout: float = 900.0):
        self.out_q = out_q
        self.in_q = in_q
        self.cfg = cfg
        # The local front-end paces the episode loop with pygame's clock. Nothing
        # here does, so without an explicit limiter the loop runs as fast as the
        # simulator can step -- measured at ~63 Hz against a 20 Hz control rate,
        # i.e. a robot moving ~3x real time that no participant could track.
        self.target_dt = 1.0 / max(1, int(cfg.env.control_freq))
        self._next_frame = 0.0
        self.late_frames = 0
        self.jpeg_quality = int(jpeg_quality)
        self.frame_size = int(frame_size)
        self.response_timeout = float(response_timeout)
        self.show_help = True
        self.clock = None
        # Kept for interface parity with the pygame display; the browser sends
        # semantic responses, so these are only sentinels.
        self.K_ESCAPE, self.K_RETURN = 27, 13
        self.latest_input: Dict[str, Any] = {"keys": [], "press": []}
        self.focused = True
        self.blur_events = 0
        self._last_screen: Optional[Dict[str, Any]] = None
        # The full outbound screen message, replayed when a socket reconnects.
        # Without this a participant who refreshes -- or who leaves for a
        # Qualtrics questionnaire and comes back -- reconnects to a blank page and
        # waits until the next screen change, which may never come because the
        # worker is blocked waiting for their answer.
        self._last_screen_msg: Optional[Dict[str, Any]] = None
        self.reconnects = 0
        # Every screen carries a sequence number that the client echoes back.
        # Replays reuse the same number, so answering a replayed screen twice is
        # idempotent and an answer to a screen we have moved past is discarded.
        # Without this, a reconnect replay produces a duplicate response that
        # satisfies the *next* wait early and silently skips a screen.
        self._screen_seq = 0

    # ------------------------------------------------------------------ outbound
    def _send(self, msg: Dict[str, Any]) -> None:
        if msg.get("t") == P.MSG_SCREEN:
            self._screen_seq += 1
            msg = dict(msg)
            msg["seq"] = self._screen_seq
            self._last_screen_msg = msg
        self.out_q.put(("json", msg))

    def _replay(self) -> None:
        if self._last_screen_msg is not None:
            self.reconnects += 1
            self.out_q.put(("json", self._last_screen_msg))
            self.reset_pacing()

    def draw(self, main_img: np.ndarray, hud: Dict[str, Any],
             side_imgs: Optional[List[np.ndarray]] = None) -> None:
        import cv2
        if self._last_screen != {"kind": P.SCREEN_TELEOP}:
            self._send(P.screen(P.SCREEN_TELEOP, keymap=KEYMAP_HELP))
            self._last_screen = {"kind": P.SCREEN_TELEOP}
        img = main_img
        if img.shape[0] != self.frame_size:
            img = cv2.resize(img, (self.frame_size, self.frame_size),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        side_jpeg = None
        if side_imgs:
            s = side_imgs[0]
            ok2, sbuf = cv2.imencode(".jpg", s[:, :, ::-1],
                                     [int(cv2.IMWRITE_JPEG_QUALITY),
                                      max(35, self.jpeg_quality - 15)])
            if ok2:
                side_jpeg = sbuf.tobytes()
        payload = P.encode_frame(buf.tobytes(), {"hud": _json_safe(hud),
                                                 "side": side_jpeg is not None})
        self.out_q.put(("frame", payload, side_jpeg))
        self._pace()

    def _pace(self) -> None:
        """Hold the episode loop to the environment's control rate.

        Keeps simulated time roughly equal to wall-clock time, so the robot moves
        at a natural speed and a key held for one second produces one second of
        motion -- which is what the demonstrations the policy learned from encode.
        """
        now = time.perf_counter()
        if self._next_frame == 0.0:
            self._next_frame = now + self.target_dt
            return
        sleep_for = self._next_frame - now
        if sleep_for > 0:
            time.sleep(sleep_for)
            self._next_frame += self.target_dt
        else:
            # Behind schedule (slow server or a long policy forward pass): resync
            # rather than accumulate debt and then sprint to catch up.
            self.late_frames += 1
            self._next_frame = time.perf_counter() + self.target_dt

    def reset_pacing(self) -> None:
        self._next_frame = 0.0

    def message(self, title: str, body: str = "",
                sub: str = "press ENTER to continue") -> None:
        scr = {"kind": P.SCREEN_MESSAGE, "title": title, "body": body}
        if scr == self._last_screen:
            return  # the runner redraws the same modal in a wait loop
        self._last_screen = scr
        self.reset_pacing()
        self._send(P.screen(P.SCREEN_MESSAGE, title=title, body=body,
                            cta=sub or "Continue"))

    # ------------------------------------------------------------------ inbound
    def pump(self) -> None:
        """Drain pending client messages without blocking."""
        while True:
            try:
                kind, data = self.in_q.get_nowait()
            except queue.Empty:
                return
            self._absorb(kind, data)

    def _absorb(self, kind: str, data: Dict[str, Any]) -> None:
        if kind == P.MSG_INPUT:
            keys = P.sanitize_keys(data.get("keys"))
            press = P.sanitize_keys(data.get("press"))
            self.latest_input["keys"] = keys
            # Edge presses must not be lost between polls, so accumulate them.
            self.latest_input["press"].extend(press)
        elif kind == P.MSG_FOCUS:
            was = self.focused
            self.focused = bool(data.get("focused", True))
            if was and not self.focused:
                self.blur_events += 1
        elif kind == P.MSG_RESEND:
            self._replay()
        elif kind == "disconnect":
            raise Disconnected()

    def wait_for_response(self, expect: str,
                          timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until the client answers the screen we are showing."""
        deadline = time.time() + (timeout or self.response_timeout)
        while time.time() < deadline:
            try:
                kind, data = self.in_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == P.MSG_RESPONSE:
                if expect is not None and data.get("screen") != expect:
                    continue
                seq = data.get("seq")
                if seq is not None and int(seq) != self._screen_seq:
                    # A stale answer: either a duplicate of a replayed screen we
                    # already consumed, or an answer that raced a screen change.
                    continue
                return dict(data.get("data") or {})
            self._absorb(kind, data)
        raise Disconnected(f"no response to '{expect}' within timeout")

    def wait_for_key(self, keys: Optional[Sequence[int]] = None) -> int:
        """Interface parity with the pygame display: 'advance' or 'quit'."""
        resp = self.wait_for_response(P.SCREEN_MESSAGE)
        return self.K_ESCAPE if resp.get("action") == "quit" else self.K_RETURN

    def redirect(self, url: str, title: str, body: str = "",
                 cta: str = "Continue") -> None:
        """Hand the tab to an external URL (a Qualtrics questionnaire).

        Qualtrics serves `X-Frame-Options: SAMEORIGIN`, so its surveys cannot be
        embedded in this page; a full-page handoff is the only option. The worker
        keeps running while the participant is away and replays its screen when
        they return.
        """
        self._send(P.screen(P.SCREEN_REDIRECT, url=url, title=title,
                            body=body, cta=cta))

    def close(self) -> None:
        self._last_screen = None


class RemoteTeleop(BaseTeleop):
    """Teleoperation driven by the browser's reported key state.

    Inherits every control semantic from `BaseTeleop`, so smoothing, gripper
    latching, authority toggling and the effort counters are bit-identical to the
    local keyboard device.
    """

    def __init__(self, cfg, action_dim: int, display: RemoteDisplay):
        super().__init__(cfg, action_dim)
        self.display = display

    def poll(self):
        self.display.pump()
        held: Set[str] = set(self.display.latest_input.get("keys") or [])
        presses = list(self.display.latest_input.get("press") or [])
        self.display.latest_input["press"] = []
        st = self._state_from(held, presses)
        # A participant who switches tabs is not "idle at the controls"; treat the
        # blurred window as no input so passivity is not inflated by tab-switching.
        if not self.display.focused:
            st.engaged = False
        return st


class RemoteSurvey:
    """Delivers survey items as a structured screen instead of key presses."""

    def __init__(self, display: RemoteDisplay,
                 items: Optional[List[SurveyItem]] = None,
                 attention_check_every: int = 0):
        self.display = display
        self.items = items or DEFAULT_SURVEY
        self.attention_check_every = int(attention_check_every)
        self._round = 0

    def run(self, header: str = "") -> Dict[str, Any]:
        self._round += 1
        items = [
            {"key": it.key, "prompt": it.prompt, "low": it.low,
             "high": it.high, "scale": it.scale}
            for it in self.items
        ]
        check = None
        if (self.attention_check_every
                and self._round % self.attention_check_every == 0):
            # Online samples need a verifiable item; a fixed correct answer lets
            # inattentive responding be detected without guessing from variance.
            check = {"key": "_attention", "prompt": "Attention check: please "
                     "select 2 for this item.", "low": "1", "high": "7",
                     "scale": 7, "expected": 2}
            items = items + [check]
        self.display._last_screen = {"kind": P.SCREEN_SURVEY, "n": self._round}
        self.display._send(P.screen(P.SCREEN_SURVEY,
                                    header=header or "Quick check-in",
                                    items=items))
        resp = self.display.wait_for_response(P.SCREEN_SURVEY)
        out: Dict[str, Any] = {}
        for it in items:
            v = resp.get(it["key"])
            if isinstance(v, (int, float)):
                out[it["key"]] = int(v)
        if check is not None and "_attention" in out:
            out["attention_passed"] = int(out.pop("_attention") == check["expected"])
        out["survey_seconds"] = float(resp.get("elapsed", 0.0))
        return out


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


# ======================================================================== flow

CONSENT_TEXT = """\
You are invited to take part in a study about teaching robots.

What you will do: you will control a simulated robot arm with your keyboard and
teach it to perform a manipulation task. The session takes about {minutes}
minutes. You will answer a few short questions between attempts.

Risks: none beyond ordinary computer use. The task can be frustrating at times.

Your data: we record your keyboard input, the robot's behaviour, and your
questionnaire answers. Records are stored under a random participant code, not
your name.

Withdrawing: you may stop at any time by closing the tab or pressing the Withdraw
button, without giving a reason and without penalty.

This study involves some information about the robot's learning that will be
clarified in a full explanation at the end of the session."""

DEBRIEF_DECEPTION = """\
Thank you for taking part. Now that the session is over we can explain the part
of the design that was not described up front.

This study is about how people respond when their teaching effort stops changing
the robot's behaviour. Participants were randomly assigned to different
conditions. In some conditions the robot's learning update was computed from your
corrections and then discarded, so the robot could not improve no matter how well
you taught it. The progress information you saw reflected that.

If you were in one of those conditions, the robot's failure to improve was not a
reflection of your teaching or your ability. It was fixed in advance by the
condition you were randomly assigned to.

Your corrections were all saved and are scientifically useful regardless of
condition. Nothing about your performance was judged.

You may ask us to delete your data at any point by quoting your participant code."""

DEBRIEF_PLAIN = """\
Thank you for taking part.

This study is about how people teach robots through demonstration and correction,
and how their teaching effort changes over repeated attempts. Your corrections
were used to update the robot's policy between attempts.

Your data is stored under your participant code, which you may quote if you would
like it deleted."""


def qualtrics_url(base: str, params: Dict[str, Any]) -> str:
    """Append join keys to a Qualtrics link as query-string parameters.

    Qualtrics reads these into Embedded Data fields declared in the survey flow
    (each field left with "Value will be set from Panel or URL"), which is how a
    response is linked back to the behavioural session. See docs/qualtrics.md.
    """
    from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
    parts = urlparse(base)
    existing = dict(parse_qsl(parts.query, keep_blank_values=True))
    existing.update({k: str(v) for k, v in params.items() if v not in (None, "")})
    return urlunparse(parts._replace(query=urlencode(existing)))


class QualtricsSurvey:
    """Per-round questionnaire delivered by handing the tab to Qualtrics.

    Only used when `--survey-mode qualtrics`. The default is the in-app survey:
    a full-page round trip to Qualtrics after every one of 8 rounds adds a lot of
    friction and breaks the round rhythm the study measures, so Qualtrics is
    normally better used for the intake and exit batteries instead.

    Returns no item scores -- the answers live in Qualtrics and are joined later
    on `participant_id` and `round` by scripts/merge_qualtrics.py -- but does
    record that the handoff happened and how long the participant was away.
    """

    def __init__(self, display: RemoteDisplay, base_url: str,
                 participant_id: str, external_id: str = "",
                 session_id: str = ""):
        self.display = display
        self.base_url = base_url
        self.participant_id = participant_id
        self.external_id = external_id
        self.session_id = session_id
        self._round = 0

    def run(self, header: str = "") -> Dict[str, Any]:
        self._round += 1
        url = qualtrics_url(self.base_url, {
            "participant_id": self.participant_id,
            "external_id": self.external_id,
            "session_id": self.session_id,
            "round": self._round,
            "stage": "round",
        })
        t0 = time.time()
        self.display.redirect(
            url,
            title=header or f"Questions after attempt {self._round}",
            body="A short questionnaire opens next. When you finish it you will "
                 "be brought back here automatically to continue.",
            cta="Open questionnaire")
        # The participant navigates away; the worker waits for the return trip.
        resp = self.display.wait_for_response(P.SCREEN_REDIRECT)
        return {
            "delivered_by": "qualtrics",
            "qualtrics_round": self._round,
            "away_seconds": round(time.time() - t0, 1),
            "returned": int(bool(resp.get("returned", True))),
        }


def _instructions_payload(cfg: Config, practice: bool) -> Dict[str, Any]:
    return {
        "task": cfg.env.env_name,
        "keymap": KEYMAP_HELP,
        "practice": practice,
        "rounds": cfg.study.num_rounds if not cfg.study.two_phase else (
            cfg.study.phase1_rounds + cfg.study.phase2_rounds),
        "hold_to_intervene": cfg.teleop.hold_to_intervene,
    }


def run_participant(cfg_dict: Dict[str, Any], in_q, out_q,
                    options: Dict[str, Any]) -> None:
    """Worker process entry point. Never raises into the parent."""
    import torch
    torch.set_num_threads(int(options.get("torch_threads", 1)))
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from ..envs import silence_robosuite_logs
    from ..policies import build_policy, load_policy, resolve_device
    from ..runner import SessionRunner, _Aborted
    from ..trainer import PolicyTrainer
    silence_robosuite_logs()

    cfg = Config.from_dict(cfg_dict)
    display = RemoteDisplay(
        out_q, in_q, cfg,
        jpeg_quality=int(options.get("jpeg_quality", 70)),
        frame_size=int(options.get("frame_size", 384)),
        response_timeout=float(options.get("response_timeout", 900.0)),
    )
    runner = None
    try:
        recruitment = dict(options.get("recruitment") or {})
        external_id = str(recruitment.get("external_id")
                          or recruitment.get("PROLIFIC_PID") or "")

        # ---------------- consent ----------------
        # Skipped when an intake questionnaire (e.g. Qualtrics) already took
        # consent; the participant should not be asked to consent twice.
        if options.get("skip_consent"):
            out_q.put(("event", {"event": "consent_external",
                                 "data": {"external_id": external_id}}))
        else:
            display._send(P.screen(
                P.SCREEN_CONSENT,
                text=CONSENT_TEXT.format(minutes=options.get("minutes", 25)),
                participant=cfg.study.participant_id,
            ))
            resp = display.wait_for_response(P.SCREEN_CONSENT)
            if not resp.get("consent"):
                display._send(P.screen(P.SCREEN_DONE, title="Session ended",
                                       body="You declined to take part. "
                                            "You may close this tab."))
                out_q.put(("event", {"event": "consent_declined"}))
                return
            out_q.put(("event", {"event": "consent_given",
                                 "data": {k: v for k, v in resp.items()
                                          if k != "consent"}}))

        # ---------------- build session ----------------
        # Build the env first so RemoteTeleop knows the action dimension, then
        # inject everything. Letting SessionRunner fall back to its defaults would
        # construct a pygame KeyboardDevice and OperatorDisplay -- pointless here,
        # and pygame has no business being initialised on a headless server.
        from ..envs import make_env
        env = make_env(cfg.env)
        teleop = RemoteTeleop(cfg.teleop, env.action_dim, display)
        round_survey_url = str(options.get("qualtrics_round_url") or "")
        if options.get("survey_mode") == "qualtrics" and round_survey_url:
            survey = QualtricsSurvey(display, round_survey_url,
                                     cfg.study.participant_id, external_id,
                                     session_id="")
        else:
            survey = RemoteSurvey(
                display,
                attention_check_every=int(options.get("attention_check_every", 3)),
            )
        runner = SessionRunner(cfg, display=display, teleop=teleop, env=env,
                               survey=survey)
        runner.logger.log("web_session", {
            "participant_id": cfg.study.participant_id,
            # `external_id` is the join key for a Qualtrics export: it is written
            # here and again into summary.json so scripts/merge_qualtrics.py can
            # match a questionnaire response to this behavioural session.
            "external_id": external_id,
            "condition": cfg.study.condition,
            "two_phase": cfg.study.two_phase,
            "survey_mode": options.get("survey_mode", "native"),
            "recruitment": recruitment,
        })
        out_q.put(("event", {"event": "session_dir",
                             "data": {"dir": runner.session_dir}}))

        device = resolve_device(cfg.train.device)
        if cfg.base_checkpoint and os.path.exists(cfg.base_checkpoint):
            policy, _ = load_policy(cfg.base_checkpoint, device, cfg.policy)
        else:
            policy = build_policy(runner.env.obs_shapes(), runner.env.action_dim,
                                  cfg.policy, cfg.env.low_dim_keys,
                                  cfg.env.image_keys, device)
        trainer = PolicyTrainer(policy, cfg.train, cfg.policy, device)
        trainer.enable_lora()

        # ---------------- instructions ----------------
        display._send(P.screen(P.SCREEN_INSTRUCTIONS,
                               **_instructions_payload(cfg, practice=True)))
        if display.wait_for_response(P.SCREEN_INSTRUCTIONS).get("action") == "quit":
            raise _Aborted()

        # ---------------- practice / qualification ----------------
        #
        # Online participants face keyboard latency an in-lab participant does
        # not, and the task is genuinely hard. Without a qualification gate a
        # participant who never could do the task is indistinguishable from one
        # who gave up because they were made helpless -- which is precisely the
        # measure the study depends on.
        n_practice = int(options.get("practice_episodes", 2))
        required = int(options.get("practice_required", 1))
        passed = 0
        if n_practice:
            display.message(
                "Practice",
                f"First, try the task yourself with full control.\n"
                f"Complete it {required} time(s) to continue.\n\n"
                f"You have {n_practice} attempt(s).",
                sub="Start practice")
            display.wait_for_key()
            for i in range(n_practice):
                res = runner.run_episode(policy=None, round_index=i,
                                         hud_extra={"phase": "practice"})
                runner.logger.log("practice_episode", {
                    "attempt": i, "success": res.success, "steps": res.steps,
                    "seconds": res.seconds, "keypresses": res.keypresses,
                    "gave_up": res.gave_up,
                })
                passed += int(res.success)
                if res.quit:
                    raise _Aborted()
                if passed >= required:
                    break
                if i < n_practice - 1:
                    display.message("Not quite",
                                    "That attempt did not complete the task.\n"
                                    "Let's try once more.", sub="Try again")
                    display.wait_for_key()
            if passed < required:
                runner.logger.log("screened_out", {"practice_successes": passed})
                out_q.put(("event", {"event": "screened_out"}))
                display._send(P.screen(
                    P.SCREEN_SCREENOUT,
                    title="Thank you",
                    body="Unfortunately the practice task was not completed, so "
                         "we cannot continue to the main session. You will still "
                         "be compensated for your time.",
                    code=options.get("screenout_code", "")))
                display.wait_for_response(P.SCREEN_SCREENOUT, timeout=120)
                return
            runner.logger.log("practice_passed", {"successes": passed})
            # Practice data seeds normalisation but is not part of the study rounds.
            if runner.episodes:
                trainer.fit_normalizer(runner.episodes)

        # ---------------- main session ----------------
        display._send(P.screen(P.SCREEN_INSTRUCTIONS,
                               **_instructions_payload(cfg, practice=False)))
        if display.wait_for_response(P.SCREEN_INSTRUCTIONS).get("action") == "quit":
            raise _Aborted()

        summary = runner.run_dagger_mode(trainer)
        summary["blur_events"] = display.blur_events
        out_q.put(("event", {"event": "study_complete", "data": _json_safe(summary)}))

        # ---------------- debrief ----------------
        deceived = cfg.study.condition != "responsive" or cfg.study.two_phase
        summary_written = _write_join_keys(runner, cfg, external_id,
                                           options.get("survey_mode", "native"))
        exit_url = str(options.get("qualtrics_exit_url") or "")

        if options.get("skip_debrief") and exit_url:
            # The exit questionnaire carries the post-task battery *and* the
            # debrief. Hand over rather than debriefing twice.
            runner.logger.log("debrief", {"delivered_by": "qualtrics_exit"})
            display.redirect(
                qualtrics_url(exit_url, {
                    "participant_id": cfg.study.participant_id,
                    "external_id": external_id,
                    "stage": "exit",
                    "task": "complete",
                }),
                title="Task complete - final questions",
                body="One last questionnaire finishes the study.",
                cta="Continue to final questions")
            out_q.put(("event", {"event": "finished"}))
            try:
                display.wait_for_response(P.SCREEN_REDIRECT, timeout=120)
            except Disconnected:
                pass
            return

        display._send(P.screen(
            P.SCREEN_DEBRIEF,
            title="About this study",
            body=DEBRIEF_DECEPTION if deceived else DEBRIEF_PLAIN,
            participant=cfg.study.participant_id,
            condition_disclosed=deceived,
        ))
        dresp = display.wait_for_response(P.SCREEN_DEBRIEF, timeout=1800)
        runner.logger.log("debrief", {
            "acknowledged": bool(dresp.get("acknowledged")),
            "withdraw_request": bool(dresp.get("withdraw")),
            "comments": str(dresp.get("comments", ""))[:2000],
        })
        if dresp.get("withdraw"):
            out_q.put(("event", {"event": "withdraw_requested"}))

        if exit_url:
            display.redirect(
                qualtrics_url(exit_url, {
                    "participant_id": cfg.study.participant_id,
                    "external_id": external_id,
                    "stage": "exit",
                    "task": "complete",
                }),
                title="A few final questions",
                body="One last questionnaire finishes the study.",
                cta="Continue to final questions")
            out_q.put(("event", {"event": "finished"}))
            try:
                display.wait_for_response(P.SCREEN_REDIRECT, timeout=120)
            except Disconnected:
                pass
            return

        display._send(P.screen(
            P.SCREEN_DONE, title="All done - thank you",
            body="Your session is complete.",
            code=options.get("completion_code", ""),
            redirect=options.get("completion_url", "")))
        out_q.put(("event", {"event": "finished"}))
        try:
            display.wait_for_response(P.SCREEN_DONE, timeout=60)
        except Disconnected:
            pass

    except Disconnected as exc:
        out_q.put(("event", {"event": "disconnected", "data": {"why": str(exc)}}))
        if runner is not None:
            runner.logger.log("disconnected", {"why": str(exc)})
    except _Aborted:
        out_q.put(("event", {"event": "aborted"}))
        if runner is not None:
            runner.logger.log("aborted", {"where": "web_worker"})
    except Exception as exc:  # noqa: BLE001 - must not kill the server
        out_q.put(("event", {"event": "error",
                             "data": {"error": repr(exc),
                                      "traceback": traceback.format_exc()[-4000:]}}))
        try:
            display._send({"t": P.MSG_ERROR,
                           "message": "The session hit a technical problem. "
                                      "Please contact the researchers."})
        except Exception:
            pass
    finally:
        try:
            if runner is not None:
                runner.close()
        except Exception:
            pass
        out_q.put(("event", {"event": "worker_exit"}))


def _write_join_keys(runner, cfg: Config, external_id: str,
                     survey_mode: str) -> bool:
    """Drop the questionnaire join keys into the session directory.

    Written as a small standalone file as well as into the log, so a Qualtrics
    export can be merged without parsing events.jsonl.
    """
    import json as _json
    try:
        path = os.path.join(runner.session_dir, "join_keys.json")
        with open(path, "w") as f:
            _json.dump({
                "participant_id": cfg.study.participant_id,
                "external_id": external_id,
                "condition": cfg.study.condition,
                "two_phase": cfg.study.two_phase,
                "survey_mode": survey_mode,
                "session_dir": runner.session_dir,
            }, f, indent=2)
        return True
    except OSError:
        return False
