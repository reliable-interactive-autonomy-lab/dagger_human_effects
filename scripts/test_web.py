#!/usr/bin/env python
"""End-to-end test of the online study over a real WebSocket, with no browser.

Starts the server in-process, enrols a simulated participant, and drives the
entire flow: consent, instructions, practice, DAgger rounds with keyboard input,
surveys, debrief and completion. Then checks that the session on disk is the same
shape the offline pipeline produces, so `scripts/analyze.py` reads online and
in-lab data identically.

Also checks the parts only a networked deployment has: frame delivery, capacity
refusal, condition assignment staying server-side, and the debrief disclosing the
manipulation for deceived arms.

    python scripts/test_web.py --rounds 2
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dagger_lh.config import Config
from dagger_lh.web import protocol as P
from dagger_lh.web.manager import Arm

FAILURES = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    if not ok:
        FAILURES.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return ok


class Participant:
    """Scripted browser: answers screens and drives the arm downward."""

    def __init__(self, ws, verbose: bool = False):
        self.ws = ws
        self.verbose = verbose
        self.frames = 0
        self.side_frames = 0
        self.huds = []
        self.screens = []
        self.surveys = []
        self.done_payload = None
        self.debrief_payload = None
        self.screenout_payload = None
        self.redirects = []
        self.screen_seq = None
        self.finished_on_exit = False
        self.error = None
        self._t = 0
        self._expect_side = False

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    async def respond(self, screen, data):
        await self.send({"t": P.MSG_RESPONSE, "screen": screen, "data": data,
                         "seq": self.screen_seq})

    async def drive(self, timeout: float = 900.0):
        """Consume messages until the session ends."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=90)
            except asyncio.TimeoutError:
                raise AssertionError("server went quiet for 90s")
            except Exception:
                return
            if isinstance(msg, (bytes, bytearray)):
                await self._on_binary(bytes(msg))
                continue
            m = json.loads(msg)
            if m.get("t") == P.MSG_PONG:
                continue
            if m.get("t") == P.MSG_ERROR:
                self.error = m.get("message")
                return
            if m.get("t") != P.MSG_SCREEN:
                continue
            kind, payload = m["kind"], m.get("payload") or {}
            self.screen_seq = m.get("seq")
            self.screens.append(kind)
            if self.verbose:
                print(f"      screen -> {kind}")
            if await self._on_screen(kind, payload):
                return

    async def _on_binary(self, buf: bytes):
        if self._expect_side:
            self._expect_side = False
            self.side_frames += 1
            return
        header, jpeg = P.decode_frame(buf)
        self.frames += 1
        self._expect_side = bool(header.get("side"))
        if header.get("hud"):
            self.huds.append(header["hud"])
        # JPEG magic, so we know real pixels arrived rather than an empty buffer
        assert jpeg[:2] == b"\xff\xd8", "frame is not a JPEG"
        await self._act()

    async def _act(self):
        """Take control and press keys, so there is human-labelled data to train on."""
        self._t += 1
        keys, press = [], []
        if self._t == 3:
            press.append("tab")                 # take control
        if 3 <= self._t <= 26:
            keys = ["f", "w"] if self._t % 2 else ["f"]   # descend, nudge forward
            if self._t == 14:
                press.append("space")           # close gripper
        if self._t == 27:
            press.append("tab")                 # hand control back
        if self._t >= 34:
            press.append("enter")               # declare the attempt over
            self._t = 0
        await self.send({"t": P.MSG_INPUT, "keys": keys, "press": press,
                         "seq": self._t})

    async def _on_screen(self, kind, payload) -> bool:
        if kind == P.SCREEN_CONSENT:
            await self.respond("consent", {"consent": True})
        elif kind == P.SCREEN_INSTRUCTIONS:
            self._t = 0
            await self.respond("instructions", {"action": "ok"})
        elif kind == P.SCREEN_MESSAGE:
            self._t = 0
            await self.respond("message", {"action": "ok"})
        elif kind == P.SCREEN_SURVEY:
            items = payload.get("items") or []
            self.surveys.append(items)
            data = {"elapsed": 4.2}
            for it in items:
                # Answer the attention check correctly; everything else mid-scale.
                data[it["key"]] = 2 if it["key"] == "_attention" else 5
            await self.respond("survey", data)
        elif kind == P.SCREEN_DEBRIEF:
            self.debrief_payload = payload
            await self.respond("debrief", {"acknowledged": True, "withdraw": False,
                                           "comments": "scripted test"})
        elif kind == P.SCREEN_REDIRECT:
            # Stand in for the participant going to Qualtrics and coming back:
            # the return trip answers the redirect screen rather than rendering it.
            self.redirects.append(payload)
            await self.respond("redirect", {"returned": True})
            if "stage=exit" in str(payload.get("url", "")):
                # With --skip-debrief the exit questionnaire is the last screen,
                # so the session ends on a handoff rather than a done screen.
                self.finished_on_exit = True
                return True
        elif kind == P.SCREEN_SCREENOUT:
            self.screenout_payload = payload
            await self.respond("screenout", {"ack": True})
            return True
        elif kind == P.SCREEN_DONE:
            self.done_payload = payload
            await self.respond("done", {"ack": True})
            return True
        return False


def start_server(cfg, options, arms, tmp, port):
    import uvicorn
    from dagger_lh.web.server import create_app
    app = create_app(cfg, max_sessions=1, arms=arms, admin_token="testtoken",
                     options=options, state_dir=os.path.join(tmp, "web"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    return server, th


async def run(args, tmp, port, arm_name, condition, two_phase):
    import urllib.request
    from websockets.asyncio.client import connect

    base = f"http://127.0.0.1:{port}"

    def post(path, body):
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, {"detail": e.read().decode()[:200]}

    def get(path):
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return json.loads(r.read())

    print(f"\n=== arm: {arm_name} (condition={condition}, two_phase={two_phase}) ===")
    check("healthz responds", get("/healthz").get("ok") is True)

    status, j = post("/api/enroll", {"arm": arm_name, "token": "testtoken",
                                     "source": "pytest",
                                     "external_id": "5f2b8test"})
    check("enroll succeeds", status == 200, str(status))
    sid = j["sid"]
    check("enroll returns no condition information",
          "arm" not in j and "condition" not in j, str(sorted(j)))

    # capacity is 1, so a second enrolment must be refused rather than queued
    status2, _ = post("/api/enroll", {})
    check("second enrolment refused at capacity", status2 == 503, str(status2))

    part = None
    async with connect(f"ws://127.0.0.1:{port}/ws/{sid}", max_size=2**22) as ws:
        part = Participant(ws, verbose=args.verbose)
        await part.send({"t": P.MSG_HELLO, "viewport": [1280, 800], "ua": "test"})
        await part.send({"t": P.MSG_PING, "ts": 0.0})
        await part.drive(timeout=args.timeout)

    check("no worker error", part.error is None, str(part.error))
    if args.qualtrics:
        check("no in-app consent (intake questionnaire took it)",
              P.SCREEN_CONSENT not in part.screens)
    else:
        check("consent shown first", part.screens[0] == P.SCREEN_CONSENT,
              part.screens[0] if part.screens else "none")
    check("instructions shown", P.SCREEN_INSTRUCTIONS in part.screens)
    n_instr = part.screens.count(P.SCREEN_INSTRUCTIONS)
    check("no duplicate screens on first connect", n_instr == 2,
          f"{n_instr} instruction screens (expect 2: practice + main)")
    # Scale with round count so the threshold stays meaningful at any --rounds.
    min_frames = 20 * args.rounds
    check("teleop frames delivered", part.frames >= min_frames,
          f"{part.frames} frames (>= {min_frames})")
    check("wrist-camera frames delivered", part.side_frames > 0,
          f"{part.side_frames}")
    check("HUD states delivered", len(part.huds) >= min_frames,
          f"{len(part.huds)} (>= {min_frames})")

    controllers = {str(h.get("controller")) for h in part.huds}
    check("HUD reported both control owners",
          {"YOU", "ROBOT"}.issubset(controllers), str(sorted(controllers)))
    check("HUD never leaks the condition",
          not any("contingent" in json.dumps(h).lower() or
                  "sham" in json.dumps(h).lower() for h in part.huds))

    if args.qualtrics:
        n_round = [r for r in part.redirects if "round" in str(r.get("url"))]
        n_exit = [r for r in part.redirects if "exit" in str(r.get("url"))]
        check("per-round Qualtrics handoff for every round",
              len(n_round) == args.rounds, f"{len(n_round)} redirects")
        check("exit questionnaire handoff", len(n_exit) == 1, f"{len(n_exit)}")
        if n_round:
            url = n_round[0]["url"]
            for need in ("participant_id=", "round=", "stage=round"):
                check(f"round link carries {need.rstrip('=')}", need in url)
        if n_exit:
            check("exit link carries stage=exit", "stage=exit" in n_exit[0]["url"])
        check("no in-app survey screens in qualtrics mode",
              P.SCREEN_SURVEY not in part.screens)
        check("no in-app debrief (exit questionnaire carries it)",
              part.debrief_payload is None)
    check("surveys delivered", args.qualtrics or len(part.surveys) >= 1,
          f"{len(part.surveys)}")
    if part.surveys and not args.qualtrics:
        keys = {it["key"] for it in part.surveys[0]}
        check("survey has the helplessness items",
              {"control", "effectiveness", "expect_success", "frustration",
               "persistence", "effort"}.issubset(keys), str(len(keys)) + " items")

    check("debrief shown", args.qualtrics or part.debrief_payload is not None)
    if part.debrief_payload and not args.qualtrics:
        disclosed = bool(part.debrief_payload.get("condition_disclosed"))
        body = part.debrief_payload.get("body", "").lower()
        if condition != "responsive" or two_phase:
            check("debrief discloses the manipulation",
                  disclosed and "discarded" in body)
        else:
            check("debrief is the non-deception version", not disclosed)

    check("completion screen shown",
          args.qualtrics or part.done_payload is not None)
    if part.done_payload and not args.qualtrics:
        check("completion code returned",
              part.done_payload.get("code") == "TESTCODE",
              str(part.done_payload.get("code")))

    # ---- admin view (researcher-only information) ----
    adm = get(f"/api/admin/sessions/{sid}?token=testtoken")
    check("admin exposes the arm to researchers", adm.get("arm") == arm_name,
          str(adm.get("arm")))
    events = [e["event"] for e in adm.get("events", [])]
    check("worker reported completion", "finished" in events, str(events[-3:]))

    # ---- on-disk artefacts must match the offline pipeline ----
    sdir = adm.get("session_dir")
    check("session directory recorded", bool(sdir and os.path.isdir(sdir)), str(sdir))
    if sdir and os.path.isdir(sdir):
        rounds = os.path.join(sdir, "rounds.csv")
        ok = os.path.exists(rounds)
        check("rounds.csv written", ok)
        if ok:
            lines = open(rounds).read().strip().split("\n")
            hdr = lines[0].split(",")
            check("rounds.csv has one row per round",
                  len(lines) - 1 == args.rounds, f"{len(lines)-1} rows")
            need = {"intervention_rate", "engagement_rate", "passivity",
                    "keypresses_per_step", "giveup_events", "condition"}
            check("rounds.csv carries the helplessness measures",
                  need.issubset(set(hdr)), f"{len(hdr)} columns")
            check("survey answers landed in the CSV",
                  any(h.startswith("survey_") for h in hdr))
        check("summary.json written",
              os.path.exists(os.path.join(sdir, "summary.json")))
        check("events.jsonl written",
              os.path.exists(os.path.join(sdir, "events.jsonl")))
        check("demos.hdf5 written",
              os.path.exists(os.path.join(sdir, "demos.hdf5")))
        check("per-round adapters saved",
              len(glob.glob(os.path.join(sdir, "adapter_round_*.pth"))) == args.rounds)
        jk = os.path.join(sdir, "join_keys.json")
        if args.qualtrics:
            ok_jk = os.path.exists(jk)
            check("join_keys.json written for the Qualtrics merge", ok_jk)
            if ok_jk:
                keys = json.load(open(jk))
                check("join key recorded",
                      keys.get("external_id") == "5f2b8test",
                      str(keys.get("external_id")))
        # the human's corrections must actually be in the dataset
        import h5py
        with h5py.File(os.path.join(sdir, "demos.hdf5")) as f:
            demos = [k for k in f["data"] if k.startswith("demo_")]
            human = sum(int(np.array(f["data"][d]["actor"]).sum()) for d in demos)
            total = sum(int(f["data"][d].attrs["num_samples"]) for d in demos)
        check("keyboard corrections recorded as human-labelled",
              0 < human < total, f"{human}/{total} steps from {len(demos)} episodes")
    return sdir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--qualtrics", action="store_true",
                    help="exercise the Qualtrics handoff flow instead of the "
                         "in-app surveys")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="dagger_web_")
    cfg = Config.from_dict({
        "mode": "dagger",
        "env": {"env_name": "Lift", "horizon": 40},
        "policy": {"algo": "bc_rnn", "lora": {"rank": 4}},
        "train": {"finetune_steps": 20, "finetune_max_seconds": 30,
                  "finetune_min_seconds": 0, "batch_size": 8, "device": "cpu"},
        "study": {"condition": "noncontingent", "num_rounds": args.rounds,
                  "rollouts_per_round": 1, "survey_enabled": True,
                  "survey_every_n_rounds": 1},
        "paths": {"root": tmp, "demos": os.path.join(tmp, "demos"),
                  "checkpoints": os.path.join(tmp, "ckpt"),
                  "sessions": os.path.join(tmp, "sessions"),
                  "logs": os.path.join(tmp, "logs")},
    })
    if args.checkpoint:
        cfg.base_checkpoint = args.checkpoint
    options = {
        "frame_size": 256, "jpeg_quality": 60,
        "practice_episodes": 0, "practice_required": 0,
        "attention_check_every": 1, "minutes": 5,
        "completion_code": "TESTCODE", "screenout_code": "SCREENOUT",
        "response_timeout": 300.0, "torch_threads": 1,
    }
    if args.qualtrics:
        options.update({
            "survey_mode": "qualtrics",
            "qualtrics_round_url": "https://example.qualtrics.com/jfe/form/SV_round",
            "qualtrics_exit_url": "https://example.qualtrics.com/jfe/form/SV_exit",
            "skip_consent": True,
            "skip_debrief": True,
        })
    arms = [Arm("nc", "noncontingent"), Arm("ctl", "responsive")]

    server, th = start_server(cfg, options, arms, tmp, args.port)
    deadline = time.time() + 30
    import urllib.request
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=2)
            break
        except Exception:
            time.sleep(0.3)
    else:
        print("server did not start", file=sys.stderr)
        return 1

    try:
        asyncio.run(run(args, tmp, args.port, "nc", "noncontingent", False))
        print("\n" + "=" * 62)
        if not FAILURES:
            print("ALL CHECKS PASSED")
        else:
            print(f"{len(FAILURES)} CHECK(S) FAILED:")
            for f in FAILURES:
                print("  -", f)
        return 1 if FAILURES else 0
    finally:
        server.should_exit = True
        th.join(timeout=10)
        if args.keep:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
