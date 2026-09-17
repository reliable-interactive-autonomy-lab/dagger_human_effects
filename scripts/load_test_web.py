#!/usr/bin/env python
"""Concurrency and latency check for the online study.

Runs N scripted participants against a live server at once and reports the
per-participant frame rate and control latency. Use it to pick `--max-sessions`
for your hardware *before* recruiting: a participant whose frame rate collapses
is a participant whose data is not comparable to the others'.

    # against a server you started separately
    python scripts/load_test_web.py --url http://127.0.0.1:8000 --n 4

    # self-contained (starts its own server)
    python scripts/load_test_web.py --n 4 --rounds 1 --spawn
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dagger_lh.config import Config
from dagger_lh.web import protocol as P
from dagger_lh.web.manager import Arm


class Load:
    def __init__(self, idx, url, rounds):
        self.idx = idx
        self.url = url
        self.rounds = rounds
        self.frames = 0
        self.rtts = []
        self.frame_times = []
        self.started = None
        self.finished = False
        self.screens = []
        self.error = None
        self._t = 0
        self._expect_side = False

    async def run(self):
        import urllib.request
        from websockets.asyncio.client import connect

        def post(path, body):
            req = urllib.request.Request(
                self.url + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())

        try:
            j = await asyncio.get_running_loop().run_in_executor(
                None, post, "/api/enroll", {"source": f"load{self.idx}"})
        except Exception as exc:
            self.error = f"enroll failed: {exc}"
            return
        sid = j["sid"]
        ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://")
        self.started = time.time()
        try:
            async with connect(f"{ws_url}/ws/{sid}", max_size=2 ** 22) as ws:
                self.ws = ws
                pinger = asyncio.create_task(self._ping())
                try:
                    await self._loop(ws)
                finally:
                    pinger.cancel()
        except Exception as exc:
            self.error = repr(exc)

    async def _ping(self):
        while True:
            t0 = time.perf_counter()
            try:
                await self.ws.send(json.dumps({"t": P.MSG_PING, "ts": t0}))
            except Exception:
                return
            await asyncio.sleep(2.0)

    async def _loop(self, ws):
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
            except (asyncio.TimeoutError, Exception):
                return
            if isinstance(msg, (bytes, bytearray)):
                if self._expect_side:
                    self._expect_side = False
                    continue
                header, _ = P.decode_frame(bytes(msg))
                self._expect_side = bool(header.get("side"))
                self.frames += 1
                self.frame_times.append(time.perf_counter())
                await self._act(ws)
                continue
            m = json.loads(msg)
            if m.get("t") == P.MSG_PONG:
                self.rtts.append((time.perf_counter() - float(m["ts"])) * 1000)
                continue
            if m.get("t") == P.MSG_ERROR:
                self.error = m.get("message")
                return
            if m.get("t") != P.MSG_SCREEN:
                continue
            kind, payload = m["kind"], m.get("payload") or {}
            self.screens.append(kind)
            if kind == P.SCREEN_CONSENT:
                await self._respond(ws, "consent", {"consent": True})
            elif kind == P.SCREEN_INSTRUCTIONS:
                self._t = 0
                await self._respond(ws, "instructions", {"action": "ok"})
            elif kind == P.SCREEN_MESSAGE:
                self._t = 0
                await self._respond(ws, "message", {"action": "ok"})
            elif kind == P.SCREEN_SURVEY:
                data = {"elapsed": 3.0}
                for it in payload.get("items", []):
                    data[it["key"]] = 2 if it["key"] == "_attention" else 4
                await self._respond(ws, "survey", data)
            elif kind == P.SCREEN_DEBRIEF:
                await self._respond(ws, "debrief", {"acknowledged": True})
            elif kind in (P.SCREEN_DONE, P.SCREEN_SCREENOUT):
                self.finished = True
                await self._respond(ws, kind, {"ack": True})
                return

    async def _respond(self, ws, screen, data):
        await ws.send(json.dumps({"t": P.MSG_RESPONSE, "screen": screen,
                                  "data": data}))

    async def _act(self, ws):
        self._t += 1
        keys, press = [], []
        if self._t == 3:
            press.append("tab")
        if 3 <= self._t <= 24:
            keys = ["f"]
            if self._t == 12:
                press.append("space")
        if self._t >= 30:
            press.append("enter")
            self._t = 0
        await ws.send(json.dumps({"t": P.MSG_INPUT, "keys": keys,
                                  "press": press, "seq": self._t}))

    def report(self):
        fps = 0.0
        if len(self.frame_times) > 4:
            span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0
        rtt = statistics.median(self.rtts) if self.rtts else float("nan")
        return {"idx": self.idx, "frames": self.frames, "fps": fps,
                "rtt_ms": rtt, "finished": self.finished, "error": self.error}


def spawn_server(port, rounds, tmp):
    import uvicorn
    from dagger_lh.web.server import create_app
    cfg = Config.from_dict({
        "mode": "dagger",
        "env": {"env_name": "Lift", "horizon": 35,
                "render_height": 384, "render_width": 384},
        "policy": {"algo": "bc_rnn", "lora": {"rank": 4}},
        "train": {"finetune_steps": 30, "finetune_max_seconds": 30,
                  "finetune_min_seconds": 0, "batch_size": 8, "device": "cpu"},
        "study": {"condition": "responsive", "num_rounds": rounds,
                  "rollouts_per_round": 1, "survey_enabled": True},
        "paths": {"root": tmp, "demos": f"{tmp}/demos",
                  "checkpoints": f"{tmp}/ckpt", "sessions": f"{tmp}/sessions",
                  "logs": f"{tmp}/logs"},
    })
    options = {"frame_size": 384, "jpeg_quality": 70, "practice_episodes": 0,
               "practice_required": 0, "attention_check_every": 0,
               "completion_code": "LOAD", "response_timeout": 300.0,
               "torch_threads": 1}
    app = create_app(cfg, max_sessions=64,
                     arms=[Arm("responsive", "responsive")],
                     admin_token="load", options=options,
                     state_dir=os.path.join(tmp, "web"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    return server, th


async def main_async(args):
    parts = [Load(i, args.url, args.rounds) for i in range(args.n)]
    t0 = time.time()
    await asyncio.gather(*(p.run() for p in parts))
    wall = time.time() - t0

    print(f"\n{args.n} concurrent participants, {wall:.0f}s wall clock")
    print(f"{'#':>3} {'frames':>7} {'fps':>7} {'rtt ms':>8}  {'status':<10}")
    for p in parts:
        r = p.report()
        status = "done" if r["finished"] else (r["error"] or "incomplete")
        print(f"{r['idx']:>3} {r['frames']:>7} {r['fps']:>7.1f} "
              f"{r['rtt_ms']:>8.1f}  {str(status)[:40]:<10}")

    reports = [p.report() for p in parts]
    fps = [r["fps"] for r in reports if r["fps"] > 0]
    rtts = [r["rtt_ms"] for r in reports if r["rtt_ms"] == r["rtt_ms"]]
    done = sum(r["finished"] for r in reports)
    print(f"\ncompleted {done}/{args.n}")
    if fps:
        print(f"frame rate  min {min(fps):.1f}  median {statistics.median(fps):.1f} "
              f"(target ~{args.target_fps})")
    if rtts:
        print(f"control rtt median {statistics.median(rtts):.0f} ms  "
              f"max {max(rtts):.0f} ms")

    ok = True
    if done != args.n:
        ok = False
        print("FAIL  not every participant finished")
    if fps and min(fps) < args.target_fps * 0.75:
        ok = False
        print(f"FAIL  slowest participant fell to {min(fps):.1f} fps against a "
              f"{args.target_fps:.0f} fps target -- lower --max-sessions or add cores")
    if fps and max(fps) > args.target_fps * 1.25:
        # Running fast is also wrong: simulated time then outruns wall clock and
        # the robot moves faster than the demonstrations the policy learned from.
        ok = False
        print(f"FAIL  fastest participant ran at {max(fps):.1f} fps, above the "
              f"{args.target_fps:.0f} fps control rate -- episode pacing is broken")
    if rtts and max(rtts) > 250:
        print(f"WARN  control latency up to {max(rtts):.0f} ms; teleoperation "
              f"gets noticeably harder past ~150 ms")
    if ok:
        print("PASS  all participants completed at an acceptable frame rate")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="server base URL")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--spawn", action="store_true", help="start a local server")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--target-fps", type=float, default=20.0)
    args = ap.parse_args()

    tmp, server, th = None, None, None
    if args.spawn or not args.url:
        tmp = tempfile.mkdtemp(prefix="dagger_load_")
        server, th = spawn_server(args.port, args.rounds, tmp)
        args.url = f"http://127.0.0.1:{args.port}"
        import urllib.request
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(args.url + "/healthz", timeout=2)
                break
            except Exception:
                time.sleep(0.3)
    try:
        return asyncio.run(main_async(args))
    finally:
        if server is not None:
            server.should_exit = True
            th.join(timeout=10)
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
