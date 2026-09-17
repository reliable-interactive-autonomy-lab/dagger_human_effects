"""FastAPI application: static client, enrolment, WebSocket relay, admin.

The server is deliberately thin. It does no simulation and holds no study logic;
it moves bytes between each participant's WebSocket and their worker process, and
exposes an admin view. All study behaviour lives in `dagger_lh/web/worker.py`,
which reuses the same `SessionRunner` as the desktop version.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import (Depends, FastAPI, HTTPException, Query, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from . import protocol as P
from .manager import DEFAULT_ARMS, Arm, SessionManager

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# How long a worker may hold a frame before we consider the client too slow.
FRAME_DRAIN_INTERVAL = 0.005


def create_app(
    cfg: Config,
    max_sessions: int = 8,
    arms: Optional[List[Arm]] = None,
    admin_token: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    state_dir: str = "data/web",
) -> FastAPI:
    app = FastAPI(title="DAgger learned-helplessness study", docs_url=None,
                  redoc_url=None)
    manager = SessionManager(cfg, arms=arms or DEFAULT_ARMS,
                             max_sessions=max_sessions, options=options,
                             state_dir=state_dir)
    app.state.manager = manager
    app.state.admin_token = admin_token or secrets.token_urlsafe(24)

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    async def index() -> Any:
        path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(path):
            return HTMLResponse("<h1>client not built</h1>", status_code=500)
        return FileResponse(path)

    @app.get("/healthz")
    async def healthz() -> Any:
        return {"ok": True, "active": manager.active_count(),
                "capacity": manager.max_sessions}

    # ------------------------------------------------------------------ enrolment
    @app.post("/api/enroll")
    async def enroll(request: Request) -> Any:
        """Assign a condition and spawn the participant's worker process."""
        body: Dict[str, Any] = {}
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            pass
        if not manager.has_capacity():
            # 503 rather than queueing: a participant kept waiting behind other
            # sessions would start the task under different conditions.
            raise HTTPException(status_code=503, detail="at_capacity")

        recruitment = {
            k: str(body.get(k))[:128]
            for k in ("PROLIFIC_PID", "STUDY_ID", "SESSION_ID", "source",
                      "external_id", "pid", "participant_id", "ResponseID")
            if body.get(k)
        }
        # Prefer an explicit external id so the behavioural session and the
        # questionnaire response share a join key. Order matters: the recruitment
        # platform's id is the most durable, then whatever the intake
        # questionnaire passed through.
        pid = (recruitment.get("PROLIFIC_PID")
               or recruitment.get("external_id")
               or recruitment.get("participant_id")
               or recruitment.get("pid")
               or None)
        if pid:
            recruitment.setdefault("external_id", pid)
        forced = None
        if body.get("arm") and body.get("token") == app.state.admin_token:
            forced = str(body["arm"])  # researcher testing a specific arm
        try:
            sess = manager.create(participant_id=pid, forced_arm=forced,
                                  recruitment=recruitment)
        except RuntimeError:
            raise HTTPException(status_code=503, detail="at_capacity")
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"sid": sess.sid, "participant_id": sess.participant_id}

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws/{sid}")
    async def ws(sock: WebSocket, sid: str) -> None:
        sess = manager.get(sid)
        if sess is None or not sess.alive():
            await sock.close(code=4404)
            return
        await sock.accept()
        reconnecting = sess.ever_connected
        sess.connected = True
        sess.ever_connected = True
        sess.last_seen = time.time()
        # On a *re*connect, ask the worker to re-send whatever screen it is on:
        # covers a refresh, a flaky network, and the return trip from an external
        # questionnaire. Skipped on the first connect, where there is nothing to
        # replay and the request would only race the worker's opening screen.
        if reconnecting:
            try:
                sess.in_q.put_nowait((P.MSG_RESEND, {}))
            except (queue.Full, ValueError, OSError):
                pass
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        async def pump_out() -> None:
            """Worker -> client. Runs the blocking queue read off the event loop."""
            while not stop.is_set():
                try:
                    item = await loop.run_in_executor(
                        None, _get_with_timeout, sess.out_q, 0.2)
                except Exception:
                    break
                if item is None:
                    continue
                kind = item[0]
                try:
                    if kind == "frame":
                        await sock.send_bytes(item[1])
                        if item[2] is not None:
                            await sock.send_bytes(item[2])
                    elif kind == "json":
                        await sock.send_text(json.dumps(item[1]))
                    elif kind == "event":
                        manager.note_event(sess, item[1])
                        if item[1].get("event") == "error":
                            await sock.send_text(json.dumps({
                                "t": P.MSG_ERROR,
                                "message": "technical problem"}))
                except (WebSocketDisconnect, RuntimeError, OSError):
                    break
            stop.set()

        async def pump_in() -> None:
            """Client -> worker, with validation at the boundary."""
            while not stop.is_set():
                try:
                    raw = await sock.receive_text()
                except (WebSocketDisconnect, RuntimeError, OSError):
                    break
                sess.last_seen = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("t")
                if t == P.MSG_PING:
                    try:
                        await sock.send_text(json.dumps(
                            {"t": P.MSG_PONG, "ts": msg.get("ts"),
                             "server_ts": time.time()}))
                    except (WebSocketDisconnect, RuntimeError, OSError):
                        break
                    continue
                if t not in (P.MSG_INPUT, P.MSG_RESPONSE, P.MSG_FOCUS, P.MSG_HELLO):
                    continue
                payload = {k: v for k, v in msg.items() if k != "t"}
                try:
                    sess.in_q.put_nowait((t, payload))
                except queue.Full:
                    # Input is a *state* report, so dropping the oldest is
                    # correct: the next message supersedes it anyway.
                    try:
                        sess.in_q.get_nowait()
                        sess.in_q.put_nowait((t, payload))
                    except (queue.Empty, queue.Full):
                        pass
            stop.set()

        try:
            await asyncio.gather(pump_out(), pump_in())
        finally:
            sess.connected = False
            sess.last_seen = time.time()
            # The peer may already be gone; closing then raises rather than
            # returning, and it is not an error worth surfacing.
            try:
                await sock.close()
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass

    # ------------------------------------------------------------------ admin
    def require_admin(token: str = Query(default="")) -> None:
        if not secrets.compare_digest(token, app.state.admin_token):
            raise HTTPException(status_code=403, detail="bad token")

    @app.get("/api/admin/status", dependencies=[Depends(require_admin)])
    async def admin_status() -> Any:
        return manager.status()

    @app.post("/api/admin/stop/{sid}", dependencies=[Depends(require_admin)])
    async def admin_stop(sid: str) -> Any:
        return {"stopped": manager.stop(sid)}

    @app.get("/api/admin/sessions/{sid}", dependencies=[Depends(require_admin)])
    async def admin_session(sid: str) -> Any:
        sess = manager.get(sid)
        if sess is None:
            raise HTTPException(status_code=404, detail="no such session")
        out = sess.public()
        # Researcher-facing only: never sent over a participant's socket.
        out["arm"] = sess.arm.name
        out["condition"] = sess.arm.condition
        out["two_phase"] = sess.arm.two_phase
        out["session_dir"] = sess.session_dir
        out["events"] = sess.events[-50:]
        out["recruitment"] = sess.recruitment
        return out

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page() -> Any:
        path = os.path.join(STATIC_DIR, "admin.html")
        if not os.path.exists(path):
            return HTMLResponse("<h1>admin client not built</h1>", status_code=500)
        return FileResponse(path)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        manager.shutdown()

    return app


def _get_with_timeout(q: Any, timeout: float) -> Optional[Any]:
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None
