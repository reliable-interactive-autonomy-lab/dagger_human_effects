"""Wire protocol between the browser client and the session worker.

Two channels share one WebSocket:

  * **text frames** carry JSON control messages (screens, survey definitions,
    status, responses);
  * **binary frames** carry video: ``[4-byte big-endian header length][UTF-8 JSON
    header][JPEG bytes]``. Keeping pixels out of JSON avoids base64's 33%
    overhead on the only high-rate message.

Continuous control uses an *input state* model rather than an event log: the
client reports the current set of held keys, and the server steps the simulator
with whatever it last heard. A dropped or late packet then costs a little control
authority instead of desynchronising the episode. Discrete presses (TAB, SPACE,
ENTER, X) cannot be resent that way, so those travel as an edge-triggered list
that the server drains exactly once.
"""
from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------- client -> server
MSG_INPUT = "input"              # {keys: [...], press: [...], seq: int}
MSG_RESPONSE = "response"        # {screen: str, data: {...}}
MSG_FOCUS = "focus"              # {focused: bool}  -- browser tab visibility
MSG_PING = "ping"                # {ts: float}
MSG_HELLO = "hello"             # {viewport: [w,h], ua: str}
MSG_RESEND = "resend"           # internal: a socket reconnected, replay the screen

# ----------------------------------------------------------------- server -> client
MSG_SCREEN = "screen"            # {kind: str, payload: {...}}
MSG_STATUS = "status"            # {...}
MSG_PONG = "pong"
MSG_ERROR = "error"

# Screen kinds the client knows how to render.
SCREEN_CONSENT = "consent"
SCREEN_INSTRUCTIONS = "instructions"
SCREEN_MESSAGE = "message"       # modal text, advanced by a button
SCREEN_TELEOP = "teleop"         # live simulation view; frames follow
SCREEN_SURVEY = "survey"         # Likert / choice items
SCREEN_DEBRIEF = "debrief"
SCREEN_DONE = "done"
SCREEN_REDIRECT = "redirect"      # hand the whole tab to an external URL
SCREEN_SCREENOUT = "screenout"   # failed the practice qualification

#: Logical key names the client may report. Anything else is ignored server-side,
#: so a stray browser key cannot inject an unexpected command.
ALLOWED_KEYS = frozenset({
    "w", "a", "s", "d", "r", "f", "q", "e",
    "up", "down", "left", "right",
    "space", "tab", "enter", "backspace", "x", "p", "h", "escape",
    "lshift", "rshift",
})


def encode_frame(jpeg: bytes, header: Dict[str, Any]) -> bytes:
    """Pack a video frame plus its HUD state into one binary WebSocket message."""
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(raw)) + raw + jpeg


def decode_frame(buf: bytes) -> tuple[Dict[str, Any], bytes]:
    """Inverse of `encode_frame`; used by the headless test client."""
    (n,) = struct.unpack(">I", buf[:4])
    header = json.loads(buf[4:4 + n].decode("utf-8"))
    return header, buf[4 + n:]


def sanitize_keys(keys: Any) -> List[str]:
    """Keep only recognised key names, de-duplicated, capped in length.

    The client is untrusted: this bounds both the key set and the work done per
    message so a malformed or hostile payload cannot drive the simulator loop.
    """
    if not isinstance(keys, (list, tuple)):
        return []
    out: List[str] = []
    for k in keys[:24]:
        if isinstance(k, str):
            k = k.lower()
            if k in ALLOWED_KEYS and k not in out:
                out.append(k)
    return out


def screen(kind: str, **payload: Any) -> Dict[str, Any]:
    return {"t": MSG_SCREEN, "kind": kind, "payload": payload}
