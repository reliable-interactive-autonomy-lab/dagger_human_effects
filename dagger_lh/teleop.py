"""Keyboard teleoperation device and pygame operator display.

Deliberately uses pygame rather than robosuite's built-in `Keyboard` device:
  * no macOS accessibility/`pynput` permissions, and no GLFW focus fighting;
  * held-key state (`pygame.key.get_pressed`) gives smooth continuous control;
  * the same surface carries the HUD the study needs (who is in control, round,
    progress feedback), which the built-in device has nowhere to put.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import TeleopConfig

# --------------------------------------------------------------------------- keymap

#: Axis bindings as *logical key names*, shared by the local pygame device and the
#: browser client. Both go through `BaseTeleop._state_from`, so the two front-ends
#: cannot drift apart -- which matters because a study may mix them.
TRANS_AXES: Dict[int, Tuple[str, str]] = {
    0: ("w", "s"),          # +X / -X
    1: ("a", "d"),          # +Y / -Y
    2: ("r", "f"),          # +Z / -Z
}
ROT_AXES: Dict[int, Tuple[str, str]] = {
    0: ("right", "left"),   # roll
    1: ("up", "down"),      # pitch
    2: ("q", "e"),          # yaw
}
#: Edge-triggered keys -> TeleopState field they set.
PRESS_EVENTS: Dict[str, str] = {
    "escape": "event_quit",
    "enter": "event_success",
    "backspace": "event_discard",
    "x": "event_giveup",
    "h": "event_toggle_help",
    "p": "event_pause",
}
HOLD_INTERVENE_KEYS = ("tab", "lshift", "rshift")

KEYMAP_HELP: List[Tuple[str, str]] = [
    ("W / S", "move +X / -X  (forward / back)"),
    ("A / D", "move +Y / -Y  (left / right)"),
    ("R / F", "move +Z / -Z  (up / down)"),
    ("<- / ->", "roll  -/+"),
    ("Up / Dn", "pitch +/-"),
    ("Q / E", "yaw  +/-"),
    ("SPACE", "toggle gripper open/close"),
    ("TAB", "take / release control"),
    ("ENTER", "episode succeeded - save"),
    ("BACKSPACE", "discard episode - retry"),
    ("X", "give up on this episode"),
    ("P", "pause"),
    ("H", "toggle this help"),
    ("ESC", "quit session"),
]


@dataclass
class TeleopState:
    """Per-step snapshot of the operator's input, consumed by the study logger."""
    action: np.ndarray = field(default_factory=lambda: np.zeros(7))
    engaged: bool = False           # is the human commanding motion this step
    intervening: bool = False       # does the human hold control authority
    keys_held: int = 0
    input_norm: float = 0.0
    gripper_closed: bool = False
    # edge-triggered, consumed once
    event_success: bool = False
    event_discard: bool = False
    event_giveup: bool = False
    event_quit: bool = False
    event_toggle_help: bool = False
    event_pause: bool = False
    event_toggle_control: bool = False


class BaseTeleop:
    """Transport-independent teleoperation semantics.

    Subclasses only have to turn their input source into a set of held logical key
    names plus a list of edge presses, then call `_state_from`. Smoothing, gripper
    latching, control authority and the effort counters all live here so the local
    and browser front-ends behave identically.
    """

    def __init__(self, cfg: TeleopConfig, action_dim: int = 7):
        self.cfg = cfg
        self.action_dim = action_dim
        self._smoothed = np.zeros(6, dtype=np.float64)
        self._gripper_closed = False
        self._intervening = False
        self._authority_pinned = False
        self.total_keypresses = 0
        self.total_engaged_steps = 0
        self.total_steps = 0

    # ------------------------------------------------------------------ lifecycle
    def reset_episode(self) -> None:
        self._smoothed[:] = 0.0
        self._gripper_closed = False
        if not self.cfg.hold_to_intervene and not self._authority_pinned:
            self._intervening = False

    @property
    def intervening(self) -> bool:
        return self._intervening

    def force_release(self) -> None:
        self._intervening = False

    def set_authority(self, engaged: bool) -> None:
        """Pin control authority, used by demonstration mode where the human
        always drives. Overridden each poll when `hold_to_intervene` is set."""
        self._intervening = bool(engaged)
        self._authority_pinned = bool(engaged)

    # ------------------------------------------------------------------ semantics
    def _state_from(self, held: Set[str], presses: Sequence[str]) -> TeleopState:
        st = TeleopState(action=np.zeros(self.action_dim))

        for key in presses:
            self.total_keypresses += 1
            field = PRESS_EVENTS.get(key)
            if field is not None:
                setattr(st, field, True)
            elif key == "space":
                self._gripper_closed = not self._gripper_closed
            elif key == "tab" and not self.cfg.hold_to_intervene:
                self._intervening = not self._intervening
                st.event_toggle_control = True

        if self.cfg.hold_to_intervene and not self._authority_pinned:
            engaged = any(k in held for k in HOLD_INTERVENE_KEYS)
            if engaged != self._intervening:
                st.event_toggle_control = True
            self._intervening = bool(engaged)

        raw = np.zeros(6, dtype=np.float64)
        n_held = 0
        for axis, (kp, km) in TRANS_AXES.items():
            raw[axis] = (float(kp in held) - float(km in held)) * self.cfg.pos_sensitivity
            n_held += int(kp in held) + int(km in held)
        for axis, (kp, km) in ROT_AXES.items():
            raw[3 + axis] = (float(kp in held) - float(km in held)) * self.cfg.rot_sensitivity
            n_held += int(kp in held) + int(km in held)

        a = float(np.clip(self.cfg.smoothing, 0.0, 1.0))
        self._smoothed = a * raw + (1.0 - a) * self._smoothed
        self._smoothed[np.abs(self._smoothed) < 1e-3] = 0.0

        action = np.zeros(self.action_dim, dtype=np.float64)
        action[:6] = np.clip(self._smoothed, -1.0, 1.0)
        if self.action_dim > 6:
            action[6] = 1.0 if self._gripper_closed else -1.0

        st.action = action
        st.keys_held = n_held
        st.input_norm = float(np.linalg.norm(raw))
        st.engaged = bool(n_held > 0)
        st.intervening = self._intervening
        st.gripper_closed = self._gripper_closed

        self.total_steps += 1
        self.total_engaged_steps += int(st.engaged)
        return st

    def poll(self) -> TeleopState:
        raise NotImplementedError


class KeyboardDevice(BaseTeleop):
    """Local pygame keyboard -> normalised delta-pose + gripper action."""

    #: pygame key constant -> logical key name
    _PYGAME_NAMES: Dict[int, str] = {}

    def __init__(self, cfg: TeleopConfig, action_dim: int = 7):
        import pygame
        super().__init__(cfg, action_dim)
        self.pg = pygame
        if not KeyboardDevice._PYGAME_NAMES:
            KeyboardDevice._PYGAME_NAMES = {
                pygame.K_w: "w", pygame.K_s: "s", pygame.K_a: "a", pygame.K_d: "d",
                pygame.K_r: "r", pygame.K_f: "f", pygame.K_q: "q", pygame.K_e: "e",
                pygame.K_UP: "up", pygame.K_DOWN: "down",
                pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
                pygame.K_SPACE: "space", pygame.K_TAB: "tab",
                pygame.K_RETURN: "enter", pygame.K_BACKSPACE: "backspace",
                pygame.K_x: "x", pygame.K_p: "p", pygame.K_h: "h",
                pygame.K_ESCAPE: "escape",
                pygame.K_LSHIFT: "lshift", pygame.K_RSHIFT: "rshift",
            }

    # ------------------------------------------------------------------ polling
    def poll(self) -> TeleopState:
        pygame = self.pg
        presses: List[str] = []
        quit_requested = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_requested = True
            elif ev.type == pygame.KEYDOWN:
                name = self._PYGAME_NAMES.get(ev.key)
                if name is not None:
                    presses.append(name)
        pressed = pygame.key.get_pressed()
        held = {name for key, name in self._PYGAME_NAMES.items() if pressed[key]}
        st = self._state_from(held, presses)
        if quit_requested:
            st.event_quit = True
        return st


class OperatorDisplay:
    """pygame window: camera view(s) on the left, study HUD on the right."""

    PANEL_W = 340
    BG = (18, 18, 22)
    FG = (232, 232, 238)
    DIM = (140, 140, 152)
    HUMAN = (90, 210, 130)
    POLICY = (245, 175, 70)
    ALERT = (235, 90, 90)

    def __init__(self, cfg: TeleopConfig, main_size: int = 512):
        import pygame
        self.pg = pygame
        self.cfg = cfg
        pygame.init()
        pygame.key.set_repeat(0)  # we read held state ourselves
        scale = float(cfg.window_scale)
        self.main = int(main_size * scale)
        self.side = int(self.main * 0.45) if cfg.show_second_view else 0
        w = self.main + self.PANEL_W + (self.side + 8 if self.side else 0)
        h = max(self.main, self.side * 2 + 8)
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("DAgger teleoperation")
        self.font = self._font(15)
        self.font_sm = self._font(12)
        self.font_lg = self._font(21, bold=True)
        self.show_help = True
        self.clock = pygame.time.Clock()
        # Exposed so callers (SessionRunner) can compare wait_for_key results
        # without importing pygame -- keeps the loops runnable headlessly.
        self.K_ESCAPE = pygame.K_ESCAPE
        self.K_RETURN = pygame.K_RETURN

    def pump(self) -> None:
        self.pg.event.pump()

    def _font(self, size: int, bold: bool = False):
        for name in ("menlo", "dejavusansmono", "couriernew", "monospace"):
            try:
                f = self.pg.font.SysFont(name, size, bold=bold)
                if f is not None:
                    return f
            except Exception:
                continue
        return self.pg.font.Font(None, size)

    # ------------------------------------------------------------------ drawing
    def _blit_image(self, img: np.ndarray, pos: Tuple[int, int], size: int) -> None:
        surf = self.pg.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        if surf.get_width() != size:
            surf = self.pg.transform.smoothscale(surf, (size, size))
        self.screen.blit(surf, pos)

    def draw(self, main_img: np.ndarray, hud: Dict[str, object],
             side_imgs: Optional[List[np.ndarray]] = None) -> None:
        pg = self.pg
        self.screen.fill(self.BG)
        self._blit_image(main_img, (0, 0), self.main)

        x = self.main + 8
        if self.side and side_imgs:
            for i, im in enumerate(side_imgs[:2]):
                self._blit_image(im, (x, i * (self.side + 8)), self.side)
            x += self.side + 8

        self._draw_panel(x, hud)
        pg.display.flip()
        if self.cfg.fps_cap:
            self.clock.tick(self.cfg.fps_cap)

    def _draw_panel(self, x: int, hud: Dict[str, object]) -> None:
        y = 10
        pad = 10

        def line(text: str, font=None, color=None, dy: Optional[int] = None):
            nonlocal y
            font = font or self.font
            surf = font.render(text, True, color or self.FG)
            self.screen.blit(surf, (x + pad, y))
            y += dy if dy is not None else font.get_height() + 2

        controller = str(hud.get("controller", "-"))
        is_human = controller.upper().startswith("YOU") or controller.upper() == "HUMAN"
        line(str(hud.get("mode", "")), self.font_lg, self.DIM)
        y += 2
        line(f"IN CONTROL: {controller}", self.font_lg,
             self.HUMAN if is_human else self.POLICY)
        y += 6

        for key in ("phase", "round", "episode", "step", "time", "gripper"):
            if key in hud:
                line(f"{key:>9}: {hud[key]}", self.font, self.FG)

        y += 6
        if "feedback" in hud and hud["feedback"]:
            self.pg.draw.line(self.screen, (60, 60, 70),
                              (x + pad, y), (x + self.PANEL_W - pad, y))
            y += 8
            line("POLICY PROGRESS", self.font_sm, self.DIM)
            for ln in str(hud["feedback"]).split("\n"):
                line(ln, self.font, self.FG)
            y += 4

        if "status" in hud and hud["status"]:
            y += 4
            for ln in str(hud["status"]).split("\n"):
                line(ln, self.font, self.ALERT)

        if self.show_help:
            y += 10
            self.pg.draw.line(self.screen, (60, 60, 70),
                              (x + pad, y), (x + self.PANEL_W - pad, y))
            y += 8
            line("CONTROLS  (H to hide)", self.font_sm, self.DIM)
            for k, desc in KEYMAP_HELP:
                surf = self.font_sm.render(f"{k:>10}  {desc}", True, self.DIM)
                self.screen.blit(surf, (x + pad, y))
                y += self.font_sm.get_height() + 1

    # ------------------------------------------------------------------ overlays
    def message(self, title: str, body: str = "",
                sub: str = "press ENTER to continue") -> None:
        """Full-window modal text screen (round breaks, instructions)."""
        pg = self.pg
        self.screen.fill(self.BG)
        w, h = self.screen.get_size()
        lines = [(title, self.font_lg, self.FG)]
        lines += [(ln, self.font, self.DIM) for ln in body.split("\n")]
        lines += [("", self.font, self.FG), (sub, self.font, self.HUMAN)]
        total = sum(f.get_height() + 6 for _, f, _ in lines)
        y = max(20, (h - total) // 2)
        for text, font, color in lines:
            surf = font.render(text, True, color)
            self.screen.blit(surf, ((w - surf.get_width()) // 2, y))
            y += font.get_height() + 6
        pg.display.flip()

    def wait_for_key(self, keys: Optional[List[int]] = None) -> int:
        """Block until one of `keys` (default ENTER/ESC) is pressed."""
        pg = self.pg
        keys = keys or [pg.K_RETURN, pg.K_ESCAPE]
        while True:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    return pg.K_ESCAPE
                if ev.type == pg.KEYDOWN and ev.key in keys:
                    return ev.key
            self.clock.tick(30)

    def close(self) -> None:
        try:
            self.pg.quit()
        except Exception:
            pass
