#!/usr/bin/env python
"""Exercise the real pygame display and keyboard device.

scripts/smoke_test.py substitutes a null display, so it never touches the code a
participant actually interacts with. This does: it builds the real
`OperatorDisplay`, renders real environment frames with the HUD, and drives the
real `KeyboardDevice` with synthetic key events to check the bindings.

Runs offscreen by default (SDL dummy driver), so it works over SSH and in CI.
Pass --window to open a visible window and try the keys by hand.

    python scripts/test_display.py            # offscreen assertions
    python scripts/test_display.py --window   # interactive, 15s
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", action="store_true",
                    help="open a real window and let the operator try the keys")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--env", default="Lift")
    args = ap.parse_args()

    if not args.window:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import numpy as np
    import pygame

    from dagger_lh.config import EnvConfig, TeleopConfig
    from dagger_lh.envs import make_env, silence_robosuite_logs
    from dagger_lh.teleop import KEYMAP_HELP, KeyboardDevice, OperatorDisplay

    silence_robosuite_logs()
    env = make_env(EnvConfig(env_name=args.env, horizon=100_000))
    obs = env.reset()
    tcfg = TeleopConfig()
    disp = OperatorDisplay(tcfg, main_size=512)
    kb = KeyboardDevice(tcfg, action_dim=env.action_dim)
    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))

    print(f"window {disp.screen.get_size()}, {len(KEYMAP_HELP)} documented keys")

    if args.window:
        print(f"\nTry the controls. Running for {args.seconds:.0f}s "
              f"(ESC or ENTER to stop early).")
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            st = kb.poll()
            if st.event_quit or st.event_success:
                break
            obs, _, done, info = env.step(st.action)
            if done:
                obs = env.reset()
            main = env.render_operator()
            side = env.render_operator(camera="robot0_eye_in_hand",
                                       height=192, width=192)
            disp.draw(main, {
                "mode": "DISPLAY TEST",
                "controller": "YOU" if st.intervening else "ROBOT",
                "round": "-", "step": f"{env.t}", "phase": "-",
                "time": f"{time.time() - t0:5.1f}s",
                "gripper": "closed" if st.gripper_closed else "open",
                "status": "move with WASD/RF, arrows, Q/E",
            }, [side])
            if st.event_toggle_help:
                disp.show_help = not disp.show_help
        print(f"  keypresses recorded: {kb.total_keypresses}")
        print(f"  engaged steps: {kb.total_engaged_steps}/{kb.total_steps}")
    else:
        main = env.render_operator()
        side = env.render_operator(camera="robot0_eye_in_hand", height=192, width=192)
        for i in range(3):
            disp.draw(main, {
                "mode": "DAgger", "controller": "YOU" if i % 2 else "ROBOT",
                "phase": "1", "round": "2/10", "step": f"{i}/400",
                "time": "3.2s", "gripper": "open",
                "feedback": "round 2/10\nsuccess this round: 40%\nchange vs last: up",
                "status": "",
            }, [side])
        check("HUD renders with both camera views", True, "3 frames")
        disp.message("Updating the robot's policy", "Learning...", "")
        check("modal message renders", True)

        expected = [
            (pygame.K_TAB, "take control", "event_toggle_control"),
            (pygame.K_SPACE, "gripper toggle", None),
            (pygame.K_x, "give up", "event_giveup"),
            (pygame.K_BACKSPACE, "discard", "event_discard"),
            (pygame.K_RETURN, "success", "event_success"),
            (pygame.K_h, "toggle help", "event_toggle_help"),
            (pygame.K_ESCAPE, "quit", "event_quit"),
        ]
        for key, label, flag in expected:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
            st = kb.poll()
            ok = getattr(st, flag) if flag else st.gripper_closed
            check(f"{label:16s} binding fires", bool(ok))
        check("TAB granted control authority", kb.intervening)
        check("keypresses counted", kb.total_keypresses == len(expected),
              str(kb.total_keypresses))
        st = kb.poll()
        check("action dim matches env", st.action.shape[0] == env.action_dim,
              f"{st.action.shape[0]} vs {env.action_dim}")
        check("action within bounds", bool(np.all(np.abs(st.action) <= 1.0)))

    disp.close()
    env.close()
    if not args.window:
        print("\n" + ("ALL CHECKS PASSED" if not failures
                      else f"{failures} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
