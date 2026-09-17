#!/usr/bin/env python
"""Launch the online study server.

    # local check
    python scripts/run_server.py --config configs/study_online.yaml \
        --checkpoint data/checkpoints/bc_rnn_Lift_study.pth --max-sessions 4

    # deployment (behind a TLS-terminating proxy; see deploy/)
    python scripts/run_server.py --host 0.0.0.0 --port 8000 \
        --config configs/study_online.yaml --max-sessions 8 \
        --admin-token "$DAGGER_ADMIN_TOKEN" --completion-code CXXXXXXX

Capacity: each participant is one OS process running MuJoCo, a policy and LoRA
finetuning. Budget roughly one core per one-to-two concurrent participants and
~400 MB RAM each, and about 1 Mbit/s of egress per participant at the default
384 px / 20 fps. `--max-sessions` is a hard cap; beyond it /api/enroll returns
503 and the client asks the participant to come back shortly, which is preferable
to queueing them into a slower, unequal session.
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dagger_lh.config import Config
from dagger_lh.study import Condition
from dagger_lh.web.manager import DEFAULT_ARMS, Arm


def parse_arms(spec: str) -> list:
    """`name:condition[:two_phase[:phase2]][ ,...]` -> list[Arm]."""
    if not spec:
        return DEFAULT_ARMS
    arms = []
    for chunk in spec.split(","):
        parts = [p.strip() for p in chunk.split(":") if p.strip()]
        if len(parts) < 2:
            raise SystemExit(f"bad --arms entry '{chunk}'")
        name, cond = parts[0], parts[1]
        two = len(parts) > 2 and parts[2].lower() in ("1", "true", "yes", "two_phase")
        p2 = parts[3] if len(parts) > 3 else Condition.RESPONSIVE
        arms.append(Arm(name, cond, two_phase=two, phase2_condition=p2))
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="base policy checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-sessions", type=int, default=4)
    ap.add_argument("--arms", default="",
                    help="override the design, e.g. "
                         "'ctl:responsive,nc:noncontingent,esc:noncontingent:1:responsive'")
    ap.add_argument("--admin-token", default=os.environ.get("DAGGER_ADMIN_TOKEN"))
    ap.add_argument("--state-dir", default="data/web")
    ap.add_argument("--ladder", default=None,
                    help="ladder.json from scripts/train_policy_ladder.py; "
                         "required if any arm uses the 'yoked' condition")
    ap.add_argument("--yoked-schedule", type=float, nargs="+", default=None,
                    help="success rate per round for 'yoked' arms")

    g = ap.add_argument_group("participant experience")
    g.add_argument("--frame-size", type=int, default=384)
    g.add_argument("--jpeg-quality", type=int, default=70)
    g.add_argument("--practice-episodes", type=int, default=2)
    g.add_argument("--practice-required", type=int, default=1,
                   help="practice successes needed to continue (0 = no gate)")
    g.add_argument("--attention-check-every", type=int, default=3,
                   help="insert a verifiable survey item every N surveys (0 = off)")
    g.add_argument("--minutes", type=int, default=25,
                   help="duration quoted on the consent page")
    g.add_argument("--completion-code", default="")
    g.add_argument("--screenout-code", default="")
    g.add_argument("--completion-url", default="",
                   help="e.g. https://app.prolific.com/submissions/complete?cc=XXXX")
    g.add_argument("--response-timeout", type=float, default=900.0)
    g.add_argument("--torch-threads", type=int, default=1,
                   help="per participant; keep low so sessions don't oversubscribe")

    q = ap.add_argument_group("Qualtrics integration (see docs/qualtrics.md)")
    q.add_argument("--survey-mode", choices=["native", "qualtrics"],
                   default="native",
                   help="where the BETWEEN-ROUND check-ins live. 'native' keeps "
                        "them in-app (recommended: a full-page round trip after "
                        "every round adds a lot of friction and breaks the round "
                        "rhythm). 'qualtrics' hands the tab over each round.")
    q.add_argument("--qualtrics-round-url", default="",
                   help="survey link for the between-round check-ins "
                        "(only used with --survey-mode qualtrics)")
    q.add_argument("--qualtrics-exit-url", default="",
                   help="survey link for the post-task battery; the task "
                        "redirects here when it finishes")
    q.add_argument("--skip-consent", action="store_true",
                   help="an intake questionnaire already took consent, so don't "
                        "ask twice")
    q.add_argument("--skip-debrief", action="store_true",
                   help="the exit questionnaire carries the debrief; requires "
                        "--qualtrics-exit-url")
    args = ap.parse_args()

    if args.survey_mode == "qualtrics" and not args.qualtrics_round_url:
        raise SystemExit("--survey-mode qualtrics requires --qualtrics-round-url")
    if args.skip_debrief and not args.qualtrics_exit_url:
        raise SystemExit(
            "--skip-debrief requires --qualtrics-exit-url: participants in the "
            "deceptive arms must be debriefed somewhere")

    cfg = Config.load(args.config)
    cfg.mode = "dagger"
    if args.checkpoint:
        cfg.base_checkpoint = args.checkpoint
    if args.ladder:
        cfg.study.ladder_path = args.ladder
    if args.yoked_schedule:
        cfg.study.yoked_schedule = list(args.yoked_schedule)
    arms_resolved = parse_arms(args.arms)
    if any(a.condition == "yoked" or a.phase2_condition == "yoked"
           for a in arms_resolved) and not cfg.study.ladder_path:
        raise SystemExit(
            "a 'yoked' arm needs --ladder (see scripts/train_policy_ladder.py)")
    if cfg.base_checkpoint and not os.path.exists(cfg.base_checkpoint):
        print(f"warning: checkpoint {cfg.base_checkpoint} not found; participants "
              f"would teach a randomly initialised policy", file=sys.stderr)

    token = args.admin_token or secrets.token_urlsafe(24)
    options = {
        "frame_size": args.frame_size,
        "jpeg_quality": args.jpeg_quality,
        "practice_episodes": args.practice_episodes,
        "practice_required": args.practice_required,
        "attention_check_every": args.attention_check_every,
        "minutes": args.minutes,
        "completion_code": args.completion_code,
        "screenout_code": args.screenout_code or args.completion_code,
        "completion_url": args.completion_url,
        "response_timeout": args.response_timeout,
        "torch_threads": args.torch_threads,
        "survey_mode": args.survey_mode,
        "qualtrics_round_url": args.qualtrics_round_url,
        "qualtrics_exit_url": args.qualtrics_exit_url,
        "skip_consent": args.skip_consent,
        "skip_debrief": args.skip_debrief,
    }

    from dagger_lh.web.server import create_app
    app = create_app(cfg, max_sessions=args.max_sessions,
                     arms=arms_resolved, admin_token=token,
                     options=options, state_dir=args.state_dir)

    print(f"task        {cfg.env.env_name} / {cfg.policy.algo}")
    print(f"checkpoint  {cfg.base_checkpoint}")
    print(f"arms        {[a.name for a in arms_resolved]}")
    if cfg.study.ladder_path:
        from dagger_lh.ladder import PolicyLadder
        print(f"ladder      {PolicyLadder(cfg.study.ladder_path).describe()}")
    print(f"capacity    {args.max_sessions} concurrent participants")
    print(f"surveys     {args.survey_mode}"
          + (f" -> {args.qualtrics_round_url}" if args.survey_mode == "qualtrics" else ""))
    if args.qualtrics_exit_url:
        print(f"exit survey {args.qualtrics_exit_url}")
    if args.skip_consent:
        print("consent     handled by the intake questionnaire (skipped in-app)")
    if args.skip_debrief:
        print("debrief     handled by the exit questionnaire (skipped in-app)")
    print(f"participant http://{args.host}:{args.port}/")
    print(f"admin        http://{args.host}:{args.port}/admin?token={token}")
    if not args.admin_token:
        print("            (generated; set --admin-token or $DAGGER_ADMIN_TOKEN "
              "to keep it stable)")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ws_ping_interval=20, ws_ping_timeout=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
