#!/usr/bin/env python
"""Aggregate study sessions and test for learned helplessness.

Reads every `rounds.csv` under a sessions directory, pools them by condition, and
reports the effort trajectory plus the two contrasts the design is built around:

  1. between-condition  -- does non-contingent feedback suppress effort relative
     to the responsive control arm?
  2. escape test        -- in two-phase sessions, does effort recover in phase 2
     once control is restored? Failure to recover is the helplessness signature.

    python scripts/analyze.py --sessions data/sessions --plot out.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

EFFORT_MEASURES = [
    ("intervention_rate", "intervention rate", True),
    ("engagement_rate", "engagement rate", True),
    ("keypresses_per_step", "keypresses / step", True),
    ("passivity", "passivity (idle while in control)", False),
    ("success_rate", "task success rate", True),
]


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_sessions(root: str) -> List[Dict[str, Any]]:
    """One record per session: its config, per-round rows and summary."""
    out = []
    for rounds_csv in sorted(glob.glob(os.path.join(root, "**", "rounds.csv"),
                                       recursive=True)):
        sdir = os.path.dirname(rounds_csv)
        with open(rounds_csv) as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        if not rows:
            continue
        summary = {}
        spath = os.path.join(sdir, "summary.json")
        if os.path.exists(spath):
            try:
                summary = json.load(open(spath))
            except json.JSONDecodeError:
                pass
        two_phase = bool(summary.get("two_phase", False))
        cond = rows[0].get("condition", summary.get("condition", "?"))
        out.append({
            "dir": sdir,
            "participant": summary.get("participant_id",
                                       os.path.basename(sdir).split("_")[0]),
            # Two-phase (escape-test) sessions are a different design from
            # single-phase ones even when phase 1 uses the same condition, so
            # they get their own group rather than being pooled.
            "condition": f"{cond}+escape" if two_phase else cond,
            "phase1_condition": cond,
            "two_phase": two_phase,
            "rows": rows,
            "summary": summary,
        })
    return out


def welch_t(a: List[float], b: List[float]) -> Dict[str, float]:
    """Welch's t-test (unequal variances). Returns nan when underpowered."""
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "n1": len(a), "n2": len(b)}
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "n1": n1, "n2": n2}
    t = (m1 - m2) / np.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except ImportError:
        pass
    return {"t": float(t), "df": float(df), "p": p,
            "mean1": float(m1), "mean2": float(m2), "n1": n1, "n2": n2}


def cohens_d(a: List[float], b: List[float]) -> float:
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
                / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / s) if s > 0 else float("nan")


def report(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for s in sessions:
        by_cond[s["condition"]].append(s)

    print(f"\nloaded {len(sessions)} session(s) across "
          f"{len(by_cond)} condition(s)")
    for cond, ss in sorted(by_cond.items()):
        parts = sorted({s['participant'] for s in ss})
        print(f"  {cond:15s} {len(ss)} session(s), {len(parts)} participant(s)")

    out: Dict[str, Any] = {"n_sessions": len(sessions), "conditions": {}}

    # ---------------- per-condition effort trajectories ----------------
    print("\n" + "=" * 74)
    print("EFFORT TRAJECTORY BY CONDITION (mean per round)")
    print("=" * 74)
    for cond, ss in sorted(by_cond.items()):
        n_rounds = max(len(s["rows"]) for s in ss)
        print(f"\n[{cond}]  n={len(ss)}")
        header = "  measure                        " + "".join(
            f"{'r' + str(i + 1):>8}" for i in range(n_rounds))
        print(header)
        cond_out: Dict[str, Any] = {"n_sessions": len(ss)}
        for key, label, _ in EFFORT_MEASURES:
            per_round = []
            for i in range(n_rounds):
                vals = [_f(s["rows"][i].get(key)) for s in ss if i < len(s["rows"])]
                vals = [v for v in vals if np.isfinite(v)]
                per_round.append(float(np.mean(vals)) if vals else float("nan"))
            cond_out[key] = per_round
            cells = "".join(f"{v:>8.3f}" if np.isfinite(v) else f"{'-':>8}"
                            for v in per_round)
            print(f"  {label:30s}{cells}")
        idx = [_f(s["summary"].get("index")) for s in ss]
        idx = [v for v in idx if np.isfinite(v)]
        if idx:
            cond_out["helplessness_index"] = idx
            print(f"  {'helplessness index':30s}{np.mean(idx):>8.3f}"
                  f"   (per-session: {', '.join(f'{v:+.2f}' for v in idx)})")
        gu = [_f(s["summary"].get("total_giveups")) for s in ss]
        gu = [v for v in gu if np.isfinite(v)]
        if gu:
            cond_out["total_giveups"] = float(np.sum(gu))
            print(f"  {'give-up events (total)':30s}{np.sum(gu):>8.0f}")
        out["conditions"][cond] = cond_out

    # ---------------- contrast 1: responsive vs non-contingent ----------------
    ctrl = by_cond.get("responsive", [])
    for arm in ("noncontingent", "degraded", "yoked"):
        exp = by_cond.get(arm, [])
        if not (ctrl and exp):
            continue
        print("\n" + "=" * 74)
        print(f"CONTRAST: {arm} vs responsive")
        print("=" * 74)
        arm_out = {}
        for key, label, _ in EFFORT_MEASURES + [("index", "helplessness index", False)]:
            if key == "index":
                a = [_f(s["summary"].get("index")) for s in exp]
                b = [_f(s["summary"].get("index")) for s in ctrl]
            else:
                # session-level mean of the measure
                a = [float(np.nanmean([_f(r.get(key)) for r in s["rows"]])) for s in exp]
                b = [float(np.nanmean([_f(r.get(key)) for r in s["rows"]])) for s in ctrl]
            st = welch_t(a, b)
            d = cohens_d(a, b)
            arm_out[key] = {**st, "cohens_d": d}
            if np.isfinite(st.get("t", float("nan"))):
                pstr = f"p={st['p']:.4f}" if np.isfinite(st["p"]) else "p=n/a (install scipy)"
                print(f"  {label:30s} {arm}={st['mean1']:+.3f}  "
                      f"responsive={st['mean2']:+.3f}  d={d:+.2f}  {pstr}")
            else:
                print(f"  {label:30s} underpowered "
                      f"(n={len(a)} vs {len(b)}; need >=2 per arm)")
        out[f"contrast_{arm}_vs_responsive"] = arm_out

    # ---------------- contrast 2: escape test ----------------
    two = [s for s in sessions if s["two_phase"]]
    if two:
        print("\n" + "=" * 74)
        print("ESCAPE TEST (phase 1 induction -> phase 2 control restored)")
        print("=" * 74)
        esc = {}
        for s in two:
            sm = s["summary"]
            p1i, p2i = _f(sm.get("phase1_intervention_rate")), _f(sm.get("phase2_intervention_rate"))
            p1e, p2e = _f(sm.get("phase1_engagement_rate")), _f(sm.get("phase2_engagement_rate"))
            p1s, p2s = _f(sm.get("phase1_success_rate")), _f(sm.get("phase2_success_rate"))
            print(f"\n  {s['participant']}  ({sm.get('condition')} -> "
                  f"{sm.get('phase2_condition')})")
            print(f"    intervention rate  {p1i:.3f} -> {p2i:.3f}  "
                  f"({p2i - p1i:+.3f})")
            print(f"    engagement rate    {p1e:.3f} -> {p2e:.3f}  "
                  f"({p2e - p1e:+.3f})")
            print(f"    success rate       {p1s:.3f} -> {p2s:.3f}  "
                  f"({p2s - p1s:+.3f})")
            recovered = np.isfinite(p2e) and np.isfinite(p1e) and p2e > p1e
            print(f"    -> effort {'RECOVERED' if recovered else 'did NOT recover'}"
                  f" when control was restored")
            esc[s["participant"]] = {
                "phase1_intervention_rate": p1i, "phase2_intervention_rate": p2i,
                "phase1_engagement_rate": p1e, "phase2_engagement_rate": p2e,
                "recovered": bool(recovered),
            }
        out["escape_test"] = esc
        rec = [v["recovered"] for v in esc.values()]
        print(f"\n  recovery: {sum(rec)}/{len(rec)} participants")

    # ---------------- surveys ----------------
    survey_keys = sorted({k for s in sessions for r in s["rows"]
                          for k in r if k.startswith("survey_")})
    if survey_keys:
        print("\n" + "=" * 74)
        print("SELF-REPORT BY CONDITION (1-7)")
        print("=" * 74)
        surv = {}
        for cond, ss in sorted(by_cond.items()):
            print(f"\n[{cond}]")
            surv[cond] = {}
            for k in survey_keys:
                vals = [_f(r.get(k)) for s in ss for r in s["rows"]]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    # first vs last response shows drift over the session
                    firsts = [_f(s["rows"][0].get(k)) for s in ss]
                    lasts = [_f(s["rows"][-1].get(k)) for s in ss]
                    firsts = [v for v in firsts if np.isfinite(v)]
                    lasts = [v for v in lasts if np.isfinite(v)]
                    drift = (np.mean(lasts) - np.mean(firsts)) if firsts and lasts else float("nan")
                    surv[cond][k] = {"mean": float(np.mean(vals)), "drift": float(drift)}
                    print(f"  {k[7:]:22s} mean={np.mean(vals):.2f}  "
                          f"first->last {drift:+.2f}")
        out["surveys"] = surv
    return out


def make_plot(sessions: List[Dict[str, Any]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for s in sessions:
        by_cond[s["condition"]].append(s)

    keys = [k for k, _, _ in EFFORT_MEASURES]
    labels = {k: l for k, l, _ in EFFORT_MEASURES}
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6),
                             squeeze=False)
    colors = {"responsive": "#2a9d5c", "noncontingent": "#c0392b",
              "degraded": "#8e44ad", "yoked": "#d68910",
              "noncontingent+escape": "#e67e22", "responsive+escape": "#16a085",
              "degraded+escape": "#5b2c6f"}
    for ax, key in zip(axes[0], keys):
        for cond, ss in sorted(by_cond.items()):
            n_rounds = max(len(s["rows"]) for s in ss)
            xs, ms, es = [], [], []
            for i in range(n_rounds):
                vals = [_f(s["rows"][i].get(key)) for s in ss if i < len(s["rows"])]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    xs.append(i + 1)
                    ms.append(np.mean(vals))
                    es.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
            if xs:
                ax.errorbar(xs, ms, yerr=es, marker="o", capsize=3,
                            label=cond, color=colors.get(cond), lw=1.8, ms=4)
        ax.set_title(labels[key], fontsize=10)
        ax.set_xlabel("DAgger round")
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("value (mean +/- s.e.)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Operator effort across DAgger rounds by contingency condition",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", default="data/sessions")
    ap.add_argument("--plot", default=None, help="write a PNG of the trajectories")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    args = ap.parse_args()

    sessions = load_sessions(args.sessions)
    if not sessions:
        print(f"no sessions with rounds.csv found under {args.sessions}",
              file=sys.stderr)
        return 1
    out = report(sessions)
    if args.plot:
        make_plot(sessions, args.plot)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
