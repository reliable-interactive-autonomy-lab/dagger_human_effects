#!/usr/bin/env python
"""Join a Qualtrics export to the behavioural sessions.

The questionnaire responses live in Qualtrics and the behaviour lives in
`data/sessions/*/rounds.csv`. This merges them on a shared participant key and
writes one analysis-ready table, so the effort measures and the self-report
scales can be modelled together.

    python scripts/merge_qualtrics.py \
        --sessions data/sessions \
        --qualtrics exports/intake.csv exports/exit.csv \
        --out merged.csv

Qualtrics CSV exports carry two header rows (column names, then the question
text) plus a JSON metadata row; `--qualtrics-header-rows` controls how many are
skipped and defaults to the standard 3-row export.

The join key is whatever the task recorded as `external_id` (the Prolific id when
recruiting there, otherwise the id the intake questionnaire passed through), and
it is matched against `--key-column` in the export.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_qualtrics(path: str, header_rows: int = 3,
                   key_column: str = "external_id") -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Return (fieldnames, {key: row}) from a Qualtrics CSV export."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], {}
    fields = rows[0]
    body = rows[max(1, header_rows):]
    if key_column not in fields:
        raise SystemExit(
            f"{path}: no '{key_column}' column. Columns present:\n  "
            + ", ".join(fields[:40])
            + "\n\nDeclare it as Embedded Data in the survey flow and pass it in "
              "the link (see docs/qualtrics.md), or pick another --key-column.")
    kidx = fields.index(key_column)
    out: Dict[str, Dict[str, str]] = {}
    dupes = 0
    for r in body:
        if len(r) <= kidx:
            continue
        key = (r[kidx] or "").strip()
        if not key:
            continue
        row = {fields[i]: (r[i] if i < len(r) else "") for i in range(len(fields))}
        if key in out:
            dupes += 1
            # Keep the most recent: a participant who restarted a questionnaire
            # leaves an earlier partial response behind.
            prev, cur = out[key].get("RecordedDate", ""), row.get("RecordedDate", "")
            if cur < prev:
                continue
        out[key] = row
    if dupes:
        print(f"  note: {dupes} duplicate key(s) in {os.path.basename(path)}; "
              f"kept the latest RecordedDate for each")
    return fields, out


def read_sessions(root: str) -> List[Dict[str, Any]]:
    """One record per session: join keys plus its per-round rows."""
    sessions = []
    for rounds_csv in sorted(glob.glob(os.path.join(root, "**", "rounds.csv"),
                                       recursive=True)):
        sdir = os.path.dirname(rounds_csv)
        keys: Dict[str, Any] = {}
        jpath = os.path.join(sdir, "join_keys.json")
        if os.path.exists(jpath):
            try:
                keys = json.load(open(jpath))
            except json.JSONDecodeError:
                pass
        if not keys:
            # Older sessions, or a local run: recover what we can from the log.
            epath = os.path.join(sdir, "events.jsonl")
            if os.path.exists(epath):
                for line in open(epath):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") in ("web_session", "session_start"):
                        keys = {
                            "participant_id": rec.get("participant_id", ""),
                            "external_id": rec.get("external_id", ""),
                            "condition": rec.get("condition", ""),
                        }
                        break
        with open(rounds_csv, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        if not rows:
            continue
        sessions.append({"dir": sdir, "keys": keys, "rows": rows})
    return sessions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", default="data/sessions")
    ap.add_argument("--qualtrics", nargs="+", required=True,
                    help="one or more Qualtrics CSV exports (intake, exit, ...)")
    ap.add_argument("--out", default="merged.csv")
    ap.add_argument("--key-column", default="external_id",
                    help="the Embedded Data column holding the join key")
    ap.add_argument("--qualtrics-header-rows", type=int, default=3,
                    help="header rows to skip in the export (Qualtrics uses 3)")
    ap.add_argument("--per-round", action="store_true",
                    help="emit one row per DAgger round (default: one per session)")
    ap.add_argument("--drop-unmatched", action="store_true",
                    help="omit sessions with no questionnaire response")
    args = ap.parse_args()

    sessions = read_sessions(args.sessions)
    if not sessions:
        print(f"no sessions under {args.sessions}", file=sys.stderr)
        return 1
    print(f"sessions: {len(sessions)}")

    exports: List[Tuple[str, Dict[str, Dict[str, str]]]] = []
    for path in args.qualtrics:
        tag = os.path.splitext(os.path.basename(path))[0]
        _, rows = read_qualtrics(path, args.qualtrics_header_rows, args.key_column)
        print(f"  {tag}: {len(rows)} response(s)")
        exports.append((tag, rows))

    merged: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    for s in sessions:
        key = str(s["keys"].get("external_id")
                  or s["keys"].get("participant_id") or "").strip()
        base: Dict[str, Any] = {
            "session_dir": s["dir"],
            "participant_id": s["keys"].get("participant_id", ""),
            "external_id": key,
            "condition": s["keys"].get("condition", ""),
            "two_phase": s["keys"].get("two_phase", ""),
        }
        hits = 0
        for tag, rows in exports:
            row = rows.get(key)
            if row is None:
                continue
            hits += 1
            for k, v in row.items():
                if k == args.key_column:
                    continue
                base[f"{tag}__{k}"] = v
        if hits == 0:
            unmatched.append(key or s["dir"])
            if args.drop_unmatched:
                continue
        base["qualtrics_sources_matched"] = hits

        if args.per_round:
            for r in s["rows"]:
                merged.append({**base, **{f"round__{k}": v for k, v in r.items()}})
        else:
            # Session-level: the summary plus each measure averaged over rounds.
            import statistics
            numeric = ["intervention_rate", "engagement_rate", "passivity",
                       "keypresses_per_step", "success_rate"]
            row = dict(base)
            row["n_rounds"] = len(s["rows"])
            for col in numeric:
                vals = []
                for r in s["rows"]:
                    try:
                        vals.append(float(r.get(col, "")))
                    except (TypeError, ValueError):
                        pass
                row[f"mean_{col}"] = (round(statistics.fmean(vals), 6)
                                      if vals else "")
            spath = os.path.join(s["dir"], "summary.json")
            if os.path.exists(spath):
                try:
                    for k, v in json.load(open(spath)).items():
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            row[f"summary__{k}"] = v
                except json.JSONDecodeError:
                    pass
            merged.append(row)

    if not merged:
        print("nothing to write", file=sys.stderr)
        return 1

    fields: List[str] = []
    for row in merged:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in merged:
            w.writerow(row)

    print(f"\nwrote {len(merged)} row(s) x {len(fields)} column(s) -> {args.out}")
    if unmatched:
        print(f"WARNING  {len(unmatched)} session(s) had no questionnaire response:")
        for k in unmatched[:10]:
            print(f"  {k}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
        print("  Usually means the participant abandoned the questionnaire, or "
              "the Embedded Data field name does not match --key-column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
