# Running the questionnaires in Qualtrics

The recommended shape is **Qualtrics → task → Qualtrics**: Qualtrics owns consent,
demographics, the baseline scales, the post-task battery and the debrief; the task
server owns the robot and the per-round check-ins.

```
Prolific  ──►  Qualtrics intake        ──►  task server        ──►  Qualtrics exit  ──►  Prolific
               consent, demographics,       teaching rounds +        post battery,       completion
               baseline scales              per-round check-ins      debrief
```

## Why the task can't just embed the survey

Qualtrics serves `X-Frame-Options: SAMEORIGIN`, so a Qualtrics survey cannot be
shown in an iframe inside the task page. That is set on Qualtrics's servers and
there is no way around it from our side. Every integration below therefore uses a
**full-page redirect**, which also avoids the third-party-cookie breakage that
iframed Qualtrics surveys suffer in current browsers.

## Decide where the per-round check-ins live

The six between-round items (perceived control, effectiveness, expected success,
frustration, persistence, effort) are the self-report half of the helplessness
measure, and they are asked after **every** round.

| | `--survey-mode native` (default) | `--survey-mode qualtrics` |
|---|---|---|
| Where | in the task page | full-page redirect each round |
| Cost per round | ~15 s | ~15 s + two page loads + Qualtrics load |
| Round rhythm | preserved | interrupted 8 times |
| Data lands in | `rounds.csv`, ready for `analyze.py` | Qualtrics, joined afterwards |
| Attention check | `--attention-check-every N` | build it in Qualtrics |

**Prefer `native` for the per-round items** and use Qualtrics for the intake and
exit batteries. Eight full-page round trips add real friction, and the
between-round moment is exactly where the manipulation is supposed to be felt —
you do not want a page transition in the middle of it. The Qualtrics mode exists
for when a protocol or IRB requires every item to be Qualtrics-hosted.

Either way the answers end up joined by `scripts/merge_qualtrics.py`.

---

## 1. Intake survey (Qualtrics)

Put consent, demographics and any baseline scales here.

**Survey flow → Add Below → Embedded Data**, and declare these with no value
(they read "Value will be set from Panel or URL"):

```
PROLIFIC_PID
STUDY_ID
SESSION_ID
```

**Survey flow → End of Survey → Customize → Redirect to a URL:**

```
https://study.example.edu/?external_id=${e://Field/PROLIFIC_PID}&STUDY_ID=${e://Field/STUDY_ID}&source=qualtrics_intake
```

`external_id` becomes the participant id on the task server and the join key for
every export. If you are not recruiting through Prolific, use
`${e://Field/ResponseID}` instead — it is unique per response:

```
https://study.example.edu/?external_id=${e://Field/ResponseID}&source=qualtrics_intake
```

Then start the server with consent already handled:

```bash
python scripts/run_server.py --config configs/study_online.yaml \
    --checkpoint data/checkpoints/bc_rnn_Lift_study.pth \
    --skip-consent \
    --qualtrics-exit-url "https://iu.qualtrics.com/jfe/form/SV_exitSurveyId"
```

## 2. Exit survey (Qualtrics)

The task redirects here when the session ends, appending the join keys:

```
https://iu.qualtrics.com/jfe/form/SV_exitSurveyId
    ?participant_id=...&external_id=...&stage=exit&task=complete
```

Declare these as Embedded Data in the exit survey, again with no value:

```
participant_id
external_id
stage
task
```

Finish with **End of Survey → Redirect to a URL** pointing at your Prolific
completion link:

```
https://app.prolific.com/submissions/complete?cc=YOUR_CODE
```

### Putting the debrief in Qualtrics

If the exit survey carries the debrief text, add `--skip-debrief` so participants
are not debriefed twice. The launcher refuses `--skip-debrief` without
`--qualtrics-exit-url`, because participants in the deceptive arms have to be
debriefed *somewhere*.

Copy the disclosure text from `DEBRIEF_DECEPTION` in `dagger_lh/web/worker.py` as
a starting point. **Branch it on condition**, otherwise participants in the
`responsive` arm are told about a manipulation that did not apply to them.

The task does not send the condition to the browser, so Qualtrics cannot branch on
it directly. Two options:

- Debrief in the task (drop `--skip-debrief`); it already branches correctly.
- Show the fuller disclosure to everyone, worded to cover both cases ("participants
  were randomly assigned; in some conditions…"). This is normal practice and
  usually the simplest thing an IRB will accept.

Do not try to pass the arm through the browser to make Qualtrics branch — the
participant can read it in the URL.

## 3. Per-round surveys in Qualtrics (optional)

```bash
python scripts/run_server.py ... \
    --survey-mode qualtrics \
    --qualtrics-round-url "https://iu.qualtrics.com/jfe/form/SV_roundSurveyId"
```

Each round the tab goes to that link with:

```
?participant_id=...&external_id=...&round=3&stage=round
```

Declare `participant_id`, `external_id`, `round`, `stage` as Embedded Data.

**The round survey must send the participant back**, with `returned=1` so the task
resumes instead of handing them out again:

```
https://study.example.edu/?returned=1
```

The task keeps the session id in `sessionStorage`, so the return lands in the same
session — the worker was parked waiting, and replays the screen it was on. Tell
participants not to close the tab; the redirect screen says so already.

> Because `round` is an embedded field, one Qualtrics survey serves all rounds and
> you get one response row per round. Join with `--per-round`.

## 4. Merging the data

Export each Qualtrics survey as CSV (**Export & Import → Export Data → CSV**, with
"Use choice text" or numeric values as you prefer), then:

```bash
python scripts/merge_qualtrics.py \
    --sessions data/sessions \
    --qualtrics exports/intake.csv exports/exit.csv \
    --out merged.csv
```

- one row per session, with each behavioural measure averaged over rounds, plus
  everything from `summary.json`
- `--per-round` gives one row per DAgger round instead (use this when the round
  survey is in Qualtrics)
- columns are namespaced by export filename: `intake__age`, `exit__frustration_post`
- Qualtrics's three header rows are skipped automatically (`--qualtrics-header-rows`)
- duplicate keys keep the latest `RecordedDate`, since a restarted questionnaire
  leaves a partial response behind
- sessions with no matching response are reported rather than silently dropped
  (`--drop-unmatched` to drop them)

The join key comes from `join_keys.json`, written into every session directory.

## Checklist before launching

- [ ] `external_id` is declared as Embedded Data in **both** surveys and appears in the export
- [ ] intake redirect points at the task server and passes `external_id`
- [ ] `--qualtrics-exit-url` is set, and the exit survey declares its embedded fields
- [ ] exit survey redirects to the Prolific completion link
- [ ] debrief happens exactly once — in the task, or in the exit survey with `--skip-debrief`
- [ ] debrief text does not tell `responsive` participants they were deceived
- [ ] if using `--survey-mode qualtrics`, the round survey redirects back with `returned=1`
- [ ] walked the whole chain yourself once, end to end, and confirmed a row comes
      out of `merge_qualtrics.py` with both questionnaire and behavioural columns
