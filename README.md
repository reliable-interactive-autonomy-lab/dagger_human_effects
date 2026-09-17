# Learned helplessness in interactive imitation learning

A keyboard-teleoperation DAgger pipeline on **robosuite** + **robomimic**, built to
run a study on whether operators become *learned helpless* when their corrections
stop improving the robot.

An operator watches a policy attempt a manipulation task, takes control when it
fails, and the policy is LoRA-finetuned on their corrections between attempts.
The experimental manipulation is whether that finetuning is real.

- **Two modes.** `demo` — pure teleoperation, human drives every step.
  `dagger` — policy drives, human intervenes, policy is finetuned each round.
- **Reference policies.** BC-RNN is robomimic's own `BC_RNN_GMM`; ACT is
  LeRobot's reference `ACT`. Neither is reimplemented here.
- **LoRA finetuning** between rounds: ~2-3% of weights trainable, a few seconds
  per round on an M-series laptop.
- **Sham updates that are exactly reversible**, so non-contingency can be
  manipulated without changing compute or perceived latency.

---

## Quick start

```bash
bash scripts/setup_env.sh                 # creates/repairs the conda env
conda activate dagger_learned_helplessness

bash scripts/download_data.sh             # robomimic's official Lift dataset (+validate)
python scripts/smoke_test.py              # headless end-to-end check, no display needed
python scripts/test_display.py            # display + key bindings (add --window to try them)

# reproduce the published BC-RNN result (Lift PH -> 100% success)
python scripts/pretrain.py \
    --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
    --algo bc_rnn --epochs 600 --eval-every 100 --eval-episodes 20

# ...but for a STUDY you want a policy with headroom, not an expert (see below)
python scripts/pretrain.py \
    --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
    --algo bc_rnn --limit-demos 8 --target-success 0.30 --eval-episodes 30 \
    --out data/checkpoints/bc_rnn_Lift_study.pth

# run a session
python scripts/run_session.py --mode dagger --config configs/study_responsive.yaml \
    --participant p001 --checkpoint data/checkpoints/bc_rnn_Lift_base.pth
```

## Controls

A pygame window shows the robot with a HUD on the right. Held keys give
continuous velocity commands; the HUD always states who is in control.

| Keys | Action |
|---|---|
| `W` / `S` | move +X / −X (forward / back) |
| `A` / `D` | move +Y / −Y (left / right) |
| `R` / `F` | move +Z / −Z (up / down) |
| `←` / `→` | roll |
| `↑` / `↓` | pitch |
| `Q` / `E` | yaw |
| `SPACE` | toggle gripper open/closed |
| `TAB` | **take / release control** (hold-to-drive with `--hold-to-intervene`) |
| `ENTER` | episode succeeded — save it |
| `BACKSPACE` | discard episode and retry |
| `X` | give up on this episode (**logged as a give-up event**) |
| `P` / `H` / `ESC` | pause / toggle help / quit |

pygame is used rather than robosuite's built-in `Keyboard` device because it needs
no macOS accessibility permissions, gives real held-key state for smooth control,
and provides a surface for the HUD the study depends on.

### Reproducing the reference numbers

BC-RNN on robomimic's Lift PH low-dim dataset reaches **100% success (mean 44
steps) in 100 epochs**, matching the published result. `scripts/pretrain.py`
prints rollout success as it trains, so this is checkable rather than asserted.

### Choose the base policy's competence deliberately

An expert base policy **breaks the experiment**. If the policy already succeeds
every time, no correction the operator gives can improve it — so the `responsive`
arm becomes non-contingent too, and the manipulation has no contrast. Conversely
a policy that always fails gives corrections nothing to build on.

`--target-success` trains in slices and stops as soon as a target is reached:

```bash
python scripts/pretrain.py --dataset ... --limit-demos 8 \
    --target-success 0.30 --eval-slice 5 --eval-episodes 30
```

`--limit-demos` also leaves the rest of the dataset held out, which
`scripts/validate_finetune.py --skip-demos` then uses as "corrections" the base
policy has genuinely not seen. Aim for roughly 20-40% base success on Lift.

### Validated: contingent finetuning actually works

`scripts/validate_finetune.py` checks the assumption the whole study rests on —
that a few hundred LoRA steps on a handful of fresh corrections produce an
improvement the operator can *see within one round*. If it did not, the
contingent arm would be indistinguishable from the sham one and the experiment
would measure nothing.

Measured on Lift, BC-RNN, base policy at 10% success, 4 held-out demos per round,
30 paired evaluation episodes:

| Arm | Success by round | Δ | Update time |
|---|---|---|---|
| `responsive` | 0.10 → 0.33 → 0.50 → 0.53 → **0.80** | **+0.70** | 3.0s avg / 3.2s max |
| `noncontingent` | 0.13 → 0.10 → 0.10 → 0.10 → 0.10 | −0.03 | 2.9s avg / 3.1s max |

The first round alone moves success 0.10 → 0.33, so corrections register
immediately rather than several rounds later. The sham arm is pinned flat while
costing the same wall-clock. A useful tell in the logs: the sham arm's
`loss_start` resets to the same value every round (≈−13.1) because the policy is
restored each time, whereas the responsive arm's descends monotonically
(−13.4 → −16.7).

```bash
python scripts/validate_finetune.py \
    --checkpoint data/checkpoints/bc_rnn_Lift_study.pth \
    --dataset data/robomimic/lift/ph/low_dim_v141.hdf5 \
    --skip-demos 8 --rounds 4 --demos-per-round 4 --eval-episodes 30 --sham
```

### Success rate is not a good dependent variable

Worth knowing before designing around it. Measured on Lift with a marginal
policy, **success rate over 20 episodes ranged 0.15-0.70 with bitwise identical
weights.** Outcomes near the competence boundary flip on perturbations as small
as 1e-4, and both reference policies are stochastic at evaluation time
(robomimic's `low_noise_eval` still samples from the GMM).

`dagger_lh/evaluate.py` mitigates this for offline measurement with paired,
seeded evaluation — a fixed set of initial simulator states shared across
policies, plus a per-episode torch seed. That tightened the same measurement to
0.40-0.43.

For the study itself the implication is structural: with one rollout per round,
per-round success is far too noisy to serve as the primary DV. **The primary
measures are the behavioural effort measures** (intervention rate, engagement,
keypresses, passivity, latency, give-ups), which is why `helplessness_index` is
built from those and not from success. Report success as a manipulation check,
and raise `rollouts_per_round` if you need it to carry weight.

---

## The study design

Learned helplessness is induced by **non-contingency**: outcomes stop depending on
what the subject does. Here the subject is the teleoperator, and the manipulated
variable is whether their corrections actually change the policy.

| Condition | What happens to the operator's corrections |
|---|---|
| `responsive` | Real LoRA update. The policy improves. **Control arm.** |
| `noncontingent` | Update runs in full, then is **discarded**. Effort and outcome are independent. **Induction arm.** |
| `degraded` | As above, plus action noise growing each round, so the policy visibly worsens. |
| `yoked` | Policy follows a prerecorded schedule, independent of this operator. |

The sham update is what makes this a clean manipulation rather than a confound.
`PolicyTrainer.finetune(commit=False)` snapshots the policy and optimizer state,
runs the **same number of gradient steps on the same data**, then restores the
snapshot bit-for-bit. Identical compute, identical data, no learning. The smoke
test asserts both halves of this (`max|Δ| == 0` across the full state dict under
sham, `> 0` under responsive).

The snapshot deliberately covers the **entire state dict, not just the trainable
tensors.** Buffers are not trainable but still change the policy: an earlier
version snapshotted only LoRA parameters, and a refit of the observation
normaliser survived a sham update and improved success by +0.40 — silently
destroying the manipulation. Anything that alters the policy has to be inside the
snapshot. For the same reason, do not refit the normaliser inside the round loop;
fit it once from the seed data before the session starts.

Because a sham update can still finish marginally faster, `finetune_min_seconds`
pins the "updating policy" screen to a fixed duration. Without it the *timing*
would leak the condition. It is set to 12s in the study configs.

### The escape test — the design that actually demonstrates helplessness

`configs/study_escape.yaml` runs two phases: 6 rounds non-contingent, then 6
rounds with control genuinely restored.

This matters because effort declining under non-contingency is, on its own, just
**rational disengagement** from a task that does not respond. Learned helplessness
is the *failure to recover* once contingency returns. Run it against
`configs/study_responsive.yaml` as a between-subjects control:

```bash
python scripts/run_session.py --config configs/study_escape.yaml --participant p003 \
    --checkpoint data/checkpoints/bc_rnn_Lift_base.pth
```

### Measures

Per round, in `rounds.csv` (see `dagger_lh/study.py::RoundMetrics`):

| Measure | Helplessness prediction |
|---|---|
| `intervention_rate` | ↓ fraction of steps the human took over |
| `engagement_rate` | ↓ fraction of steps with actual key input |
| `keypresses_per_step` | ↓ raw motor effort |
| `passivity` | ↑ holding control while doing nothing |
| `time_to_first_intervention` | ↑ latency to step in |
| `giveup_events` | ↑ explicit `X` presses |
| `success_rate` | task outcome |
| `survey_*` | perceived control, effectiveness, expected success, frustration, persistence, effort (1–7) |

`helplessness_index` ∈ [−1, 1] aggregates the three effort measures as *relative*
declines (1 = effort ceased, 0 = steady, negative = worked harder). It normalises
deliberately: the raw deltas are in incommensurable units, so an unweighted mean
would let keypresses-per-step swamp two rates bounded in [0, 1]. Latency,
passivity and give-ups are reported alongside rather than folded in, because they
can dissociate — an operator may keep taking control while doing nothing once
there.

### Analysis

```bash
python scripts/analyze.py --sessions data/sessions --plot effort.png --json report.json
```

Reports per-round effort trajectories by condition, Welch t-tests with Cohen's *d*
for each arm against `responsive`, the escape-test recovery contrast, and survey
drift. Two-phase sessions are grouped separately from single-phase ones even when
phase 1 shares a condition, since they are different designs.

Before recruiting anyone, `scripts/simulate_study.py` generates synthetic sessions
with a **known** effect so you can check the analysis recovers it and size your
sample:

```bash
python scripts/simulate_study.py --out data/sim --n-per-condition 12 --two-phase
python scripts/analyze.py --sessions data/sim --plot sim.png
```

Its generative model is an explicit assumption about behaviour, not a prediction —
it exists so the analysis code can be tested against ground truth.

---

## DAgger variant

This is **HG-DAgger** (human-gated; Kelly et al. 2019): the policy acts, and only
states where the operator held control enter the training set. Classic DAgger
would require labelling every state the policy visits, which is not something a
person can do at 20 Hz through a keyboard.

Every timestep is still recorded with an `actor` flag, so the aggregated dataset
retains the policy-visited states too — `SequenceDataset(human_only=False)` trains
on all of it if you want to compare.

Fresh corrections are oversampled via `TrainConfig.new_data_prob` (default 0.5,
i.e. half the sampling mass on the newest round) so that a correction has a
visible effect within one round. This matters for the study: if improvement lagged
several rounds behind, the `responsive` arm would itself feel non-contingent.

## Policies

Both are the published implementations, wrapped only for the DAgger loop:

**BC-RNN** — `dagger_lh/policies/bc_rnn.py` builds robomimic's `BC_RNN_GMM`
through `robomimic.algo.algo_factory`, configured to the paper's low-dim
hyperparameters (`generate_paper_configs.py`): seq_length 10, LSTM hidden 400,
5-mode GMM head, **no MLP layers** between RNN and output, lr 1e-4. ~2.0M
parameters. Loss comes from robomimic's own `_compute_losses`.

**ACT** — `dagger_lh/policies/act.py` wraps LeRobot's `ACT` with `ACTConfig`
defaults (dim 512, 8 heads, FFN 3200, 4 encoder / 1 decoder layer, latent 32,
kl_weight 10, ImageNet-pretrained ResNet18) and its `ACTTemporalEnsembler`. ~40M
parameters. `chunk_size` is the one deliberate change — the reference 100 targets
50 Hz ALOHA, so 20 (≈1s at robosuite's 20 Hz) is used here.

Action chunking makes ACT visibly smoother than BC-RNN, which is not cosmetic in
this study: it changes how an operator judges whether the robot is improving.

### LoRA

`dagger_lh/lora.py` is hand-rolled rather than using `peft`, for two reasons the
reference architectures force:

- BC-RNN's recurrent core keeps its weights as raw `nn.Parameter`s on `nn.LSTM`,
  not `nn.Linear` submodules, so a Linear-only injector cannot adapt the part of a
  BC-RNN that matters most. `LoRALSTM` puts low-rank deltas on `weight_ih_l*` /
  `weight_hh_l*` and rebuilds `_flat_weights` each forward, keeping the fused
  kernel.
- LeRobot's ACT uses `nn.MultiheadAttention`, which packs q/k/v into one
  `in_proj_weight` and reads `out_proj.weight` directly — neither can carry an
  adapter. `patch_attention_for_lora` rewrites those blocks into explicit q/k/v/out
  `nn.Linear`s. This is verified numerically identical (≤4e-7, including under
  key-padding masks), so it does not change the reference model.

Adapters initialise to an exact no-op (B = 0), asserted in both policies. Layers
too small for a low-rank factorisation to save anything (e.g. the 400→5 GMM
mixture logits) are fully finetuned instead of skipped. Each round's delta is
checkpointed as `adapter_round_N.pth` — a few hundred KB, so the whole session's
trajectory is recoverable.

Measured on an M3 Max (MPS), low-dim Lift:

| Policy | Params | Trainable | Per step | 300 steps |
|---|---|---|---|---|
| BC-RNN | 2.0M | 3.4% | ~14 ms | ~4 s |
| ACT | 40.1M | 2.3% | ~45 ms | ~13 s |

---

## A version trap worth knowing about

**robosuite 1.5 silently changed observation conventions relative to 1.4.1, which
is what robomimic's published datasets were generated with.** Nothing raises; a
policy trains normally and then fails at rollout.

`ManipulationEnv._get_obj_eef_sensor` now returns `obj_pos - eef_pos` where 1.4.1
stored `eef_pos - obj_pos`. For Lift that negates 3 of the 10 `object` dimensions.
It cost this pipeline a policy that trained to convergence and scored **0%** until
it was found. `EnvConfig.legacy_object_state` (on by default) corrects it, and
logs which sensors it flipped.

A second, unrelated bug in the same area: robosuite caches observables, so reading
observations after writing simulator state directly requires
`_get_observations(force_update=True)`. Without it you get values from the previous
step.

**Always run the validator before pretraining on an external dataset:**

```bash
python scripts/validate_dataset.py --dataset <file.hdf5> --env Lift
```

It replays recorded simulator states and compares observations key by key, and
separately replays recorded actions open-loop to check the action convention
(control frame, scaling, gripper sign) independently.

| Dataset | Status |
|---|---|
| `lift/ph/low_dim` | **Validated** — observations match to 3e-8, actions replay 3/3 |
| `can/ph/low_dim` | **Not usable as-is** — actions replay fine, but robosuite 1.5 reports PickPlace object positions in world rather than bin-relative coordinates. Not auto-corrected. Collect demos with this env instead, or add the frame correction. |

Other pins that are load-bearing (see `scripts/setup_env.sh`): mujoco must be
3.3.x (3.12 makes robosuite assert on the Panda model), numpy must stay 1.x, and
robomimic/lerobot must be installed `--no-deps` or they downgrade torch and pull
numpy 2.

---

## Online deployment

The study can run in a browser with participants recruited from Prolific/MTurk.
The simulator cannot: robosuite is Python and MuJoCo needs a real GL context, so
there is no WASM path. The architecture is therefore a **thin client over
server-side simulation** — the browser sends key state and receives JPEG frames,
while robosuite, the policy and the LoRA finetuning stay on the server.

```bash
python scripts/run_server.py --config configs/study_online.yaml \
    --checkpoint data/checkpoints/bc_rnn_Lift_study.pth \
    --max-sessions 8 --admin-token "$DAGGER_ADMIN_TOKEN" \
    --arms responsive:responsive,noncontingent:noncontingent,escape:noncontingent:1:responsive

python scripts/test_web.py           # full flow over a real WebSocket, no browser
python scripts/load_test_web.py --n 8 --spawn   # concurrency + latency check
```

Participants get `/`; researchers get `/admin?token=…` (live sessions, assignment
balance, per-session event log, force-stop).

### The study logic is not duplicated

`dagger_lh/web/worker.py` provides `RemoteDisplay`, `RemoteTeleop` and
`RemoteSurvey`, which implement the same interfaces the local pygame front-end
does, and injects them into **the same `SessionRunner`** the desktop version
uses. Conditions, sham updates, metrics and logging run identical code online and
in-lab, so the two cannot drift apart — and `scripts/analyze.py` reads sessions
from either without changes (verified in `scripts/test_web.py`).

Control semantics are shared the same way: both front-ends go through
`BaseTeleop._state_from`, so smoothing, gripper latching and authority toggling
are identical whether keys arrive from pygame or a WebSocket.

### One process per participant

Threads do not work. MuJoCo's offscreen GL contexts are not thread-safe; a
threaded test of 4 concurrent envs had to be killed. Processes also isolate
robomimic's module-level observation-spec globals and each participant's torch
thread pool, and stop one crash taking the study down.

Measured on an M3 Max, 8 concurrent participants through the real server:

| Concurrent | Frame rate each | Control RTT (local) |
|---|---|---|
| 4 | 20.5 fps | 1 ms |
| 8 | 20.4–20.6 fps | 2–3 ms |

Sizing: ~1 core per 1–2 concurrent participants, ~400 MB RAM each, and ~1 Mbit/s
egress each at the default 384 px / 20 fps (`--frame-size`, `--jpeg-quality`
trade quality against bandwidth; 256 px is ~0.5 Mbit/s). `--max-sessions` is a
hard cap — past it `/api/enroll` returns 503 and the client asks the participant
to come back, which is better than queueing them into a degraded session.

### Things that only matter online

**Episode pacing.** The local front-end paces the loop with pygame's clock. The
first version of the web worker had no clock and ran at **63 fps against a 20 Hz
control rate** — a robot moving ~3× real time, which no participant could track
and which does not match the demonstrations the policy learned from.
`RemoteDisplay._pace` now holds the loop to `env.control_freq`, and
`load_test_web.py` fails if the rate drifts more than 25% either way, *including
too fast*.

**Input as state, not events.** The client reports the currently held key set
rather than a keystroke log, so a dropped packet costs a little control authority
instead of desynchronising the episode. Discrete presses (TAB, SPACE, ENTER, X)
can't be resent that way, so they travel as an edge-triggered list drained exactly
once. Held keys are cleared on blur — otherwise a tab switch leaves a direction
key stuck down forever.

**A practice qualification gate.** Online participants face latency an in-lab
participant does not, and the task is genuinely hard. Without a gate, someone who
*never could* do the task is indistinguishable from someone who **gave up because
they were made helpless** — which is the measure the whole study rests on.
`--practice-required` sets how many practice successes are needed; failures get a
polite screen-out and are still paid, and their assignment is returned to the pool
so the design stays balanced.

**Attention checks.** `--attention-check-every N` inserts a verifiable item
("select 2") into every Nth survey; `survey_attention_passed` lands in
`rounds.csv`. Tab-blur events are logged too, and a blurred window counts as *no
input* rather than idle-at-the-controls, so passivity is not inflated by
tab-switching.

**Condition never reaches the client.** `/api/enroll` returns only a session id,
the HUD carries no condition text (asserted in `test_web.py`), and arm assignment
is visible only through the token-gated admin API. A participant reading the
network tab learns nothing.

**Reconnection.** Sessions survive a dropped socket: the client retries with the
session id in `sessionStorage`, so an accidental refresh resumes rather than
restarting the study. Abandoned sessions are reaped after a grace period.

**Balanced assignment that survives restarts.** `Assigner` allocates to whichever
arm is furthest behind and persists counts atomically, so restarting the server
mid-study does not silently unbalance the design. Declines and screen-outs return
their assignment.

### Deploying

`deploy/` has a `Dockerfile` (EGL headless rendering, the same load-bearing pins
as `setup_env.sh`, plus build-time assertions that the stack is correct *and* that
offscreen rendering actually works in the container), a `docker-compose.yml` with
Caddy for automatic TLS, and an `nginx.conf` alternative.

```bash
export DAGGER_ADMIN_TOKEN=$(openssl rand -base64 24)
export STUDY_DOMAIN=study.example.edu
docker compose -f deploy/docker-compose.yml up -d --build
```

TLS is not optional: browsers only allow WebSockets from an `https` page over
`wss`. Both proxy configs set a 2-hour read timeout — a session holds one
connection for 20–30 minutes and nginx's 60s default would drop it mid-task.

### Questionnaires in Qualtrics

Supported, and documented step by step in **[docs/qualtrics.md](docs/qualtrics.md)**.
The recommended shape is Qualtrics → task → Qualtrics: Qualtrics owns consent,
demographics, the baseline and exit batteries and (optionally) the debrief, while
the task owns the robot and the per-round check-ins.

```bash
python scripts/run_server.py --config configs/study_online.yaml \
    --checkpoint data/checkpoints/bc_rnn_Lift_study.pth \
    --skip-consent \
    --qualtrics-exit-url "https://iu.qualtrics.com/jfe/form/SV_exitId"

python scripts/merge_qualtrics.py --sessions data/sessions \
    --qualtrics exports/intake.csv exports/exit.csv --out merged.csv
```

**Qualtrics cannot be embedded.** It serves `X-Frame-Options: SAMEORIGIN`, so the
survey cannot live in an iframe inside the task page — that is set on their
servers, with no workaround from ours. The integration is therefore a full-page
handoff, which also dodges the third-party-cookie breakage iframed Qualtrics
surveys hit in current browsers. Sessions survive it: the worker keeps running,
the session id sits in `sessionStorage`, and on return the worker replays the
screen it was parked on.

**Keep the per-round check-ins native** (`--survey-mode native`, the default).
Those six items are asked after every round, and eight full-page round trips add
real friction at exactly the moment the manipulation is meant to be felt.
`--survey-mode qualtrics` exists for protocols that require every item to be
Qualtrics-hosted.

`merge_qualtrics.py` joins the exports to `rounds.csv` on `external_id` (recorded
in `join_keys.json` in each session directory), namespaces columns by export
(`intake__age`, `exit__frustration_post`), skips Qualtrics's three header rows,
and reports unmatched sessions rather than dropping them silently.

One ethics detail worth catching early: if the debrief moves into the exit survey
(`--skip-debrief`), Qualtrics cannot branch on condition — the task deliberately
never sends it to the browser. Either debrief in the task, which already branches
correctly, or word one disclosure to cover both arms. Do not pass the arm through
the URL to make Qualtrics branch; the participant can read it.

### Recruitment plumbing

`?PROLIFIC_PID=…&STUDY_ID=…&SESSION_ID=…` is captured into the session log, and
`PROLIFIC_PID` becomes the participant id so records line up with the platform.
`--completion-code` and `--completion-url` drive the final screen's code and
return button. Researchers can force an arm for piloting with
`?arm=escape&token=…`.

### Caveat worth designing around

Network latency makes teleoperation harder and raises baseline frustration. It is
*constant across conditions*, so it does not confound the between-condition
contrast — but it does compress the range of the effort measures and will push
attrition up. Expect to need a larger sample than an in-lab version; use
`scripts/simulate_study.py` to size it. `configs/study_online.yaml` shortens
sessions (8 rounds, horizon 300) and raises input smoothing for this reason.

---

## Controlling the policy's success rate

`scripts/calibrate_ladder.py` builds a ladder of policies at target success rates
(0%, 10%, … 100%). Each rung is one base policy plus a **calibrated rollout
action-noise level**, found by bisection. Because every rung shares the same
weights, competence is the only thing that varies across the ladder.

```bash
python scripts/calibrate_ladder.py \
    --checkpoint data/checkpoints/bc_rnn_Lift_base.pth \
    --out-dir data/checkpoints/ladder \
    --targets 0 10 20 30 40 50 60 70 80 90 100 \
    --horizon 300 --select-episodes 60 --verify-episodes 100

python scripts/show_ladder.py --ladder data/checkpoints/ladder/ladder.json \
    --plot ladder.png --reverify 100
```

`--refine-from ladder.json` re-bisects only the bands that came out off-target,
inside the bracket implied by their neighbours, so a ladder can be tightened
without rebuilding it.

### Three methods that did not work

All were built and measured on Lift before settling on the fourth. They are kept
(`scripts/train_policy_ladder.py`, `scripts/build_ladder_noise.py`) because they
may work on a task with a less forgiving success basin, but none worked here.

**Earlier checkpoints of one training run.** Assumes success rate is a stable
function of training step. It is not: consecutive 200-step snapshots, scored on a
*fixed* evaluation set, went 0.90 → 0.37 → 0.57 → 0.47. Selecting the snapshot
that happens to measure 20% selects a fluke — targets of 10/20/30% all verified at
**0.34–0.38** on held-out states, and two were the same snapshot. Kept as evidence
in `data/checkpoints/ladder_snapshots_FAILED/`.

**Weight interpolation** between the random init and the trained policy. A step,
not a ramp — success stayed 0.00 from α 0.0 through 0.8, then 0.05 at 0.9 and 0.95
at 1.0. Nothing to bisect. (Linear mode connectivity holds *along* a training
path, not between an init and its endpoint.)

**Training on corrupted demonstrations** — the intuitive fix, and non-monotone.
Gaussian noise on demonstrated actions acts as *regularisation* at moderate levels
(the DART effect): σ 0.09 → 0.25 but σ 0.20 → 0.57. Replacing a fraction of demos
with heavily corrupted ones behaved no better (0.0 → 0.92, 0.25 → 0.80, 0.5 →
0.96), and corrupting **every** demonstration still left the policy at 0.40. Lift
simply tolerates sloppy demonstrations too well to serve as a competence knob.

### Why rollout noise works

It is monotone by construction over its useful range, needs no retraining, and
bisection converges reliably:

| noise | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|
| success | 1.00 | 0.93 | 0.70 | 0.40 | 0.17 | 0.10 | 0.00 |

It also composes with the rest of the pipeline for free: `action_noise_std` is
already applied per round in the episode loop (it is what the `degraded` condition
uses), so a rung's noise level flows straight through.

The behaviour it produces is an imprecise reach-and-grasp rather than a policy
doing something random, which is the right *kind* of failure for a study where
participants judge whether the robot is improving — a policy that fails randomly
reads as broken instead.

### How accurate the placement actually is

The delivered Lift ladder (`data/checkpoints/ladder/`), each rung a separate
self-contained checkpoint with its calibrated noise baked in as a buffer — loading
`bc_rnn_Lift_p030.pth` *is* a 30% policy, nothing external to remember:

| target | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noise | .450 | .216 | .178 | .148 | .111 | .104 | .089 | .077 | .056 | .042 | .000 |
| verified (100 states) | .02 | .17 | .23 | .25 | .43 | .45 | .57 | .71 | .80 | .93 | 1.00 |

Mean deviation from target 0.029, max 0.07, monotone, 11 distinct noise levels.

**Budget about ±0.05–0.09 of measurement noise on any single evaluation.** Three
rungs re-measured on four independent 60-state samples each:

| rung | label | s101 | s202 | s303 | s404 | pooled mean | SD |
|---|---|---|---|---|---|---|---|
| p020 | 20% | 0.28 | 0.23 | 0.17 | 0.28 | **0.24** | 0.055 |
| p050 | 50% | 0.60 | 0.50 | 0.43 | 0.62 | **0.54** | 0.086 |
| p080 | 80% | 0.85 | 0.80 | 0.82 | 0.92 | **0.85** | 0.052 |

Those SDs match the binomial standard error at n = 60 almost exactly, so this is
ordinary sampling noise rather than anything systematic — and the pooled means
(over 240 states) land within ~0.05 of the labels. The ladder is well calibrated.

One trap worth knowing: a single evaluation run **shares its state set across all
rungs**, so an unlucky draw moves every rung in the same direction at once. A
60-state re-verification of this ladder came out low on 10 of 11 rungs for exactly
that reason, which looks alarmingly like bias until you re-sample. Judge a ladder
from several state samples, or one large one:

```bash
python scripts/show_ladder.py --ladder data/checkpoints/ladder/ladder.json \
    --reverify 300 --seed <a seed the ladder was not built with>
```

At ~7–8 ms per simulator step, a 300-state pass costs roughly 12 minutes per rung
(~2 hours for eleven), so run it with nothing else competing for the GPU.

### Two measurement details that decide whether a ladder reproduces

**Paired, seeded evaluation.** Every candidate is scored from the same fixed set
of initial simulator states with a per-episode torch seed. Without this a policy
with *unchanged weights* scored anywhere from 0.15 to 0.70 on Lift (see "Success
rate is not a good dependent variable").

**Selection and verification use disjoint state sets.** Bisecting to the noise
level whose measured score hits a target is a selection on a noisy quantity, so
that score is optimistically biased. The manifest's `verified_success` comes from
a second set the selection never saw — exactly what exposed the snapshot method's
failure above. `--reverify` measures a third time from fresh states.

Both the builder and `show_ladder.py` also report how many **distinct** policy
settings a ladder contains: one claiming 11 levels but holding 8 would make any
manipulation built on the duplicates a silent no-op.

### What the ladder unlocks: a genuinely scripted condition

The `yoked` condition is supposed to be "the policy follows a prerecorded schedule
independent of this operator". Until the ladder existed it could not do that and
behaved **identically to `noncontingent`** — a mislabelled duplicate arm. Now a
success curve can be prescribed directly:

```bash
python scripts/run_session.py --mode dagger --condition yoked \
    --ladder data/checkpoints/ladder/ladder.json \
    --yoked-schedule 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4
```

Each round the runner swaps in the rung nearest that round's target, so the
operator watches a policy of known, predetermined competence. The trainable policy
is still updated and rolled back, so the between-round wait is unchanged, and the
human's corrections are still recorded — which keeps the debrief's promise that
their data was preserved. Every swap is logged as a `ladder_rung` event.

Both `run_session.py` and `run_server.py` refuse a `yoked` arm without `--ladder`
rather than silently degrading it back into a second non-contingent condition.

Other uses: choosing a starting competence with the right headroom (an expert base
policy makes even the contingent arm non-contingent), and holding competence fixed
across participants so that only contingency differs between conditions.

---

## Layout

```
dagger_lh/
  config.py       dataclass configs, YAML + dotted overrides
  envs.py         robosuite wrapper; observation conventions & compatibility
  teleop.py       pygame keyboard device + operator HUD
  lora.py         LoRA for nn.Linear and nn.LSTM; inject / mark / snapshot
  policies/
    base.py       policy interface, obs normalisation, device resolution
    bc_rnn.py     robomimic BC_RNN_GMM + paper config
    act.py        LeRobot reference ACT + LoRA-compatible attention
  data.py         robomimic-format HDF5 writer/loader, sequence sampling
  trainer.py      pretraining and time-budgeted LoRA finetuning (incl. sham)
  study.py        conditions, measures, event logging, surveys
  runner.py       the demo and DAgger session loops
  evaluate.py     headless rollouts / success rate
  ladder.py       checkpoints of known competence; scripted success curves
  web/
    protocol.py   client<->worker wire format and key-name validation
    worker.py     per-participant process; remote display/teleop/survey
    manager.py    process lifecycle, balanced condition assignment
    server.py     FastAPI: enrolment, WebSocket relay, admin API
    static/       participant client and admin page (vanilla JS, no build)

deploy/
  Dockerfile            EGL headless image with build-time stack assertions
  docker-compose.yml    server + Caddy (automatic TLS)
  Caddyfile, nginx.conf reverse-proxy configs (WebSocket + long timeouts)

scripts/
  setup_env.sh          environment creation with the load-bearing pins
  download_data.sh      official robomimic datasets + validation
  validate_dataset.py   observation & action convention checker
  pretrain.py           base policy training
  run_session.py        main entry point (demo | dagger)
  smoke_test.py         headless end-to-end test with a scripted operator
  test_display.py       real pygame display + key bindings (offscreen or --window)
  validate_finetune.py  does contingent finetuning actually improve the policy?
  run_server.py         launch the online study server
  test_web.py           full online flow over a real WebSocket, no browser
  load_test_web.py      concurrency + latency check; sizes --max-sessions
  merge_qualtrics.py    join Qualtrics exports to the behavioural sessions
  calibrate_ladder.py     build a ladder at controlled success rates (the method
                          that worked; calibrated rollout noise)
  build_ladder_noise.py   ladder by corrupting the training data (non-monotone
                          on Lift; kept for harder tasks)
  train_policy_ladder.py  ladder by snapshot selection / weight interpolation
                          (both measured and rejected on Lift; see README)
  show_ladder.py        inspect / re-verify / plot a ladder
  show_ladder.py        inspect / re-verify / plot a ladder

docs/
  qualtrics.md          step-by-step Qualtrics survey-flow setup
  simulate_study.py     synthetic sessions for power analysis
  analyze.py            aggregate sessions, contrasts, plots
```

### Output per session

`data/sessions/<participant>_<mode>_<timestamp>/`

```
config.yaml            exact config used
demos.hdf5             robomimic-format demos + actor/intervention arrays
events.jsonl           timestamped events (interventions, give-ups, updates, surveys)
rounds.csv             analysis-ready per-round measures
summary.json           helplessness index and escape-test contrasts
adapter_round_N.pth    LoRA delta after each round
policy_final.pth       final policy
```

`demos.hdf5` is valid robomimic, with per-round `mask/round_N` filter keys, so it
can be fed straight to `robomimic/scripts/train.py` or its dataset tooling. The
extra `actor` / `intervention` arrays are ignored by robomimic.

## Ethics note

The non-contingent arms work by deceiving participants about whether their effort
matters, and are designed to induce frustration and disengagement. That needs IRB
approval before you recruit anyone.

The pipeline supports the obligations rather than leaving them to you:

- **Consent** is a gated screen — two explicit checkboxes, and a decline path that
  returns the participant's condition assignment to the pool. The text flags that
  something about the robot's learning is withheld until debriefing, which is the
  standard way to consent to an incomplete-disclosure design.
- **Debriefing is automatic and condition-aware.** Deceived arms get an explicit
  disclosure that the update was computed from their corrections and then
  discarded, and that the robot's failure was fixed in advance by random
  assignment rather than reflecting their teaching. `test_web.py` asserts this
  disclosure actually reaches the deceived arms.
- **Withdrawal** is a checkbox on the debrief that records a deletion request
  against the participant code, plus `ESC` / closing the tab at any point.
- **The condition is never shown** to the participant, in the HUD or the API.
- **Screened-out participants are still paid** (`--screenout-code`).
- Every episode is saved regardless of condition, so the debrief's claim that
  their corrections were preserved and are scientifically useful is true.

Edit the consent and debrief text in `dagger_lh/web/worker.py`
(`CONSENT_TEXT`, `DEBRIEF_DECEPTION`, `DEBRIEF_PLAIN`) to match what your IRB
approves — the defaults are a starting point, not approved language, and they
carry no contact details or protocol number.

One online-specific point: because debriefing happens in-browser, a participant
who abandons the tab mid-session never sees it. If your IRB requires debriefing
for all enrolled participants, plan to send it through the recruitment platform
using the `PROLIFIC_PID` recorded in the session log.
