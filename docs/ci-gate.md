# CI gate for robot policies

`breakpoint gate` is pytest for robot policies. On every new checkpoint, your own CI
runner evaluates the candidate and the baseline checkpoint on the same fixed conditions
and the same seeds, compares them condition by condition, and fails the build when the
candidate falls in a condition the baseline survived.

Weights never leave the runner. The report carries per-condition fall counts and the
sha256 of each checkpoint file, nothing else.

## Command

```
breakpoint gate --robot go1 \
    --baseline checkpoints/prod.pkl --candidate checkpoints/candidate.pkl \
    [--battery legged-quadruped-v1] [--budget smoke|standard] [--tolerance 0.0] \
    [--out gate.json] [--summary-out summary.md] [--no-upload]
```

| exit | meaning |
|---|---|
| 0 | pass: the candidate survives every condition the baseline survived |
| 1 | regression: prints `CI GATE FAILED (regression detected)` and the cells |
| 2 | usage error (missing checkpoint, unknown theta dim, negative tolerance) |

Checkpoints are loaded exactly as `breakpoint check-policy` loads them: a brax params
pickle by default, or an adapter spec (`onnx:policy.onnx`, `params:weights.npz`).

## How the comparison works

- **Cells.** A battery is a list of named, fixed conditions (theta overrides on the
  robot's theta space; every other dim stays at nominal). Every cell runs every seed.
- **Same worlds, same keys.** Seed `s` runs all cells in one batch under `PRNGKey(s)`;
  row `i` uses `fold_in(PRNGKey(s), i)`. Baseline and candidate therefore see identical
  conditions and identical rollout keys.
- **Fall.** The engine's fall rule (trunk height above the floor < 0.18 m, or the trunk
  tips past the up-projection threshold), the same one every BreakPoint report uses.
- **Pass.** A cell passes when its fall rate is at most the battery's `pass_rule`
  (1/8 by default). Each cell also gets an exact (Clopper-Pearson) 95% interval.
- **Regression.** A cell regresses when the baseline passed it, the candidate failed it,
  and the candidate's fall rate exceeds the baseline's by more than `--tolerance`
  (a fall-rate margin; 0 means any newly failing cell regresses). A cell the baseline
  also failed is reported but never fails the build: the gate only protects conditions
  the baseline actually survived.
- If the baseline fails the nominal cell, the summary says so: the gate cannot protect
  conditions the baseline never survived, and the policy probably has an observation or
  action mismatch (`breakpoint check-policy` diagnoses that).

### Battery

When `breakpoint_mjx/batteries/<battery>.yaml` exists it is used as declared (format in
the shared report contract), and the report's `battery.sha256` is the hash of that file.
On a branch without `batteries/`, the gate falls back to a small built-in condition set
(`builtin-quadruped-gate-v0`: nominal, low friction, payload, lag plus dropout, sensor
noise, push, soft gains, slope; 8 seeds; horizon 500) and the report names it as such,
never as the full battery. A parallel branch adds the full battery; switching to a
shared loader is the single assignment `BATTERY_LOADER = ...` in `gate.py`.

### Budgets

| budget | seeds | horizon | where it fits |
|---|---|---|---|
| `smoke` | first 2 of the battery's seeds | min(battery, 150) | GitHub-hosted (CPU-only) runners |
| `standard` | all of the battery's seeds | as declared | a self-hosted GPU runner |

A smoke gate is a screen for gross regressions. With 2 seeds per cell the pass rule
means "no falls", and the intervals are wide; it can miss a regression that only shows
up over more seeds or a longer horizon. Real gates need `standard` on a GPU.

## Report

`gate.json` is a `kind: "gate"` report per the shared contract
(`schema: breakpoint.report/v1`). `cells` and `summary` describe the candidate; the
baseline enters through `gate`:

```json
"gate": {"baseline_sha256": "...", "candidate_sha256": "...",
         "new_failing_cells": ["push"], "regressed": true,
         "tolerance": "fall_rate_delta<=0"}
```

`run.runner` is `self` and `attestation` is `null` (signing is added by a separate
step). `policy.name` is the checkpoint's file name; no paths, weights or trajectories
are written. Override the name with `--candidate-name`.

## Optional upload

When `BREAKPOINT_API_KEY` and `BREAKPOINT_PROJECT` are set, the gate POSTs the report
(contract fields only, canonical JSON) to
`<BREAKPOINT_API_URL>/api/projects/<project>/gate` with `Authorization: Bearer <key>`.
`BREAKPOINT_API_URL` defaults to `https://breakpoint-inoki.vercel.app`. Without the two
variables the upload is skipped silently. Any upload error prints a warning and never
changes the exit code. `--no-upload` disables it.

Status: the dashboard's gate endpoint may not accept `POST` yet; if the upload gets an
HTTP error, the gate warns and carries on.

## GitHub Action

`action.yml` at the repo root is a composite action. Inputs: `robot`, `baseline`,
`candidate`, `battery`, `budget` (default `smoke`), `tolerance`, `python-version`, plus
`engine-ref`, `repo-token`, `api-key`, `project`, `out`. It installs the engine, runs the
gate, appends the markdown summary to `$GITHUB_STEP_SUMMARY`, uploads `gate.json` as the
`breakpoint-gate-report` artifact (even on failure), then fails the job on a regression.
Output `regressed` is `true` or `false`.

Installing from a private fork or mirror needs a token with read access; set
`repo-token` and the action installs with it (the token is only in that step's
environment). A public engine repo needs no token:

```
pip install "breakpoint-mjx[sim] @ git+https://github.com/InokiLabs/breakpoint@<ref>"
```

### Runners

- GitHub-hosted runners (`runs-on: ubuntu-latest`) are CPU-only. They fit
  `budget: smoke`.
- Real gates need a self-hosted GPU runner, e.g. `runs-on: [self-hosted, linux, x64, gpu]`,
  with CUDA jax installed (`pip install "jax[cuda12]"`, or baked into the runner image).
  Without it the gate runs on CPU; `run.backend` in the report says which backend ran.

See `examples/github-workflow.yml` for a pull-request workflow with both jobs.

## Sample (real CPU smoke run)

The bundled demo policy against itself, `--budget smoke`, CPU. Identical checkpoints on
identical seeds must pass:

```
breakpoint gate --robot go1 \
    --baseline breakpoint_mjx/assets/demo/go1_policy.pkl \
    --candidate breakpoint_mjx/assets/demo/go1_policy.pkl \
    --budget smoke --out gate.json
```

Exit 0. The summary it wrote (the note's path is shown in the relative form the gate
now prints):

```markdown
## BreakPoint gate: PASSED

The candidate survives every condition the baseline survived.

- robot: `go1`; battery: `builtin-quadruped-gate-v0` (sha256 `ed0831d9d951`)
- baseline: `go1_policy.pkl` (sha256 `b442aaa99d03`)
- candidate: `go1_policy.pkl` (sha256 `b442aaa99d03`)
- budget: `smoke`; seeds [0, 1]; horizon 150 steps; backend `mjx-cpu`; tolerance `fall_rate_delta<=0`
- candidate: 7/8 cells passed

| cell | baseline falls | candidate falls | candidate 95% CI | status |
|---|---|---|---|---|
| `nominal` | 0/2 | 0/2 | [0.00, 0.84] | pass |
| `low-friction` | 0/2 | 0/2 | [0.00, 0.84] | pass |
| `payload` | 0/2 | 0/2 | [0.00, 0.84] | pass |
| `lag-dropout` | 1/2 | 1/2 | [0.01, 0.99] | fails (baseline too) |
| `sensor-noise` | 0/2 | 0/2 | [0.00, 0.84] | pass |
| `push` | 0/2 | 0/2 | [0.00, 0.84] | pass |
| `soft-gains` | 0/2 | 0/2 | [0.00, 0.84] | pass |
| `slope` | 0/2 | 0/2 | [0.00, 0.84] | pass |

> note: battery 'legged-quadruped-v1' not found under breakpoint_mjx/batteries/; using the built-in builtin-quadruped-gate-v0 (8 cells). The full battery lands with breakpoint_mjx/batteries/.
> note: budget smoke: 2 of 8 battery seeds, horizon 150 (battery declares 500); a smoke gate is a screen, not a rating

Simulation only. A cell passes when its fall rate is at most the battery's pass rule; weights never left this runner.
```

The demo policy falls in 1 of 2 `lag-dropout` rollouts for both checkpoints, so that cell
fails for both, is reported, and does not fail the build.
