# CI gate for robot policies

`breakpoint gate` is pytest for robot policies. On every new checkpoint, your own CI
runner evaluates the candidate and the baseline checkpoint on the same fixed conditions
and the same seeds, then asks one question: is there statistical evidence that the
candidate falls more than the baseline? It fails the build only when there is, and it
keeps the probability of a false alarm at or below `--alpha` (default 5%), however many
conditions the battery has.

Weights never leave the runner. The report carries per-condition fall counts and the
sha256 of each checkpoint file, nothing else.

## Command

```
breakpoint gate --robot go1 \
    --baseline checkpoints/prod.pkl --candidate checkpoints/candidate.pkl \
    [--battery legged-quadruped-v1] [--budget smoke|standard] \
    [--decision stat|rule] [--alpha 0.05] [--escalate-to N] [--tolerance 0.0] \
    [--out gate.json] [--summary-out summary.md] [--no-upload] \
    [--obs-spec contract.yaml ...the other policy-adapter flags `rate` takes]
```

| exit | meaning |
|---|---|
| 0 | pass: no evidence the candidate falls more than the baseline |
| 1 | regression: prints `CI GATE FAILED (regression detected)` and the cells |
| 2 | anything else: usage error, a policy that will not load, a battery for another robot class, an internal error |

A crash is never reported as a regression. Checkpoints load exactly as `breakpoint
check-policy` and `breakpoint rate` load them: a brax params pickle, an adapter spec
(`onnx:policy.onnx`, `params:weights.npz`), or anything with a `--obs-spec` contract.

## How the verdict works

**Same worlds, same keys.** Seed `s` runs every cell in one batch under `PRNGKey(s)`,
row `i` keyed `fold_in(PRNGKey(s), i)`, so baseline and candidate see identical
conditions and identical rollout keys. Each (cell, seed) is therefore a *pair* of
outcomes. The battery, fall rule, theta validation and simulator-artifact handling are
shared with `breakpoint rate` (a rollout flagged `sim_invalid` counts as a fall), so a
gate and a rating of the same checkpoint agree cell for cell.

**Exact paired tests (`--decision stat`, the default).** A pair where exactly one
checkpoint falls is discordant. If the candidate were no worse than the baseline, each
discordant pair would be at most a fair coin flip, which gives an exact one-sided sign
test (McNemar's test in exact form; no normal approximation, valid at two seeds). The
gate runs two families of tests, each at `alpha / 2`:

- a **pooled** sign test over every discordant pair in the battery, which has the most
  power when a regression is spread thin or sits in one cell while the rest are quiet;
- a **per-cell** sign test, Holm-adjusted across cells, which catches one badly
  regressed cell even when improvements elsewhere cancel it out of the pooled count.

By the union bound the build's false-alarm probability is at most `alpha`. A cell the
baseline already fails is tested too: 2/8 falls becoming 8/8 is a regression.
`--tolerance` is a practical-significance floor: a fall-rate increase at or below it
never fails the build, however significant.

**Escalation (`--escalate-to`).** Eight seeds cannot tell a real one-cell regression from
noise: even a clean sweep (0/8 to 8/8) has p = 0.0039, which no per-cell test across 15
cells can call significant. So cells where the candidate fell more, but not
significantly, are re-run alone on fresh seeds (`10000, 10001, ...`, never a battery
seed) and decided again. Stage 1 (battery seeds) and stage 2 (fresh seeds, suspect cells
only) each get `alpha / 2`; stage 2 sees only new, independent seeds, so choosing cells
from stage-1 data does not bias it and the overall false-alarm bound still holds.
Escalation costs only the suspect cells. It defaults to 32 seeds for `standard` and 16 for
`smoke`. Smoke used to default to off, but with 4 seeds a single cell can never reach
significance on stage 1 (best one-sided p = 1/16 before Holm), so a candidate that fell on
every seed of one condition passed. `--escalate-to 0` turns it off.

A cell that was escalated is labelled on its fresh-seed evidence: `REGRESSED
(escalated)`, `cleared on more seeds`, or `worse, not significant` (a warning, exit 0).

**Tracking degradation.** Falls are binary, and most seeds of a healthy policy never fall,
so the fall tests ignore most of the evidence a run produces. On every (cell, seed) where
*neither* checkpoint fell, the gate also compares the mean velocity-tracking error: a
pair counts as worse when the candidate's error exceeds the baseline's by more than
`--degradation-margin` (default 10%), better when the reverse holds, and an exact
one-sided sign test runs over the counted pairs. The deadband is symmetric in the error
ratio, so under "no change" a counted pair is a fair coin and the test is exact. This
catches a policy that is getting worse at walking before it starts falling. It is
advisory by default (reported, exit 0); `--fail-on-degradation` lets it fail the build,
and then alpha is split in half between it and the fall tests so the overall false-alarm
bound stays at `--alpha`. Seeds that fell are left out: after a fall the error measures
the fall, not the gait.

**The original rule (`--decision rule`).** A cell regresses when the baseline passed it
(fall rate at most the battery's pass rule, 1/8), the candidate failed it, and the fall
rate rose by more than `--tolerance`. It is easy to explain, but with 15 cells it raises
false alarms on most builds of a policy with any noisy cell (measured below). It is kept
for teams that want it; the report says which decision ran.

## What the statistics buy (measured)

`scripts/gate_power.py` simulates two checkpoints on the same seeds, with a per-cell
baseline fall-rate profile shaped like the shipped battery (6 cells at 0%, 5 at 2%, cells
at 10%, 15%, 20% and 40%) and a shared per-seed difficulty (`rho = 0.5`: a hard world
tends to topple both checkpoints). 4000 simulated builds per row, alpha 0.05, escalation
to 32 fresh seeds. Raw numbers, including `rho = 0`, are in `docs/gate_power.json`.

| scenario (8 battery seeds) | `rule` | `stat` | `stat` + escalation |
|---|---|---|---|
| **no change** (any alarm is false) | **50.2%** | 0.8% | **1.1%** |
| one cell 2% to 50% | 97.9% | 12.0% | **99.1%** |
| one cell 2% to 25% | 79.8% | 3.8% | 44.0% |
| every cell +10 points | 98.4% | 79.8% | 98.1% |
| every cell +5 points | 84.9% | 29.3% | 45.3% |

| scenario (4 seeds, smoke) | `rule` | `stat` | `stat` + escalation |
|---|---|---|---|
| **no change** | **64.6%** | 0.4% | 0.6% |
| one cell 2% to 50% | 94.3% | 3.0% | 92.3% |

The old rule's high "detection" rates are not worth much when it also fails half of the
builds where nothing changed: a gate that cries wolf on every other build gets disabled.
The statistical gate with escalation matches the old rule on large regressions at one
fiftieth of its false-alarm rate. Small regressions (one cell 2% to 25%) need more seeds
than any default here; the summary says what the run could detect, so a pass is read for
what it is.

## Measured on the simulator (CPU, smoke)

The bundled Go1 demo policy as the baseline; as candidates, copies with multiplicative
Gaussian noise on every policy weight (`w * (1 + sigma * N(0, 1))`). Smoke budget (4
seeds, 150 steps), escalation to 16 or 32 fresh seeds.

| candidate | `rule` | `stat` + escalation |
|---|---|---|
| identical file | pass | pass (0 discordant pairs) |
| sigma 0.1 | **FAIL**: `obs-dropout` 0/4 to 1/4 | pass: `obs-dropout` re-run on 16 fresh seeds, 0 vs 0 falls, **cleared** |
| sigma 0.3 | **FAIL**: `obs-dropout` 0/4 to 1/4 | pass with a warning: 3 vs 0 on 16 fresh seeds (p = 0.125) |
| sigma 0.5 | FAIL (9 cells) | **FAIL**: pooled 16 vs 1 (p = 1.4e-4); fresh seeds attribute it, e.g. `latency-20ms` 13 vs 0, `nominal` 0 vs 0 (cleared) |
| sigma 0.7 | FAIL (12+ cells) | **FAIL**: pooled 49 vs 0; `latency-40ms`, which the baseline already failed, went 8 vs 2 on fresh seeds |

At sigma 0.1 the old rule fails the build on one fall in four seeds, a blip that 16 fresh
seeds show was noise. At sigma 0.5 and 0.7 both decisions fail the build; the statistical
one also says which cells carry the regression and which only looked worse by chance.

## Battery

`--battery` takes a shipped id (`breakpoint_mjx/batteries/<id>.yaml`) or a path to a
battery YAML. Without it the gate uses the robot class's battery, exactly as `breakpoint
rate` does (`legged-quadruped-v1` for go1 and barkour, `legged-humanoid-v1` for g1), and
refuses a battery written for another robot class. The report's `battery.sha256` is the
hash of that file (line endings normalized), so a silent edit is visible.

### Boundary battery (more sensitive)

Most cells of the shipped battery are conditions a good policy survives, so a small
regression shows up there as a few new falls the gate cannot tell from noise. A
**boundary battery** holds conditions where *your baseline* falls on some seeds but not
all of them. The policy is at its edge there, so a slightly worse candidate falls
measurably more often.

```
breakpoint gate-boundary --robot go1 --baseline checkpoints/prod.pkl --out boundary.yaml
breakpoint gate --robot go1 --baseline checkpoints/prod.pkl \
    --candidate checkpoints/candidate.pkl --battery boundary.yaml
```

`gate-boundary` samples `--conditions` (default 128) conditions uniformly in the plausible
band. `--regimes near_nominal conservative` also samples the near-nominal box (each band
shrunk toward nominal to 25% of its width), split evenly. On the go1 demo policy that box
produced no boundary condition, so it is off by default. It runs the **baseline only** on 20 selection seeds and keeps the
conditions with a fall rate in [`--p-min`, `--p-max`] (default [0.10, 0.60]), up to
`--max-cells` (default 24). The written battery runs `--seeds` (default 8) fresh seeds
per cell. These seeds are disjoint from the selection seeds and the escalation seeds, so
the false-alarm bound above is unchanged. The cost is one baseline run of
conditions × selection seeds (2,560 rollouts by default), paid once per baseline.

The battery is tied to that baseline. Its `built_from.baseline_sha256` is checked, and
the gate's summary carries a note when you gate against a different baseline. Rebuild it
when you promote a new baseline. Run it at `--budget standard`, because the smoke
budget's horizon cap changes what these conditions mean. Exit codes: 0 written, 1 no
condition fell in the range (sample more or widen the range), 2 usage error.

Measured in the benchmark (`benchmark/reports/TECHNICAL_REPORT.md` §9.3): go1 demo
policy, 24 boundary cells × 8 paired seeds, synthetic weight-noise candidates:

| candidate | shipped battery (gate) | boundary battery |
|---|---|---|
| identical | pass | pass, 80 → 80 falls |
| σ 0.1 | pass, 15 → 17 | **regression, 80 → 97 (p = 0.006)** |
| σ 0.3 | regression, 15 → 42 | regression, 80 → 159 |
| σ 0.5 | regression, 15 → 103 | regression, 80 → 174 |

These candidates are synthetic, and the identical pair is a single false-alarm check. The
benchmark chose its cells from search results, while `gate-boundary` samples uniformly;
the selection rule is the same.

Running both batteries in one CI job spends `--alpha` on each, so the build's false-alarm
bound is the sum of the two (pass `--alpha 0.025` to each to keep 5% overall).

### Budgets

| budget | seeds | horizon | escalation default | where it fits |
|---|---|---|---|---|
| `smoke` | first 4 of the battery's seeds | min(battery, 150) | off | GitHub-hosted (CPU-only) runners |
| `standard` | all of the battery's seeds | as declared | 32 fresh seeds | a self-hosted GPU runner |

A smoke gate screens for gross regressions. With 4 seeds no single cell can fail the
build on stage-1 evidence alone; the summary's `sensitivity:` line says so. Real gates
need `standard` on a GPU.

## Report

`gate.json` is a `kind: "gate"` report per the shared contract
(`schema: breakpoint.report/v1`), validated by the same checker as ratings before it is
written. `cells` and `summary` describe the candidate; `run` carries `budget` and
`horizon`, so a smoke gate cannot pass for a standard one. The baseline and the verdict
enter through `gate`:

```json
"gate": {"baseline_sha256": "...", "candidate_sha256": "...",
         "new_failing_cells": ["push-firm"], "regressed": true,
         "tolerance": "fall_rate_delta<=0",
         "decision": "stat", "alpha": 0.05,
         "method": "paired exact sign test (pooled + Holm per cell)",
         "pooled": {"up": 9, "down": 0, "p": 0.00195, "delta": 0.075, "significant": true},
         "baseline_falls": {"nominal": 0, "push-firm": 0, "...": 0},
         "cell_tests": [{"id": "push-firm", "up": 8, "down": 0, "p": 0.0039,
                         "p_holm": 0.0586, "status": "REGRESSED (escalated)"}],
         "warnings": [],
         "min_detectable": {"per_cell": null, "pooled": 6},
         "escalation": {"seeds": [10000, "..."], "cells": ["push-firm"],
                        "pooled": {"...": "..."}, "cell_tests": ["..."]}}
```

`run.runner` is `self` and `attestation` is `null` (signing is a separate step).
`policy.name` is the checkpoint's file name; no paths, weights or trajectories are
written. Override the name with `--candidate-name`.

## Optional upload

When `BREAKPOINT_API_KEY` and `BREAKPOINT_PROJECT` are set, the gate POSTs the report
(contract fields only, canonical JSON) to
`<BREAKPOINT_API_URL>/api/projects/<project>/gate` with `Authorization: Bearer <key>`.
`BREAKPOINT_API_URL` defaults to `https://breakpoint-inoki.vercel.app`. Without the two
variables the upload is skipped silently. Any upload error prints a warning and never
changes the exit code. `--no-upload` disables it.

## GitHub Action

`action.yml` at the repo root is a composite action. Inputs: `robot`, `baseline`,
`candidate`, `battery` (empty = the robot's battery), `budget` (default `smoke`),
`decision` (default `stat`), `alpha` (default `0.05`), `escalate-to` (empty = the budget's
default), `tolerance`, `python-version`, plus `engine-ref`, `repo-token`, `api-key`,
`project`, `out`. It installs the engine, runs the gate, appends the markdown summary to
`$GITHUB_STEP_SUMMARY`, uploads `gate.json` as the `breakpoint-gate-report` artifact (even
on failure), then fails the job on a regression and errors (not "regressed") on exit 2.
Output `regressed` is `true` or `false`.

Installing from a private fork or mirror needs a token with read access; set
`repo-token` and the action installs with it (the token is only in that step's
environment). A public engine repo needs no token:

```
pip install "breakpoint-mjx[sim] @ git+https://github.com/InokiLabs/breakpoint@<ref>"
```

### Runners

- GitHub-hosted runners (`runs-on: ubuntu-latest`) are CPU-only. They fit
  `budget: smoke` (about 40 s for the two checkpoints on a laptop CPU, plus the compile).
- Real gates need a self-hosted GPU runner, e.g. `runs-on: [self-hosted, linux, x64, gpu]`,
  with CUDA jax installed (`pip install "jax[cuda12]"`, or baked into the runner image).
  Without it the gate runs on CPU; `run.backend` in the report says which backend ran.

See `examples/github-workflow.yml` for a pull-request workflow with both jobs.

## Sample (real CPU smoke run)

The bundled demo policy against itself, `--budget smoke`, CPU (41.6 s wall, exit 0):

```
breakpoint gate --robot go1 \
    --baseline breakpoint_mjx/assets/demo/go1_policy.pkl \
    --candidate breakpoint_mjx/assets/demo/go1_policy.pkl \
    --budget smoke --out gate.json
```

```markdown
## BreakPoint gate: PASSED

No condition shows the candidate falling more than the baseline.

- robot: `go1`; battery: `legged-quadruped-v1` (sha256 `0857eb402abf`)
- baseline: `go1_policy.pkl` (sha256 `b442aaa99d03`)
- candidate: `go1_policy.pkl` (sha256 `b442aaa99d03`)
- budget: `smoke`; seeds [0, 1, 2, 3]; horizon 150 steps; backend `mjx-cpu`; tolerance `fall_rate_delta<=0`
- decision: `stat` (paired exact sign test (pooled + Holm per cell), alpha 0.05)
- pooled: candidate fell alone in 0 seed-pairs, the baseline alone in 0 (p = 1.000)
- sensitivity: no single cell can fail the build on 4 seeds alone; the battery does at 6 net seed-pairs
- candidate: SMOKE PREVIEW, not a rating: 13/15 cells passed (4 seeds, 150 steps)

| cell | baseline falls | candidate falls | candidate 95% CI | p (Holm) | status |
|---|---|---|---|---|---|
| `nominal` | 0/4 | 0/4 | [0.00, 0.60] | 1.000 | pass |
| `latency-40ms` | 3/4 | 3/4 | [0.19, 0.99] | 1.000 | fails (baseline too) |
| `lag-dropout` | 4/4 | 4/4 | [0.40, 1.00] | 1.000 | fails (baseline too) |
| ... 12 more cells, all 0/4 vs 0/4 ... | | | | | |

> note: budget smoke: 4 of 8 battery seeds, horizon 150 (battery declares 500); a smoke gate screens for gross regressions only
> note: baseline and candidate are the same file (identical sha256)
```

The demo policy fails the two 40 ms latency cells (`latency-40ms`, `lag-dropout`) for both checkpoints; they are reported and
do not fail the build, since the candidate is no worse there.
