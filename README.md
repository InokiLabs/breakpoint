# BreakPoint

Crash-testing for robot control policies, in simulation.

BreakPoint inverts domain randomization. Instead of sampling environment parameters
(friction, mass, latency, pushes, sensor noise, slope) at random to make a policy robust,
it treats that parameter space as a search space and hunts for the settings where a
fixed, already-trained policy breaks. MuJoCo MJX simulates thousands of environments at
once, so one generation of search is one batched rollout. Failures are ranked by a
transparent plausibility heuristic, grouped into named failure families, and the worst
member of each family is replayed bit-exactly into a self-contained HTML report.

On top of the search engine sit three evidence layers:

- **Ratings** (`breakpoint rate`): a fixed, versioned battery of named conditions and
  seeds. Every cell reports falls, a fall rate and a 95% interval, so two policies rated
  on the same battery are directly comparable.
- **CI gate** (`breakpoint gate`): pytest for robot policies. A candidate checkpoint is
  run on the same conditions and seeds as the baseline, and the build fails only when
  exact paired tests show it falls more, with false alarms held at or below 5%. Suspect
  conditions are re-run on fresh seeds before the verdict. Ships as a GitHub Action
  (`action.yml`).
- **Signed reports** (`breakpoint keys` / `sign` / `verify`): ed25519 attestations over
  a canonical JSON report, so anyone holding the public key can check a rating was not
  edited after it was produced.

Licensed under the Apache License, Version 2.0 (see `LICENSE` and `NOTICE`).

## Open source vs commercial edition

This repository is the open-source engine. It contains everything that **finds and
tests** failures: search, evaluation, clustering, evidence (cross-config and
cross-engine checks, minimal causes, coverage and risk certificates), reports, ratings,
the CI gate and self-signed attestations. It is free for any use, commercial included,
with no caps.

The **commercial edition** (Inoki Labs) adds what **fixes and certifies** policies:
hardening and robust retraining against the failures BreakPoint finds, Inoki-signed
certification of ratings, a hosted CI dashboard and enterprise support. That code is not
in this repository. `breakpoint harden` and `breakpoint license` say so when you run
them; nothing in the open-source engine is time-limited or crippled.

## Install

Python 3.11 or 3.12.

```bash
git clone https://github.com/InokiLabs/breakpoint.git && cd breakpoint
python -m pip install -U pip wheel
python -m pip install -e ".[sim]"        # jax + mujoco + brax + MuJoCo Playground
python -m pip install -r requirements.txt  # the full research stack (search extras, rendering)
```

Optional extras: `.[attest]` (ed25519 signing via `cryptography`), `.[onnx]` (ONNX policy
import), `.[llm]` (LLM-seeded search and plain-language explanations; off unless you set
`BP_LLM=1` and an API key), `.[dev]` (pytest).

A real search wants an NVIDIA GPU with CUDA 12 on Linux
(`pip install -U "jax[cuda12]<0.9"`). JAX GPU is Linux-only; on Windows use WSL2 or the
Colab notebook in `colab/`. CPU works for `check-policy`, `rate --budget smoke`, the
`smoke` search budget and the test suite.

## Quickstart

```bash
breakpoint doctor                        # what is installed, GPU or not, memory headroom
breakpoint demo --view                   # open the bundled sample report, nothing is run
```

The sample report under `breakpoint_mjx/assets/demo/sample_report/` is the output of a
real CPU smoke-budget run of a Go1 locomotion policy. It is a smoke-budget tour, not a
converged GPU search.

**Get a policy to test.** Either bring your own (below) or train a Go1 baseline with
MuJoCo Playground's PPO recipe (about 10 minutes on a modern GPU):

```bash
python scripts/train.py --target go1 --out out    # writes out/go1/go1_policy.pkl
```

**Check it walks, then search it:**

```bash
breakpoint check-policy --robot go1 --policy out/go1/go1_policy.pkl
breakpoint run --robot go1 --policy out/go1/go1_policy.pkl --budget standard
```

`check-policy` rolls the policy out at nominal parameters and exits with code 3 (with a
diagnosis: observation statistics, action saturation, likely cause) if it cannot walk.
A policy fed the wrong observation order still runs, it just falls everywhere, and the
resulting report would look exactly like a real one; this check catches that first.

**Read the report.** `<out>/<robot>/report.html` opens in any browser with the replays
embedded. Each family is a region of parameter space where the policy fails, with a
name, a representative replay, how often it failed and its plausibility score. The same
run writes `failures.json` (every evaluated world) and `families.json`.

**Defects vs coverage gaps.** Tell BreakPoint what the policy was trained with
(`--train-dr training_dr.yaml`, or a contract from `breakpoint init --from ...`, which
carries it) and every family is classified against it. A failure whose certified cause
sits inside the trained ranges is a **defect** of the policy. A failure on an axis or
value training never covered (latency on a policy trained with no delay) is a
**coverage gap**: expected, and the fix is a randomization term, so the run also writes
`training_dr_patch.yaml` with the ranges to add, capped at the plausible band, and the
Isaac Lab / legged_gym field each one maps to. `breakpoint demo` does this for the
bundled policy (`breakpoint_mjx/assets/demo/go1_training_dr.yaml`).

### Bring your own policy

Policies: `onnx:/path/policy.onnx`, `params:/path/weights.npz` (named numpy arrays, e.g.
a torch `state_dict` dumped to numpy), or a brax params `.pkl`. Policies trained on a
stack of past observations (Isaac Lab `history_length`, legged_gym-style frame stacks)
and RSL-RL recurrent (LSTM/GRU) actors are supported; see "Stateful policies" in
`docs/byo-policy.md`. Robots: `go1`, `barkour`,
`g1`, `spot`, `t1`, `h1`, `berkeley_humanoid`, or a path to your own `.urdf`. Trained in
Isaac Lab, RSL-RL, legged_gym or unitree_rl_gym? `breakpoint init --from rsl_rl <run dir>`
writes a policy contract (observation order, scales, PD gains, training randomization).
See `docs/byo-policy.md` and `docs/contract.md`.

## Rate, gate, verify

**Rate** a policy on a fixed battery (`breakpoint_mjx/batteries/*.yaml`):

```bash
breakpoint rate --robot go1 --policy ./policy.pkl --battery legged-quadruped-v1 --out report.json
breakpoint rate --robot go1 --policy ./policy.pkl --budget smoke     # a quick CPU preview
```

The report is `breakpoint.report/v1` JSON (schema in `docs/ratings.md`): per-cell
falls, fall rate, 95% interval and pass/fail, plus the battery and checkpoint sha256
hashes. Reports never contain weights, trajectories or file paths beyond the policy name.

**Gate** a candidate against the policy in production:

```bash
breakpoint gate --robot go1 --baseline checkpoints/prod.pkl --candidate checkpoints/candidate.pkl \
    --battery legged-quadruped-v1 --budget smoke --out gate.json
```

The gate also compares velocity-tracking error on every seed where neither checkpoint
fell (an exact paired sign test with a 10% deadband), which catches a policy getting
worse at walking before it starts falling. It is advisory unless you pass
`--fail-on-degradation` (alpha is then split with the fall tests).

Exit 0 means no evidence the candidate falls more than the baseline; exit 1 means a
regression (the cells are printed); exit 2 means the gate could not run (never reported
as a regression). The verdict is an exact paired sign test, pooled across the battery and
Holm-adjusted per cell, so false alarms stay at or below `--alpha` (default 0.05) however
many cells there are; cells that got worse without enough evidence are re-run alone on
fresh seeds (`--escalate-to`, 32 by default for `standard`). With 8 seeds and 15 cells the
original "newly failing cell" rule raised a false alarm on about half of all builds where
nothing changed; it is still available as `--decision rule`. Measured false-alarm and
detection rates: `docs/ci-gate.md`. For GitHub Actions, copy
`examples/github-workflow.yml` into your policy repo.

**Sign and verify** a report (needs `.[attest]`):

```bash
breakpoint keys new --out ~/.breakpoint/keys
breakpoint sign report.json --key ~/.breakpoint/keys/breakpoint.key --out signed.json
breakpoint verify signed.json --pubkey ~/.breakpoint/keys/breakpoint.pub --policy ./policy.pkl
```

A self-signed report proves integrity, not independence: trust comes from whose public
key verifies it. Details: `docs/attest.md`.

## Limits (read these)

- **Simulation only.** Everything runs in MuJoCo MJX (with optional cross-checks in the
  C MuJoCo engine). No real hardware is involved at any stage, and a clean BreakPoint
  result is not evidence that a policy is safe on a real robot.
- **Search is not proof.** A run that finds no failure family is a search result under a
  budget, not a proof that no failure exists.
- **Plausibility is a heuristic.** The sim-to-real plausibility score is a transparent,
  hand-set function of parameter ranges, not a measured or calibrated sim-to-real metric.
  The search weights failures by its joint form (each axis at its hard bound halves it),
  so compound worst-case corners rank low instead of costing 1/16 per axis.
- **Observation adversaries act on sensors only.** Noise, bias and dropout touch the
  sensor channels a target declares, never the velocity command, the policy's previous
  action or a gait clock; a dropped channel holds its last reading. Ratings record this
  as `engine.obs_model` (`sensor-hold-v2`); ratings under the earlier model are not
  comparable.
- **Per-target validation.** Results for one robot or simulator configuration do not
  generalise to another without re-running.
- **Scope.** Velocity-tracking locomotion (quadrupeds and humanoids) plus the Panda
  pick-cube target. MLP, observation-history and LSTM/GRU policies; no vision or VLA
  policies; no heightfield terrain yet.
- **Not a new idea.** Searching environment parameters for a fixed policy's failures is
  established in research (e.g. adversarial attacks on quadruped controllers, RSS 2024;
  RoboGate, 2026). What BreakPoint adds is the engineering around it; `docs/landscape.md`
  compares it with the alternatives, including a measured head-to-head.
- **Synthetic benchmarks are labelled.** Several docs (`docs/contract.md`,
  `docs/coverage.md`, `docs/famrisk.md`, `docs/mincause.md`, `docs/riskcert.md`)
  validate methods on planted, synthetic landscapes or scrambles of a real policy. They
  say so in their headings; they are method checks, not field results.

## About the bundled demo policy

`breakpoint demo` (without `--view`) runs a bundled Go1 checkpoint end to end, so you
can see a full report before training anything. The checkpoint
(`breakpoint_mjx/assets/demo/go1_policy.pkl`) is a brax PPO params pickle for MuJoCo
Playground's `Go1JoystickFlatTerrain`, trained by Inoki Labs with this repository's
`scripts/train.py`, and released under the same Apache-2.0 licence as the code. It is a
baseline demo policy, not a tuned or hardened one: it is meant to show what BreakPoint
finds, not to be deployed.

## Tests

```bash
python -m pip install -e ".[dev,attest]"
python -m pytest -q
```

GPU tests skip unless a GPU is present and `BP_RUN_GPU=1` is set.

## Repository layout

- `breakpoint_mjx/`: the engine (search, evaluation, targets, adapters, evidence,
  ratings, gate, attest, CLI)
- `breakpoint_mjx/batteries/`: rating batteries; `breakpoint_mjx/configs/`: theta
  spaces and search config
- `scripts/`: training, benchmarks, evaluation helpers
- `docs/`: method notes and measured results
- `colab/`: GPU notebooks and cells (validation, ratings, evidence, baselines)
- `examples/`, `action.yml`: the CI gate as a GitHub Action

## Contributing

See `CONTRIBUTING.md`. Security issues: please report privately (see `CONTRIBUTING.md`)
rather than in a public issue.
