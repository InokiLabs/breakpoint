# BreakPoint benchmark suite

A reproducible benchmark of whether BreakPoint's directed failure search finds meaningful
robot-policy failures more efficiently than conventional evaluation, plus validation of
those failures and a policy-regression experiment. **Simulation only**: no result here
involves a physical robot.

* `BENCHMARK_PLAN.md`: what exists, what is new, what cannot be evaluated, and every
  predefined criterion (written before any result).
* `reports/TECHNICAL_REPORT.md`: methodology, results, limitations.
* `reports/INVESTOR_BENCHMARK_SUMMARY.md`: one-page results summary.
* `configs/perturbations.md`: every perturbation range with its physical rationale.

## Experiments

| | question | how |
|---|---|---|
| A | Does directed search find distinct failures faster than standard sampling? | Uniform, scrambled Sobol, LHS + boundary (RoboGate-style) and BreakPoint directed (multi-restart CMA-ES), same evaluator, same box, same budget, 3 bound regimes |
| B | Are the failures real? | Each selected failure re-run on 20 fresh seeds (95% Clopper-Pearson), physical-validity screen, replay of validated failures in C MuJoCo with full-body collision |
| C | Can the release gate tell a regression from noise? | `breakpoint gate` on the fixed `legged-quadruped-v1` battery, plus a separate adversarial battery built from Experiment B's validated failures |

## Layout

```
benchmark/
  BENCHMARK_PLAN.md       plan and predefined criteria
  run_benchmark.py        the runner (stages: search, replay, regression, analyze)
  configs/                smoke.yaml, pilot.yaml, full.yaml, perturbations.md
  baselines/samplers.py   the scrambled Sobol baseline (uniform, LHS + boundary and
                          directed are reused unchanged from breakpoint_mjx.equal_budget)
  evaluation/             config resolution, the recording evaluator, search wrapper,
                          validation, replay, regression, provenance
  analysis/               metrics (pure, unit-tested), CSVs, figures, tables
  tests/                  unit tests (no simulator needed)
  results/<config>/       raw run files, CSVs, figures (PNG + PDF), provenance
  reports/                technical report and investor summary
```

Each run directory holds:

* `provenance.json`: software versions, CPU/GPU, git commit, config and checkpoint hashes
* `runs/<policy>__<regime>__<method>__rep<k>.json`: every evaluated condition (theta,
  failure label, validity flag and reason, time to failure, min height), timing per batch,
  clustering, and the validation results for that run's selected candidates
* `replay/*.json`, `regression/*.json` (+ the gate's own `*.gate.json` reports)
* `csv/runs.csv`, `csv/summary.csv`, `csv/ratios.csv`, `csv/validation.csv`,
  `csv/regression.csv`
* `figures/fig1..fig5*.png|pdf`, `RESULTS_TABLES.md`

## Reproduce

Environment (Python 3.11 or 3.12):

```bash
git clone https://github.com/InokiLabs/breakpoint.git && cd breakpoint
pip install -e ".[sim,dev,bench]"
```

Tests (no simulator, about a second):

```bash
python -m pytest benchmark/tests -q
```

Benchmarks, smallest first. Each stage resumes where it stopped; artifacts are never
overwritten (use `--out` for a fresh directory).

```bash
python -m benchmark.run_benchmark --config benchmark/configs/smoke.yaml   # plumbing, ~4 min on a laptop CPU
python -m benchmark.run_benchmark --config benchmark/configs/pilot.yaml   # ~10 min
python -m benchmark.run_benchmark --config benchmark/configs/full.yaml    # ~3-4 h on an Apple M3 CPU
```

Single stages: `--stage search | replay | regression | analyze`.

On a GPU machine install the CUDA build of JAX first (`pip install "jax[cuda12]"`); the
provenance file records which backend ran. Larger budgets, more repetitions or more
policies are config changes only.

### Adding a policy

Add an entry to `policies:` in a config:

```yaml
  - name: my-policy
    robot: go1                         # a registered target: go1, barkour, g1, ...
    checkpoint: path/to/policy.pkl     # or onnx:..., params:...npz (see docs/byo-policy.md)
    obs_spec: path/to/contract.yaml    # optional policy contract
```

A configuration whose nominal world falls in more than 25% of rollouts is recorded as not
walking and excluded.

### Real regression comparison

Add a real candidate checkpoint of the same policy:

```yaml
regression:
  pairs:
    - {name: real-pair, kind: real, candidate: path/to/candidate.pkl}
```

## Fairness rules the code enforces

* Every method spends exactly the same number of rollouts (`budget` must divide by the
  shared batch size; the runner refuses otherwise and checks the recorded count).
* Every method calls the same compiled evaluator; the failure label is computed from the
  same rollout the method saw, by the engine's own signals, so no method gets a different
  criterion.
* The directed search algorithm is the engine's unchanged; its plan (restarts, popsize,
  generations) is fixed in the config before running.
* Validation, replay and adversarial-battery seeds come from streams no search uses.
* Wall-clock is measured per method including its own overhead (sampler, CMA-ES updates).
