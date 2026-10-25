# Benchmark plan

Written 2026-10-18, before any benchmark result was produced. Everything under
"Predefined criteria" was fixed here first and is not changed after seeing data; any later
deviation is listed in the technical report with the reason.

The code and recorded outputs in this repository are the source of truth, not the pitch
deck.

## 1. What already exists (inspected)

| piece | where | used by the benchmark |
|---|---|---|
| Policy loading (brax pickle, ONNX, params npz, contracts, history and recurrent actors) | `check_policy.load_policy`, `check_policy._load_env`, `adapters/` | yes, unchanged |
| Batched rollouts (`jax.vmap` over a perturbed MJX model, one compile per batch shape) | `evaluator.make_evaluator`, `rollout.rollout_one` | yes, unchanged |
| Perturbation space: 16 dims per legged robot, each with stress-test `bounds`, a `nominal` and a `plausible` deployment band | `configs/<robot>/theta_space.yaml`, `thetas.apply_thetas`, `rollout.py` | yes |
| Failure signals per rollout: fall (trunk below a robot-specific height or tipped past an orientation limit), time to failure, min height, tracking error, instability | `failure.component_signals` | yes: the failure label |
| Physical-validity gate: non-finite state, floor penetration beyond 4 cm, joint accelerations above 1e4, causal window and settle window | `failure._validity`, `docs/validity.md` | yes: invalid rollouts are never failures |
| Directed search: multi-restart CMA-ES, fitness = failure score x plausibility weight | `multi_search.multi_cmaes_search` | yes, as "BreakPoint directed", unchanged |
| Equal-budget harness with uniform and RoboGate-style LHS + boundary arms, exact rollout parity, C-MuJoCo replay hook | `equal_budget.py` | yes: its arms and replay are reused |
| Second engine: C MuJoCo on Playground's full-collision Go1 scene, compiled env step | `backends/mujoco_c.py`, `crossval_engines.replay_family`, `equal_budget.make_replay_fn` | yes (Go1 only) |
| Release gate: paired seeds, exact sign tests, Holm per cell, escalation to fresh seeds | `gate.py`, `gate_stats.py`, `batteries/legged-quadruped-v1.yaml` | yes, for Experiment C |
| Interval helpers (Clopper-Pearson, Wilson) | `ratings.clopper_pearson`, `stats.wilson_interval` | yes |

### Where the deck's numbers come from

* **"About 92% failure rate"**: `docs/equal_budget.md` / `docs/equal_budget_lhs_cpu_smoke.json`.
  Uniform sampling over the full stress-test `bounds`, horizon 150, bundled Go1 policy:
  92.5-93.5% of worlds fall. That is a property of the broad bounds, not a real-world
  failure probability.
* **"2x vs random"**: same file. Directed search found 304 failures inside the plausible
  bands per 384 rollouts vs 155.5 for uniform (2 seeds, CPU smoke, horizon 150). The
  directed objective is weighted toward plausible conditions, so this metric partly
  measures what it optimizes.
* **Go1 latency result** (site hero): one world from `media/demo/demo_measurements.json`
  (latency 1.86 steps, rounded to 2 = 40 ms; dropout 0.134; slope -3.9 deg). Re-verified
  2026-10-17: 56/64 falls with only those three dims perturbed, 0/64 nominal.
* **Repair table** (deck slide 5): the hardened checkpoints it describes
  (`retrain_bp_wide`, `media/dl/...`) are **not in this repository**, so it cannot be
  re-checked or used here.

## 2. Policies and robots actually available

| configuration | checkpoint | status |
|---|---|---|
| `go1-demo` | `breakpoint_mjx/assets/demo/go1_policy.pkl` (Playground Go1 joystick PPO), native gains Kp 35 | walks (verified in the run) |
| `go1-demo-kp25` | **same weights**, actuator gains Kp 25 / Kd 0.5 via `docs/byo/bundled_go1_kp25.yaml` | walks 32/32 (docs/byo-policy.md); a deployment variant, **not an independent policy** |
| Isaac Lab Go2 (third-party, fetchable) | `scripts/fetch_thirdparty.py` | **excluded**: does not walk on the Go1 model at nominal (8/32 and 22/32 survival), so every world "fails" and the comparison would be meaningless |
| Barkour, G1, T1, Spot | none in the repo | targets exist (Barkour, G1) but no trained checkpoint; training needs a GPU (100M steps, ~28 h at the ~1,000 env-steps/s measured on this machine) |

So the benchmark runs on **one real policy in two actuation configurations**. The
three-independent-policy requirement **cannot be met** in this environment; the runner
accepts any number of configurations and needs only a checkpoint path (plus a contract)
per additional policy.

## 3. Compute available

Apple M3, 8 cores, 16 GB, **no GPU** (JAX CPU backend). Measured MJX throughput with the
bundled policy: 25 ms per rollout at horizon 250, 49-57 ms at horizon 500 (batch 64-128).

## 4. What is new

* `benchmark/` (this directory): the runner, configs, a Sobol baseline, a recording
  evaluator, validation, replay and regression stages, metrics, figures, reports, tests.
* No change to the search algorithm, the failure signals, the validity gate or policy
  loading. The Sobol sampler is the only new search method, and it is a baseline.

## 5. What cannot be evaluated here

* **Physical hardware**: none. Every result is simulation only.
* **Three or more independent policies**: only one real walking checkpoint (see 2).
* **A real checkpoint pair for regression testing**: none in the repo. Experiment C is
  runnable on any pair; here it runs only calibration controls (identical pair, and
  explicitly synthetic weight-noise candidates), and the real comparison is marked
  **incomplete**.
* **Torque / motor-strength limits**: not a perturbation the engine implements.
  "Motor strength" is represented only by PD-gain scaling (`kp_scale`, `kd_scale`), which
  changes stiffness, not the torque ceiling. Not faked.
* **GPU time**: no GPU. Cost is reported as rollouts, simulation steps and CPU wall time.
* **Cross-engine replay beyond Go1**: the C-MuJoCo full-collision scene exists only for Go1.

## 6. Methods compared (Experiment A)

All methods use the same configuration, perturbation box, horizon, failure label and
compiled evaluator (same batch shape), and spend exactly B rollouts.

1. **Uniform**: i.i.d. uniform over the box (`equal_budget` arm, unchanged).
2. **Sobol**: scrambled Sobol sequence over the box (new; `benchmark/baselines`).
3. **LHS + boundary** (RoboGate-style, arXiv 2603.22126): half Latin hypercube, half
   Gaussian refinement around worlds nearest the failure threshold (`equal_budget` arm,
   our implementation of their published description).
4. **BreakPoint directed**: multi-restart CMA-ES (`multi_search`, unchanged). Plan
   restarts x popsize x generations = B; set in the config before running, not tuned.

CMA-ES is not added as a separate baseline because BreakPoint's directed method already
is multi-restart CMA-ES; a second CMA-ES arm would compare it with itself.

## 7. Perturbations and bound regimes

Supported perturbations (Go1 `theta_space.yaml`; physical justification in
`configs/perturbations.md`): control latency, ground friction, trunk mass, centre-of-mass
offset, PD-gain scaling (stiffness proxy for motor strength), joint friction loss and
armature, sensor noise / dropout / bias, an external push (magnitude and timing), and
floor slope.

Three regimes, run separately (the first was added after the smoke run; see section 11):

* **Near-nominal bounds**: each plausible band shrunk toward nominal to 25% of its
  distance on each side (a well-controlled deployment). Note: latency tops out at 0.5
  control steps here, which the engine rounds to 0, so latency is effectively off.
* **Conservative operating bounds**: each dim's `plausible` band (realistic deployment
  conditions per the repository's own plausibility model).
* **Broad stress-test bounds**: each dim's `bounds`.

## 8. Predefined criteria

* **Failure**: a physically valid fall. `fall == 1` from `failure.component_signals`
  using the target's own thresholds (Go1: trunk below 0.18 m or trunk up-axis below 0.5;
  G1 uses 0.40 m), and `sim_invalid == 0`. A rollout flagged invalid is recorded as an
  artifact, never a failure.
* **Severity**: time to fall (earlier = more severe) and minimum trunk height.
* **Distinct failure conditions**: greedy leader clustering in discovery order over
  failures, in coordinates normalized to the active regime's box ((theta - lo) / span).
  A failure starts a new cluster when its L-infinity distance to every existing leader
  exceeds eps = 0.10 (every parameter within 10% of its range = same condition).
  Sensitivity is reported at eps = 0.05, 0.20 and 0.30. Caveat fixed in advance: in 16
  dimensions most random points are far apart in L-infinity, so space-filling samplers
  will produce many clusters; this is why the mode-diversity metric below is also reported.
* **Failure modes** (diversity): the direction-aware dominant deviating dimension of each
  failure ("high latency_steps"), from `equal_budget.dominant_modes`. At most 32.
* **Validation (Experiment B)**: each selected candidate is re-run on N = 20 fresh seeds
  from a seed stream disjoint from every search stream. p = valid falls / N with a 95%
  Clopper-Pearson interval. A candidate is **validated** if the CP 95% lower bound is
  >= 0.5 (at N = 20 that needs >= 15/20 valid falls). Invalid rollouts count as
  non-failures and are reported separately.
* **Candidate selection** (per configuration x regime x method x repetition, fixed rule,
  no choice by outcome): (a) the first 3 failures in discovery order, for time to first
  validated failure; (b) up to 8 cluster leaders drawn uniformly at random (seeded) from
  all clusters. If a method has <= 8 clusters, all are validated.
* **Estimated distinct validated failures** = clusters x validation rate of the sampled
  leaders, with a CP interval on the rate (exact when all leaders were validated).
* **Time to first validated failure**: rollouts spent before the first of the in-order
  failures that validates; reported as "not reached" when none of the first 3 validates.
* **Configuration check**: a configuration whose nominal world gives valid falls in more
  than 25% of 2 batches of rollouts does not walk and is excluded (recorded, not hidden).
* **Cross-engine replay**: 2 validated candidates per configuration x regime x method x
  repetition (random, seeded), 8 C-MuJoCo seeds each, the repository's existing
  reproduction rule. Cross-engine agreement is **not** evidence of real-world transfer.
* **Experiment C acceptance criterion** (fixed before running): the gate's statistical
  decision (`--decision stat`, alpha = 0.05, tolerance 0, escalation to fresh seeds) on
  `legged-quadruped-v1`. A candidate is a regression only if that test rejects. The
  adversarial battery (validated failures from Experiment A) is reported separately and
  never feeds the gate decision.
* **Comparisons between methods**: per-repetition values, mean and range across
  repetitions; ratios use the across-repetition means with a bootstrap 95% interval
  over repetitions. With 5 repetitions these intervals are wide and are reported as such.

## 9. Seeds and leakage

* Search repetition r uses seed r for every method (each method has its own rollout key
  stream derived from it, as in `equal_budget`).
* Validation keys: base 9,000,000 + a per-candidate index; never used by any search.
* Replay keys: the C-MuJoCo backend's own seed stream (base 77,000).
* Experiment C: the battery's fixed seeds; adversarial-battery seeds base 5,000,000.
* Nothing is trained, so there is no training data to leak.

## 10. Staging

smoke (plumbing) -> pilot -> full, all from `benchmark/configs/`. The largest run that
fits this CPU is executed; any reduction from the suggested full settings is listed in
the report with the reason.

## 11. Changes made after the smoke run (before any pilot or full result)

The smoke run (horizon 150, 200 rollouts, 1 repetition) is a plumbing check, not a result.
It exposed two design problems, fixed before the pilot:

1. **Replay seeds 2 -> 8.** The repository's reproduction rule (one-sided 90%
   Clopper-Pearson lower bound >= 0.5) cannot be met with 2 seeds even at 2/2, so every
   replay was "not reproduced" by construction. At 4 seeds only 4/4 passes; 8 seeds lets
   7/8 or 8/8 reproduce.
2. **A near-nominal regime was added.** About 54% of worlds inside the plausible bands
   already failed for this policy (0/200 at nominal). When a random sample fails half the
   time, search efficiency barely matters, so the comparison would not test what directed
   search is for: finding rare failures. The near-nominal regime is where failures should
   be rare. The conservative and broad regimes are kept unchanged.

Also confirmed, not changed: at eps = 0.10 (and even 0.30) L-infinity leader clustering
merges almost no space-filling samples in 16 dimensions, so the distinct-cluster count is
close to the raw failure count for uniform and Sobol, while it does merge the
near-duplicate refinements of LHS + boundary and directed search. That is the metric
working as defined (it removes near-identical conditions), but distinct parameter vectors
are not distinct failure mechanisms; the report says so and leans on validation rate,
closeness to nominal and mode diversity alongside it.

## 12. Improvement round (written after the full run, before any tuning run)

The full run (§1–11, `results/full/`) showed four weaknesses: directed search finds fewer
distinct failures than random sampling at coarse grouping (eps 0.3), concentrates on fewer
failure modes, finds no reproducible failure near nominal (no method does), and the
validated-failure adversarial battery cannot detect a regression (baseline already fails
91% there). This round tries to fix them **without changing the original method**:
`directed` is still `multi_cmaes_search` with its defaults and stays in every table.

**Candidates** (new, named methods; same budget of 1,000 rollouts, same evaluator):

| name | what |
|---|---|
| `nov_r06` | engine's existing opt-in novelty sharing, radius 0.6, strength 0.8 (its defaults) |
| `nov_r12` | the same, radius 1.2 |
| `restarts10` | 10 restarts × popsize 10 instead of 5 × 20 (the engine docs report more, smaller restarts cover more modes) |
| `robust_k2` | new `breakpoint_mjx.robust_search`: each condition scored as the mean over 2 seeds; 5 restarts × popsize 10 |
| `robust_k4` | the same, 4 seeds per condition; 5 restarts × popsize 5 |

**Tuning (dev) runs.** go1-demo, all three regimes, 2 repetitions on **disjoint seeds**
(rep indices 100–101; the full run used 0–4), candidates plus `directed` and `uniform` as
references (`configs/tune.yaml`, `results/tune/`).

**Selection rule (fixed now).** Pick the one candidate with the highest mean coarse
estimated validated distinct failures (eps-0.3 clusters × leader validation rate), summed
over conservative and broad, among candidates whose eps-0.1 estimated validated distinct in
conservative is at least `uniform`'s. If none qualifies, report that no candidate fixed the
diversity gap. Separately, any candidate that validates at least one near-nominal failure in
the dev runs is also confirmed in near_nominal.

**Confirmation.** The selected variant runs on reps 0–4 for both configurations and all
three regimes (`configs/confirm.yaml`, `results/confirm/`), next to the existing baseline
and directed runs of `results/full/` (linked, not copied or re-run). Seeds 0–4 were never
used for tuning. Success, stated now: in conservative on go1-demo, the variant's ratio to
uniform on coarse estimated validated distinct has a 95% bootstrap lower bound above 1
**and** its eps-0.1 estimated validated distinct is not significantly below `directed`'s.
Every outcome is reported, including failure.

**Boundary regression battery.** A third battery, reported separately: Experiment B
candidates of go1-demo (near_nominal and conservative) whose 20-seed fall rate is in
[0.10, 0.60], up to 24 conditions, 8 paired seeds each (seed base 5,000,000), chosen from
baseline data only. Run on every existing regression pair; the gate's paired statistics,
alpha 0.05.

**Not attempted (needs a GPU or new training).** More independent policies and a real
regression checkpoint pair. Never substituted with fabricated ones.
