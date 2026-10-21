# BreakPoint benchmark: technical report

**Basis: simulation only.** Every number below comes from MuJoCo MJX rollouts, with replay in
C MuJoCo. No physical robot was used. Replay in a second simulator is a check against
artifacts of one solver; it is **not** physical validation.

Raw data: `benchmark/results/full/` (120 run files, replay, regression, provenance).
Generated tables: `benchmark/results/full/RESULTS_TABLES.md`. CSVs: `csv/` and
`supplementary/`. Figures: `figures/`. Plan and predefined criteria, written before any
result: `benchmark/BENCHMARK_PLAN.md`.

## 1. Summary

| question | answer (measured) |
|---|---|
| Does directed search find more failures per rollout? | **Yes in the realistic (conservative) regime**: failure rate 0.88 vs 0.70 for uniform random on go1-demo (ratio 1.24, 95% CI [1.22, 1.28]); 0.96 vs 0.90 on the Kp 25 variant (1.07 [1.05, 1.08]). **No difference in the broad regime** (about 97% of all random samples already fail). **Nothing to find near nominal**: about 1% failure rate for every method, none validated. |
| Does it find more *distinct* failures? | **It depends on what counts as distinct, and the answer is not robust.** At the predefined cluster radius (L∞ 0.10 of the normalised box), directed finds 1.24× more clusters than uniform (conservative, go1-demo). At a coarser radius (0.30) it finds **0.77× [0.73, 0.82]** as many as uniform, and 0.63× in the broad regime. Directed concentrates its rollouts; its extra failures are partly near-duplicates. |
| Are its failures more reproducible? | **Yes, modestly.** Selected failures re-run on 20 fresh seeds: 107/110 validated for directed vs 99/109 for uniform, 93/110 for Sobol and 88/110 for LHS + boundary (conservative regime, both configurations pooled). |
| Do they survive a second simulator? | Validated failures replayed in C MuJoCo with full-body collision, 8 seeds each, conservative + broad: directed 36/40, Sobol 31/40, uniform 29/40, LHS + boundary 26/40. Small samples; intervals overlap. |
| Is it faster? | **No.** Same rollout budget, median search wall-clock 50–56 s for directed vs 48–57 s for the baselines (CMA-ES overhead ~1 s). Rollouts to the first validated failure: 1 for every method in conservative and broad (the first failure found is almost always real), so this metric does not separate methods here. |
| Does it beat the RoboGate-style LHS + boundary baseline? | **Yes, on every metric and at every cluster radius** (e.g. 2.21× [1.87, 2.83] estimated validated distinct failures, conservative, go1-demo; 1.26× clusters even at radius 0.30). |
| Can the release gate distinguish a regression? | **On synthetic candidates, yes; on real ones, untested.** No false alarm on an identical policy (15/120 → 15/120); flagged weight-noise candidates at σ 0.3 (15 → 42 falls of 120) and σ 0.5 (15 → 103). No real checkpoint pair exists, so **the real comparison is incomplete**. The separate adversarial battery missed the σ 0.5 candidate (ceiling effect, Section 6). |

The honest overall reading: in a realistic perturbation box, BreakPoint's directed search
spends more of a fixed budget on conditions that really break the policy, and the failures
it finds are slightly more reproducible across seeds and simulators. It does **not**
discover a wider variety of failure modes than plain random sampling, and it is no faster
in wall-clock. It clearly beats a RoboGate-style LHS + boundary design. Against uniform
random and Sobol the advantage is real but moderate (1.07–1.48× depending on configuration
and metric), and it reverses on coarse diversity.

## 2. What was run

| item | value |
|---|---|
| Robot | Unitree Go1 (MuJoCo Playground model), 16 perturbation dimensions (`configs/perturbations.md`) |
| Policies | **go1-demo**: Playground Go1 joystick PPO checkpoint (sha256 `b442aaa9…`). **go1-demo-kp25**: the *same weights* deployed with actuator gains Kp 25 / Kd 0.5. Not an independent policy (labelled as such everywhere). |
| Nominal check | go1-demo 0/200 valid falls; go1-demo-kp25 3/200. Both walk (exclusion rule: > 25%). |
| Regimes | near_nominal (25% of each plausible band around nominal), conservative (the plausible bands), broad (the stress-test bounds) |
| Methods | Uniform random; scrambled Sobol; LHS + boundary (RoboGate-style, as implemented in `breakpoint_mjx.equal_budget`); BreakPoint directed (multi-restart CMA-ES, the engine's algorithm unchanged; 5 restarts × popsize 20 × 10 generations) |
| Budget | 1,000 rollouts per method per repetition, horizon 500 control steps (10 s), so 500,000 simulation steps per run. Identical for every method; checked by the runner. |
| Repetitions | 5 independent seeds per (configuration, regime, method): **120 search runs, 120,000 search rollouts** |
| Validation | Per run, the first 3 failures plus up to 8 randomly chosen cluster leaders, each re-run on 20 fresh seeds (seed stream disjoint from all search seeds). Validated = one-sided 95% Clopper-Pearson lower bound on fall probability ≥ 0.5 (≥ 15/20). 1,218 candidates, 24,360 validation rollouts. |
| Replay | 2 validated failures per run (conservative + broad), 8 seeds each in C MuJoCo with full-body collision; confirmed by the engine's one-sided 90% reproduction rule |
| Compute | One Apple M3 laptop, JAX CPU backend (no GPU). Search + validation + replay: 3 h 14 min. Provenance in `provenance.json` (versions, git commit, config and checkpoint hashes). |

### Reductions from the requested design (and why)

* **Fewer than 3 real policies.** Only one walking checkpoint exists in the repository. The
  Go2 checkpoint does not walk on the Go1 model; Barkour and G1 have no checkpoints, and
  training them was estimated at ~28 h of this machine's compute. The second configuration
  is a deployment variant of the same weights, not an independent policy. Results
  therefore describe **one policy**; generalisation to other policies is untested.
* **No CMA-ES-only baseline separate from BreakPoint**: BreakPoint's directed method *is*
  multi-restart CMA-ES with a plausibility-weighted objective; a plain CMA-ES arm would
  differ only in the objective weighting. Not run.
* **CPU only.** The protocol is unchanged on GPU; only wall-clock would differ.

### Changes after the smoke run (before any pilot or full result; plan §11)

1. Replay seeds raised from 2 to 8 (with 2 seeds the reproduction rule can never confirm).
2. The near_nominal regime was added because the conservative box already fails about half
   the time for random samples.

### Fix after the full run (Experiment C only)

The first full run passed `--budget full` to the release gate, which accepts only `smoke`
or `standard`; all four gate calls exited with a usage error (exit 2) and the runner
recorded them as "ran". The broken outputs were moved, not deleted, to
`results/full/regression_invalid_gate_budget/`; the config now says `standard`; the runner
now records any exit code other than 0/1 as `gate_error`; and the config loader rejects an
unknown gate budget (unit-tested). The regression stage was then re-run
(`results/full_regression_rerun.log`). Experiments A and B were not affected. Note that
`results/full/config.yaml` is the copy taken at the start of the run and still shows the
old value; `benchmark/configs/full.yaml` is the corrected config.

## 3. Experiment A: discovery at equal budget

Means over 5 repetitions; brackets are the min–max across repetitions for counts and the
95% bootstrap interval over repetitions for ratios. "Est. validated distinct" = clusters ×
that method's validation rate (predefined).

### go1-demo (the independent policy)

| regime | method | failure rate | clusters (eps 0.10) | leaders validated | est. validated distinct | closest failure to nominal (L2, normalised) |
|---|---|---|---|---|---|---|
| conservative | BreakPoint directed | **0.88** | **875.8** [861–906] | 38/40 | **832** | 1.12 |
| conservative | Uniform random | 0.70 | 704.2 [683–725] | 32/40 | 563 | 1.03 |
| conservative | Sobol | 0.71 | 709.2 [703–723] | 29/40 | 514 | 1.08 |
| conservative | LHS + boundary | 0.66 | 581.2 [570–589] | 26/40 | 377 | 1.09 |
| broad | BreakPoint directed | 0.96 | 962.0 [946–970] | 40/40 | 962 | **0.73** |
| broad | Uniform random | 0.97 | 974.2 [968–982] | 40/40 | 974 | 0.98 |
| broad | Sobol | 0.98 | 976.0 [969–983] | 39/40 | 951 | 0.93 |
| broad | LHS + boundary | 0.97 | 741.4 [728–765] | 39/40 | 723 | 1.01 |
| near_nominal | all four | 0.01 | 5–7 | 0/27–0/34 | 0 | 1.44–1.58 |

Directed / baseline ratios, go1-demo:

| regime | metric | vs uniform | vs Sobol | vs LHS + boundary |
|---|---|---|---|---|
| conservative | failure rate | 1.24 [1.22, 1.28] | 1.23 [1.21, 1.26] | 1.33 [1.30, 1.36] |
| conservative | est. validated distinct (eps 0.10) | 1.48 [1.27, 1.72] | 1.62 [1.30, 2.14] | 2.21 [1.87, 2.83] |
| conservative | clusters, eps 0.20 | 1.21 [1.18, 1.23] | 1.20 [1.18, 1.22] | 1.96 [1.93, 1.99] |
| conservative | clusters, eps 0.30 | **0.77 [0.73, 0.82]** | **0.76 [0.72, 0.81]** | 1.26 [1.20, 1.33] |
| conservative | failure modes (dominant dimension) | **0.88 [0.81, 0.96]** | **0.85 [0.75, 0.94]** | **0.88 [0.81, 0.96]** |
| conservative | closest-failure distance | 1.09 [0.96, 1.20] | 1.04 [0.92, 1.13] | 1.03 [0.90, 1.15] |
| broad | est. validated distinct | 0.99 [0.98, 1.00] | 1.01 [0.98, 1.06] | 1.33 [1.27, 1.42] |
| broad | clusters, eps 0.30 | **0.63 [0.59, 0.66]** | **0.62 [0.59, 0.66]** | 1.21 [1.14, 1.28] |
| broad | closest-failure distance (lower = nearer nominal) | **0.75 [0.69, 0.81]** | **0.79 [0.71, 0.87]** | **0.73 [0.67, 0.80]** |
| broad | failure modes | 1.28 [1.15, 1.43] | 1.32 [1.16, 1.50] | 1.32 [1.24, 1.43] |

go1-demo-kp25 shows the same pattern with smaller gains in the conservative regime
(failure rate 1.07 [1.05, 1.08] vs uniform; est. validated distinct 1.09 [1.06, 1.15];
clusters at eps 0.30: 0.65 [0.62, 0.69]) because uniform sampling already fails 90% of the
time there. Full tables: `RESULTS_TABLES.md`, `supplementary/eps_sensitivity.csv`.

Figures: `fig1_discovery_vs_budget` (estimated validated distinct failures against rollouts
spent), `fig1b_discovery_vs_wallclock`, `fig4_failure_distribution_<regime>`.

### Efficiency and time

| | directed | uniform | Sobol | LHS + boundary |
|---|---|---|---|---|
| median search wall-clock per 1,000 rollouts (s), range over configs × regimes | 50–56 | 48–55 | 48–57 | 48–115* |
| search-side overhead (sampler / CMA-ES) | ~1 s | < 0.1 s | < 0.1 s | < 0.3 s |
| rollouts to first validated failure (conservative, broad) | 1 | 1 | 1 | 1–2 |

\* One LHS run (go1-demo-kp25, conservative, rep 3) contains a single 711 s batch where the
others take 5–7 s: a host stall (the laptop sleeping or other load), not a cost of the
method. Medians are reported for this reason (`supplementary/wall_clock.csv`). Simulation steps are identical by construction.

## 4. Experiment B: are the failures real? (simulation only)

| regime | method | candidates validated (20 fresh seeds, CP lower ≥ 0.5) | invalid validation rollouts | C MuJoCo replay confirmed |
|---|---|---|---|---|
| conservative | BreakPoint directed | **107/110** | 2/2,200 | **18/20** |
| conservative | Uniform random | 99/109 | 5/2,180 | 14/20 |
| conservative | Sobol | 93/110 | 2/2,200 | 13/20 |
| conservative | LHS + boundary | 88/110 | 1/2,200 | 10/20 |
| broad | BreakPoint directed | 110/110 | 12/2,200 | 18/20 |
| broad | Uniform random | 109/110 | 11/2,200 | 15/20 |
| broad | Sobol | 109/110 | 15/2,200 | 18/20 |
| broad | LHS + boundary | 109/110 | 14/2,200 | 16/20 |
| near_nominal | all four | **0/339** | 0 | none to replay |

(Both configurations pooled. `csv/validation.csv`, `replay/*.json`, `fig3_reproducibility`.)

* **Physical-validity screen.** During search, at most 11 of 1,000 rollouts per run were
  flagged by the engine's validity gate (non-finite state, floor penetration, joint
  accelerations) and none were counted as failures. Invalid rollouts are rare, and most
  of them occur in the broad regime.
* **Near nominal, nothing validates.** Every method found a handful of falls (about 1%;
  9–12% for the Kp 25 variant), but none reproduces on fresh seeds: the highest per-candidate
  fall probability seen was 13/20. These are seed-dependent events, not reliable failure
  conditions. With 339 candidates, all 0/339, the policy has no reproducible failure inside
  the near-nominal box *at this budget*.
* **Replay.** Most validated MJX failures also fail in C MuJoCo with full-body collision. The
  per-cell samples are small (20 failures per method per regime), so the differences
  between methods are suggestive only. Replay disagreements were not investigated
  individually.

## 5. Why directed does not win on diversity (investigation)

1. **CMA-ES concentrates by design.** Each restart converges on a region of high failure
   score. Within the predefined radius (0.10), points 0.1–0.3 apart count as distinct, so
   concentration still produces many "distinct" clusters. At 0.30 they merge (directed:
   876 → 535 clusters; uniform: 704 → 697). Space-filling methods barely change, because
   their points are spread out by construction.
2. **The 0.10 radius is close to degenerate in 16 dimensions.** For the space-filling
   methods nearly every failure is its own cluster (clusters ≈ failures). The predefined
   metric therefore mostly measures failure *count*, which is where directed is strong.
   We kept the predefined metric as the primary result and report the sensitivity
   alongside it (plan §8 predefined the 0.05/0.2/0.3 sensitivity values).
3. **The objective rewards plausibility and severity, not novelty.** The directed fitness
   is failure score × transfer likelihood, with no diversity term. In the conservative box
   its failures concentrate on the dominant modes "high latency" (30% of its failures vs
   18% for uniform) and "high sensor dropout": effective number of modes 6.3 vs 7.3
   (`supplementary/mode_concentration.csv`).
4. **In the broad box, failures are not rare.** 97% of random samples fail, so a search
   method cannot improve the hit rate; directed instead finds failures **closer to
   nominal** (25–37% shorter normalised distance) and with more distinct dominant modes.
   That is consistent with its plausibility weighting doing what it is designed to do.
5. **Near nominal there is nothing reproducible to find**, so no method can show an
   advantage there.

Implication: if the product claim is "more diverse failures", the current algorithm does not
support it against random sampling; a novelty or coverage term in the objective (not
changed in this benchmark, per the brief) would be the obvious thing to test next. If the
claim is "higher yield of realistic, reproducible failures in a deployment-plausible box",
the data support it moderately.

## 6. Experiment C: release-gate regression testing

Acceptance criterion (predefined): the gate's statistical decision on the fixed
`legged-quadruped-v1` battery (15 cells × 8 seeds, horizon 500), alpha 0.05, tolerance 0.
Exit 0 = pass, exit 1 = regression. The adversarial battery (24 conditions drawn from
Experiment B's validated conservative-regime failures, 8 paired seeds each) is reported
**separately** and does not enter the gate decision.

| pair | candidate | fixed battery falls (baseline → candidate) | cells passed | gate verdict | adversarial battery falls (baseline → candidate, of 192) | adversarial verdict |
|---|---|---|---|---|---|---|
| real-pair | **no real second checkpoint exists** | — | — | **incomplete** | — | — |
| null-identical | the same checkpoint | 15/120 → 15/120 | 13/15 | pass (correct) | 174 → 174 | no regression (correct) |
| synthetic σ 0.1 | weight noise, 0.1 × per-tensor std (**synthetic**) | 15/120 → 17/120 | 13/15 | pass | 174 → 181 | no regression |
| synthetic σ 0.3 | weight noise 0.3 (**synthetic**) | 15/120 → 42/120 | 5/15 | **regression** (p < 0.001) | 174 → 185 | regression |
| synthetic σ 0.5 | weight noise 0.5 (**synthetic**) | 15/120 → 103/120 | 0/15 | **regression** (p < 0.001) | 174 → 182 | **no regression (missed)** |

Reading:

* The gate did not false-alarm on an identical policy, and flagged both large synthetic
  degradations. The σ 0.1 candidate's change (+2 falls) is within noise; whether that is a
  true negative or a miss cannot be decided, since its true effect is unknown.
* **The adversarial battery is a poor regression detector as built.** Its conditions are,
  by construction, ones where the baseline already falls (174/192 = 91%), so there is
  almost no room to observe a candidate getting worse. It missed the σ 0.5 candidate, which
  the fixed battery caught decisively. Validated failures are useful for diagnosing a
  policy, but a regression battery needs conditions where the baseline *passes*. This
  is a design finding about the benchmark's adversarial battery, not about the gate.
* **Synthetic noise is not a real regression.** These candidates show that the statistics
  work; they do not show that the gate catches the kinds of regressions real retraining
  produces. Running `real-pair` needs a second real checkpoint of the same policy (see
  `README.md`). Never substitute a fabricated one.

Files: `regression/*.json`, `regression/*.gate.json`, `csv/regression.csv`,
`fig5_regression`.

## 9. Improvement round (after the full run)

Protocol and success criteria were written into `BENCHMARK_PLAN.md` §12 **before** any
tuning run. The original `directed` method was not changed; improvements are new, named,
opt-in methods. Tuning used disjoint seeds (reps 100–101); confirmation used reps 0–4,
next to the existing baseline runs (linked from `results/full/`, not re-run).

### 9.1 Diversity: directed + novelty (radius 1.2)

Five candidates were tuned on go1-demo (`results/tune/`, `selection.json`): the engine's
existing novelty sharing at radius 0.6 and 1.2, 10 × 10 restarts, and a new noise-robust
search (`breakpoint_mjx/robust_search.py`, each condition scored on 2 or 4 seeds). The
pre-registered rule selected **`nov_r12`** (`multi_cmaes_search(novelty=True,
novelty_radius=1.2, novelty_strength=0.8)`), which pushes later samples away from regions
where failures were already found.

Confirmation, 5 reps × 2 configurations × 3 regimes (`results/confirm/`):

| | conservative, go1-demo | conservative, Kp 25 | broad, go1-demo | broad, Kp 25 |
|---|---|---|---|---|
| coarse validated distinct (eps 0.3), nov_r12 / original directed | **1.25 [1.13, 1.36]** | **1.24 [1.13, 1.35]** | **1.24 [1.16, 1.32]** | **1.22 [1.13, 1.32]** |
| coarse validated distinct, nov_r12 / uniform | 1.14 [0.97, 1.34] | 0.83 [0.76, 0.90] | 0.78 [0.75, 0.80] | 0.76 [0.73, 0.80] |
| (for reference) original directed / uniform | 0.91 [0.80, 1.05] | 0.67 [0.62, 0.72] | 0.63 [0.59, 0.66] | 0.63 [0.59, 0.66] |
| fine validated distinct (eps 0.1), nov_r12 / original directed | 0.99 [0.90, 1.08] | 0.97 [0.93, 1.01] | 1.01 [1.00, 1.02] | 1.00 [0.99, 1.00] |
| fine validated distinct, nov_r12 / uniform | 1.46 [1.26, 1.69] | 1.07 [1.00, 1.13] | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] |

**Pre-registered success test: not met.** Lower bound of nov_r12 / uniform on coarse
diversity (conservative, go1-demo) is 0.97, not above 1. The second condition (no
significant loss against directed on the fine metric) is met.

Reading: novelty sharing recovers about a quarter more coarse-distinct validated failures
than the original directed search in every cell, at no cost in failure yield. That closes
most of the diversity gap to random sampling in the realistic box for go1-demo, but it does
not overtake random sampling on coarse diversity, and it still trails it in the stress-test
box and on the Kp 25 variant. Failure modes (dominant dimension) are unchanged against
directed in conservative and still below uniform (0.86 [0.78, 0.96]). A wall-clock of
230 s in one cell is a host stall again (the run went overnight); simulation steps are
identical by construction.

### 9.2 Near nominal: noise-robust search

Scoring each condition on 2 or 4 seeds (fitness = mean failure score) did **not** find a
reproducible near-nominal failure on the tuning seeds (0 validated, like every other
method), and it cuts the number of conditions tested by 2–4×, which halves or quarters the
failure count elsewhere. Not selected. Combined with 0/339 in the full run, the
best-supported reading is that the policy has no reproducible failure inside the
near-nominal box at this budget, not that the search misses one.

### 9.3 Regression detection: boundary battery

A third battery (`results/full/regression/*.boundary.json`): 24 conditions where go1-demo
fell on 10–60% of its 20 validation seeds (selected from baseline data only), 8 paired
seeds each.

| pair | fixed battery (gate) | validated-failure battery | **boundary battery** (falls of 192) |
|---|---|---|---|
| identical | pass, 15 → 15 | no regression, 174 → 174 | **no regression, 80 → 80** |
| synthetic σ 0.1 | pass, 15 → 17 | no regression, 174 → 181 | **regression, 80 → 97 (29 worse / 12 better, p = 0.006)** |
| synthetic σ 0.3 | regression, 15 → 42 | regression, 174 → 185 | **regression, 80 → 159 (p < 1e-20)** |
| synthetic σ 0.5 | regression, 15 → 103 | missed, 174 → 182 | **regression, 80 → 174 (p < 1e-24)** |

The boundary battery had no false alarm, caught everything the fixed battery caught, and
also caught the smallest perturbation, which the fixed battery passed. This is the most
clearly positive result of the round. Same caveat as Section 6: the candidates are
synthetic, and one identical-policy null is a single check of the false-alarm rate.

### 9.4 Still not addressed

* More independent policies and a real regression checkpoint pair (need training on a GPU).
* Wall-clock speed: at an equal rollout budget the methods cost the same; the gains above
  are per rollout.

## 7. Limitations

* **One policy.** All conclusions are about one Go1 PPO policy (plus a gain-variant of the
  same weights). Other robots, policies and training methods are untested.
* **Simulation only.** No hardware. C MuJoCo replay reduces the chance that a failure is an
  MJX artifact; it says nothing about the sim-to-real gap.
* **The perturbation model is the repository's.** Plausible bands were set by hand from
  typical deployment figures, not measured on a Go1. "Conservative" means "inside those
  bands", not "observed in the field". Torque limits, terrain roughness and stairs are not
  modelled.
* **Distinctness is a modelling choice.** Results on diversity flip with the cluster
  radius (Section 5). No ground-truth catalogue of failure modes exists.
* **Validation selects a subset.** 11 candidates per run (first 3 + 8 leaders) are re-run;
  the est. validated distinct count extrapolates that rate to all clusters.
* **Replay samples are small** (20 per method per regime).
* **Wall-clock is from a laptop CPU** with one host stall; GPU timings would differ in
  absolute terms but the budgets are identical across methods.
* **The deck's earlier numbers** ("92%" and "2×") came from a different, smaller analysis
  (plan §1). This benchmark does not reproduce a 2× advantage over uniform random in
  failure rate; it measures 1.07–1.24× in the conservative box and about 1.0× in the broad
  box.

## 8. Reproduction

```bash
git clone https://github.com/InokiLabs/breakpoint.git && cd breakpoint
pip install -e ".[sim,dev,bench]"
python -m pytest benchmark/tests -q                                    # 33 benchmark unit tests (+ tests/test_robust_search.py)
python -m benchmark.run_benchmark --config benchmark/configs/smoke.yaml  # plumbing, minutes
python -m benchmark.run_benchmark --config benchmark/configs/pilot.yaml
python -m benchmark.run_benchmark --config benchmark/configs/full.yaml   # ~3.5 h on an M3 CPU, plus the gate
python -m benchmark.analysis.supplementary --results benchmark/results/full
# improvement round (BENCHMARK_PLAN.md §12)
python -m benchmark.run_benchmark --config benchmark/configs/tune.yaml
python -m benchmark.analysis.select_variant --results benchmark/results/tune --config benchmark/configs/tune.yaml
python -m benchmark.run_benchmark --config benchmark/configs/full.yaml --stage regression   # boundary battery
# confirmation: link results/full/{runs,replay,configs_status}/*.json into results/confirm/, then
python -m benchmark.run_benchmark --config benchmark/configs/confirm.yaml
```

Runs resume where they stop and never overwrite existing artifacts; use `--out <dir>` for a
fresh directory. Every seed is derived deterministically from the run id
(`policy__regime__method__repK`); validation seeds start at 9,000,000 and the adversarial
battery at 5,000,000, disjoint from search seeds.
