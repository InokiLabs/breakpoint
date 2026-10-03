# Certified Minimal Cause (CMC)

Status: pre-registration written before the benchmark was run. Results are appended below
the pre-registration and are labeled SYNTHETIC wherever they come from analytic functions.

## What it is

BreakPoint used to name a failure family by the dims whose family mean deviated most from
the mean of all failures, with all 16 domain-randomization dims perturbed at once. That
produced names that contradicted the numbers printed next to them (an "underweight trunk"
family whose representative carries +1.11 kg). CMC names a family by the minimal set of
perturbations the simulator says are needed:

1. Candidate set C: dims where the representative theta* differs from nominal by more than
   1% of the dim's span.
2. Batched delta debugging (ddmin) toward nominal. Every round builds all candidate
   reductions (revert each dim of S; chunks and chunk complements) and scores them in one
   batched call, each row replicated over noise seeds. Certification is SEQUENTIAL (round
   2): seeds are added in rounds of K/2 (K=8 default), one batched call per round, and a
   configuration stops receiving seeds as soon as it crosses an exact binomial
   group-sequential boundary on either side of 0.5 (FAIL if k >= c_r, PASS if k <= d_r;
   for K=8 the looks are n = 4, 8, ..., 32 with c = 4, 7, 10, 13, 15, 18, 20, 24 and
   d = 0, 1, 2, 3, 5, 6, 8, 8). The boundaries are computed by dynamic programming at the
   least favorable rate q = 0.5 so the overall one-sided error over all looks is at most
   alpha = 0.1 on each side. This is a standard group-sequential test, not a new method.
   The loop stops when no reduction certifies, so S is 1-minimal.
3. Necessity: theta* with only S reverted to nominal must not certify as failing.
4. Magnitude shrink: each dim of S is bisected between nominal and theta*[s] (6 batched
   steps), giving a minimal counterexample theta_min and its normalized L-inf distance from
   nominal.
5. Label: conjunction over S, direction = sign(theta_min[s] - nominal[s]), asserted to agree
   with the displayed representative values.
6. Families with the same (S, directions) are merged.

Code: `breakpoint_mjx/mincause.py` (pure core with an injectable
`fail_fn(thetas, seeds) -> bool`, plus `make_evaluator_fail_fn`, the GPU adapter).

Prior work this builds on: delta debugging (Zeller and Hildebrandt 2002); ddmin for
cyber-physical systems with flaky executions (arXiv 2607.25695); counterfactual minimal,
necessary and sufficient explanations for CPS (DeCaF, arXiv 2604.07679). What is new here is
narrower: the ddmin rounds, the necessity check and the bisection are run as single
GPU-parallel batches over continuous DR parameters of a learned policy; per-reduction
Clopper-Pearson certification with pooled top-ups replaces repeated-run voting; and the
certified cause becomes the family identity (name, merge key, distance from nominal).

### Round 2: sequential certification (exact, no sampling)

The round-1 certifier applied the one-sided Clopper-Pearson test at full alpha after
every K-seed wave. Computed exactly (`python scripts/bench_mincause.py --oc`), that made
its real overall false-FAIL rate for a configuration failing exactly half the time
0.118, above the nominal 0.1. The round-2 boundaries hold it at 0.100 and spend fewer
seeds when the answer is clear. Per configuration, K=8, at most 32 seeds:

| true failure rate q | round-1 P(FAIL) | round-1 E[seeds] | round-2 P(FAIL) | round-2 E[seeds] |
|---|---|---|---|---|
| 1.0 | 1.000 | 8.00 | 1.000 | 4.00 |
| 0.9 | 1.000 | 9.62 | 0.999 | 6.50 |
| 0.8 | 0.989 | 13.68 | 0.939 | 11.66 |
| 0.7 | 0.829 | 19.99 | 0.649 | 19.11 |
| 0.5 | 0.118 | 28.82 | 0.100 | 27.22 |
| 0.3 | 0.002 | 19.99 | 0.009 | 19.11 |

The price of the honest alpha is power for configurations that fail only 70-80% of the
time (0.939 vs 0.989 at q = 0.8); see the p = 0.2 row of the benchmark below.

## Pre-registration (written before any benchmark number existed)

Benchmark: `scripts/bench_mincause.py`, pure numpy on CPU, over the real Go1 ThetaSpace
(16 dims, `breakpoint_mjx/configs/go1/theta_space.yaml`).

- 300 planted analytic failure functions. Each has a causal set of size 1 to 3 with AND
  structure: it fails iff every causal dim is past its threshold in its direction. Thresholds
  are drawn uniformly between 20% and 80% of the way from nominal to the bound.
- Seed flakiness: every evaluation outcome is flipped with probability p in {0, 0.1, 0.2}.
- For each function a representative theta* is sampled as a failing point with all 16 dims
  perturbed (every dim more than 1% of span away from nominal).
- Methods: CMC (K=8); ablation CMC with K=1 (plain ddmin, no certification); baseline A =
  current naming (`naming._ranked_dims`, family mean vs the mean of all failures in a run
  with 3 distractor families, given the true |S| as its depth, which is generous);
  baseline B = `margins.compute_attribution` finite-difference importance, top-|S| dims
  (also given the true |S|), scored on a continuous planted failure score.
- Metrics: exact causal-set recovery (predicted dim set equals the planted set), label
  direction consistency (every labeled direction agrees with the sign of theta* minus
  nominal), simulations used per family.

Bars (a missed bar is reported as a negative result and not claimed):

| bar | threshold |
|---|---|
| CMC exact recovery at p = 0 | >= 0.95 |
| CMC exact recovery at p = 0.1 | >= 0.85 |
| CMC label-direction consistency | 100% |
| CMC minus best baseline (A or B), exact recovery at p = 0.1 | >= 20 points |
| K=1 ablation | reported, no bar |

Real-path check: the pipeline completes with CMC on, and every family margin string lists a
minimal cause whose directions agree with the displayed values; median |S| reported.

One clarification was made after a 20-function smoke look and before the full run: both
baselines are given NOISELESS data at every p (A's search population is labeled with the
true failure set, B finite-differences the noiseless continuous score), in addition to
being told the true |S|. Flaky labels drowned baseline A's family mean in the smoke look,
which would have made the comparison look better for CMC than is fair. So the baseline
numbers below are upper bounds for those methods; CMC and its K=1 ablation see the flaky
outcomes.

## Results: SYNTHETIC analytic benchmark (planted failure sets, no physics, no policy)

Command (308 s measured on this Windows CPU in round 2, which also runs the round-1
certifier as `cmc_fixed` for comparison):

```bash
PYTHONPATH=. python scripts/bench_mincause.py        # writes docs/mincause_results.json
```

300 planted functions per p (the same 300 functions at every p; sizes 1/2/3 are
99/104/97 of them). "exact" = predicted dim set equals the planted set. "exact+dir" also
requires every direction to match the planted one. "dir-consist" = every labeled direction
agrees with the representative's displayed value. "sims/fam" = rollouts spent per family.

Round-1 numbers (fixed-K certifier; the round-2 re-run is in the next subsection):

| p | method | exact | exact+dir | dir-consist | sims/fam |
|---|---|---|---|---|---|
| 0.0 | CMC (K=8) | **1.000** | 1.000 | 1.000 | 841 |
| 0.0 | CMC K=1 (plain ddmin) | 1.000 | 1.000 | 1.000 | 105 |
| 0.0 | A: current naming (noiseless, true \|S\|) | 0.713 | 0.710 | 0.820 | 0 extra |
| 0.0 | B: finite-diff attribution (noiseless, true \|S\|) | 0.500 | 0.500 | 1.000 | 32 |
| 0.1 | CMC (K=8) | **1.000** | 1.000 | 1.000 | 1024 |
| 0.1 | CMC K=1 | 0.240 | 0.240 | 1.000 | 77 |
| 0.1 | A | 0.697 | 0.693 | 0.800 | 0 extra |
| 0.1 | B | 0.500 | 0.500 | 1.000 | 32 |
| 0.2 | CMC (K=8) | **0.980** | 0.980 | 1.000 | 1463 |
| 0.2 | CMC K=1 | 0.113 | 0.113 | 1.000 | 61 |
| 0.2 | A | 0.707 | 0.700 | 0.820 | 0 extra |
| 0.2 | B | 0.500 | 0.500 | 1.000 | 32 |

Pre-registered bars, all met on this synthetic benchmark:

| bar | measured | met |
|---|---|---|
| CMC exact at p = 0 >= 0.95 | 1.000 | yes |
| CMC exact at p = 0.1 >= 0.85 | 1.000 | yes |
| CMC direction consistency 100% | 100% at every p | yes |
| CMC minus best baseline at p = 0.1 >= 20 pts | 1.000 - 0.697 (A) = 30.3 pts | yes |

Other measured CMC numbers (K=8): necessity check returned "necessary" on 100% / 100% /
97.3% of functions at p = 0 / 0.1 / 0.2 (the planted sets are AND structures, so this is
the expected answer; `tests/test_mincause.py` covers the OR case where it must say "no");
mean magnitude-shrink error vs the planted threshold 0.004 / 0.004 / 0.009 of span; batched
`fail_fn` calls per family 16.7 / 28.2 / 43.5. At p = 0.2 there are 6 misses out of 300:
5 representatives did not certify as failing within 32 seeds (CMC then declines to rename
the family rather than guess), and 1 returned a 4-dim cause for a planted 3-dim set (one
spurious extra dim that never certified as removable).

What the ablation says: certification is what makes this work under flaky rollouts.
Plain ddmin (K=1) is perfect with deterministic outcomes but collapses to 0.24 at p = 0.1,
because a single spurious pass or fail sends the reduction down the wrong branch.
Certification costs roughly 8x to 24x the rollouts of plain ddmin.

What this does NOT show. The planted functions are axis-aligned AND-of-thresholds, which
is exactly the structure ddmin is built for, and the flakiness is symmetric outcome
flipping. Real failure sets can be non-monotone, have OR structure (then the cause is
1-minimal but not unique, and necessity comes back "no"), or depend on interactions no
single-dim reversion exposes. Baseline B's weakness here is structural: a finite
difference around the representative of an AND-of-thresholds moves only the binding dim.
None of this is a robot result.

### Round 2 re-run: sequential certifier vs the round-1 certifier (SYNTHETIC)

Same 300 functions, same command, 2026-09-27. `cmc` is the round-2 sequential certifier,
`cmc_fixed` is the round-1 certifier run in the same process on its own seed stream, and
"round 1" is the committed number from before this change. Medians and means are
rollouts per family.

| p | method | exact | median rollouts | mean rollouts | batched calls/family |
|---|---|---|---|---|---|
| 0.0 | round 1 (committed) | 1.000 | 652 | 841 | 16.7 |
| 0.0 | cmc_fixed (same run) | 1.000 | 652 | 841 | 16.7 |
| 0.0 | **cmc, sequential** | **1.000** | **326** | 420 | 16.7 |
| 0.1 | round 1 (committed) | 1.000 | 788 | 1024 | 28.2 |
| 0.1 | cmc_fixed (same run) | 1.000 | 804 | 1019 | 28.1 |
| 0.1 | **cmc, sequential** | **0.993** | **550** | 687 | 48.6 |
| 0.2 | round 1 (committed) | 0.980 | 1156 | 1463 | 43.5 |
| 0.2 | cmc_fixed (same run) | 0.973 | 1148 | 1486 | 44.0 |
| 0.2 | **cmc, sequential** | **0.850** | **938** | 1181 | 85.3 |

At p = 0.1: exact recovery 0.993 (bar 0.95) and median rollouts 550 vs 788 committed,
30.2% fewer (32% vs the same-run fixed certifier). The pre-registered bars above still
all hold for the sequential certifier (p = 0: 1.000; p = 0.1: 0.993; consistency 100%;
margin over the best baseline at p = 0.1: 0.993 - 0.697 = 29.6 pts).

The costs, stated plainly:

- p = 0.2 recovery drops from 0.97-0.98 to 0.850. At a 20% flip rate a configuration
  that truly fails fails 80% of the time, and the honest alpha buys less power there
  (P(FAIL) 0.939 vs 0.989 per configuration, table above), which compounds over a ddmin
  run. The round-1 certifier's extra power came partly from its uncontrolled error rate
  (0.118 at q = 0.5).
- Batched calls per family roughly double at p > 0 (48.6 vs 28.1 at p = 0.1) because
  seeds come in rounds of K/2. On a GPU each call has fixed latency, so fewer rollouts
  does not automatically mean less wall-clock there; on the CPU demo each call costs about
  3 s at the 64-row shape. This has not been measured on a GPU.

## Real demo run (round 2, CPU, smoke budget)

`python -m breakpoint_mjx.cli demo --out <dir>` on a Windows 11 Pro laptop (i7-1360P,
16 GB, jax CPU), 2026-09-27, one run: exit 0, 473 s total, CMC stage 117 s (83 s to
200 s), longest silence inside CMC 40 s (the first 64-row call, which includes a jit
compile). CMC certified 2 of 4 families by priority under the 600-rollout cap and used
364 rollouts:

- Priority-1 family (`low joint frictionloss scale`): representative failed 6/8 seeds, not
  certified at the smoke K; clustering name kept.
- Priority-2 family: 1-minimal set `obs_dropout_prob >= 0.242 AND obs_bias_scale >= 0.0245`,
  but the final re-check failed only 4/8 seeds, so the margin says `NOT re-certified
  (flaky)` and `necessary: undecided`. The round-1 demo cause was obs_dropout AND latency;
  this run found obs_dropout AND obs_bias instead. Both are observation-corruption
  pairs on a flaky representative; at K=4 / 8 seeds this is a preview, not a certified
  cause.

The 0xC0000409 native crash two outside graders hit after 24 min did not reproduce in
three runs on this machine with the new per-call cap (64 rows at smoke on CPU; 256 by
default on CPU). The cause was not isolated, so this is "not reproduced", not "fixed".

## Real-path check (CPU, toy budget, UNVERIFIED on GPU)

Command (about 5 minutes on this Windows CPU, 1.3 GB RAM):

```bash
PYTHONPATH=. python scripts/mincause_realpath.py --horizon 100 --n 64 --families 2 --k 4
```

This runs the real code path: Go1 MJX env, `make_evaluator`, `random_search`,
`build_families`, the pipeline's CMC stage (`run_stage` + `make_evaluator_fail_fn`, the GPU
adapter, here on CPU jax) and `write_report`. There is no trained Go1 checkpoint on this
machine, so the policy is a ZERO-ACTION policy (the robot holds its default pose under PD).
That makes the physics real and the policy meaningless: nothing below is a finding about a
locomotion policy. 55 of 64 random worlds failed at horizon 100.

| family | clustering name (before) | CMC name | CMC margin |
|---|---|---|---|
| fam-0 | rear CoM shift | payload overload + lateral CoM shift | fails when torso_mass_add_kg >= 4.52 AND com_offset_y_m <= -0.0465; certified 4/4 seeds (CP lower 0.56); necessary: no; distance from nominal 0.65 of span (magnitudes not shrunk) |
| fam-1 | forward CoM shift | high push mag + high floor slope rad | fails when push_mag >= 0.692 AND floor_slope_rad >= 0.119; certified 9/12 seeds (CP lower 0.52); necessary: yes; distance from nominal 0.35 of span (magnitudes not shrunk) |

- Pipeline completed with CMC on; both margins list a minimal cause; every clause direction
  agrees with the displayed representative value and with the counterexample value
  (checked by the script, `directions_agree: true`). Median |S| = 2 (2 families).
- Neither certified cause involves com_offset_x_m, the dim both old names were built from.
- 500 and 844 rollouts, 49 and 59 batched evaluate calls (many are pooled top-ups for
  configurations that were 2/4 or 3/4). At K=4 a certified failure needs 4/4 or a pooled
  equivalent, so these representatives sit close to their failure boundary.
- The magnitude shrink moved nothing on either family: every bisection midpoint failed to
  certify, so theta_min is the representative's own value on S. On this toy policy the
  failures are marginal at the representative (fam-1 reproduced 9 of 12 seeds). Whether
  the shrink earns its cost on a trained policy is an open question for the GPU run.
- fam-0 "necessary: no": with mass and lateral CoM reverted, some other combination of the
  representative's perturbations still fails, so this cause is sufficient and 1-minimal but
  not the only way the world breaks.
- A first version of the shrink combined all per-dim moves at once and fell back to the
  unshrunk point when the combination did not certify. It was replaced by a
  speculative-combine-with-rollback shrink (test
  `test_shrink_never_reports_an_uncertified_point_under_interaction`). The table above is from
  the rerun with the new shrink; the earlier run gave the same causes and names (fam-1 then
  certified 4/4 seeds) and also moved no magnitudes.

## GPU run (UNVERIFIED: never executed)

`colab/mincause_cell.py`, pasted after the bootstrap (cell 1 of
`colab/breakpoint_validation.ipynb`) on an A100. It runs a fresh go1 search with `BP_MINCAUSE=0`, then re-reads the same
`failures.json` with CMC on (K=8), and prints old names vs CMC names, margins, cause
sizes, necessity, min distance, whether directions agree with the displayed values, and
wall-clock.

```python
# paste colab/mincause_cell.py (kept in the repo so it stays in sync with the code)
exec(open("/content/breakpoint-mjx/colab/mincause_cell.py").read())
```

## Using it

- On by default in `python -m breakpoint_mjx.pipeline` / `scripts/run_pipeline.py`.
  `BP_MINCAUSE=0` turns it off. Its size follows `BP_BUDGET` (set by `--budget`):

  | budget | K | max seeds | families certified | hard rollout cap |
  |---|---|---|---|---|
  | `smoke` | 4 | 8 | top 2 by priority | 600 (3 shrink steps, one 64-row shape on CPU) |
  | `standard` | 8 | 24 | top 4 | none |
  | `full` (default) | 8 | 32 | all | none |

  Families outside the top N, or whose run hits the cap, keep their clustering names and
  their margin line says `CMC skipped at smoke budget (...)`. `BP_MINCAUSE_K`,
  `BP_MINCAUSE_MAXMULT`, `BP_MINCAUSE_TOP` and `BP_MINCAUSE_CAP` (`all` = no limit)
  override any preset field. `BP_MINCAUSE_MAXBATCH` caps rows per evaluate call (default
  256 on CPU, 64 on CPU at smoke, 1024 on GPU via `jax.default_backend()`); chunks are
  padded to at most three fixed shapes (max/4, max/2, max, none below 64), so the jit
  compiles at most three shapes (one at smoke).
- Progress: one flushed line per batched call, e.g. `CMC family 1/2 | ddmin round 3
  (|S|=5) | 48 rows | 312/600 rollouts | 71s elapsed | ETA ~66s`.
- Integer dims (`latency_steps`, rounded by the rollout) print as the effective integer
  threshold: `latency_steps >= 1`, not `>= 0.516`.
- Output: `FailureFamily.name`, `.margin` (rendered as the report's margin line),
  `.minimal_cause` ([{dim, value, direction}]), `.min_distance`, `.cause_certified`,
  `.cause_necessary`, `.merged_from`; the same fields are in `families.json`.
- With `BP_MARGINS=1` too, the old single-axis sweep is appended after the CMC margin.
- If CMC errors (or a label/value direction assertion fails) the stage logs it and the
  run keeps the clustering names; it never fails a run.
- Tests: `python -m pytest -q tests/test_mincause.py`.
