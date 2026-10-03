# Operational risk per family (Cause-Stratified Operational Risk)

`breakpoint_mjx/famrisk.py`. **Off by default; `BP_FAMRISK=1` enables it.**

Why it is opt-in: Bar 3 below failed in the unsafe direction. Some cause regions that lie
inside the declared operating domain are labelled `outside_odd`, and a buyer reads that
label as "not my deployment risk". Until the label errs only toward "in domain", the stage
stays off unless you ask for it, and its out-of-domain verdicts should be treated as
unverified.

## The question

BreakPoint's search box is deliberately wide. On the bundled Go1, 85-93% of uniformly drawn
worlds fail. That makes every family look like an artifact of the chosen bounds. A buyer
needs a different answer: **how often does this family's failure happen inside the
conditions I actually deploy in?** It also helps to split out families that need
conditions outside that domain.

`riskcert.py` (`BP_RISK=1`) gives one number for the whole policy. This stage gives one
number per family, and it runs by default.

## Method

Certified Minimal Cause (`mincause.py`) returns, for each family, a 1-minimal AND-region, for
example `obs_dropout_prob >= 0.242 AND obs_bias_scale >= 0.0245`. Every cause dim in the
region is moved away from nominal, and CMC certifies that the region's corner fails across
seeds. This stage treats each such region `R_f` as a stratum of the declared operating
distribution `p` (`riskcert.OperationalDist`, a product of independent per-dim uniform or
truncated-normal marginals).

1. **Exact stratum mass.** `m_f = P_p(theta in R_f)` is a product of 1-D tail
   probabilities. A uniform marginal gives a length ratio. A truncated normal gives a ratio
   of `erfc` terms, which stays accurate in far tails. The mass has no sampling error.
   `latency_steps`, which the simulator rounds, uses the effective integer edge:
   `round(x) >= k` is `x >= k - 0.5`.
2. **Fresh conditional rollouts.** Draw `n_f` worlds from `p` conditioned on `R_f`: each
   cause dim by exact inverse CDF inside its region, every other dim from `p`. Roll them out
   with the pipeline's own `fail_fn` (the same batched MJX evaluator CMC uses, on seeds far
   from the CMC seed range). The result is a Clopper-Pearson interval `[a_f, b_f]` on the
   conditional failure rate, at level `delta / F`, Bonferroni across the `F` families that
   spend rollouts (`delta = 0.05`).
3. **Per-family operational risk** is `m_f * [a_f, b_f]`. The region is fixed before these
   rollouts are drawn and the search records are never reused, so the interval is valid
   even though the region was chosen from the data.
4. **Class.**
   - `outside_odd`: `m_f == 0`, or `m_f * b_f < BP_FAMRISK_OUT_EPS` (1e-6).
   - `in_domain_risk`: `m_f * a_f > BP_FAMRISK_MIN` (1e-3).
   - `in_domain_low`: everything else.
5. **Residual.** `sum_f m_f * b_f` upper-bounds `P_p(fail AND theta in the union of the
   named regions)`. When `BP_RISK=1` ran under the same `p`, `total_lo - sum_f m_f*b_f` is a
   lower bound (confidence >= 1 - 2 delta) on the operational failure mass that no named
   family explains. If the two stages used different domains they are not subtracted, and
   the report says so.

**Boundary relaxation (added after the first measurement; see "Honest record").** CMC's
shrunk value is a certified *failing* point, and bisection only resolves the true boundary to
`span / 2**shrink_steps` (6 steps at standard/full, 3 at smoke). When a certified region has
zero mass under `p`, the stage moves each threshold toward nominal by that resolution. If
the relaxed region has positive mass, the rollouts are spent on it and the family records
`region_basis: "relaxed"`. The interval is then for `P_p(fail AND theta in relaxed region)`,
which is still valid for the same reason as step 3. The report says when this happened.
`BP_FAMRISK_RELAX=0` turns it off.

### Where the operating domain comes from

In this order:

1. The `operational:` block of the YAML at `BP_RISK_DIST` (`--risk-dist`).
2. The `operational:` block of the contract YAML at `BP_OBS_SPEC`.
3. Each dim's `plausible` band in `theta_space.yaml`.

The schema is `riskcert.OperationalDist.from_space`:
`{dim: {range: [lo, hi]}}` or `{dim: {mean: m, sd: s}}`. A contract can carry this block
today, but the byo build does **not** yet write it from training DR ranges automatically.

### Budget

Rollouts per certified family: smoke 32, standard 256, full 1024. `BP_FAMRISK_N`
overrides. A zero-mass family spends none.

### Where it shows up

- `families.json`: `families[i].odd_risk` for every CMC-certified family (null otherwise),
  plus a top-level `family_operational_risk` summary (the ODD spec and its source, counts
  per class, union upper bound, rollouts, and the residual when `BP_RISK=1`).
- `report.html`:
  - Every certified family gets an `Operational risk (declared ODD)` row with the mass,
    the conditional k/n with its CP interval, and the risk interval.
  - A section "Operational risk by family (declared operating domain)" lists in-domain
    families ordered by upper risk bound, and folds out-of-ODD families into a collapsed
    "Outside your operating domain" block.
  - If no family certified at the budget, the section says so.
  - Everything is labelled SIMULATION.

## Pre-registered SYNTHETIC benchmark

`scripts/bench_famrisk.py`. Pure numpy, no physics, no policy.

**Setup**

- Planted AND-regions on the real 16-dim Go1 ThetaSpace, with `p` uniform on the plausible
  bands.
- Exact masses 1e-2, 1e-3 and 1e-4, crossed with conditional failure rates 0.3, 0.7 and
  1.0.
- 500 reps per cell, N = 2000 rollouts per family per method.
- The CMC-recovered regions come from the real `certified_minimal_cause`. It runs on the
  same failure process plus p = 0.1 label flips.
- Truth for a recovered region `R_hat` is `r * P_p(R_hat intersect R)`, computed exactly.

**Baselines at equal rollouts**

- (a) Plain MC from `p`, counting failures inside the region, with Clopper-Pearson.
- (b) `riskcert.operational_risk` on the indicator `fail AND in region`. This uses the
  defensive-mixture IS with the betting CS. No surrogate is given, so this is the IS
  variant, not DR.
- (c) This stage's estimator.

    PYTHONPATH=. python scripts/bench_famrisk.py --reps 500 --n 2000          # docs/famrisk_results.json
    PYTHONPATH=. python scripts/bench_famrisk.py --reps 500 --n 2000 --no-relax

All numbers below are **SYNTHETIC**. Wall time is about 16 min on a Windows CPU laptop.

### Bar 1, coverage (>= 0.93 at every mass): PASS

Minimum empirical coverage of the 95% per-family interval across the rate cells:

| mass | given region | CMC-recovered region |
|---|---|---|
| 1e-2 | 0.948 | 0.976 |
| 1e-3 | 0.952 | 0.938 |
| 1e-4 | 0.948 | 0.946 |

CMC-recovered coverage comes from only the r = 0.7 and r = 1.0 cells. At r = 0.3 CMC
never certifies, 0/500 in every mass cell, because its bar is a failure fraction >= 0.5.
**Families whose conditional failure rate is below 0.5 get no per-family risk at all.**
That is a CMC limit, and it is inherited here.

### Bar 2, tightness (mass <= 1e-3: stratified width at least 10x smaller than plain MC): PASS

The smallest ratio of median 95% interval width (plain MC / stratified) over every cell with
mass <= 1e-3, given and CMC-recovered, is **45.4**. Most cells are far larger; for example,
at mass 1e-4 and r = 1 the stratified width is 1.8e-7 against plain MC's 1.8e-3.

The riskcert IS + betting baseline was wider than plain MC in every cell. Its median width
was 0.012 to 0.025. This matches the floor documented in `docs/riskcert.md`.

### Bar 3, out-of-ODD labelling: FAIL (reported as measured)

**Design.** Each rep plants six regions on disjoint dims: each mass twice, with rates
cycling 0.3 / 0.7 / 1.0. The declared ODD (passed as a real `BP_RISK_DIST` YAML) clips one
cause dim of one region per mass to halfway between nominal and its threshold. Families are
built from the CMC-recovered regions and run through the real `famrisk.run_stage`.

| | with boundary relaxation (default) | certified region only (`--no-relax`) |
|---|---|---|
| out-of-ODD regions labelled out, among CMC-certified | 684/684 = **1.000** | 684/684 = 1.000 |
| out-of-ODD, end to end (uncertified counts as a miss) | 684/1500 = 0.456 | 684/1500 = 0.456 |
| in-ODD regions mislabelled out (bar: 0) | **173** of 704 certified | 571 of 704 certified |

The accuracy half of the bar passes among certified families. The "0 in-ODD mislabelled"
half fails.

**Where the misses come from** (breakdown of the default run by planted region):

- 169 of the 173 are the in-ODD region with mass 1e-3 and conditional rate 0.7. That region
  reaches only 0.1% (one dim) or 3.2% (two dims) of the band width into the ODD.
- For that region, the effective failure fraction inside the region is 0.66 after label
  flips, only just above CMC's 0.5 bar. CMC then often certifies a non-minimal or overshot
  region: exact dims in 77 of 203 certified cases, and the shrunk threshold lands beyond
  the ODD edge by more than the bisection resolution.
- The mass-1e-4, rate-1.0 in-ODD region is mislabelled in 3 of 500.

So the classifier is only as good as the certified region's boundary. When CMC's threshold
overshoots by more than `span / 2**steps`, the stage calls the family out of domain. That
claim is correct for the certified region and wrong for the underlying failure mode.

**Untested next steps:**

- Certify the bisection's last *passing* point as well, so the boundary is bracketed.
- Or spend a few conditional rollouts on the slab between nominal and the certified
  threshold before issuing an `outside_odd` verdict.

### Honest record

- The first quick run (5 reps) showed in-ODD regions labelled out. Part of that was a bench
  bug: the multi-region representative was not kept out of the other planted regions, so
  CMC certified a different region's cause. The rest is the boundary effect above.
- The relaxation was added *after* seeing that result. The `--no-relax` column is the
  pre-registered method.
- An earlier run showed given-region coverage of 0.6 at r = 1. That was float roundoff:
  the planted mass 0.01 against the exact product 0.00999... The truth is now computed
  from the exact float mass.
- The oracle's uniform draws are salted per rep and per cell, so cells are independent.

## Real path: Go1 CPU smoke (not a benchmark)

This is a rerun of `python -m breakpoint_mjx.pipeline` at `BP_BUDGET=smoke` on the bundled
Go1 policy, **reusing a copied `failures.json`** from an earlier `breakpoint demo` smoke run
(192 search-evaluated worlds). Settings: `BP_RISK=1 BP_RISK_N=512 BP_RISK_TOL=0
BP_EMBED_VIDEO=0 JAX_PLATFORMS=cpu`, on a Windows CPU laptop.

Per-family output, pasted from the run log (**CPU smoke, SIMULATION**):

    CMC fam-1: minimal cause not certified: representative failed 6/8 seeds (CP lower 0.46 < 0.5)
    CMC fam-0: fails when obs_dropout_prob >= 0.242 AND obs_bias_scale >= 0.0245; final re-check
      failed only 4/8 seeds, NOT re-certified (flaky); necessary: undecided; distance from nominal
      0.48 of span (magnitudes not shrunk)
    CMC fam-3, fam-2: skipped at smoke budget (not in the top 2)
      ODD risk fam-0 [in_domain_risk]: obs_dropout_prob >= 0.1798 AND obs_bias_scale >= 0.012
      (certified region has zero mass in the ODD; measured on the region relaxed by the CMC
      bisection resolution); mass 0.0708; 24/32 fail (CP 0.566-0.885); risk 0.0401-0.0627
    per-family operational risk (ODD: plausible bands (theta_space.yaml)): 1 in-domain risk,
      0 in-domain low, 0 outside the operating domain; union upper bound 0.0627; 32 fresh rollouts
    operational failure probability (declared p, SIMULATION): 0.4434 (95% anytime CI
      0.2726-0.5627) from 512 fresh rollouts

`families.json` carries `odd_risk` on the one CMC-certified family and `null` on the other
three. `family_operational_risk.residual` reads: union upper bound 0.063 against the
whole-policy BP_RISK total 0.27-0.56 under the same plausible-band domain, so **at least 0.21
of the operational failure mass lies outside every named family region** at this budget. At
smoke, only the top two families get CMC and only one certified, so the named families
explain little of the total. `report.html` renders the section and the per-family row.

How to read it:

- CMC's certified thresholds (0.242 on `obs_dropout_prob`) sit outside the plausible band
  (upper edge 0.20), so the certified region has zero mass under this domain.
- The magnitudes were not shrunk at smoke (3 bisection steps), and the true boundary can be
  anywhere toward nominal. So the stage measured the relaxed region (thresholds moved
  toward nominal by span/8) instead of calling the family out of domain. The report says so.
- The family's final CMC re-check was flaky (4/8), and the independent C-MuJoCo replay was
  inconclusive (5/7). The operational-risk interval is valid for the relaxed region as
  measured, but it does not strengthen the case that the cause itself is confirmed.
- Wall time for the stage: 32 extra rollouts, a few seconds on CPU.

Reproduce (Windows, CPU):

    mkdir -p smoke/go1 && cp <earlier demo>/go1/{failures.json,go1_policy.pkl} smoke/go1/
    BP_OUT=smoke BP_BUDGET=smoke BP_RESTARTS=1 BP_EMBED_VIDEO=0 BP_RISK=1 BP_RISK_N=512 \
      BP_RISK_TOL=0 JAX_PLATFORMS=cpu PYTHONPATH=. python -m breakpoint_mjx.pipeline

The first attempt at this rerun died with `mujoco.FatalError: Could not allocate memory` while
rendering videos. That was after the per-family stage had already printed the same numbers
as above, and it happened because two benchmark processes were running at the same time on a
low-RAM laptop. The rerun above ran alone and completed.

