# Operational risk certificate (`riskcert.py`, `BP_RISK=1`, `breakpoint run --risk N`)

**Status: shipped as an engineering and product feature. The width claim is a
measured NEGATIVE at every CPU budget tested. Every number below is SYNTHETIC (an
analytic failure set on CPU) unless it says otherwise. There is no GPU or robot result
yet; the Colab cell for one is UNVERIFIED.**

## 1. What problem this fixes

Before this change the only probability BreakPoint reported was a Wilson CI over every
env the run evaluated (`pipeline.py`, `stats.compute_report_stats`). Those envs were
chosen by an adversarial search. So the `0.967` failure rate in `media/dl/families.json`
(go1, BP_SEARCH=cmaes, 12,800 evals; that file sits untracked in the main checkout, not
in git) is the failure rate **among worlds the search picked**. It is not
the probability that the policy fails in deployment, and it was never labelled that way.

Relabels that ship regardless of the estimator result:

| Where | Before | After |
|---|---|---|
| `pipeline.py` console line | `overall failure rate` | `failure rate among search-evaluated worlds (adversarially selected)` |
| `report.py` stats header | `Overall failure rate` | `Overall failure rate among search-evaluated worlds` + "NOT a deployment failure probability" |
| `webpush.build_families_json` | `stats.failure_rate` | same key (web compatibility) + `stats.failure_rate_basis: "search_selected"` |
| `surrogate.predict_proba` docstring | "Calibrated failure probability" | `clip(score/(2*threshold))`, "NOT calibrated" |
| `envelope.py` M3 | "machine-verified certificate" | the box check samples the SURROGATE; `verified_by: "surrogate"`; `certify_envelope` added for a simulator-backed bound |

## 2. What the new stage computes

Given an operator-declared deployment distribution `p(theta)` (default: uniform on each
`ThetaDim.plausible` band in `configs/<robot>/theta_space.yaml`, overridable with an
`operational:` YAML block via `--risk-dist`), the stage draws FRESH rollouts and returns
an `operational_risk` block in `families.json` and a section in `report.html`:

`estimate, lo, hi` (95%, anytime-valid), `n_rollouts, n_failures, stopped, alpha,
w_max_bound, w_max_observed, ess, surrogate_term, dr_floor_halfwidth,
plain_mc_baseline` (Clopper-Pearson on the defensive p-draws inside the same rollouts),
`proposal`, the declared `dist`, and the caveat that this is a SIMULATION probability
under the declared p, not a real-world failure rate.

It also certifies the M3 envelope box with 299 fresh simulator rollouts
(`envelope.simulator_certificate`): 0 failures in 299 gives a 95% upper bound of 0.00997,
the smallest budget that can certify 1%.

### The estimator, and whose it is

This is **not a new estimator**. It is doubly-robust importance sampling with a betting
confidence sequence:

* Karampatziakis, Mineiro and Ramdas, *Off-Policy Confidence Sequences*, ICML 2021.
* Waudby-Smith, Wu, Ramdas, Karampatziakis and Mineiro, *Anytime-valid off-policy
  inference for contextual bandits*, ACM/IMS Journal of Data Science.
* Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting*,
  JRSS-B 2023 (the hedged-capital CS used here).

Mapping: logging policy = proposal `q`, target policy = `p`, reward = `1{fail}`, reward
model = the surrogate `g_hat`. The surrogate term plus the importance-weighted residual
is also the prediction-powered-inference rectifier (Angelopoulos et al. 2023; SureSim,
arXiv 2510.04354, uses the same idea for sim-to-real). The proposal is a defensive
mixture (Owen and Zhou 2000). Seeding an IS proposal from failure-search output is
established practice (Bayesian Safety Validation, arXiv 2305.02449; Kochenderfer et al.,
*Algorithms for Validation*, adaptive-IS chapters). Finite-sample, anytime-valid
certification of an operational failure rate is also being published now: Ray and
Thakkar, arXiv 2609.20037, and arXiv 2604.03827 (IS confidence intervals for AV
evaluation). Only search snippets of those two were seen, not the papers.

The honest claim is product-level: BreakPoint would be the only tool in its segment
that ships an anytime-valid operational failure-probability bound next to its search
output. That is a product claim, not a research contribution.

### Mechanics (`breakpoint_mjx/riskcert.py`)

1. `OperationalDist`: independent per-dim uniform or truncated normal on a box.
2. `build_proposal`: `q = alpha*p + (1-alpha) * sum_k pi_k TN(mu_k, diag(s_k^2))`.
   Components are **diagonal** truncated normals, so the truncation normaliser is a
   product of 1-D normal-CDF differences and `q` is exact. That keeps
   `w = p/q <= 1/alpha` exact (unit-tested at d=6 and d=12; the benchmark logs the max
   observed weight, which equals the bound 3.333 and never exceeds it). A full covariance
   on a d=12 box has no closed-form normaliser, so it is not offered. Centers come from
   family `mean_theta` vectors and the top failing records. `pi_k` is the surrogate-
   predicted failure mass in each center's cell. With `fit="moments"` (default) each
   component is moment-matched to the surrogate-predicted failure mass in its cell
   (one cross-entropy-method step on the surrogate).
3. `estimate_from_samples`: `"dr"` uses `X = c_hat + w (f - g)` with range
   `[c - 1/alpha, c + 1/alpha]`; `c_hat = E_p[g]` from up to 10^6 cheap surrogate calls,
   with an empirical-Bernstein radius (Maurer and Pontil 2009) paid at delta/2 because
   "many draws" is not "exact" at P_fail ~ 1e-3. `"is"` uses `X = w f` in `[0, 1/alpha]`.
   `"is_wald"` / `"dr_wald"` are the same point estimates with asymptotic normal CIs,
   reported only to show what the finite-sample guarantee costs.
4. `betting_cs`: the WSR hedged-capital CS (predictable plug-in bets, running
   intersection), valid at any stopping time. `side="upper"` gives the one-sided bound
   for a `P(fail) <= eps` release gate. `clopper_pearson` is pure stdlib and matches
   scipy to 1e-6 in the tests.
5. `certify_box` / `envelope.certify_envelope`: Clopper-Pearson upper bound from fresh
   simulator rollouts inside a box.
6. `dr_floor_halfwidth` / `crossover_n`: the analytic floor (below).

## 3. The floor, computed BEFORE the benchmark

A distribution-free CI for a mean bounded in a range of width `2/alpha` cannot rule out
one unseen draw at the bound. So even with a PERFECT surrogate (every rectifier term
exactly 0) the betting CS half-width has a floor that depends only on `N` and `alpha`.
`dr_floor_halfwidth` computes it by running the CS on constant data. Plain-MC
Clopper-Pearson is evaluated at the expected count `k = round(N * P_fail)`.

Command: `PYTHONPATH=. python scripts/bench_riskcert.py` (first three printed lines).

| half-width | N=2,000 | N=20,000 | N=200,000 |
|---|---|---|---|
| DR floor, alpha=0.1 | 0.03679 | 0.00369 | 0.000369 |
| DR floor, alpha=0.3 | 0.01226 | 0.00123 | 0.000123 |
| DR floor, alpha=0.5 | 0.00736 | 0.00074 | 0.000074 |
| plain-MC CP, P=1e-2 | 0.00464 | 0.00140 | 0.000439 |
| plain-MC CP, P=1e-3 | 0.00174 | 0.00047 | 0.000141 |

The pre-registered 5x target at `P=1e-3, N=2000` needs a half-width of 0.00035. The
floor at alpha=0.3 is 0.01226, 35x too wide, before any variance. **The target is
unreachable by construction, as the review predicted.**

Optimistic crossover (smallest N where the perfect-surrogate floor drops under CP):

| | alpha=0.1 | alpha=0.3 | alpha=0.5 |
|---|---|---|---|
| P=1e-2 | ~158,000 | ~15,800 | ~6,300 |
| P=1e-3 | > 500,000 | ~158,000 | ~63,000 |

So width gains are only possible once `N * P_fail >> 1/alpha`, which is the GPU regime
(tens to hundreds of thousands of rollouts), not the CPU toy. And the floor is only the
first obstacle: the measured widths below are well above it.

## 4. Measurement (SYNTHETIC)

Everything in this section is **SYNTHETIC**: `theta` in `[0,1]^d`, `p` uniform on
`[0.1, 0.9]^d`, failure = inside one of 4 disjoint ellipsoids in the first 3 dims (the
other dims are inert), so `P_p(fail)` is known EXACTLY (1e-2 or 1e-3). The search is
the real `multi_cmaes_search` (16 restarts x 128, 15 generations, 30,720 evals) fed a
legged-contract synthetic evaluate so the real scorer runs; the surrogate is the real
`Ensemble` (3 members) on 4,000 of those records; the proposal is the real
`build_proposal` at alpha = 0.3. Every method gets the same number N of FRESH draws.
Reps: 200 at N=2,000, 50 at N=20,000, 6 at N=200,000 (so the N=200,000 rows are
indicative only). Median over reps.

Command: `PYTHONPATH=. python scripts/bench_riskcert.py` (3,755 s on this machine's
CPU, results in `docs/riskcert_results.json`).

Columns: `mc_bet` plain MC with the same betting CS; `is_bet` IS, no surrogate;
`dr_bet` the shipped default (DR, search centers, moment-matched components);
`dr_bet_wrong_g` a deliberately wrong surrogate (mirror image of the space);
`dr_bet_bad_centers` centers placed away from every failure region; `dr_bet_fixed_sd`
the first version of the proposal (fixed narrow components); `dr_bet_oracle_q` an
ORACLE proposal built from the hidden truth (a reference no real run can build);
`*_wald` the same estimators with an asymptotic normal CI (NOT finite-sample valid).

### Pre-registered primary outcome (b): FAIL

At `P_fail = 1e-3, N = 2,000` the ratio `CP half-width / DR-betting half-width` was
**0.23 (d=6) and 0.24 (d=12)**. That means the certified interval is about 4x WIDER than
plain Monte Carlo with Clopper-Pearson, not 5x narrower. The pass bar was 5 and the fail
bar was 2. This is the failure the analytic floor predicted.

### Width ratio, CP half-width / method half-width (above 1 means narrower than plain MC)

| d | P_fail | N | reps | CP half-width | mc_bet | is_bet | dr_bet | dr_bet_wrong_g | dr_bet_bad_centers | dr_bet_fixed_sd | dr_bet_oracle_q | is_wald | dr_wald | dr_wald_oracle_q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.01 | 2,000 | 200 | 0.004642 | 0.69 | 0.51 | 0.37 | 0.34 | 0.37 | 0.36 | 0.38 | 1.34 | 1.39 | 5.70 |
| 6 | 0.01 | 20,000 | 50 | 0.001396 | 0.56 | 0.72 | 0.57 | 0.28 | 0.54 | 0.46 | 0.83 | 1.27 | 1.30 | 5.44 |
| 6 | 0.01 | 200,000 | 6 | 0.000436 | 0.45 | 0.73 | 0.51 | 0.24 | 0.45 | 0.38 | 1.10 | 1.26 | 1.30 | 5.37 |
| 6 | 0.001 | 2,000 | 200 | 0.001743 | 0.69 | 0.27 | 0.23 | 0.22 | 0.23 | 0.23 | 0.22 | 7.35 | 12.44 | 18.36 |
| 6 | 0.001 | 20,000 | 50 | 0.000456 | 0.62 | 0.41 | 0.34 | 0.31 | 0.35 | 0.36 | 0.36 | 0.81 | 0.81 | 15.03 |
| 6 | 0.001 | 200,000 | 6 | 0.000141 | 0.57 | 0.45 | 0.35 | 0.27 | 0.31 | 0.34 | 0.71 | 0.78 | 0.79 | 14.69 |
| 12 | 0.01 | 2,000 | 200 | 0.004642 | 0.68 | 0.44 | 0.36 | 0.34 | 0.37 | 0.36 | 0.38 | 0.85 | 0.87 | 4.91 |
| 12 | 0.01 | 20,000 | 50 | 0.001405 | 0.56 | 0.47 | 0.40 | 0.30 | 0.42 | 0.36 | 0.86 | 0.78 | 0.80 | 4.63 |
| 12 | 0.01 | 200,000 | 6 | 0.000439 | 0.46 | 0.41 | 0.35 | 0.26 | 0.36 | 0.30 | 1.21 | 0.77 | 0.79 | 4.43 |
| 12 | 0.001 | 2,000 | 200 | 0.001743 | 0.69 | 0.28 | 0.24 | 0.24 | 0.24 | 0.24 | 0.22 | 13.16 | 14.63 | 15.92 |
| 12 | 0.001 | 20,000 | 50 | 0.000433 | 0.63 | 0.35 | 0.31 | 0.31 | 0.31 | 0.30 | 0.35 | 0.57 | 0.59 | 12.47 |
| 12 | 0.001 | 200,000 | 6 | 0.000142 | 0.56 | 0.33 | 0.29 | 0.28 | 0.28 | 0.27 | 0.85 | 0.58 | 0.59 | 12.95 |

The shipped estimator (`dr_bet`) never beat plain-MC Clopper-Pearson in any cell. Its
best ratio was 0.57. Even the ORACLE proposal with the betting CS only drew about level
with CP at N=200,000 and P=1e-2 (1.10 and 1.21, from 6 reps).

### Coverage (sanity check only; betting CSs are conservative by construction)

| d | P_fail | N | reps | mc_cp | mc_bet | is_bet | dr_bet | dr_bet_wrong_g | dr_bet_bad_centers | dr_bet_fixed_sd | dr_bet_oracle_q | is_wald | dr_wald | dr_wald_oracle_q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.01 | 2,000 | 200 | 0.94 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 | 0.77 | 0.93 |
| 6 | 0.01 | 20,000 | 50 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.96 | 0.90 |
| 6 | 0.01 | 200,000 | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 0.83 | 0.50 |
| 6 | 0.001 | 2,000 | 200 | 0.96 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 | 0.33 | 0.98 |
| 6 | 0.001 | 20,000 | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.92 | 0.94 |
| 6 | 0.001 | 200,000 | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 12 | 0.01 | 2,000 | 200 | 0.96 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 0.92 | 0.95 |
| 12 | 0.01 | 20,000 | 50 | 0.98 | 1.00 | 0.98 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.92 | 0.96 | 0.92 |
| 12 | 0.01 | 200,000 | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 |
| 12 | 0.001 | 2,000 | 200 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.49 | 0.48 | 0.94 |
| 12 | 0.001 | 20,000 | 50 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.94 | 0.92 |
| 12 | 0.001 | 200,000 | 6 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Every finite-sample method (`*_bet`, and CP) covered, including with the wrong
surrogate and the misplaced centers. That is the robustness claim, and it holds on this
landscape. The asymptotic Wald intervals are the warning. At N=2,000 and P=1e-3 they
look 7x to 15x narrower than CP and cover the truth only **33% to 49%** of the time. A
tool that reported that number would be wrong most of the time. That gap is what the
guarantee buys, and at CPU budgets it costs the whole width advantage.

### Why: proposal quality (variance factor V = E_p[w f] / mu; plain MC = 1, lower is better)

| d | P_fail | regions hit | E_p[g] | V search+moments | V fixed | V bad centers | V oracle |
|---|---|---|---|---|---|---|---|
| 6 | 0.01 | 3/4 | 0.00887 | 0.654 | 2.282 | 0.996 | 0.070 |
| 6 | 0.001 | 2/4 | 0.00049 | 1.736 | 2.303 | 1.981 | 0.007 |
| 12 | 0.01 | 3/4 | 0.00446 | 1.663 | 3.308 | 1.734 | 0.070 |
| 12 | 0.001 | 3/4 | 0.00006 | 3.309 | 3.308 | 2.996 | 0.007 |

The IS variance is `mu*V - mu^2`, so V > 1 means the proposal is worse than sampling
from p directly. Findings:

1. **The first proposal (`fixed`) was worse than plain MC everywhere (V = 2.3 to 3.3).**
   Search centers with narrow components cover the failure set only near where CMA-ES
   converged. The failure set is wide along the inert axes, so most failing p-mass sat
   at `w ~ 1/alpha`. This was found on the first landscape and caused the one design
   change (`fit="moments"`, a cross-entropy step on the surrogate). The change was made
   after seeing that result, and both versions are reported.
2. **Moment matching helped only where the surrogate was good.** At d=6, P=1e-2 it gave
   V = 0.65. At P=1e-3, and at d=12, the surrogate saw too few failures
   (`E_p[g]` = 0.00049 and 0.00006 against a truth of 0.001), and V stayed at 1.7 to 3.3.
3. **An oracle proposal gets V = 0.007 to 0.07**, and its asymptotic CI is 4x to 18x
   narrower than CP. So the IS machinery can help a lot. The binding constraints are
   (a) proposals built from search output were nowhere near the oracle, and (b) the
   anytime-valid betting CS on a range of width `2/alpha` costs about 4x to 80x in width
   over the asymptotic interval at these budgets (most at small N and P=1e-3).

### Sequential stopping (d): holds

d=6, P=1e-2, 500-rollout batches, stop at half-width 0.005, cap 20,000, 60 reps:
coverage 1.00, median 7,000 rollouts used. This shows only that stopping on the CI does
not break validity. It does not show efficiency: plain MC CP reaches +/-0.005 at P=1e-2
with about 1,800 draws.

### Weight bound

Max observed `w` across every draw in every cell: 3.3333 = 1/alpha, never above it.

## 5. Limits

* **Simulation probability only.** Under the declared `p`, on this simulator. Not a
  real-world failure rate; sim-to-real calibration needs paired hardware runs
  (SureSim-style PPI on top), which do not exist.
* **Validity assumes the fresh rollouts are independent draws.** Rollouts in one MJX
  batch share a batch key but use per-env `fold_in` keys; that is treated as
  independent. The search's own records are never reused.
* **`p` is operator-declared.** The default (uniform on each plausible band) is a
  convenience, not a measured deployment distribution. The number is only as meaningful
  as the declared `p`.
* **The DR range is symmetric.** Betting on `[c - 1/alpha, c + 1/alpha]` pays for both
  tails. On the benchmark, the no-surrogate IS bet on `[0, 1/alpha]` was often narrower
  than the DR bet at the same budget. Tighter variants were not tried here: a fixed-N
  (not anytime) betting or empirical-Bernstein CI, a one-sided DR form, and stratifying
  p so most weights sit well below `1/alpha`.
* **The proposal is the weak part.** Search-seeded centers with a surrogate fit did not
  get close to the oracle proposal (section 4), especially at P=1e-3, where the
  surrogate saw too few failures to fit components.
* **Head-to-head baselines are missing.** If a paper is ever wanted, the only
  defensible question is empirical: do multi-start CMA-ES modes give better proposals
  than cross-entropy IS, BSV-style GP acquisition, or KDE-AIS at equal simulator budget
  on real MJX locomotion? Nothing here answers that.

## 6. GPU follow-up (UNVERIFIED)

`colab/riskcert_cell.py`: go1 and barkour, 16x128 multi-start search, BP_RISK with 512-
rollout batches up to 8,192 fresh rollouts, stop at +/-0.005, plus a 65,536-draw
brute-force plain-MC reference (`BP_RISK_MC_N`). It
has never run. The CPU analysis predicts no width gain at 8,192 unless the true
P_fail is well above 1e-2, so a "no gain" result there is the expected outcome.

## 7. Reproducing

```bash
PYTHONPATH=. python scripts/bench_riskcert.py          # full SYNTHETIC run, ~63 min CPU here
PYTHONPATH=. python scripts/bench_riskcert.py --quick  # smoke, ~2 min
python -m pytest -q tests/test_riskcert.py
```

Results land in `docs/riskcert_results.json`.
