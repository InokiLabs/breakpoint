# Campaign coverage: uniform residual audit (BP_COVERAGE=1)

Product question: a validation lead running BreakPoint asks "are we done testing, and
what did we miss?" Until now the report's only statistic was a Wilson CI over
search-selected envs, which answers neither. This feature adds a uniform residual audit and one
release-review sentence with its assumptions stated. It was also built to provide a
stop condition. That part failed calibration (see below) and is off by default.

It is a product feature, not a research contribution. Every estimator here is standard.
The closest precedent is the multistart stopping-rule literature: Boender and Rinnooy Kan,
"Bayesian stopping rules for multistart global optimization methods" (Math. Programming
37, 1987) and its follow-ups on optimal and sequential stopping rules. They run
independent local searches, treat the distinct optima as a multinomial sample over
basins, and stop when few unfound optima are expected. What we do is transplant that
setup to adversarial DR failure search. Capture-recapture with independent units is also
standard in software inspection (Eick et al. 1992; Briand et al. IEEE TSE 2000) and
Chao2 on incidence data is already used for fuzzing saturation (Liyanage et al. ICSE
2023). The uniform audit is plain Monte Carlo failure-probability estimation with a
Clopper-Pearson bound (covered by the Corso et al. JAIR 2021 taxonomy).

## Method (frozen in commit 239eba2 before the benchmark ran)

Code: `breakpoint_mjx/coverage.py`. Wiring: `BP_SEARCH=multi BP_COVERAGE=1` in
`pipeline.py`. Knobs: `BP_COVERAGE_WAVES` (default 4), `BP_COVERAGE_RULE` (`none` |
`chao2` | `brk` | `chao1`, default `none`), `BP_COVERAGE_AUDIT` (default 2048).

1. **Uniform residual audit (the lead number).** Draw n scrambled-Sobol worlds from the
   declared DR box and evaluate them once. A failing audit world counts as covered if the
   straight segment from it to the nearest incumbent of some discovered mode stays
   failing. Otherwise it is unseen. The report states `k_unseen/n` and its one-sided 95%
   Clopper-Pearson upper bound. This is the only part that carries no adaptive-sampling
   bias and does not depend on a mode-count estimator.
2. **Modes are connected failing regions between restart incumbents.** Each independent
   CMA-ES restart (restart 0 excluded, since it starts at the LLM seed or box centre)
   contributes one point: its best-scoring sample, if that sample fails. Two incumbents
   are the same mode if the segment between them stays failing at 8 interior probes.
   Modes are the connected components of that graph. There is no radius and nothing is
   tied to a known K. The rejected alternative was leader clustering of every failing
   sample at r = 0.30. On `make_landscape(K=6)` it gave 1118 "modes" for 5 real ones,
   because in 16-D the failing set is a shell wider than r.
3. **Stop rules on those basins.** Each wave calls `multi_cmaes_search` once with 16
   fresh restarts (`multi_search.py` itself is unchanged). After each wave we compute:
   - Chao2 (bias-corrected when Q2 = 0), with a log-normal 95% CI, plus the Chao and
     Jost (2012) new-mode probability.
   - Bias-corrected Chao1 on basin hit counts.
   - Boender-Rinnooy Kan: `w_est = w(N-1)/(N-w-2)`, using the closed form as it is usually
     quoted. I did not fetch the paper's full text.

   The opt-in Chao2 rule stops when `Chao2 - S_obs < 0.5` and the new-mode probability is
   below 0.05. The BRK rule stops when `w_est - w < 0.5`.

Output: `families.json["coverage"]` and `out/<target>/coverage.json`, plus one line in
report.html. That line is written outside the learned layer's swallowed try/except.

## Measurement (ALL SYNTHETIC: planted landscapes on CPU, no physics, no robot)

`scripts/bench_coverage.py`. The real `multi_cmaes_search` and the real scorer run over
planted landscapes with known answers:
- homog-K: `make_landscape`, K in {3, 6, 10}.
- hetero-K: one flat slab holding 10% of the box, plus K-1 small tall modes.

Protocol: 10 seeds per group, 4 waves of 16 restarts x popsize 128 x 15 generations, a
2048-world Sobol audit, and 200,000 uniform draws as truth. True modes are connected
components of the planted failing set. Bars were frozen in commit 239eba2 before the
first full run. The fresh-seed confirmation was frozen in commit 2b4ee4c before it ran.
Raw per-run JSON and the printed summaries are in `docs/data/`.

```
PYTHONPATH=. <venv>/python scripts/bench_coverage.py --family homog --K 3 6 --out out/cov_homog36.json
PYTHONPATH=. <venv>/python scripts/bench_coverage.py --family homog --K 10 --out out/cov_homog10.json
PYTHONPATH=. <venv>/python scripts/bench_coverage.py --family hetero --out out/cov_hetero.json
PYTHONPATH=. <venv>/python scripts/bench_coverage.py --summarize out/cov_*.json [--rule brk]
# confirmation: the same three runs with --seed-start 10, then --rule brk --summarize
```

Each run takes about 2 minutes on this CPU, so one 50-run set is about 45 minutes with
three processes in parallel.

### Results

| SYNTHETIC | seeds 0-9 (exploratory) | seeds 10-19 (confirmation) |
|---|---|---|
| B1: audit 95% CP upper >= true unseen mass | 50/50 runs, 200/200 prefixes: PASS | 50/50, 200/200: PASS |
| B2': audit power with the slab counterfactually dropped | 20/20 PASS. Mean k_unseen/n 0.1000 vs truth 0.0999, mean CP upper 0.1116 | 20/20 PASS. 0.1001 vs 0.0997, CP upper 0.1117 |
| B3: false-complete, Chao2 rule | **28.0% FAIL** | 38.0% |
| B3: false-complete, BRK rule | 8.0% (would pass) | **16.0% FAIL** (frozen confirmation) |
| B3: false-complete, Chao1 rule | 50.0% | 56.0% |
| B4: declares on homog-3 (Chao2 / BRK) | 100% / 100% | 100% / 100% |
| BRK declares at all on homog-6 or homog-10 | 0% | 0% |
| B6: mean abs(Chao2 - K) vs mean abs(S_obs - K) | 0.43 vs 0.22: FAIL | 0.35 vs 0.14 |
| Chao2 95% CI contains K | 62% | 76% |
| Scorer vs analytic truth, audit points | 1.000 | 1.000 |

Modes found, fixed 16 restarts vs fixed 64 (mean true components):
- Seeds 0-9: hetero-7 3.7 vs 6.2, homog-10 7.1 vs 9.0.
- Seeds 10-19: hetero-7 2.9 vs 6.1, homog-10 6.7 vs 8.6.

The segment-connectivity mode definition held up on these landscapes:
- The S_obs to K_conn ratio was 0.95 to 1.00 in every group.
- Over-splitting averaged 0.1 modes at most.
- Incumbents falling outside the analytic failing set, summed over prefixes: 3 on seeds
  0-9 and 0 on seeds 10-19.

Compare the rejected fixed-radius rule, which gave 1118 modes for 5.

### What this means

- **No stop rule ships.** The Chao2 rule failed the pre-registered safety bar. BRK passed
  it on the exploratory seeds, but that choice was post hoc, and it failed the frozen
  confirmation on fresh seeds (16%). BRK is also very conservative: within 64 restarts it
  never declared completion on homog-6 or homog-10.
- **The default is therefore a fixed wave budget with no completeness claim.**
  `BP_COVERAGE_RULE=chao2|brk|chao1` opts in anyway, and the output labels the result
  uncalibrated.
- **Chao2 did not beat the raw count.** Its mean error was larger than that of S_obs in
  both sets. It stays in the JSON as an informational number only.
- **The audit is the part that held up, in both directions.**
  - Every false-complete under the Chao2 and BRK rules (Chao1 not checked) missed only
    tiny modes with at most 6.5e-5 true uniform mass. The audit bound (about 0.0015 at
    k_unseen = 0) covered that every time.
  - When a mode holding 10% of the mass was removed from the found set, the audit
    recovered it almost exactly.
  - So the sentence the product can defend is "failure probability in modes we never
    found <= x", not "we found all the modes".
- **The real search never missed the slab**, even at 16 restarts. The hetero landscape
  was meant to contain a large mode that search tends to miss, and on this landscape it
  does not. That is why audit power is shown through the disclosed counterfactual (B2').
  The pre-registered B2 had no qualifying run.

## Limits

- Everything above is SYNTHETIC. The Go1 Colab cell (`colab/build_multi_notebook.py`,
  cell 9) is UNVERIFIED and has never run on a GPU.
- The go1 box was 0.967 failing in the search-selected sample (`media/dl/families.json`).
  The Go1 audit will therefore likely report a large total failure mass and one big
  connected mode. That is a true statement about that box, not a certificate of rare
  corner cases.
- The audit bound concerns probability mass under the declared DR distribution. It says
  nothing about the worst undiscovered region. A rare but severe mode with mass below
  about 3/n stays invisible to it (rule of three, arithmetic).
- Segment connectivity can split a curved region into several modes. For the audit this
  errs toward reporting more unseen mass, never less. On a stochastic simulator the
  probes are noisy; a fixed key is used.
- Probe cost: a mean of about 13,800 extra evaluations per 4-wave run on these
  landscapes (measured). That is roughly 7 extra batches of 2048 on a GPU.
- The BRK formulas are the closed forms as they are usually quoted, implemented without
  the paper's full text.
- With one incidence per restart, Chao2 differs from Chao1 only by the (m-1)/m factor and
  by what m counts.
