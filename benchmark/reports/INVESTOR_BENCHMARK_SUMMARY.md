# Inoki Labs / BreakPoint: benchmark summary

*Simulation only (MuJoCo MJX, replay in C MuJoCo). No physical robot. Run on 19 Oct 2026, one
Apple M3 CPU, 120 search runs (120,000 rollouts) plus 24,360 validation rollouts. Full
method, tables and limitations: `TECHNICAL_REPORT.md`.*

**1. Does directed search find distinct validated failures faster than random testing?**
In a realistic perturbation box, yes, per rollout: it finds more failures, and they
validate more often. It is **not** faster in wall-clock (same budget, about 50 s per 1,000
rollouts for every method), and it does **not** find a wider *variety* of failures than
random sampling. At a coarser definition of "distinct" it finds fewer (0.77× uniform).

**2. By what factor?** Directed vs uniform random, realistic (conservative) box, 5 repetitions, 95% CIs:

| | independent policy (go1-demo) | same weights at Kp 25 |
|---|---|---|
| failure rate | 1.24× [1.22, 1.28] | 1.07× [1.05, 1.08] |
| estimated validated distinct failures (predefined metric) | 1.48× [1.27, 1.72] | 1.09× [1.06, 1.15] |
| distinct failures at coarser grouping (radius 0.3) | **0.77× [0.73, 0.82]** | **0.65× [0.62, 0.69]** |

There is no advantage in the stress-test box (97% of random samples already fail) and
none near nominal (no method found a reproducible failure). Against the RoboGate-style LHS
+ boundary design, directed is better on every metric (2.21× [1.87, 2.83] validated distinct
failures). This does not reproduce the deck's earlier "2×" claim against random testing.

*Improvement round (pre-registered, tuned on separate seeds):* enabling novelty sharing in
the directed search raises coarse-distinct validated failures by 1.22–1.25× over the original
directed search in every cell, with no loss of yield. It reaches 1.14× uniform in the
realistic box (95% CI 0.97–1.34, so not a significant win) and still trails uniform in the
stress-test box (0.78×).

**3. Across how many configurations?** One independent policy (Unitree Go1 PPO) and one
deployment variant of the same weights. That is fewer than the three policies the plan
called for; no other walking checkpoint exists in the repository.

**4. How reproducible are the failures?** Re-run on 20 fresh seeds, 107/110 of directed's
selected failures in the realistic box validated (fall probability ≥ 0.5 with 95%
confidence). For the baselines: uniform 99/109, Sobol 93/110, LHS + boundary 88/110.

**5. Are they inside realistic ranges?** The conservative box is the repository's own
hand-set "plausible" bands (e.g. 0–40 ms latency, friction 0.4–1.0, up to 1 m/s push). They
are not measured on hardware. The dominant failure causes there are high control latency
and sensor dropout.

**6. Does cross-engine replay support it?** Mostly: 36/40 of directed's replayed failures
also fail in C MuJoCo with full-body collision (baselines 26–31/40). The samples are small.
A second simulator is not physical validation.

**7. Can the system distinguish regressions?** On synthetic degradations, yes: no false
alarm on an identical policy, and it flagged weight-noise candidates that raised falls from
15/120 to 42/120 and 103/120. A new "boundary" battery, built from conditions where the policy fails only
sometimes, also caught the smallest degradation (falls 80 → 97 of 192, p = 0.006), which the
standard battery passed, with no false alarm. **A real before/after checkpoint pair was not
available, so this is untested on real regressions.**

**8. What is not validated on hardware?** Everything. No failure here has been reproduced
on a physical Go1, and the perturbation ranges are not field-measured.

## Headlines the data support

1. "In a realistic perturbation range, BreakPoint's directed search found 1.5× more
   validated failures than random testing at the same simulation budget (simulation, one
   Go1 policy, 95% CI 1.3–1.7×)."
2. "97% of the failures BreakPoint selected reproduced on 20 fresh random seeds, versus
   91% for random testing (simulation)."
3. "BreakPoint's boundary regression battery detected every synthetic policy degradation
   tested, including one the standard battery missed, with no false alarm on an unchanged
   policy (simulation; real checkpoint pairs not yet tested)."

Do not claim: broader failure-mode coverage than random testing, faster wall-clock,
results across multiple robots or policies, or hardware validation.
