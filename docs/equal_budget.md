# Equal-budget baseline

`breakpoint_mjx/equal_budget.py`, also `breakpoint baseline ...`.

## The question

Does BreakPoint's directed search find failures that standard robustness evaluation
would miss, when both spend the same number of rollouts, and when "better" is judged by
metrics the search does not optimize?

Two earlier pieces of evidence did not answer this:

- `scripts/benchmark_search.py` scores "plausible yield" with `transfer.transfer_likelihood`,
  the same heuristic CMA-ES maximizes (its fitness is `failure_score * transfer`). A
  directed win on that metric is partly circular.
- The pipeline's built-in random baseline (`pipeline.py`, `random_search` with
  `n_eval = popsize`) is one population of evals, far smaller than the directed budget.

## Arms

All arms share one evaluator (`evaluator.make_evaluator`), one theta space, one horizon
and one rollout budget B per arm per seed. They also share one batch shape
(`restarts * popsize`), so the jitted rollout compiles once.

| arm | what it samples | budget |
|---|---|---|
| `uniform` | B iid uniform draws over the theta box (what "evaluate on N random DR configs" means) | exactly B |
| `directed` | `multi_search.multi_cmaes_search`, the engine's multi-restart CMA-ES | `restarts * popsize * generations == B`, enforced by `plan_directed` (a non-divisible B is refused, never rounded) |
| `train_dr` | B draws from the declared operational distribution (`famrisk.resolve_odd`: `BP_RISK_DIST`, then the `BP_OBS_SPEC` contract's `operational:` block) | exactly B; skipped with a logged reason when nothing is declared, because the plausible-band fallback is not the policy's training DR |

The directed arm gets no LLM seed: restart 0 starts at the box centre, the rest are
scattered (`multi_search.restart_means`). This keeps the arm deterministic and offline.
The pipeline's default `BP_SEARCH` is single-start `cmaes`; this benchmark uses the
multi-restart search the engine recommends for filling a batch.

Uniform and train_dr batches that do not fill the last batch are padded with a repeat
row whose result is discarded; the padded rollouts are not counted toward B (they only
exist to avoid a recompile). With a B that is a multiple of the batch there is no padding.

A nominal check (one extra batch of the all-nominal world on distinct rollout keys,
outside every arm's budget) is reported alongside, because if the policy already fails
at nominal the distance metric is meaningless.

## Metrics, and why each is independent of the heuristic

None of the metrics calls `transfer_likelihood`; a unit test replaces it with a function
that raises and the metrics still compute.

| metric | definition | relation to what the search optimizes |
|---|---|---|
| failures | rollouts with `failure_score >= failure_threshold` | the search maximizes `score * transfer`, not the count of threshold crossings |
| closest L2 / closest Linf | min distance from nominal over failures in box-normalized units `(theta - lo) / (hi - lo)`; lower is better (the "most-likely failure" proxy used in falsification) | uses only the hard bounds and the nominal, never the plausible bands. **Caveat:** the transfer weighting rewards staying inside the plausible bands, which contain nominal, so the directed arm has an indirect pull toward nominal. The metric is not what is optimized, but it is not orthogonal to it either. |
| in-band failures | failures that pass `plausibility.impossible_reasons` plus its two count rules (`n_extreme <= MAX_EXTREME`, `n_outside <= max_outside_default`) | `plausibility.py` DOES call the heuristic: its `low_transfer` rule is `transfer_likelihood < MIN_TRANSFER`. That rule is dropped. The two count rules read the same per-dim plausible bands the heuristic averages, so this metric is a different functional of shared inputs, not a fully independent one. |
| modes, in-band modes | distinct direction-aware dominant-deviating dims among failures ("high latency_steps"), the transfer-free fingerprint from `scripts/benchmark_search.py::_dominant_mode` | independent of transfer. Caveat: it rewards spread, so uniform sampling gets modes cheaply when failures are dense. |
| replay confirmed (optional) | the k closest failures per arm/seed (default 4) re-run in C MuJoCo on the full-collision scene via `crossval_engines.replay_family` (the pipeline's replay stage); rate = reproduced / replayed with the CMC Clopper-Pearson bar | independent engine and contact model. Off by default; `BP_BASELINE_REPLAY=1` or `--replay` |

Across seeds: mean, min and max per metric, plus the per-seed values in `equal_budget.json`.

## Running it

```
python -m breakpoint_mjx.equal_budget --target go1 --budget 384 --seeds 2 \
    --preset smoke --replay --replay-k 4 --replay-seeds 4 --label "CPU smoke" --out out/eb
breakpoint baseline --target go1 --budget 5120 --seeds 3 --preset standard --out out/eb
```

`--preset` picks horizon and popsize from `config.BUDGET_PRESETS` and the restart count
from `BUDGET_RESTARTS` (smoke: 2 x 32, horizon 150; standard: 4 x 128, horizon 300). B
must be a multiple of `restarts * popsize`. `--policy`, `--obs-spec` and `--risk-dist`
behave as in `breakpoint run`; without `--policy`, go1 uses the bundled demo policy.
Outputs: `equal_budget.json` and `equal_budget.md`, both carrying the basis (budget,
horizon, seeds, "simulation only").

GPU: `colab/baseline_cell.py` (UNVERIFIED) runs go1 at the standard preset, B=5120,
3 seeds, with the replay on.

## CPU smoke result

Measured 2026-09-28 on this Windows laptop (JAX CPU backend), 525 s wall for everything.

**CPU smoke, horizon 150, B=384 per arm per seed, seeds 0 and 1, simulation only.**
Bundled Go1 demo policy, failure threshold 0.5, directed = 2 restarts x 32 popsize x 6
generations. Nominal check: 0/64 all-nominal rollouts fail. Replay: 4 closest failures
per arm/seed, 4 C-MuJoCo seeds each.

```
python -m breakpoint_mjx.equal_budget --target go1 --budget 384 --seeds 2 --preset smoke \
    --replay --replay-k 4 --replay-seeds 4 --label "CPU smoke" --out <dir>
```

| arm | failures | closest L2 (lower=better) | closest Linf (lower=better) | in-band failures | modes | in-band modes | replay confirmed |
|---|---|---|---|---|---|---|---|
| uniform | 354.5 [354, 355] | 1.058 [1.033, 1.083] | 0.472 [0.454, 0.490] | 153.5 [152, 155] | 10 | 10 | 0.12 [0.00, 0.25] |
| directed | 344.0 [342, 346] | 0.822 [0.737, 0.906] | 0.379 [0.346, 0.412] | 291.0 [272, 310] | 10 | 10 | 0.75 [0.50, 1.00] |
| train_dr | skipped (the demo policy declares no operational distribution) | | | | | | |

Cells: mean [min, max] over the two seeds.

What it says, plainly:

- **Raw failure count: no win for directed.** Uniform finds slightly MORE failures
  (354.5 vs 344). This policy fails in about 92% of uniform worlds at horizon 150, so the
  count is saturated and says little either way.
- **Modes: a tie at 10 for both arms,** also saturated. One of the 10 modes in every arm
  is "high push_step_frac", which only matters when push_mag is non-zero, so the
  fingerprint overcounts.
- **Closest failure to nominal: directed is closer on both seeds** (L2 0.74 / 0.91 vs
  1.03 / 1.08; the ranges do not overlap). Caveat from the metrics table: the transfer
  weighting pulls the search toward the plausible bands, which contain nominal.
- **In-band failures: directed has about 1.9x as many** (291 vs 153.5). Same caveat,
  stronger: the count rules read the bands the heuristic averages.
- **Replay: 6/8 directed vs 1/8 uniform closest failures reproduced** in C MuJoCo
  full-collision. With only 4 replay seeds the CP bar needs 4/4, so one miss makes a
  failure "inconclusive" (not confirmed); most uniform misses were inconclusive after 1-2
  seeds, not decided "not reproduced". Eight failures per arm is a small sample.

The directed advantage shows up exactly on the metrics nearest to what it optimizes
(closeness, in-band) and on the independent replay, and not at all on raw counts or
modes. Two seeds at a smoke budget are not enough to call it; the standard-budget GPU
run (`colab/baseline_cell.py`) is the measurement that would.

## Limits

- Simulation only. Nothing here measures real hardware.
- The CPU smoke is small (see its basis line). Two seeds give a range, not a confidence
  interval. The standard-budget GPU run has not been executed.
- Closest-to-nominal and in-band failures are not fully independent of the transfer
  heuristic (see the caveats above); failures, modes and the replay are.
- L-inf distance over 16 dims is weakly informative for uniform samples: magnitude dims
  whose nominal sits at a bound (noise, latency, push) put a uniform sample far from
  nominal on at least one axis almost always. `push_step_frac` counts toward distance
  even when `push_mag` is zero and the timing has no effect.
- The directed arm is one configuration of one search (multi-restart CMA-ES, no LLM seed).
  Other directed modes (active, llm, atlas) are not benchmarked here.
- The train_dr arm needs a declared distribution; the bundled demo policy declares none,
  so it is skipped in the smoke.
