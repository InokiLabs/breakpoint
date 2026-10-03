# BreakPoint Crash Ratings (v1)

## What a rating is

A Crash Rating runs a policy through a **fixed, versioned battery** of named conditions and
reports, per condition, how often the robot fell. Every policy rated on the same battery
runs the same cells, the same seeds, the same horizon and the same fall rule, so two
ratings are directly comparable, the way two cars are comparable on one crash-test
protocol.

It is the opposite of `breakpoint run`. The search adapts to a policy to find where it
breaks; a rating never adapts. It only answers: on these named, plausible deployment
conditions, how often does this policy fall?

## The battery: `legged-quadruped-v1`

File: `breakpoint_mjx/batteries/legged-quadruped-v1.yaml` (each cell has a comment saying
why it exists). Robots: `go1`, `barkour`. 15 cells x seeds 0-7 x 500 control steps (10 s).

| cell | condition | axis |
|---|---|---|
| `nominal` | every dim at nominal | reference |
| `low-friction` | floor friction 0.45 (nominal 0.80) | contact |
| `latency-20ms` | 1 control step of actuation latency | timing |
| `latency-40ms` | 2 control steps of latency | timing |
| `obs-noise` | observation noise scale 0.03 | sensing |
| `obs-dropout` | 10% per-step observation dropout | sensing |
| `obs-bias` | constant sensor bias scale 0.03 | sensing |
| `lag-dropout` | 40 ms latency + 20% dropout | sensing + timing |
| `payload-2kg` | +2 kg on the trunk | morphology |
| `payload-offcenter` | +1.5 kg, CoM 3 cm forward | morphology |
| `weak-actuators` | PD gains at 85% | actuation |
| `push-light` | 0.5 m/s horizontal impulse at mid-episode | pushes |
| `push-firm` | 0.9 m/s horizontal impulse at mid-episode | pushes |
| `tilt-pos-5deg` | floor pitched +5 deg about world y | terrain |
| `tilt-neg-5deg` | floor pitched -5 deg about world y | terrain |

`legged-humanoid-v1` (robot `g1`) has the same cells with the payloads scaled to the
heavier robot (4 kg, and 3 kg off-centre) and a 0.40 m fall height. No humanoid rating has
been run yet; there is no bundled humanoid policy.

Rules the loader enforces (`breakpoint_mjx/ratings.py`):

* every cell dim must exist in the robot's theta space (`configs/<robot>/theta_space.yaml`),
  else the rating refuses to run and names the dim;
* every value must lie inside the dim's hard bounds **and** inside its `plausible` band
  (the same per-dim deployment band `plausibility.py` / `transfer.py` score against), so no
  cell is a stacked extreme;
* dims a cell does not list stay at nominal; latency must be a whole number of steps.

**Fall rule:** trunk height above the floor plane (floor-relative, so a tilted floor is not
misread as a fall) < 0.18 m, or the trunk up-axis z-projection < 0.5. Both numbers are
parsed from the battery's `fall_rule` sentence and used as the thresholds, so the battery,
not the robot target, defines a fall.

**Pass rule:** a cell passes if its fall rate is at most 1/8 across seeds (compared exactly
in integers). With 8 seeds that means 0 or 1 falls.

**Interval:** each cell reports an exact two-sided 95% Clopper-Pearson interval on its fall
rate, rounded outward to 4 decimals. With n = 8, zero falls still means the true fall rate
could be as high as 0.37: a rating separates clearly broken from not-obviously-broken, it
does not certify a low rate.

**Simulator artifacts:** a rollout the engine's physical-validity gate flags `sim_invalid`
(NaN state, torso through the floor while upright, deep contact penetration, acceleration
blow-up) is **counted as a fall**, and the per-cell `sim_invalid` count is reported next to
it. A rating can therefore never look better because the simulator misbehaved; if
`sim_invalid` is non-zero, read that cell with care.

**Determinism:** the rollout for cell i, seed s uses JAX key `fold_in(PRNGKey(s), i)`. One
evaluator batch per seed holds every cell, compiled once.

**Versioning:** the report carries the sha256 of the battery file (line endings normalized
to LF). Any change to a cell, the seeds, the horizon or a rule is a new battery version
(`-v2`), never an edit in place.

## The report

`breakpoint rate` writes `breakpoint.report/v1` JSON with `kind: "rating"` (the shared
report contract, `schema: breakpoint.report/v1`). Additions beyond the contract's example,
all additive: `run.budget` (`full` or `smoke`), `run.horizon`, and a per-cell `sim_invalid`
count. `attestation` is always null here; signing is a separate step. `policy.sha256` is the
hash of the checkpoint file; `policy.source` is a URL or `local`; no file paths, weights or
trajectories are written.

## How to reproduce

```bash
pip install -e ".[sim]"
breakpoint rate --robot go1 --policy breakpoint_mjx/assets/demo/go1_policy.pkl \
    --battery legged-quadruped-v1 --out report.json            # the full rating (GPU)
breakpoint rate --robot go1 --policy ./policy.onnx --obs-spec contract.yaml \
    --policy-name my-policy --out report.json                  # a BYO policy
breakpoint rate --robot go1 --policy <ckpt> --budget smoke     # a CPU preview
```

The full battery is 120 rollouts of 500 steps; on a GPU that is minutes, on a laptop CPU it
is long. `colab/ratings_cell.py` runs it on a Colab A100 (UNVERIFIED: written, not yet run).

`--budget smoke` runs the first 4 seeds for 150 steps. It exists to check the plumbing. A
smoke report says `"budget": "smoke"` in its run block and its summary label begins
`SMOKE PREVIEW, not a rating`. Never publish a smoke report as a rating.

## Measured so far

`docs/ratings/smoke-go1-demo.json`: a **SMOKE** run (CPU, 4 seeds, 150 steps) of the bundled
demo Go1 policy, 2026-10-03. It is a plumbing check, not a rating: at n = 4 a zero-fall
cell still has a 95% upper bound of 0.60.

| cell (SMOKE) | falls / n | 95% CI | result |
|---|---|---|---|
| nominal | 0/4 | [0.00, 0.60] | pass |
| low-friction | 0/4 | [0.00, 0.60] | pass |
| latency-20ms | 0/4 | [0.00, 0.60] | pass |
| latency-40ms | 3/4 | [0.19, 0.99] | fail |
| obs-noise | 0/4 | [0.00, 0.60] | pass |
| obs-dropout | 0/4 | [0.00, 0.60] | pass |
| obs-bias | 0/4 | [0.00, 0.60] | pass |
| lag-dropout | 4/4 | [0.40, 1.00] | fail |
| payload-2kg | 0/4 | [0.00, 0.60] | pass |
| payload-offcenter | 0/4 | [0.00, 0.60] | pass |
| weak-actuators | 0/4 | [0.00, 0.60] | pass |
| push-light | 0/4 | [0.00, 0.60] | pass |
| push-firm | 0/4 | [0.00, 0.60] | pass |
| tilt-pos-5deg | 0/4 | [0.00, 0.60] | pass |
| tilt-neg-5deg | 0/4 | [0.00, 0.60] | pass |

SMOKE PREVIEW, not a rating: 13/15 cells passed (4 seeds, 150 steps). No rollout was
flagged `sim_invalid`. The two failing cells are the two cells with 40 ms latency.
With a 150-step horizon the pushes land at step 74 and the robot has ~1.5 s to recover; the full
500-step battery may differ.

No full rating has been run yet.

## What a rating does NOT claim

* **Simulation only.** Every number comes from MuJoCo MJX on the Playground Go1 model. The
  Go1 search scene is feet-only collision (only the foot spheres touch the floor). Nothing
  here has been checked against real hardware.
* **Not a safety certification.** Passing every cell means the policy did not fall more than
  once in 8 seeds on 15 specific conditions. It does not bound the real-world fall rate, it
  does not cover conditions outside the battery, and it is not evidence that a robot is
  safe to deploy around people.
* **Not a search result.** A rating does not look for failures; a policy can pass every cell
  and still fail somewhere the search would find in minutes.
* **Small n.** 8 seeds per cell gives wide intervals (see above). Differences of one or two
  falls between policies are within noise.
* **Tied to the simulator model.** The theta dims perturb the model the target loads; a
  policy trained for a different embodiment (for example a Go2 policy on the Go1 model) is
  rated on the wrong robot, and the rating says nothing about its own robot.
