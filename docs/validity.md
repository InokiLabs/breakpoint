# Physical-validity gate and floor-relative fall metric

BreakPoint scores a rollout as a failure when the policy falls. Two things used to go
wrong with that verdict on the Go1:

1. Falls were judged on the world-frame trunk height `qpos[2]`. On a tilted floor
   (`floor_slope_rad`) the robot walks downhill, world z drops below
   `trunk_height_min = 0.18`, and an upright walk was scored as a fall.
2. Nothing checked whether the simulator itself misbehaved (NaN state, torso passing
   through the floor, contacts sinking far deeper than walking ever produces, an
   acceleration blow-up). A rollout like that could be scored as a critical failure.

Both are now handled by default. Nothing to enable: it is on whenever the trajectory
carries the three validity channels, which the legged summary now always emits.

## What changed

| Piece | File | Behaviour |
|---|---|---|
| Summary channels | `breakpoint_mjx/objectives/legged.py` | `legged_summarize` keeps the original 5 keys (`trunk_height` is still `qpos[2]`) and adds `floor_height` (torso height along the floor's own normal), `penetration` (min contact `dist` this step, negative = interpenetration) and `qacc_max` (max abs `qacc`). |
| Fall metric | `breakpoint_mjx/failure.py` | `component_signals`, `behavior_descriptor` and `rollout_signature` use `floor_height` when present, else `trunk_height`. |
| Gate | `breakpoint_mjx/failure.py` | `component_signals` returns `sim_invalid` (0/1) and `sim_invalid_reason` (bitmask: 1 nonfinite, 2 below floor, 4 penetration, 8 blow-up). `failure_score` returns 0.0 for a flagged rollout, so the search never climbs toward it and families, stats and riskcert never count it. Helpers: `is_sim_invalid(sig)`, `sim_invalid_reasons(sig)`. |
| C-MuJoCo replay | `breakpoint_mjx/backends/mujoco_c.py` (no change needed) | The C backend already builds its per-step summary through the same `legged_summarize` on an `mjx.put_data` view of the C state, so it emits the same three keys. Checked by `tests/test_mujoco_c_parity.py::test_c_summary_carries_the_validity_keys` (run with `BP_RUN_CBACKEND=1`; passed on CPU). The Genesis backend's duck state has no contact or geom arrays, so it emits only the original 5 keys and scores exactly as before. |
| Stats | `breakpoint_mjx/stats.py` | `compute_report_stats(..., records=all_records)` counts flagged rollouts (`sim_validity_counts`) and computes every rate over physically valid rollouts only. |
| Report | `breakpoint_mjx/report.py` | A "Physical validity" line at the top of the report ("N of M evaluated worlds were excluded as simulator artifacts (penetration / below-floor / blow-up)" plus per-reason counts). Per-family numbers are relabeled "Share of search-selected failures" (members / failures), with the member fraction of the search-evaluated worlds as a secondary figure. `write_report` picks the records up from the run's `failures.json` automatically (it holds every evaluated record and is only used when its count equals `n_total`). A zero-volume safety envelope reads "No failure-free box found around nominal" instead of a list of `[nominal, nominal]` intervals (`report.py` when an envelope is passed, `envelope._summarize` for the `families.json` / console summary). |

Records written before this change carry no `sim_invalid` key; the report then says the
gate was not active instead of implying 0 artifacts were checked.

## Thresholds (all overridable in `cfg.thresholds` / a target's `thresholds`)

| Key | Default | Meaning |
|---|---|---|
| `penetration_max` | 0.04 m | flag if contact penetration is deeper than this |
| `qacc_max` | 1e4 | flag if max abs `qacc` is larger than this |
| `validity_settle_steps` | 10 | penetration is not checked in the first N steps |
| `validity_whole_rollout` | 0 | 1 = inspect every step (see "Causal window" below) |

Calibration (CPU, demo checkpoint `media/dl/frozen_go1_policy.pkl`, horizon 150, 32
rollouts at the nominal theta): worst nominal penetration **-0.0266 m**, worst nominal
max abs `qacc` **2466**. The defaults sit at about 1.5x and 4x those. With them,
**0 of 32** nominal rollouts are flagged.

## Measured sweep (CPU, horizon 150, 32 rollouts per row, one axis at a time)

CPU / short horizon. These are NOT GPU or 500-step numbers. Horizon 150 is 3 s at 50 Hz.
Reproduce with:

```
cd <repo>
PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/validity_probe.py \
    --ckpt media/dl/frozen_go1_policy.pkl --horizon 150 --n 32 \
    --out docs/validity_probe.json
```

(about 60 s on a laptop CPU; raw per-rollout rows are in `docs/validity_probe.json`).

| Setting | flagged (default gate) | falls (new metric) | falls (old world-z metric) | flagged (whole-rollout rule) | end below floor | min penetration (m) | max abs qacc | min floor_height (m) | min world z (m) |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 0 | 0 | 0 | 0 | 0 | -0.0266 | 2466 | 0.276 | 0.277 |
| obs_dropout_prob=0.05 | 0 | 0 | 0 | 0 | 0 | -0.0257 | 2528 | 0.270 | 0.270 |
| obs_dropout_prob=0.1 | 0 | 1 | 1 | 1 | 1 | -0.0324 | 2804 | -0.338 | -0.337 |
| obs_dropout_prob=0.15 | 0 | 7 | 7 | 6 | 6 | -0.0435 | 3508 | -0.354 | -0.352 |
| obs_dropout_prob=0.2 | 0 | 18 | 18 | 18 | 18 | -0.0641 | 3296 | -0.359 | -0.359 |
| floor_slope_rad=-0.1 | 0 | 0 | 7 | 0 | 0 | -0.0593 | 2524 | 0.238 | -0.091 |
| floor_slope_rad=-0.05 | 0 | 0 | 1 | 0 | 0 | -0.0342 | 2172 | 0.259 | 0.120 |
| floor_slope_rad=+0.05 | 0 | 0 | 5 | 0 | 0 | -0.0393 | 2453 | 0.257 | 0.101 |
| floor_slope_rad=+0.1 | 0 | 0 | 14 | 0 | 0 | -0.0641 | 2247 | 0.236 | -0.110 |

Mean failure score on `floor_slope_rad=+0.1` drops from 0.394 (old metric) to 0.048;
on `obs_dropout_prob=0.2` it is unchanged at 0.586.

### Finding 1: slope "falls" were a metric bug (fixed)

With the old world-z metric, 14 of 32 upright walks on `floor_slope_rad=+0.1` (and 7 of
32 on -0.1) were scored as falls. Their floor-relative height never went below 0.236 m,
so none of them was a fall. With `floor_height` they are all scored as non-falls.

### Finding 2: the dropout "sink through the floor" is a real fall followed by an artifact

The Go1 Playground model gives floor collision ONLY to the four foot spheres (geoms 15,
26, 38, 50, `conaffinity=1`; the floor has `contype=1`; no explicit pairs). Trunk and legs
have no floor collision at all. Measured on `obs_dropout_prob=0.2`: 18 of 32 rollouts end
with the torso below the floor plane, and **in all 18 the robot tipped over first**
(up_proj < 0.5 came 6 to 12 steps before floor_height < 0; up_proj at the below-floor step
was between -0.38 and -0.92). The perception dropout makes the policy fall over. After
that, the trunk has nothing to land on and sinks.

So these are genuine policy falls with a post-failure simulator artifact. Scoring them 0
would erase 18 real failures at dropout 0.2. The default gate therefore uses a causal
window:

- nonfinite / penetration / blow-up are checked only up to and including the first fall
  step (an artifact after the policy has already failed does not change the verdict);
- "below floor" counts only while the robot has not yet tipped over (an upright torso
  below the floor plane is physically impossible, so that is an artifact).

The literal rule from the original spec ("flag if the torso is below the floor at any
step") is still available as `validity_whole_rollout: 1`. Under it, the 18 dropout-0.2
rollouts ARE flagged and score 0, which is exactly the problem above.

This is also why the failure videos for such families show the robot sinking through
the ground after it falls. Two follow-ups are left open: truncate the render at the fall
step, or give the trunk a collision geom in the Go1 model (that changes the physics, so it
needs its own measurement).

### Finding 3: tilted floors spawn the feet inside the plane

The env's reset pose assumes a flat floor. On `floor_slope_rad=+/-0.1` the feet start up to
6.4 cm inside the tilted plane. Every step with penetration deeper than 3 cm on the slope
sweep fell in steps 0 to 9, in runs of at most 2 steps. That spawn transient is why
penetration is not checked in the first `validity_settle_steps = 10` steps.

## Acceptance status

| Criterion | Status |
|---|---|
| 0 of at least 32 nominal rollouts flagged (CPU, horizon 150) | Met: 0 / 32. |
| `floor_slope_rad=+0.1` alone is not scored as a fall | Met: 0 / 32 falls (was 14 / 32). |
| `obs_dropout_prob=0.2` rollouts ending below the floor are flagged sim_invalid with score 0 | **Not met by default, on purpose.** Measured: all 18 are real tip-overs that sink afterwards (Finding 2). The literal behaviour is available with `validity_whole_rollout: 1` (18 / 32 flagged, score 0). No rollout in this sweep sank while still upright, so the default gate flagged 0. |
| Report shows excluded count with reasons, per-family relabel, zero-volume envelope suppressed | Met in `report.py` + tests. The count reaches the real `run_pipeline` report through `failures.json`. The envelope block needs `write_report(..., envelope=families_payload.get("envelope"))` in `pipeline.py`, which this change was not allowed to touch; the zero-volume wording already reaches `families.json` and the console through `envelope._summarize`. |
| Trajectories without the new keys score byte-identically | Met: `tests/test_validity_gate.py::test_missing_validity_keys_score_byte_identically` compares against a verbatim copy of the old scorer. |

## GPU check (UNVERIFIED, not run here)

The numbers above are CPU and horizon 150. To repeat at horizon 500 on a GPU runtime
(Colab), paste this cell. It has NOT been run.

```python
# UNVERIFIED: GPU/500-step repeat of the CPU validity probe. Not run by the author.
!git clone https://github.com/InokiLabs/breakpoint.git breakpoint-mjx
%cd breakpoint-mjx
!pip install -q -r requirements.txt
!PYTHONPATH=. python scripts/validity_probe.py --ckpt media/dl/frozen_go1_policy.pkl \
    --horizon 500 --n 256 --out validity_probe_gpu.json
```

Check: nominal flagged count stays 0 (if not, raise `penetration_max` / `qacc_max` to
just above the printed nominal worst case); slope rows keep 0 falls. On GPU the MJX
implementation may differ (`warp` vs `jax`); `legged_summarize` reads
`data._impl.contact.dist` and falls back to "no contacts" when that array is absent,
which would silently disable the penetration check, so confirm `min_pen` is not 1000.
