# Skeptic-grade evidence: minimal counterexamples, independent full-collision replay

**Every number on this page is a CPU smoke measurement** (Windows laptop, no GPU,
`BP_BUDGET=smoke`: horizon 150, a 192-world search reused from an earlier smoke run of the
bundled Go1 demo policy). It is a plumbing and honesty check of the evidence layer, not a
converged search and not a statement about the policy. The standard-budget GPU run is the
Colab cell at the bottom, which is UNVERIFIED (never run).

## What a report now shows per family

1. **Minimal counterexample vs nominal twin.** After Certified Minimal Cause (CMC,
   [mincause.md](mincause.md)), each certified family's minimal counterexample (every dim
   at nominal except the certified cause dims) is rendered next to a nominal twin with the
   same seed and every dim nominal. Both videos end 1 s after the counterexample's fall
   step. The "Triggering parameters" table lists only the certified dims, as
   `nominal -> counterexample`. The original 16-dim search representative is kept as a
   collapsed secondary video.
2. **Independent replay badge** (on by default, `BP_REPLAY=0` disables). The minimal
   counterexample is re-run on `BP_REPLAY_K=8` seeds in C MuJoCo (host CPU,
   `backends/mujoco_c.py`) on Playground's `scene_mjx_fullcollisions_flat_terrain.xml`:
   hip, thigh, calf and trunk primitives collide with the floor, where the MJX search
   scene is feet-only. Same robot, same masses, gains, damping, solver options and floor
   friction (copied from the search model); only the contact geometry differs. The badge
   reads `reproduces in C-MuJoCo full-collision: k/n (CP interval lo-hi, bar 0.50)`, with
   the same one-sided 90% Clopper-Pearson rule CMC uses: **reproduced** if the lower bound
   clears 0.5, **did not reproduce** if the upper bound is below it, otherwise
   **inconclusive at this seed budget**. Only reproduced families stay ranked; the others
   move to "Did not reproduce outside MJX at the certification bar".
   `BP_REPLAY_MAXMULT=4` tops up an inconclusive result in waves of K up to 32 seeds, the
   way CMC pools. If the full-collision model cannot be built, the report says so.
3. **Physics-fidelity disclosure.** The report states the search scene's collision model
   and flags any family whose MJX representative rollout ends with the torso more than
   2 cm below the floor plane.
4. **Report hygiene.** A family CMC could not certify is listed only under "Unconfirmed
   (not reproducible under seed noise)" as `uncertified: <dims>`, with no rank and no
   severity. Priority ranks are renumbered 1..n after CMC merges. Every margin and label
   string is html-escaped. The surrogate R2 is 5-fold cross-validated and hidden, with the
   reason, when there are fewer than 200 labeled worlds or fewer than 20 of the rarer
   class. When more than 80% of the envelope surrogate's labels are failures, an empty
   envelope reads "no failure-free box found at this budget and label balance", next to
   the measured nominal pass count.
5. `--policy C:/path/policy.pkl` (a Windows path) now loads as a brax params pickle
   instead of failing with "unknown policy kind c".

## How the replay is made cheap on CPU

The eager C loop costs about 4 s per reset plus about 70 ms per step on this laptop (an
eager mjx forward inside the env's reset, then several hundred eager jax ops per
`env.step`). `jit_env=True` compiles the env's OWN reset and step in two passes around the
host `mj_step` (`_JitEnvStep` in `backends/mujoco_c.py`): no env logic is re-implemented,
only compiled. Measured: about 19 ms per step including `mjx.put_data`, after a one-time
compile of about 10 s. Rollouts also stop 1 s after the fall, which cannot change the
verdict (a physically valid fall scores at least 0.7 and the validity gate reads only
steps up to the fall). The opt-in test `test_jit_env_matches_eager_c_loop_early` checks
the compiled loop against the eager one.

## Measured CPU smoke run (2026-09-27)

Command (reuses a prior smoke search so the run skips the search):

```bash
mkdir -p run/go1 && cp <prior smoke out>/go1/failures.json run/go1/
BP_BUDGET=smoke BP_OUT=run BP_EMBED_VIDEO=0 \
XLA_FLAGS="--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true" \
python -m breakpoint_mjx.pipeline --policy media/dl/frozen_go1_policy.pkl
```

The policy file is byte-identical (md5 `d626f46e...`) to the bundled
`breakpoint_mjx/assets/demo/go1_policy.pkl`. Wall time: 31 min 19 s, almost all of it
CMC's MJX rollouts on CPU. The replay stage took 104.3 s.

**Families.** Clustering found 4. CMC certified 3 and merged two with the same cause, so
2 certified families plus 1 uncertified remain.

| family | CMC (MJX, feet-only) | replay (C MuJoCo, full collision) | verdict |
|---|---|---|---|
| latency_steps >= 1.52 AND obs_dropout_prob >= 0.0249 | 12/16 seeds, CP lower 0.56 | 6/8, CP interval 0.46-0.93 | inconclusive |
| obs_dropout_prob >= 0.227 AND obs_bias_scale >= 0.00613 | 7/8 seeds, CP lower 0.59 | 4/6, CP interval 0.33-0.91 (stopped at 6: 7/8 out of reach) | inconclusive |
| uncertified: push_step_frac, kd_scale, com_offset_x_m | representative 4/16, CP lower 0.11 | not replayed | unconfirmed |

**Result: 0 of 2 certified families reproduced at the bar outside MJX; both are
inconclusive at 8 seeds, not refuted.** Both still fall on most full-collision seeds
(fall steps `[38, 84, 90, 52, 52, 97, -, -]` and `[53, 127, -, 48, 130, -]`), and in
both cases the nominal twin (same seed 0) did not fall. The report therefore ranks
nothing and says so. Whether the two causes hold outside MJX needs more seeds
(`BP_REPLAY_MAXMULT=4`) or the standard-budget run below; this page does not claim
either way.

**Physics fidelity.** Lowest torso height above the floor over all replay rollouts:
0.055 m and 0.054 m (full collision). The same families' MJX representative rollouts end
0.318 m, 0.382 m and 0.301 m BELOW the floor plane (feet-only scene), and all three are
flagged in the report. The opt-in test `test_full_collision_replay_keeps_torso_above_floor`
asserts torso height >= -0.02 m on a hard fall in the full-collision replay.

**Videos (frames at 25 fps, every 2nd 50 Hz step).** `fam-1-mincause.mp4` 45 frames (fall
at step 38, plus 50 steps), `fam-1-nominal.mp4` 45 (cut to the same length),
`fam-0-mincause.mp4` 53 (fall at step 53), `fam-0-nominal.mp4` 53.

**Other lines in the same report.** Nominal check: 16/16 seeds pass at the space nominal.
Surrogate R2: not shown (192 labeled worlds, 179 fail / 13 pass). Envelope: "no
failure-free box found at this budget and label balance (surrogate trained on 96 worlds,
94% labeled as failures; nominal parameters: 16/16 seeds pass)". The graders' review of
the report before this change quoted an in-sample R2 of 1.000 and "no failure-free box"
next to 0/16 nominal falls (quoted from the review, not re-measured here).

`families.json` excerpt (CPU smoke):

```json
{"name": "latency gait collapse + high obs dropout prob",
 "status": "not_confirmed_outside_mjx", "priority_rank": 1,
 "minimal_cause": [{"dim": "latency_steps", "value": 1.5155, "direction": "high"},
                   {"dim": "obs_dropout_prob", "value": 0.0249, "direction": "high"}],
 "replay": {"verdict": "inconclusive", "k": 6, "n": 8, "cp_lo": 0.4618, "cp_hi": 0.9314,
            "fall_steps": [38, 84, 90, 52, 52, 97, null, null],
            "min_floor_height": 0.0549,
            "nominal_twin": {"fell": false, "min_floor_height": 0.2814, "steps": 89}},
 "evidence_videos": {"counterexample": "fam-1-mincause.mp4",
                     "nominal": "fam-1-nominal.mp4",
                     "source": "c_mujoco_full_collision_replay"},
 "mjx_min_floor_height": -0.318}
{"evidence": {"replay": {"status": "ran", "k": 8, "n_replayed": 2, "n_reproduced": 0,
                         "n_demoted": 2, "n_inconclusive": 2, "seconds": 104.3,
                         "collision_model": {"source": "playground_scene",
                                             "scene": "scene_mjx_fullcollisions_flat_terrain.xml",
                                             "n_collidable": 31}},
              "surrogate_fit": {"r2": null, "method": "5-fold cross-validated", "n": 192,
                                "n_fail": 179, "n_pass": 13},
              "nominal_check": {"n": 16, "fails": 0}}}
```

**Replay cost against the acceptance bar (<= 3 min for 4 families).** Measured 104.3 s
for 2 families, including the one-time compile of about 10 s and 4 video renders. This
run had only 2 certified families, so 4 families were NOT measured. At the measured rate
(about 47 s per family), 4 families would take about 200 s, over the 3-minute bar. That
figure is an extrapolation, not a measurement. Early stopping makes a clearly reproducing
family cheaper than an inconclusive one, and `BP_REPLAY_K` trades seeds for time.

## Tests

- Default suite: `tests/test_report.py` (sections, contiguous ranks, escaping, cause
  table, videos, fidelity flag, envelope wording), `tests/test_crossval_engines.py`
  (replay verdicts, early stop, CMC-style pooling, escape hatch, pristine-model restore,
  enable-contacts fallback on a real tiny MuJoCo model), `tests/test_surrogate.py` (CV R2
  vs in-sample, hiding rule), `tests/test_pipeline_dispatch.py` (Windows `.pkl` routing,
  line buffering, stage order).
- Opt-in (`BP_RUN_CBACKEND=1`, real Go1 env, about 2 min on CPU):
  `tests/test_mujoco_c_parity.py -k "full_collision or jit_env"` checks that the
  full-collision scene is a layout-compatible drop-in, that the compiled loop matches
  the eager loop, and that the torso stays above the floor on a hard fall. 3 passed on
  this laptop.

## GPU run (UNVERIFIED)

[`colab/evidence_cell.py`](../colab/evidence_cell.py): a fresh standard-budget Go1 search
with CMC and the replay pooled like CMC (`BP_REPLAY_MAXMULT=4`). It prints each family's
status, badge, replay and MJX floor heights, and the replay seconds. It has never been
run.
