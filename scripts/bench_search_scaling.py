"""Measure multi-restart CMA-ES against single-start on a landscape with known answers.

Runs on CPU. The physics is replaced by a SYNTHETIC failure landscape whose ground
truth we control: K Gaussian "failure modes" at known locations in the real 16-dim
theta space. That is the only way to say something honest about search quality
without a GPU, because on real physics nobody knows how many failure modes exist.

The synthetic evaluate returns trajectories in the legged summary contract, so the
REAL scoring path (component_signals -> failure_score -> transfer weighting) runs
unmodified. What is swapped out is the simulator, not the search or the scorer.

Two comparisons, because they answer different questions:

  EQUAL GPU CALLS  - the efficiency claim. Both do the same number of sequential
                     batched calls; multi-restart packs `restarts` x more worlds into
                     each one, using the device capacity single-start leaves idle.
  EQUAL EVALS      - the quality claim. Same total simulations; single-start gets
                     proportionally more generations. This asks whether restarts are
                     worth anything once you pay for them, and it is the comparison
                     that could embarrass the change.

  python scripts/bench_search_scaling.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

HORIZON = 8
N_MODES = 6
#: How close a sample must get to count as having FOUND a planted mode.
MODE_RADIUS = 0.30
#: Width of the score kernel. This is deliberately MUCH wider than MODE_RADIUS.
#: In 16 dimensions two random points sit ~1.0-1.2 apart, so a kernel as narrow as
#: the found-radius makes exp(-d^2/2s^2) underflow to 0.0 everywhere except at the
#: centre itself: the modes become isolated needles with no gradient, and EVERY
#: optimizer scores 0/6. (Measured: at sigma=0.18 a point 0.73 away scored exactly
#: 0.000.) A landscape no search can climb compares nothing, so the signal has to be
#: broad enough to follow while the success test stays strict.
DRIVE_SIGMA = 0.55


def make_landscape(space, n_modes=N_MODES, seed=0):
    """K mode centres in NORMALIZED [0,1]^D theta space, plus a scorer.

    Centres are placed away from the space centre, because a landscape whose modes
    sit where CMA-ES already starts would flatter multi-start for the wrong reason.
    """
    D = len(space.names())
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.12, 0.88, size=(n_modes, D))
    # Push each centre away from the middle so none is trivially reachable.
    for c in centres:
        far = np.abs(c - 0.5) < 0.2
        c[far] = np.where(rng.random(far.sum()) < 0.5, 0.15, 0.85)
    return centres


def normalized(space, thetas):
    lo = np.array(space.lower())
    hi = np.array(space.upper())
    return (np.asarray(thetas) - lo) / np.maximum(hi - lo, 1e-9)


def make_evaluate(space, centres):
    """Synthetic evaluate: closeness to the nearest mode centre drives a fall.

    Returns the legged trajectory contract so the real scorer runs on it.
    """
    def evaluate(thetas, key):
        u = normalized(space, thetas)                        # (N, D)
        d = np.linalg.norm(u[:, None, :] - centres[None, :, :], axis=2)  # (N, K)
        nearest = d.min(axis=1)
        # drive in [0,1]: 1 at a mode centre, decaying with distance
        drive = np.exp(-(nearest ** 2) / (2 * (DRIVE_SIGMA ** 2)))
        return trajectories_from_drive(drive)

    return evaluate


def trajectories_from_drive(drive):
    """Map a per-world drive in [0,1] to the legged trajectory contract (monotone:
    more drive, lower trunk, more tilt, more spin). Shared with scripts/bench_coverage.py
    so every planted landscape goes through the same real scorer."""
    import jax.numpy as jnp

    drive = np.asarray(drive, dtype=float)
    n = drive.shape[0]
    t = np.arange(HORIZON)[None, :]
    height = 0.30 - 0.25 * drive[:, None] * (t / HORIZON)
    up = 1.0 - 0.9 * drive[:, None] * (t / HORIZON)
    return {
        "trunk_height": jnp.asarray(height),
        "up_proj": jnp.asarray(up),
        "lin_vel": jnp.zeros((n, HORIZON, 3)),
        "ang_vel": jnp.asarray(np.repeat(drive[:, None, None], HORIZON, 1)
                               * np.ones((1, 1, 3)) * 9.0),
        "command": jnp.zeros((n, HORIZON, 3)),
    }


def calibrate_radius(space, centres, cfg):
    """Derive the found-radius from the landscape instead of picking one by hand.

    A mode counts as FOUND when a sample got close enough to score at or above the
    pipeline's own ``failure_threshold`` -- i.e. close enough that the search actually
    produced a scored failure belonging to that mode. Choosing the number any other
    way risks tuning the metric until it says what you hoped.
    """
    import jax.numpy as jnp  # noqa: F401  (evaluate returns jnp arrays)

    from breakpoint_mjx.search import scores_from_trajectories

    evaluate = make_evaluate(space, centres)
    lo = np.array(space.lower())
    hi = np.array(space.upper())
    c = centres[0]
    # Walk outward from a mode centre along a fixed direction and find the distance
    # at which the score falls below the failure threshold.
    direction = np.ones_like(c) / np.sqrt(len(c))
    dists = np.linspace(0.0, 1.5, 61)
    pts = np.clip(c[None, :] + dists[:, None] * direction[None, :], 0.0, 1.0)
    actual = np.linalg.norm(pts - c[None, :], axis=1)
    scores, _, _ = scores_from_trajectories(
        evaluate(lo + pts * (hi - lo), None), cfg.thresholds, HORIZON)
    ok = np.asarray(scores) >= cfg.failure_threshold
    return float(actual[ok].max()) if ok.any() else MODE_RADIUS


def modes_found(space, records, centres, radius):
    """How many of the K planted modes any sample actually reached."""
    if not records:
        return 0
    u = normalized(space, np.array([r.theta_vector for r in records]))
    d = np.linalg.norm(u[:, None, :] - centres[None, :, :], axis=2)
    return int((d.min(axis=0) <= radius).sum())


def closest_per_mode(space, records, centres):
    """Distance to each planted mode's nearest sample -- richer than a binary count."""
    if not records:
        return np.full(len(centres), np.inf)
    u = normalized(space, np.array([r.theta_vector for r in records]))
    return np.linalg.norm(u[:, None, :] - centres[None, :, :], axis=2).min(axis=0)


def run_single(evaluate, space, cfg, generations):
    from breakpoint_mjx.search import cmaes_search
    import jax

    c = replace(cfg, generations=generations)
    t0 = time.perf_counter()
    recs = cmaes_search(evaluate, space, c, jax.random.PRNGKey(0))
    return recs, time.perf_counter() - t0, generations


def run_multi(evaluate, space, cfg, generations, restarts, novelty=False):
    from breakpoint_mjx.multi_search import multi_cmaes_search
    import jax

    c = replace(cfg, generations=generations)
    t0 = time.perf_counter()
    recs = multi_cmaes_search(evaluate, space, c, jax.random.PRNGKey(0),
                              restarts=restarts, novelty=novelty)
    return recs, time.perf_counter() - t0, generations


def summarize(space, recs, centres, label, calls, secs, radius):
    scores = np.array([r.failure_score for r in recs])
    top20 = np.sort(scores)[::-1][:20].mean() if scores.size else 0.0
    closest = closest_per_mode(space, recs, centres)
    return {
        "label": label,
        "evals": len(recs),
        "gpu_calls": calls,
        "seconds": round(secs, 1),
        "modes_found": modes_found(space, recs, centres, radius),
        "mean_closest": round(float(closest.mean()), 3),
        "best": round(float(scores.max()) if scores.size else 0.0, 3),
        "top20_mean": round(float(top20), 3),
    }


def main():
    space = load_theta_space(config_path("go1/theta_space.yaml"))
    cfg = load_search_config(config_path("search.yaml"))
    cfg = replace(cfg, popsize=128, horizon=HORIZON)

    centres = make_landscape(space)
    evaluate = make_evaluate(space, centres)
    restarts = 8
    gens = 15
    radius = calibrate_radius(space, centres, cfg)

    print(f"landscape: {N_MODES} planted failure modes in {len(space.names())}-dim "
          f"theta space")
    print(f"found-radius {radius:.3f} (calibrated: the distance at which a sample "
          f"still scores >= the {cfg.failure_threshold} failure threshold)")
    print(f"popsize={cfg.popsize}  restarts={restarts}  horizon={HORIZON}\n")

    rows = []

    # --- EQUAL GPU CALLS: same number of sequential batched evaluations ---------
    r_single, t_single, g = run_single(evaluate, space, cfg, gens)
    rows.append(summarize(space, r_single, centres, "single-start", g, t_single, radius))

    r_multi, t_multi, g = run_multi(evaluate, space, cfg, gens, restarts,
                                    novelty=False)
    rows.append(summarize(space, r_multi, centres, f"multi x{restarts}", g, t_multi,
                          radius))

    r_nov, t_nov, g = run_multi(evaluate, space, cfg, gens, restarts, novelty=True)
    rows.append(summarize(space, r_nov, centres, f"multi x{restarts} + novelty",
                          g, t_nov, radius))

    # --- EQUAL EVALS: single-start gets `restarts` x more generations -----------
    r_fair, t_fair, g = run_single(evaluate, space, cfg, gens * restarts)
    rows.append(summarize(space, r_fair, centres, "single-start (equal evals)",
                          g, t_fair, radius))

    print("\n" + "=" * 78)
    hdr = (f"{'strategy':<28}{'evals':>8}{'calls':>7}{'modes':>7}"
           f"{'closest':>9}{'best':>8}{'top20':>8}")
    print(hdr)
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<28}{r['evals']:>8}{r['gpu_calls']:>7}"
              f"{r['modes_found']:>5}/{N_MODES}{r['mean_closest']:>9}"
              f"{r['best']:>8}{r['top20_mean']:>8}")
    print("=" * 78)

    s, m, nov, f = rows[0], rows[1], rows[2], rows[3]
    print(f"\nEQUAL GPU CALLS ({s['gpu_calls']} batched evaluations each):")
    print(f"  work per call        : {s['evals']} -> {m['evals']} evals "
          f"({m['evals'] / max(s['evals'], 1):.0f}x)")
    print(f"  modes found          : {s['modes_found']}/{N_MODES} -> "
          f"{m['modes_found']}/{N_MODES}")
    print(f"  mean closest approach: {s['mean_closest']} -> {m['mean_closest']} "
          f"(lower is better)")
    print(f"\nEQUAL EVALS ({f['evals']} vs {m['evals']}):")
    print(f"  single-start needed  : {f['gpu_calls']} sequential calls vs "
          f"{m['gpu_calls']} ({f['gpu_calls'] / max(m['gpu_calls'], 1):.0f}x more)")
    print(f"  modes found          : {f['modes_found']}/{N_MODES} (single) vs "
          f"{m['modes_found']}/{N_MODES} (multi)")
    print(f"  top20 mean           : {f['top20_mean']} (single) vs "
          f"{m['top20_mean']} (multi)")
    return rows


if __name__ == "__main__":
    main()
