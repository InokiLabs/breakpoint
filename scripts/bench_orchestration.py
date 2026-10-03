"""Measure the orchestration fixes: recompiles-per-policy-version and scoring throughput.

Runs on CPU with a tiny env/horizon, because what is being measured is COMPILE COUNT
and HOST-TRANSFER COST, not physics throughput -- both are hardware-independent.

  python scripts/bench_orchestration.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bench_recompiles():
    """Count traces when the SAME evaluator is reused across 3 'policy versions'.

    A retrace re-runs the Python body, so a counter inside the body counts compiles.
    Uses a stand-in rollout with the same closure structure as the real evaluator
    (params baked into a closure vs threaded as an argument), so the result is about
    the plumbing rather than about MJX.
    """
    import jax
    import jax.numpy as jnp

    versions = [
        {"w": jnp.array([float(k) + 1.0]), "b": jnp.array([0.1 * k])} for k in range(3)
    ]
    thetas = jnp.ones((8, 4))
    key = jax.random.PRNGKey(0)

    # --- closure style (make_evaluator): params baked in, new callable per version
    closure_traces = {"n": 0}

    def make_closure_evaluator(params):
        def _ev(th, k):
            closure_traces["n"] += 1
            act = th @ jnp.ones((4, 1)) * params["w"] + params["b"]
            return jax.lax.scan(lambda c, _: (c + act.sum(), c), 0.0, jnp.arange(64))[0]
        return jax.jit(_ev)

    for p in versions:
        make_closure_evaluator(p)(thetas, key).block_until_ready()

    # --- argument style (make_param_evaluator): one compile serves all versions
    arg_traces = {"n": 0}

    def _ev_params(params, th, k):
        arg_traces["n"] += 1
        act = th @ jnp.ones((4, 1)) * params["w"] + params["b"]
        return jax.lax.scan(lambda c, _: (c + act.sum(), c), 0.0, jnp.arange(64))[0]

    ev = jax.jit(_ev_params)
    for p in versions:
        ev(p, thetas, key).block_until_ready()

    return closure_traces["n"], arg_traces["n"], len(versions)


def bench_scoring(n=256, horizon=500):
    """Time scores_from_trajectories on a device-resident trajectory batch.

    Compares the shipped batched-conversion path against the previous per-row
    conversion, on identical data, asserting identical scores.
    """
    import jax.numpy as jnp

    from breakpoint_mjx.config import load_search_config
    from breakpoint_mjx.search import scores_from_trajectories

    cfg = load_search_config(config_path("search.yaml"))
    rng = np.random.default_rng(0)
    traj_np = {
        "trunk_height": rng.uniform(0.05, 0.35, (n, horizon)),
        "up_proj": rng.uniform(0.2, 1.0, (n, horizon)),
        "lin_vel": rng.normal(0, 0.5, (n, horizon, 3)),
        "ang_vel": rng.normal(0, 2.0, (n, horizon, 3)),
        "command": rng.normal(0, 0.5, (n, horizon, 3)),
    }
    traj = {k: jnp.asarray(v) for k, v in traj_np.items()}

    # previous behaviour: convert each row of each key separately
    def old_scores(traj, thresholds, horizon):
        from breakpoint_mjx.objectives import LEGGED_OBJECTIVE as obj
        m = int(np.asarray(next(iter(traj.values()))).shape[0])
        out = np.zeros(m)
        for i in range(m):
            one = {k: np.asarray(v[i]) for k, v in traj.items()}
            out[i] = obj.failure_score(obj.component_signals(one, thresholds), horizon)
        return out

    t0 = time.perf_counter()
    old = old_scores(traj, cfg.thresholds, horizon)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new, _, _ = scores_from_trajectories(traj, cfg.thresholds, horizon)
    t_new = time.perf_counter() - t0

    assert np.allclose(old, new), "scores changed -- the optimization is not safe"
    return t_old, t_new, n, horizon


if __name__ == "__main__":
    closure_n, arg_n, versions = bench_recompiles()
    print(f"[recompiles] {versions} policy versions: "
          f"closure-style traces={closure_n}  param-arg traces={arg_n}")

    t_old, t_new, n, h = bench_scoring()
    print(f"[scoring] N={n} H={h}: per-row {t_old*1000:.0f} ms -> batched "
          f"{t_new*1000:.0f} ms ({t_old/t_new:.1f}x), scores identical")
