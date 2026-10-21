"""CPU tests for parallel multi-restart CMA-ES.

The pure coordination logic (starting points, score scattering) is tested directly;
the loop is tested end to end against a synthetic evaluator, so the real scoring path
runs but no physics does.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("evosax")

from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.multi_search import (  # noqa: E402
    multi_cmaes_search, restart_means, split_scores,
)
from breakpoint_mjx.paths import config_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 6


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def cfg():
    c = load_search_config(config_path("search.yaml"))
    return replace(c, popsize=8, horizon=HORIZON, generations=2)


# ---------------------------------------------------------------------------
# restart_means
# ---------------------------------------------------------------------------

def test_restarts_start_from_different_points(space):
    means = restart_means(space, restarts=6, seed=0)
    assert means.shape == (6, len(space.names()))
    # Identical starts would make the restarts decorative: run the same search 6x.
    pairwise = [np.linalg.norm(means[i] - means[j])
                for i in range(6) for j in range(i + 1, 6)]
    assert min(pairwise) > 1e-6


def test_restart_zero_keeps_the_informed_start(space):
    """Multi-start must never be WORSE than the single-start run it replaces."""
    seed_theta = {n: space.dims[n].nominal for n in space.names()}
    seed_theta["latency_steps"] = space.dims["latency_steps"].bounds[1]
    means = restart_means(space, restarts=4, seed=0, seed_theta=seed_theta)

    from breakpoint_mjx.search import theta_to_cmaes_mean
    assert np.allclose(means[0], theta_to_cmaes_mean(space, seed_theta))


def test_no_seed_theta_starts_restart_zero_at_the_centre(space):
    means = restart_means(space, restarts=3, seed=0, seed_theta=None)
    assert np.allclose(means[0], 0.0)


def test_restart_means_are_deterministic_for_a_seed(space):
    a = restart_means(space, restarts=5, seed=7)
    b = restart_means(space, restarts=5, seed=7)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# split_scores
# ---------------------------------------------------------------------------

def test_split_scores_partitions_in_order():
    vals = np.arange(12.0)
    parts = split_scores(vals, restarts=3, popsize=4)
    assert len(parts) == 3
    assert np.allclose(parts[0], [0, 1, 2, 3])
    assert np.allclose(parts[2], [8, 9, 10, 11])


def test_split_scores_rejects_a_mismatched_batch():
    """A silently mis-sliced batch would train each restart on another's results."""
    with pytest.raises(ValueError, match="expected 12"):
        split_scores(np.arange(10.0), restarts=3, popsize=4)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def _synthetic_evaluate(space):
    """Latency drives a fall; everything else is inert. Real scorer runs on it."""
    import jax.numpy as jnp

    names = space.names()
    j = names.index("latency_steps")
    lo = np.array(space.lower())
    hi = np.array(space.upper())

    def evaluate(thetas, key):
        th = np.asarray(thetas)
        drive = (th[:, j] - lo[j]) / (hi[j] - lo[j])
        n = th.shape[0]
        t = np.arange(HORIZON)[None, :]
        return {
            "trunk_height": jnp.asarray(0.30 - 0.25 * drive[:, None] * (t / HORIZON)),
            "up_proj": jnp.asarray(1.0 - 0.9 * drive[:, None] * (t / HORIZON)),
            "lin_vel": jnp.zeros((n, HORIZON, 3)),
            "ang_vel": jnp.zeros((n, HORIZON, 3)),
            "command": jnp.zeros((n, HORIZON, 3)),
        }

    return evaluate


def test_one_batch_per_generation_covers_every_restart(space, cfg):
    """The efficiency claim: restarts share ONE evaluate call per generation."""
    calls = {"n": 0, "rows": []}
    inner = _synthetic_evaluate(space)

    def counting_evaluate(thetas, key):
        calls["n"] += 1
        calls["rows"].append(int(np.asarray(thetas).shape[0]))
        return inner(thetas, key)

    restarts = 4
    recs = multi_cmaes_search(counting_evaluate, space, cfg, jax.random.PRNGKey(0),
                              restarts=restarts)

    assert calls["n"] == cfg.generations           # not generations * restarts
    assert set(calls["rows"]) == {restarts * cfg.popsize}
    assert len(recs) == cfg.generations * restarts * cfg.popsize


def test_records_carry_the_restart_index_and_are_unique(space, cfg):
    restarts = 3
    recs = multi_cmaes_search(_synthetic_evaluate(space), space, cfg,
                              jax.random.PRNGKey(0), restarts=restarts)
    assert len({r.id for r in recs}) == len(recs)
    tags = {r.id.split("-")[1] for r in recs}
    assert tags == {f"r{i}" for i in range(restarts)}
    assert all(np.isfinite(r.failure_score) for r in recs)


def test_search_actually_climbs_the_synthetic_objective(space, cfg):
    """Sanity: a broken tell() would leave the population at its random start.

    Asserted on the MEAN, not the max: this objective is one-dimensional, so a
    32-sample first generation already contains a near-ceiling sample and the max has
    nowhere to climb. CMA-ES's actual job is to concentrate the whole population,
    which is what the mean measures.
    """
    c = replace(cfg, generations=5)
    recs = multi_cmaes_search(_synthetic_evaluate(space), space, c,
                              jax.random.PRNGKey(0), restarts=4)
    scores = np.array([r.failure_score for r in recs])
    per_gen_mean = scores.reshape(c.generations, -1).mean(axis=1)
    assert per_gen_mean[-1] > per_gen_mean[0], (
        f"population did not concentrate: {per_gen_mean}"
    )


def test_novelty_is_on_by_default_at_the_benchmarked_radius():
    """Default chosen by the pre-registered benchmark (TECHNICAL_REPORT.md section 9.1)."""
    import inspect

    from breakpoint_mjx.multi_search import multi_cmaes_search

    p = inspect.signature(multi_cmaes_search).parameters
    assert p["novelty"].default is True
    assert p["novelty_radius"].default == 1.2 and p["novelty_strength"].default == 0.8
