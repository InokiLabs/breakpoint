"""CPU tests for the noise-robust multi-restart CMA-ES (synthetic evaluator, no physics)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("evosax")

from dataclasses import replace  # noqa: E402

from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402
from breakpoint_mjx.robust_search import robust_multi_cmaes_search  # noqa: E402

HORIZON = 6


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def cfg():
    c = load_search_config(config_path("search.yaml"))
    return replace(c, popsize=6, horizon=HORIZON, generations=3, seed=0)


def _evaluate(space, calls=None):
    """Latency drives a fall deterministically; every other dim is inert."""
    import jax.numpy as jnp

    j = space.names().index("latency_steps")
    lo, hi = np.array(space.lower()), np.array(space.upper())

    def evaluate(thetas, key):
        th = np.asarray(thetas)
        if calls is not None:
            calls.append(th.copy())
        drive = (th[:, j] - lo[j]) / (hi[j] - lo[j])
        n = th.shape[0]
        t = np.arange(HORIZON)[None, :]
        return {"trunk_height": jnp.asarray(0.30 - 0.25 * drive[:, None] * (t / HORIZON)),
                "up_proj": jnp.asarray(1.0 - 0.9 * drive[:, None] * (t / HORIZON)),
                "lin_vel": jnp.zeros((n, HORIZON, 3)), "ang_vel": jnp.zeros((n, HORIZON, 3)),
                "command": jnp.zeros((n, HORIZON, 3))}

    return evaluate


def test_spends_exactly_the_budget_in_one_call_per_generation(space, cfg):
    calls = []
    X, S = robust_multi_cmaes_search(_evaluate(space, calls), space, cfg, restarts=2,
                                     seeds_per_theta=3)
    assert len(calls) == cfg.generations
    assert {c.shape[0] for c in calls} == {2 * cfg.popsize * 3}
    assert X.shape == (cfg.generations * 2 * cfg.popsize * 3, len(space.names()))
    assert S.shape == (X.shape[0],)


def test_each_condition_is_repeated_contiguously(space, cfg):
    X, _ = robust_multi_cmaes_search(_evaluate(space), space, cfg, restarts=2,
                                     seeds_per_theta=4)
    blocks = X.reshape(-1, 4, X.shape[1])
    assert np.allclose(blocks, blocks[:, :1, :])          # same theta within a block
    assert not np.allclose(blocks[0, 0], blocks[1, 0])     # different between blocks


def test_rows_stay_inside_the_box(space, cfg):
    X, _ = robust_multi_cmaes_search(_evaluate(space), space, cfg, restarts=2,
                                     seeds_per_theta=2, novelty=True)
    assert np.all(X >= np.array(space.lower()) - 1e-6)
    assert np.all(X <= np.array(space.upper()) + 1e-6)


def test_deterministic_for_a_seed(space, cfg):
    a, _ = robust_multi_cmaes_search(_evaluate(space), space, cfg, restarts=2, seeds_per_theta=2)
    b, _ = robust_multi_cmaes_search(_evaluate(space), space, cfg, restarts=2, seeds_per_theta=2)
    assert np.allclose(a, b)


def test_rejects_zero_seeds(space, cfg):
    with pytest.raises(ValueError):
        robust_multi_cmaes_search(_evaluate(space), space, cfg, restarts=1, seeds_per_theta=0)
