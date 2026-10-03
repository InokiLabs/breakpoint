"""Real-env proof that threading params as an ARGUMENT removes the per-policy recompile.

Opt-in (BP_RUN_EVAL_SMOKE=1) because it loads the Go1 MJX env; the CPU stack in
.venv-cpu runs it in seconds at a tiny horizon. What is measured is COMPILE COUNT via
wall-clock (first call compiles, later calls do not), which is hardware-independent:
the same structure on GPU is the difference between one multi-minute compile and one
per policy version.
"""
import os
import time

import numpy as np
import pytest

RUN = os.environ.get("BP_RUN_EVAL_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="set BP_RUN_EVAL_SMOKE=1 to run")

HORIZON = 6
N = 4


def _weights(in_dim, act_dim, seed):
    """Random MLP weights as plain numpy [(W, b), ...]."""
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(0, 0.05, (in_dim, 32)), np.zeros(32)),
        (rng.normal(0, 0.05, (32, act_dim)), np.zeros(act_dim)),
    ]


def _make_policy(params):
    """params -> policy_fn, trace-safe.

    NOTE: make_policy runs INSIDE the jit trace, so it must be pure jnp. The
    adapters' MLPPolicy validates its weights with numpy at construction and
    therefore cannot be built here -- build it outside and pass its .policy_fn to
    make_evaluator, or use a forward like this one for the params-as-argument path.
    """
    import jax.numpy as jnp

    def policy_fn(obs, key=None):
        x = obs["state"] if isinstance(obs, dict) else obs
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            if i < len(params) - 1:
                x = jnp.tanh(x)
        return x, {}

    return policy_fn


@pytest.fixture  # function-scoped ON PURPOSE: a jitted evaluator patches env._mjx_model
def rig():        # under vmap, so sharing one env across evaluators leaks tracers
    from breakpoint_mjx.config import load_theta_space
    from breakpoint_mjx.targets import get_target

    target = get_target("go1")
    env = target.load_env()
    consts = target.resolve_constants(env)
    space = load_theta_space(target.theta_space_path)
    obs_dim = int(env.observation_size["state"][0])
    return target, env, consts, space, obs_dim


def test_param_evaluator_compiles_once_across_policy_versions(rig):
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.evaluator import make_param_evaluator

    target, env, consts, space, obs_dim = rig
    stacked = [
        [(jnp.asarray(W), jnp.asarray(b)) for (W, b) in _weights(obs_dim, env.action_size, s)]
        for s in (0, 1, 2)
    ]

    evaluate = make_param_evaluator(env, _make_policy, space, HORIZON, target, consts)
    thetas = jnp.asarray(np.tile(space.nominal_vector(), (N, 1)))
    key = jax.random.PRNGKey(0)

    t0 = time.perf_counter()
    out0 = evaluate(stacked[0], thetas, key)
    jax.block_until_ready(out0)
    t_first = time.perf_counter() - t0

    later = []
    for p in stacked[1:]:
        t0 = time.perf_counter()
        jax.block_until_ready(evaluate(p, thetas, key))
        later.append(time.perf_counter() - t0)

    # A recompile costs the same order as the first call; a cache hit is far cheaper.
    assert max(later) < t_first / 3, (
        f"a later policy version appears to have recompiled: first={t_first:.2f}s "
        f"later={later}"
    )
    assert np.isfinite(np.asarray(out0["trunk_height"])).all()


def test_param_evaluator_matches_closure_evaluator(rig):
    """Same weights, same key, same thetas -> identical trajectories.

    The optimization is only worth having if it is numerically a no-op.
    """
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.evaluator import make_evaluator, make_param_evaluator

    target, env, consts, space, obs_dim = rig
    params = [(jnp.asarray(W), jnp.asarray(b))
              for (W, b) in _weights(obs_dim, env.action_size, 7)]

    thetas = jnp.asarray(np.tile(space.nominal_vector(), (N, 1)))
    key = jax.random.PRNGKey(3)

    # Same weights, one baked into a closure, one passed as an argument.
    closure_eval = make_evaluator(env, _make_policy(params), space, HORIZON, target, consts)
    param_eval = make_param_evaluator(env, _make_policy, space, HORIZON, target, consts)

    a = closure_eval(thetas, key)
    b = param_eval(params, thetas, key)
    for k in a:
        assert np.allclose(np.asarray(a[k]), np.asarray(b[k]), rtol=1e-6, atol=1e-6), k
