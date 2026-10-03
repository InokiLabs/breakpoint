"""End-to-end test of the agentic search loop with a synthetic evaluator.

Real jax, real config, real record construction; the physics and the LLM are the two
injected seams. The synthetic world has a KNOWN answer -- only latency and pushes
break the "policy" -- so the test can assert the coordinator actually finds it and
moves budget toward the timing role, which is the whole claim of the design.
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from breakpoint_mjx.agentic_search import ROLES, agentic_search  # noqa: E402
from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 20


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def cfg():
    c = load_search_config(config_path("search.yaml"))
    return type(c)(**{**c.__dict__, "popsize": 40, "horizon": HORIZON})


def _fake_evaluate(space, sensitive_dims):
    """A synthetic world: the 'robot' falls when `sensitive_dims` are pushed high.

    Returns trajectories in the legged summary contract, so the real scoring path
    (component_signals / failure_score / rollout_signature) runs unmodified.
    """
    names = space.names()
    idx = [names.index(d) for d in sensitive_dims]
    lo = np.array(space.lower())
    hi = np.array(space.upper())

    def evaluate(thetas, key):
        th = np.asarray(thetas)
        frac = (th - lo) / np.maximum(hi - lo, 1e-9)
        drive = frac[:, idx].mean(axis=1)          # 0..1, high = should fall
        n = th.shape[0]
        t = np.arange(HORIZON)[None, :]
        # Height decays with drive; below 0.18 counts as fallen for the legged gate.
        height = 0.30 - 0.25 * drive[:, None] * (t / HORIZON)
        up = 1.0 - 0.9 * drive[:, None] * (t / HORIZON)
        return {
            "trunk_height": jnp.asarray(height),
            "up_proj": jnp.asarray(up),
            "lin_vel": jnp.zeros((n, HORIZON, 3)),
            "ang_vel": jnp.asarray(np.repeat(drive[:, None, None], HORIZON, 1) * np.ones((1, 1, 3)) * 9.0),
            "command": jnp.zeros((n, HORIZON, 3)),
        }

    return evaluate


def _fake_complete(space):
    """Stand-in LLM: returns valid JSON proposals that push each role's own dials high.

    Deterministic, so the test asserts coordination behaviour rather than model luck.
    """
    import json

    def complete(prompt):
        # Recover which role is asking from its stance text (the prompt is the contract).
        role = next((r for r, (stance, _) in ROLES.items() if stance[:40] in prompt), None)
        dims = ROLES[role][1] if role else ()
        cfgs = []
        for j in range(3):
            c = {}
            for name, d in space.dims.items():
                if name in dims:
                    # push this role's own dials toward their upper bound
                    c[name] = d.bounds[1] - 0.05 * j * (d.bounds[1] - d.bounds[0])
                else:
                    c[name] = d.nominal
            cfgs.append(c)
        return json.dumps(cfgs)

    return complete


def test_agentic_loop_runs_and_labels_records_by_role(space, cfg):
    recs = agentic_search(
        _fake_evaluate(space, ["latency_steps", "push_mag"]),
        space, cfg, jax.random.PRNGKey(0),
        rounds=2, n_proposals=3, robot_name="go1",
        complete=_fake_complete(space),
    )
    assert len(recs) == 2 * cfg.popsize

    roles = {r.id.split("-")[2] for r in recs}
    assert roles == set(ROLES), f"every role must appear in the ids, got {roles}"
    assert all(r.id.startswith("agentic-r") for r in recs)
    # ids must be unique -- benchmark_search splits per-role off them
    assert len({r.id for r in recs}) == len(recs)
    assert all(np.isfinite(r.failure_score) for r in recs)


def test_budget_moves_toward_the_role_that_actually_finds_failures(space, cfg, capsys):
    """The coordinator's claim: fund what works. Timing is the only lethal axis here."""
    agentic_search(
        _fake_evaluate(space, ["latency_steps", "push_mag", "push_step_frac"]),
        space, cfg, jax.random.PRNGKey(0),
        rounds=3, n_proposals=3, robot_name="go1",
        complete=_fake_complete(space),
    )
    out = capsys.readouterr().out
    allocs = [line for line in out.splitlines() if "next_alloc" in line]
    assert len(allocs) == 3

    def alloc_of(line, role):
        seg = line.split("next_alloc[")[1].rstrip("]")
        return float(dict(kv.split("=") for kv in seg.split(", "))[role])

    first, last = allocs[0], allocs[-1]
    assert alloc_of(last, "timing") > alloc_of(first, "timing"), (
        "timing is the only lethal axis; its budget share must grow"
    )
    assert alloc_of(last, "timing") > alloc_of(last, "morphology")
    # ...but exploration never dies: the floor holds for every role.
    for role in ROLES:
        assert alloc_of(last, role) >= 0.10 - 1e-6, f"{role} starved below the floor"


def test_no_api_key_degrades_to_random_without_crashing(space, cfg):
    """A keyless run must still produce a full batch (as random), never a short one."""
    recs = agentic_search(
        _fake_evaluate(space, ["latency_steps"]),
        space, cfg, jax.random.PRNGKey(0),
        rounds=1, n_proposals=3, robot_name="go1",
        complete=lambda prompt: None,      # what _complete returns with no key set
    )
    assert len(recs) == cfg.popsize
    assert len({r.id for r in recs}) == cfg.popsize
