"""Pure tests for llm_search helpers (no LLM calls, no JAX).

Uses the real configs/go1/theta_space.yaml.  All tests are deterministic / offline.
"""
from pathlib import Path
import json
import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.llm_search import _build_batch, _parse_configs
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


# ---------------------------------------------------------------------------
# _parse_configs
# ---------------------------------------------------------------------------

def test_parse_configs_clips_and_fills(space):
    """Two configs: one out-of-bounds friction_floor, one missing kp_scale.
    Both should be returned as complete dicts with clipped / nominal values."""
    # friction_floor bounds are [0.15, 1.20]; supply a value above the upper bound.
    friction_upper = space.dims["friction_floor"].bounds[1]   # 1.20
    kp_nominal = space.dims["kp_scale"].nominal               # 1.0

    # Build a prose string with a JSON array embedded in it.
    cfg1 = {name: d.nominal for name, d in space.dims.items()}
    cfg1["friction_floor"] = friction_upper + 5.0   # out-of-bounds high (1.20 + 5.0 = 6.20)

    cfg2 = {name: d.nominal for name, d in space.dims.items()}
    del cfg2["kp_scale"]   # missing key -> should fill with nominal

    text = f"Here are my proposals:\n{json.dumps([cfg1, cfg2])}\nEnjoy!"
    result = _parse_configs(text, space)

    assert len(result) == 2, f"Expected 2 configs, got {len(result)}"

    # Every dim must be present in both
    for cfg in result:
        assert set(cfg.keys()) == set(space.names()), \
            f"Config is missing dims: {set(space.names()) - set(cfg.keys())}"

    # Config 1: friction_floor clipped to upper bound
    assert result[0]["friction_floor"] == pytest.approx(friction_upper), (
        f"friction_floor should be clipped to {friction_upper}, got {result[0]['friction_floor']}"
    )

    # Config 2: kp_scale filled with nominal
    assert result[1]["kp_scale"] == pytest.approx(kp_nominal), (
        f"Missing kp_scale should be nominal {kp_nominal}, got {result[1]['kp_scale']}"
    )

    # Bounds satisfied for every value in both configs
    for cfg in result:
        for name, d in space.dims.items():
            assert d.bounds[0] <= cfg[name] <= d.bounds[1], (
                f"{name}={cfg[name]} out of bounds {d.bounds}"
            )


def test_parse_configs_garbage_returns_empty(space):
    """Garbage text with no JSON array returns an empty list."""
    assert _parse_configs("no json here at all", space) == []
    assert _parse_configs("", space) == []
    assert _parse_configs(None, space) == []
    assert _parse_configs("{not an array}", space) == []


# ---------------------------------------------------------------------------
# _build_batch
# ---------------------------------------------------------------------------

def _make_proposals(space, n):
    """Build n distinct proposal dicts (all-nominal but with a distinguishing tweak)."""
    dims = list(space.dims.items())
    proposals = []
    for i in range(n):
        d_name, d = dims[i % len(dims)]
        cfg = {name: dim.nominal for name, dim in space.dims.items()}
        # Nudge one dim so proposals are distinct (stay within bounds)
        mid = (d.bounds[0] + d.bounds[1]) / 2.0
        cfg[d_name] = mid
        proposals.append(cfg)
    return proposals


def test_build_batch_shape_and_bounds(space):
    """4 proposals, n=64 -> shape (64, D), all rows within bounds."""
    D = len(space.dims)
    proposals = _make_proposals(space, 4)
    rng = np.random.default_rng(0)
    batch = _build_batch(space, proposals, 64, rng)

    assert batch.shape == (64, D), f"Expected (64, {D}), got {batch.shape}"

    lo = np.array(space.lower())
    hi = np.array(space.upper())
    assert np.all(batch >= lo - 1e-9), "Some rows below lower bound"
    assert np.all(batch <= hi + 1e-9), "Some rows above upper bound"


def test_build_batch_first_row_unjitered(space):
    """The first row of each proposal block must equal the un-jittered proposal vector."""
    D = len(space.dims)
    n_props = 4
    n = 64
    proposals = _make_proposals(space, n_props)
    rng = np.random.default_rng(0)
    batch = _build_batch(space, proposals, n, rng)

    per = max(1, n // n_props)   # = 16 for n=64, n_props=4
    for i, cfg in enumerate(proposals):
        expected = np.array([cfg[name] for name in space.names()], dtype=float)
        actual = batch[i * per]
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"Proposal {i}: first row of its block is not the un-jittered vector"
        )


def test_build_batch_empty_proposals(space):
    """Empty proposals list -> (32, D) of uniform random rows within bounds."""
    D = len(space.dims)
    rng = np.random.default_rng(0)
    batch = _build_batch(space, [], 32, rng)

    assert batch.shape == (32, D), f"Expected (32, {D}), got {batch.shape}"

    lo = np.array(space.lower())
    hi = np.array(space.upper())
    assert np.all(batch >= lo - 1e-9), "Some rows below lower bound"
    assert np.all(batch <= hi + 1e-9), "Some rows above upper bound"
