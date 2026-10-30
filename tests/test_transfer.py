"""Unit tests for breakpoint_mjx.transfer -- pure, no JAX/GPU required."""
from pathlib import Path

import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.transfer import (
    EXTREME_AXIS_FACTOR,
    joint_plausibility,
    transfer_likelihood,
    severity_from_score,
    priority_key,
)

from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def nominal_named(space):
    return space.vector_to_named(space.nominal_vector())


# ---------------------------------------------------------------------------
# transfer_likelihood: nominal vector -> 1.0
# ---------------------------------------------------------------------------

def test_nominal_is_fully_transferable(space, nominal_named):
    tl = transfer_likelihood(nominal_named, space)
    assert tl == 1.0, f"Expected 1.0 for all-nominal theta, got {tl}"


# ---------------------------------------------------------------------------
# transfer_likelihood: plausible > extreme
# ---------------------------------------------------------------------------

def test_plausible_beats_extreme(space, nominal_named):
    """friction_floor=0.8 is inside plausible [0.40, 1.00]; friction_floor=0.15 is at the
    hard lower bound, entirely outside plausible -> lower transfer."""
    plausible_named = dict(nominal_named)
    plausible_named["friction_floor"] = 0.8

    extreme_named = dict(nominal_named)
    extreme_named["friction_floor"] = 0.15  # at the bound edge, outside plausible

    tl_plausible = transfer_likelihood(plausible_named, space)
    tl_extreme = transfer_likelihood(extreme_named, space)

    assert tl_plausible > tl_extreme, (
        f"Plausible friction (tl={tl_plausible:.4f}) should exceed extreme "
        f"friction (tl={tl_extreme:.4f})"
    )


# ---------------------------------------------------------------------------
# joint_plausibility: compounding extremes cost geometrically, not 1/n_dims each
# ---------------------------------------------------------------------------

def _with(nominal_named, **kw):
    out = dict(nominal_named)
    out.update(kw)
    return out


def test_joint_nominal_and_in_band_are_one(space, nominal_named):
    assert joint_plausibility(nominal_named, space) == 1.0
    assert joint_plausibility(_with(nominal_named, friction_floor=0.45, latency_steps=2.0),
                              space) == 1.0


def test_joint_each_axis_at_bound_multiplies(space, nominal_named):
    f = EXTREME_AXIS_FACTOR
    ice = _with(nominal_named, friction_floor=0.15)
    corner3 = _with(ice, torso_mass_add_kg=6.0, latency_steps=4.0)
    corner5 = _with(corner3, obs_dropout_prob=0.5, push_mag=2.0)
    assert joint_plausibility(ice, space) == pytest.approx(f)
    assert joint_plausibility(corner3, space) == pytest.approx(f ** 3)
    assert joint_plausibility(corner5, space) == pytest.approx(f ** 5)
    # the mean barely notices the same corner, which is why the search no longer uses it
    assert transfer_likelihood(corner5, space) > 0.6


def test_joint_partial_excess_costs_proportionally(space, nominal_named):
    # friction 0.275 is halfway from the plausible edge 0.40 to the bound 0.15
    half = joint_plausibility(_with(nominal_named, friction_floor=0.275), space)
    assert half == pytest.approx(1.0 - 0.5 * (1.0 - EXTREME_AXIS_FACTOR))
    assert EXTREME_AXIS_FACTOR < half < 1.0


# ---------------------------------------------------------------------------
# severity_from_score
# ---------------------------------------------------------------------------

def test_severity_critical():
    assert severity_from_score(0.95) == "critical"


def test_severity_high():
    assert severity_from_score(0.80) == "high"


def test_severity_medium():
    assert severity_from_score(0.50) == "medium"


def test_severity_low():
    assert severity_from_score(0.20) == "low"


# ---------------------------------------------------------------------------
# priority_key ordering
# ---------------------------------------------------------------------------

def test_priority_critical_beats_low():
    high_pri = priority_key("critical", 0.9, 0.9)
    low_pri = priority_key("low", 0.2, 0.5)
    assert high_pri > low_pri, (
        f"critical/high priority key ({high_pri:.4f}) should exceed "
        f"low priority key ({low_pri:.4f})"
    )
