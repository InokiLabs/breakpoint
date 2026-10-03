"""Unit tests for breakpoint_mjx.transfer -- pure, no JAX/GPU required."""
from pathlib import Path

import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.transfer import (
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
