"""Config loading, regime boxes and the Sobol baseline (no simulator needed)."""
from pathlib import Path

import numpy as np
import pytest

from benchmark.baselines.samplers import sobol_points
from benchmark.evaluation.setup import load_config, regime_space
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def go1_space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


def test_conservative_regime_uses_plausible_bands(go1_space):
    c = regime_space(go1_space, "conservative")
    for n, d in go1_space.dims.items():
        assert c.dims[n].bounds == d.plausible
        lo, hi = d.plausible
        assert lo <= c.dims[n].nominal <= hi
        assert d.bounds[0] <= lo and hi <= d.bounds[1]     # inside the stress box


def test_broad_regime_is_unchanged(go1_space):
    assert regime_space(go1_space, "broad") is go1_space


@pytest.mark.parametrize("name", ["smoke", "pilot", "full"])
def test_shipped_configs_load_with_equal_budgets(name):
    bc = load_config(CONFIGS / f"{name}.yaml")
    assert bc.plan.budget % bc.plan.batch == 0
    assert bc.repetitions >= 1 and bc.validation_n >= 1


def test_config_rejects_non_divisible_budget(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\npolicies: []\nrepetitions: 1\nhorizon: 10\n"
                 "search_plan: {budget: 101, restarts: 2, popsize: 5}\n")
    with pytest.raises(ValueError, match="multiple"):
        load_config(p)


def test_sobol_in_box_deterministic_and_seed_dependent(go1_space):
    a = sobol_points(go1_space, 100, seed=0)
    b = sobol_points(go1_space, 100, seed=0)
    c = sobol_points(go1_space, 100, seed=1)
    lo, hi = np.asarray(go1_space.lower()), np.asarray(go1_space.upper())
    assert a.shape == (100, len(lo))
    assert np.all(a >= lo) and np.all(a <= hi)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_near_nominal_regime_shrinks_toward_nominal(go1_space):
    n = regime_space(go1_space, "near_nominal")
    for k, d in go1_space.dims.items():
        lo, hi = n.dims[k].bounds
        plo, phi = d.plausible
        nom = n.dims[k].nominal
        assert plo <= lo <= nom <= hi <= phi + 1e-6
        assert (hi - lo) <= 0.25 * (phi - plo) + 1e-6


def test_config_rejects_unknown_gate_budget(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\npolicies: []\nrepetitions: 1\nhorizon: 10\n"
                 "search_plan: {budget: 100, restarts: 2, popsize: 5}\n"
                 "regression: {budget: full}\n")
    with pytest.raises(ValueError, match="gate accepts"):
        load_config(p)


def test_gate_budgets_match_engine():
    from benchmark.evaluation.setup import GATE_BUDGETS
    from breakpoint_mjx.gate import BUDGETS
    assert set(GATE_BUDGETS) == set(BUDGETS)
