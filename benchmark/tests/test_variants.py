"""Directed-search variants: config validation and boundary-battery selection."""
import json

import numpy as np
import pytest

from benchmark.evaluation.setup import load_config, variant_plan, SearchPlan

BASE = ("name: v\npolicies: []\nrepetitions: 2\nhorizon: 10\nrep_offset: 100\n"
        "search_plan: {budget: 1000, restarts: 5, popsize: 20}\n")


def _cfg(tmp_path, extra):
    p = tmp_path / "c.yaml"
    p.write_text(BASE + extra)
    return load_config(p)


def test_variant_is_a_new_named_method_and_reps_are_offset(tmp_path):
    bc = _cfg(tmp_path, "methods: [directed, nov]\n"
                        "method_variants: {nov: {kind: novelty, radius: 0.6, label: N}}\n")
    assert bc.methods == ["directed", "nov"]
    assert list(bc.reps()) == [100, 101]
    assert bc.label("nov") == "N" and bc.label("directed") == "BreakPoint directed"


def test_variant_cannot_shadow_a_builtin(tmp_path):
    with pytest.raises(ValueError, match="shadows"):
        _cfg(tmp_path, "methods: [directed]\nmethod_variants: {directed: {kind: novelty}}\n")


def test_variant_must_spend_the_shared_budget_exactly(tmp_path):
    with pytest.raises(ValueError, match="divide the budget"):
        _cfg(tmp_path, "methods: [r]\n"
                       "method_variants: {r: {kind: robust, restarts: 3, popsize: 7, "
                       "seeds_per_theta: 4}}\n")


def test_variant_plan_accounts_for_seeds_per_theta():
    plan = SearchPlan(1000, 5, 20)
    assert variant_plan({"kind": "robust", "restarts": 5, "popsize": 5,
                         "seeds_per_theta": 4}, plan) == (5, 5, 4, 10)
    assert variant_plan({"kind": "novelty"}, plan) == (5, 20, 1, 10)
    # seeds_per_theta is ignored outside the robust kind
    assert variant_plan({"kind": "restarts", "restarts": 10, "popsize": 10,
                         "seeds_per_theta": 4}, plan) == (10, 10, 1, 10)


def test_boundary_battery_selects_only_baseline_conditions_in_range(tmp_path):
    from benchmark.run_benchmark import boundary_thetas

    runs = tmp_path / "runs"
    runs.mkdir()
    theta = np.arange(12, dtype=float).reshape(4, 3)
    ps = {"0": 0.05, "1": 0.3, "2": 0.6, "3": 0.95}
    rec = {"rows": {"theta": theta.tolist()},
           "validation": {"results": {k: {"p": v} for k, v in ps.items()}}}
    (runs / "pol__conservative__uniform__rep0.json").write_text(json.dumps(rec))
    (runs / "other__conservative__uniform__rep0.json").write_text(json.dumps(rec))
    out = boundary_thetas(tmp_path, "pol", ["conservative"], [0.1, 0.6], cap=10)
    assert np.allclose(out, theta[[1, 2]])
    assert boundary_thetas(tmp_path, "pol", ["broad"], [0.1, 0.6], cap=10).size == 0
