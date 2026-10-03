"""CPU tests for the search-strategy benchmark (scripts/benchmark_search.py) -- pure."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_search as bs  # noqa: E402

from breakpoint_mjx.config import load_theta_space  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


def _rec(space, rid, score, transfer, overrides=None):
    tn = {n: space.dims[n].nominal for n in space.names()}
    tn.update(overrides or {})
    return SimpleNamespace(
        id=rid, failure_score=score, transfer_score=transfer,
        theta_vector=[tn[n] for n in space.names()],
    )


def test_strategy_of_prefixes():
    assert bs._strategy_of("random-3") == "random"
    assert bs._strategy_of("cmaes-40") == "cmaes"
    assert bs._strategy_of("llm-r2-7") == "llm"
    assert bs._strategy_of("active-r0-1") == "active"
    assert bs._strategy_of("weird-1") == "other"
    assert bs._strategy_of(None) == "other"


def test_score_strategy_metrics(space):
    recs = [
        _rec(space, "cmaes-0", 0.2, 0.9),                                   # not a failure
        _rec(space, "cmaes-1", 0.9, 0.9, {"friction_floor": 0.2}),          # plausible fail
        _rec(space, "cmaes-2", 0.9, 0.1, {"latency_steps": 4.0}),           # implausible fail
        _rec(space, "cmaes-3", 0.8, 0.8, {"torso_mass_add_kg": 5.0}),       # plausible fail
    ]
    m = bs.score_strategy(recs, space, threshold=0.5, min_transfer=0.6)
    assert m["evals"] == 4
    assert m["plausible_failures"] == 2       # the 0.1-transfer failure is excluded
    assert m["plausible_yield"] == pytest.approx(0.5)
    assert m["evals_to_first_plausible"] == 2  # cmaes-1 is the 2nd eval
    assert m["modes_found"] == 2               # low friction_floor + high torso_mass
    assert "low friction_floor" in m["modes"] and "high torso_mass_add_kg" in m["modes"]
    assert m["dispersion"] > 0.0
    # top20: mean of the weighted scores (all 4 rows, best 20)
    expect = np.mean(sorted([0.2 * 0.9, 0.9 * 0.9, 0.9 * 0.1, 0.8 * 0.8], reverse=True))
    assert m["top20_weighted"] == pytest.approx(expect, abs=1e-4)


def test_score_strategy_empty(space):
    assert bs.score_strategy([], space, 0.5, 0.6) == {"evals": 0}


def test_benchmark_splits_by_prefix(space):
    runs = {"main": [
        _rec(space, "random-0", 0.9, 0.9, {"friction_floor": 0.2}),
        _rec(space, "random-1", 0.1, 0.9),
        _rec(space, "cmaes-0", 0.9, 0.9, {"torso_mass_add_kg": 5.0}),
    ]}
    out = bs.benchmark(runs, space, 0.5, 0.6)
    assert set(out) == {"main/random", "main/cmaes"}
    assert out["main/random"]["evals"] == 2
    assert out["main/cmaes"]["plausible_yield"] == pytest.approx(1.0)


def test_directed_beats_random_on_engineered_data(space):
    """Sanity: a directed strategy that always finds plausible failures must dominate a
    random strategy that rarely does, on yield and evals-to-first."""
    rng = np.random.default_rng(0)
    rand = [_rec(space, f"random-{i}", 0.9 if rng.random() < 0.1 else 0.2, 0.9,
                 {"friction_floor": 0.3}) for i in range(100)]
    directed = [_rec(space, f"active-r0-{i}", 0.9, 0.9, {"friction_floor": 0.3})
                for i in range(100)]
    out = bs.benchmark({"a": rand, "b": directed}, space, 0.5, 0.6)
    assert out["b/active"]["plausible_yield"] > out["a/random"]["plausible_yield"]
    assert out["b/active"]["evals_to_first_plausible"] <= out["a/random"]["evals_to_first_plausible"]


def test_table_formats(space):
    runs = {"m": [_rec(space, "random-0", 0.9, 0.9, {"friction_floor": 0.2})]}
    out = bs.benchmark(runs, space, 0.5, 0.6)
    txt = bs._fmt_table(out)
    assert "plausible_yield" in txt and "m/random" in txt
