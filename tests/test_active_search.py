"""CPU unit tests for the surrogate-guided active search -- the PURE proposal logic only.

`propose_batch` is the tested core (cold start, exploit/explore split, bounds, dtype,
determinism, blind-spot escape hatch). The outer `active_search` loop imports jax and is
exercised by the GPU smoke test on Colab (tests/test_gpu_smoke.py). The acquisition
scoring itself (boundary x uncertainty + diversity) is covered by
tests/test_acquisition_envelope.py; here we test the WIRING around it.

Ensemble training is real (pure numpy) but shrunk via fit_kw for speed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from breakpoint_mjx.active_search import (
    MIN_RECORDS_TO_FIT,
    propose_batch,
    transferable_seed_records,
)
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]

# Tiny fit budget: the tests assert wiring/shape/determinism, not surrogate quality.
FIT = {"epochs": 20, "batch": 64}
KW = dict(explore_frac=0.25, pool=512, ensemble_k=2, threshold=0.5, fit_kw=FIT)


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


def _labeled(space, n, seed=0):
    """Fake labeled records: failure iff floor friction is low (a learnable 1-dim rule).
    propose_batch only reads .theta_vector / .failure_score, so SimpleNamespace suffices."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(space.lower())
    hi = np.asarray(space.upper())
    j = space.names().index("friction_floor")
    recs = []
    for _ in range(n):
        v = lo + rng.random(len(lo)) * (hi - lo)
        score = 1.0 if v[j] < 0.5 else 0.05
        recs.append(SimpleNamespace(theta_vector=[float(x) for x in v],
                                    failure_score=float(score)))
    return recs


def test_cold_start_is_uniform_explore(space):
    rng = np.random.default_rng(0)
    batch, info = propose_batch([], space, 64, rng, **KW)
    assert info["mode"] == "explore" and info["r2"] is None and info["n_exploit"] == 0
    assert batch.shape == (64, len(space.names()))
    assert batch.dtype == np.float32


def test_cold_start_threshold_is_min_records(space):
    rng = np.random.default_rng(0)
    few = _labeled(space, MIN_RECORDS_TO_FIT - 1)
    _, info = propose_batch(few, space, 16, rng, **KW)
    assert info["mode"] == "explore"
    enough = _labeled(space, MIN_RECORDS_TO_FIT)
    _, info2 = propose_batch(enough, space, 16, rng, **KW)
    assert info2["mode"] == "acquire"


def test_acquire_shape_bounds_and_split(space):
    rng = np.random.default_rng(1)
    recs = _labeled(space, 80)
    n = 32
    batch, info = propose_batch(recs, space, n, rng, **KW)
    assert batch.shape == (n, len(space.names()))
    assert batch.dtype == np.float32
    lo = np.asarray(space.lower())
    hi = np.asarray(space.upper())
    assert (batch >= lo - 1e-5).all() and (batch <= hi + 1e-5).all()
    # explore_frac=0.25 of 32 -> 8 explore rows, 24 exploit rows.
    assert info["n_exploit"] == 24
    assert info["r2"] is not None


def test_explore_frac_one_is_pure_uniform_even_with_records(space):
    """The blind-spot escape hatch: explore_frac=1.0 must yield zero exploit rows
    (no ensemble influence on the batch) even when plenty of records exist."""
    rng = np.random.default_rng(2)
    recs = _labeled(space, 80)
    kw = dict(KW)
    kw["explore_frac"] = 1.0
    batch, info = propose_batch(recs, space, 16, rng, **kw)
    assert info["n_exploit"] == 0
    assert batch.shape == (16, len(space.names()))


def test_deterministic_given_same_generator_state(space):
    recs = _labeled(space, 60)
    a, _ = propose_batch(recs, space, 24, np.random.default_rng(7), **KW)
    b, _ = propose_batch(recs, space, 24, np.random.default_rng(7), **KW)
    np.testing.assert_array_equal(a, b)
    c, _ = propose_batch(recs, space, 24, np.random.default_rng(8), **KW)
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------- #
# transferable_seed_records (cross-policy warm-start, BP_PRIOR)
# --------------------------------------------------------------------------- #
def _prior(space, overrides=None, score=0.7, shuffle_names=False, drop_dim=None,
           named=True, vec=None):
    names = list(space.names())
    tn = {n: space.dims[n].nominal for n in names}
    tn.update(overrides or {})
    if drop_dim:
        tn.pop(drop_dim)
    if shuffle_names:  # a prior YAML that listed dims in another order
        items = sorted(tn.items(), key=lambda kv: kv[0])
        tn = dict(items)
    return SimpleNamespace(
        theta_named=tn if named else None,
        theta_vector=vec if vec is not None else [tn.get(n, 0.0) for n in names],
        failure_score=score,
    )


def test_transfer_maps_by_name_regardless_of_prior_order(space):
    r = _prior(space, {"friction_floor": 0.3, "push_mag": 1.5}, shuffle_names=True)
    out = transferable_seed_records([r], space)
    assert len(out) == 1
    j = space.names().index("friction_floor")
    k = space.names().index("push_mag")
    assert out[0].theta_vector[j] == pytest.approx(0.3)
    assert out[0].theta_vector[k] == pytest.approx(1.5)
    assert out[0].failure_score == pytest.approx(0.7)


def test_transfer_clips_out_of_bounds_prior_values(space):
    # A prior version searched wider bounds: friction below the current lower bound.
    r = _prior(space, {"friction_floor": 0.01})
    out = transferable_seed_records([r], space)
    j = space.names().index("friction_floor")
    assert out[0].theta_vector[j] == pytest.approx(space.dims["friction_floor"].bounds[0])


def test_transfer_drops_records_missing_a_dim(space):
    ok = _prior(space)
    missing = _prior(space, drop_dim="push_mag")
    out = transferable_seed_records([missing, ok], space)
    assert len(out) == 1  # the incomplete record is dropped, never nominal-filled


def test_transfer_falls_back_to_vector_only_when_length_matches(space):
    D = len(space.names())
    good = _prior(space, named=False, vec=[0.5] * D)
    bad = _prior(space, named=False, vec=[0.5] * (D - 2))  # older, smaller theta space
    out = transferable_seed_records([good, bad], space)
    assert len(out) == 1


def test_transfer_drops_nonfinite_and_caps(space):
    nan_score = _prior(space, score=float("nan"))
    nan_theta = _prior(space, {"friction_floor": float("nan")})
    ok = [_prior(space, score=0.1 * i) for i in range(10)]
    out = transferable_seed_records([nan_score, nan_theta] + ok, space, max_records=4)
    assert len(out) == 4
    # newest-last preserved: the cap keeps the LAST records.
    assert out[-1].failure_score == pytest.approx(0.9)


def test_transfer_empty_input(space):
    assert transferable_seed_records([], space) == []
    assert transferable_seed_records(None, space) == []


def test_transfer_max_records_cap_boundaries(space):
    # BOUNDARIES: cap of 0 must keep NOTHING (regression: out[-0:] used to return ALL);
    # a cap larger than the input keeps everything; a negative cap keeps nothing.
    recs = [_prior(space, score=0.1 * i) for i in range(5)]
    assert transferable_seed_records(recs, space, max_records=0) == []
    assert len(transferable_seed_records(recs, space, max_records=1)) == 1
    assert len(transferable_seed_records(recs, space, max_records=99)) == 5
    assert transferable_seed_records(recs, space, max_records=-3) == []


def test_exploit_rows_concentrate_near_learned_boundary(space):
    """With a strongly learnable rule (fail iff friction < 0.5) and a real fit budget,
    the exploit rows' friction should sit MUCH closer to the 0.5 boundary than uniform
    draws over [0.15, 1.20] would (uniform mean distance ~0.28)."""
    rng = np.random.default_rng(3)
    recs = _labeled(space, 400, seed=5)
    kw = dict(KW)
    kw["explore_frac"] = 0.0        # all exploit, isolate the acquisition behavior
    kw["fit_kw"] = {"epochs": 120, "batch": 128}
    batch, info = propose_batch(recs, space, 32, rng, **kw)
    assert info["mode"] == "acquire" and info["n_exploit"] == 32
    j = space.names().index("friction_floor")
    mean_dist = float(np.abs(batch[:, j] - 0.5).mean())
    assert mean_dist < 0.20, f"exploit rows not boundary-focused (mean |f-0.5|={mean_dist:.3f})"
