"""CPU tests for the learned failure surrogate (M1). Pure numpy, no jax."""
import numpy as np
import pytest

from breakpoint_mjx.surrogate import Surrogate


class _Space:
    """Minimal ThetaSpace stand-in (names/lower/upper) for a 2-D toy problem."""
    def __init__(self):
        self._names = ["friction_floor", "latency_steps"]
    def names(self):
        return self._names
    def lower(self):
        return [0.1, 0.0]
    def upper(self):
        return [1.0, 4.0]


def _toy_data(n=1200, seed=0):
    """A learnable failure function: fails (high score) when friction < 0.3 OR latency > 3."""
    rng = np.random.default_rng(seed)
    friction = rng.uniform(0.1, 1.0, n)
    latency = rng.uniform(0.0, 4.0, n)
    score = (
        0.9 * (friction < 0.3).astype(float)
        + 0.7 * (latency > 3.0).astype(float)
        + rng.normal(0, 0.03, n)
    )
    return np.stack([friction, latency], axis=1), np.clip(score, 0, 1.6)


def test_surrogate_learns_the_failure_region():
    space = _Space()
    X, y = _toy_data()
    sur = Surrogate(space.names(), space.lower(), space.upper(), hidden=(48, 48))
    sur.fit(X, y, epochs=300, seed=1)
    assert sur.r2(X, y) > 0.7  # explains most variance on a learnable function

    # low friction -> high predicted failure; safe corner -> low
    hot = float(sur.predict_score([[0.15, 1.0]])[0])
    safe = float(sur.predict_score([[0.8, 1.0]])[0])
    assert hot > safe + 0.3
    # probability is bounded and ordered the same way
    assert 0.0 <= float(sur.predict_proba([[0.8, 1.0]])[0]) <= 1.0
    assert sur.predict_proba([[0.15, 1.0]])[0] > sur.predict_proba([[0.8, 1.0]])[0]


def test_raw_theta_is_normalized_internally():
    space = _Space()
    sur = Surrogate(space.names(), space.lower(), space.upper())
    # out-of-range theta is clipped, not extrapolated to absurdity
    sur.fit(*_toy_data(400), epochs=50)
    a = sur.predict_score([[5.0, 99.0]])  # far above bounds -> clipped to the hi corner
    b = sur.predict_score([[1.0, 4.0]])   # the hi corner
    assert np.allclose(a, b, atol=1e-9)


def test_json_round_trip_is_bit_identical():
    space = _Space()
    sur = Surrogate(space.names(), space.lower(), space.upper()).fit(*_toy_data(400), epochs=40)
    probe = np.array([[0.2, 3.5], [0.9, 0.2], [0.5, 2.0]])
    before = sur.predict_score(probe)
    restored = Surrogate.from_json(sur.to_json())
    after = restored.predict_score(probe)
    assert np.allclose(before, after, atol=1e-12)
    assert restored.dim_names == sur.dim_names


def test_serialized_format_is_portable_plain_json():
    space = _Space()
    d = Surrogate(space.names(), space.lower(), space.upper()).fit(*_toy_data(200), epochs=20).to_dict()
    assert d["format"] == "breakpoint-surrogate-1"
    # only JSON-native types (so the browser can read it)
    import json
    text = json.dumps(d)
    assert isinstance(json.loads(text)["weights"][0][0][0], float)


# --------------------------------------------------------------------------- #
# Honest fit reporting: cross-validated R2, hidden when the labels cannot support it
# --------------------------------------------------------------------------- #
class _Rec:
    def __init__(self, theta, score):
        self.theta_vector = list(theta)
        self.failure_score = float(score)


def test_r2_hidden_when_too_few_or_one_sided():
    from breakpoint_mjx.surrogate import r2_report

    space = _Space()
    rng = np.random.default_rng(0)
    # 192 worlds, 179 failures: the smoke-demo shape the graders saw R2=1.000 on.
    recs = [_Rec(rng.random(2), 0.9 if i < 179 else 0.1) for i in range(192)]
    out = r2_report(recs, space, 0.5)
    assert out["r2"] is None and "not shown" in out["hidden_reason"]
    assert (out["n"], out["n_fail"], out["n_pass"]) == (192, 179, 13)
    # enough worlds but a minority class of 10 is still hidden
    recs = [_Rec(rng.random(2), 0.9 if i < 390 else 0.1) for i in range(400)]
    assert r2_report(recs, space, 0.5)["r2"] is None


def test_r2_is_cross_validated_not_in_sample():
    from breakpoint_mjx.surrogate import cv_r2, r2_report

    space = _Space()
    rng = np.random.default_rng(1)
    X = rng.random((300, 2)) * np.array([1.0, 4.0])
    y = rng.random(300)                                  # pure noise: nothing to learn
    sur = Surrogate(space.names(), space.lower(), space.upper()).fit(X, y, epochs=300)
    in_sample = sur.r2(X, y)
    cv = cv_r2(X, y, space.lower(), space.upper(), space.names(), epochs=300)
    assert cv < in_sample - 0.1 and cv < 0.2             # memorization does not survive CV
    recs = [_Rec(x, s) for x, s in zip(X, y)]
    out = r2_report(recs, space, 0.5, epochs=60)
    assert out["method"] == "5-fold cross-validated" and out["r2"] is not None
    assert out["hidden_reason"] is None
