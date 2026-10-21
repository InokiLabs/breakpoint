"""Boundary battery (breakpoint_mjx/boundary.py): sampling, selection, YAML, gate round trip.

All tests use a mocked evaluator; no JAX or simulator is touched. The theta space is the
shipped go1 one (pure YAML).
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from breakpoint_mjx import boundary as B
from breakpoint_mjx import gate as G
from breakpoint_mjx import ratings
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path

SPACE = load_theta_space(config_path("configs/go1/theta_space.yaml"))


@pytest.fixture
def ckpts(tmp_path):
    b = tmp_path / "base.pkl"
    c = tmp_path / "cand.pkl"
    b.write_bytes(b"baseline-weights")
    c.write_bytes(b"candidate-weights")
    return b, c


def fall_rate(theta, worse=0.0):
    """A toy policy: falls more the harder the push; ``worse`` shifts every rate up."""
    return min(1.0, theta.get("push_mag", 0.0) * 0.9 + worse)


def mock_eval(worse_for=None, worse=0.0):
    """Seed s falls when a fixed per-(cell, seed) uniform is below the cell's fall rate,
    so baseline and candidate share their randomness (paired, like the real evaluator)."""
    calls = []

    def fn(spec, cells, seeds, horizon):
        calls.append((str(spec), len(cells), list(seeds), horizon))
        w = worse if worse_for and str(spec).endswith(worse_for) else 0.0
        out = {}
        for c in cells:
            p = fall_rate(c["theta"], w)
            out[c["id"]] = [int(np.random.default_rng([s, hash(c["id"]) % 2**32]).random() < p)
                            for s in seeds]
        return out

    fn.calls = calls
    return fn


def _args(b, out, **kw):
    ns = dict(robot="go1", baseline=str(b), battery=None, out=str(out),
              conditions=B.DEFAULTS["n_conditions"], select_seeds=B.DEFAULTS["select_seeds"],
              p_min=0.1, p_max=0.6, max_cells=B.DEFAULTS["max_cells"],
              seeds=B.DEFAULTS["seeds"], sample_seed=B.DEFAULTS["sample_seed"],
              obs_spec=None, obs_norm=None, squash=None, activation=None,
              action_scale=None, mean_dim=None)
    ns.update(kw)
    return argparse.Namespace(**ns)


# ---------------------------------------------------------------- pure parts

def test_regime_boxes_stay_inside_the_plausible_band():
    plo, phi = B.regime_box(SPACE, "conservative")
    nlo, nhi = B.regime_box(SPACE, "near_nominal")
    nom = np.array(SPACE.nominal_vector())
    assert np.all(plo <= nlo + 1e-12) and np.all(nhi <= phi + 1e-12)
    assert np.all(nlo <= nom) and np.all(nom <= nhi)
    i = SPACE.names().index("friction_floor")      # plausible [0.40, 1.00], nominal 0.80
    assert nlo[i] == pytest.approx(0.70) and nhi[i] == pytest.approx(0.85)
    with pytest.raises(ValueError):
        B.regime_box(SPACE, "broad")


def test_sampled_conditions_load_as_a_battery():
    conds = B.sample_conditions(SPACE, 40, rng=np.random.default_rng(0))
    assert len(conds) == 40 and len({c for c, _ in conds}) == 40
    assert sum(c.startswith("near-") for c, _ in conds) == 20
    for _, theta in conds:
        if "latency_steps" in theta:
            assert theta["latency_steps"] == round(theta["latency_steps"])
    import dataclasses
    probe = dataclasses.replace(ratings.load_battery("legged-quadruped-v1"),
                                cells=tuple(conds))
    assert ratings.cell_thetas(probe, SPACE).shape == (40, len(SPACE.names()))


def test_sampling_is_deterministic_given_the_seed():
    a = B.sample_conditions(SPACE, 10, rng=np.random.default_rng(7))
    b = B.sample_conditions(SPACE, 10, rng=np.random.default_rng(7))
    assert a == b


def test_select_keeps_only_the_band_and_caps():
    conds = [(f"c{i}", {}) for i in range(6)]
    outcomes = {"c0": [0] * 10, "c1": [1] + [0] * 9, "c2": [1] * 6 + [0] * 4,
                "c3": [1] * 7 + [0] * 3, "c4": [1] * 10, "c5": [1] * 3 + [0] * 7}
    keep = B.select_cells(conds, outcomes, (0.1, 0.6), 10)
    assert [k[0] for k in keep] == ["c1", "c2", "c5"]
    assert keep[0][2:] == (1, 10)
    capped = B.select_cells(conds, outcomes, (0.1, 0.6), 2, np.random.default_rng(0))
    assert len(capped) == 2 and set(k[0] for k in capped) <= {"c1", "c2", "c5"}
    with pytest.raises(ValueError):
        B.select_cells(conds, outcomes, (0.6, 0.1), 2)


# ---------------------------------------------------------------- build + gate

def test_build_writes_a_loadable_battery(ckpts, tmp_path):
    b, _ = ckpts
    out = tmp_path / "boundary.yaml"
    ev = mock_eval()
    assert B.build(_args(b, out), evaluate_fn=ev, space=SPACE) == 0
    bat = ratings.load_battery(str(out))
    base = ratings.load_battery("legged-quadruped-v1")
    assert bat.id.startswith("boundary-") and bat.robot_class == base.robot_class
    assert (bat.horizon, bat.fall_rule, bat.pass_rule) == (base.horizon, base.fall_rule,
                                                           base.pass_rule)
    assert 0 < len(bat.cells) <= 24
    assert bat.baseline_sha256 == ratings.sha256_file(b)
    ratings.cell_thetas(bat, SPACE)                     # every value plausible
    # one evaluator call, baseline only, on selection seeds disjoint from battery seeds
    assert len(ev.calls) == 1 and ev.calls[0][0] == str(b)
    sel = set(ev.calls[0][2])
    assert len(sel) == 20 and not sel & set(bat.seeds)
    assert not set(bat.seeds) & {G.ESCALATION_SEED_BASE + i for i in range(64)}
    # every kept cell's baseline selection rate is inside the band
    again = ev(str(b), G.battery_cells(bat), sorted(sel), bat.horizon)
    for cid, _ in bat.cells:
        assert 0.1 <= sum(again[cid]) / 20 <= 0.6


def test_no_condition_in_range_writes_nothing(ckpts, tmp_path, capsys):
    b, _ = ckpts
    out = tmp_path / "boundary.yaml"
    never = lambda spec, cells, seeds, horizon: {c["id"]: [0] * len(seeds) for c in cells}
    assert B.build(_args(b, out), evaluate_fn=never, space=SPACE) == 1
    assert not out.exists()
    assert "nothing written" in capsys.readouterr().err


def test_missing_baseline_is_a_usage_error(tmp_path, capsys):
    rc = B.build(_args(tmp_path / "nope.pkl", tmp_path / "o.yaml"), evaluate_fn=mock_eval(),
                 space=SPACE)
    assert rc == 2 and "not found" in capsys.readouterr().err


def _built(b, tmp_path):
    out = tmp_path / "boundary.yaml"
    assert B.build(_args(b, out), evaluate_fn=mock_eval(), space=SPACE) == 0
    return out


def test_gate_on_boundary_battery_catches_a_small_regression(ckpts, tmp_path):
    b, c = ckpts
    out = _built(b, tmp_path)
    report, comp, _ = G.run_gate(robot="go1", baseline=str(b), candidate=str(c),
                                 battery=str(out), evaluate_fn=mock_eval("cand.pkl", 0.25),
                                 backend="mjx-cpu")
    assert report["gate"]["regressed"] is True
    assert report["battery"]["id"].startswith("boundary-")


def test_gate_on_boundary_battery_passes_an_identical_candidate(ckpts, tmp_path):
    b, c = ckpts
    out = _built(b, tmp_path)
    report, _, md = G.run_gate(robot="go1", baseline=str(b), candidate=str(c),
                               battery=str(out), evaluate_fn=mock_eval(), backend="mjx-cpu")
    assert report["gate"]["regressed"] is False
    assert "different baseline" not in md


def test_gate_notes_a_battery_built_from_another_baseline(ckpts, tmp_path):
    b, c = ckpts
    out = _built(b, tmp_path)
    other = tmp_path / "other.pkl"
    other.write_bytes(b"another-baseline")
    _, _, md = G.run_gate(robot="go1", baseline=str(other), candidate=str(c),
                          battery=str(out), evaluate_fn=mock_eval(), backend="mjx-cpu")
    assert "built from a different baseline" in md
    _, _, md = G.run_gate(robot="go1", baseline=str(b), candidate=str(c), battery=str(out),
                          budget="smoke", evaluate_fn=mock_eval(), backend="mjx-cpu")
    assert "capped-horizon budget weakens it" in md


def test_cli_registers_gate_boundary():
    from breakpoint_mjx import cli

    a = cli.build_parser().parse_args(["gate-boundary", "--baseline", "a.pkl"])
    assert a.cmd == "gate-boundary" and a.func is cli.cmd_gate_boundary
    assert a.max_cells == 24 and a.seeds == 8 and a.out == "boundary-battery.yaml"
