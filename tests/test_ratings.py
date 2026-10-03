"""Crash Ratings (breakpoint_mjx/ratings.py): battery loading/hashing, theta mapping,
Clopper-Pearson intervals, the report contract, and smoke labelling. Fast: the evaluator is
mocked; no rollout runs here (the real CPU smoke rating is docs/ratings/smoke-go1-demo.json)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from breakpoint_mjx import ratings as R
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path

DEMO = Path(R.__file__).resolve().parent / "assets" / "demo" / "go1_policy.pkl"
if not DEMO.exists():
    # The open-source export ships without the bundled checkpoint. These tests mock the
    # evaluator and only hash the file, so a stand-in under a dir named "demo" (the
    # policy name is taken from it) exercises the same code.
    import tempfile

    DEMO = Path(tempfile.mkdtemp(prefix="bp_ratings_")) / "demo" / "go1_policy.pkl"
    DEMO.parent.mkdir()
    DEMO.write_bytes(b"\x80stand-in checkpoint; the mocked evaluator never unpickles it")


def _space(robot):
    return load_theta_space(config_path(f"configs/{robot}/theta_space.yaml"))


def _battery_yaml(tmp_path, cells, name="t.yaml", **over):
    body = {
        "battery": "test-v1", "version": 1, "robot_class": "legged-quadruped",
        "horizon": 500, "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
        "fall_rule": "trunk height above the floor (floor-relative) < 0.18 m, or trunk "
                     "up-axis z-projection < 0.5 (tipped over)",
        "cells": cells, "pass_rule": "a cell passes if fall rate <= 1/8 across seeds",
    }
    body.update(over)
    p = tmp_path / name
    p.write_text(json.dumps(body))          # JSON is valid YAML
    return p


# --------------------------------------------------------------------------- battery

@pytest.mark.parametrize("bid,robots", [("legged-quadruped-v1", ("go1", "barkour")),
                                        ("legged-humanoid-v1", ("g1",))])
def test_shipped_batteries_load_and_fit_their_robots(bid, robots):
    b = R.load_battery(bid)
    assert b.id == bid and b.version == 1
    assert b.seeds == tuple(range(8))
    assert 10 <= len(b.cells) <= 16
    assert b.cells[0] == ("nominal", {})
    assert (b.pass_num, b.pass_den) == (1, 8)
    assert b.up_proj_min == 0.5
    for robot in robots:
        R.check_robot(b, robot)
        th = R.cell_thetas(b, _space(robot), require_plausible=True)
        assert th.shape == (len(b.cells), len(_space(robot).dims))


def test_quadruped_fall_height_matches_search_default():
    assert R.load_battery("legged-quadruped-v1").fall_height_m == 0.18
    assert R.load_battery("legged-humanoid-v1").fall_height_m == 0.40


def test_battery_hash_is_stable_and_line_ending_independent(tmp_path):
    p = R.battery_path("legged-quadruped-v1")
    data = p.read_bytes()
    h1 = R.load_battery("legged-quadruped-v1").sha256
    assert h1 == R.load_battery(str(p)).sha256 == R.hash_battery_bytes(data)
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    (tmp_path / "crlf.yaml").write_bytes(crlf)
    assert R.load_battery(tmp_path / "crlf.yaml").sha256 == h1
    (tmp_path / "edit.yaml").write_bytes(lf.replace(b"0.45", b"0.46"))
    assert R.load_battery(tmp_path / "edit.yaml").sha256 != h1


def test_unknown_battery_and_bad_files(tmp_path):
    with pytest.raises(R.BatteryError, match="unknown battery"):
        R.load_battery("no-such-battery-v9")
    with pytest.raises(R.BatteryError, match="duplicate"):
        R.load_battery(_battery_yaml(tmp_path, [{"id": "a"}, {"id": "a"}]))
    with pytest.raises(R.BatteryError, match="seeds"):
        R.load_battery(_battery_yaml(tmp_path, [{"id": "a"}], seeds=[0, 0]))
    with pytest.raises(R.BatteryError, match="fall_rule"):
        R.load_battery(_battery_yaml(tmp_path, [{"id": "a"}], fall_rule="falls are bad"))
    with pytest.raises(R.BatteryError, match="pass_rule"):
        R.load_battery(_battery_yaml(tmp_path, [{"id": "a"}], pass_rule="be good"))


def test_thetas_outside_bounds_are_rejected(tmp_path):
    b = R.load_battery(_battery_yaml(tmp_path, [{"id": "ice", "theta": {"friction_floor": 5.0}}]))
    with pytest.raises(R.BatteryError, match="outside bounds"):
        R.cell_thetas(b, _space("go1"))


def test_implausible_values_unknown_dims_and_fractional_latency_rejected(tmp_path):
    sp = _space("go1")
    # inside bounds [0.15, 1.2] but below the plausible band [0.40, 1.00]
    b = R.load_battery(_battery_yaml(tmp_path, [{"id": "slick", "theta": {"friction_floor": 0.2}}]))
    with pytest.raises(R.BatteryError, match="plausible"):
        R.cell_thetas(b, sp)
    assert R.cell_thetas(b, sp, require_plausible=False)[0][0] == pytest.approx(0.2)
    b = R.load_battery(_battery_yaml(tmp_path, [{"id": "x", "theta": {"wind_mps": 1.0}}]))
    with pytest.raises(R.BatteryError, match="unknown theta dim 'wind_mps'"):
        R.cell_thetas(b, sp)
    b = R.load_battery(_battery_yaml(tmp_path, [{"id": "x", "theta": {"latency_steps": 1.5}}]))
    with pytest.raises(R.BatteryError, match="whole number"):
        R.cell_thetas(b, sp)


def test_unlisted_dims_stay_nominal():
    b = R.load_battery("legged-quadruped-v1")
    sp = _space("go1")
    th = R.cell_thetas(b, sp)
    names = sp.names()
    assert list(th[0]) == sp.nominal_vector()
    i = [c for c, _ in b.cells].index("low-friction")
    for j, n in enumerate(names):
        expect = 0.45 if n == "friction_floor" else sp.dims[n].nominal
        assert th[i][j] == pytest.approx(expect)


def test_robot_class_checks():
    q = R.load_battery("legged-quadruped-v1")
    with pytest.raises(R.BatteryError, match="legged-humanoid"):
        R.check_robot(q, "g1")
    with pytest.raises(R.BatteryError, match="no rating battery"):
        R.check_robot(q, "spot")
    assert R.default_battery("go1") == "legged-quadruped-v1"
    assert R.default_battery("g1") == "legged-humanoid-v1"


# --------------------------------------------------------------------------- statistics

def test_clopper_pearson_known_cases():
    a = 0.05
    lo, hi = R.clopper_pearson(0, 8)
    assert lo == 0.0 and hi == pytest.approx(1 - (a / 2) ** (1 / 8), abs=1e-6)  # 0.3694
    lo, hi = R.clopper_pearson(8, 8)
    assert hi == 1.0 and lo == pytest.approx((a / 2) ** (1 / 8), abs=1e-6)      # 0.6306
    lo, _ = R.clopper_pearson(1, 8)
    assert lo == pytest.approx(1 - (1 - a / 2) ** (1 / 8), abs=1e-6)            # 0.00316
    _, hi = R.clopper_pearson(7, 8)
    assert hi == pytest.approx((1 - a / 2) ** (1 / 8), abs=1e-6)                # 0.99684
    lo, hi = R.clopper_pearson(4, 8)
    assert lo == pytest.approx(1 - hi, abs=1e-6)
    assert lo == pytest.approx(0.1570, abs=1e-4) and hi == pytest.approx(0.8430, abs=1e-4)


def test_clopper_pearson_matches_scipy_beta():
    st = pytest.importorskip("scipy.stats")
    for n in (4, 8, 13):
        for k in range(n + 1):
            lo, hi = R.clopper_pearson(k, n)
            elo = 0.0 if k == 0 else st.beta.ppf(0.025, k, n - k + 1)
            ehi = 1.0 if k == n else st.beta.ppf(0.975, k + 1, n - k)
            assert lo == pytest.approx(elo, abs=1e-6) and hi == pytest.approx(ehi, abs=1e-6)


def test_pass_rule_is_exact():
    assert R.cell_passed(1, 8, 1, 8) and R.cell_passed(0, 8, 1, 8)
    assert not R.cell_passed(2, 8, 1, 8)
    assert R.cell_passed(0, 4, 1, 8) and not R.cell_passed(1, 4, 1, 8)
    assert not R.cell_passed(0, 0, 1, 8)


def test_summarize_counts_sim_invalid_as_fall_and_widens_ci():
    b = R.load_battery("legged-quadruped-v1")
    sigs = [{"fall": 0.0, "sim_invalid": 0.0}] * 6 + [{"fall": 1.0, "sim_invalid": 0.0},
                                                      {"fall": 0.0, "sim_invalid": 1.0}]
    c = R.summarize_cell("x", sigs, b)
    assert (c["n"], c["falls"], c["sim_invalid"], c["passed"]) == (8, 2, 1, False)
    lo, hi = R.clopper_pearson(2, 8)
    assert c["ci95"][0] <= lo and c["ci95"][1] >= hi
    assert c["ci95"][0] <= c["fall_rate"] <= c["ci95"][1]


# --------------------------------------------------------------------------- rate (mocked)

def _fake_factory(fail_cells, calls):
    """A mocked evaluator: cells whose index is in ``fail_cells`` fall on every seed."""
    def factory(target, env, policy_fn, space, horizon, thresholds):
        calls.append({"horizon": horizon, "thresholds": dict(thresholds)})

        def run(thetas, seed):
            calls.append({"seed": seed, "rows": thetas.shape[0]})
            return [{"fall": float(i in fail_cells), "sim_invalid": 0.0}
                    for i in range(thetas.shape[0])]
        return run
    return factory


def test_rate_full_report_matches_contract():
    calls = []
    rep = R.rate("go1", str(DEMO), "legged-quadruped-v1", budget="full",
                 policy_name="demo", _runner_factory=_fake_factory({1, 2}, calls),
                 _backend="mjx-cpu", log=lambda *a, **k: None)
    assert R.validate_report(rep) == []
    assert rep["schema"] == "breakpoint.report/v1" and rep["kind"] == "rating"
    assert rep["attestation"] is None and rep["gate"] is None
    b = R.load_battery("legged-quadruped-v1")
    assert rep["battery"] == {"id": b.id, "sha256": b.sha256}
    assert rep["policy"] == {"name": "demo", "source": "local", "sha256": R.sha256_file(DEMO)}
    assert rep["run"]["seeds"] == list(range(8)) and rep["run"]["budget"] == "full"
    assert rep["run"]["horizon"] == 500 and rep["run"]["runner"] == "self"
    assert [c["id"] for c in rep["cells"]] == [c for c, _ in b.cells]
    assert rep["cells"][1]["falls"] == 8 and not rep["cells"][1]["passed"]
    assert rep["cells"][0]["falls"] == 0 and rep["cells"][0]["passed"]
    n = len(b.cells)
    assert rep["summary"] == {"cells_passed": n - 2, "cells_total": n,
                              "label": f"{n - 2}/{n} cells passed"}
    # one batch per seed, all cells per batch; the battery's fall rule sets the thresholds
    assert [c["seed"] for c in calls if "seed" in c] == list(range(8))
    assert all(c["rows"] == n for c in calls if "rows" in c)
    assert calls[0]["thresholds"]["trunk_height_min"] == 0.18
    assert calls[0]["thresholds"]["up_proj_min"] == 0.5
    json.dumps(rep)  # serializable


def test_smoke_runs_are_labelled_smoke():
    calls = []
    rep = R.rate("go1", str(DEMO), budget="smoke", _runner_factory=_fake_factory(set(), calls),
                 _backend="mjx-cpu", log=lambda *a, **k: None)
    assert R.validate_report(rep) == []
    assert rep["run"]["budget"] == "smoke"
    assert rep["run"]["seeds"] == [0, 1, 2, 3] and rep["run"]["horizon"] == R.SMOKE_HORIZON
    assert rep["summary"]["label"].startswith("SMOKE PREVIEW, not a rating")
    assert all(c["n"] == 4 for c in rep["cells"])
    assert calls[0]["horizon"] == R.SMOKE_HORIZON


def test_rate_refuses_missing_checkpoint_and_wrong_robot(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.rate("go1", str(tmp_path / "nope.pkl"), _runner_factory=_fake_factory(set(), []))
    with pytest.raises(R.BatteryError):
        R.rate("g1", str(DEMO), "legged-quadruped-v1", _runner_factory=_fake_factory(set(), []))


def test_policy_file_strips_adapter_prefixes_not_drive_letters():
    assert R.policy_file("onnx:/a/p.onnx") == Path("/a/p.onnx")
    assert R.policy_file("params:w.npz") == Path("w.npz")
    assert R.policy_file("C:/runs/go1.pkl") == Path("C:/runs/go1.pkl")


def test_validate_report_catches_problems():
    calls = []
    rep = R.rate("go1", str(DEMO), budget="smoke", _runner_factory=_fake_factory(set(), calls),
                 _backend="mjx-cpu", log=lambda *a, **k: None)
    bad = json.loads(json.dumps(rep))
    del bad["policy"]["sha256"]
    bad["cells"][0]["ci95"] = [0.5, 0.9]
    bad["summary"]["cells_passed"] = 0
    probs = R.validate_report(bad)
    assert any("policy.sha256" in p for p in probs)
    assert any("ci95" in p for p in probs)
    assert any("cells_passed" in p for p in probs)
    assert R.validate_report({}) and "missing top-level key 'schema'" in R.validate_report({})[0]


def test_build_report_rejects_unknown_budget():
    b = R.load_battery("legged-quadruped-v1")
    with pytest.raises(ValueError):
        R.build_report(battery=b, robot="go1", policy={}, cells=[], backend="mjx-cpu",
                       seeds=[0], horizon=1, budget="huge")


# --------------------------------------------------------------------------- CLI

def test_cli_rate_writes_the_report(tmp_path, monkeypatch, capsys):
    from breakpoint_mjx import cli

    calls = []
    real = R.rate

    def fake_rate(robot, policy, **kw):
        return real(robot, policy, battery_name=kw["battery_name"], budget=kw["budget"],
                    policy_name=kw["policy_name"], policy_source=kw["policy_source"],
                    _runner_factory=_fake_factory({3}, calls), _backend="mjx-cpu",
                    log=lambda *a, **k: None)

    monkeypatch.setattr(R, "rate", fake_rate)
    out = tmp_path / "r" / "report.json"
    rc = cli.main(["rate", "--robot", "go1", "--policy", str(DEMO), "--battery",
                   "legged-quadruped-v1", "--budget", "smoke", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert R.validate_report(rep) == [] and rep["run"]["budget"] == "smoke"
    text = capsys.readouterr().out
    assert "SMOKE PREVIEW" in text and "latency-40ms" in text and "FAIL" in text


def test_cli_rate_bad_battery_exits_2(tmp_path, capsys):
    from breakpoint_mjx import cli

    rc = cli.main(["rate", "--robot", "go1", "--policy", str(DEMO), "--battery",
                   "nope-v9", "--out", str(tmp_path / "x.json")])
    assert rc == 2
    assert "unknown battery" in capsys.readouterr().err


def test_committed_smoke_rating_is_labelled_and_valid():
    p = Path(R.__file__).resolve().parent.parent / "docs" / "ratings" / "smoke-go1-demo.json"
    if not p.exists():
        pytest.skip("smoke rating not generated in this checkout")
    rep = json.loads(p.read_text())
    assert R.validate_report(rep) == []
    assert rep["run"]["budget"] == "smoke"
    assert rep["summary"]["label"].startswith("SMOKE PREVIEW")
    assert rep["battery"]["sha256"] == R.load_battery(rep["battery"]["id"]).sha256
    assert math.isclose(sum(c["passed"] for c in rep["cells"]), rep["summary"]["cells_passed"])
