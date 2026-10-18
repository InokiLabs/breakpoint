"""CI gate (breakpoint_mjx/gate.py): verdict, tolerance, contract fields, upload, markdown.

All tests use a mocked evaluator; no JAX or simulator is touched. The battery is the
shipped legged-quadruped-v1 (15 cells, 8 seeds, pass rule <= 1/8) unless a test writes
its own YAML.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error

import pytest

from breakpoint_mjx import gate as G
from breakpoint_mjx import ratings

BATTERY = ratings.load_battery("legged-quadruped-v1")
CELL_IDS = [cid for cid, _ in BATTERY.cells]


@pytest.fixture
def ckpts(tmp_path):
    b = tmp_path / "base.pkl"
    c = tmp_path / "cand.pkl"
    b.write_bytes(b"baseline-weights")
    c.write_bytes(b"candidate-weights")
    return b, c


def mock_eval(table, per_seed=True):
    """table: {spec_basename: {cell_id: k | [0/1 per seed]}}; missing cells never fall.
    An int k is a fall rate of k/8: seed s falls when s % 8 < k, so equal k give equal
    outcomes and the pattern carries over to escalation seeds. With per_seed=False the
    evaluator returns counts only, as an older evaluator would."""
    calls = []

    def fn(spec, cells, seeds, horizon):
        calls.append((str(spec), [c["id"] for c in cells], list(seeds), horizon))
        name = str(spec).replace("\\", "/").rsplit("/", 1)[-1]
        out = {}
        for c in cells:
            v = table.get(name, {}).get(c["id"], 0)
            if callable(v):
                outcomes = [int(v(s)) for s in seeds]
            elif isinstance(v, list):
                outcomes = list(v)
            else:
                outcomes = [int(s % 8 < v) for s in seeds]
            out[c["id"]] = outcomes if per_seed else sum(outcomes)
        return out

    fn.calls = calls
    return fn


def _args(b, c, out, **kw):
    ns = dict(robot="go1", baseline=str(b), candidate=str(c), battery=None,
              budget="standard", tolerance=0.0, decision="stat", alpha=0.05, out=str(out),
              summary_out=None, candidate_name=None, baseline_name=None, no_upload=False,
              obs_spec=None, obs_norm=None, action_scale=None, squash=None,
              activation=None, mean_dim=None)
    ns.update(kw)
    return argparse.Namespace(**ns)


def run(b, c, table, **kw):
    per_seed = kw.pop("per_seed", True)
    return G.run_gate(robot="go1", baseline=str(b), candidate=str(c),
                      evaluate_fn=mock_eval(table, per_seed), backend="mjx-cpu", **kw)


def row(comp, cid):
    return next(r for r in comp["rows"] if r["id"] == cid)


# ---------------------------------------------------------------- statistical verdict

def test_clear_regression_detected(ckpts):
    b, c = ckpts
    report, comp, _ = run(b, c, {"cand.pkl": {"push-firm": 8}})
    assert report["gate"]["regressed"] is True
    assert report["gate"]["new_failing_cells"] == ["push-firm"]
    assert row(comp, "push-firm")["status"].startswith("REGRESSED")
    assert row(comp, "push-firm")["up"] == 8


def test_no_regression_when_identical(ckpts):
    b, c = ckpts
    report, comp, _ = run(b, c, {"base.pkl": {"tilt-pos-5deg": 3},
                                 "cand.pkl": {"tilt-pos-5deg": 3}})
    assert report["gate"]["regressed"] is False
    assert report["gate"]["new_failing_cells"] == []
    assert report["gate"]["pooled"]["p"] == 1.0


def test_small_increase_is_a_warning_not_a_failure(ckpts):
    # 0/8 -> 2/8 fails the pass rule (old gate: REGRESSED) but is two coin flips
    b, c = ckpts
    report, comp, md = run(b, c, {"cand.pkl": {"push-light": 2}}, escalate_to=0)
    assert report["gate"]["regressed"] is False
    assert row(comp, "push-light")["status"] == "worse, not significant"
    assert row(comp, "push-light")["rule_status"] == "REGRESSED"
    assert report["gate"]["warnings"] == ["push-light"]
    assert "within noise" in md


def test_escalation_confirms_a_real_small_regression(ckpts):
    # a true 0% -> 25% rise: inconclusive on 8 seeds, decided on 32 fresh ones
    b, c = ckpts
    ev = mock_eval({"cand.pkl": {"push-light": 2}})
    report, comp, md = G.run_gate(robot="go1", baseline=str(b), candidate=str(c),
                                  evaluate_fn=ev, backend="mjx-cpu")
    assert report["gate"]["regressed"] is True
    assert row(comp, "push-light")["status"] == "REGRESSED (escalated)"
    esc = report["gate"]["escalation"]
    assert esc["cells"] == ["push-light"]
    assert esc["seeds"] == [G.ESCALATION_SEED_BASE + i for i in range(32)]
    # stage 2 re-ran only the suspect cell, for both checkpoints, on fresh seeds
    stage2_calls = [call for call in ev.calls if call[2] != list(range(8))]
    assert [call[1] for call in stage2_calls] == [["push-light"], ["push-light"]]
    assert not set(stage2_calls[0][2]) & set(range(8))
    assert "escalated: `push-light` re-run on 32 fresh seeds" in md


def test_escalation_clears_noise(ckpts):
    # the candidate happened to fall on two battery seeds; on fresh seeds it never does
    b, c = ckpts
    report, comp, md = run(b, c, {"cand.pkl": {"push-light": lambda s: s in (3, 5)}})
    assert report["gate"]["regressed"] is False
    assert row(comp, "push-light")["status"] == "cleared on more seeds"
    assert report["gate"]["warnings"] == []


def test_smoke_does_not_escalate_by_default(ckpts):
    b, c = ckpts
    report, _, _ = run(b, c, {"cand.pkl": {"push-light": 2}}, budget="smoke")
    assert report["gate"]["escalation"]["seeds"] == []


def test_rule_decision_keeps_the_original_behaviour(ckpts):
    b, c = ckpts
    report, comp, _ = run(b, c, {"cand.pkl": {"push-light": 2}}, decision="rule")
    assert report["gate"]["regressed"] is True
    assert report["gate"]["new_failing_cells"] == ["push-light"]
    assert report["gate"]["decision"] == "rule"
    assert "no significance test" in report["gate"]["method"]


def test_cell_the_baseline_already_fails_getting_worse_is_caught(ckpts):
    # old gate: "fails (baseline too)" -> passed the build
    b, c = ckpts
    report, comp, _ = run(b, c, {"base.pkl": {"tilt-pos-5deg": 2},
                                 "cand.pkl": {"tilt-pos-5deg": 8}})
    assert report["gate"]["regressed"] is True
    assert row(comp, "tilt-pos-5deg")["rule_status"] == "fails (baseline too)"


def test_spread_out_regression_caught_by_pooled_test(ckpts):
    # one extra fall in each of 10 cells: no cell alone is significant, the battery is
    b, c = ckpts
    table = {"cand.pkl": {cid: 1 for cid in CELL_IDS[:10]}}
    report, comp, _ = run(b, c, table, escalate_to=0)
    assert report["gate"]["pooled"]["significant"] is True
    assert report["gate"]["regressed"] is True
    assert all(row(comp, cid)["status"] == "REGRESSED (pooled)" for cid in CELL_IDS[:10])
    # with escalation the fresh seeds confirm it, cell by cell attribution included
    report, comp, _ = run(b, c, table)
    assert report["gate"]["regressed"] is True
    assert all(row(comp, cid)["status"] == "REGRESSED (escalated, pooled)"
               for cid in CELL_IDS[:10])


def test_improvement_reported_as_better(ckpts):
    b, c = ckpts
    report, comp, _ = run(b, c, {"base.pkl": {"payload-2kg": 6}})
    assert report["gate"]["regressed"] is False
    assert row(comp, "payload-2kg")["status"] == "better"
    assert row(comp, "payload-2kg")["rule_status"] == "fixed"


def test_count_only_evaluator_falls_back_to_fisher(ckpts):
    b, c = ckpts
    report, _, md = run(b, c, {"cand.pkl": {"push-firm": 8}}, per_seed=False)
    assert "Fisher" in report["gate"]["method"]
    assert report["gate"]["regressed"] is True
    assert "seed pairing was not used" in md


def test_alpha_controls_strictness(ckpts):
    b, c = ckpts
    table = {"cand.pkl": {"push-firm": 6}}           # sign test p = 1/64 per cell
    assert run(b, c, table, alpha=0.05, escalate_to=0)[0]["gate"]["regressed"] is True
    assert run(b, c, table, alpha=0.01, escalate_to=0)[0]["gate"]["regressed"] is False
    with pytest.raises(ValueError):
        run(b, c, table, alpha=1.5)


def test_both_checkpoints_get_same_cells_and_seeds(ckpts):
    b, c = ckpts
    ev = mock_eval({})
    G.run_gate(robot="go1", baseline=str(b), candidate=str(c), evaluate_fn=ev,
               backend="mjx-cpu", budget="smoke")
    (sb, cb, seb, hb), (sc, cc, sec, hc) = ev.calls
    assert sb.endswith("base.pkl") and sc.endswith("cand.pkl")
    assert cb == cc == CELL_IDS
    assert seb == sec == list(range(ratings.SMOKE_SEEDS))     # same smoke as `rate`
    assert hb == hc == ratings.SMOKE_HORIZON


# ---------------------------------------------------------------- exit codes

def test_exit_codes_and_failed_message(ckpts, tmp_path, capsys):
    b, c = ckpts
    rc = G.gate(_args(b, c, tmp_path / "g.json", no_upload=True),
                evaluate_fn=mock_eval({"cand.pkl": {"nominal": 8}}), backend="mjx-cpu")
    assert rc == 1
    out = capsys.readouterr().out
    assert G.GATE_FAILED_MSG in out and "0/8 -> 8/8" in out
    rc = G.gate(_args(b, c, tmp_path / "g2.json", no_upload=True),
                evaluate_fn=mock_eval({}), backend="mjx-cpu")
    assert rc == 0


def test_missing_checkpoint_is_usage_error(tmp_path, capsys):
    rc = G.gate(_args(tmp_path / "nope.pkl", tmp_path / "nope2.pkl", tmp_path / "g.json"),
                evaluate_fn=mock_eval({}))
    assert rc == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [RuntimeError("XLA blew up"), ModuleNotFoundError("jax"),
                                 KeyError("nope")])
def test_a_crash_is_exit_2_never_a_regression(ckpts, tmp_path, capsys, exc):
    b, c = ckpts

    def boom(*a, **k):
        raise exc

    rc = G.gate(_args(b, c, tmp_path / "g.json", no_upload=True), evaluate_fn=boom)
    assert rc == 2
    assert G.GATE_FAILED_MSG not in capsys.readouterr().out


def test_byo_dim_mismatch_systemexit_is_exit_2(ckpts, tmp_path, capsys):
    b, c = ckpts

    def boom(*a, **k):
        raise SystemExit("observation dim 45 != 48")

    rc = G.gate(_args(b, c, tmp_path / "g.json", no_upload=True), evaluate_fn=boom)
    assert rc == 2
    assert "cannot load a policy" in capsys.readouterr().err


def test_wrong_robot_class_battery_is_refused(ckpts, tmp_path, capsys):
    b, c = ckpts
    rc = G.gate(_args(b, c, tmp_path / "g.json", robot="g1",
                      battery="legged-quadruped-v1", no_upload=True),
                evaluate_fn=mock_eval({}))
    assert rc == 2
    assert "legged-quadruped" in capsys.readouterr().err


def test_default_battery_follows_the_robot(ckpts):
    b, c = ckpts
    report, _, _ = G.run_gate(robot="g1", baseline=str(b), candidate=str(c),
                              evaluate_fn=mock_eval({}), backend="mjx-cpu")
    assert report["battery"]["id"] == "legged-humanoid-v1"


# ---------------------------------------------------------------- tolerance

def test_tolerance_filters_small_but_significant_increases(ckpts):
    b, c = ckpts
    table = {"cand.pkl": {"push-firm": 8}}
    assert run(b, c, table, tolerance=0.0)[0]["gate"]["regressed"] is True
    assert run(b, c, table, tolerance=1.0)[0]["gate"]["regressed"] is False


def test_tolerance_in_report_and_negative_rejected(ckpts):
    b, c = ckpts
    report, _, _ = run(b, c, {}, tolerance=0.25)
    assert report["gate"]["tolerance"] == "fall_rate_delta<=0.25"
    with pytest.raises(ValueError):
        run(b, c, {}, tolerance=-0.1)


# ---------------------------------------------------------------- contract

def test_report_fields_per_contract(ckpts, tmp_path):
    b, c = ckpts
    report, _, _ = run(b, c, {"cand.pkl": {"push-firm": 2}})
    assert list(report) == list(G.CONTRACT_KEYS)
    assert ratings.validate_report(report) == []
    assert report["schema"] == "breakpoint.report/v1" and report["kind"] == "gate"
    assert report["battery"] == {"id": "legged-quadruped-v1", "sha256": BATTERY.sha256}
    assert report["policy"] == {"name": "cand.pkl", "source": "local",
                                "sha256": hashlib.sha256(b"candidate-weights").hexdigest()}
    run_ = report["run"]
    assert run_["backend"] == "mjx-cpu" and run_["runner"] == "self"
    assert run_["seeds"] == list(range(8))
    assert run_["budget"] == "standard" and run_["horizon"] == 500   # was missing
    for cell in report["cells"]:
        assert set(cell) == {"id", "n", "falls", "fall_rate", "ci95", "passed"}
    assert report["summary"]["cells_total"] == 15
    assert report["summary"]["cells_passed"] == 14
    g = report["gate"]
    assert g["baseline_sha256"] == hashlib.sha256(b"baseline-weights").hexdigest()
    assert g["candidate_sha256"] == report["policy"]["sha256"]
    assert g["decision"] == "stat" and g["alpha"] == 0.05
    assert {t["id"] for t in g["cell_tests"]} == set(CELL_IDS)
    assert g["baseline_falls"]["push-firm"] == 0
    assert g["min_detectable"]["per_cell"] is None      # 8 seeds cannot reach 0.025/15
    assert report["attestation"] is None
    assert str(tmp_path) not in json.dumps(report)      # never a path


def test_smoke_report_is_labelled_smoke(ckpts):
    b, c = ckpts
    report, _, md = run(b, c, {}, budget="smoke")
    assert report["run"]["budget"] == "smoke"
    assert report["summary"]["label"].startswith("SMOKE PREVIEW")
    assert "budget smoke" in md


def test_cell_ci_matches_contract_example():
    cell = G.cell_result("nominal", 0, 8, 1, 8)
    assert cell == {"id": "nominal", "n": 8, "falls": 0, "fall_rate": 0.0,
                    "ci95": [0.0, 0.3695], "passed": True}


def test_custom_battery_yaml_path(ckpts, tmp_path):
    b, c = ckpts
    y = tmp_path / "mini.yaml"
    y.write_text("battery: mini-v1\nversion: 1\nrobot_class: legged-quadruped\n"
                 "horizon: 300\nseeds: [0, 1, 2, 3]\n"
                 "fall_rule: \"trunk height above the floor < 0.18 m, or trunk up-axis "
                 "z-projection < 0.5\"\n"
                 "cells:\n  - id: nominal\n    theta: {}\n"
                 "  - id: lag\n    theta: {latency_steps: 2}\n"
                 "pass_rule: \"a cell passes if fall rate <= 1/4 across seeds\"\n",
                 encoding="utf-8")
    report, _, _ = run(b, c, {"cand.pkl": {"lag": 1}}, battery=str(y))
    assert report["battery"]["id"] == "mini-v1"
    assert [x["id"] for x in report["cells"]] == ["nominal", "lag"]
    assert report["gate"]["regressed"] is False


def test_gate_writes_json(ckpts, tmp_path):
    b, c = ckpts
    out = tmp_path / "sub" / "gate.json"
    G.gate(_args(b, c, out, no_upload=True), evaluate_fn=mock_eval({}), backend="mjx-cpu")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "gate"


# ---------------------------------------------------------------- upload

class _Resp:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _report(ckpts):
    b, c = ckpts
    return run(b, c, {})[0]


def test_upload_skipped_without_env(ckpts):
    called = []
    res = G.maybe_upload(_report(ckpts), environ={},
                         opener=lambda *a, **k: called.append(1))
    assert res == "skipped" and called == []
    res = G.maybe_upload(_report(ckpts), environ={"BREAKPOINT_API_KEY": "bp_x"},
                         opener=lambda *a, **k: called.append(1))
    assert res == "skipped" and called == []


def test_upload_posts_contract_payload(ckpts):
    seen = {}

    def opener(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(200)

    env = {"BREAKPOINT_API_KEY": "bp_secret", "BREAKPOINT_PROJECT": "my proj",
           "BREAKPOINT_API_URL": "https://example.test/"}
    assert G.maybe_upload(_report(ckpts), environ=env, opener=opener) == "uploaded"
    assert seen["url"] == "https://example.test/api/projects/my%20proj/gate"
    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer bp_secret"
    assert set(seen["body"]) == set(G.CONTRACT_KEYS)


@pytest.mark.parametrize("exc", [
    urllib.error.HTTPError("u", 405, "Method Not Allowed", {}, None),
    urllib.error.URLError("dns"),
    TimeoutError("slow"),
])
def test_upload_failure_warns_only(ckpts, exc):
    logs = []

    def opener(req, timeout=None):
        raise exc

    env = {"BREAKPOINT_API_KEY": "k", "BREAKPOINT_PROJECT": "p"}
    assert G.maybe_upload(_report(ckpts), environ=env, opener=opener,
                          log=logs.append) == "failed"
    assert logs and logs[0].startswith("WARNING")


def test_upload_failure_never_changes_exit_code(ckpts, tmp_path, capsys):
    b, c = ckpts

    def opener(req, timeout=None):
        raise urllib.error.URLError("down")

    env = {"BREAKPOINT_API_KEY": "k", "BREAKPOINT_PROJECT": "p"}
    rc = G.gate(_args(b, c, tmp_path / "g.json"), evaluate_fn=mock_eval({}),
                backend="mjx-cpu", environ=env, opener=opener)
    assert rc == 0
    assert "WARNING" in capsys.readouterr().err
    rc = G.gate(_args(b, c, tmp_path / "g2.json"),
                evaluate_fn=mock_eval({"cand.pkl": {"nominal": 8}}), backend="mjx-cpu",
                environ=env, opener=opener)
    assert rc == 1


def test_no_upload_flag(ckpts, tmp_path):
    b, c = ckpts
    called = []
    env = {"BREAKPOINT_API_KEY": "k", "BREAKPOINT_PROJECT": "p"}
    G.gate(_args(b, c, tmp_path / "g.json", no_upload=True), evaluate_fn=mock_eval({}),
           backend="mjx-cpu", environ=env,
           opener=lambda *a, **k: called.append(1) or _Resp())
    assert called == []


# ---------------------------------------------------------------- markdown

def test_markdown_summary_on_regression(ckpts):
    b, c = ckpts
    _, _, md = run(b, c, {"cand.pkl": {"push-firm": 8, "tilt-neg-5deg": 7}})
    assert md.startswith("## BreakPoint gate: FAILED: regression detected")
    assert "`push-firm`" in md and "`tilt-neg-5deg`" in md
    assert "| `push-firm` | 0/8 | 8/8 |" in md
    assert "**REGRESSED" in md
    assert "legged-quadruped-v1" in md
    assert "pooled: candidate fell alone in 15 seed-pairs" in md
    assert "Simulation only" in md
    assert "base.pkl" in md and "cand.pkl" in md


def test_markdown_summary_on_pass(ckpts):
    b, c = ckpts
    _, _, md = run(b, c, {}, budget="smoke")
    assert md.startswith("## BreakPoint gate: PASSED")
    assert "15/15 cells passed" in md
    assert "- sensitivity: no single cell can fail the build on 4 seeds alone" in md


def test_markdown_flags_baseline_failing_nominal(ckpts):
    b, c = ckpts
    _, _, md = run(b, c, {"base.pkl": {"nominal": 8}, "cand.pkl": {"nominal": 8}})
    assert "BASELINE fails the nominal cell" in md


def test_summary_out_appends(ckpts, tmp_path):
    b, c = ckpts
    s = tmp_path / "summary.md"
    s.write_text("existing\n", encoding="utf-8")
    G.gate(_args(b, c, tmp_path / "g.json", summary_out=str(s), no_upload=True),
           evaluate_fn=mock_eval({}), backend="mjx-cpu")
    assert s.read_text(encoding="utf-8").startswith("existing\n## BreakPoint gate: PASSED")


# ---------------------------------------------------------------- cli wiring

def test_cli_registers_gate():
    from breakpoint_mjx.cli import build_parser

    a = build_parser().parse_args(["gate", "--baseline", "a.pkl", "--candidate", "b.pkl",
                                   "--budget", "smoke", "--tolerance", "0.1",
                                   "--obs-spec", "c.yaml", "--alpha", "0.01"])
    assert a.cmd == "gate" and a.budget == "smoke" and a.tolerance == 0.1
    assert a.battery is None and a.out == "gate.json"
    assert a.decision == "stat" and a.alpha == 0.01 and a.obs_spec == "c.yaml"


def test_escalated_cell_is_attributed_on_fresh_seed_evidence(ckpts):
    # the battery is significantly worse overall (stage-1 pooled), but one of the cells
    # that contributed is clean on 32 fresh seeds: it must not be blamed
    b, c = ckpts
    table = {"cand.pkl": {cid: 1 for cid in CELL_IDS[:10]}}
    table["cand.pkl"]["nominal"] = lambda s: s == 0          # one battery-seed blip only
    report, comp, _ = run(b, c, table)
    assert report["gate"]["regressed"] is True
    assert row(comp, "nominal")["status"] == "cleared on more seeds"
    assert "nominal" not in report["gate"]["new_failing_cells"]
