"""CI gate (breakpoint_mjx/gate.py): comparison, tolerance, contract fields, upload, markdown.

All tests use a mocked evaluator; no JAX or simulator is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error

import pytest

from breakpoint_mjx import gate as G

CELL_IDS = [c["id"] for c in G.BUILTIN_BATTERY["cells"]]


@pytest.fixture
def ckpts(tmp_path):
    b = tmp_path / "base.pkl"
    c = tmp_path / "cand.pkl"
    b.write_bytes(b"baseline-weights")
    c.write_bytes(b"candidate-weights")
    return b, c


def mock_eval(table):
    """table: {spec_basename: {cell_id: falls}}; missing cells fall 0 times."""
    calls = []

    def fn(spec, cells, seeds, horizon):
        calls.append((str(spec), [c["id"] for c in cells], list(seeds), horizon))
        name = str(spec).replace("\\", "/").rsplit("/", 1)[-1]
        return {c["id"]: int(table.get(name, {}).get(c["id"], 0)) for c in cells}

    fn.calls = calls
    return fn


def _args(b, c, out, **kw):
    ns = dict(robot="go1", baseline=str(b), candidate=str(c), battery="legged-quadruped-v1",
              budget="standard", tolerance=0.0, out=str(out), summary_out=None,
              candidate_name=None, baseline_name=None, no_upload=False)
    ns.update(kw)
    return argparse.Namespace(**ns)


def run(b, c, table, tmp_path, **kw):
    return G.run_gate(robot="go1", baseline=str(b), candidate=str(c),
                      evaluate_fn=mock_eval(table), backend="mjx-cpu",
                      batteries_dir=tmp_path / "no-batteries", **kw)


# ---------------------------------------------------------------- detection

def test_regression_detected(ckpts, tmp_path):
    b, c = ckpts
    report, comp, md = run(b, c, {"cand.pkl": {"push": 5}}, tmp_path)
    assert report["gate"]["regressed"] is True
    assert report["gate"]["new_failing_cells"] == ["push"]
    row = next(r for r in comp["rows"] if r["id"] == "push")
    assert row["status"] == "REGRESSED"


def test_no_regression_when_identical(ckpts, tmp_path):
    b, c = ckpts
    report, comp, _ = run(b, c, {}, tmp_path)
    assert report["gate"]["regressed"] is False
    assert report["gate"]["new_failing_cells"] == []
    assert all(r["status"] == "pass" for r in comp["rows"])


def test_cell_baseline_also_fails_is_not_a_regression(ckpts, tmp_path):
    b, c = ckpts
    report, comp, _ = run(b, c, {"base.pkl": {"slope": 8}, "cand.pkl": {"slope": 8}},
                          tmp_path)
    assert report["gate"]["regressed"] is False
    assert next(r for r in comp["rows"] if r["id"] == "slope")["status"] == \
        "fails (baseline too)"


def test_fixed_cell_reported(ckpts, tmp_path):
    b, c = ckpts
    _, comp, _ = run(b, c, {"base.pkl": {"payload": 6}}, tmp_path)
    assert next(r for r in comp["rows"] if r["id"] == "payload")["status"] == "fixed"


def test_both_checkpoints_get_same_cells_and_seeds(ckpts, tmp_path):
    b, c = ckpts
    ev = mock_eval({})
    G.run_gate(robot="go1", baseline=str(b), candidate=str(c), evaluate_fn=ev,
               backend="mjx-cpu", batteries_dir=tmp_path / "none", budget="smoke")
    (sb, cb, seb, hb), (sc, cc, sec, hc) = ev.calls
    assert sb.endswith("base.pkl") and sc.endswith("cand.pkl")
    assert cb == cc == CELL_IDS
    assert seb == sec == [0, 1]
    assert hb == hc == 150


def test_exit_codes_and_failed_message(ckpts, tmp_path, capsys):
    b, c = ckpts
    rc = G.gate(_args(b, c, tmp_path / "g.json", no_upload=True),
                evaluate_fn=mock_eval({"cand.pkl": {"nominal": 3}}), backend="mjx-cpu")
    assert rc == 1
    assert G.GATE_FAILED_MSG in capsys.readouterr().out
    rc = G.gate(_args(b, c, tmp_path / "g2.json", no_upload=True),
                evaluate_fn=mock_eval({}), backend="mjx-cpu")
    assert rc == 0


def test_missing_checkpoint_is_usage_error(tmp_path, capsys):
    rc = G.gate(_args(tmp_path / "nope.pkl", tmp_path / "nope2.pkl", tmp_path / "g.json"),
                evaluate_fn=mock_eval({}))
    assert rc == 2
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------- tolerance

def test_compare_tolerance_absorbs_small_increase():
    pr = G.DEFAULT_PASS_FALL_RATE
    base = [G.cell_result("a", 1, 8, pr)]         # 0.125 -> passes
    cand = [G.cell_result("a", 2, 8, pr)]         # 0.25  -> fails, delta 0.125
    assert G.compare_cells(base, cand, 0.0)["regressed"] is True
    loose = G.compare_cells(base, cand, 0.125)
    assert loose["regressed"] is False
    assert loose["rows"][0]["status"] == "within tolerance"
    assert G.compare_cells(base, cand, 0.1)["regressed"] is True


def test_tolerance_in_report_and_negative_rejected(ckpts, tmp_path):
    b, c = ckpts
    report, _, _ = run(b, c, {}, tmp_path, tolerance=0.25)
    assert report["gate"]["tolerance"] == "fall_rate_delta<=0.25"
    with pytest.raises(ValueError):
        run(b, c, {}, tmp_path, tolerance=-0.1)


def test_pass_rule_parsing():
    assert G.parse_pass_fall_rate({"pass_rule": "fall rate <= 1/8 across seeds"}) == 0.125
    assert G.parse_pass_fall_rate({"pass_rule": "fall rate <= 0.25"}) == 0.25
    assert G.parse_pass_fall_rate({}) == G.DEFAULT_PASS_FALL_RATE


# ---------------------------------------------------------------- contract

def test_report_fields_per_contract(ckpts, tmp_path):
    b, c = ckpts
    report, _, _ = run(b, c, {"cand.pkl": {"push": 2}}, tmp_path)
    assert list(report) == list(G.CONTRACT_KEYS)
    assert report["schema"] == "breakpoint.report/v1"
    assert report["kind"] == "gate"
    assert set(report["battery"]) == {"id", "sha256"}
    assert report["battery"]["id"] == "builtin-quadruped-gate-v0"
    assert set(report["engine"]) == {"version", "commit"}
    assert report["robot"] == "go1"
    assert report["policy"] == {"name": "cand.pkl", "source": "local",
                                "sha256": hashlib.sha256(b"candidate-weights").hexdigest()}
    run_ = report["run"]
    assert set(run_) == {"date", "backend", "seeds", "runner"}
    assert run_["backend"] == "mjx-cpu" and run_["runner"] == "self"
    assert run_["seeds"] == list(range(8))
    for cell in report["cells"]:
        assert set(cell) == {"id", "n", "falls", "fall_rate", "ci95", "passed"}
        assert cell["n"] == 8
    assert report["summary"] == {"cells_passed": 7, "cells_total": 8,
                                 "label": "7/8 cells passed"}
    g = report["gate"]
    assert set(g) == {"baseline_sha256", "candidate_sha256", "new_failing_cells",
                      "regressed", "tolerance"}
    assert g["baseline_sha256"] == hashlib.sha256(b"baseline-weights").hexdigest()
    assert g["candidate_sha256"] == report["policy"]["sha256"]
    assert report["attestation"] is None
    # never a path: only the file's name
    assert str(tmp_path) not in json.dumps(report)


def test_cell_ci_matches_contract_example():
    cell = G.cell_result("nominal", 0, 8, G.DEFAULT_PASS_FALL_RATE)
    assert cell == {"id": "nominal", "n": 8, "falls": 0, "fall_rate": 0.0,
                    "ci95": [0.0, 0.3694], "passed": True}


def test_battery_yaml_is_loaded_when_present(ckpts, tmp_path):
    b, c = ckpts
    d = tmp_path / "batteries"
    d.mkdir()
    y = d / "legged-quadruped-v1.yaml"
    y.write_text("battery: legged-quadruped-v1\nversion: 1\nhorizon: 300\n"
                 "seeds: [0, 1, 2, 3]\ncells:\n  - id: nominal\n    theta: {}\n"
                 "  - id: lag\n    theta: {latency_steps: 2}\n"
                 "pass_rule: \"a cell passes if fall rate <= 1/4 across seeds\"\n",
                 encoding="utf-8")
    report, _, md = G.run_gate(robot="go1", baseline=str(b), candidate=str(c),
                               evaluate_fn=mock_eval({"cand.pkl": {"lag": 1}}),
                               backend="mjx-cpu", batteries_dir=d)
    assert report["battery"] == {"id": "legged-quadruped-v1",
                                 "sha256": hashlib.sha256(y.read_bytes()).hexdigest()}
    assert [x["id"] for x in report["cells"]] == ["nominal", "lag"]
    assert report["gate"]["regressed"] is False      # 1/4 <= 1/4 passes
    assert "built-in" not in md


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


def _report(ckpts, tmp_path):
    b, c = ckpts
    return run(b, c, {}, tmp_path)[0]


def test_upload_skipped_without_env(ckpts, tmp_path):
    called = []
    res = G.maybe_upload(_report(ckpts, tmp_path), environ={},
                         opener=lambda *a, **k: called.append(1))
    assert res == "skipped" and called == []
    res = G.maybe_upload(_report(ckpts, tmp_path), environ={"BREAKPOINT_API_KEY": "bp_x"},
                         opener=lambda *a, **k: called.append(1))
    assert res == "skipped" and called == []


def test_upload_posts_contract_payload(ckpts, tmp_path):
    seen = {}

    def opener(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(200)

    rep = _report(ckpts, tmp_path)
    env = {"BREAKPOINT_API_KEY": "bp_secret", "BREAKPOINT_PROJECT": "my proj",
           "BREAKPOINT_API_URL": "https://example.test/"}
    assert G.maybe_upload(rep, environ=env, opener=opener) == "uploaded"
    assert seen["url"] == "https://example.test/api/projects/my%20proj/gate"
    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer bp_secret"
    assert set(seen["body"]) == set(G.CONTRACT_KEYS)
    assert seen["body"]["kind"] == "gate"


@pytest.mark.parametrize("exc", [
    urllib.error.HTTPError("u", 405, "Method Not Allowed", {}, None),
    urllib.error.URLError("dns"),
    TimeoutError("slow"),
])
def test_upload_failure_warns_only(ckpts, tmp_path, exc):
    logs = []

    def opener(req, timeout=None):
        raise exc

    env = {"BREAKPOINT_API_KEY": "k", "BREAKPOINT_PROJECT": "p"}
    assert G.maybe_upload(_report(ckpts, tmp_path), environ=env, opener=opener,
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
                evaluate_fn=mock_eval({"cand.pkl": {"nominal": 4}}), backend="mjx-cpu",
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

def test_markdown_summary_on_regression(ckpts, tmp_path):
    b, c = ckpts
    _, _, md = run(b, c, {"cand.pkl": {"push": 5, "slope": 2}}, tmp_path)
    assert md.startswith("## BreakPoint gate: FAILED: regression detected")
    assert "`push`, `slope`" in md
    assert "| `push` | 0/8 | 5/8 |" in md
    assert "**REGRESSED**" in md
    assert "builtin-quadruped-gate-v0" in md
    assert "built-in" in md                  # the fallback note is shown
    assert "Simulation only" in md
    assert "base.pkl" in md and "cand.pkl" in md


def test_markdown_summary_on_pass_and_smoke_note(ckpts, tmp_path):
    b, c = ckpts
    _, _, md = run(b, c, {}, tmp_path, budget="smoke")
    assert md.startswith("## BreakPoint gate: PASSED")
    assert "budget smoke" in md and "not a rating" in md
    assert "8/8 cells passed" in md


def test_markdown_flags_baseline_failing_nominal(ckpts, tmp_path):
    b, c = ckpts
    _, _, md = run(b, c, {"base.pkl": {"nominal": 8}, "cand.pkl": {"nominal": 8}}, tmp_path)
    assert "BASELINE fails the nominal cell" in md


def test_summary_out_appends(ckpts, tmp_path):
    b, c = ckpts
    s = tmp_path / "summary.md"
    s.write_text("existing\n", encoding="utf-8")
    G.gate(_args(b, c, tmp_path / "g.json", summary_out=str(s), no_upload=True),
           evaluate_fn=mock_eval({}), backend="mjx-cpu")
    text = s.read_text(encoding="utf-8")
    assert text.startswith("existing\n## BreakPoint gate: PASSED")


# ---------------------------------------------------------------- cli wiring

def test_cli_registers_gate():
    from breakpoint_mjx.cli import build_parser

    a = build_parser().parse_args(["gate", "--baseline", "a.pkl", "--candidate", "b.pkl",
                                   "--budget", "smoke", "--tolerance", "0.1"])
    assert a.cmd == "gate" and a.budget == "smoke" and a.tolerance == 0.1
    assert a.battery == "legged-quadruped-v1" and a.out == "gate.json"


def test_spec_path_strips_adapter_prefix():
    assert G.spec_path("onnx:./p.onnx") == "./p.onnx"
    assert G.spec_path("params:w.npz") == "w.npz"
    assert G.spec_path("C:/x/p.pkl") == "C:/x/p.pkl"     # drive letter is not a prefix
