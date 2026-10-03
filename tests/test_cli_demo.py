"""`breakpoint demo`, `run --budget`, the check-policy preflight wiring and the
'pipeline exited' path. subprocess is mocked: no jax, no env load."""
import subprocess
import sys
import types

import pytest

from breakpoint_mjx import cli

needs_demo = pytest.mark.skipif(
    not cli.demo_asset().exists(),
    reason="bundled demo checkpoint not shipped in this install")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BREAKPOINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BREAKPOINT_LICENSE_KEY", raising=False)
    for k in ("BP_BUDGET", "BP_RESTARTS", "BP_HORIZON", "BP_N_EVAL", "BP_GENERATIONS"):
        monkeypatch.delenv(k, raising=False)
    if not cli.demo_asset().exists():
        # The open-source export ships without the bundled checkpoint; these tests mock
        # the pipeline subprocess, so any pickle-looking file exercises the same path.
        fake = tmp_path / "demo_asset" / "go1_policy.pkl"
        fake.parent.mkdir()
        fake.write_bytes(b"\x80stand-in checkpoint; the mocked pipeline never loads it")
        monkeypatch.setattr(cli, "demo_asset", lambda: fake)
    return tmp_path


class Recorder:
    """Stands in for subprocess.run; returns the queued return codes in order."""

    def __init__(self, rcs=(0,), on_call=None):
        self.calls, self.rcs, self.on_call = [], list(rcs), on_call

    def __call__(self, cmd, env=None, **kw):
        self.calls.append((list(cmd), dict(env or {})))
        if self.on_call:
            self.on_call(cmd, env)
        return types.SimpleNamespace(returncode=self.rcs.pop(0) if self.rcs else 0)


@needs_demo
def test_bundled_demo_asset_is_present_and_a_pickle():
    p = cli.demo_asset()
    assert p.exists() and p.stat().st_size > 100_000
    assert p.read_bytes()[:1] == b"\x80"  # pickle protocol marker


def test_demo_copies_asset_and_runs_go1_cmaes_smoke(home, monkeypatch, capsys):
    out = home / "demo"

    def write_report(cmd, env):
        (out / "go1" / "report.html").write_text("<html/>", encoding="utf-8")

    rec = Recorder(on_call=write_report)
    monkeypatch.setattr(cli.subprocess, "run", rec)
    rc = cli.main(["demo", "--out", str(out)])
    assert rc == 0
    ckpt = out / "go1" / "go1_policy.pkl"
    assert ckpt.read_bytes() == cli.demo_asset().read_bytes()
    assert len(rec.calls) == 1  # no preflight: the bundled policy is the default path
    cmd, env = rec.calls[0]
    assert cmd == [sys.executable, "-m", "breakpoint_mjx.pipeline", "--target", "go1"]
    assert env["BP_BUDGET"] == "smoke" and env["BP_SEARCH"] == "cmaes"
    assert env["BP_RESTARTS"] == "2"
    assert env["BP_OUT"] == str(out.resolve())
    assert "BP_POLICY" not in env
    o = capsys.readouterr().out
    assert "open the report:" in o and "report.html" in o
    assert o.index("running:") < o.index("open the report:")


def test_demo_budget_flag_is_forwarded(home, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec)
    cli.main(["demo", "--out", str(home / "d"), "--budget", "standard"])
    assert rec.calls[0][1]["BP_BUDGET"] == "standard"
    assert rec.calls[0][1]["BP_RESTARTS"] == "4"


def test_user_restarts_are_not_overridden(home, monkeypatch):
    monkeypatch.setenv("BP_RESTARTS", "9")
    rec = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec)
    cli.main(["demo", "--out", str(home / "d")])
    assert rec.calls[0][1]["BP_RESTARTS"] == "9"


def test_run_without_budget_leaves_bp_budget_unset(home, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec)
    rc = cli.main(["run", "--robot", "go1", "--policy", "params:/w.npz", "--out",
                   str(home / "o"), "--no-contribute", "--skip-check"])
    assert rc == 0
    assert "BP_BUDGET" not in rec.calls[0][1]


def test_run_with_policy_preflights_then_runs(home, monkeypatch):
    rec = Recorder(rcs=(0, 0))
    monkeypatch.setattr(cli.subprocess, "run", rec)
    rc = cli.main(["run", "--robot", "go1", "--policy", "params:/w.npz", "--out",
                   str(home / "o"), "--no-contribute", "--budget", "smoke"])
    assert rc == 0
    assert len(rec.calls) == 2
    check_cmd, check_env = rec.calls[0]
    assert check_cmd[:3] == [sys.executable, "-m", "breakpoint_mjx.check_policy"]
    assert check_cmd[check_cmd.index("--policy") + 1] == "params:/w.npz"
    assert check_env["BP_BUDGET"] == "smoke"
    assert rec.calls[1][0][:3] == [sys.executable, "-m", "breakpoint_mjx.pipeline"]


def test_failing_preflight_stops_before_the_pipeline(home, monkeypatch, capsys):
    rec = Recorder(rcs=(3,))
    monkeypatch.setattr(cli.subprocess, "run", rec)
    rc = cli.main(["run", "--robot", "go1", "--policy", "params:/w.npz", "--out",
                   str(home / "o"), "--no-contribute"])
    assert rc == 3
    assert len(rec.calls) == 1
    assert "--skip-check" in capsys.readouterr().err


def test_skip_check_skips_preflight(home, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec)
    cli.main(["run", "--robot", "go1", "--policy", "params:/w.npz", "--out",
              str(home / "o"), "--no-contribute", "--skip-check"])
    assert len(rec.calls) == 1
    assert "breakpoint_mjx.pipeline" in rec.calls[0][0]


@pytest.mark.parametrize("child_rc,expect", [(1, 1), (-9, 137), (3221225477, 1),
                                             (4294967295, 1)])
def test_pipeline_failure_is_non_zero_with_message(home, monkeypatch, capsys, child_rc,
                                                   expect):
    monkeypatch.setattr(cli.subprocess, "run", Recorder(rcs=(child_rc,)))
    rc = cli.main(["demo", "--out", str(home / "d")])
    assert rc == expect and rc != 0
    err = capsys.readouterr().err
    assert f"pipeline exited {child_rc}" in err


def test_real_crashing_child_gives_non_zero(home, monkeypatch, capsys):
    """Not mocked: a real child process that dies with rc 7 in place of the pipeline."""
    real = subprocess.run
    monkeypatch.setattr(cli, "_pipeline_cmd",
                        lambda extra: [sys.executable, "-c", "import sys; sys.exit(7)"])
    monkeypatch.setattr(cli.subprocess, "run", real)
    rc = cli.main(["demo", "--out", str(home / "d")])
    assert rc == 7
    assert "pipeline exited 7" in capsys.readouterr().err


def test_check_policy_subcommand_parses():
    a = cli.build_parser().parse_args(
        ["check-policy", "--robot", "go1", "--policy", "x.pkl", "--n", "4"])
    assert a.func is cli.cmd_check_policy and a.n == 4 and a.horizon == 150
