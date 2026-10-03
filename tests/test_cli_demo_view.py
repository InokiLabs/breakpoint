"""`breakpoint demo --view` (the bundled measured sample report, no JAX), `--low-mem`
forwarding, the doctor memory section and the crash-recovery hint. subprocess is
mocked except for the one real no-JAX check."""
import os
import subprocess
import sys
import time
import types

import pytest

from breakpoint_mjx import cli, stagecache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BREAKPOINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BREAKPOINT_LICENSE_KEY", raising=False)
    monkeypatch.setenv("BP_NO_BROWSER", "1")
    for k in ("BP_BUDGET", "BP_RESTARTS", "BP_LOWMEM"):
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
    def __init__(self, rcs=(0,), on_call=None):
        self.calls, self.rcs, self.on_call = [], list(rcs), on_call

    def __call__(self, cmd, env=None, **kw):
        self.calls.append((list(cmd), dict(env or {})))
        if self.on_call:
            self.on_call(cmd, env)
        return types.SimpleNamespace(returncode=self.rcs.pop(0) if self.rcs else 0)


def test_sample_report_is_bundled_and_labelled():
    p = cli.sample_report_path()
    assert p.exists(), "run scripts/bundle_sample_report.py to regenerate it"
    html = p.read_text(encoding="utf-8")
    assert "CPU smoke run" in html and "not a GPU" in html
    fam = p.parent / "families.json"
    assert fam.exists() and fam.stat().st_size > 100


def test_view_prints_path_without_running(home, monkeypatch, capsys):
    rec = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec)
    assert cli.main(["demo", "--view", "--no-open"]) == 0
    assert rec.calls == []
    o = capsys.readouterr().out
    assert str(cli.sample_report_path()) in o and "CPU smoke run" in o
    assert "colab/evidence_cell.py" in o and "UNVERIFIED" in o


def test_view_imports_no_jax_and_is_fast():
    code = ("import sys, time; t=time.time(); from breakpoint_mjx import cli; "
            "rc = cli.main(['demo', '--view', '--no-open']); "
            "assert rc == 0, rc; assert 'jax' not in sys.modules, 'jax imported'; "
            "print('elapsed', time.time() - t)")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=REPO, BP_NO_BROWSER="1"),
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 5.0


def test_low_mem_flag_is_forwarded(home, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec)
    cli.main(["demo", "--out", str(home / "d"), "--low-mem"])
    assert rec.calls[0][1]["BP_LOWMEM"] == "1"
    rec2 = Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec2)
    cli.main(["demo", "--out", str(home / "d2")])
    assert "BP_LOWMEM" not in rec2.calls[0][1]


def test_native_crash_prints_partial_report_and_resume_command(home, monkeypatch, capsys):
    out = home / "demo"

    def crash_after_report(cmd, env):
        d = out / "go1"
        (d / "report.html").write_text("<html/>", encoding="utf-8")
        (d / "failures.json").write_text("[]", encoding="utf-8")
        stagecache.save_stage(d, "cmc", "k", [])

    monkeypatch.setattr(cli.subprocess, "run",
                        Recorder(rcs=(3221225477,), on_call=crash_after_report))
    rc = cli.main(["demo", "--out", str(out)])
    assert rc != 0
    err = capsys.readouterr().err
    assert str(out / "go1" / "report.html") in err
    assert "finished stages saved: cmc" in err
    assert f"breakpoint demo --out {out}" in err
    assert "--low-mem" in err


def test_crash_before_search_says_nothing_to_resume(home, monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "run", Recorder(rcs=(1,)))
    assert cli.main(["demo", "--out", str(home / "x")]) != 0
    assert "nothing to resume yet" in capsys.readouterr().err


def test_run_resume_command_keeps_flags():
    args = types.SimpleNamespace(robot="go1", out="o", search="active", policy="onnx:p.onnx",
                                 budget="standard", obs_spec=None, low_mem=True)
    cmd = cli._resume_command(args)
    assert cmd == ("breakpoint run --robot go1 --out o --search active --policy onnx:p.onnx "
                   "--budget standard --low-mem")


def test_doctor_reports_memory(home, monkeypatch, capsys):
    monkeypatch.setattr(stagecache, "free_memory", lambda: {
        "free_ram_mb": 900, "total_ram_mb": 16000, "free_commit_mb": 1000,
        "commit_limit_mb": 30000, "source": "test"})
    cli.main(["doctor"])
    o = capsys.readouterr().out
    assert "free RAM" in o and "free commit" in o and "demo peak" in o
    assert "warning: free RAM 900 MB" in o
