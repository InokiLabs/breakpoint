"""check_policy: the pure diagnosis on synthetic signals (CPU), plus an opt-in real check
of the bundled checkpoint (loads the Go1 MJX env; ~80 s on a laptop CPU, so it runs only
with BP_RUN_SLOW=1)."""
import os
import types
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx import check_policy as cp

TH = {"trunk_height_min": 0.18, "up_proj_min": 0.5, "tracking_err_max": 0.5,
      "ang_vel_max": 8.0}
N, H, D, A = 8, 150, 6, 4


def _sig(fall, ttf=H, trk=0.2):
    return {"fall": float(fall), "time_to_failure": float(ttf), "tracking_error": trk}


def _good(rng):
    obs = rng.normal(0.0, 1.0, (N, H, D))
    act = np.clip(rng.normal(0.0, 0.3, (N, H, A)), -1, 1)
    return [_sig(0) for _ in range(N)], obs, act, (np.zeros(D), np.ones(D))


def test_good_policy_passes():
    sig, obs, act, norm = _good(np.random.default_rng(0))
    d = cp.diagnose(sig, obs, act, TH, H, obs_norm=norm)
    assert d["passed"] and d["reasons"] == [] and d["causes"] == []
    assert d["fall_rate"] == 0.0 and d["flagged_dims"] == []
    assert "PASS" in cp.format_diagnosis(d)


def test_falling_policy_with_scrambled_obs_fails_and_names_the_mismatch():
    rng = np.random.default_rng(1)
    sig = [_sig(1, ttf=10 + i) for i in range(N)]
    obs = rng.normal(0.0, 1.0, (N, H, D))
    obs[..., 2] += 12.0          # a dim fed from the wrong slot: far off its norm mean
    obs[..., 4] *= 20.0          # and one with the wrong scale
    act = np.clip(rng.normal(0.0, 0.3, (N, H, A)), -1, 1)
    d = cp.diagnose(sig, obs, act, TH, H, obs_norm=(np.zeros(D), np.ones(D)))
    assert not d["passed"]
    assert set(d["flagged_dims"]) == {2, 4}
    text = " ".join(d["causes"])
    assert "OBSERVATION MISMATCH" in text and "EARLY FALLS" in text
    out = cp.format_diagnosis(d)
    assert "FAIL" in out and "likely causes" in out and "time-to-fall" in out


def test_saturating_actions_name_action_scale():
    rng = np.random.default_rng(2)
    sig = [_sig(1, ttf=60) for _ in range(N)]
    obs = rng.normal(0.0, 1.0, (N, H, D))
    act = np.sign(rng.normal(size=(N, H, A)))  # always at the rail
    d = cp.diagnose(sig, obs, act, TH, H, obs_norm=None)
    assert not d["passed"] and d["sat_frac"] > 0.9
    text = " ".join(d["causes"])
    assert "ACTION SCALE" in text and "NO OBS NORMALIZATION" in text


def test_tracking_only_failure_names_command_channels():
    rng = np.random.default_rng(3)
    sig, obs, act, norm = _good(rng)
    sig = [_sig(0, trk=1.2) for _ in range(N)]
    d = cp.diagnose(sig, obs, act, TH, H, obs_norm=norm)
    assert not d["passed"] and "tracking error" in d["reasons"][0]
    assert any("DOES NOT TRACK" in c for c in d["causes"])


def test_fall_rate_boundary_is_inclusive():
    rng = np.random.default_rng(4)
    _, obs, act, norm = _good(rng)
    sig = [_sig(1, ttf=100)] * 2 + [_sig(0)] * 6          # 0.25 exactly -> still passes
    assert cp.diagnose(sig, obs, act, TH, H, obs_norm=norm)["passed"]
    sig = [_sig(1, ttf=100)] * 3 + [_sig(0)] * 5          # 0.375 -> fails
    assert not cp.diagnose(sig, obs, act, TH, H, obs_norm=norm)["passed"]


def test_post_fall_steps_are_excluded_from_stats():
    obs = np.zeros((1, 10, 1))
    obs[0, 5:, 0] = 1000.0       # lying on the floor after the fall at step 5
    rows = cp.obs_dim_stats(obs, cp._valid_mask([5], 1, 10), np.zeros(1), np.ones(1))
    assert rows[0]["mean"] == 0.0 and not rows[0]["flag"]


def test_ttf_histogram_counts():
    h = cp.ttf_histogram([0, 10, 29, 149], 150, bins=5)
    assert [c for _, _, c in h] == [3, 0, 0, 0, 1]


def test_adapter_spec_detection():
    assert cp.is_adapter_spec("params:/w.npz") and cp.is_adapter_spec("ONNX:/p.onnx")
    assert not cp.is_adapter_spec("C:/x/go1_policy.pkl")  # drive letter is not a kind
    assert not cp.is_adapter_spec("./go1_policy.pkl")


def test_check_returns_3_on_fail_and_0_on_pass(monkeypatch, capsys):
    rng = np.random.default_rng(5)
    sig, obs, act, norm = _good(rng)
    good = cp.diagnose(sig, obs, act, TH, H, obs_norm=norm)
    bad = cp.diagnose([_sig(1, ttf=5)] * N, obs, act, TH, H, obs_norm=norm)
    args = types.SimpleNamespace(robot="go1", policy="x.pkl", n=N, horizon=H, seed=0,
                                 obs_norm=None, action_scale=None, squash=False,
                                 activation=None, mean_dim=None)
    for d, want in ((good, 0), (bad, cp.EXIT_FAIL)):
        monkeypatch.setattr(cp, "run_check",
                            lambda *a, _d=dict(d, seconds=1.0), **k: _d)
        assert cp.check(args) == want
    assert "FAIL" in capsys.readouterr().out


def test_check_turns_adapter_dim_error_into_exit_3(monkeypatch, capsys):
    def boom(*a, **k):
        raise SystemExit("BYO policy expects obs dim 45 but env obs['state'] is 48")

    monkeypatch.setattr(cp, "run_check", boom)
    args = types.SimpleNamespace(robot="go1", policy="params:/w.npz", n=4, horizon=10,
                                 seed=0, obs_norm=None, action_scale=None, squash=False,
                                 activation=None, mean_dim=None)
    assert cp.check(args) == cp.EXIT_FAIL
    assert "obs dim 45" in capsys.readouterr().err


@pytest.mark.skipif(
    not (Path(cp.__file__).resolve().parent / "assets" / "demo" / "go1_policy.pkl").exists(),
    reason="bundled demo checkpoint not shipped in this install")
@pytest.mark.skipif(os.environ.get("BP_RUN_SLOW") != "1",
                    reason="loads the Go1 MJX env (~80 s CPU); set BP_RUN_SLOW=1")
def test_bundled_checkpoint_walks_at_nominal():
    pytest.importorskip("mujoco_playground")
    from breakpoint_mjx.cli import demo_asset

    d = cp.run_check("go1", str(demo_asset()), n=8, horizon=150, verbose=False)
    assert d["passed"], cp.format_diagnosis(d)


# ---------------------------------------------------------------------------
# contracts: --obs-spec, --infer-contract, `breakpoint init`, `run --obs-spec`
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "contract"


def _args(**kw):
    base = dict(robot="go1", policy="x.pkl", n=N, horizon=H, seed=0, obs_norm=None,
                action_scale=None, squash=False, activation=None, mean_dim=None,
                obs_spec=None, infer_contract=False, contract_out="contract.yaml",
                bci_budget=64)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_new_flags_parse():
    from breakpoint_mjx.cli import build_parser

    a = build_parser().parse_args(["check-policy", "--robot", "go1", "--policy", "p.onnx",
                                   "--obs-spec", "c.yaml", "--infer-contract",
                                   "--contract-out", "o.yaml", "--bci-budget", "256"])
    assert a.obs_spec == "c.yaml" and a.infer_contract and a.contract_out == "o.yaml"
    assert a.bci_budget == 256
    r = build_parser().parse_args(["run", "--robot", "go1", "--policy", "p.onnx",
                                   "--obs-spec", "c.yaml"])
    assert r.obs_spec == "c.yaml"
    i = build_parser().parse_args(["init", "--from", "rsl_rl", "some/run"])
    assert i.source == "rsl_rl" and i.robot == "go1" and i.out == "contract.yaml"


def test_obs_map_applies_contract_but_not_normalization():
    from breakpoint_mjx.adapters import ObsSpec

    sp = ObsSpec(indices=[1, 0], offset=[0.0, 1.0], scale=[2.0, 3.0],
                 mean=[100.0, 100.0], std=[7.0, 7.0])
    f = cp._obs_map(sp)
    obs = np.array([[[10.0, 20.0]]])
    np.testing.assert_allclose(f(obs), [[[40.0, 27.0]]])
    assert cp._obs_map(ObsSpec()) is None


def test_infer_contract_runs_only_when_nominal_fails(monkeypatch, capsys):
    rng = np.random.default_rng(9)
    sig, obs, act, norm = _good(rng)
    good = dict(cp.diagnose(sig, obs, act, TH, H, obs_norm=norm), seconds=1.0)
    bad = dict(cp.diagnose([_sig(1, ttf=5)] * N, obs, act, TH, H, obs_norm=norm),
               seconds=1.0)
    calls = []

    def fake_infer(args, opts, ctx):
        calls.append(opts)
        return cp.EXIT_FAIL, None                   # ambiguous / failed: no file

    monkeypatch.setattr(cp, "infer_contract", fake_infer)
    monkeypatch.setattr(cp, "run_check", lambda *a, **k: good)
    assert cp.check(_args(infer_contract=True)) == 0 and calls == []
    monkeypatch.setattr(cp, "run_check", lambda *a, **k: bad)
    assert cp.check(_args(infer_contract=True)) == cp.EXIT_FAIL and len(calls) == 1
    assert cp.check(_args(infer_contract=False)) == cp.EXIT_FAIL and len(calls) == 1


def test_infer_contract_accept_rechecks_through_the_written_file(monkeypatch, tmp_path):
    rng = np.random.default_rng(10)
    sig, obs, act, norm = _good(rng)
    good = dict(cp.diagnose(sig, obs, act, TH, H, obs_norm=norm), seconds=1.0)
    bad = dict(cp.diagnose([_sig(1, ttf=5)] * N, obs, act, TH, H, obs_norm=norm),
               seconds=1.0)
    seen = []

    def fake_run_check(robot, spec, **k):
        seen.append(k["opts"].get("obs_spec"))
        return good if k["opts"].get("obs_spec") else bad

    out = tmp_path / "contract.yaml"
    monkeypatch.setattr(cp, "run_check", fake_run_check)
    monkeypatch.setattr(cp, "infer_contract", lambda a, o, c: (0, out))
    assert cp.check(_args(infer_contract=True)) == 0
    assert seen == [None, str(out)]


def test_bp_obs_spec_env_is_honored(monkeypatch):
    seen = {}

    def fake_run_check(robot, spec, **k):
        seen.update(k["opts"])
        raise FileNotFoundError("stop here")

    monkeypatch.setenv("BP_OBS_SPEC", "/from/env.yaml")
    monkeypatch.setattr(cp, "run_check", fake_run_check)
    assert cp.check(_args()) == 2
    assert seen["obs_spec"] == "/from/env.yaml"


def test_init_writes_contract_and_refuses_fatal(tmp_path, capsys):
    from breakpoint_mjx import cli
    from breakpoint_mjx.adapters import load_contract

    out = tmp_path / "contract.yaml"
    rc = cli.main(["init", "--from", "rsl_rl", str(FIX / "rsl_rl_go1_flat"),
                   "--out", str(out)])
    assert rc == 0 and out.exists()
    text = capsys.readouterr().out
    assert "ASSUMED" in text and "stiffness" in text
    c = load_contract(out)
    assert c.obs_dim == 48 and c.policy_joint_names[0] == "FL_hip_joint"
    # a policy that needs a phase clock: table printed, nothing written, exit 2
    out2 = tmp_path / "c2.yaml"
    rc = cli.main(["init", "--from", "unitree_rl_gym", str(FIX / "unitree_phase_deploy.yaml"),
                   "--out", str(out2)])
    assert rc == 2 and not out2.exists()
    assert "FATAL" in capsys.readouterr().out


def test_run_exports_obs_spec_and_unbuffered_to_children(tmp_path, monkeypatch):
    from breakpoint_mjx import cli

    monkeypatch.setenv("BREAKPOINT_HOME", str(tmp_path / "home"))
    contract = tmp_path / "c.yaml"
    contract.write_text("version: 1\nrobot: go1\nobs: [joint_vel]\n", encoding="utf-8")
    calls = []

    class R:
        returncode = 0

    def fake_run(cmd, env=None, **k):
        calls.append((cmd, env))
        return R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    pkl = tmp_path / "x.pkl"
    pkl.write_bytes(b"x")
    rc = cli.main(["run", "--robot", "go1", "--policy", str(pkl), "--obs-spec",
                   str(contract), "--out", str(tmp_path / "out"), "--no-contribute"])
    assert rc == 0 and len(calls) == 2
    (pre_cmd, pre_env), (pipe_cmd, pipe_env) = calls
    assert "--obs-spec" in pre_cmd and pre_cmd[pre_cmd.index("--obs-spec") + 1] == str(
        contract.resolve())
    for env in (pre_env, pipe_env):
        assert env["BP_OBS_SPEC"] == str(contract.resolve())
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["BP_POLICY"] == str(pkl)
    # a missing contract file is an actionable error, not a child-process traceback
    assert cli.main(["run", "--robot", "go1", "--policy", str(pkl), "--obs-spec",
                     str(tmp_path / "nope.yaml"), "--no-contribute"]) == 2


@pytest.mark.skipif(
    not (Path(cp.__file__).resolve().parent / "assets" / "demo" / "go1_policy.pkl").exists(),
    reason="bundled demo checkpoint not shipped in this install")
@pytest.mark.skipif(os.environ.get("BP_RUN_SLOW") != "1",
                    reason="loads the Go1 MJX env (~80 s CPU); set BP_RUN_SLOW=1")
def test_bundled_checkpoint_walks_through_an_imported_rsl_rl_contract(tmp_path):
    """`init --from rsl_rl` on a config describing the bundled policy, then the check
    through that YAML contract: the full declarative path on a policy that walks."""
    pytest.importorskip("mujoco_playground")
    from breakpoint_mjx import cli
    from breakpoint_mjx.cli import demo_asset

    out = tmp_path / "contract.yaml"
    assert cli.main(["init", "--from", "rsl_rl", str(FIX / "bundled_go1_rsl_rl"),
                     "--out", str(out)]) == 0
    d = cp.run_check("go1", str(demo_asset()), n=8, horizon=150, verbose=False,
                     opts={"obs_spec": str(out)})
    assert d["passed"], cp.format_diagnosis(d)


@pytest.mark.skipif(
    not (Path(cp.__file__).resolve().parent / "assets" / "demo" / "go1_policy.pkl").exists(),
    reason="bundled demo checkpoint not shipped in this install")
def test_pipeline_accepts_a_windows_pkl_path_and_a_contract(tmp_path, monkeypatch):
    """`breakpoint run --policy C:/.../x.pkl` used to die in the pipeline with
    'unknown policy kind c'. The pipeline's BYO loader now weight-imports the brax
    checkpoint, and BP_OBS_SPEC applies a contract on top, with no pipeline edit."""
    pytest.importorskip("jax")
    pytest.importorskip("brax")
    from breakpoint_mjx.cli import demo_asset
    from breakpoint_mjx.pipeline import _maybe_adapter_policy

    env = types.SimpleNamespace(action_size=12, observation_size={"state": (48,)})
    ns = types.SimpleNamespace(policy=str(demo_asset().resolve()), obs_norm=None,
                               activation=None, action_scale=None, action_bias=None,
                               squash=False, weight_layout=None, mean_dim=None)
    monkeypatch.delenv("BP_OBS_SPEC", raising=False)
    # Without a contract the pipeline rebuilds the pickle natively (make_policy_fn; see
    # test_pipeline_dispatch). The weight-imported reference comes from build_policy.
    from breakpoint_mjx.adapters import build_policy
    fn = build_policy(str(demo_asset().resolve())).policy_fn
    assert fn.__self__.action_size == 12
    from breakpoint_mjx import cli

    out = tmp_path / "c.yaml"
    assert cli.main(["init", "--from", "rsl_rl", str(FIX / "bundled_go1_rsl_rl"),
                     "--out", str(out)]) == 0
    monkeypatch.setenv("BP_OBS_SPEC", str(out))
    fn2 = _maybe_adapter_policy(env, ns)
    o = {"state": np.random.default_rng(0).normal(size=48)}
    np.testing.assert_allclose(fn2.__self__.act_np(o), fn.__self__.act_np(o), atol=1e-12)
