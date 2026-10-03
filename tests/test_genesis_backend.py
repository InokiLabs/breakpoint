"""Tests for the Genesis cross-SIMULATOR backend (``backends/genesis_backend.py``).

Two tiers, following ``tests/test_gpu_smoke.py`` / ``tests/test_urdf_target.py``:

* **Default tier** -- pure numpy/stdlib: the theta-coverage guard, the quaternion math,
  the observation mirror's block layout, and every REFUSAL (non-URDF env, non-legged
  objective, legacy record, unmappable theta dim). Nothing here imports genesis, jax or
  mujoco, so it runs in the normal suite. Importing Genesis alone costs >60 s on a cold
  CPU box, which is exactly why ``GenesisBackend.available`` uses ``find_spec``.

* **Opt-in tier (``BP_RUN_GENESIS=1``)** -- the actual cross-engine measurement: build
  the URDF quadruped in BOTH engines and compare. This is the evidence behind
  ``READY = True``; without it the flag would be an intention, and a false READY is the
  worst failure mode this feature has (it would silently inflate cross-engine agreement
  by scoring a rollout nobody checked).

      BP_RUN_GENESIS=1 XLA_FLAGS="--xla_backend_optimization_level=0 \
          --xla_llvm_disable_expensive_passes=true" \
          python -m pytest tests/test_genesis_backend.py -q

  MEASURED on tests/fixtures/urdf/quadruped.urdf, horizon 20, random-weight MLP
  (2026-09-04, mujoco 3.10 / jax 0.8.3 CPU / genesis 1.3.3):

    observation mirror vs URDFEnv._obs (5 random states)  max |diff| 6.7e-08
    quaternion -> R vs MuJoCo xmat                        max |diff| 1.1e-16
    step-0 observation, MJX vs Genesis (nominal / slope)  max |diff| 0.0
    step-0 observation, MJX vs Genesis (all adversaries)  max |diff| 3.3e-09
    step-0 action,      MJX vs Genesis                    max |diff| <= 1.5e-08
    trunk_height, first 5 steps (nominal / adversarial)   max |diff| 4.2e-04  (0.13%)
    up_proj,      first 5 steps (nominal / adversarial)   max |diff| 2.0e-06
    trunk_height, first 5 steps (10 deg slope)            max |diff| 7.4e-03  (2.3%)
    up_proj,      first 5 steps (10 deg slope)            max |diff| 2.4e-03

  The step-0 observation/action agreement is EXACT because both engines are seeded from
  the same ``env.reset(key)`` state and URDFEnv's reset is deterministic. The ~1e-4
  trunk-height offset after one control step is the two engines' contact-stiffness
  difference (MuJoCo solref vs Genesis constraint_timeconst) -- a real solver difference,
  which is the product, not a porting bug. Long-horizon agreement is NOT asserted.
"""
from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest

from breakpoint_mjx.backends.genesis_backend import (
    READY,
    SUPPORTED_MODEL_DIMS,
    SUPPORTED_ROLLOUT_DIMS,
    GenesisBackend,
    _quat_to_R,
    _urdf_obs_numpy,
    unsupported_theta_dims,
)
from breakpoint_mjx.config import ThetaDim, ThetaSpace
from breakpoint_mjx.schemas import FailureRecord

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "urdf")
QUADRUPED = os.path.join(FIXTURES, "quadruped.urdf")


def _space(**kinds):
    """A minimal ThetaSpace: ``{name: kind}``."""
    return ThetaSpace(dims={
        n: ThetaDim(name=n, kind=k, bounds=(0.0, 1.0), nominal=0.0, plausible=(0.0, 1.0))
        for n, k in kinds.items()
    })


def _record(rid="g0", theta=None, **kw):
    theta = theta or {}
    return FailureRecord(
        id=rid, theta_vector=list(theta.values()), theta_named=dict(theta),
        failure_score=0.9, component_signals={}, rollout_signature=[], seed=0,
        batch_key=kw.pop("batch_key", [0, 11]), env_index=kw.pop("env_index", 3), **kw,
    )


# --------------------------------------------------------------------------- #
# Default tier: theta coverage -- an unmappable dim must RAISE, never no-op
# --------------------------------------------------------------------------- #
def test_every_legged_theta_dim_is_expressible():
    """The full legged/URDF theta space maps onto Genesis with nothing left over.

    If a future dim is added to ``configs/legged_default/theta_space.yaml`` without a
    Genesis mapping, this catches it at test time instead of letting the Genesis re-sim
    quietly run that axis at nominal -- which would show up as engine AGREEMENT on an
    axis that was never actually varied in one of the engines.
    """
    from breakpoint_mjx.config import load_theta_space
    from breakpoint_mjx.paths import config_path

    space = load_theta_space(config_path("configs/legged_default/theta_space.yaml"))
    assert unsupported_theta_dims(space) == []
    assert set(space.names()) <= (SUPPORTED_MODEL_DIMS | SUPPORTED_ROLLOUT_DIMS)


def test_unsupported_dim_is_reported_with_a_reason():
    space = _space(friction_floor="model", gear_backlash_rad="model",
                   obs_noise_scale="rollout", wind_gust="rollout")
    bad = dict(unsupported_theta_dims(space))
    assert set(bad) == {"gear_backlash_rad", "wind_gust"}
    assert all(isinstance(v, str) and v for v in bad.values())


def test_simulate_refuses_an_unmappable_theta_dim(monkeypatch):
    """Refusal, not a silent nominal replay, when the space has a dim Genesis cannot set."""
    monkeypatch.setattr(GenesisBackend, "available", staticmethod(lambda: True))
    monkeypatch.setattr(GenesisBackend, "supports_env", staticmethod(lambda env: True))
    b = GenesisBackend(object(), None, target=None,
                       space=_space(gear_backlash_rad="model"), consts=(0, 1), horizon=5)
    with pytest.raises(NotImplementedError, match="gear_backlash_rad"):
        b.simulate(_record())


# --------------------------------------------------------------------------- #
# Default tier: the refusals that keep a fabricated reproduction off the report
# --------------------------------------------------------------------------- #
def test_playground_env_is_refused_not_hand_ported():
    """A mujoco_playground env is NOT supported, and says why.

    This is the honest scope line: Playground owns reset/observation/command, Genesis
    cannot call Playground's code, and hand-porting a 48-channel observation layout is
    the silent-mismatch this repo refuses to take on.
    """
    fake_playground_env = types.SimpleNamespace(observation_size={"state": (48,)})
    assert GenesisBackend.supports_env(fake_playground_env) is False
    reason = GenesisBackend.skip_reason(fake_playground_env)
    assert reason and "URDF" in reason


def test_skip_reason_is_none_only_when_everything_lines_up(monkeypatch):
    monkeypatch.setattr(GenesisBackend, "available", staticmethod(lambda: False))
    assert "not importable" in GenesisBackend.skip_reason(None)
    monkeypatch.setattr(GenesisBackend, "available", staticmethod(lambda: True))
    monkeypatch.setattr(GenesisBackend, "supports_env", staticmethod(lambda env: True))
    assert GenesisBackend.skip_reason(object()) is None


def test_legged_only_guard():
    manip = types.SimpleNamespace(objective=types.SimpleNamespace(name="manipulation"))
    b = GenesisBackend(object(), None, manip, _space(), (0, 1), 5)
    with pytest.raises(ValueError, match="legged-only"):
        b.simulate(_record())


def test_legacy_record_refused_before_any_engine_work(monkeypatch):
    """No batch_key/env_index -> no exact key -> refuse rather than replay a fabricated
    rollout (the same discipline as MjxBackend / MujocoCBackend)."""
    monkeypatch.setattr(GenesisBackend, "available", staticmethod(lambda: True))
    monkeypatch.setattr(GenesisBackend, "supports_env", staticmethod(lambda env: True))
    b = GenesisBackend(object(), None, None, _space(), (0, 1), 5)
    rec = _record()
    rec.batch_key = None
    with pytest.raises(ValueError, match="exact-repro"):
        b.simulate(rec)


def test_ready_is_a_measurement_not_an_intention():
    """READY is True; the numbers behind it are in this module's docstring and in the
    opt-in tier below. It gates on genesis being installed as well, so a box without
    Genesis degrades to an MJX-only report rather than erroring."""
    assert READY is True
    assert GenesisBackend.READY is READY
    assert GenesisBackend.ready() == (READY and GenesisBackend.available())


# --------------------------------------------------------------------------- #
# Default tier: the numerics that are reimplemented (and therefore need proving)
# --------------------------------------------------------------------------- #
def test_quat_to_R_matches_known_rotations_and_is_orthonormal():
    np.testing.assert_allclose(_quat_to_R((1, 0, 0, 0)), np.eye(3), atol=1e-15)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    # 90 deg about z: x -> y
    R = _quat_to_R((c, 0.0, 0.0, s))
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = _quat_to_R(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_obs_mirror_block_layout():
    """The mirror lays out URDFEnv's blocks in URDFEnv's order, with its conventions:
    gravity/lin/ang in the BODY frame, joint_pos as an offset from the default pose."""
    nu = 2
    R = _quat_to_R((math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)))  # 90 deg z
    qpos = np.array([0, 0, 0.3, 1, 0, 0, 0, 0.7, -0.2])
    qvel = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, -0.5])
    obs = _urdf_obs_numpy(R, qpos, qvel, last_act=np.array([0.1, 0.2]),
                          command=np.array([0.5, 0.0, 0.0]), floating=True,
                          q0=7, v0=6, nu=nu, default_pose=np.array([0.5, 0.0]))
    assert obs.shape == (3 + 3 + 3 + nu * 3 + 3,)
    np.testing.assert_allclose(obs[0:3], R.T @ np.array([0, 0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(obs[3:6], R.T @ qvel[:3], atol=1e-12)   # world lin -> body
    np.testing.assert_allclose(obs[6:9], R.T @ qvel[3:6], atol=1e-12)  # MuJoCo's local omega
    np.testing.assert_allclose(obs[9:11], [0.2, -0.2], atol=1e-12)     # qpos - default_pose
    np.testing.assert_allclose(obs[11:13], [0.5, -0.5], atol=1e-12)
    np.testing.assert_allclose(obs[13:15], [0.1, 0.2], atol=1e-12)
    np.testing.assert_allclose(obs[15:18], [0.5, 0.0, 0.0], atol=1e-12)


def test_obs_mirror_fixed_base_drops_the_base_blocks():
    obs = _urdf_obs_numpy(np.eye(3), np.array([0.4]), np.array([0.1]),
                          last_act=np.array([0.0]), command=np.array([0.0, 0.0, 0.0]),
                          floating=False, q0=0, v0=0, nu=1, default_pose=np.array([0.0]))
    assert obs.shape == (1 + 1 + 1 + 3,)


# --------------------------------------------------------------------------- #
# Opt-in tier: the real two-engine measurement behind READY
# --------------------------------------------------------------------------- #
# A marker, not a module-level skip: the default tier above must still run on a bare box.
# Every genesis / jax / mujoco import below lives INSIDE a fixture or a test, so
# collection stays cheap (importing genesis alone is >60 s here).
requires_genesis = pytest.mark.skipif(
    os.environ.get("BP_RUN_GENESIS") != "1",
    reason="builds the URDF robot in BOTH engines (imports genesis, ~minutes); "
           "set BP_RUN_GENESIS=1 to run (see this module's docstring for XLA_FLAGS)",
)

HORIZON = 20
EARLY = 5
CONTRACT_KEYS = ("trunk_height", "up_proj", "lin_vel", "ang_vel", "command")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    """The URDF quadruped as a real Target + a small RANDOM-weight MLP policy.

    Parity is about the two engines agreeing, not about the policy being good; small
    weights keep the actions mild so the early-agreement window measures the port rather
    than chaotic contact divergence.
    """
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from breakpoint_mjx.adapters import from_params
    from breakpoint_mjx.config import load_theta_space
    from breakpoint_mjx.urdf_target import make_urdf_target

    ts = tmp_path_factory.mktemp("gen") / "theta_space.yaml"
    target = make_urdf_target("miniquad_test", QUADRUPED, theta_space_path=ts)
    env = target.load_env()
    consts = target.resolve_constants(env)
    space = load_theta_space(target.theta_space_path)
    rng = np.random.default_rng(0)
    obs_dim, act_dim = int(env.observation_size), int(env.action_size)
    policy_fn = from_params([
        (rng.normal(size=(obs_dim, 32)) * 0.15, np.zeros(32)),
        (rng.normal(size=(32, act_dim)) * 0.15, np.zeros(act_dim)),
    ], activation="tanh").policy_fn
    return types.SimpleNamespace(target=target, env=env, consts=consts, space=space,
                                 policy_fn=policy_fn, base_model=env.mjx_model)


def _rec(fx, rid="p", **overrides):
    theta = {n: fx.space.dims[n].nominal for n in fx.space.names()}
    theta.update(overrides)
    return FailureRecord(
        id=rid, theta_vector=[theta[n] for n in fx.space.names()], theta_named=theta,
        failure_score=0.9, component_signals={}, rollout_signature=[], seed=0,
        batch_key=[0, 11], env_index=3,
    )


@pytest.fixture(scope="module")
def nominal(fx):
    """Both engines replayed once on an all-nominal theta (module-scoped: each MJX
    re-sim costs an eager XLA compile of the whole scan body, and the first Genesis
    scene costs a taichi kernel compile)."""
    import jax

    from breakpoint_mjx.backends.mjx_backend import MjxBackend

    rec = _rec(fx, "nominal")
    mjx = MjxBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts,
                     fx.base_model, HORIZON)
    m = mjx.simulate(rec, collect_obs_act=True)
    # Two live XLA executables do not fit in a small commit limit; drop the MJX one
    # before Genesis brings up taichi + torch.
    jax.clear_caches()
    g = GenesisBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts,
                       HORIZON).simulate(rec, collect_obs_act=True)
    return rec, m, g


# --- (3) OBSERVATION parity: the mirror IS URDFEnv._obs ---------------------- #
@requires_genesis
def test_obs_mirror_matches_env_obs(fx):
    """``_urdf_obs_numpy`` reproduces ``URDFEnv._obs`` on random states.

    This is the guard that makes the one reimplemented piece safe -- the same discipline
    ``apply_legged_theta_numpy`` follows against ``build_legged_fields``. Tolerance is
    the float32 round-off floor of MJX's own observation build (measured 6.7e-08); an
    ordering, scaling or frame mistake would be an O(1) mismatch on whole channels.
    """
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    env, rep, mjm = fx.env, fx.env.report, fx.env.mj_model
    q0, v0, nu = rep["qpos_joint_start"], rep["qvel_joint_start"], int(env.action_size)
    d = mujoco.MjData(mjm)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(5):
        qp = np.asarray(mjm.qpos0, dtype=float).copy()
        qp[:3] = rng.normal(size=3) * 0.1 + [0, 0, 0.35]
        q = rng.normal(size=4)
        qp[3:7] = q / np.linalg.norm(q)
        qp[q0:] = rng.normal(size=nu) * 0.3
        qv = rng.normal(size=mjm.nv) * 0.5
        d.qpos[:], d.qvel[:] = qp, qv
        mujoco.mj_forward(mjm, d)
        info = {"last_act": jnp.asarray(rng.normal(size=nu)),
                "command": jnp.asarray([0.5, 0.0, 0.0])}
        ref = np.asarray(env._obs(mjx.put_data(mjm, d), info))
        R = np.asarray(d.xmat[rep["root_body_id"]]).reshape(3, 3)
        mine = _urdf_obs_numpy(R, qp, qv, np.asarray(info["last_act"]),
                               np.asarray(info["command"]), floating=True, q0=q0, v0=v0,
                               nu=nu, default_pose=np.asarray(rep["default_pose"]))
        assert ref.shape == mine.shape
        worst = max(worst, float(np.abs(ref - mine).max()))
    assert worst < 1e-6, f"observation mirror drifted from URDFEnv._obs by {worst}"


@requires_genesis
def test_quat_to_R_matches_mujoco_xmat(fx):
    """The Genesis->MuJoCo orientation conversion is the SAME rotation MuJoCo computes,
    so ``up_proj`` and every body-frame observation block mean the same thing on both
    sides (measured floor: 1.1e-16)."""
    import mujoco

    mjm = fx.env.mj_model
    d = mujoco.MjData(mjm)
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        d.qpos[:] = mjm.qpos0
        d.qpos[3:7] = q
        mujoco.mj_forward(mjm, d)
        ref = np.asarray(d.xmat[fx.env.report["root_body_id"]]).reshape(3, 3)
        np.testing.assert_allclose(_quat_to_R(q), ref, atol=1e-12)


# --- (2) RESET/COMMAND + (1) MODEL parity ----------------------------------- #
@requires_genesis
def test_step0_observation_and_action_parity(nominal):
    """Both engines are seeded from the env's OWN ``reset(key)``, so the first
    observation the policy sees -- and therefore its first action -- are identical."""
    _, m, g = nominal
    o_m, o_g = np.asarray(m["obs_state"])[0], np.asarray(g["obs_state"])[0]
    assert o_m.shape == o_g.shape
    np.testing.assert_allclose(o_g, o_m, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(g["action"])[0], np.asarray(m["action"])[0],
                               rtol=1e-4, atol=1e-5)


@requires_genesis
def test_command_is_the_envs_own(nominal):
    """URDFEnv samples the velocity command at reset; the Genesis loop carries that same
    command, so the tracking-error term of the score is computed against the same target."""
    _, m, g = nominal
    np.testing.assert_allclose(np.asarray(g["command"]), np.asarray(m["command"]),
                               rtol=1e-6, atol=1e-6)


@requires_genesis
def test_genesis_model_matches_the_mujoco_model(fx):
    """Genesis's import of the URDF agrees with the MuJoCo model URDFEnv runs on:
    same links, same masses, same actuated-DOF block. A mismatch RAISES at world-build
    time (this asserts the validation runs and passes, not that it is skipped)."""
    from breakpoint_mjx.backends.genesis_backend import _GenesisWorld

    mjm = fx.env.mj_model
    world = _GenesisWorld(fx.env.report["urdf_path"], mjm, fx.env.report, (0.0, 0.0, 1.0))
    assert len(world.link_of_body) == int(mjm.nbody) - 1
    assert world.act_dofs == list(range(6, 6 + int(mjm.nu)))


# --- (4)+(5) trajectory: close EARLY, then free to diverge ------------------ #
@requires_genesis
def test_early_trajectory_tracks_then_is_free_to_diverge(nominal):
    """The two engines track each other over the first few control steps -- which is what
    proves the ctrl mapping, substep count, gains, armature and inertials were ported --
    and are NOT asserted to agree after that. Two different contact solvers integrating a
    legged robot diverge, and that divergence is the product."""
    _, m, g = nominal
    h_m, h_g = np.asarray(m["trunk_height"]), np.asarray(g["trunk_height"])
    u_m, u_g = np.asarray(m["up_proj"]), np.asarray(g["up_proj"])
    np.testing.assert_allclose(h_g[:EARLY], h_m[:EARLY], rtol=0.02)
    np.testing.assert_allclose(u_g[:EARLY], u_m[:EARLY], atol=0.02)
    for k in CONTRACT_KEYS:
        a, b = np.asarray(m[k]), np.asarray(g[k])
        assert a.shape == b.shape and a.shape[0] == HORIZON
        assert np.isfinite(a).all() and np.isfinite(b).all()


@requires_genesis
def test_summary_feeds_the_same_scorer(fx, nominal):
    """A Genesis summary is a first-class input to the SAME
    ``component_signals -> failure_score`` path the search used."""
    from breakpoint_mjx.crossval_engines import resolve_thresholds, score_summary_fn

    _, m, g = nominal
    score = score_summary_fn(resolve_thresholds(fx.target), HORIZON, fx.target.objective)
    s_m, s_g = score(m), score(g)
    assert np.isfinite(s_m) and np.isfinite(s_g) and s_m >= 0.0 and s_g >= 0.0


@requires_genesis
def test_theta_actually_reaches_genesis(fx, nominal):
    """A model-dim theta CHANGES the Genesis trajectory.

    Without this, every Genesis setter could silently be a no-op and every failure would
    "reproduce" -- the exact way a bad port inflates agreement. kp_scale is used because
    it is live on any URDF robot (armature/frictionloss are dead on a URDF, see the
    import report's ``dead_theta_dims``).
    """
    _, _, g_nominal = nominal
    b = GenesisBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, HORIZON)
    g_soft = b.simulate(_rec(fx, "soft", kp_scale=0.55, kd_scale=1.45))
    assert not np.allclose(np.asarray(g_soft["trunk_height"]),
                           np.asarray(g_nominal["trunk_height"]))


@requires_genesis
def test_genesis_backend_is_deterministic(fx):
    """Same record + same key -> bit-identical Genesis re-sim. A re-sim that drifted
    between calls could not be evidence of anything."""
    b = GenesisBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, HORIZON)
    rec = _rec(fx, "det", friction_floor=0.5)
    a, c = b.simulate(rec), b.simulate(rec)
    for k in CONTRACT_KEYS:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(c[k]))
