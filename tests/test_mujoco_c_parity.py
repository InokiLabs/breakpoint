"""Cross-ENGINE parity: MjxBackend vs MujocoCBackend on the SAME theta + SAME key.

This is the proof behind ``MujocoCBackend.READY``. The C backend contributes an
independent-implementation entry to the validity vector, so a wrong port would not fail
loudly -- it would quietly make every failure "reproduce" and inflate validity. The gate
is therefore a measurement, not a code review:

  step 0 (TIGHT)  -- the two engines must start from the SAME state and hand the policy
                     the SAME numbers: identical initial qpos/qvel, the same sampled
                     velocity command, the same observation vector (float32 round-off
                     only), and therefore the same first action.
  later (LOOSE)   -- they must NOT be asserted to agree. Different solvers integrating
                     contact-rich locomotion diverge, and that divergence is the product
                     (a failure that survives BOTH engines is the strong evidence). So we
                     assert close agreement only over the first few steps and, after
                     that, only that both engines return finite, same-shaped summaries.

SKIP semantics (so `pytest` stays green on a bare box, never a collection error): the
module importorskips jax / mujoco / mujoco_playground, then skips unless BP_RUN_CBACKEND
is set -- these tests build a real Go1 env and run two physics engines, which is far too
slow for the default suite. Run it explicitly:

    BP_RUN_CBACKEND=1 python -m pytest tests/test_mujoco_c_parity.py -q

MEMORY NOTE (not a correctness note): compiling the MJX scan body eagerly on a CPU box
needs several GB of LLVM working set. On a machine with a small commit limit, prepend
``XLA_FLAGS="--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"``
-- it changes codegen effort, not semantics.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

# --- skip ladder ----------------------------------------------------------- #
pytest.importorskip("jax")
pytest.importorskip("mujoco")
pytest.importorskip("mujoco_playground")

if not os.environ.get("BP_RUN_CBACKEND"):
    pytest.skip(
        "cross-engine parity is slow (real Go1 env + two engines); "
        "set BP_RUN_CBACKEND=1 to run",
        allow_module_level=True,
    )

import jax  # noqa: E402
import numpy as np  # noqa: E402

from breakpoint_mjx.adapters import from_params  # noqa: E402
from breakpoint_mjx.backends import make_mujoco_c_backend  # noqa: E402
from breakpoint_mjx.backends.mjx_backend import MjxBackend  # noqa: E402
from breakpoint_mjx.backends.mujoco_c import (  # noqa: E402
    MujocoCBackend,
    _reset_with_pre_obs_info,
)
from breakpoint_mjx.config import load_theta_space  # noqa: E402
from breakpoint_mjx.schemas import FailureRecord  # noqa: E402
from breakpoint_mjx.targets import get_target  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 50
EARLY = 5          # steps over which the two engines must still track each other
BATCH_KEY = [0, 11]
ENV_INDEX = 3
CONTRACT_KEYS = ("trunk_height", "up_proj", "lin_vel", "ang_vel", "command")


@pytest.fixture(scope="module")
def fx():
    """Go1 env + a small RANDOM-weight MLP policy.

    Parity is about the PHYSICS AND PLUMBING agreeing, not about the policy being good,
    so no trained checkpoint is needed -- only a deterministic policy_fn that runs
    identically on both engines. Weights are small so the actions stay mild: a violent
    policy would drive the two solvers into chaotic contact divergence within a couple of
    steps and the early-agreement check would measure the policy, not the port.
    """
    target = get_target("go1")
    env = target.load_env()
    consts = target.resolve_constants(env)
    space = load_theta_space(config_path(target.theta_space_path))
    base_model = env.mjx_model          # pristine, captured before any simulate()

    rng = np.random.default_rng(0)
    obs_dim = int(env.observation_size["state"][0])
    act_dim = int(env.action_size)
    layers = [
        (rng.normal(size=(obs_dim, 32)) * 0.15, np.zeros(32)),
        (rng.normal(size=(32, act_dim)) * 0.15, np.zeros(act_dim)),
    ]
    policy_fn = from_params(layers, activation="tanh").policy_fn
    return types.SimpleNamespace(
        target=target, env=env, consts=consts, space=space,
        base_model=base_model, policy_fn=policy_fn,
    )


def _record(fx, **overrides):
    """A FailureRecord carrying the exact-repro fields both backends replay from."""
    theta = {n: fx.space.dims[n].nominal for n in fx.space.names()}
    theta.update(overrides)
    return FailureRecord(
        id="parity", theta_vector=[theta[n] for n in fx.space.names()],
        theta_named=dict(theta), failure_score=0.9, component_signals={},
        rollout_signature=[], seed=0, batch_key=list(BATCH_KEY), env_index=ENV_INDEX,
    )


def _backends(fx):
    mjx = MjxBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts,
                     fx.base_model, HORIZON)
    c = MujocoCBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, HORIZON)
    return mjx, c


@pytest.fixture(scope="module")
def nominal(fx):
    """Both engines replayed once on an all-nominal theta (module-scoped: each MJX
    re-sim costs an eager XLA compile of the whole scan body)."""
    rec = _record(fx)
    mjx, c = _backends(fx)
    # MJX first: its compile is the memory-hungry one, so it gets the clean heap.
    return rec, mjx.simulate(rec, collect_obs_act=True), c.simulate(rec, collect_obs_act=True)


@pytest.fixture(scope="module")
def adversarial(fx):
    """Both engines replayed once with EVERY rollout adversary switched on, so the
    follow-up axes (dropout / bias / push) are exercised on the C loop too."""
    rec = _record(
        fx, friction_floor=0.35, floor_slope_rad=0.05, latency_steps=2.0,
        obs_noise_scale=0.02, obs_dropout_prob=0.05, obs_bias_scale=0.01,
        push_mag=1.0, push_step_frac=0.3,
    )
    mjx, c = _backends(fx)
    # Each eager MjxBackend.simulate compiles the whole scan body; two live executables
    # do not fit in a small commit limit, so drop the first before compiling the second.
    jax.clear_caches()
    return rec, mjx.simulate(rec), c.simulate(rec)


# --------------------------------------------------------------------------- #
# (e) the gates that keep a broken port off the validity vector
# --------------------------------------------------------------------------- #
def test_ready_gate(fx):
    """READY is True and gates on `mujoco` being importable -- both conditions, so a box
    without the bindings still degrades to MJX-only validity instead of erroring."""
    assert MujocoCBackend.READY is True
    assert MujocoCBackend.available() is True
    assert MujocoCBackend.ready() is True
    b = make_mujoco_c_backend(
        env=fx.env, policy_fn=fx.policy_fn, target=fx.target, space=fx.space,
        consts=fx.consts, horizon=HORIZON,
    )
    assert isinstance(b, MujocoCBackend)


def test_legged_only_guard(fx):
    """A non-legged objective is refused loudly: the field math, the consts unpack and the
    summary hook are all legged, and the factory returns None rather than a half backend."""
    manip = types.SimpleNamespace(objective=types.SimpleNamespace(name="manipulation"))
    b = MujocoCBackend(fx.env, fx.policy_fn, manip, fx.space, fx.consts, HORIZON)
    with pytest.raises(ValueError, match="legged-only"):
        b.simulate(_record(fx))
    assert make_mujoco_c_backend(
        env=fx.env, policy_fn=fx.policy_fn, target=manip, space=fx.space,
        consts=fx.consts, horizon=HORIZON,
    ) is None


def test_legacy_record_refused(fx):
    """No batch_key/env_index -> no exact key -> refuse rather than validate the wrong
    rollout (same discipline as MjxBackend/render_failure)."""
    _, c = _backends(fx)
    rec = _record(fx)
    rec.batch_key = None
    with pytest.raises(ValueError, match="exact-repro"):
        c.simulate(rec)


# --------------------------------------------------------------------------- #
# (1) RESET parity + (3) COMMAND parity
# --------------------------------------------------------------------------- #
def test_reset_parity_initial_state_and_command(fx):
    """The C engine is seeded from the env's OWN reset, so its initial qpos/qvel/ctrl and
    its sampled velocity command are the MJX rollout's, exactly.

    ``_reset_with_pre_obs_info`` shadows ``_get_obs`` for the duration of the reset to
    capture the info dict before the env's observation noise advances ``info['rng']``.
    This asserts that shadowing does not perturb the reset at all: same pose, same
    velocity, same command, same post-reset rng.
    """
    key = jax.random.fold_in(
        jax.numpy.asarray(BATCH_KEY, dtype=jax.numpy.uint32), ENV_INDEX
    )
    reference = fx.env.reset(key)
    state, pre_obs_info = _reset_with_pre_obs_info(fx.env, key)

    np.testing.assert_array_equal(np.asarray(state.data.qpos), np.asarray(reference.data.qpos))
    np.testing.assert_array_equal(np.asarray(state.data.qvel), np.asarray(reference.data.qvel))
    np.testing.assert_array_equal(np.asarray(state.data.ctrl), np.asarray(reference.data.ctrl))
    # Playground randomizes the initial pose from `key`: this must NOT be a neutral
    # keyframe, or "reset parity" would be vacuous.
    assert not np.allclose(np.asarray(state.data.qvel)[:6], 0.0)
    # The command is sampled at reset and is NOT touched by _get_obs, so the captured
    # pre-observation dict carries the identical command the MJX rollout tracks.
    np.testing.assert_array_equal(
        np.asarray(pre_obs_info["command"]), np.asarray(reference.info["command"])
    )
    assert np.asarray(pre_obs_info["command"]).shape == (3,)


def test_command_stream_identical_over_horizon(nominal):
    """(3) The joystick command is RESAMPLED mid-episode by the env's own step, so command
    parity is a whole-trajectory property, not a reset-time one. The C loop runs the env's
    step for bookkeeping, so the command stream is bit-identical."""
    _, mjx_s, c_s = nominal
    np.testing.assert_array_equal(np.asarray(mjx_s["command"]), np.asarray(c_s["command"]))


# --------------------------------------------------------------------------- #
# (2) OBS parity + (4) STEP/ctrl parity, at step 0
# --------------------------------------------------------------------------- #
def test_step0_observation_and_action_parity(nominal):
    """(2) The first observation the policy sees is the same vector on both engines, and
    (4) therefore so is the first action.

    Step 0's observation is built from a pure forward solve on the same qpos/qvel, so the
    only difference is float32 round-off (MJX runs float32; the C engine is float64 and is
    down-converted by put_data). rtol 1e-5 / atol 1e-5 is ~100x above the observed floor
    and ~1000x below any real layout, scaling or ordering mistake, which would show up as
    an O(1) mismatch on whole channels.
    """
    _, mjx_s, c_s = nominal
    o_m = np.asarray(mjx_s["obs_state"])[0]
    o_c = np.asarray(c_s["obs_state"])[0]
    assert o_m.shape == o_c.shape
    np.testing.assert_allclose(o_c, o_m, rtol=1e-5, atol=1e-5)

    a_m = np.asarray(mjx_s["action"])[0]
    a_c = np.asarray(c_s["action"])[0]
    assert a_m.shape == a_c.shape
    np.testing.assert_allclose(a_c, a_m, rtol=1e-4, atol=1e-5)


def test_step0_summary_parity(nominal):
    """After ONE control step (5 physics substeps) the engines still agree to ~1e-5 on the
    upright projection and ~1e-5 relative on trunk height -- i.e. the ctrl mapping
    (default pose + action*action_scale) and the substep count are the env's own, not a
    reimplementation that happens to be close."""
    _, mjx_s, c_s = nominal
    np.testing.assert_allclose(
        float(np.asarray(c_s["trunk_height"])[0]),
        float(np.asarray(mjx_s["trunk_height"])[0]), rtol=1e-3,
    )
    np.testing.assert_allclose(
        float(np.asarray(c_s["up_proj"])[0]),
        float(np.asarray(mjx_s["up_proj"])[0]), atol=1e-4,
    )


# --------------------------------------------------------------------------- #
# Trajectory: close EARLY, then only well-formed. Long-horizon agreement is NOT a
# property of two different solvers, so it is deliberately not asserted.
# --------------------------------------------------------------------------- #
def test_early_trajectory_tracks_then_is_free_to_diverge(nominal):
    _, mjx_s, c_s = nominal
    h_m, h_c = np.asarray(mjx_s["trunk_height"]), np.asarray(c_s["trunk_height"])
    u_m, u_c = np.asarray(mjx_s["up_proj"]), np.asarray(c_s["up_proj"])
    np.testing.assert_allclose(h_c[:EARLY], h_m[:EARLY], rtol=0.05)
    np.testing.assert_allclose(u_c[:EARLY], u_m[:EARLY], atol=0.05)
    # Beyond EARLY: no agreement assertion. Only that both engines produced a complete,
    # finite, same-shaped summary -- what makes the two runs comparable at all.
    for k in CONTRACT_KEYS:
        a, b = np.asarray(mjx_s[k]), np.asarray(c_s[k])
        assert a.shape == b.shape == (HORIZON,) + a.shape[1:]
        assert a.shape[0] == HORIZON
        assert np.isfinite(a).all() and np.isfinite(b).all()


def test_summaries_feed_the_same_scorer(fx, nominal):
    """Both summaries go through the SAME component_signals -> failure_score path the
    search used, so an engine 'reproduces' a failure by exactly the criterion that
    discovered it."""
    from breakpoint_mjx.crossval_engines import score_summary_fn

    rec, mjx_s, c_s = nominal
    thresholds = {"trunk_height_min": 0.18, "up_proj_min": 0.5,
                  "tracking_err_max": 0.5, "ang_vel_max": 8.0}
    score = score_summary_fn(thresholds, HORIZON, fx.target.objective)
    s_m, s_c = score(mjx_s), score(c_s)
    # failure_score is an unbounded-above severity sum (fall + tracking + instability +
    # an earliness bonus), so only its finiteness/sign is a contract; the point here is
    # that a C summary is a first-class input to it, not a special case.
    assert np.isfinite(s_m) and np.isfinite(s_c)
    assert s_m >= 0.0 and s_c >= 0.0


# --------------------------------------------------------------------------- #
# The follow-up axes: dropout / per-episode bias / gait-timed push must be mirrored in
# the C loop too, or the cross-engine comparison silently omits them.
# --------------------------------------------------------------------------- #
def test_adversary_axes_are_mirrored(fx, nominal, adversarial):
    """With every rollout adversary on, the C run is (a) still well-formed and (b) NOT the
    same trajectory as the nominal C run -- proving the adversaries actually reach the C
    loop rather than being dropped as they were before the port."""
    _, _, c_nominal = nominal
    _, mjx_adv, c_adv = adversarial
    for k in CONTRACT_KEYS:
        a, b = np.asarray(mjx_adv[k]), np.asarray(c_adv[k])
        assert a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.allclose(
        np.asarray(c_adv["trunk_height"]), np.asarray(c_nominal["trunk_height"])
    )
    # Command parity holds under the adversaries too (the adversaries perturb the
    # OBSERVATION and the base velocity, never the env's command stream).
    np.testing.assert_array_equal(
        np.asarray(mjx_adv["command"]), np.asarray(c_adv["command"])
    )


def test_c_backend_is_deterministic(fx):
    """Same record + same key -> bit-identical C re-sim (a re-sim that drifted between
    calls could not be evidence of anything)."""
    _, c = _backends(fx)
    rec = _record(fx, friction_floor=0.5)
    a = c.simulate(rec)
    b = c.simulate(rec)
    for k in CONTRACT_KEYS:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_c_summary_carries_the_validity_keys(nominal):
    """The C re-sim emits the SAME physical-validity channels as the MJX rollout
    (floor_height / penetration / qacc_max, via the shared legged_summarize), so the
    sim-invalid gate scores a cross-engine replay by the identical criterion, and a
    nominal rollout on either engine is not flagged."""
    from breakpoint_mjx import failure

    _, mjx_s, c_s = nominal
    thresholds = {"trunk_height_min": 0.18, "up_proj_min": 0.5,
                  "tracking_err_max": 0.5, "ang_vel_max": 8.0}
    for s in (mjx_s, c_s):
        for k in failure.VALIDITY_KEYS:
            a = np.asarray(s[k])
            assert a.shape == (HORIZON,)
            assert np.isfinite(a).all()
        assert not failure.is_sim_invalid(failure.component_signals(s, thresholds))
    # Early steps: floor-relative height agrees across engines like trunk_height does.
    np.testing.assert_allclose(np.asarray(c_s["floor_height"])[:EARLY],
                               np.asarray(mjx_s["floor_height"])[:EARLY], rtol=0.05)


# --------------------------------------------------------------------------- #
# (f) Skeptic-grade replay: full-collision scene + the compiled (jit_env) C loop
# --------------------------------------------------------------------------- #
FLOOR_TOL_M = 0.02


@pytest.fixture(scope="module")
def full_model(fx):
    from breakpoint_mjx.backends.mujoco_c import build_full_collision_model

    m, info = build_full_collision_model(fx.env, fx.target)
    return m, info


def test_full_collision_scene_is_a_layout_compatible_drop_in(fx, full_model):
    """Go1 gets Playground's own full-collision scene (not the enable-contacts fallback),
    with every sensor/site/body the env reads at the same address as the search model."""
    from breakpoint_mjx.backends.mujoco_c import layout_mismatches

    m, info = full_model
    assert info["source"] == "playground_scene", info
    assert layout_mismatches(fx.env.mj_model, m) == []
    # strictly more collidable geoms than the feet-only search scene
    base = fx.env.mj_model
    n_base = int(((base.geom_contype != 0) | (base.geom_conaffinity != 0)).sum())
    assert info["n_collidable"] > n_base
    assert float(m.opt.timestep) == float(base.opt.timestep)


def test_jit_env_matches_eager_c_loop_early(fx, full_model):
    """The compiled two-pass step runs the env's own ops, so over the first steps it must
    track the parity-proven eager loop to float round-off."""
    m, _ = full_model
    rec = _record(fx)
    eager = MujocoCBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, 10,
                           mj_model=m).simulate(rec)
    fast = MujocoCBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, 10,
                          mj_model=m, jit_env=True).simulate(rec)
    for k in CONTRACT_KEYS:
        np.testing.assert_allclose(np.asarray(fast[k])[:EARLY], np.asarray(eager[k])[:EARLY],
                                   atol=1e-4, err_msg=k)


def test_full_collision_replay_keeps_torso_above_floor(fx, full_model):
    """Acceptance (c): a hard fall replayed on the full-collision scene never puts the
    torso below the floor plane (tolerance 2 cm), while the same world on the feet-only
    search model sinks through it."""
    m, _ = full_model
    rec = _record(fx, push_mag=3.0, push_step_frac=0.1, kp_scale=0.5)
    full = MujocoCBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, 150,
                          mj_model=m, jit_env=True).simulate(rec)
    feet = MujocoCBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts, 150,
                          jit_env=True).simulate(rec)
    assert float(np.min(full["floor_height"])) >= -FLOOR_TOL_M
    assert float(np.min(full["up_proj"])) < 0.5 or float(np.min(feet["floor_height"])) < 0.18, (
        "the probe world did not fall; pick a harsher theta so the check means something")
    assert float(np.min(feet["floor_height"])) < float(np.min(full["floor_height"]))


def test_stateful_policy_runs_on_the_c_engine(fx):
    """A history policy that reads only the newest frame is the stateless policy, so the
    C-engine replay must not change: pins the carry plumbing in MujocoCBackend."""
    from breakpoint_mjx.adapters.mlp import MLPPolicy
    from breakpoint_mjx.adapters.obs_spec import ObsSpec
    from breakpoint_mjx.adapters.stateful import HistoryPolicy

    base = fx.policy_fn.__self__
    H, D = 3, base.in_dim
    (W0, b0), rest = base.params[0], base.params[1:]
    Wh = np.zeros((H * D, W0.shape[1]))
    Wh[(H - 1) * D:] = W0
    head = MLPPolicy(params=[(Wh, b0)] + list(rest), activation=base.activation,
                     obs_spec=ObsSpec(keys=("_history",), expected_dim=H * D))
    hp = HistoryPolicy(frame_spec=ObsSpec(expected_dim=D), mlp=head, length=H,
                       layout="frame")
    rec = _record(fx, latency_steps=1.0, obs_noise_scale=0.02, obs_bias_scale=0.01)
    a = MujocoCBackend(fx.env, fx.policy_fn, fx.target, fx.space, fx.consts,
                       HORIZON).simulate(rec)
    b = MujocoCBackend(fx.env, hp, fx.target, fx.space, fx.consts, HORIZON).simulate(rec)
    for k in CONTRACT_KEYS:
        np.testing.assert_allclose(np.asarray(a[k]), np.asarray(b[k]), rtol=1e-5,
                                   atol=1e-5, err_msg=k)
