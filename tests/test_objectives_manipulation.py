"""CPU unit tests for T7: the per-target objective abstraction + the Franka Panda
MANIPULATION target class. No JAX/GPU: the pure objective (component_signals /
failure_score / behavior_descriptor / rollout_signature), the fixed-base validator, the
arm field math (via a numpy-backed fake model), the config parse, the registry entry, and
-- crucially -- a byte-identical regression that the LEGGED path is unchanged after the
refactor. The GPU summarize hook + rollout are Colab-validated.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx import failure
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.objectives import LEGGED_OBJECTIVE, Objective
from breakpoint_mjx.objectives.legged import legged_summarize
from breakpoint_mjx.objectives import manipulation as M
from breakpoint_mjx.search import scores_from_trajectories
from breakpoint_mjx.targets import TARGETS, Target, get_target
from breakpoint_mjx.targets._manipulation import (
    MANIPULATION_MODEL_FIELDS,
    build_manipulation_fields,
    validate_manipulation_model,
)

from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]

MANIP_THR = dict(
    success_dist=0.05, grasp_dist=0.06, drop_dist=0.15, joint_margin_min=0.02,
)
T = 150


def _const(x, val):
    return np.full(T, float(val))


def _manip_traj(obj_to_target, ee_to_obj, obj_height=0.1, joint_margin=0.5, reached=0.0):
    """Build a synthetic manipulation summary traj. Each arg is a scalar (constant fill)
    or an array of length T."""
    def arr(v):
        return np.asarray(v, dtype=float) if np.ndim(v) else _const(None, v)
    return dict(
        obj_to_target=arr(obj_to_target),
        ee_to_obj=arr(ee_to_obj),
        obj_height=arr(obj_height),
        joint_margin=arr(joint_margin),
        reached=arr(reached),
    )


# --------------------------------------------------------------------------- #
# Registry + target metadata (manipulation vertical)
# --------------------------------------------------------------------------- #
def test_get_target_returns_panda():
    p = get_target("panda")
    assert isinstance(p, Target)
    assert p.name == "panda"
    assert p.env_name == "PandaPickCube"
    assert p.checkpoint_name == "panda_policy.pkl"
    assert "panda" in TARGETS


def test_panda_uses_manipulation_objective_and_domain():
    p = get_target("panda")
    assert p.objective is M.MANIPULATION_OBJECTIVE
    assert p.objective.name == "manipulation"
    assert p.domain == "manipulation"
    # Its own failure modes, distinct from the legged ones.
    assert p.objective.failure_modes == ("missed_target", "dropped_object", "joint_limit", "timeout")
    assert p.objective.failure_modes != LEGGED_OBJECTIVE.failure_modes


def test_panda_model_fields_are_arm_appropriate():
    p = get_target("panda")
    assert p.model_fields == MANIPULATION_MODEL_FIELDS
    # NOT the legged fields (no body_ipos/com or geom_quat floor-slope).
    assert "body_ipos" not in p.model_fields
    assert "geom_quat" not in p.model_fields


def test_panda_thresholds_are_manipulation_keys():
    p = get_target("panda")
    assert p.thresholds is not None
    for k in ("success_dist", "grasp_dist", "drop_dist", "joint_margin_min"):
        assert k in p.thresholds


def test_panda_build_fields_signature():
    import inspect
    p = get_target("panda")
    assert list(inspect.signature(p.build_fields).parameters) == [
        "scalars", "base_model", "consts",
    ]


# --------------------------------------------------------------------------- #
# Config parse + honest-to-the-task theta space
# --------------------------------------------------------------------------- #
def test_panda_theta_space_parses_and_is_honest():
    p = get_target("panda")
    assert p.theta_space_path == "configs/panda/theta_space.yaml"
    space = load_theta_space(config_path(p.theta_space_path))
    names = set(space.names())
    # Manipulation-relevant dims present.
    for n in ("object_mass_scale", "object_friction_scale", "floor_friction_scale",
              "kp_scale", "joint_frictionloss_scale", "joint_armature_scale",
              "obs_noise_scale", "obs_dropout_prob", "obs_bias_scale", "latency_steps"):
        assert n in names, n
    # Locomotion-only dims OMITTED (honest to a fixed-base arm).
    for n in ("torso_mass_add_kg", "com_offset_x_m", "com_offset_y_m", "com_offset_z_m",
              "floor_slope_rad", "push_mag", "push_step_frac"):
        assert n not in names, n
    # The model dims exactly match the arm field-math dims.
    model_dims = [n for n in space.names() if space.dims[n].kind == "model"]
    assert set(model_dims) == {
        "object_mass_scale", "object_friction_scale", "floor_friction_scale",
        "kp_scale", "joint_frictionloss_scale", "joint_armature_scale",
    }


# --------------------------------------------------------------------------- #
# Fixed-base validator (mirror of the legged one)
# --------------------------------------------------------------------------- #
def _arm_model(jnt0=3, nu=8):
    """A minimal fake model for the validator: jnt_type[0] and an actuator matrix."""
    return types.SimpleNamespace(
        jnt_type=np.array([jnt0] + [3] * 5),
        actuator_gainprm=np.zeros((nu, 3)),
    )


def test_validate_manipulation_passes_on_fixed_base_arm():
    validate_manipulation_model(_arm_model(jnt0=3, nu=8))   # hinge joint 0 -> ok
    validate_manipulation_model(_arm_model(jnt0=2, nu=8))   # slide joint 0 -> ok


def test_validate_manipulation_rejects_free_base():
    # jnt_type[0] == 0 (mjJNT_FREE): a floating base, i.e. a legged robot -> hard fail.
    with pytest.raises(ValueError, match=r"FIXED-BASE|mjJNT_FREE|free"):
        validate_manipulation_model(_arm_model(jnt0=0, nu=8))


def test_validate_manipulation_rejects_no_actuators():
    with pytest.raises(ValueError, match="actuator"):
        validate_manipulation_model(_arm_model(jnt0=3, nu=0))


# --------------------------------------------------------------------------- #
# Arm field math (numpy-backed fake of the JAX .at[...] API)
# --------------------------------------------------------------------------- #
class _AtProxy:
    def __init__(self, arr, idx):
        self._arr, self._idx = arr, idx

    def set(self, v):
        new = self._arr.copy(); new[self._idx] = v; return _FakeArr(new)

    def add(self, v):
        new = self._arr.copy(); new[self._idx] = new[self._idx] + v; return _FakeArr(new)

    def multiply(self, v):
        new = self._arr.copy(); new[self._idx] = new[self._idx] * v; return _FakeArr(new)


class _AtIndexer:
    def __init__(self, fa):
        self._fa = fa

    def __getitem__(self, idx):
        return _AtProxy(self._fa.arr, idx)


class _FakeArr:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    @property
    def at(self):
        return _AtIndexer(self)

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, idx):
        return self.arr[idx]


def _fake_manip_model(nbody=4, ngeom=3, nu=8, ndof=15):
    rng = np.random.default_rng(1)
    return types.SimpleNamespace(
        body_mass=_FakeArr(rng.uniform(0.1, 3.0, size=(nbody,))),
        geom_friction=_FakeArr(rng.uniform(0.5, 1.0, size=(ngeom, 3))),
        actuator_gainprm=_FakeArr(rng.uniform(10.0, 100.0, size=(nu, 3))),
        actuator_biasprm=_FakeArr(rng.uniform(-100.0, 0.0, size=(nu, 3))),
        dof_frictionloss=_FakeArr(rng.uniform(0.0, 0.5, size=(ndof,))),
        dof_armature=_FakeArr(rng.uniform(0.01, 0.1, size=(ndof,))),
    )


def _manip_consts():
    return dict(
        obj_body_id=2,
        obj_geom_ids=np.array([1]),
        floor_geom_id=0,
        arm_dof_ids=np.array([0, 1, 2]),        # arm dofs FIRST (not [6:])
        arm_actuator_ids=np.array([0, 1, 2, 3, 4, 5, 6]),  # 7 arm actuators; #7 = gripper
    )


def _manip_scalars():
    return dict(
        object_mass_scale=1.5, object_friction_scale=0.5, floor_friction_scale=1.3,
        kp_scale=1.2, joint_frictionloss_scale=1.4, joint_armature_scale=1.6,
    )


def test_build_manipulation_fields_returns_exactly_model_fields():
    out = build_manipulation_fields(_manip_scalars(), _fake_manip_model(), _manip_consts())
    assert set(out.keys()) == set(MANIPULATION_MODEL_FIELDS)


def test_build_manipulation_fields_applies_arm_math_at_right_indices():
    base = _fake_manip_model()
    c = _manip_consts()
    s = _manip_scalars()
    out = build_manipulation_fields(s, base, c)

    # object payload mass scaled; other bodies untouched.
    assert out["body_mass"].arr[2] == pytest.approx(base.body_mass.arr[2] * s["object_mass_scale"])
    assert out["body_mass"].arr[3] == pytest.approx(base.body_mass.arr[3])

    # object geom friction[.,0] scaled; floor geom friction[0,0] scaled; other cols untouched.
    assert out["geom_friction"].arr[1, 0] == pytest.approx(
        base.geom_friction.arr[1, 0] * s["object_friction_scale"])
    assert out["geom_friction"].arr[0, 0] == pytest.approx(
        base.geom_friction.arr[0, 0] * s["floor_friction_scale"])
    assert out["geom_friction"].arr[1, 1] == pytest.approx(base.geom_friction.arr[1, 1])

    # arm Kp: gainprm[:,0] and biasprm[:,1] scaled by kp_scale on the ARM actuators ONLY
    # ([0..6]); the gripper actuator (index 7) is UNTOUCHED (grip strength not perturbed).
    np.testing.assert_allclose(out["actuator_gainprm"].arr[0:7, 0],
                               base.actuator_gainprm.arr[0:7, 0] * s["kp_scale"])
    np.testing.assert_allclose(out["actuator_biasprm"].arr[0:7, 1],
                               base.actuator_biasprm.arr[0:7, 1] * s["kp_scale"])
    assert out["actuator_gainprm"].arr[7, 0] == pytest.approx(base.actuator_gainprm.arr[7, 0])
    assert out["actuator_biasprm"].arr[7, 1] == pytest.approx(base.actuator_biasprm.arr[7, 1])

    # arm dof friction/armature scaled on the arm dofs ONLY ([0,1,2]); rest untouched
    # (NOT the legged [6:] slice -- the arm dofs are FIRST here).
    np.testing.assert_allclose(out["dof_frictionloss"].arr[[0, 1, 2]],
                               base.dof_frictionloss.arr[[0, 1, 2]] * s["joint_frictionloss_scale"])
    np.testing.assert_allclose(out["dof_frictionloss"].arr[3:],
                               base.dof_frictionloss.arr[3:])
    np.testing.assert_allclose(out["dof_armature"].arr[[0, 1, 2]],
                               base.dof_armature.arr[[0, 1, 2]] * s["joint_armature_scale"])
    np.testing.assert_allclose(out["dof_armature"].arr[3:], base.dof_armature.arr[3:])


# --------------------------------------------------------------------------- #
# Manipulation objective: component_signals / failure_score / behavior_descriptor
# --------------------------------------------------------------------------- #
def _mode(name):
    return M.FAILURE_MODES.index(name)


def test_success_is_not_a_failure():
    """Box reaches the target, grasped, joints safe -> low score, success flag set."""
    o2t = np.linspace(0.4, 0.02, T)     # ends inside success_dist=0.05
    e2o = np.linspace(0.3, 0.01, T)     # gripper reaches box, stays
    traj = _manip_traj(o2t, e2o, obj_height=0.15, joint_margin=0.5, reached=1.0)
    sig = M.component_signals(traj, MANIP_THR)
    assert sig["success"] == 1.0
    assert sig["dropped"] == 0.0
    assert sig["joint_limit_hit"] == 0.0
    score = M.failure_score(sig, horizon=T)
    assert score < 0.5, f"success scored too high: {score}"


def test_missed_target_is_a_failure():
    traj = _manip_traj(obj_to_target=0.4, ee_to_obj=0.4)   # never reaches, never grasps
    sig = M.component_signals(traj, MANIP_THR)
    assert sig["success"] == 0.0
    assert M.failure_score(sig, horizon=T) >= 0.5
    mode, ttf = M.behavior_descriptor(traj, MANIP_THR, horizon=T)
    assert mode == _mode("missed_target"), M.FAILURE_MODES[mode]
    assert ttf == 1.0                                      # pure miss: no timed event


def test_dropped_object_is_a_failure():
    """Grasped (ee < grasp_dist) then dropped (ee grows past drop_dist), never reaches."""
    e2o = np.concatenate([np.linspace(0.3, 0.01, 60), np.full(40, 0.01), np.full(50, 0.4)])
    o2t = np.concatenate([np.linspace(0.3, 0.08, 100), np.full(50, 0.3)])  # min 0.08 > 0.05
    traj = _manip_traj(o2t, e2o, obj_height=0.2, joint_margin=0.5, reached=1.0)
    sig = M.component_signals(traj, MANIP_THR)
    assert sig["grasped"] == 1.0
    assert sig["dropped"] == 1.0
    assert sig["success"] == 0.0
    assert M.failure_score(sig, horizon=T) >= 0.5
    mode, ttf = M.behavior_descriptor(traj, MANIP_THR, horizon=T)
    assert mode == _mode("dropped_object"), M.FAILURE_MODES[mode]
    assert 0.0 < ttf < 1.0                                 # timed drop event


def test_joint_limit_is_a_failure_even_when_task_succeeds():
    """Task succeeds but an arm joint is driven to its limit -> still flagged, mode joint_limit."""
    o2t = np.linspace(0.4, 0.02, T)                        # reaches target (success)
    e2o = np.linspace(0.3, 0.01, T)
    jm = np.full(T, 0.5); jm[30:60] = 0.0                  # joint at limit for a stretch
    traj = _manip_traj(o2t, e2o, obj_height=0.15, joint_margin=jm, reached=1.0)
    sig = M.component_signals(traj, MANIP_THR)
    assert sig["success"] == 1.0
    assert sig["joint_limit_hit"] == 1.0
    assert M.failure_score(sig, horizon=T) >= 0.5
    mode, ttf = M.behavior_descriptor(traj, MANIP_THR, horizon=T)
    assert mode == _mode("joint_limit"), M.FAILURE_MODES[mode]
    assert ttf == pytest.approx(30 / T, abs=1e-6)


def test_nonfinite_is_max_failure_and_dropped_mode():
    traj = _manip_traj(obj_to_target=0.4, ee_to_obj=0.4)
    traj["obj_to_target"][70] = np.nan
    sig = M.component_signals(traj, MANIP_THR)
    assert sig["nonfinite"] == 1.0
    assert M.failure_score(sig, horizon=T) == pytest.approx(1.0)
    mode, _ = M.behavior_descriptor(traj, MANIP_THR, horizon=T)
    assert mode == _mode("dropped_object"), M.FAILURE_MODES[mode]


def test_descriptor_always_valid_and_precomputed_sig_matches():
    trajs = [
        _manip_traj(0.4, 0.4),
        _manip_traj(np.linspace(0.4, 0.02, T), np.linspace(0.3, 0.01, T), reached=1.0),
    ]
    for traj in trajs:
        mode, ttf = M.behavior_descriptor(traj, MANIP_THR, horizon=T)
        assert 0 <= mode < len(M.FAILURE_MODES)
        assert 0.0 <= ttf <= 1.0
        sig = M.component_signals(traj, MANIP_THR)
        assert M.behavior_descriptor(traj, MANIP_THR, T, sig=sig) == \
               M.behavior_descriptor(traj, MANIP_THR, T)


def test_manip_signature_length():
    traj = _manip_traj(0.4, 0.4)
    assert len(M.rollout_signature(traj)) == M.SIGNATURE_LEN


def test_obj_reach_norm_is_single_source():
    """The graded-miss normalizer is a module constant used by BOTH failure_score and
    behavior_descriptor, and is NOT a Target.thresholds override -- so injecting a bogus
    'obj_reach_norm' threshold changes NOTHING (score + descriptor identical)."""
    assert hasattr(M, "OBJ_REACH_NORM")
    assert "obj_reach_norm" not in M._DEFAULTS
    assert "obj_reach_norm" not in (get_target("panda").thresholds or {})

    traj = _manip_traj(obj_to_target=0.15, ee_to_obj=0.4)   # partial miss, no catastrophe
    sig_a = M.component_signals(traj, MANIP_THR)
    sig_b = M.component_signals(traj, {**MANIP_THR, "obj_reach_norm": 999.0})
    assert M.failure_score(sig_a, T) == M.failure_score(sig_b, T)
    assert M.behavior_descriptor(traj, MANIP_THR, T) == \
           M.behavior_descriptor(traj, {**MANIP_THR, "obj_reach_norm": 999.0}, T)

    # The graded-miss term in the score uses OBJ_REACH_NORM: a run whose closest approach
    # equals OBJ_REACH_NORM contributes a full 0.20 graded-miss term.
    at_norm = _manip_traj(obj_to_target=M.OBJ_REACH_NORM, ee_to_obj=0.4)
    sig = M.component_signals(at_norm, MANIP_THR)
    # missed target (0.5) + full graded miss (0.2) = 0.7 (no drop/joint/time event).
    assert M.failure_score(sig, T) == pytest.approx(0.7, abs=1e-9)


def test_scores_from_trajectories_with_manipulation_objective():
    """Batched (N, T) manip traj through the shared scores helper: a dropped-object env
    scores higher than a success env, using the manipulation objective."""
    def batch(*trajs):
        keys = trajs[0].keys()
        return {k: np.stack([t[k] for t in trajs]) for k in keys}

    success = _manip_traj(np.linspace(0.4, 0.02, T), np.linspace(0.3, 0.01, T), reached=1.0)
    dropped = _manip_traj(
        np.concatenate([np.linspace(0.3, 0.08, 100), np.full(50, 0.3)]),
        np.concatenate([np.linspace(0.3, 0.01, 60), np.full(40, 0.01), np.full(50, 0.4)]),
        obj_height=0.2, reached=1.0,
    )
    traj = batch(success, dropped)
    scores, sigs, comps = scores_from_trajectories(
        traj, MANIP_THR, T, objective=M.MANIPULATION_OBJECTIVE)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]
    assert comps[1]["dropped"] == 1.0 and comps[0]["success"] == 1.0
    assert len(sigs) == 2 and len(sigs[0]) == M.SIGNATURE_LEN


# --------------------------------------------------------------------------- #
# LEGGED byte-identical regression: the refactor must not move the legged path
# --------------------------------------------------------------------------- #
_LEGGED_THR = dict(trunk_height_min=0.18, up_proj_min=0.5, tracking_err_max=0.5, ang_vel_max=8.0)


def _legged_batch(N=2, Tl=50):
    trunk_height = np.stack([np.full(Tl, 0.28), np.full(Tl, 0.05)])
    up_proj = np.stack([np.full(Tl, 1.0), np.full(Tl, 0.1)])
    command = np.zeros((N, Tl, 3)); command[:, :, 0] = 1.0
    lin_vel = np.zeros((N, Tl, 3)); lin_vel[0, :, 0] = 0.95
    ang_vel = np.zeros((N, Tl, 3))
    return dict(trunk_height=trunk_height, up_proj=up_proj, lin_vel=lin_vel,
               ang_vel=ang_vel, command=command)


def test_legged_objective_wraps_the_exact_failure_functions():
    """The legged objective's hooks ARE the failure.py functions (identity), so routing
    through it cannot change legged behavior."""
    assert LEGGED_OBJECTIVE.component_signals is failure.component_signals
    assert LEGGED_OBJECTIVE.failure_score is failure.failure_score
    assert LEGGED_OBJECTIVE.behavior_descriptor is failure.behavior_descriptor
    assert LEGGED_OBJECTIVE.rollout_signature is failure.rollout_signature
    assert LEGGED_OBJECTIVE.failure_modes is failure.FAILURE_MODES
    assert isinstance(LEGGED_OBJECTIVE, Objective)


def test_scores_from_trajectories_legged_default_is_byte_identical():
    """scores_from_trajectories with the default (None) objective == explicitly passing the
    legged objective == the pre-T7 direct failure.* computation, bit-for-bit."""
    traj = _legged_batch()
    s_default, sig_default, c_default = scores_from_trajectories(traj, _LEGGED_THR, 50)
    s_legged, sig_legged, c_legged = scores_from_trajectories(
        traj, _LEGGED_THR, 50, objective=LEGGED_OBJECTIVE)
    np.testing.assert_array_equal(s_default, s_legged)
    assert sig_default == sig_legged
    assert c_default == c_legged
    # And equal to computing failure.* directly on each row.
    for i in range(2):
        one = {k: np.asarray(v[i]) for k, v in traj.items()}
        c = failure.component_signals(one, _LEGGED_THR)
        assert c_default[i] == c
        assert s_default[i] == pytest.approx(failure.failure_score(c, 50))


def test_go1_barkour_g1_all_route_through_the_legged_objective():
    for name in ("go1", "barkour", "g1"):
        t = get_target(name)
        assert t.objective is LEGGED_OBJECTIVE
        assert t.domain == "locomotion"


def test_legged_summarize_matches_the_former_inline_summary():
    """legged_summarize reproduces the exact 5-key dict rollout_one used to build inline,
    for a fake state (numpy stand-in for the traced State)."""
    torso = 1
    xmat = np.zeros((3, 9))
    # A non-trivial rotation matrix for body `torso` (row-major 3x3), rest identity-ish.
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    xmat[torso] = R.reshape(9)
    qpos = np.array([0.1, 0.2, 0.31, 0.0, 0.0, 0.0, 1.0])
    qvel = np.array([0.5, -0.3, 0.2, 0.01, 0.02, 0.03])
    command = np.array([1.0, 0.0, 0.2])
    state = types.SimpleNamespace(
        data=types.SimpleNamespace(xmat=xmat, qpos=qpos, qvel=qvel),
        info={"command": command},
    )
    out = legged_summarize(state, (0, torso))
    assert out["trunk_height"] == pytest.approx(0.31)
    assert out["up_proj"] == pytest.approx(R[2, 2])
    np.testing.assert_allclose(out["lin_vel"], R.T @ qvel[:3])
    np.testing.assert_allclose(out["ang_vel"], qvel[3:6])
    np.testing.assert_allclose(out["command"], command)


# --------------------------------------------------------------------------- #
# nominal_performance (the hardening Pareto performance axis for manipulation)
# --------------------------------------------------------------------------- #
def _sig(success=1.0, min_o2t=0.0, dropped=0.0, joint=0.0, nonfinite=0.0):
    return {
        "success": success,
        "min_obj_to_target": min_o2t,
        "dropped": dropped,
        "joint_limit_hit": joint,
        "nonfinite": nonfinite,
    }


def test_manip_nominal_performance_success_is_one():
    from breakpoint_mjx.objectives.manipulation import nominal_performance

    out = nominal_performance([_sig(success=1.0, min_o2t=0.0)], {})
    assert out["nominal_reward"] == pytest.approx(1.0)
    assert out["success_rate"] == pytest.approx(1.0)


def test_manip_nominal_performance_graded_near_miss_beats_far_miss():
    from breakpoint_mjx.objectives.manipulation import nominal_performance

    near = nominal_performance([_sig(success=0.0, min_o2t=0.03)], {})
    far = nominal_performance([_sig(success=0.0, min_o2t=0.50)], {})
    assert near["nominal_reward"] > far["nominal_reward"]
    # A miss beyond OBJ_REACH_NORM earns zero closeness credit.
    assert far["nominal_reward"] == pytest.approx(0.0)


def test_manip_nominal_performance_catastrophe_zeroes_the_rollout():
    from breakpoint_mjx.objectives.manipulation import nominal_performance

    # Succeeded but smashed a joint into its stop: no partial credit for that rollout.
    out = nominal_performance([_sig(success=1.0, joint=1.0)], {})
    assert out["nominal_reward"] == pytest.approx(0.0)
    assert out["joint_limit_rate"] == pytest.approx(1.0)
    drop = nominal_performance([_sig(success=1.0, dropped=1.0)], {})
    assert drop["nominal_reward"] == pytest.approx(0.0)
    blow = nominal_performance([_sig(success=1.0, nonfinite=1.0)], {})
    assert blow["nominal_reward"] == pytest.approx(0.0)


def test_manip_nominal_performance_mixed_mean_and_empty():
    from breakpoint_mjx.objectives.manipulation import nominal_performance

    out = nominal_performance([_sig(success=1.0), _sig(success=0.0, min_o2t=0.50)], {})
    assert out["nominal_reward"] == pytest.approx(0.5)  # mean of 1.0 and 0.0
    z = nominal_performance([], {})
    assert z["nominal_reward"] == 0.0 and z["success_rate"] == 0.0


def test_manipulation_objective_carries_the_hook():
    from breakpoint_mjx.objectives.manipulation import (
        MANIPULATION_OBJECTIVE,
        nominal_performance,
    )

    assert MANIPULATION_OBJECTIVE.nominal_performance is nominal_performance
