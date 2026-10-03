"""Tests for the bring-your-own-robot URDF importer (``breakpoint_mjx/urdf_target.py``).

Two tiers, following the convention in ``tests/test_gpu_smoke.py``:

* **Default tier** -- everything that is pure XML injection + one mujoco (C) compile:
  the audit's fail-loud guards, the freejoint/actuator/floor/camera injection, mass
  conservation, the measured standing height, the derived theta space, and the YAML
  round-trip. These need ``mujoco`` and ``numpy`` only, so they run in the normal suite
  (``importorskip`` keeps a mujoco-less box on SKIPPED, never a collection error).
* **Opt-in tier (``BP_RUN_URDF=1``)** -- anything that builds an MJX env or a jitted
  evaluator. Eager MJX compilation is minutes-and-gigabytes on a CPU box, so it is gated,
  and every fixture that touches it is FUNCTION-scoped: ``rollout_one`` mutates
  ``env._mjx_model``, so a shared env would leak traced models between tests.

      BP_RUN_URDF=1 XLA_FLAGS="--xla_backend_optimization_level=0 \
          --xla_llvm_disable_expensive_passes=true" python -m pytest tests/test_urdf_target.py -q
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest

# mujoco is the only hard dependency of the default tier (no jax).
from breakpoint_mjx.paths import config_path

mujoco = pytest.importorskip("mujoco")

from breakpoint_mjx.config import load_theta_space  # noqa: E402
from breakpoint_mjx.urdf_target import (  # noqa: E402
    FLOOR_GEOM_NAME,
    OBS_CONVENTION_WARNING,
    TRACK_CAMERA_NAME,
    URDFImportError,
    derive_theta_space,
    derive_thresholds,
    format_report,
    load_urdf_model,
    make_urdf_target,
    urdf_to_mjcf,
    write_theta_space_yaml,
)

FIXTURES = Path(__file__).parent / "fixtures" / "urdf"
QUADRUPED = FIXTURES / "quadruped.urdf"

# Ground truth read straight off tests/fixtures/urdf/quadruped.urdf.
URDF_MASS = 8.2                 # 5.0 trunk + 4 * (0.5 thigh + 0.3 calf)
N_ACTUATED = 8                  # 4 legs x (hip, knee)
GEOMETRIC_STANDING_HEIGHT = 0.32  # 0.15 thigh + 0.15 calf + 0.02 foot radius
EFFORT = 24.0                   # every joint's URDF effort limit


# --------------------------------------------------------------------------- #
# Session-scoped, jax-free: one compile shared by the whole default tier.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quad():
    """(mj_model, report) for the good quadruped fixture. mujoco-C only, ~0.5 s."""
    return load_urdf_model(QUADRUPED)


# =========================================================================== #
# (1) FAIL-LOUD: a description that would compile into a meaningless model
#     must raise, with the specific reason in the message.
# =========================================================================== #
@pytest.mark.parametrize(
    "fixture, must_mention",
    [
        ("no_collision.urdf", ["collision", "visual"]),
        ("zero_mass.urdf", ["ZERO total mass", "mass"]),
        ("missing_mesh.urdf", ["no_such_mesh.stl", "meshdir", "Searched"]),
        ("mimic_arm.urdf", ["mimic", "finger_right_joint"]),
    ],
)
def test_broken_urdf_raises_with_actionable_message(fixture, must_mention):
    with pytest.raises(URDFImportError) as ei:
        load_urdf_model(FIXTURES / fixture)
    msg = str(ei.value)
    for token in must_mention:
        assert token in msg, f"{fixture}: message does not mention {token!r}:\n{msg}"


def test_missing_file_raises():
    with pytest.raises(URDFImportError, match="URDF not found"):
        load_urdf_model(FIXTURES / "does_not_exist.urdf")


def test_xacro_template_is_rejected_with_the_expand_command(tmp_path):
    """A xacro template is a TEMPLATE; guessing at it would produce a wrong robot."""
    p = tmp_path / "robot.urdf"
    p.write_text(
        '<robot name="x" xmlns:xacro="http://www.ros.org/wiki/xacro">'
        '<link name="a"><inertial><mass value="${m}"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>'
        "</robot>",
        encoding="utf-8",
    )
    with pytest.raises(URDFImportError) as ei:
        load_urdf_model(p)
    assert "xacro" in str(ei.value) and "ros2 run xacro" in str(ei.value)


def test_planar_joint_is_rejected(tmp_path):
    """MuJoCo silently expands a planar joint into three joints (j_TX/j_TY/j_RZ), which
    breaks the one-dof-per-actuated-joint layout the legged field math slices on."""
    p = tmp_path / "planar.urdf"
    p.write_text(
        '<robot name="p">'
        '<link name="a"><inertial><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
        "<collision><geometry><box size=\"0.1 0.1 0.1\"/></geometry></collision></link>"
        '<link name="b"><inertial><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
        "<collision><geometry><box size=\"0.1 0.1 0.1\"/></geometry></collision></link>"
        '<joint name="j" type="planar"><parent link="a"/><child link="b"/>'
        '<origin xyz="0 0 0.2"/><axis xyz="0 1 0"/></joint>'
        "</robot>",
        encoding="utf-8",
    )
    with pytest.raises(URDFImportError, match="planar"):
        load_urdf_model(p)


def test_no_actuated_joints_is_rejected(tmp_path):
    """A single welded link has nothing for a policy to control -- a vacuous search."""
    p = tmp_path / "static.urdf"
    p.write_text(
        '<robot name="s"><link name="a"><inertial><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
        "<collision><geometry><box size=\"0.1 0.1 0.1\"/></geometry></collision>"
        "</link></robot>",
        encoding="utf-8",
    )
    with pytest.raises(URDFImportError, match="NO actuated joints"):
        load_urdf_model(p)


def test_multi_root_urdf_is_rejected(tmp_path):
    p = tmp_path / "two.urdf"
    link = ('<link name="{n}"><inertial><mass value="1"/>'
            '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
            '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>')
    p.write_text(
        '<robot name="t">' + link.format(n="a") + link.format(n="b") + "</robot>",
        encoding="utf-8",
    )
    with pytest.raises(URDFImportError, match="exactly ONE root link"):
        load_urdf_model(p)


# =========================================================================== #
# (2) THE INJECTION: freejoint, actuators, floor, camera.
# =========================================================================== #
def test_urdf_alone_has_no_actuators_and_no_floating_base():
    """The premise of this module, asserted rather than asserted-in-prose: a raw URDF
    imports with nu == 0, and the root link is fused into worldbody -- taking its 5 kg
    with it (8.2 kg declared -> 3.2 kg imported)."""
    raw = mujoco.MjModel.from_xml_path(str(QUADRUPED))
    assert raw.nu == 0
    assert mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_BODY, "trunk") < 0
    assert raw.body_mass.sum() == pytest.approx(URDF_MASS - 5.0)
    assert not any(int(t) == mujoco.mjtJoint.mjJNT_FREE for t in raw.jnt_type)


def test_freejoint_injected_and_layout_is_free_base_first(quad):
    mj_model, report = quad
    assert report["injected"]["freejoint"] is True
    assert "injected by BreakPoint" in report["injected"]["freejoint_source"]
    # jnt_type[0] == mjJNT_FREE is exactly what validate_legged_model requires.
    assert int(mj_model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    # A floating base adds 7 qpos / 6 qvel on top of the 8 hinge joints.
    assert mj_model.nq == 7 + N_ACTUATED
    assert mj_model.nv == 6 + N_ACTUATED
    assert report["nq"] == mj_model.nq and report["nv"] == mj_model.nv
    # dof vectors are length (6 + nu), so [6:] is exactly the actuated joints.
    assert mj_model.dof_armature.shape[0] == 6 + mj_model.nu
    assert report["qpos_joint_start"] == 7 and report["qvel_joint_start"] == 6


def test_actuator_count_equals_actuated_joint_count(quad):
    mj_model, report = quad
    assert mj_model.nu == N_ACTUATED
    assert report["n_actuated_joints"] == N_ACTUATED
    assert report["injected"]["actuators"] == N_ACTUATED
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(mj_model.njnt)]
    # Every actuated joint the report names is a real joint, and none of them is the base.
    for jn in report["actuated_joint_names"]:
        assert jn in names
    assert len(set(report["actuated_joint_names"])) == N_ACTUATED


def test_actuator_ranges_come_from_the_urdf_limits(quad):
    """ctrlrange is the URDF position limit and forcerange the URDF effort limit -- not a
    guess. hip [-1, 1] / knee [-2.2, 0] / effort 24 are read straight off the fixture."""
    mj_model, report = quad
    want = {"hip": (-1.0, 1.0), "knee": (-2.2, 0.0)}
    for i in range(mj_model.nu):
        jid = int(mj_model.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = want["hip" if jname.endswith("hip") else "knee"]
        assert mj_model.actuator_ctrlrange[i] == pytest.approx([lo, hi])
        assert mj_model.actuator_forcerange[i] == pytest.approx([-EFFORT, EFFORT])
    # kp derived from effort: 24 N.m / 0.35 rad ~= 68.6, and kv = 0.05 * kp.
    assert report["actuator_kp"]["sources"] == ["urdf_effort"]
    assert report["actuator_kp"]["min"] == pytest.approx(EFFORT / 0.35, rel=1e-6)
    assert mj_model.actuator_gainprm[:, 0] == pytest.approx(EFFORT / 0.35, rel=1e-6)
    # The affine-actuator relation the legged Kp/Kd field math assumes.
    assert mj_model.actuator_biasprm[:, 1] == pytest.approx(-mj_model.actuator_gainprm[:, 0])
    assert mj_model.actuator_biasprm[:, 2] == pytest.approx(-0.05 * EFFORT / 0.35, rel=1e-6)


def test_explicit_kp_kd_override_the_derivation():
    _, report = load_urdf_model(QUADRUPED, kp=25.0, kd=0.7)
    assert report["actuator_kp"] == {"min": 25.0, "max": 25.0, "sources": ["caller"]}
    assert all(s["kv"] == 0.7 for s in report["joint_specs"])


def test_floor_and_camera_are_injected(quad):
    mj_model, report = quad
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    assert gid >= 0, "no named floor geom -- resolve_constants_from_candidates would fail"
    assert int(mj_model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    assert mj_model.geom_friction[gid, 0] == pytest.approx(0.8)
    assert report["floor_geom_id"] == gid
    cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, TRACK_CAMERA_NAME)
    assert cid >= 0, "no 'track' camera -- render_failure(camera='track') would fail"
    assert report["injected"]["camera"] == TRACK_CAMERA_NAME
    # The camera is mounted on the root body so it follows the robot.
    assert int(mj_model.cam_bodyid[cid]) == report["root_body_id"]


def test_custom_floor_friction_flows_through_to_the_model_and_the_space():
    mj_model, report = load_urdf_model(QUADRUPED, floor_friction=(0.4, 0.01, 0.001))
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    assert mj_model.geom_friction[gid] == pytest.approx([0.4, 0.01, 0.001])
    space = derive_theta_space(report)
    assert space.dims["friction_floor"].nominal == pytest.approx(0.4)


def test_mass_matches_the_urdf(quad):
    """The floating-base injection exists precisely so the root link keeps its mass."""
    mj_model, report = quad
    assert mj_model.body_mass.sum() == pytest.approx(URDF_MASS)
    assert report["model_total_mass_kg"] == pytest.approx(URDF_MASS)
    assert report["urdf_total_mass_kg"] == pytest.approx(URDF_MASS)
    assert report["base_mass_kg"] == pytest.approx(5.0)


def test_collision_geometry_survives(quad):
    mj_model, report = quad
    # trunk box + 4 * (thigh box, calf box, foot sphere) = 13, plus the injected floor.
    assert report["n_collision_geoms"] == 13
    assert mj_model.ngeom == 14


def test_dead_theta_dims_are_reported_and_can_be_made_live(quad):
    """URDF has NO armature concept and usually no <dynamics friction=>, so both dof
    vectors import as all-zero -- which makes joint_armature_scale /
    joint_frictionloss_scale multiply zero by a number, i.e. search nothing while still
    appearing as axes. That must be reported, never silently tolerated."""
    mj_model, report = quad
    assert report["dead_theta_dims"] == [
        "joint_armature_scale", "joint_frictionloss_scale"
    ]
    assert report["dof_armature_range"] == [0.0, 0.0]
    assert any("NO-OPS" in w and "default_armature" in w for w in report["warnings"])
    assert np.all(np.asarray(mj_model.dof_armature) == 0)

    live_model, live = load_urdf_model(
        QUADRUPED, default_armature=0.01, default_frictionloss=0.1
    )
    assert live["dead_theta_dims"] == []
    # The free base keeps armature 0; every actuated dof gets the requested value.
    assert np.asarray(live_model.dof_armature)[6:] == pytest.approx(0.01)
    assert np.asarray(live_model.dof_frictionloss)[6:] == pytest.approx(0.1)
    assert not any("NO-OPS" in w for w in live["warnings"]), live["warnings"]


def test_urdf_dynamics_friction_is_imported(tmp_path):
    """<dynamics friction=> DOES reach dof_frictionloss (armature has no URDF spelling)."""
    src = QUADRUPED.read_text(encoding="utf-8").replace(
        '<axis xyz="0 1 0"/>', '<axis xyz="0 1 0"/><dynamics damping="0.5" friction="0.2"/>'
    )
    p = tmp_path / "damped.urdf"
    p.write_text(src, encoding="utf-8")
    mj_model, report = load_urdf_model(p)
    assert np.asarray(mj_model.dof_frictionloss)[6:] == pytest.approx(0.2)
    assert np.asarray(mj_model.dof_damping)[6:] == pytest.approx(0.5)
    assert report["dead_theta_dims"] == ["joint_armature_scale"]


def test_option_overrides_are_applied():
    mj_model, _ = load_urdf_model(QUADRUPED, option={"timestep": 0.005})
    assert float(mj_model.opt.timestep) == pytest.approx(0.005)


# =========================================================================== #
# (3) THE MEASUREMENT: standing height comes from settling under gravity.
# =========================================================================== #
def test_standing_height_is_measured_and_sane(quad):
    _, report = quad
    h = report["standing_height_m"]
    assert math.isfinite(h) and h > 0
    # The legs are straight at the default pose, so the geometric stance is 0.32 m; the
    # settled value must be that, minus a little contact compliance, never a free-fall or
    # a collapse.
    assert 0.9 * GEOMETRIC_STANDING_HEIGHT < h <= GEOMETRIC_STANDING_HEIGHT + 1e-3
    assert report["settled_up_proj"] > 0.99      # still upright
    assert report["settle_contacts"] >= 4        # all four feet on the floor
    assert report["standing_height_is_stance"] is True
    # The only warning this well-formed fixture earns is the dead-theta-dim one below.
    assert all("NO-OPS" in w for w in report["warnings"]), report["warnings"]


def test_fall_threshold_is_derived_from_this_robot_not_from_go1(quad):
    _, report = quad
    th = derive_thresholds(report)
    assert set(th) == {"trunk_height_min"}
    assert th["trunk_height_min"] == pytest.approx(0.5 * report["standing_height_m"], abs=1e-6)
    assert th["trunk_height_min"] != pytest.approx(0.18)  # the go1 value, not copied


# =========================================================================== #
# (4) MESHES: package:// resolution, and the loud failure when one is missing.
# =========================================================================== #
def _write_binary_stl(path, tris):
    """MuJoCo requires BINARY STL (measured: an ASCII file fails the decoder)."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def _mesh_urdf(tmp_path, filename):
    (tmp_path / "meshes").mkdir(exist_ok=True)
    _write_binary_stl(
        tmp_path / "meshes" / "tetra.stl",
        [((0, 0, 0), (0, .1, 0), (.1, 0, 0)),
         ((0, 0, 0), (.1, 0, 0), (0, 0, .1)),
         ((0, 0, 0), (0, 0, .1), (0, .1, 0)),
         ((.1, 0, 0), (0, .1, 0), (0, 0, .1))],
    )
    p = tmp_path / "meshbot.urdf"
    p.write_text(
        '<robot name="meshbot">'
        '<link name="base"><inertial><mass value="1.0"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
        '<collision><origin xyz="0 0 -0.05"/>'   # so the base origin stands at z ~= 0.05
        f'<geometry><mesh filename="{filename}"/></geometry></collision></link>'
        '<link name="tip"><inertial><mass value="0.2"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>'
        '<collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>'
        '<joint name="j1" type="revolute"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>'
        '<limit lower="-1" upper="1" effort="5" velocity="5"/></joint>'
        "</robot>",
        encoding="utf-8",
    )
    return p


def test_package_uri_meshes_are_resolved(tmp_path):
    """MuJoCo does NOT resolve package:// (measured: it tries to open the literal string),
    so the importer rewrites it to an absolute path before compiling."""
    urdf = _mesh_urdf(tmp_path, "package://meshbot/meshes/tetra.stl")
    mj_model, report = load_urdf_model(urdf, meshdir=tmp_path)
    assert mj_model.nmesh == 1
    assert len(report["meshes"]) == 1
    assert report["meshes"][0]["declared"].startswith("package://")
    assert report["meshes"][0]["resolved"].endswith("meshes/tetra.stl")
    assert Path(report["meshes"][0]["resolved"]).is_file()


def test_relative_mesh_paths_resolve_against_the_urdf_directory(tmp_path):
    urdf = _mesh_urdf(tmp_path, "meshes/tetra.stl")
    mj_model, _ = load_urdf_model(urdf)
    assert mj_model.nmesh == 1


def test_unresolvable_mesh_lists_every_missing_path_and_where_it_looked(tmp_path):
    urdf = _mesh_urdf(tmp_path, "package://meshbot/meshes/absent.stl")
    with pytest.raises(URDFImportError) as ei:
        load_urdf_model(urdf, meshdir=tmp_path)
    msg = str(ei.value)
    # The message must name the missing file AND every directory that was searched, so the
    # fix (pass meshdir, or ship the mesh) is obvious from the traceback alone.
    assert "absent.stl" in msg and "Searched" in msg and tmp_path.name in msg


# =========================================================================== #
# (5) FIXED BASE (an arm): supported for import, refused for a legged Target.
# =========================================================================== #
def test_fixed_base_import_skips_the_freejoint_and_the_base_obs_blocks():
    mj_model, report = load_urdf_model(FIXTURES / "arm2.urdf", floating_base=False)
    assert report["injected"]["freejoint"] is False
    assert not any(int(t) == mujoco.mjtJoint.mjJNT_FREE for t in mj_model.jnt_type)
    assert mj_model.nq == 2 and mj_model.nu == 2
    assert [n for n, _, _ in report["obs_layout"]] == [
        "joint_pos", "joint_vel", "last_action", "command"
    ]
    assert report["obs_dim"] == 2 + 2 + 2 + 3
    assert report["standing_height_is_stance"] is False
    assert any("fixed base" in w for w in report["warnings"])
    # Per-joint gains differ because the two joints declare different effort limits.
    assert report["actuator_kp"]["min"] == pytest.approx(40.0 / 0.35)
    assert report["actuator_kp"]["max"] == pytest.approx(80.0 / 0.35)


def test_make_urdf_target_refuses_a_fixed_base_robot(tmp_path):
    with pytest.raises(URDFImportError) as ei:
        make_urdf_target("arm", FIXTURES / "arm2.urdf", floating_base=False,
                         theta_space_path=tmp_path / "ts.yaml")
    assert "floating_base=True" in str(ei.value)
    assert "manipulation" in str(ei.value)


def test_derive_theta_space_refuses_a_fixed_base_report():
    _, report = load_urdf_model(FIXTURES / "arm2.urdf", floating_base=False)
    with pytest.raises(URDFImportError, match="FIXED-base"):
        derive_theta_space(report)


# =========================================================================== #
# (6) THE DERIVED THETA SPACE: scaled from THIS robot, loadable by the pipeline.
# =========================================================================== #
def test_derived_space_has_the_dims_the_legged_field_math_reads(quad):
    from breakpoint_mjx.targets._legged import build_legged_fields  # noqa: F401

    _, report = quad
    space = derive_theta_space(report)
    required_model = {
        "friction_floor", "torso_mass_add_kg", "com_offset_x_m", "com_offset_y_m",
        "kp_scale", "kd_scale", "com_offset_z_m", "joint_frictionloss_scale",
        "joint_armature_scale", "floor_slope_rad",
    }
    required_rollout = {
        "obs_noise_scale", "latency_steps", "obs_dropout_prob", "obs_bias_scale",
        "push_mag", "push_step_frac",
    }
    names = set(space.names())
    assert required_model <= names and required_rollout <= names
    for n in required_model:
        assert space.dims[n].kind == "model"
    for n in required_rollout:
        assert space.dims[n].kind == "rollout"
    # Every dim must bracket its nominal, and every plausible band sit inside the bounds.
    for d in space.dims.values():
        lo, hi = d.bounds
        assert lo <= d.nominal <= hi, d
        assert lo <= d.plausible[0] <= d.plausible[1] <= hi, d


def test_dimensional_bounds_scale_with_the_robot_not_with_go1(quad):
    _, report = quad
    space = derive_theta_space(report)
    mass = report["model_total_mass_kg"]
    h = report["standing_height_m"]
    # payload: a FRACTION of this robot's mass, so an 8.2 kg robot never sees go1's +6 kg.
    assert space.dims["torso_mass_add_kg"].bounds[1] == pytest.approx(0.47 * mass)
    assert space.dims["torso_mass_add_kg"].bounds[1] < 6.0
    assert space.dims["com_offset_x_m"].bounds[1] == pytest.approx(0.28 * h)
    assert space.dims["com_offset_z_m"].bounds[1] == pytest.approx(0.18 * h)
    # push: Froude-scaled to the stance height.
    assert space.dims["push_mag"].bounds[1] == pytest.approx(1.2 * math.sqrt(9.81 * h))
    # dimensionless dims are copied verbatim from configs/legged_default/theta_space.yaml.
    default = load_theta_space(
        config_path("legged_default/theta_space.yaml")
    )
    for n in ("kp_scale", "kd_scale", "joint_armature_scale", "joint_frictionloss_scale",
              "floor_slope_rad", "obs_noise_scale", "latency_steps", "obs_dropout_prob",
              "obs_bias_scale", "push_step_frac"):
        assert space.dims[n].bounds == default.dims[n].bounds, n
        assert space.dims[n].plausible == default.dims[n].plausible, n


def test_theta_space_yaml_round_trips_through_the_pipeline_loader(quad, tmp_path):
    _, report = quad
    space = derive_theta_space(report)
    path = write_theta_space_yaml(space, tmp_path / "theta_space.yaml", header="derived")
    reloaded = load_theta_space(path)
    assert reloaded.names() == space.names()
    for n in space.names():
        a, b = space.dims[n], reloaded.dims[n]
        assert b.kind == a.kind
        assert b.bounds == pytest.approx(a.bounds, abs=1e-6)
        assert b.nominal == pytest.approx(a.nominal, abs=1e-6)
        assert b.plausible == pytest.approx(a.plausible, abs=1e-6)


# =========================================================================== #
# (7) THE TARGET + the report's honesty contract.
# =========================================================================== #
def test_make_urdf_target_builds_a_usable_target(tmp_path):
    from breakpoint_mjx.targets import Target
    from breakpoint_mjx.targets._legged import LEGGED_MODEL_FIELDS, build_legged_fields

    target = make_urdf_target("miniquad", QUADRUPED,
                              theta_space_path=tmp_path / "theta_space.yaml")
    assert isinstance(target, Target)
    assert target.name == "miniquad"
    assert target.env_name == "urdf:quadruped.urdf"
    assert target.checkpoint_name == "miniquad_policy.pkl"
    assert target.model_fields == LEGGED_MODEL_FIELDS
    assert target.build_fields is build_legged_fields
    assert target.camera == TRACK_CAMERA_NAME
    assert target.objective.name == "legged"
    # The theta space is written where the target points, and the pipeline's
    # `config_path(target.theta_space_path)` join resolves an absolute path to itself.
    assert Path(target.theta_space_path).is_file()
    assert (Path("/anything") / target.theta_space_path) == Path(target.theta_space_path)
    space = load_theta_space(target.theta_space_path)
    assert len(space.dims) == 16
    # Per-robot fall threshold, derived from the measured stance.
    report = target.load_env.report
    assert target.thresholds["trunk_height_min"] == pytest.approx(
        0.5 * report["standing_height_m"], abs=1e-6
    )


def test_report_is_truthful_and_loudly_flags_the_obs_convention(quad):
    _, report = quad
    assert report["obs_convention_warning"] == OBS_CONVENTION_WARNING
    assert "ObsSpec" in report["obs_convention_warning"]
    # The obs layout is exposed programmatically, contiguous, and matches obs_dim.
    layout = report["obs_layout"]
    assert [n for n, _, _ in layout] == [
        "gravity_body", "lin_vel_body", "ang_vel_body",
        "joint_pos", "joint_vel", "last_action", "command",
    ]
    cursor = 0
    for _, a, b in layout:
        assert a == cursor
        cursor = b
    assert cursor == report["obs_dim"] == 9 + 3 * N_ACTUATED + 3
    assert report["act_dim"] == N_ACTUATED
    # The report says what was ignored rather than pretending it was handled.
    text = format_report(report)
    assert "INJECTED" in text and "MEASURED" in text and "ObsSpec" in text


def test_ros_only_blocks_are_reported_as_ignored_not_silently_dropped(tmp_path):
    src = QUADRUPED.read_text(encoding="utf-8").replace(
        "</robot>",
        '<transmission name="t1"><type>transmission_interface/SimpleTransmission</type>'
        '</transmission>'
        '<gazebo reference="trunk"><mu1>0.8</mu1></gazebo>'
        '<ros2_control name="rc" type="system"><hardware><plugin>fake</plugin></hardware>'
        "</ros2_control></robot>",
    )
    p = tmp_path / "ros.urdf"
    p.write_text(src, encoding="utf-8")
    _, report = load_urdf_model(p)
    joined = " ".join(report["unsupported"])
    assert "<transmission>" in joined and "<gazebo>" in joined and "<ros2_control>" in joined
    # ...and the robot still imports correctly alongside them.
    assert report["n_actuated_joints"] == N_ACTUATED


def test_continuous_joints_get_a_default_ctrlrange_and_a_warning(tmp_path):
    src = QUADRUPED.read_text(encoding="utf-8").replace(
        '<joint name="fr_hip" type="revolute">', '<joint name="fr_hip" type="continuous">'
    ).replace('<limit lower="-1.0" upper="1.0" effort="24.0" velocity="20.0"/>\n  </joint>\n'
              '  <link name="fr_thigh">',
              '<limit effort="24.0" velocity="20.0"/>\n  </joint>\n  <link name="fr_thigh">',
              1)
    p = tmp_path / "cont.urdf"
    p.write_text(src, encoding="utf-8")
    mj_model, report = load_urdf_model(p)
    assert any("continuous" in w for w in report["warnings"])
    spec = next(s for s in report["joint_specs"] if s["name"] == "fr_hip")
    assert spec["ctrlrange"] == pytest.approx([-math.pi, math.pi])
    assert "unlimited" in spec["ctrlrange_source"]


def test_urdf_to_mjcf_returns_wellformed_xml_with_every_injection(tmp_path):
    import xml.etree.ElementTree as ET

    mjcf, info = urdf_to_mjcf(QUADRUPED, work_dir=tmp_path)
    root = ET.fromstring(mjcf)                       # well-formed by construction
    assert root.tag == "mujoco"
    assert root.find("actuator") is not None
    assert len(root.find("actuator").findall("position")) == N_ACTUATED
    floor = [g for g in root.find("worldbody").findall("geom")
             if g.get("name") == FLOOR_GEOM_NAME]
    assert len(floor) == 1 and floor[0].get("type") == "plane"
    assert root.find("option").get("solver") == "Newton"
    assert info["injected"]["actuators"] == N_ACTUATED
    # No jax needed to get here.
    assert "jax" not in [m for m in ("jax",) if m in globals()]


# =========================================================================== #
# (8) OPT-IN: MJX env + the full evaluator path. BP_RUN_URDF=1.
# =========================================================================== #
_RUN_MJX = os.environ.get("BP_RUN_URDF") == "1"
requires_mjx = pytest.mark.skipif(
    not _RUN_MJX,
    reason="builds an MJX env / jitted evaluator; set BP_RUN_URDF=1 to run "
           "(see this module's docstring for the recommended XLA_FLAGS)",
)

E2E_HORIZON = 10   # tiny: one XLA compile, ~1.5 GB peak on this box
E2E_N = 4


@pytest.fixture
def target_and_env(tmp_path):
    """FUNCTION-scoped on purpose: rollout_one mutates ``env._mjx_model``, so a shared env
    would leak traced models (and their tracers) between tests."""
    target = make_urdf_target("miniquad_e2e", QUADRUPED,
                              theta_space_path=tmp_path / "theta_space.yaml")
    env = target.load_env()
    return target, env


@requires_mjx
def test_env_implements_the_rollout_contract(target_and_env):
    import jax
    import jax.numpy as jnp

    target, env = target_and_env
    report = target.load_env.report
    assert env.action_size == N_ACTUATED
    assert env.observation_size == report["obs_dim"]
    assert [n for n, _ in env.obs_layout] == [n for n, _, _ in report["obs_layout"]]
    assert env.dt == pytest.approx(0.02)

    state = env.reset(jax.random.PRNGKey(0))
    assert isinstance(state.obs, dict) and set(state.obs) == {"state"}
    assert state.obs["state"].shape == (env.observation_size,)
    assert state.info["command"].shape == (3,)
    # The obs blocks mean what the docstring says: at reset the robot is upright and
    # still, so gravity_body ~= (0, 0, -1) and both velocity blocks are zero.
    obs = np.asarray(state.obs["state"])
    blocks = dict(env.obs_layout)
    np.testing.assert_allclose(obs[blocks["gravity_body"]], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[blocks["lin_vel_body"]], 0, atol=1e-6)
    np.testing.assert_allclose(obs[blocks["ang_vel_body"]], 0, atol=1e-6)
    np.testing.assert_allclose(obs[blocks["joint_pos"]], 0, atol=1e-6)
    np.testing.assert_allclose(obs[blocks["last_action"]], 0, atol=1e-6)
    assert float(state.data.qpos[2]) == pytest.approx(report["standing_height_m"])

    nxt = env.step(state, jnp.zeros((env.action_size,)))
    assert nxt.obs["state"].shape == (env.observation_size,)
    assert np.all(np.isfinite(np.asarray(nxt.obs["state"])))
    assert int(nxt.info["step"]) == 1
    # state.replace / data.replace are what rollout_one's push adversary needs.
    bumped = nxt.replace(data=nxt.data.replace(qvel=nxt.data.qvel.at[:3].add(1.0)))
    assert float(bumped.data.qvel[0]) == pytest.approx(float(nxt.data.qvel[0]) + 1.0)


@requires_mjx
def test_validate_legged_model_accepts_the_imported_model(target_and_env):
    target, env = target_and_env
    floor_id, base_id = target.resolve_constants(env)   # runs validate_legged_model
    assert floor_id == target.load_env.report["floor_geom_id"]
    assert base_id == target.load_env.report["root_body_id"]


@requires_mjx
def test_end_to_end_evaluator_over_the_derived_theta_space(target_and_env):
    """THE proof: URDF -> Target -> derived space -> random MLP sized to the derived dims
    -> make_evaluator -> a finite trajectory dict of the legged contract shape."""
    import jax

    from breakpoint_mjx.adapters import from_params
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.thetas import sample_uniform

    target, env = target_and_env
    consts = target.resolve_constants(env)
    space = load_theta_space(target.theta_space_path)

    rng = np.random.default_rng(0)
    obs_dim, act_dim = env.observation_size, env.action_size
    policy = from_params(
        [(rng.normal(size=(obs_dim, 16)) * 0.1, np.zeros(16)),
         (rng.normal(size=(16, act_dim)) * 0.1, np.zeros(act_dim))],
        activation="tanh",
    )
    evaluate = make_evaluator(env, policy.policy_fn, space, E2E_HORIZON, target, consts)
    thetas = sample_uniform(space, jax.random.PRNGKey(0), E2E_N)
    traj = evaluate(thetas, jax.random.PRNGKey(1))

    for key, shape in (
        ("trunk_height", (E2E_N, E2E_HORIZON)),
        ("up_proj", (E2E_N, E2E_HORIZON)),
        ("lin_vel", (E2E_N, E2E_HORIZON, 3)),
        ("ang_vel", (E2E_N, E2E_HORIZON, 3)),
        ("command", (E2E_N, E2E_HORIZON, 3)),
    ):
        assert key in traj, key
        arr = np.asarray(traj[key])
        assert arr.shape == shape, (key, arr.shape)
        assert np.all(np.isfinite(arr)), key

    h = np.asarray(traj["trunk_height"])
    up = np.asarray(traj["up_proj"])
    stance = target.load_env.report["standing_height_m"]
    # 10 steps (0.2 s) of a random policy cannot teleport the trunk or invert the robot.
    assert np.all(h > 0.5 * stance) and np.all(h < 1.5 * stance)
    assert np.all(up > 0.5)

    # The scores machinery consumes this trajectory dict unchanged.
    from breakpoint_mjx.failure import component_signals

    thresholds = {"trunk_height_min": 0.18, "up_proj_min": 0.5,
                  "tracking_err_max": 0.5, "ang_vel_max": 8.0}
    thresholds.update(target.thresholds or {})
    sig = component_signals({k: np.asarray(v)[0] for k, v in traj.items()}, thresholds)
    assert np.isfinite(sig["fall"]) and np.isfinite(sig["tracking_error"])
