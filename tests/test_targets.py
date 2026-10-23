"""Pure tests for the Target abstraction (Upgrade #1) -- no JAX/GPU required.

Covers the registry lookup, target metadata, and the Go1 build_fields per-row math.
build_fields uses the JAX in-place update API (x.at[idx].set/add/multiply); we drive
it with a tiny numpy-backed fake that mimics that chainable API, so the verbatim
Go1 field math is exercised on CPU without importing jax.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.targets import TARGETS, Target, get_target
from breakpoint_mjx.targets._legged import (
    build_legged_fields,
    validate_legged_model,
)

from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Registry + metadata
# --------------------------------------------------------------------------- #

def test_get_target_returns_go1():
    g = get_target("go1")
    assert isinstance(g, Target)
    assert g.name == "go1"
    assert g.env_name == "Go1JoystickFlatTerrain"
    assert g.checkpoint_name == "go1_policy.pkl"
    assert "go1" in TARGETS


def test_unknown_target_raises():
    with pytest.raises(KeyError):
        get_target("does-not-exist")


def test_go1_model_fields_are_the_eight():
    g = get_target("go1")
    assert g.model_fields == (
        "geom_friction",
        "body_mass",
        "body_ipos",
        "actuator_gainprm",
        "actuator_biasprm",
        "dof_frictionloss",
        "dof_armature",
        "geom_quat",   # T6 floor-slope (GEOMETRY axis)
        "pair_friction",   # explicit floor contact pairs (G1) carry their own friction
    )


def test_go1_theta_space_path_exists():
    g = get_target("go1")
    assert g.theta_space_path == "configs/go1/theta_space.yaml"
    assert (config_path(g.theta_space_path)).is_file()


def test_callables_are_unbound_functions():
    """Stored as plain functions on the frozen instance, so they are NOT bound as
    methods -- calling target.load_env() must not inject an implicit self."""
    g = get_target("go1")
    assert callable(g.load_env)
    assert callable(g.resolve_constants)
    assert callable(g.build_fields)
    # build_fields takes exactly (scalars, base_model, consts) -- 3 params, no self.
    import inspect
    assert list(inspect.signature(g.build_fields).parameters) == [
        "scalars", "base_model", "consts",
    ]


# --------------------------------------------------------------------------- #
# build_fields math (numpy-backed fake of the JAX .at[...] API)
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
    """Minimal numpy-backed stand-in for a JAX array supporting .at[idx].{set,add,multiply},
    plus .shape and read indexing (x[:, 0]) for validate_legged_model."""
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


class _FakeModel:
    def __init__(self, nbody=4, ngeom=3, nu=12, ndof=18, jnt0=0):
        rng = np.random.default_rng(0)
        self.geom_friction = _FakeArr(rng.uniform(0.5, 1.0, size=(ngeom, 3)))
        self.body_mass = _FakeArr(rng.uniform(0.5, 3.0, size=(nbody,)))
        self.body_ipos = _FakeArr(rng.uniform(-0.01, 0.01, size=(nbody, 3)))
        self.actuator_gainprm = _FakeArr(rng.uniform(10.0, 30.0, size=(nu, 10)))
        self.actuator_biasprm = _FakeArr(rng.uniform(-30.0, 30.0, size=(nu, 10)))
        self.dof_frictionloss = _FakeArr(rng.uniform(0.0, 0.5, size=(ndof,)))
        self.dof_armature = _FakeArr(rng.uniform(0.01, 0.1, size=(ndof,)))
        # geom_quat: identity (w,x,y,z)=(1,0,0,0) per geom; the floor-slope dim pitches
        # the floor row. Identity base keeps the intended-rotation math clean to check.
        self.geom_quat = _FakeArr(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (ngeom, 1)))
        # explicit contact pairs: (floor 0, geom 1) and (geom 1, geom 2); only the first
        # touches the floor, so only it may take friction_floor
        self.pair_geom1 = np.array([0, 1])
        self.pair_geom2 = np.array([1, 2])
        self.pair_friction = _FakeArr(np.full((2, 5), 0.6))
        # jnt_type: joint 0 is the free base (mjJNT_FREE == 0) by default, then nu
        # hinge joints (mjJNT_HINGE == 3). Tests flip jnt0 to simulate a model whose
        # joint 0 is NOT the free base (count can still match, but [6:] would mis-slice).
        self.jnt_type = _FakeArr(np.concatenate([[jnt0], np.full(nu, 3)]))


def _scalars():
    return dict(
        friction_floor=0.3,
        torso_mass_add_kg=2.5,
        com_offset_x_m=0.01,
        com_offset_y_m=-0.02,
        kp_scale=1.2,
        kd_scale=0.8,
        com_offset_z_m=0.03,
        joint_frictionloss_scale=1.5,
        joint_armature_scale=2.0,
        floor_slope_rad=0.15,   # T6 GEOMETRY axis
    )


def test_build_fields_returns_exactly_model_fields():
    g = get_target("go1")
    base = _FakeModel()
    out = g.build_fields(_scalars(), base, (0, 1))   # (floor_geom_id=0, torso_body_id=1)
    assert set(out.keys()) == set(g.model_fields)


def test_build_fields_applies_go1_math_at_the_right_indices():
    g = get_target("go1")
    base = _FakeModel()
    floor_geom_id, torso_body_id = 0, 1
    s = _scalars()
    out = g.build_fields(s, base, (floor_geom_id, torso_body_id))

    # explicit contact pairs that include the floor take friction_floor on both tangential
    # coefficients (they override geom friction); other pairs and coefficients untouched
    pf = out["pair_friction"].arr
    assert pf[0, :2] == pytest.approx([s["friction_floor"]] * 2)
    assert pf[0, 2:] == pytest.approx(base.pair_friction.arr[0, 2:])
    assert pf[1] == pytest.approx(base.pair_friction.arr[1])

    # geom_friction[floor, 0] is SET to friction_floor; other entries untouched.
    assert out["geom_friction"].arr[floor_geom_id, 0] == pytest.approx(s["friction_floor"])
    assert out["geom_friction"].arr[floor_geom_id, 1] == pytest.approx(
        base.geom_friction.arr[floor_geom_id, 1]
    )

    # body_mass[torso] += torso_mass_add_kg.
    assert out["body_mass"].arr[torso_body_id] == pytest.approx(
        base.body_mass.arr[torso_body_id] + s["torso_mass_add_kg"]
    )
    # other bodies untouched.
    assert out["body_mass"].arr[2] == pytest.approx(base.body_mass.arr[2])

    # body_ipos[torso, {0,1,2}] += com offsets.
    assert out["body_ipos"].arr[torso_body_id, 0] == pytest.approx(
        base.body_ipos.arr[torso_body_id, 0] + s["com_offset_x_m"]
    )
    assert out["body_ipos"].arr[torso_body_id, 1] == pytest.approx(
        base.body_ipos.arr[torso_body_id, 1] + s["com_offset_y_m"]
    )
    assert out["body_ipos"].arr[torso_body_id, 2] == pytest.approx(
        base.body_ipos.arr[torso_body_id, 2] + s["com_offset_z_m"]
    )

    # Kp scales gainprm[:,0] and biasprm[:,1]; Kd scales biasprm[:,2].
    np.testing.assert_allclose(
        out["actuator_gainprm"].arr[:, 0], base.actuator_gainprm.arr[:, 0] * s["kp_scale"]
    )
    np.testing.assert_allclose(
        out["actuator_biasprm"].arr[:, 1], base.actuator_biasprm.arr[:, 1] * s["kp_scale"]
    )
    np.testing.assert_allclose(
        out["actuator_biasprm"].arr[:, 2], base.actuator_biasprm.arr[:, 2] * s["kd_scale"]
    )

    # dof_frictionloss/dof_armature scale ONLY the 12 actuated joints ([6:]); [:6] untouched.
    np.testing.assert_allclose(
        out["dof_frictionloss"].arr[6:],
        base.dof_frictionloss.arr[6:] * s["joint_frictionloss_scale"],
    )
    np.testing.assert_allclose(out["dof_frictionloss"].arr[:6], base.dof_frictionloss.arr[:6])
    np.testing.assert_allclose(
        out["dof_armature"].arr[6:], base.dof_armature.arr[6:] * s["joint_armature_scale"]
    )
    np.testing.assert_allclose(out["dof_armature"].arr[:6], base.dof_armature.arr[:6])

    # geom_quat[floor] is pitched about y by floor_slope_rad (identity base -> pure
    # pitch quaternion [cos(h), 0, sin(h), 0], h = slope/2); other geoms untouched.
    h = 0.5 * s["floor_slope_rad"]
    np.testing.assert_allclose(
        out["geom_quat"].arr[floor_geom_id],
        [np.cos(h), 0.0, np.sin(h), 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out["geom_quat"].arr[1:], base.geom_quat.arr[1:]  # non-floor geoms unchanged
    )


# --------------------------------------------------------------------------- #
# Barkour: second quadruped target (BYO validation, Upgrade #1 Definition of Done)
# --------------------------------------------------------------------------- #

def test_get_target_returns_barkour():
    b = get_target("barkour")
    assert isinstance(b, Target)
    assert b.name == "barkour"
    assert b.env_name == "BarkourJoystick"
    assert b.checkpoint_name == "barkour_policy.pkl"
    assert "barkour" in TARGETS


def test_barkour_model_fields_match_go1():
    """Same morphology class -> the second quadruped batches the exact same 7 fields."""
    b = get_target("barkour")
    g = get_target("go1")
    assert b.model_fields == g.model_fields


def test_barkour_callables_are_unbound_functions():
    b = get_target("barkour")
    assert callable(b.load_env)
    assert callable(b.resolve_constants)
    assert callable(b.build_fields)
    import inspect
    assert list(inspect.signature(b.build_fields).parameters) == [
        "scalars", "base_model", "consts",
    ]


def test_barkour_theta_space_path_exists_and_parses():
    b = get_target("barkour")
    assert b.theta_space_path == "configs/barkour/theta_space.yaml"
    path = config_path(b.theta_space_path)
    assert path.is_file()
    space = load_theta_space(path)
    assert len(space.dims) > 0


def test_barkour_theta_space_matches_go1_dims():
    """Dims are robot-agnostic (scale/offset based), so the second target keeps the
    same dim names, kinds, and count as go1 unless intentionally changed."""
    go1_space = load_theta_space(config_path(get_target("go1").theta_space_path))
    bk_space = load_theta_space(config_path(get_target("barkour").theta_space_path))
    assert bk_space.names() == go1_space.names()
    for name in go1_space.names():
        assert bk_space.dims[name].kind == go1_space.dims[name].kind


def test_go1_and_barkour_share_the_same_build_fields():
    """The refactor's whole point: both quadruped targets reuse ONE field-math
    helper rather than copy-pasting, so their build_fields are the same object."""
    g = get_target("go1")
    b = get_target("barkour")
    assert g.build_fields is b.build_fields
    assert g.build_fields is build_legged_fields


def test_barkour_build_fields_math_matches_go1_byte_for_byte():
    """Identical helper -> identical output for the same inputs (incl. the [6:] slice
    and the affine-actuator Kp/Kd math), at Barkour-resolved constants."""
    g = get_target("go1")
    b = get_target("barkour")
    base_g, base_b = _FakeModel(), _FakeModel()  # same seed -> identical arrays
    s = _scalars()
    consts = (0, 1)  # (floor_geom_id, torso_body_id) by name-resolution convention
    out_g = g.build_fields(s, base_g, consts)
    out_b = b.build_fields(s, base_b, consts)
    assert set(out_b.keys()) == set(b.model_fields)
    for f in b.model_fields:
        np.testing.assert_allclose(out_b[f].arr, out_g[f].arr)


# --------------------------------------------------------------------------- #
# g1: first HUMANOID target (cross-morphology proof, 6 + N=29 layout)
# --------------------------------------------------------------------------- #

def test_get_target_returns_g1():
    g = get_target("g1")
    assert isinstance(g, Target)
    assert g.name == "g1"
    assert g.env_name == "G1JoystickFlatTerrain"
    assert g.checkpoint_name == "g1_policy.pkl"
    assert "g1" in TARGETS


def test_g1_model_fields_match_go1():
    """A humanoid is still a floating-base legged robot -> it batches the same 7 fields."""
    assert get_target("g1").model_fields == get_target("go1").model_fields


def test_g1_shares_the_same_build_fields_as_go1():
    """The whole point of generalizing the quadruped-only helper into targets._legged:
    the humanoid reuses the ONE field-math helper (N-agnostic), not a humanoid copy."""
    g1 = get_target("g1")
    assert g1.build_fields is get_target("go1").build_fields
    assert g1.build_fields is build_legged_fields


def test_g1_callables_are_unbound_functions():
    g1 = get_target("g1")
    assert callable(g1.load_env)
    assert callable(g1.resolve_constants)
    assert callable(g1.build_fields)
    import inspect
    assert list(inspect.signature(g1.build_fields).parameters) == [
        "scalars", "base_model", "consts",
    ]


def test_g1_theta_space_path_exists_and_parses():
    g1 = get_target("g1")
    assert g1.theta_space_path == "configs/g1/theta_space.yaml"
    path = config_path(g1.theta_space_path)
    assert path.is_file()
    space = load_theta_space(path)
    # Same dim names/kinds as go1 (scale/offset dims transfer); only bounds differ.
    go1_space = load_theta_space(config_path(get_target("go1").theta_space_path))
    assert space.names() == go1_space.names()
    for name in go1_space.names():
        assert space.dims[name].kind == go1_space.dims[name].kind


def test_g1_widens_torso_mass_add_range_vs_go1():
    """The G1 is materially heavier than the Go1, so its mass-add range is wider
    (documented in configs/g1/theta_space.yaml); every OTHER dim's bounds match go1."""
    g1_space = load_theta_space(config_path(get_target("g1").theta_space_path))
    go1_space = load_theta_space(config_path(get_target("go1").theta_space_path))
    g1_lo, g1_hi = g1_space.dims["torso_mass_add_kg"].bounds
    go1_lo, go1_hi = go1_space.dims["torso_mass_add_kg"].bounds
    assert (g1_hi - g1_lo) > (go1_hi - go1_lo)  # strictly wider
    for name in go1_space.names():
        if name == "torso_mass_add_kg":
            continue
        assert g1_space.dims[name].bounds == go1_space.dims[name].bounds


def test_g1_has_taller_fall_threshold_override():
    """Per-target thresholds: the humanoid raises trunk_height_min above the
    search.yaml default (a knee-high cutoff would never trigger on a ~0.8 m pelvis);
    go1/barkour declare no override."""
    g1 = get_target("g1")
    assert g1.thresholds is not None
    assert g1.thresholds["trunk_height_min"] > 0.18  # taller than the Go1 default
    assert get_target("go1").thresholds is None
    assert get_target("barkour").thresholds is None


# --------------------------------------------------------------------------- #
# validate_legged_model: guard the 6-dof-free-base + N-actuated-joint assumption
# --------------------------------------------------------------------------- #

def _affine_model(nu=12, ndof=18, jnt0=0):
    """A _FakeModel whose actuators satisfy biasprm[:,1] == -gainprm[:,0] (affine),
    so the soft invariant holds and no warning prints on the happy path. jnt0 defaults
    to a free base (mjJNT_FREE == 0)."""
    m = _FakeModel(nu=nu, ndof=ndof, jnt0=jnt0)
    m.actuator_biasprm.arr[:, 1] = -m.actuator_gainprm.arr[:, 0]
    return m


def test_validate_legged_model_passes_on_12act_18dof(capsys):
    # Go1/Barkour: free base (jnt_type[0]==0), 12 actuators, 6 + 12 = 18 dofs, valid
    # affine -> no raise, no warning.
    validate_legged_model(_affine_model(nu=12, ndof=18))
    assert "WARNING" not in capsys.readouterr().out


def test_validate_legged_model_passes_on_humanoid_29act_35dof(capsys):
    # G1 humanoid: free base, 29 actuators, 6 + 29 = 35 dofs -> the validator passes.
    validate_legged_model(_affine_model(nu=29, ndof=35))
    assert "WARNING" not in capsys.readouterr().out


def test_validate_legged_model_passes_on_generic_6_plus_n(capsys):
    # Any free-base 6 + N layout passes (here N=23, 6 + 23 = 29 dofs).
    validate_legged_model(_affine_model(nu=23, ndof=29))
    assert "WARNING" not in capsys.readouterr().out


def test_validate_legged_model_raises_when_dof_not_6_plus_nu():
    # Free base, but nu=16 with ndof=18 (expected 6 + 16 = 22) -> hard fail, naming
    # expected vs actual.
    with pytest.raises(ValueError, match="dof_armature"):
        validate_legged_model(_affine_model(nu=16, ndof=18))


def test_validate_legged_model_raises_when_joint0_not_free():
    # The dof COUNT matches (6 + 12 = 18) but joint 0 is NOT a free base (jnt_type[0]=3,
    # mjJNT_HINGE) -> hard fail, because [6:] / qpos[2] / qvel[:6] would mis-slice.
    # This is the case the count check alone could not catch.
    with pytest.raises(ValueError, match=r"FREE base|mjJNT_FREE|jnt_type"):
        validate_legged_model(_affine_model(nu=12, ndof=18, jnt0=3))


def test_validate_legged_model_raises_on_fixed_base_like_layout():
    # nu=12, ndof=12 (no 6-dof free base; expected 18) -> hard fail.
    with pytest.raises(ValueError, match=r"6-dof free|dof_armature"):
        validate_legged_model(_affine_model(nu=12, ndof=12))


def test_validate_legged_model_warns_on_broken_biasprm_relation(capsys):
    # Default _FakeModel has random biasprm, so biasprm[:,1] != -gainprm[:,0]:
    # must WARN (heads-up) but NOT raise -- the layout shapes are still valid.
    validate_legged_model(_FakeModel())  # nu=12, ndof=18 by default
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Generic legged-target factory (targets/generic.py) + the factory-made robots
# --------------------------------------------------------------------------- #

FACTORY_TARGETS = ("spot", "h1", "t1", "berkeley_humanoid")


def test_factory_targets_registered_with_expected_metadata():
    for name in FACTORY_TARGETS:
        t = get_target(name)
        assert isinstance(t, Target)
        assert t.name == name
        assert t.checkpoint_name == f"{name}_policy.pkl"
        assert name in TARGETS


def test_factory_targets_share_the_legged_field_math():
    """The factory's whole point: every generic legged robot reuses THE one shared
    field-math helper and the same 8 batched model fields as go1."""
    go1 = get_target("go1")
    for name in FACTORY_TARGETS:
        t = get_target(name)
        assert t.build_fields is build_legged_fields
        assert t.model_fields == go1.model_fields


def test_factory_targets_theta_space_is_the_shared_default_and_parses():
    go1_space = load_theta_space(config_path(get_target("go1").theta_space_path))
    for name in FACTORY_TARGETS:
        t = get_target(name)
        assert t.theta_space_path == "configs/legged_default/theta_space.yaml"
        path = config_path(t.theta_space_path)
        assert path.is_file()
        space = load_theta_space(path)
        # Same dims/kinds/bounds as go1 (it is a documented copy of the proven space).
        assert space.names() == go1_space.names()
        for n in go1_space.names():
            assert space.dims[n].kind == go1_space.dims[n].kind
            assert space.dims[n].bounds == go1_space.dims[n].bounds


def test_factory_targets_have_scale_matched_fall_thresholds():
    """Every factory robot must override trunk_height_min (the go1-scale 0.18 default
    under-triggers on taller robots), humanoids strictly taller than quadrupeds."""
    th = {name: get_target(name).thresholds["trunk_height_min"] for name in FACTORY_TARGETS}
    for name, v in th.items():
        assert v > 0.18, name
    assert th["h1"] > th["spot"]          # ~1 m humanoid pelvis vs ~0.5 m quadruped body
    assert th["t1"] > th["spot"]


def test_factory_target_callables_are_unbound_functions():
    import inspect

    for name in FACTORY_TARGETS:
        t = get_target(name)
        assert callable(t.load_env)
        assert callable(t.resolve_constants)
        assert list(inspect.signature(t.build_fields).parameters) == [
            "scalars", "base_model", "consts",
        ]


def test_factory_is_import_safe_without_jax_or_mujoco():
    """Building factory targets (import time for the registry) must not pull jax /
    mujoco / mujoco_playground -- they load only inside the callables."""
    import subprocess
    import sys
    code = (
        "import sys; from breakpoint_mjx.targets import TARGETS; "
        "assert {'spot','h1','t1','berkeley_humanoid'} <= set(TARGETS); "
        "assert 'jax' not in sys.modules and 'mujoco' not in sys.modules "
        "and 'mujoco_playground' not in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
