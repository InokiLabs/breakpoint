"""CPU unit tests for T1 cross-config validity (breakpoint_mjx/crossval.py) + the
C-MuJoCo backend's NUMPY field application (breakpoint_mjx/backends/mujoco_c.py).

No jax / mujoco required: only the pure config-building, the validity aggregation, and
the numpy field-parity are exercised here. The live MJX / C-MuJoCo re-sim needs a real
env and is measured in tests/test_mujoco_c_parity.py (opt-in BP_RUN_CBACKEND=1).
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from breakpoint_mjx.backends import make_mujoco_c_backend
from breakpoint_mjx.backends.mjx_backend import MjxBackend
from breakpoint_mjx.backends.mujoco_c import MujocoCBackend, apply_legged_theta_numpy
from breakpoint_mjx.crossval import (
    PhysicsConfig,
    SOLVER_CG,
    SOLVER_NEWTON,
    annotate_families,
    build_config_variants,
    combined_transfer,
    default_physics_configs,
    rank_families_by_validity,
    reproduction_vector,
    reproduces,
    select_top,
    validate_failures,
    validity_score,
)
from breakpoint_mjx.schemas import FailureFamily, FailureRecord
from breakpoint_mjx.targets._legged import build_legged_fields


# --------------------------------------------------------------------------- #
# Config-variant builder: K distinct, valid opt configs + one reference
# --------------------------------------------------------------------------- #
def test_default_configs_are_distinct_and_valid():
    configs = build_config_variants()
    # K (>=5) variants + exactly one nominal reference.
    assert len(configs) >= 5
    names = [c.name for c in configs]
    assert len(set(names)) == len(names)          # distinct names
    assert sum(1 for c in configs if c.reference) == 1
    assert all(c.is_valid() for c in configs)
    # The reference is a true no-op (reproduces the original scored physics).
    ref = next(c for c in configs if c.reference)
    assert ref.is_noop()
    # The variants actually perturb something (not all no-ops).
    assert any(not c.is_noop() for c in configs)


def test_default_configs_cover_solver_and_timestep_and_contact():
    configs = default_physics_configs()
    assert any(c.solver == SOLVER_NEWTON for c in configs)
    assert any(c.solver == SOLVER_CG for c in configs)
    assert any(c.timestep_scale != 1.0 for c in configs)
    assert any(c.solref_scale != 1.0 for c in configs)


def test_build_config_variants_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        build_config_variants([
            PhysicsConfig("nominal", reference=True),
            PhysicsConfig("dup", timestep_scale=0.5),
            PhysicsConfig("dup", timestep_scale=2.0),
        ])


def test_build_config_variants_rejects_invalid_config():
    with pytest.raises(ValueError, match="invalid"):
        build_config_variants([
            PhysicsConfig("nominal", reference=True),
            PhysicsConfig("bad_timestep", timestep_scale=0.0),  # non-positive
        ])


def test_build_config_variants_requires_exactly_one_reference():
    with pytest.raises(ValueError, match="reference"):
        build_config_variants([
            PhysicsConfig("a", timestep_scale=0.5),
            PhysicsConfig("b", timestep_scale=2.0),
        ])
    with pytest.raises(ValueError, match="reference"):
        build_config_variants([
            PhysicsConfig("a", reference=True),
            PhysicsConfig("b", reference=True),
        ])


def test_physics_config_validity_rules():
    assert PhysicsConfig("ok", solver=SOLVER_NEWTON, iterations=10).is_valid()
    assert PhysicsConfig("cg", solver=SOLVER_CG).is_valid()
    assert not PhysicsConfig("neg_iter", iterations=0).is_valid()
    assert not PhysicsConfig("neg_ls", ls_iterations=-1).is_valid()
    assert not PhysicsConfig("bad_solver", solver=99).is_valid()
    # PGS (mjtSolver 0) is NOT supported by MJX, so a PGS config is rejected up front
    # rather than passing validation and then blowing up at runtime.
    assert not PhysicsConfig("pgs", solver=0).is_valid()
    assert not PhysicsConfig("neg_solref", solref_scale=-1.0).is_valid()


# --------------------------------------------------------------------------- #
# Validity aggregation: reproduction vector + fraction math on synthetic scores
# --------------------------------------------------------------------------- #
def test_reproduces_threshold_boundary():
    assert reproduces(0.5, 0.5) is True       # >= threshold counts (same rule as search)
    assert reproduces(0.49999, 0.5) is False
    assert reproduces(0.9, 0.5) is True


def test_reproduction_vector_and_validity_fraction():
    scores = [0.9, 0.4, 0.55, 0.2, 0.5]
    repro = reproduction_vector(scores, threshold=0.5)
    assert repro == [True, False, True, False, True]
    assert validity_score(repro) == pytest.approx(3 / 5)


def test_validity_all_and_none():
    assert validity_score([True, True, True]) == 1.0
    assert validity_score([False, False]) == 0.0
    assert validity_score([]) == 0.0          # no evidence -> 0


def test_combined_transfer_validity_dominates_and_falls_back():
    # validity present: 0.75*validity + 0.25*plausibility
    assert combined_transfer(1.0, 0.0) == pytest.approx(0.75)
    assert combined_transfer(0.0, 1.0) == pytest.approx(0.25)
    assert combined_transfer(0.8, 0.4, validity_weight=0.75) == pytest.approx(
        0.75 * 0.8 + 0.25 * 0.4
    )
    # validity None -> plausibility only (clean degrade to pre-T1 behavior).
    assert combined_transfer(None, 0.42) == pytest.approx(0.42)
    assert combined_transfer(None, None) == 0.0


# --------------------------------------------------------------------------- #
# select_top + family annotation / re-ranking
# --------------------------------------------------------------------------- #
def _rec(rid, score, transfer=0.5, severity="high", **kw):
    return FailureRecord(
        theta_vector=[0.0], theta_named={"friction_floor": 0.3},
        failure_score=score, component_signals={"fall": 1.0},
        rollout_signature=[0.1], seed=0, severity=severity,
        transfer_score=transfer, id=rid, **kw,
    )


def test_select_top_orders_by_priority_and_limits():
    recs = [_rec("a", 0.6), _rec("b", 0.95), _rec("c", 0.7)]
    top = select_top(recs, top_n=2)
    assert len(top) == 2
    assert top[0].id == "b"                    # highest failure/priority first
    assert set(select_top(recs, None)[i].id for i in range(3)) == {"a", "b", "c"}


def _fam(fid, rep_id, rank):
    return FailureFamily(
        id=fid, name=f"fam {fid}", member_ids=[rep_id], representative_id=rep_id,
        triggering_parameters={"friction_floor": 0.3}, mean_failure_score=0.8,
        mean_transfer_score=0.6, priority_rank=rank,
    )


def test_annotate_families_copies_representative_validity():
    r1 = _rec("r1", 0.9, validity=1.0, reproduction=[True, True, True])
    r2 = _rec("r2", 0.8, validity=0.25, reproduction=[True, False, False, False])
    fams = [_fam("f1", "r1", 1), _fam("f2", "r2", 2)]
    annotate_families(fams, [r1, r2])
    assert fams[0].validity == 1.0 and fams[0].reproduction == [True, True, True]
    assert fams[1].validity == 0.25


def test_rank_families_by_validity_prefers_measured_then_priority():
    f1 = _fam("f1", "r1", 1)   # best priority but...
    f2 = _fam("f2", "r2", 2)
    f1.validity = 0.2
    f2.validity = 0.9          # ...worse-priority family reproduces more -> ranks first
    ranked = rank_families_by_validity([f1, f2])
    assert [f.id for f in ranked] == ["f2", "f1"]


def test_rank_families_unvalidated_keeps_priority_order():
    # No validity anywhere -> identical to the original priority ordering.
    f1 = _fam("f1", "r1", 1)
    f2 = _fam("f2", "r2", 2)
    ranked = rank_families_by_validity([f2, f1])
    assert [f.id for f in ranked] == ["f1", "f2"]


# --------------------------------------------------------------------------- #
# FailureRecord round-trip with the new fields + legacy back-compat
# --------------------------------------------------------------------------- #
def test_failure_record_roundtrips_validity_fields():
    r = _rec("x", 0.9, validity=0.6, reproduction=[True, False, True, True, False])
    back = FailureRecord.from_json(r.to_json())
    assert back == r
    assert back.validity == pytest.approx(0.6)
    assert back.reproduction == [True, False, True, True, False]


def test_failure_record_legacy_json_without_validity_fields():
    """Records serialized before T1 (no validity/reproduction) still load, defaulting
    the new fields to None (same back-compat discipline as Upgrade #6's batch_key)."""
    legacy = json.dumps({
        "theta_vector": [1.0], "theta_named": {"a": 1.0},
        "failure_score": 0.9, "component_signals": {"fall": 1.0},
        "rollout_signature": [0.1], "seed": 7, "severity": "high",
        "transfer_score": 0.8, "priority_rank": 1, "id": "x-0",
        "batch_key": [1, 2], "env_index": 3,
    })
    back = FailureRecord.from_json(legacy)
    assert back.validity is None
    assert back.reproduction is None
    assert back.batch_key == [1, 2] and back.env_index == 3


def test_failure_family_roundtrips_validity_fields():
    f = _fam("f1", "r1", 1)
    f.validity = 0.5
    f.reproduction = [True, False]
    back = FailureFamily(**json.loads(f.to_json()))
    assert back.validity == 0.5 and back.reproduction == [True, False]


# --------------------------------------------------------------------------- #
# Backend safety: the C backend must NOT contribute a fabricated reproduction, and
# validity must refuse a legacy record instead of validating the wrong rollout.
# --------------------------------------------------------------------------- #
def test_c_backend_ready_gate_is_two_conditions():
    """READY (parity ported + measured in tests/test_mujoco_c_parity.py) is now True, but
    ``ready()`` must STILL gate on `mujoco` importing -- the two conditions stay separate
    so a box without the bindings degrades to MJX-only validity, and so a future
    un-ported target can turn the flag off without touching any caller."""
    assert MujocoCBackend.READY is True
    assert MujocoCBackend.ready() is (MujocoCBackend.available() and True)


def test_make_mujoco_c_backend_returns_none_when_not_ready(monkeypatch):
    """Whenever the C backend is not ready, the factory returns None -- not a half-working
    backend -- so validity runs on the MJX configs alone rather than being corrupted by a
    fabricated independent-backend entry."""
    monkeypatch.setattr(MujocoCBackend, "READY", False)
    b = make_mujoco_c_backend(
        env=None, policy_fn=None, target=None, space=None, consts=(0, 1), horizon=8
    )
    assert b is None


def test_c_backend_simulate_raises_not_implemented_when_not_ready(monkeypatch):
    """Even if someone constructs it directly, simulate refuses to run (loud) rather than
    feeding an unverified rollout into validity."""
    monkeypatch.setattr(MujocoCBackend, "READY", False)
    b = MujocoCBackend(
        env=None, policy_fn=None, target=None, space=None, consts=(0, 1), horizon=8
    )
    with pytest.raises(NotImplementedError, match="not ready"):
        b.simulate(_rec("r", 0.9, batch_key=[1, 2], env_index=0))


def test_c_backend_legged_only_guard_precedes_everything():
    """A non-legged objective is refused before any env work, with a clear message -- the
    field math, the consts unpack and the summary hook are all legged."""
    manip = types.SimpleNamespace(objective=types.SimpleNamespace(name="manipulation"))
    b = MujocoCBackend(
        env=None, policy_fn=None, target=manip, space=None, consts=(0, 1), horizon=8
    )
    with pytest.raises(ValueError, match="legged-only"):
        b.simulate(_rec("r", 0.9, batch_key=[1, 2], env_index=0))
    assert make_mujoco_c_backend(
        env=None, policy_fn=None, target=manip, space=None, consts=(0, 1), horizon=8
    ) is None


def test_mjx_backend_key_raises_on_legacy_record():
    """A record without batch_key/env_index cannot reconstruct its exact scored key, so
    the MJX backend raises (mirrors render_failure) instead of fabricating a key -- the
    guard is checked before importing jax, so this runs on a CPU box."""
    b = MjxBackend(
        env=None, policy_fn=None, target=None, space=None, consts=None,
        base_model=None, horizon=8,
    )
    with pytest.raises(ValueError, match="exact-repro"):
        b._key(_rec("legacy", 0.9))          # batch_key/env_index default None


def test_validate_failures_raises_on_record_missing_key():
    """validate_failures refuses a legacy record: it re-sims via the record's OWN recorded
    key, and there is none, so it raises before any (fabricated) validity is computed.
    Reaches the raise without jax: the nominal config is a no-op and the key guard fires
    first."""
    rec = _rec("legacy", 0.9)                 # no batch_key/env_index
    cfg = types.SimpleNamespace(failure_threshold=0.5, thresholds={}, horizon=8)
    with pytest.raises(ValueError, match="exact-repro"):
        validate_failures(
            None, None, [rec], None, cfg, None,
            env=None, consts=None, base_model=None, top_n=1,
        )


# --------------------------------------------------------------------------- #
# C-MuJoCo backend NUMPY field application matches build_legged_fields
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
    """numpy-backed stand-in for a JAX array supporting .at[idx].{set,add,multiply}."""
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    @property
    def at(self):
        return _AtIndexer(self)

    def __getitem__(self, idx):
        return self.arr[idx]


def _source_arrays(nbody=4, ngeom=3, nu=12, ndof=18):
    """Deterministic source arrays for both the _FakeArr model (build_legged_fields)
    and the plain-numpy model (apply_legged_theta_numpy) so they start identical."""
    rng = np.random.default_rng(0)
    return dict(
        geom_friction=rng.uniform(0.5, 1.0, size=(ngeom, 3)),
        body_mass=rng.uniform(0.5, 3.0, size=(nbody,)),
        body_ipos=rng.uniform(-0.01, 0.01, size=(nbody, 3)),
        actuator_gainprm=rng.uniform(10.0, 30.0, size=(nu, 10)),
        actuator_biasprm=rng.uniform(-30.0, 30.0, size=(nu, 10)),
        dof_frictionloss=rng.uniform(0.0, 0.5, size=(ndof,)),
        dof_armature=rng.uniform(0.01, 0.1, size=(ndof,)),
        # geom_quat: identity per geom; the floor-slope dim pitches the floor row.
        geom_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (ngeom, 1)),
        # explicit contact pairs (G1-style): pair 0 touches the floor (geom 0), pair 1 not
        pair_geom1=np.array([0, 1]),
        pair_geom2=np.array([1, 2]),
        pair_friction=np.full((2, 5), 0.6),
    )


def _fake_model(src):
    """``_FakeArr`` fields for build_legged_fields; pair geom ids stay plain int arrays
    (they are static numpy arrays on a real MJX model too)."""
    return types.SimpleNamespace(**{k: (v if k.startswith("pair_geom") else _FakeArr(v))
                                    for k, v in src.items()})


def _scalars():
    return dict(
        friction_floor=0.3, torso_mass_add_kg=2.5, com_offset_x_m=0.01,
        com_offset_y_m=-0.02, kp_scale=1.2, kd_scale=0.8, com_offset_z_m=0.03,
        joint_frictionloss_scale=1.5, joint_armature_scale=2.0,
        floor_slope_rad=0.15,   # T6 GEOMETRY axis
    )


def test_cbackend_numpy_field_application_matches_build_legged_fields():
    """The C-backend host field math must equal the MJX build_legged_fields output on
    the SAME inputs (incl. the [6:] actuated slice + affine Kp/Kd), so cross-backend
    reproduction compares like-for-like physics parameters."""
    src = _source_arrays()
    consts = (0, 1)  # (floor_geom_id, torso_body_id)
    s = _scalars()

    # MJX path: build_legged_fields on a _FakeArr model -> dict of new _FakeArr.
    fake = _fake_model(src)
    expected = build_legged_fields(s, fake, consts)

    # C-MuJoCo path: apply_legged_theta_numpy mutating a plain-numpy model in place.
    numpy_model = types.SimpleNamespace(**{k: v.copy() for k, v in src.items()})
    apply_legged_theta_numpy(numpy_model, s, consts)

    for f in expected:
        np.testing.assert_allclose(
            getattr(numpy_model, f), expected[f].arr,
            err_msg=f"C-backend numpy field {f!r} diverged from build_legged_fields",
        )


def test_cbackend_numpy_field_application_only_touches_expected_entries():
    """Untouched entries (non-floor geoms, non-torso bodies, the 6-dof free base) must
    stay at their base values -- same targeting as build_legged_fields."""
    src = _source_arrays()
    consts = (0, 1)
    s = _scalars()
    numpy_model = types.SimpleNamespace(**{k: v.copy() for k, v in src.items()})
    apply_legged_theta_numpy(numpy_model, s, consts)

    # geom_friction: only [floor, 0] changed.
    np.testing.assert_allclose(numpy_model.geom_friction[0, 1:], src["geom_friction"][0, 1:])
    np.testing.assert_allclose(numpy_model.geom_friction[1:], src["geom_friction"][1:])
    # dof arrays: the 6-dof free base slice [:6] is untouched.
    np.testing.assert_allclose(numpy_model.dof_frictionloss[:6], src["dof_frictionloss"][:6])
    np.testing.assert_allclose(numpy_model.dof_armature[:6], src["dof_armature"][:6])
    # body_mass: non-torso bodies untouched.
    np.testing.assert_allclose(numpy_model.body_mass[2:], src["body_mass"][2:])
    # geom_quat: only the floor row (index 0) is tilted; other geoms stay identity.
    np.testing.assert_allclose(numpy_model.geom_quat[1:], src["geom_quat"][1:])
    # pair_friction: only the floor pair's two tangential coefficients changed.
    np.testing.assert_allclose(numpy_model.pair_friction[0, :2], [s["friction_floor"]] * 2)
    np.testing.assert_allclose(numpy_model.pair_friction[0, 2:], src["pair_friction"][0, 2:])
    np.testing.assert_allclose(numpy_model.pair_friction[1], src["pair_friction"][1])


def test_floor_slope_produces_intended_pitch_quaternion():
    """The numpy mirror tilts the floor geom by the intended pitch about y: from an
    identity base, geom_quat[floor] == [cos(slope/2), 0, sin(slope/2), 0]. This is the
    correctness check the parity test alone can't give (both mirrors could be wrong the
    same way); the parity test then proves build_legged_fields matches this."""
    from breakpoint_mjx.backends.mujoco_c import _quat_mul_np

    slope = 0.15
    numpy_model = types.SimpleNamespace(**{k: v.copy() for k, v in _source_arrays().items()})
    apply_legged_theta_numpy(numpy_model, {**_scalars(), "floor_slope_rad": slope}, (0, 1))
    h = 0.5 * slope
    np.testing.assert_allclose(
        numpy_model.geom_quat[0], [np.cos(h), 0.0, np.sin(h), 0.0], atol=1e-12
    )

    # Hamilton product correctness: composing two y-pitches adds their angles
    # (pitch(a) (x) pitch(b) == pitch(a + b)), which a wrong product order would break.
    def _pitch(a):
        return np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])

    a, b = 0.2, 0.1
    np.testing.assert_allclose(_quat_mul_np(_pitch(a), _pitch(b)), _pitch(a + b), atol=1e-12)


def test_floor_slope_zero_is_identity_noop():
    """floor_slope_rad == 0 leaves geom_quat untouched (identity pitch composed with the
    base), so a nominal slope is a true no-op -- matching the rollout's nominal-safe axes."""
    numpy_model = types.SimpleNamespace(**{k: v.copy() for k, v in _source_arrays().items()})
    src = _source_arrays()
    apply_legged_theta_numpy(numpy_model, {**_scalars(), "floor_slope_rad": 0.0}, (0, 1))
    np.testing.assert_allclose(numpy_model.geom_quat, src["geom_quat"], atol=1e-12)
