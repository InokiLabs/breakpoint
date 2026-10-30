"""Genesis engine backend (cross-SIMULATOR rung 2): replay a FailureRecord in the
Genesis physics engine -- a genuinely different implementation from BOTH MJX and
C MuJoCo (its own collider, its own constraint solver, its own integrator, written
in taichi/torch rather than being another build of the same MuJoCo code).

SCOPE: **URDF-DERIVED TARGETS ONLY** (``breakpoint_mjx.urdf_target.URDFEnv``).
--------------------------------------------------------------------------------
``MujocoCBackend`` reaches parity with a MuJoCo Playground env by handing the foreign
engine's state back to the ENV'S OWN CODE (``mjx.put_data`` -> ``env._get_obs``, and a
scoped swap of ``mjx_env.step``). Genesis cannot do that: it is not a MuJoCo build, so
there is no ``mjx.Data`` to hand Playground, and Playground owns the observation, reset
and command semantics. Hand-porting a 48-channel Playground observation layout is exactly
the silent-mismatch this repo refuses to take on, so **a Playground target is REFUSED**
(clear error, never a fabricated rollout).

A URDF target is different, and that is what makes this shippable: for ``URDFEnv`` *we*
define the observation layout, the reset, the command and the action mapping (see
``urdf_target.URDFEnv``), and the robot is a plain URDF that Genesis loads natively. Both
engines are therefore built from the SAME description, and the mirror below is small
enough to be proven line-by-line against the env's own code
(``tests/test_genesis_backend.py`` asserts ``_urdf_obs_numpy`` reproduces
``URDFEnv._obs`` exactly on random states, the way ``apply_legged_theta_numpy`` is
proven against ``build_legged_fields``).

THE FIVE PARITY POINTS (each measured in tests/test_genesis_backend.py)
----------------------------------------------------------------------
1. MODEL  -- Genesis loads ``report["urdf_path"]`` with ``fixed=False``. Link names,
             link masses and the actuated-joint DOF layout are VALIDATED against the
             MuJoCo model ``URDFEnv`` runs on; any mismatch raises. Every simulation
             parameter is then copied from that MuJoCo model AFTER
             ``apply_legged_theta_numpy`` has applied the record's theta, so the theta
             math is the SAME proven numpy mirror the C backend uses -- Genesis mirrors
             the resulting MODEL, it does not reinterpret the theta.
             Genesis's own defaults (kp=100, kv=10, armature=0.1) are overwritten, not
             inherited: leaving armature at Genesis's 0.1 against MuJoCo's 0 would be a
             silent physics difference that looks like "engine disagreement".
2. RESET + COMMAND -- ``env.reset(key)`` is CALLED (not reimplemented) on the record's
             own key, and Genesis is seeded from the resulting qpos/qvel; the velocity
             command comes out of that same ``State.info``.
3. OBS    -- ``_urdf_obs_numpy``, a numpy mirror of ``URDFEnv._obs`` proven equal to it
             in tests, evaluated on GENESIS's state.
4. STEP   -- ``ctrl = clip(default_pose + action_scale*act, ctrlrange)`` driven into
             Genesis's PD controller (``kp*(target-q) - kv*qd``, clipped to the force
             range -- algebraically MuJoCo's ``position`` actuator), then exactly
             ``env._n_substeps`` Genesis steps at the MuJoCo model's timestep.
5. SUMMARY -- the target's REAL ``objective.summarize`` is called on a duck-typed state
             carrying Genesis's numbers, so the summary is not reimplemented either.

FRAME CONVENTION (measured, and the single most likely place to go silently wrong):
MuJoCo's free joint stores ``qvel[:3]`` in the WORLD frame but ``qvel[3:6]`` in the
BODY-LOCAL frame; Genesis's free-base DOF velocities are WORLD frame for both. So
``mj_qvel[3:6] = R.T @ genesis_ang_world`` and ``genesis_ang_world = R @ mj_qvel[3:6]``.
Everything downstream (``legged_summarize``'s ``ang_vel``, ``URDFEnv._obs``'s
``ang_vel_body``) is expressed in MuJoCo's convention, so the conversion happens once,
at the Genesis boundary.

Import-safety: ``genesis`` / ``jax`` / ``mujoco`` are imported INSIDE methods, so this
module imports on a box with none of them.
"""
from __future__ import annotations

import copy

import numpy as np

from breakpoint_mjx.backends.mujoco_c import (
    LMAX,
    _model_scalars,
    _rollout_scalars,
    apply_legged_theta_numpy,
)

#: Model-dim theta names this backend can express in Genesis. Anything outside this set
#: is REPORTED (raised), never silently ignored -- a theta dim that quietly does nothing
#: in one engine would show up as "engine disagreement" that is really a porting hole.
SUPPORTED_MODEL_DIMS = frozenset({
    "friction_floor",            # -> plane geom friction (Genesis combines pairs with max, as MuJoCo does)
    "torso_mass_add_kg",         # -> links mass_shift (added to inertial_mass, inertia tensor untouched: MuJoCo's body_mass semantics)
    "com_offset_x_m",            # -> links COM shift (added to inertial_pos: MuJoCo's body_ipos semantics)
    "com_offset_y_m",
    "com_offset_z_m",
    "kp_scale",                  # -> set_dofs_kp
    "kd_scale",                  # -> set_dofs_kv
    "joint_frictionloss_scale",  # -> set_dofs_frictionloss
    "joint_armature_scale",      # -> set_dofs_armature
    "floor_slope_rad",           # -> plane morph normal (build-time; the scene is rebuilt when it changes)
})

#: Rollout-dim theta names. These are engine-independent (they live in ``rollout_one``'s
#: step loop, not in the model), so all of them are mirrored here.
SUPPORTED_ROLLOUT_DIMS = frozenset({
    "obs_noise_scale", "latency_steps", "obs_dropout_prob",
    "obs_bias_scale", "push_mag", "push_step_frac",
})

# Set True ONLY with the numbers in tests/test_genesis_backend.py behind it (step-0
# state/observation parity against MJX on the same URDF + close early-trajectory
# agreement). A false READY silently inflates cross-engine agreement -- it would make
# BreakPoint claim engine agreement it never measured -- so this flag is a measurement,
# not an intention.
READY = True


def unsupported_theta_dims(space):
    """``[(dim_name, reason), ...]`` for every theta dim this backend cannot express.

    Called by ``simulate`` BEFORE any Genesis work, so an unmappable dim is a loud
    refusal rather than a re-sim that quietly ran at nominal on that axis.
    """
    out = []
    for name in space.names():
        kind = space.dims[name].kind
        if kind == "model" and name not in SUPPORTED_MODEL_DIMS:
            out.append((name, "no Genesis equivalent for this model field"))
        elif kind == "rollout" and name not in SUPPORTED_ROLLOUT_DIMS:
            out.append((name, "not part of the mirrored rollout_one adversary set"))
        elif kind not in ("model", "rollout"):
            out.append((name, f"unknown theta kind {kind!r}"))
    return out


# --------------------------------------------------------------------------- #
# Small numeric helpers (numpy; no engine imported)
# --------------------------------------------------------------------------- #

def _quat_to_R(q):
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix (both engines use w-first)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _np(x):
    """Genesis returns torch tensors; make them plain float64 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64)


def _urdf_obs_numpy(R, qpos, qvel, last_act, command, *, floating, q0, v0, nu,
                    default_pose):
    """NUMPY mirror of ``urdf_target.URDFEnv._obs`` -- the ONE reimplemented piece.

    ``qpos`` / ``qvel`` are in MUJOCO convention (free base: qpos ``[xyz, wxyz]``,
    qvel ``[v_world, omega_LOCAL]``); ``R`` is the root body's world rotation matrix.
    Block order and contents are copied from ``URDFEnv._obs`` line for line:

        gravity_body   R.T @ [0, 0, -1]        (floating base only)
        lin_vel_body   R.T @ qvel[0:3]         (floating base only)
        ang_vel_body   R.T @ qvel[3:6]         (floating base only)
        joint_pos      qpos[q0:q0+nu] - default_pose
        joint_vel      qvel[v0:v0+nu]
        last_action    info["last_act"]
        command        info["command"]

    ``tests/test_genesis_backend.py::test_obs_mirror_matches_env_obs`` asserts this is
    equal to ``URDFEnv._obs`` on random states, which is what makes it safe to run the
    policy on -- exactly the discipline ``apply_legged_theta_numpy`` follows.
    """
    parts = []
    if floating:
        parts += [
            R.T @ np.array([0.0, 0.0, -1.0]),
            R.T @ np.asarray(qvel[0:3], dtype=float),
            R.T @ np.asarray(qvel[3:6], dtype=float),
        ]
    parts += [
        np.asarray(qpos[q0: q0 + nu], dtype=float) - np.asarray(default_pose, dtype=float),
        np.asarray(qvel[v0: v0 + nu], dtype=float),
        np.asarray(last_act, dtype=float),
        np.asarray(command, dtype=float),
    ]
    return np.concatenate(parts)


class _DuckData:
    """The three ``mjx.Data`` fields ``legged_summarize`` reads, backed by numpy."""

    __slots__ = ("xmat", "qpos", "qvel")

    def __init__(self, xmat, qpos, qvel):
        self.xmat, self.qpos, self.qvel = xmat, qpos, qvel


class _DuckState:
    """Enough of a Playground ``State`` for the target's REAL ``objective.summarize``.

    Summaries are NOT reimplemented here: ``legged_summarize`` is pure indexing /
    reshape / matmul, so it runs unchanged on these numpy arrays and produces the same
    5-key dict ``component_signals`` scores.
    """

    __slots__ = ("data", "info")

    def __init__(self, data, info):
        self.data, self.info = data, info


# --------------------------------------------------------------------------- #
# Genesis process/scene management
# --------------------------------------------------------------------------- #

_GS_INITED = False


def _ensure_genesis(seed=0):
    """Import + one-time ``gs.init`` (CPU, headless). Returns the ``genesis`` module.

    ``gs.init`` is process-global and refuses a second call, so it is done once and
    remembered. Headless by construction: no viewer, no renderer is ever constructed, so
    no GL/display context is required.
    """
    global _GS_INITED
    import genesis as gs

    if not _GS_INITED:
        gs.init(backend=gs.cpu, logging_level="warning", seed=int(seed))
        _GS_INITED = True
    return gs


class _GenesisWorld:
    """One built Genesis scene: a ground plane + the URDF robot, with the index maps.

    Built once per distinct floor NORMAL (the only theta axis Genesis cannot change
    after ``scene.build()``); everything else -- friction, masses, CoM, gains, armature,
    frictionloss, force ranges -- is written per re-sim.
    """

    def __init__(self, urdf_path, mj_model, report, floor_normal, seed=0):
        """``mj_model`` is the NOMINAL (un-perturbed) model: it supplies the timestep,
        gravity and the reference link masses / inertial offsets that the per-re-sim
        shifts in ``apply_model`` are measured against."""
        gs = _ensure_genesis(seed)
        self.gs = gs
        self.floor_normal = tuple(float(v) for v in floor_normal)
        self._nominal_ipos = np.asarray(mj_model.body_ipos, dtype=float).copy()

        grav = tuple(float(v) for v in np.asarray(mj_model.opt.gravity, dtype=float))
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=float(mj_model.opt.timestep), substeps=1, gravity=grav,
            ),
            # Genesis's OWN solver settings are kept (this is a second opinion, not a
            # MuJoCo re-implementation). Torsional/rolling friction are off by default,
            # which matches the URDF floor's condim=3.
            rigid_options=gs.options.RigidOptions(),
            show_viewer=False,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane(normal=self.floor_normal))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=str(urdf_path), fixed=False)
        )
        self.scene.build(n_envs=0)
        self._build_index_maps(mj_model, report)

    # -- validation + index maps -------------------------------------------- #
    def _build_index_maps(self, mj_model, report):
        """Map MuJoCo body/joint ids onto Genesis link/dof indices, or FAIL LOUD.

        Two independent importers read the same URDF; if they disagree about which links
        survived (Genesis merges fixed-joint links by default, MuJoCo welds them) or
        about the DOF ordering, every downstream number would be quietly indexed wrong.
        So the mapping is by NAME, and anything missing raises.
        """
        import mujoco

        self.link_of_body = {}
        for bid in range(1, int(mj_model.nbody)):     # 0 == world
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            link = next((i for i, l in enumerate(self.robot.links) if l.name == name), None)
            if link is None:
                raise NotImplementedError(
                    f"MuJoCo body {name!r} has no Genesis link of the same name "
                    f"(Genesis links: {[l.name for l in self.robot.links]}). The two "
                    f"importers built different kinematic trees, so no cross-engine "
                    f"comparison on this URDF would be meaningful."
                )
            self.link_of_body[bid] = link

        # Actuated joints: model order (== report['actuated_joint_names']) -> Genesis dofs.
        self.act_dofs = []
        for jname in report["actuated_joint_names"]:
            j = next((jj for jj in self.robot.joints if jj.name == jname), None)
            if j is None or int(j.n_dofs) != 1:
                raise NotImplementedError(
                    f"actuated joint {jname!r} is missing from Genesis or is not 1-dof "
                    f"(n_dofs={None if j is None else j.n_dofs}); the URDF target's "
                    f"one-dof-per-actuated-joint layout does not survive the Genesis import."
                )
            self.act_dofs.append(int(np.atleast_1d(_np(j.dofs_idx_local))[0]))

        self.base_dofs = list(range(6))
        nq_expected = int(mj_model.nq)
        nv_expected = int(mj_model.nv)
        if int(self.robot.n_qs) != nq_expected or int(self.robot.n_dofs) != nv_expected:
            raise NotImplementedError(
                f"Genesis built nq={self.robot.n_qs} nv={self.robot.n_dofs} for this URDF "
                f"but the MuJoCo model URDFEnv runs has nq={nq_expected} nv={nv_expected}. "
                f"Refusing to compare two different models."
            )
        if self.act_dofs != list(range(6, 6 + len(self.act_dofs))):
            raise NotImplementedError(
                f"Genesis actuated dofs {self.act_dofs} are not the contiguous block after "
                f"the 6-dof free base; URDFEnv's obs/field slicing assumes they are."
            )
        # Mass parity: a mesh Genesis failed to load (or a link it merged away) shows up
        # here as a mass mismatch rather than as a mysterious dynamics difference later.
        gm = _np(self.robot.get_links_inertial_mass())
        for bid, lid in self.link_of_body.items():
            mm = float(mj_model.body_mass[bid])
            if abs(gm[lid] - mm) > 1e-6 * max(1.0, mm):
                raise NotImplementedError(
                    f"link mass mismatch on body {bid}: MuJoCo {mm:.6g} kg vs Genesis "
                    f"{gm[lid]:.6g} kg. The two importers read different inertials."
                )
        self._nominal_mass = gm.copy()

    # -- per-re-sim model write ---------------------------------------------- #
    def apply_model(self, mj_model, report, consts):
        """Write the (theta-perturbed) MuJoCo model's parameters into Genesis.

        Genesis is mirroring the MODEL, not the theta: the theta was already applied by
        ``apply_legged_theta_numpy`` (the same numpy mirror the C backend uses, unit-
        tested against ``build_legged_fields``), so there is exactly one implementation
        of the theta math across all three engines.
        """
        floor_geom_id, torso_body_id = consts
        robot, plane = self.robot, self.plane

        # -- contact friction. Both engines mix a contact PAIR as max(mu_a, mu_b)
        # (MuJoCo's default friction mixing rule; Genesis's collider does
        # ``max(max(a, b), 1e-2)``), so writing the same per-body numbers reproduces the
        # same effective coefficient -- including the fact that a floor mu BELOW the
        # robot's own geom mu is a no-op in both engines, not just in one.
        plane.set_friction(float(np.clip(mj_model.geom_friction[floor_geom_id, 0], 1e-2, 5.0)))
        by_body = {}
        for g in range(int(mj_model.ngeom)):
            if g == floor_geom_id:
                continue
            by_body.setdefault(int(mj_model.geom_bodyid[g]), []).append(
                float(mj_model.geom_friction[g, 0])
            )
        for bid, mus in by_body.items():
            if max(mus) - min(mus) > 1e-9:
                # Genesis's per-geom friction is not addressable by MuJoCo geom id (a
                # URDF's geoms are unnamed), so per-BODY is the finest granularity that
                # can be mapped safely. Refuse rather than pick one silently.
                raise NotImplementedError(
                    f"body {bid} has geoms with different sliding friction {sorted(set(mus))}; "
                    f"the Genesis mapping is per-link, so this model cannot be mirrored "
                    f"exactly."
                )
            lid = self.link_of_body.get(bid)
            if lid is not None:
                robot.links[lid].set_friction(float(np.clip(mus[0], 1e-2, 5.0)))

        # -- inertial: mass ADD and CoM OFFSET, expressed as Genesis shifts (which are
        # added to inertial_mass / inertial_pos and leave the inertia TENSOR alone --
        # exactly MuJoCo's body_mass / body_ipos semantics).
        n_links = int(robot.n_links)
        mass_shift = np.zeros(n_links)
        com_shift = np.zeros((n_links, 3))
        for bid, lid in self.link_of_body.items():
            mass_shift[lid] = float(mj_model.body_mass[bid]) - float(self._nominal_mass[lid])
            com_shift[lid] = np.asarray(mj_model.body_ipos[bid], dtype=float) - np.asarray(
                self._nominal_ipos[bid], dtype=float
            )
        robot.set_mass_shift(mass_shift, links_idx_local=list(range(n_links)))
        robot.set_COM_shift(com_shift, links_idx_local=list(range(n_links)))

        # -- actuation. MuJoCo's `position` actuator is force = kp*(ctrl - q) - kv*qd,
        # clipped to forcerange; Genesis's PD controller is the same expression.
        kp = np.asarray(mj_model.actuator_gainprm[:, 0], dtype=float)
        kv = -np.asarray(mj_model.actuator_biasprm[:, 2], dtype=float)
        robot.set_dofs_kp(kp, dofs_idx_local=self.act_dofs)
        robot.set_dofs_kv(kv, dofs_idx_local=self.act_dofs)
        fr = np.asarray(mj_model.actuator_forcerange, dtype=float)
        robot.set_dofs_force_range(fr[:, 0], fr[:, 1], dofs_idx_local=self.act_dofs)

        # -- drivetrain. Genesis DEFAULTS armature to 0.1 and kv to 10; a URDF has no
        # armature concept at all, so the MuJoCo model's 0 must be written explicitly or
        # the two engines would be simulating different robots.
        v0 = int(report["qvel_joint_start"])
        nu = int(mj_model.nu)
        robot.set_dofs_armature(np.asarray(mj_model.dof_armature[v0:v0 + nu], dtype=float),
                                dofs_idx_local=self.act_dofs)
        robot.set_dofs_frictionloss(
            np.asarray(mj_model.dof_frictionloss[v0:v0 + nu], dtype=float),
            dofs_idx_local=self.act_dofs)
        robot.set_dofs_damping(np.asarray(mj_model.dof_damping[v0:v0 + nu], dtype=float),
                               dofs_idx_local=self.act_dofs)
        robot.set_dofs_armature(np.asarray(mj_model.dof_armature[:6], dtype=float),
                                dofs_idx_local=self.base_dofs)
        robot.set_dofs_damping(np.asarray(mj_model.dof_damping[:6], dtype=float),
                               dofs_idx_local=self.base_dofs)

    def reset(self):
        """Clear solver state (contacts, mass/COM shifts, controls) before a re-sim."""
        self.scene.reset()

    # -- state I/O (MuJoCo convention in, MuJoCo convention out) ------------- #
    def base_R(self):
        return _quat_to_R(_np(self.robot.get_quat()).reshape(4))

    def set_state_mj(self, qpos_mj, qvel_mj):
        """Seed Genesis from a MUJOCO-convention (qpos, qvel) pair."""
        self.robot.set_qpos(np.asarray(qpos_mj, dtype=float), zero_velocity=True)
        R = self.base_R()
        vel = np.asarray(qvel_mj, dtype=float).copy()
        vel[3:6] = R @ vel[3:6]        # MuJoCo stores base omega LOCAL; Genesis wants WORLD
        self.robot.set_dofs_velocity(vel)

    def get_state_mj(self):
        """Genesis state as ``(qpos, qvel, R)`` in MUJOCO convention."""
        qpos = _np(self.robot.get_qpos()).reshape(-1)
        vel = _np(self.robot.get_dofs_velocity()).reshape(-1)
        R = self.base_R()
        qvel = vel.copy()
        qvel[3:6] = R.T @ vel[3:6]     # WORLD -> body LOCAL, MuJoCo's free-joint convention
        return qpos, qvel, R

    def add_base_linvel(self, dv):
        """The gait-timed push: ``qvel[:3] += dv``. Both engines store base LINEAR
        velocity in the world frame, so no rotation is involved."""
        vel = _np(self.robot.get_dofs_velocity()).reshape(-1)
        vel[:3] = vel[:3] + np.asarray(dv, dtype=float)
        self.robot.set_dofs_velocity(vel)

    def xmat_all(self, nbody):
        """A ``(nbody, 9)`` xmat array, filled for every mapped body (world row zeros)."""
        quats = _np(self.robot.get_links_quat()).reshape(-1, 4)
        out = np.zeros((int(nbody), 9))
        for bid, lid in self.link_of_body.items():
            out[bid] = _quat_to_R(quats[lid]).reshape(9)
        return out


# --------------------------------------------------------------------------- #
# The backend
# --------------------------------------------------------------------------- #

class GenesisBackend:
    """Re-simulate a recorded failure in Genesis. URDF targets only (see module docs)."""

    name = "genesis"
    READY = READY

    def __init__(self, env, policy_fn, target, space, consts, horizon, mj_model=None,
                 seed=0):
        self.env = env
        self.policy_fn = policy_fn
        self.target = target
        self.space = space
        self.consts = consts          # legged (floor_geom_id, torso_body_id)
        self.horizon = int(horizon)
        self._mj_model = mj_model
        self._seed = int(seed)
        self._world = None            # built lazily, rebuilt when the floor normal changes

    # -- gates --------------------------------------------------------------- #
    @staticmethod
    def available() -> bool:
        """True iff the ``genesis`` package is INSTALLED.

        Deliberately ``find_spec`` and not ``import genesis``: importing Genesis pulls in
        torch + taichi and takes MINUTES on a cold CPU box (measured: >100 s), and
        ``available_engines`` is called on every pipeline run, including runs where no
        Genesis re-sim will happen. The real import is deferred to ``_ensure_genesis`` at
        the first ``simulate``; if it then fails, ``replay_matrix`` captures it per-cell
        and it shows up in ``error_rate`` -- never as a silent skip.

        NOT the same as being safe to run -- see ``ready``/``skip_reason``.
        """
        import importlib.util

        try:
            return importlib.util.find_spec("genesis") is not None
        except Exception:
            return False

    @classmethod
    def ready(cls) -> bool:
        return bool(READY) and cls.available()

    @staticmethod
    def supports_env(env) -> bool:
        """True iff this env is a URDF-derived target (the only supported scope)."""
        try:
            from breakpoint_mjx.urdf_target import URDFEnv
        except Exception:
            return False
        return isinstance(env, URDFEnv)

    @classmethod
    def skip_reason(cls, env=None):
        """``None`` if this backend can run here, else a one-line reason for the report.

        ``available_engines`` prints this, so a run that silently dropped Genesis can
        never be mistaken for one where Genesis agreed.
        """
        # The structural reason first: installing genesis would not make a Playground env
        # runnable, so that is the reason to report whether or not genesis is installed.
        if env is not None and not cls.supports_env(env):
            return (
                "target is not URDF-derived; Genesis cannot run a MuJoCo Playground env "
                "(Playground owns reset/obs/command and cannot be called from Genesis)"
            )
        if not cls.available():
            return "genesis not importable"
        if not READY:
            return "URDF parity not ported (READY=False)"
        return None

    # -- entry point --------------------------------------------------------- #
    def simulate(self, record, collect_obs_act=False):
        """Guarded entry point. Refuses -- never fabricates -- outside the ported scope."""
        obj = getattr(self.target, "objective", None)
        if obj is not None and getattr(obj, "name", "legged") != "legged":
            raise ValueError(
                "GenesisBackend is legged-only (legged consts unpack + "
                "apply_legged_theta_numpy + stay-upright summary); target objective "
                f"{getattr(obj, 'name', '?')!r} is not supported."
            )
        if not self.ready():
            raise NotImplementedError(
                "Genesis backend not ready (genesis missing or READY=False)"
            )
        if not self.supports_env(self.env):
            raise NotImplementedError(
                f"GenesisBackend supports URDF-derived targets only; got "
                f"{type(self.env).__name__}. A MuJoCo Playground env owns its own "
                f"observation/reset/command semantics and Genesis cannot call them, so a "
                f"Playground 'Genesis reproduction' would be a hand-ported guess. Import "
                f"the robot with breakpoint_mjx.urdf_target.make_urdf_target to get a "
                f"cross-simulator-checkable target."
            )
        bad = unsupported_theta_dims(self.space)
        if bad:
            raise NotImplementedError(
                "theta dims Genesis cannot express: "
                + "; ".join(f"{n} ({why})" for n, why in bad)
                + ". Refusing rather than re-simulating those axes at nominal, which "
                  "would look like engine agreement on an axis never tested."
            )
        return self._simulate_impl(record, collect_obs_act=collect_obs_act)

    # -- the re-sim ---------------------------------------------------------- #
    def _simulate_impl(self, record, collect_obs_act=False):
        """Mirror ``rollout_one`` step-for-step on Genesis; return its summary dict keys.

        Same recorded key, same per-step 3-way split, same episode-level draws
        (``fold_in(key, 1)`` obs bias, ``fold_in(key, 2)`` push direction), same
        noise -> bias -> dropout (sample-and-hold, sensor channels) order, same latency ring
        buffer (newest at 0), same gait-timed push applied BEFORE the step, same per-target summary hook. The only
        intended difference is the engine underneath.
        """
        # The exact-repro key FIRST: a legacy record must be refused before any engine
        # work, so the failure mode is a clear error rather than a rollout off a
        # fabricated key (same ordering as MjxBackend / MujocoCBackend).
        key = self._key(record)
        env = self.env
        report = env.report

        nominal_model = self._mj_model if self._mj_model is not None else env.mj_model
        mj_model = copy.deepcopy(nominal_model)
        apply_legged_theta_numpy(mj_model, _model_scalars(self.space, record.theta_named),
                                 self.consts)

        rs = _rollout_scalars(self.space, record.theta_named)
        obs_noise_scale = float(rs.get("obs_noise_scale", 0.0))
        obs_dropout_prob = float(rs.get("obs_dropout_prob", 0.0))
        obs_bias_scale = float(rs.get("obs_bias_scale", 0.0))
        push_mag = float(rs.get("push_mag", 0.0))
        push_step_frac = float(rs.get("push_step_frac", 0.0))
        latency_steps = int(np.clip(round(rs.get("latency_steps", 0.0)), 0, LMAX))

        floor_geom_id, torso_body_id = self.consts
        normal = _quat_to_R(mj_model.geom_quat[floor_geom_id]) @ np.array([0.0, 0.0, 1.0])
        world = self._world_for(normal, nominal_model, report)
        world.reset()
        world.apply_model(mj_model, report, self.consts)

        # (2) RESET + COMMAND parity: the env's OWN reset on the record's own key.
        st0 = env.reset(key)
        qpos0 = np.asarray(st0.data.qpos, dtype=np.float64)
        qvel0 = np.asarray(st0.data.qvel, dtype=np.float64)
        command = np.asarray(st0.info["command"], dtype=np.float64)
        world.set_state_mj(qpos0, qvel0)

        floating = bool(report["floating_base"] and report["free_joints"])
        q0, v0 = int(report["qpos_joint_start"]), int(report["qvel_joint_start"])
        nu = int(mj_model.nu)
        default_pose = np.asarray(report["default_pose"], dtype=np.float64)
        ctrl_lo = np.asarray(mj_model.actuator_ctrlrange[:, 0], dtype=np.float64)
        ctrl_hi = np.asarray(mj_model.actuator_ctrlrange[:, 1], dtype=np.float64)
        action_scale = float(env._action_scale)
        n_substeps = int(env._n_substeps)
        summarize = self._summarize()
        horizon = self.horizon

        qpos, qvel, R = world.get_state_mj()
        last_act = np.zeros(nu)
        obs_vec = _urdf_obs_numpy(R, qpos, qvel, last_act, command, floating=floating,
                                  q0=q0, v0=v0, nu=nu, default_pose=default_pose)

        # Episode-level adversary draws -- byte-identical derivation to rollout_one.
        obs_bias = obs_bias_scale * self._normal(self._fold_in(key, 1), obs_vec.shape)
        push_angle = float(self._uniform(self._fold_in(key, 2), (), 0.0, 2.0 * np.pi))
        push_vec = push_mag * np.array([np.cos(push_angle), np.sin(push_angle), 0.0])
        push_step = int(np.clip(round(push_step_frac * (horizon - 1)), 0, horizon - 1))

        from breakpoint_mjx.obs_adversary import obs_sensor_mask, perturb_obs

        sensor_mask = obs_sensor_mask(env, obs_vec.shape[-1])
        held = perturb_obs(np, obs_vec, 0.0, obs_bias, np.zeros(obs_vec.shape, bool),
                           obs_vec, sensor_mask)[0]
        buf = np.zeros((LMAX + 1, nu))
        steps = []
        for t in range(horizon):
            key, nk_noise, nk_policy = self._split3(key)
            gauss = obs_noise_scale * self._normal(nk_noise, obs_vec.shape)
            dropped = (self._uniform(self._fold_in(nk_noise, 1), obs_vec.shape)
                       < obs_dropout_prob)
            # Same sensor-only, sample-and-hold rule as rollout_one.
            perturbed, held = perturb_obs(np, obs_vec, gauss, obs_bias, dropped, held,
                                          sensor_mask)
            if callable(getattr(self.policy_fn, "init_carry", None)):
                raise NotImplementedError(
                    "stateful (history / recurrent) policies are not wired into the "
                    "Genesis backend; use the MJX or C MuJoCo backend")
            act, _ = self.policy_fn({"state": perturbed}, nk_policy)
            act = np.asarray(act, dtype=np.float64).reshape(nu)
            buf = np.concatenate([act[None], buf[:-1]], axis=0)   # newest at index 0
            delayed = buf[latency_steps]                          # lat=0 => no delay

            if push_mag != 0.0 and t == push_step:
                world.add_base_linvel(push_vec)

            # (4) STEP parity: URDFEnv.step's ctrl mapping, driven into Genesis's PD.
            clipped = np.clip(delayed, -1.0, 1.0)
            ctrl = np.clip(default_pose + action_scale * clipped, ctrl_lo, ctrl_hi)
            world.robot.control_dofs_position(ctrl, dofs_idx_local=world.act_dofs)
            for _ in range(n_substeps):
                world.scene.step()

            qpos, qvel, R = world.get_state_mj()
            last_act = clipped
            obs_vec = _urdf_obs_numpy(R, qpos, qvel, last_act, command, floating=floating,
                                      q0=q0, v0=v0, nu=nu, default_pose=default_pose)
            # (5) SUMMARY parity: the target's REAL summarize on Genesis's numbers.
            state = _DuckState(
                _DuckData(world.xmat_all(mj_model.nbody), qpos, qvel),
                {"command": command},
            )
            summary = summarize(state)
            if collect_obs_act:
                summary = {**summary, "obs_state": perturbed, "action": act}
            steps.append(summary)

        return {k: np.asarray([np.asarray(s[k]) for s in steps]) for k in steps[0]}

    # -- scene cache --------------------------------------------------------- #
    def _world_for(self, normal, nominal_model, report):
        """Reuse the built scene unless the floor NORMAL changed.

        The plane's orientation is the ONE theta axis Genesis cannot change after
        ``scene.build()``, so a differently-sloped floor gets its own scene; every other
        axis is written per re-sim by ``apply_model``.
        """
        n = tuple(round(float(v), 9) for v in normal)
        if self._world is None or self._world.floor_normal != n:
            self._world = _GenesisWorld(
                report["urdf_path"], nominal_model, report, n, seed=self._seed
            )
        return self._world

    def _summarize(self):
        obj = getattr(self.target, "objective", None)
        if obj is not None:
            return lambda state: obj.summarize(state, self.consts)
        from breakpoint_mjx.objectives.legged import legged_summarize

        return lambda state: legged_summarize(state, self.consts)

    # --- RNG helpers (jax used ONLY to match rollout_one's key stream) -------- #
    def _key(self, record):
        if record.batch_key is None or record.env_index is None:
            raise ValueError(
                "record predates exact-repro (Upgrade #6); re-run the search to validate"
            )
        import jax
        import jax.numpy as jnp

        return jax.random.fold_in(
            jnp.asarray(record.batch_key, dtype=jnp.uint32), int(record.env_index)
        )

    @staticmethod
    def _split3(key):
        import jax

        k, a, b = jax.random.split(key, 3)
        return k, a, b

    @staticmethod
    def _normal(key, shape):
        import jax

        return np.asarray(jax.random.normal(key, shape), dtype=np.float64)

    @staticmethod
    def _uniform(key, shape, minval=0.0, maxval=1.0):
        import jax

        return np.asarray(jax.random.uniform(key, shape, minval=minval, maxval=maxval),
                          dtype=np.float64)

    @staticmethod
    def _fold_in(key, i):
        import jax

        return jax.random.fold_in(key, i)
