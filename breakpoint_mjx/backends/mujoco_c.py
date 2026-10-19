"""Independent C-MuJoCo re-simulation backend (T1, stronger cross-backend evidence).

Cross-backend agreement (a failure that also reproduces on the standard C MuJoCo
engine, a genuinely DIFFERENT implementation/solver than MJX) is stronger evidence
than same-backend MJX config variation. So this backend re-sims a SINGLE failure on
host CPU via the `mujoco` python bindings and contributes one extra entry to the
validity reproduction vector.

What IS unit-tested on CPU (no mujoco needed): `apply_legged_theta_numpy` -- the NUMPY
mirror of `targets._legged.build_legged_fields` (geom_friction / body_mass / body_ipos
/ actuator_gainprm+biasprm / dof_frictionloss / dof_armature over the [6:] actuated
slice). test_crossval asserts it matches build_legged_fields byte-for-byte on a fake
numpy model, so the field application -- the part most likely to silently diverge from
the MJX path -- is proven independent of a live run.

PARITY STRATEGY (ported; see tests/test_mujoco_c_parity.py for the numbers): the four
env-specific parity points -- reset, observation, command, ctrl -- are NOT hand-ported.
Hand-porting a 48-channel observation layout is exactly where a silent mismatch lives,
so instead the C state is handed BACK to the env's own code:

  * RESET   -- `_reset_with_pre_obs_info` runs the env's own ``reset(key)`` and returns
               both the resulting State and the info dict as it stood when the env
               called ``_get_obs`` (reset mutates ``info['rng']`` in place). The C
               ``MjData`` is seeded from that State's qpos/qvel/ctrl, so the initial
               pose AND the reset key stream are the env's, bit-for-bit.
  * OBS     -- ``env._get_obs(mjx.put_data(mj_model, mj_data), info)``: the env's own
               observation builder, run on the C engine's state. No layout is
               reimplemented here, so no layout can drift.
  * COMMAND -- carried in that same info dict, sampled by the env's reset and resampled
               by the env's own ``step`` (Playground joystick envs resample mid-episode),
               so ``info['command']`` is the env's, not a placeholder.
  * STEP    -- ``_c_engine_physics`` swaps ``mjx_env.step`` (the ONE function a
               Playground env calls to advance physics) for a C-MuJoCo ``mj_step`` loop
               for the duration of ``env.step``. The env therefore computes its own
               ``motor_targets`` (default pose + action*action_scale), its own substep
               count, its own contact/air-time/command bookkeeping and its own obs --
               only the integrator underneath is the independent C solver.

Import-safety: `mujoco` / `jax` / `mujoco_playground` are imported INSIDE methods;
`apply_legged_theta_numpy` uses only numpy, so this module imports on a CPU-only box.
Use `backends.make_mujoco_c_backend(...)` to construct it only when `mujoco` is
available.
"""
from __future__ import annotations

import contextlib

import numpy as np

# Maximum supported latency buffer depth in steps -- kept in lockstep with
# rollout.LMAX (buffer size = LMAX + 1). Duplicated as a plain int here so this
# module stays import-safe (rollout.py imports jax at module top).
LMAX = 4


def _quat_mul_np(q1, q2):
    """Hamilton product of (w, x, y, z) quaternions (numpy mirror of
    targets._legged._quat_mul; kept byte-identical so the floor tilt composes the same)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def apply_legged_theta_numpy(model, scalars, consts):
    """NUMPY mirror of build_legged_fields: apply a failure's theta to a MjModel IN PLACE.

    Operates on plain numpy-array attributes (a real `mujoco.MjModel` exposes exactly
    these as writable numpy arrays; a fake numpy model works identically in tests). The
    math is byte-identical to `targets._legged.build_legged_fields`:

      geom_friction[floor, 0]  = friction_floor            (SET)
      body_mass[torso]        += torso_mass_add_kg         (ADD)
      body_ipos[torso, {0,1,2}]+= com_offset_{x,y,z}_m     (ADD, per axis)
      actuator_gainprm[:, 0]  *= kp_scale                  (Kp)
      actuator_biasprm[:, 1]  *= kp_scale                  (Kp; biasprm[:,1] = -kp)
      actuator_biasprm[:, 2]  *= kd_scale                  (Kd; biasprm[:,2] = -kd)
      dof_frictionloss[6:]    *= joint_frictionloss_scale  (6-dof free base excluded)
      dof_armature[6:]        *= joint_armature_scale      (6-dof free base excluded)
      geom_quat[floor]         = q_pitch(floor_slope_rad) (x) geom_quat[floor]  (tilt)

    ``floor_slope_rad`` is optional (``.get``): absent / 0 -> identity pitch, a no-op.

    Parameters
    ----------
    model   : object with writable numpy-array fields (a mujoco.MjModel, or a fake).
    scalars : {model-dim name: scalar value} -- the failure's model-dim theta values.
    consts  : (floor_geom_id, torso_body_id).

    Returns the same ``model`` (mutated in place), for convenience.
    """
    floor_geom_id, torso_body_id = consts

    model.geom_friction[floor_geom_id, 0] = scalars["friction_floor"]
    model.body_mass[torso_body_id] += scalars["torso_mass_add_kg"]
    model.body_ipos[torso_body_id, 0] += scalars["com_offset_x_m"]
    model.body_ipos[torso_body_id, 1] += scalars["com_offset_y_m"]
    model.body_ipos[torso_body_id, 2] += scalars["com_offset_z_m"]
    # Affine/position actuators: gainprm[:,0]=kp, biasprm[:,1]=-kp, biasprm[:,2]=-kd.
    model.actuator_gainprm[:, 0] *= scalars["kp_scale"]
    model.actuator_biasprm[:, 1] *= scalars["kp_scale"]
    model.actuator_biasprm[:, 2] *= scalars["kd_scale"]
    # dof vectors are length (6 free base + nu actuated); [6:] = the nu actuated joints.
    model.dof_frictionloss[6:] *= scalars["joint_frictionloss_scale"]
    model.dof_armature[6:] *= scalars["joint_armature_scale"]
    # Floor slope (GEOMETRY axis): pitch the floor geom about y by floor_slope_rad,
    # composed with its base orientation. Mirror of build_legged_fields' geom_quat math.
    slope = float(scalars.get("floor_slope_rad", 0.0))
    half = 0.5 * slope
    q_pitch = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    base_q = np.asarray(model.geom_quat[floor_geom_id], dtype=float)
    model.geom_quat[floor_geom_id] = _quat_mul_np(q_pitch, base_q)
    return model


def _model_scalars(space, theta_named):
    """Extract the kind=="model" theta values (the ones build_legged_fields consumes)."""
    return {
        n: float(theta_named[n])
        for n in space.names()
        if space.dims[n].kind == "model" and n in theta_named
    }


def _rollout_scalars(space, theta_named):
    """Extract the kind=="rollout" theta values (obs_noise_scale / latency_steps)."""
    return {
        n: float(theta_named[n])
        for n in space.names()
        if space.dims[n].kind == "rollout" and n in theta_named
    }


def _env_impl(env):
    """The mjx array implementation ("jax"/"warp") this env's model was put under.

    ``mjx.put_data`` must build a Data of the SAME impl the env's ``_get_obs`` will read,
    so it is read off the env rather than assumed.
    """
    impl = getattr(getattr(env, "mjx_model", None), "impl", None)
    return getattr(impl, "value", impl) or "jax"


def _reset_with_pre_obs_info(env, key):
    """``env.reset(key)`` + the info dict AS IT STOOD when the env called ``_get_obs``.

    A Playground reset builds ``info`` (command, rng, last_act, feet_air_time, the
    perturbation schedule, ...) and then calls ``self._get_obs(data, info)``, which
    advances ``info['rng']`` IN PLACE (one split per noisy observation channel). The
    returned ``State.info`` is therefore already past those splits, so it cannot be used
    to rebuild the reset observation on a different engine's state.

    Shadowing the bound ``_get_obs`` with a spy for the duration of the reset captures
    the pre-observation dict without reimplementing any of the reset's key stream -- the
    env still does every draw itself. Returns ``(state, pre_obs_info)``.
    """
    get_obs = getattr(type(env), "_get_obs", None)
    if get_obs is None:
        raise NotImplementedError(
            "MujocoCBackend needs the env's own observation builder (_get_obs) to reach "
            f"observation parity; {type(env).__name__} does not define one. Hand-porting "
            "the observation layout is exactly the silent-mismatch risk this backend "
            "refuses to take."
        )
    holder = {}

    def spy(data, info):
        holder.setdefault("info", dict(info))
        return get_obs(env, data, info)

    env._get_obs = spy          # instance attribute shadows the bound class method
    try:
        state = env.reset(key)
    finally:
        env.__dict__.pop("_get_obs", None)
    if "info" not in holder:
        raise NotImplementedError(
            f"{type(env).__name__}.reset did not call _get_obs, so the pre-observation "
            "info (command + rng) could not be captured; the C backend cannot claim obs "
            "parity for this env."
        )
    return state, holder["info"]


@contextlib.contextmanager
def _c_engine_physics(mj_model, mj_data, impl, calls):
    """Run ``env.step``'s PHYSICS on C MuJoCo while the env does everything else.

    Every ``mujoco_playground`` env advances physics through the single module-level
    ``mjx_env.step(model, data, action, n_substeps)``. Swapping that one function for a
    C ``mj_step`` loop is what makes this an independent-ENGINE re-sim rather than a
    reimplementation: the env still computes its own motor targets from the action
    (default pose + action*action_scale), its own substep count, its own contact /
    air-time / command bookkeeping, and its own observation.

    ``mj_data`` is the persistent float64 C state (the authority); the incoming mjx
    ``data`` is only read for ``xfrc_applied`` so an env-side perturbation still lands.
    ``calls`` is a one-element list incremented per call, so a caller can assert the
    swap actually fired instead of silently re-simulating on MJX.

    The swap is a module-global rebind held only for the duration of one ``env.step``,
    so two C re-sims must not run CONCURRENTLY in one process (the replay loops here are
    sequential). Restored in ``finally``, so an env exception cannot leave MJX unpatched.
    """
    import mujoco
    from mujoco import mjx
    from mujoco_playground._src import mjx_env

    original = mjx_env.step

    def c_step(model, data, action, n_substeps=1):
        calls[0] += 1
        xfrc = getattr(data, "xfrc_applied", None)
        if xfrc is not None:
            mj_data.xfrc_applied[:] = np.asarray(xfrc, dtype=np.float64)
        mj_data.ctrl[:] = np.asarray(action, dtype=np.float64)
        for _ in range(int(n_substeps)):
            mujoco.mj_step(mj_model, mj_data)
        return mjx.put_data(mj_model, mj_data, impl=impl)

    mjx_env.step = c_step
    try:
        yield
    finally:
        mjx_env.step = original


class _JitEnvStep:
    """The env's own reset/step, COMPILED, with the physics still on C MuJoCo.

    ``env.step`` cannot be jitted whole because its physics call is a host-side C
    ``mj_step``. It is split in two compiled passes around that call instead, each
    tracing the env's OWN ``step`` with ``mjx_env.step`` swapped:

      * ``pre(state, action)``       -- the swap records ``(ctrl, xfrc_applied)`` the env
        hands to physics and returns them (everything after is dead code, pruned).
      * host                         -- ``mj_step`` x n_substeps on the float64 C state.
      * ``post(state, action, data)`` -- the swap returns the C result, so the env computes
        its own contacts, obs, rewards, command resampling on it.

    No env logic is re-implemented; the ops are the same as the eager path, only
    compiled. ``reset`` is ``_reset_with_pre_obs_info`` jitted. One instance per
    (env, policy_fn), cached, so the compile (~10 s on CPU) is paid once per process.
    """

    _cache = {}

    def __init__(self, env, policy_fn):
        import jax

        self.env = env
        self.n_substeps = None
        self.reset = jax.jit(lambda key: _reset_with_pre_obs_info(env, key))
        self.policy = jax.jit(policy_fn)
        self._pre = jax.jit(self._pre_impl)
        self._post = jax.jit(self._post_impl)

    @classmethod
    def for_env(cls, env, policy_fn):
        k = (id(env), id(policy_fn))
        if k not in cls._cache:
            cls._cache[k] = cls(env, policy_fn)
        return cls._cache[k]

    @staticmethod
    @contextlib.contextmanager
    def _swap(fn):
        from mujoco_playground._src import mjx_env

        original = mjx_env.step
        mjx_env.step = fn
        try:
            yield
        finally:
            mjx_env.step = original

    def _pre_impl(self, state, action):
        cap = {}

        def record(model, data, ctrl, n_substeps=1):
            cap["ctrl"], cap["xfrc"] = ctrl, data.xfrc_applied
            self.n_substeps = int(n_substeps)
            return data

        with self._swap(record):
            self.env.step(state, action)
        if "ctrl" not in cap:
            raise NotImplementedError(
                f"{type(self.env).__name__}.step did not route physics through "
                "mjx_env.step; refusing to run it on the C engine.")
        return cap["ctrl"], cap["xfrc"]

    def _post_impl(self, state, action, next_data):
        with self._swap(lambda model, data, ctrl, n_substeps=1: next_data):
            return self.env.step(state, action)

    def step(self, state, action, mj_model, mj_data, impl, calls):
        import mujoco
        from mujoco import mjx

        ctrl, xfrc = self._pre(state, action)
        mj_data.xfrc_applied[:] = np.asarray(xfrc, dtype=np.float64)
        mj_data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        for _ in range(int(self.n_substeps)):
            mujoco.mj_step(mj_model, mj_data)
        calls[0] += 1
        return self._post(state, action, mjx.put_data(mj_model, mj_data, impl=impl))


class MujocoCBackend:
    """Re-simulate a single recorded failure on the standard C MuJoCo engine (host CPU).

    Independent of MJX: a different constraint solver / integrator implementation. Used
    for the "cross-backend agreement" evidence entry in the validity vector.

    READY (parity ported): reset, observation, command and ctrl all come from the env's
    own code running on the C engine's state (see the module docstring), so the frozen
    policy sees in-distribution input and a reproduction means something. The parity is
    proven numerically in ``tests/test_mujoco_c_parity.py`` (opt-in ``BP_RUN_CBACKEND=1``):
    step-0 qpos/qvel/command are identical to the MJX rollout's and the first observation
    agrees to ~1e-7 absolute, which is the float32 round-off floor of the two forward
    solves. Later steps DIVERGE -- that is the product, not a bug -- so nothing here
    asserts long-horizon agreement.

    The two backends are only comparable because they share everything except the
    integrator: the same recorded key, the same per-step key split, the same obs-space
    adversaries (noise / dropout / per-episode bias), the same latency ring buffer, the
    same gait-timed push and the same per-target summary hook.
    """

    name = "mujoco_c"

    # True only because reset / obs / command / ctrl parity is ported AND measured
    # (tests/test_mujoco_c_parity.py). Flipping this without those numbers would let a
    # bogus rollout contribute a fabricated reproduction and silently inflate validity.
    READY = True

    def __init__(self, env, policy_fn, target, space, consts, horizon, mj_model=None,
                 jit_env=False, early_stop=None):
        self.env = env
        self.policy_fn = policy_fn
        self.target = target
        self.space = space
        self.consts = consts  # legged (floor_geom_id, torso_body_id) -- this backend is
                              # legged-ONLY (see the guard in `simulate`).
        self.horizon = int(horizon)
        self._mj_model = mj_model  # optional pre-supplied host MjModel
        # jit_env: run the env's OWN reset/step compiled (``_JitEnvStep``) instead of
        # eagerly. Same ops, same key stream, same C physics; ~20x faster on CPU. Off by
        # default so the parity-proven eager path stays the reference.
        self.jit_env = bool(jit_env)
        # early_stop: optional ``fn(t, summary) -> bool``; when it returns True the
        # rollout ends after that step (e.g. 1 s after a fall, when the verdict can no
        # longer change). The returned arrays are then SHORTER than ``horizon``.
        self.early_stop = early_stop

    @staticmethod
    def available() -> bool:
        """True iff the `mujoco` python bindings import.

        NOTE: importability alone does NOT mean the backend is safe to run -- the reset /
        obs / command parity must also be ported (see ``ready``). That separation is the
        guard that kept an importable-but-stubbed build off the validity vector.
        """
        try:
            import mujoco  # noqa: F401
        except Exception:
            return False
        return True

    @classmethod
    def ready(cls) -> bool:
        """True iff the C backend can produce a TRUSTWORTHY reproduction.

        Requires both (a) `mujoco` importable and (b) ``READY`` -- the env-reset / obs /
        command parity actually ported. Both hold now; the two conditions stay separate
        so a box without the bindings still degrades to MJX-only validity, and so a
        future un-ported target can turn the flag off without touching callers.
        """
        return cls.available() and cls.READY

    def _host_model(self):
        """The host-side C `mujoco.MjModel` to perturb + step.

        A mujoco_playground MJX env exposes the host MjModel as `env.mj_model` (fall back
        to `env.unwrapped.mj_model`). Copied per re-sim so a theta perturbation never
        leaks across failures.
        """
        import copy

        if self._mj_model is not None:
            return copy.deepcopy(self._mj_model)
        m = getattr(self.env, "mj_model", None)
        if m is None:
            m = self.env.unwrapped.mj_model
        return copy.deepcopy(m)

    def simulate(self, record, collect_obs_act=False, collect_states=False):
        """Guarded entry point: refuse to run unless parity is ported (see ``ready``).

        Raises NotImplementedError while ``READY`` is False so a stubbed rollout can NEVER
        be scored into validity (which would inflate it with a fake reproduction). With
        parity ported and measured, this delegates to ``_simulate_impl``.

        Legged-ONLY (T7): the field math (apply_legged_theta_numpy), the consts unpack
        (floor_geom_id, torso_body_id), and the stay-upright summary are all legged, so a
        non-legged target is refused with a clear error rather than a cryptic dict unpack.
        """
        obj = getattr(self.target, "objective", None)
        if obj is not None and getattr(obj, "name", "legged") != "legged":
            raise ValueError(
                "MujocoCBackend is legged-only (hardcoded legged consts + "
                "apply_legged_theta_numpy + stay-upright summary); target objective "
                f"{getattr(obj, 'name', '?')!r} is not supported. A manipulation "
                "cross-backend port is needed for that vertical."
            )
        if not self.ready():
            raise NotImplementedError(
                "C-MuJoCo backend not ready (mujoco missing or READY=False)"
            )
        return self._simulate_impl(record, collect_obs_act=collect_obs_act,
                                   collect_states=collect_states)

    def _simulate_impl(self, record, collect_obs_act=False, collect_states=False):
        """Re-sim `record` on C MuJoCo and return the SAME summary dict keys as rollout_one.

        Mirrors `rollout_one` step-for-step so `failure_score` is comparable -- the same
        recorded key, the same per-step 3-way split, the same episode-level adversary
        draws (fold_in(key, 1) obs bias, fold_in(key, 2) push direction), the same
        noise -> bias -> dropout order, the same latency ring buffer (newest at index 0,
        delayed = buf[lat]), the same gait-timed push before the step, and the same
        per-target ``summarize`` hook. The ONLY intended difference is the integrator:
        C MuJoCo instead of MJX.

        The four parity points (see the module docstring for why none of them is
        hand-ported):
          1. RESET   -- ``_reset_with_pre_obs_info`` runs the env's own reset; the C
                        MjData is seeded from that state's qpos/qvel/ctrl.
          2. OBS     -- ``env._get_obs`` on ``mjx.put_data`` of the C state, threaded
                        through the env's own (mutating) info dict.
          3. COMMAND -- lives in that info dict; the env's reset samples it and the env's
                        own ``step`` resamples it mid-episode.
          4. STEP    -- ``_c_engine_physics`` swaps only ``mjx_env.step``, so the env
                        still builds motor_targets from the action and picks n_substeps.

        The C ``MjData`` is float64 and stays the authority across steps; the mjx Data
        handed to the env is a float32 view of it (``put_data``), which is exactly the
        precision the MJX side runs at.

        ``collect_states=True`` returns ``(summary, states)`` where ``states`` is the reset
        state followed by every post-step state (the same list layout render_failure
        feeds to ``env.render``), so an independent replay can be rendered.
        """
        import mujoco
        from mujoco import mjx

        from breakpoint_mjx.rollout import _obs_state, _obs_with_state

        key = self._key(record)  # raises on a legacy record (no exact-repro key)
        model = self._host_model()
        scalars = _model_scalars(self.space, record.theta_named)
        apply_legged_theta_numpy(model, scalars, self.consts)

        rs = _rollout_scalars(self.space, record.theta_named)
        obs_noise_scale = float(rs.get("obs_noise_scale", 0.0))
        obs_dropout_prob = float(rs.get("obs_dropout_prob", 0.0))
        obs_bias_scale = float(rs.get("obs_bias_scale", 0.0))
        push_mag = float(rs.get("push_mag", 0.0))
        push_step_frac = float(rs.get("push_step_frac", 0.0))
        latency_steps = int(np.clip(round(rs.get("latency_steps", 0.0)), 0, LMAX))

        env = self.env
        horizon = self.horizon
        impl = _env_impl(env)
        summarize = self._summarize()

        # (1) RESET parity: the env's own reset supplies qpos/qvel (randomized pose), the
        # command, and the pre-observation info dict (rng included).
        fast = _JitEnvStep.for_env(env, self.policy_fn) if self.jit_env else None
        st_mjx, info = fast.reset(key) if fast else _reset_with_pre_obs_info(env, key)
        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray(st_mjx.data.qpos, dtype=np.float64)
        data.qvel[:] = np.asarray(st_mjx.data.qvel, dtype=np.float64)
        data.ctrl[:] = np.asarray(st_mjx.data.ctrl, dtype=np.float64)
        mujoco.mj_forward(model, data)

        # (2)+(3) OBS/COMMAND parity: the env's own _get_obs on the C state, advancing the
        # same info dict (command + rng) the env's step will keep advancing.
        c_data0 = mjx.put_data(model, data, impl=impl)
        obs = env._get_obs(c_data0, info)
        state = st_mjx.replace(data=c_data0, obs=obs, info=info)

        # Episode-level adversary draws -- byte-identical derivation to rollout_one.
        vec_shape = np.asarray(_obs_state(obs)).shape
        obs_bias = obs_bias_scale * self._normal(self._fold_in(key, 1), vec_shape)
        push_angle = float(self._uniform(self._fold_in(key, 2), (), 0.0, 2.0 * np.pi))
        push_vec = push_mag * np.array([np.cos(push_angle), np.sin(push_angle), 0.0])
        push_step = int(np.clip(round(push_step_frac * (horizon - 1)), 0, horizon - 1))

        buf = np.zeros((LMAX + 1, int(env.action_size)), dtype=np.float32)
        # Stateful policy (history / recurrent): same carry rule as rollout_one, built
        # from the reset obs plus the episode's sensor bias.
        pol = fast.policy if fast else self.policy_fn
        stateful = callable(getattr(self.policy_fn, "init_carry", None))
        pcarry = (self.policy_fn.init_carry(_obs_with_state(
            state.obs, np.asarray(_obs_state(state.obs)) + obs_bias)) if stateful else None)
        calls = [0]
        steps = []
        states = [state] if collect_states else None
        for t in range(horizon):
            key, nk_noise, nk_policy = self._split3(key)
            vec = np.asarray(_obs_state(state.obs))
            gauss = obs_noise_scale * self._normal(nk_noise, vec.shape)
            keep = (
                self._uniform(self._fold_in(nk_noise, 1), vec.shape) >= obs_dropout_prob
            ).astype(vec.dtype)
            # Dropout zeros channels LAST, so a dropped channel reads exactly 0 (rollout_one).
            perturbed = _obs_with_state(state.obs, (vec + gauss + obs_bias) * keep)
            if stateful:
                act, pcarry = pol(perturbed, nk_policy, pcarry)
            else:
                act, _ = pol(perturbed, nk_policy)
            buf = np.concatenate([np.asarray(act, dtype=np.float32)[None], buf[:-1]], axis=0)
            delayed = buf[latency_steps]                      # lat=0 => no delay
            # Gait-timed push on the base linear velocity, applied to the float64 C state
            # BEFORE stepping so the impulse is integrated this step (rollout_one order).
            if push_mag != 0.0 and t == push_step:
                data.qvel[:3] += push_vec
            if fast:
                state = fast.step(state, delayed, model, data, impl, calls)
            else:
                with _c_engine_physics(model, data, impl, calls):
                    state = env.step(state, delayed)
            if calls[0] != t + 1:
                raise NotImplementedError(
                    f"{type(env).__name__}.step did not route physics through "
                    "mjx_env.step, so this re-sim would have run on MJX, not the C "
                    "engine. Refusing to report a cross-engine reproduction."
                )
            summary = summarize(state)
            if collect_obs_act:
                # Same pair rollout_one collects: the obs the policy actually SAW
                # (post noise/bias/dropout) and its RAW action (pre-latency buffer).
                summary = {**summary, "obs_state": _obs_state(perturbed), "action": act}
            steps.append(summary)
            if collect_states:
                states.append(state)
            if self.early_stop is not None and self.early_stop(t, summary):
                break

        out = {k: np.asarray([np.asarray(s[k]) for s in steps]) for k in steps[0]}
        return (out, states) if collect_states else out

    def _summarize(self):
        """The ACTIVE target's per-step summary hook bound to consts (same as MjxBackend).

        Falls back to the legged summary when no target is supplied (direct construction
        in tests); ``simulate`` has already refused any non-legged objective.
        """
        obj = getattr(self.target, "objective", None)
        if obj is not None:
            return lambda state: obj.summarize(state, self.consts)
        from breakpoint_mjx.objectives.legged import legged_summarize

        return lambda state: legged_summarize(state, self.consts)

    # --- RNG + obs helpers (jax used only to MATCH rollout_one's key stream) --------- #
    def _key(self, record):
        """Reconstruct the recorded eval key (see MjxBackend._key); jax kept lazy.

        RAISES for a legacy record (no batch_key/env_index) rather than fabricating a key
        -- validating against the wrong rollout would be silent corruption. Guard checked
        before importing jax so it is CPU-testable.
        """
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

        return np.asarray(jax.random.normal(key, shape))

    @staticmethod
    def _uniform(key, shape, minval=0.0, maxval=1.0):
        import jax

        return np.asarray(jax.random.uniform(key, shape, minval=minval, maxval=maxval))

    @staticmethod
    def _fold_in(key, i):
        import jax

        return jax.random.fold_in(key, i)


# --------------------------------------------------------------------------- #
# Full-collision variant of the search scene (independent-replay evidence)
# --------------------------------------------------------------------------- #
# env_name -> "module:function" returning (xml_path, assets) of a FULL-collision scene of
# the same robot. Only targets listed here get the Playground scene; every other legged
# Playground target falls back to enabling body-floor contacts on its own loaded model.
FULL_COLLISION_SCENES = {
    "Go1JoystickFlatTerrain": "breakpoint_mjx.targets.go1:full_collision_scene",
}

# Model arrays copied from the env's (possibly env-modified) model onto the full-collision
# model so the two differ ONLY in contact geometry: same masses, gains, damping, limits.
_COPY_FIELDS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_damping", "dof_armature",
    "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
    "jnt_range", "qpos0",
)
_COPY_OPT = (
    "timestep", "iterations", "ls_iterations", "solver", "cone", "impratio", "tolerance",
    "ls_tolerance", "noslip_iterations", "ccd_iterations", "integrator", "disableflags",
    "enableflags",
)


def _names(model, objtype, n):
    import mujoco

    return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


def layout_mismatches(base, cand):
    """Everything the env's own reset/step/_get_obs index by id that differs between the
    env's model ``base`` and a candidate replacement ``cand``. Empty list = drop-in safe.

    The env reads qpos/qvel/ctrl, sensordata by its OWN model's sensor_adr, sites by id and
    the torso by body id. Geoms are only read through ``consts`` (the floor id), so the
    geom COUNT may differ but the floor id may not.
    """
    import mujoco

    out = []
    for f in ("nq", "nv", "nu", "nbody", "nsite", "nsensordata"):
        if getattr(base, f) != getattr(cand, f):
            out.append(f"{f} {getattr(base, f)} != {getattr(cand, f)}")
    for objtype, n, label in ((mujoco.mjtObj.mjOBJ_BODY, "nbody", "body"),
                              (mujoco.mjtObj.mjOBJ_SITE, "nsite", "site")):
        nb, nc = getattr(base, n), getattr(cand, n)
        if nb == nc and _names(base, objtype, nb) != _names(cand, objtype, nc):
            out.append(f"{label} names/order differ")
    for i, nm in enumerate(_names(base, mujoco.mjtObj.mjOBJ_SENSOR, base.nsensor)):
        try:
            j = cand.sensor(nm).id
        except Exception:  # noqa: BLE001 -- KeyError across mujoco versions
            out.append(f"sensor {nm} missing")
            continue
        if (int(base.sensor_adr[i]), int(base.sensor_dim[i])) != (
                int(cand.sensor_adr[j]), int(cand.sensor_dim[j])):
            out.append(f"sensor {nm} address differs")
    try:
        if base.geom("floor").id != cand.geom("floor").id:
            out.append("floor geom id differs")
    except Exception:  # noqa: BLE001
        out.append("no geom named 'floor'")
    return out


def _match_physics(base, cand):
    """Copy every non-geometry physics parameter from ``base`` onto ``cand`` in place."""
    for f in _COPY_FIELDS:
        a, b = getattr(base, f, None), getattr(cand, f, None)
        if a is not None and b is not None and np.shape(a) == np.shape(b):
            b[...] = a
    for f in _COPY_OPT:
        if hasattr(base.opt, f) and hasattr(cand.opt, f):
            setattr(cand.opt, f, getattr(base.opt, f))
    fl = base.geom("floor").id
    for f in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority"):
        getattr(cand, f)[fl] = getattr(base, f)[fl]
    return cand


def enable_body_floor_contacts(base):
    """Copy of ``base`` with every robot geom made collidable against the floor.

    Sets ``conaffinity=1`` (``contype`` untouched) on each non-world geom that currently
    collides with nothing, so it contacts the floor (``contype=1``) but not other robot
    parts. On a feet-only scene these are the visual meshes, so contacts use their convex
    hulls. Returns ``(model, n_geoms_enabled)``.
    """
    import copy

    m = copy.deepcopy(base)
    sel = (np.asarray(m.geom_bodyid) != 0) & (np.asarray(m.geom_contype) == 0) & (
        np.asarray(m.geom_conaffinity) == 0)
    m.geom_conaffinity[sel] = 1
    # The compiled model caches per-body OR-ed collision masks for the broadphase and a
    # per-body BVH over the geoms that were collidable AT COMPILE TIME (midphase). Both
    # must follow the edit or the new contacts are silently filtered out (measured: the
    # trunk falls straight through). Update the masks and turn the midphase off; it is
    # only an acceleration structure, so contacts are unchanged, just found by brute force.
    import mujoco

    if hasattr(m, "body_conaffinity"):
        for b in set(int(x) for x in np.asarray(m.geom_bodyid)[sel]):
            m.body_conaffinity[b] = int(m.body_conaffinity[b]) | 1
    m.opt.disableflags = int(m.opt.disableflags) | int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    return m, int(sel.sum())


def build_full_collision_model(env, target=None):
    """A FULL-collision C MuJoCo model that is a drop-in for ``env``'s own model.

    Returns ``(mj_model, info)``. ``info["source"]`` says which of three things happened,
    so the report states it rather than silently falling back:

    * ``"playground_scene"`` -- the target has a Playground full-collision scene
      (``FULL_COLLISION_SCENES``); it is compiled with the sensors the env does not know
      pruned, passes ``layout_mismatches`` and is given the env's physics parameters.
    * ``"enabled_contacts"`` -- no such scene (or it failed the layout check): the env's
      own model with body-floor contacts switched on (``enable_body_floor_contacts``).
    * ``"unavailable"``      -- ``mujoco`` missing or no host model; ``mj_model`` is None
      and ``info["reason"]`` says why.
    """
    try:
        import mujoco
    except Exception as e:  # noqa: BLE001
        return None, {"source": "unavailable", "reason": f"mujoco not importable ({e})"}
    base = getattr(env, "mj_model", None)
    if base is None:
        return None, {"source": "unavailable", "reason": "env exposes no host mj_model"}
    info = {"source": None, "reason": None}
    env_name = getattr(target, "env_name", None) or type(env).__name__
    hook = FULL_COLLISION_SCENES.get(env_name)
    if hook:
        try:
            import importlib

            mod, fn = hook.split(":")
            xml, assets = getattr(importlib.import_module(mod), fn)()
            spec = mujoco.MjSpec.from_file(str(xml))
            keep = set(_names(base, mujoco.mjtObj.mjOBJ_SENSOR, base.nsensor))
            for s in list(spec.sensors):
                if s.name not in keep:
                    spec.delete(s)
            spec.assets = assets
            cand = spec.compile()
            bad = layout_mismatches(base, cand)
            if not bad:
                _match_physics(base, cand)
                n_col = int(((np.asarray(cand.geom_contype) != 0)
                             | (np.asarray(cand.geom_conaffinity) != 0)).sum())
                info.update(source="playground_scene",
                            scene=str(xml).replace("\\", "/").rsplit("/", 1)[-1],
                            n_collidable=n_col)
                return cand, info
            info["reason"] = ("playground full-collision scene failed the layout check: "
                              + "; ".join(bad[:3]))
        except Exception as e:  # noqa: BLE001 -- fall back, but say why
            info["reason"] = (f"playground full-collision scene unusable "
                              f"({type(e).__name__}: {e})")
    m, n = enable_body_floor_contacts(base)
    if n == 0:
        return None, {"source": "unavailable",
                      "reason": ((info["reason"] + "; ") if info["reason"] else "")
                      + "no non-colliding robot geoms to enable"}
    info.update(source="enabled_contacts", n_enabled=n,
                reason=info["reason"] or f"no full-collision scene registered for {env_name}")
    return m, info
