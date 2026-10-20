"""Single source of truth for a one-env rollout (Upgrade #6: exact reproduction).

Both the GPU evaluator (vmapped over N) and the renderer call rollout_one, so the
rendered failure reproduces the scored failure bit-for-bit: same physics, same
per-step key split, same latency buffer, same summary. The renderer can no longer
drift from the evaluator because there is exactly one step loop.

Runs on CPU jax as well as GPU; tests/test_gpu_smoke.py asserts the determinism
invariant below on a real simulator.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Maximum supported latency buffer depth in steps (buffer size = LMAX + 1).
LMAX = 4


def env_key(batch_key, i):
    """Canonical per-env key derivation: fold_in(batch_key, i).

    Position-stable: row i always gets the same key regardless of batch size N
    (unlike split(key, N)). This is the single rule the evaluator and the renderer
    share, so a recorded (batch_key, env_index) reconstructs the exact eval key.
    """
    return jax.random.fold_in(batch_key, i)


def _obs_state(obs):
    """Read the perturbable observation vector from ``obs``, whichever shape a target uses.

    Legged Playground envs return a DICT ``{"state": ..., "privileged_state": ...}`` and we
    perturb only ``obs["state"]``; manipulation envs (e.g. PandaPickCube) return a PLAIN
    ARRAY and we perturb the whole vector. The dict-vs-array test is a static Python-type
    check resolved at trace time, so the legged (dict) path is byte-identical to before.
    """
    return obs["state"] if isinstance(obs, dict) else obs


def _obs_with_state(obs, new_state):
    """Write ``new_state`` back, mirroring ``_obs_state``: replace ``obs["state"]`` for a
    dict obs (passing every other key through unchanged), or return the array itself."""
    return {**obs, "state": new_state} if isinstance(obs, dict) else new_state


def rollout_one(
    env,
    policy_fn,
    model,
    rollout_scalars,
    key,
    horizon,
    summarize,
    collect_states=False,
    collect_obs_act=False,
    collect_energy=False,
):
    """One-env rollout -- THE shared rollout body.

    The evaluator vmaps this (collect_states=False) over N envs; the renderer calls
    it directly (collect_states=True), which ALSO returns the per-step States for
    env.render. collect_states is a Python bool resolved at trace time, so the eval
    path never materializes states (no throughput regression).

    ``summarize`` (T7) is the ACTIVE target's per-step summary hook, ``summarize(state)
    -> dict``, already bound to the target's consts by the caller (evaluator / render /
    backend build it from ``target.objective.summarize`` + ``consts``). rollout_one calls
    it once per step instead of hardcoding the legged "stay-upright" summary, so a
    manipulation target produces object/gripper/joint signals over the SAME step loop.
    The legged summarize returns the identical 5-key dict this function used to build
    inline, so the legged path stays byte-identical.

    The obs-space adversaries operate on the observation VECTOR generically: a dict obs
    perturbs ``obs["state"]`` (legged, byte-identical); a plain-array obs perturbs the
    whole array (manipulation). See ``_obs_state`` / ``_obs_with_state``.

    rollout_scalars are read by meaning; every dim degrades to a no-op (0.0) when a
    space does not declare it (``.get(name, 0.0)``), so a target without a given
    adversary is unaffected. Recognized ROLLOUT dims (all pure-jax + vmappable):
      * obs_noise_scale   -- per-step Gaussian noise on obs["state"].
      * latency_steps     -- action ring-buffer delay (0..LMAX).
      * obs_dropout_prob  -- per-step Bernoulli channel dropout on obs["state"]
                             (each channel zeroed with prob p; obs-noise key stream).
      * obs_bias_scale    -- per-EPISODE constant bias added to obs["state"] (sampled
                             once at reset from a key child; models a miscalibrated sensor).
      * push_mag          -- magnitude (m/s) of a gait-timed velocity impulse on the base.
      * push_step_frac    -- WHEN the push lands, as a fraction of the horizon [0,1].

    Step semantics (identical in both paths -- the divergence #6 fixes):
      key, nk_noise, nk_policy = split(key, 3)
      state = (obs["state"] + obs_noise_scale*normal(nk_noise) + obs_bias) * dropout_mask
      act = policy_fn({**obs, "state": state}, nk_policy)
      latency ring buffer: newest at index 0, delayed = buf[lat]
      at step==round(push_step_frac*(H-1)): base qvel[:3] += push_mag * dir (key-sampled)

    Stateful policies (observation history, LSTM/GRU; adapters/stateful.py) expose
    ``init_carry(obs)`` and are called ``policy_fn(obs, key, carry) -> (act, carry)``.
    The carry is built from the (perturbed) reset observation and lives in the scan
    state, so it is per-env under vmap and replays bit-exactly. Whether a policy is
    stateful is a trace-time Python check: a stateless policy compiles to exactly the
    previous program.

    Nominal-safe: with every new adversary at its nominal 0 the rollout is
    byte-identical to the pre-adversary path (dropout mask is all-ones, bias/push are 0),
    so eval_nominal and existing behavior do not shift.

    Returns
    -------
    collect_states=False : summary dict (per-step, scan-stacked).
    collect_states=True  : (summary dict, list[State] of length horizon+1) where
                           states[0] is the reset state and states[1:] are the
                           post-step states. This list shape matches the prior
                           render.py behavior (reset state prepended); it is NOT
                           eval_nominal.py's list (which keeps post-step states only).
    """
    e = env
    e._mjx_model = model        # binds this env's model under vmap (the key trick)
    st0 = e.reset(key)

    # Read rollout dims by meaning so a target/space without one degrades gracefully.
    obs_noise_scale = rollout_scalars.get("obs_noise_scale", 0.0)
    latency_steps = rollout_scalars.get("latency_steps", 0.0)
    obs_dropout_prob = rollout_scalars.get("obs_dropout_prob", 0.0)
    obs_bias_scale = rollout_scalars.get("obs_bias_scale", 0.0)
    push_mag = rollout_scalars.get("push_mag", 0.0)
    push_step_frac = rollout_scalars.get("push_step_frac", 0.0)

    lat = jnp.clip(
        jnp.round(latency_steps).astype(jnp.int32), 0, LMAX
    )
    A = env.action_size
    buf0 = jnp.zeros((LMAX + 1, A))

    # Per-EPISODE obs bias: a constant vector added to obs["state"] every step, sampled
    # ONCE at reset from a child of `key` (fold_in), so obs_bias_scale=0 is a byte-exact
    # no-op and eval + render (which reconstruct the same `key`) stay bit-identical.
    state_shape = _obs_state(st0.obs).shape
    obs_bias = obs_bias_scale * jax.random.normal(jax.random.fold_in(key, 1), state_shape)

    # Gait-timed PUSH: a horizontal velocity impulse added to the base linear qvel at one
    # chosen step. Direction is sampled once at reset (key-derived, xy-plane); magnitude
    # push_mag; the landing step = round(push_step_frac*(H-1)). push_mag=0 => no-op.
    push_angle = jax.random.uniform(
        jax.random.fold_in(key, 2), (), minval=0.0, maxval=2.0 * jnp.pi
    )
    push_vec = push_mag * jnp.array([jnp.cos(push_angle), jnp.sin(push_angle), 0.0])
    push_step = jnp.clip(
        jnp.round(push_step_frac * (horizon - 1)).astype(jnp.int32), 0, horizon - 1
    )

    stateful = callable(getattr(policy_fn, "init_carry", None))
    # The carry starts from the clean reset obs plus the episode's sensor bias (no
    # per-step noise or dropout is drawn for it: those keys belong to step 0).
    pcarry0 = (policy_fn.init_carry(_obs_with_state(st0.obs, _obs_state(st0.obs) + obs_bias))
               if stateful else None)

    def body(carry, step_idx):
        st, buf, key, pcarry = carry
        key, nk_noise, nk_policy = jax.random.split(key, 3)
        # Observation adversaries all draw from the obs-noise key stream (nk_noise) so
        # they stay deterministic + vmappable. Gaussian noise uses nk_noise directly (so
        # obs_noise-only behavior is unchanged vs. before); dropout uses a folded child.
        state = _obs_state(st.obs)
        gauss = obs_noise_scale * jax.random.normal(nk_noise, state.shape)
        keep = (
            jax.random.uniform(jax.random.fold_in(nk_noise, 1), state.shape)
            >= obs_dropout_prob
        ).astype(state.dtype)
        # Dropout zeros channels LAST, so a dropped channel reads exactly 0 (not noise).
        obs = _obs_with_state(st.obs, (state + gauss + obs_bias) * keep)
        if stateful:
            act, pcarry = policy_fn(obs, nk_policy, pcarry)
        else:
            act, _ = policy_fn(obs, nk_policy)
        buf = jnp.concatenate([act[None], buf[:-1]], axis=0)  # newest at index 0
        delayed = buf[lat]                                     # lat=0 => no delay
        # Apply the gait-timed push to the base linear velocity at the chosen step, BEFORE
        # stepping, so the impulse is integrated this step. do_push is 0/1 (no branching).
        do_push = (step_idx == push_step).astype(st.data.qvel.dtype)
        v_before = st.data.qvel[:3]
        st = st.replace(data=st.data.replace(qvel=st.data.qvel.at[:3].add(do_push * push_vec)))
        # Energy the perturbation INJECTED this step: the kinetic-energy jump the
        # impulse caused (0 on every non-push step). Base mass is folded in by the
        # caller, so this is per-unit-mass; the ratio in autolytic.py is scale-free
        # only if W_self uses the same convention, hence the plain 0.5*dv term here.
        v_after = st.data.qvel[:3]
        # Mass-weighted so E_in is in JOULES, matching qfrc_actuator.qvel below. Using
        # 0.5*dv^2 alone divides per-unit-mass energy into real work and inflates every
        # gain by the robot's mass (~12.7 kg for a Go1). body_subtreemass[1] is the whole
        # robot and tracks the payload dim, so a heavier world prices its push correctly.
        base_mass = model.body_subtreemass[1]
        input_power = 0.5 * base_mass * (jnp.sum(v_after ** 2) - jnp.sum(v_before ** 2))
        st = e.step(st, delayed)
        # Per-target summary hook (T7): the legged summarize returns the identical 5-key
        # dict this used to build inline; a manipulation target returns object/gripper/
        # joint signals over the SAME step loop.
        summary = summarize(st)
        if collect_energy:
            # Autolytic accounting (see breakpoint_mjx/autolytic.py): the mechanical
            # work the host's OWN actuators spend this step, and the energy the
            # perturbation injected. Comparing self_work against a same-seed twin
            # world says whether the policy destroyed itself or was simply shoved.
            self_work = jnp.sum(st.data.qfrc_actuator * st.data.qvel)
            summary = {**summary, "self_work": self_work, "input_power": input_power}
        if collect_obs_act:
            # Demonstration collection (portfolio distillation): the obs vector the policy
            # actually SAW (post noise/dropout/bias) and its RAW output action (pre-latency
            # buffer) -- the (input, label) pair a behavior-cloned student must reproduce.
            # Trace-time bool like collect_states, so eval throughput is untouched.
            summary = {**summary, "obs_state": _obs_state(obs), "action": act}
        out = (summary, st) if collect_states else summary
        return (st, buf, key, pcarry), out

    # xs = arange(H) so `body` gets the step index (drives the gait-timed push); this is
    # also the scan length. With push_mag=0 the step index is unused, so nominal is a no-op.
    (_, _, _, _), scanned = jax.lax.scan(body, (st0, buf0, key, pcarry0),
                                         jnp.arange(horizon))

    if collect_states:
        traj, stacked = scanned
        # Unstack the scan-stacked States (leading dim = horizon) into a list,
        # prepended with the reset state, for env.render(states[::render_every]).
        states = [st0] + [
            jax.tree_util.tree_map(lambda x, i=i: x[i], stacked)
            for i in range(horizon)
        ]
        return traj, states
    return scanned
