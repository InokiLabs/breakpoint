"""Observation adversaries as pure array code (numpy or jax.numpy), no jax import.

``rollout.rollout_one`` (MJX, vmapped), the C MuJoCo re-sim and the Genesis re-sim all
call ``perturb_obs`` so the three engines apply the same rule to the same channels.
"""
from __future__ import annotations

import numpy as np

# Observation adversary semantics, recorded in rating/gate reports. v2: noise, bias and
# dropout touch only SENSOR channels (a target declares its command / last-action / phase
# channels via ``set_obs_non_sensor``), and a dropped channel HOLDS its last delivered
# reading instead of reading 0. v1 perturbed every channel and zeroed dropped ones.
OBS_PERTURBATION_MODEL = "sensor-hold-v2"


def set_obs_non_sensor(env, slices, frame=None):
    """Declare which ``obs["state"]`` channels are NOT sensor readings.

    ``slices`` is a sequence of ``(start, length)``: the velocity command, the policy's own
    previous action, a gait-phase clock. A negative ``start`` counts from the end of the
    vector (or of each frame). These are inputs the robot computes, not readings
    a sensor can corrupt, so the obs adversaries leave them alone. ``frame`` is the period
    of a stacked observation history (e.g. Barkour's 15 x 31): the slices then repeat in
    every frame. A target calls this from its ``load_env``; an env without a declaration
    keeps every channel perturbable.
    """
    env._bp_obs_non_sensor = (tuple((int(a), int(n)) for a, n in slices),
                              None if frame is None else int(frame))
    return env


def obs_sensor_mask(env, n):
    """Bool numpy mask of length ``n``: True where the obs adversaries may act, or None
    (every channel) when the env declared nothing. Built at trace time."""
    spec = getattr(env, "_bp_obs_non_sensor", None)
    if not spec:
        return None
    slices, frame = spec
    period = frame or n
    mask = np.ones(n, dtype=bool)
    for base in range(0, n, period):
        for start, length in slices:
            a = base + (start if start >= 0 else period + start)
            mask[a: a + length] = False
    return mask


def perturb_obs(xp, clean, gauss, bias, dropped, held, sensor_mask):
    """One step of the observation adversaries. Pure; ``xp`` is jax.numpy or numpy.

    A reading is ``clean + gauss + bias``; a dropped channel instead repeats the last
    reading the policy was given (``held``, a sample-and-hold bus); non-sensor channels
    pass through clean. Returns ``(delivered, new_held)``, which are the same array.
    With every adversary at 0 and nothing dropped this returns ``clean`` exactly.
    """
    read = xp.where(dropped, held, clean + gauss + bias)
    if sensor_mask is not None:
        read = xp.where(sensor_mask, read, clean)
    return read, read
