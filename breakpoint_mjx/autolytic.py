"""Autolytic gain: does a perturbation make the policy destroy ITSELF?

The measurement under the "autolytic germline" idea, isolated from the ecology so
it can be validated on its own. For a perturbed ("infected") world and its
same-seed unperturbed TWIN, define

    E_in  = energy the perturbation injected      (integral of applied wrench . velocity)
    G     = W_self(infected) - W_self(twin)       (excess actuator work the host spent,
                                                   W_self = integral of qfrc_actuator . qvel)
    gain  = G / E_in

gain <= 1 means the perturbation did the work itself -- a sledgehammer, uninteresting
and physically unsurprising. gain >> 1 means a small input recruited the controller
into spending large energy fighting itself: a resonance, reflex exploit, or gait
entrainment. That ratio is the quantity no failure-search tool measures today, and
it is what makes a discovered failure a CONTROLLER defect rather than a big shove.

Sign discipline (the refuter's main catch): a policy that collapses early does LESS
actuator work, so integrating to the end of the episode can make G negative for the
worst failures. All windows here therefore end at (or before) the collapse step and
the accumulation is explicitly one-sided; `gain_window` returns both the value and
the window it used so a caller can never silently compare different windows.

Everything here is pure numpy over per-step arrays MJX already computes, so it is
CPU-testable; the GPU side only has to hand back two extra per-step scalars.
"""
from __future__ import annotations

import numpy as np

# Below this, E_in is numerical dust and the ratio is meaningless rather than large.
_EPS_E_IN = 1e-6


def step_work(qfrc_actuator, qvel):
    """Per-step mechanical work of the actuators: sum_j qfrc_actuator_j * qvel_j.

    Accepts (T, nv) arrays (or (nv,) for a single step) and returns (T,) (or scalar).
    """
    f = np.asarray(qfrc_actuator, dtype=float)
    v = np.asarray(qvel, dtype=float)
    if f.shape != v.shape:
        raise ValueError(f"qfrc_actuator {f.shape} and qvel {v.shape} must match")
    return np.sum(f * v, axis=-1)


def step_input_power(xfrc_applied, body_vel):
    """Per-step power injected by the perturbation: sum_b wrench_b . vel_b.

    xfrc_applied/body_vel are (T, nbody, 6) (or (nbody, 6) for one step).
    """
    w = np.asarray(xfrc_applied, dtype=float)
    v = np.asarray(body_vel, dtype=float)
    if w.shape != v.shape:
        raise ValueError(f"xfrc_applied {w.shape} and body_vel {v.shape} must match")
    return np.sum(w * v, axis=(-1, -2))


def collapse_step(fallen_flags):
    """First step index where the host is fallen, or None if it never falls."""
    idx = np.flatnonzero(np.asarray(fallen_flags, dtype=bool))
    return int(idx[0]) if idx.size else None


def gain_window(work_infected, work_twin, input_power, fallen_flags=None,
                window_steps=None, bind_step=0):
    """Autolytic gain over an explicit, collapse-aware window.

    Returns (gain, G, E_in, (lo, hi)) with hi EXCLUSIVE. The window starts at
    ``bind_step`` (when the perturbation first acts) and ends at the collapse step
    if the host falls, else the end of the episode; ``window_steps`` further caps
    it to the last N steps before that end, which is the "1.5 s before collapse"
    the experiment calls for.

    G is accumulated over that window, but E_in is accumulated from ``bind_step`` to
    the window's end: a gait-timed push is a SINGLE impulse that usually lands before
    a late window opens, and integrating it over the narrow window alone reports
    E_in = 0 and throws the row away as unmeasured. The denominator is "energy this
    perturbation has injected so far", which is what the ratio is meant to divide by.

    gain is None when E_in is dust: a ratio with no denominator is not a large
    gain, and returning None keeps it out of a mean.
    """
    wi = np.asarray(work_infected, dtype=float)
    wt = np.asarray(work_twin, dtype=float)
    pin = np.asarray(input_power, dtype=float)
    if not (wi.shape == wt.shape == pin.shape):
        raise ValueError("work_infected, work_twin, input_power must have equal length")

    end = len(wi)
    if fallen_flags is not None:
        c = collapse_step(fallen_flags)
        if c is not None:
            end = c  # exclude the collapse step itself: post-fall work is not autolytic
    lo = int(max(bind_step, 0))
    if window_steps is not None:
        lo = max(lo, end - int(window_steps))
    hi = int(max(end, lo))

    G = float(np.sum(wi[lo:hi] - wt[lo:hi]))
    E_in = float(np.sum(pin[max(bind_step, 0):hi]))
    gain = G / E_in if abs(E_in) > _EPS_E_IN else None
    return gain, G, E_in, (lo, hi)


def classify(gain, sledgehammer_max=1.0, autolytic_min=3.0):
    """Label a world by what its gain means.

    'damped'       (gain < 0): the perturbed host spent LESS actuator energy than its
                   twin. Not a sledgehammer -- the opposite sign of effect, and usually
                   a host that collapsed early and stopped actuating. Kept separate so
                   it can never be read as "the push did the work".
    'sledgehammer' (0 <= gain <= 1): the input did the work; physically unremarkable.
    'amplified'    (1 < gain < autolytic_min): the controller contributed.
    'autolytic'    (gain >= autolytic_min): the controller is the main destroyer --
                   the finding worth putting in a report.
    'unmeasured'   : no usable denominator.
    """
    if gain is None:
        return "unmeasured"
    if gain < 0:
        return "damped"
    if gain <= sledgehammer_max:
        return "sledgehammer"
    if gain < autolytic_min:
        return "amplified"
    return "autolytic"


def rank_worlds(gains, failure_scores, autolytic_min=3.0):
    """Order world indices by autolytic evidence: gain first, failure as tie-break.

    Worlds with no usable gain sink below every measured one instead of being
    dropped, so a caller sees them rather than silently losing rows.
    """
    rows = []
    for i, (g, fs) in enumerate(zip(gains, failure_scores)):
        rows.append((i, -np.inf if g is None else float(g), float(fs)))
    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
    return [r[0] for r in rows]


def twin_thetas(space, thetas):
    """The same worlds with every ROLLOUT perturbation at nominal (the control).

    Model dims are left ALONE on purpose: the twin must be the same robot in the
    same world, differing only by the attack. Zeroing model dims too would compare
    two different robots and the "excess work" would be meaningless.
    """
    names = space.names()
    twin = np.array(thetas, dtype=float, copy=True)
    for k, d in space.dims.items():
        if d.kind == "rollout":
            twin[:, names.index(k)] = d.nominal
    return twin


def measure_records(records, evaluate_energy, space, thresholds, horizon,
                    top_n=32, window_steps=None):
    """Annotate the top-N records in place with autolytic gain, and return a summary.

    ``evaluate_energy(thetas, key) -> traj`` must be an evaluator built with
    ``rollout_one(..., collect_energy=True)`` so the trajectory carries ``self_work``
    and ``input_power``. Each selected world is run TWICE under the SAME key (perturbed
    and twin), which is why this is opt-in: it doubles the rollout cost for those rows.

    Records outside the top-N keep ``autolytic_gain=None`` rather than 0.0, so "not
    measured" can never be misread as "no amplification".
    """
    import jax
    import jax.numpy as jnp

    chosen = sorted(records, key=lambda r: r.failure_score, reverse=True)[:top_n]
    if not chosen:
        return summarize_batch([])

    thetas = np.array([r.theta_vector for r in chosen], dtype=float)
    twins = twin_thetas(space, thetas)
    key = jax.random.PRNGKey(0)
    inf = evaluate_energy(jnp.asarray(thetas), key)
    twin = evaluate_energy(jnp.asarray(twins), key)

    h = np.asarray(inf["trunk_height"])
    up = np.asarray(inf["up_proj"])
    fallen = ((h < thresholds["trunk_height_min"]) | (up < thresholds["up_proj_min"]))
    w_inf = np.asarray(inf["self_work"])
    w_twin = np.asarray(twin["self_work"])
    p_in = np.asarray(inf["input_power"])

    gains = []
    for i, rec in enumerate(chosen):
        g, _, _, _ = gain_window(w_inf[i], w_twin[i], p_in[i],
                                 fallen_flags=fallen[i], window_steps=window_steps)
        rec.autolytic_gain = None if g is None else float(g)
        rec.autolytic_class = classify(g)
        gains.append(g)
    return summarize_batch(gains)


def summarize_batch(gains, autolytic_min=3.0):
    """Batch-level report: what fraction of discovered failures are host-amplified?

    This is the week-1 experiment's headline: if a meaningful fraction of the
    failures the search already finds are autolytic (gain >= 3), the mechanism has
    signal; if essentially all are sledgehammers, it does not and the idea dies.
    """
    labels = [classify(g, autolytic_min=autolytic_min) for g in gains]
    n = len(labels)
    measured = [g for g in gains if g is not None]
    counts = {k: labels.count(k) for k in
              ("autolytic", "amplified", "sledgehammer", "damped", "unmeasured")}
    return {
        "n": n,
        "counts": counts,
        "autolytic_fraction": counts["autolytic"] / n if n else 0.0,
        "measured_fraction": len(measured) / n if n else 0.0,
        "peak_gain": max(measured) if measured else None,
        "median_gain": float(np.median(measured)) if measured else None,
    }
