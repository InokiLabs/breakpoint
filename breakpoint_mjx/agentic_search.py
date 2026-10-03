"""Agentic failure-world search: MULTIPLE role-specialized LLM proposers with a
yield-driven budget coordinator (the multi-agent extension of llm_search).

Each round, every role-agent proposes failure worlds from its own attack stance
(contact physics, sensing, timing, morphology, cross-axis combining); the whole
proposal set is evaluated in ONE GPU batch; the coordinator then reallocates the
next round's simulation budget toward the roles whose slices actually yielded
transfer-weighted failures (EMA + a floor so no role starves and exploration
never dies). Role memory is split: every agent sees the global leaderboard, but
only its OWN last-round results, which keeps the stances from collapsing into
one voice.

Records carry ids like ``agentic-r2-timing-17`` so benchmark_search can split
per-role yield out of a single run. Pure coordination logic (slice counts,
reallocation) is dependency-free and unit-tested on CPU; the GPU evaluate and
LLM calls are injected, mirroring llm_search.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.llm_search import (
    _build_batch, _complete, _parse_configs, _space_desc, gated_proposals,
)
from breakpoint_mjx.search import _records, scores_from_trajectories

# Role stances: name -> (attack brief, dims to emphasize). Dims are advisory (the
# prompt steers, the parser still allows every dim) so a role can discover that its
# axis only bites in combination with another.
ROLES = {
    "contact": (
        "You attack CONTACT AND TERRAIN physics: friction, floor slope, joint friction "
        "loss. Find slips, foot-catch, and slope-induced falls.",
        ("friction_floor", "floor_slope_rad", "joint_frictionloss_scale"),
    ),
    "sensing": (
        "You attack PERCEPTION: observation noise, per-channel dropout, constant sensor "
        "bias. Find where the policy's state estimate breaks.",
        ("obs_noise_scale", "obs_dropout_prob", "obs_bias_scale"),
    ),
    "timing": (
        "You attack TIMING AND IMPULSES: actuation latency and gait-timed pushes. Find "
        "the phase-sensitive windows where a small delay or shove becomes a fall.",
        ("latency_steps", "push_mag", "push_step_frac"),
    ),
    "morphology": (
        "You attack the ROBOT'S BODY: payload mass, center-of-mass offsets, actuator "
        "gain and armature scaling. Find loadouts the controller cannot carry.",
        ("torso_mass_add_kg", "com_offset_x_m", "com_offset_y_m", "com_offset_z_m",
         "kp_scale", "kd_scale", "joint_armature_scale"),
    ),
    "combiner": (
        "You COMBINE what is already working: read the leaderboard and compose "
        "cross-axis worlds from the strongest finds of the other agents, staying inside "
        "realistic bands so transfer stays high.",
        (),
    ),
}

_FLOOR = 0.10  # minimum budget fraction per role (exploration guarantee)
_EMA = 0.5     # how fast the coordinator chases yield (0 = never, 1 = instant)


def slice_counts(alloc, n):
    """Integer per-role slice sizes summing exactly to n (largest-remainder)."""
    alloc = np.asarray(alloc, dtype=float)
    alloc = alloc / alloc.sum()
    raw = alloc * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(raw - counts)[::-1][: n - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def reallocate(alloc, yields, floor=_FLOOR, ema=_EMA):
    """Next-round budget fractions: EMA toward observed per-role yield, floored.

    yields are per-role transfer-weighted scores (any nonnegative scale). A round
    where every role yields zero keeps the current allocation (no signal, no move).
    """
    alloc = np.asarray(alloc, dtype=float)
    y = np.asarray(yields, dtype=float)
    if y.sum() <= 0:
        return (alloc / alloc.sum()).tolist()
    target = y / y.sum()
    nxt = (1 - ema) * (alloc / alloc.sum()) + ema * target
    # Floor projection: give every role the floor outright, split the remainder
    # proportionally. Guarantees min >= floor AND sum == 1 (a plain clamp+renorm
    # would scale floored roles back below the floor).
    return (floor + (1.0 - floor * len(nxt)) * (nxt / nxt.sum())).tolist()


def role_yield(scores, transfers, top=5):
    """A slice's yield: mean of its top-`top` transfer-weighted failure scores.

    Dense enough to rank roles at 2048-batch scale, and weighting by transfer keeps
    the coordinator from funding roles that only find unsurvivable absurd worlds.
    """
    w = np.asarray(scores, dtype=float) * np.asarray(transfers, dtype=float)
    if w.size == 0:
        return 0.0
    return float(np.sort(w)[::-1][: min(top, w.size)].mean())


def _role_prompt(space, stance, dims, leaderboard, own_last, n_proposals, robot):
    lb = "\n".join(
        f"  fs={r['fs']:.2f} tr={r['tr']:.2f} :: "
        + ", ".join(f"{k}={v:.3g}" for k, v in r["theta"].items())
        for r in leaderboard
    ) or "  (none yet)"
    own = "\n".join(
        f"  fs={r['fs']:.2f} tr={r['tr']:.2f} :: "
        + ", ".join(f"{k}={v:.3g}" for k, v in r["theta"].items())
        for r in own_last
    ) or "  (first round)"
    focus = f"Your primary dials: {', '.join(dims)}.\n" if dims else ""
    names = list(space.dims.keys())
    return (
        f"You are one specialist on a team of adversarial agents searching for worlds "
        f"that break a {robot or 'quadruped'} locomotion policy (trained with domain "
        f"randomization; robust to mild single-axis perturbations, breaks under "
        f"realistic combinations). {stance}\n{focus}\n"
        f"Parameter space:\n{_space_desc(space)}\n\n"
        f"TEAM leaderboard (all agents):\n{lb}\n\nYOUR last round:\n{own}\n\n"
        f"Propose {n_proposals} NEW, DIVERSE configurations from your stance. Maximize "
        f"failure_score while keeping every value inside its realistic band (transfer "
        f"matters). Output ONLY a JSON array of {n_proposals} objects, each mapping "
        f"every key in {names} to a number. No prose."
    )


def agentic_search(evaluate, space, cfg, key, rounds=5, n_proposals=6, robot_name=None,
                   objective=None, roles=None, complete=_complete, gate_opts=None):
    """Closed-loop multi-agent search. Returns list[FailureRecord] across all rounds.

    ``complete`` is injectable for tests (same contract as llm_search._complete).

    ``gate_opts`` is forwarded to ``plausibility.gate``, which every role's proposals pass
    through before ``_build_batch``. Gating PER ROLE matters here beyond saving GPU time:
    the coordinator reallocates budget by measured yield, so a role that only ever finds
    unsurvivable-corner worlds would otherwise be rewarded for producing failures nobody
    can reproduce on hardware. A role's slice size is unaffected by what its gate rejects
    -- ``_build_batch`` pads with random rows -- so gating cannot starve a role either."""
    import jax
    import jax.numpy as jnp

    roles = roles or ROLES
    names = list(roles.keys())
    N = cfg.popsize
    rng = np.random.default_rng(int(cfg.seed))
    alloc = [1.0 / len(names)] * len(names)
    leaderboard, own_last, all_records = [], {n: [] for n in names}, []

    for r in range(rounds):
        counts = slice_counts(alloc, N)
        batch_parts, spans, at = [], {}, 0
        for role, cnt in zip(names, counts):
            stance, dims = roles[role]
            text = complete(_role_prompt(space, stance, dims, leaderboard,
                                         own_last[role], n_proposals, robot_name))
            cfgs = _parse_configs(text, space)
            if text and not cfgs:
                print(f"  ({role}: proposal parse failed; slice falls back to random)")
            if cfgs:
                cfgs = gated_proposals(space, cfgs, horizon=cfg.horizon,
                                       gate_opts=gate_opts, label=f"gate[{role}]")
            batch_parts.append(_build_batch(space, cfgs, cnt, rng))
            spans[role] = (at, at + cnt)
            at += cnt
        batch = np.concatenate(batch_parts, axis=0)

        key, ek = jax.random.split(key)
        traj = evaluate(jnp.asarray(batch), ek)
        scores, sigs, comps = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon,
                                                       objective)
        recs = _records(space, batch, scores, sigs, comps, cfg.seed, f"agentic-r{r}",
                        len(all_records), ek)

        yields = []
        for role in names:
            lo, hi = spans[role]
            role_recs = recs[lo:hi]
            for j, rec in enumerate(role_recs):
                rec.id = f"agentic-r{r}-{role}-{lo + j}"
            tr = [rec.transfer_score or 0.0 for rec in role_recs]
            fs = [rec.failure_score for rec in role_recs]
            yields.append(role_yield(fs, tr))
            own_last[role] = sorted(
                ({"fs": rec.failure_score, "tr": rec.transfer_score or 0.0,
                  "theta": {k: round(v, 3) for k, v in rec.theta_named.items()}}
                 for rec in role_recs),
                key=lambda x: x["fs"] * x["tr"], reverse=True)[:3]

        all_records += recs
        for rec in recs:
            leaderboard.append({"fs": rec.failure_score, "tr": rec.transfer_score or 0.0,
                                "theta": {k: round(v, 3) for k, v in rec.theta_named.items()}})
        leaderboard.sort(key=lambda x: x["fs"] * x["tr"], reverse=True)
        leaderboard = leaderboard[:12]

        alloc = reallocate(alloc, yields)
        alloc_str = ", ".join(f"{n}={a:.2f}" for n, a in zip(names, alloc))
        print(f"agentic round {r}: max_score={float(scores.max()):.3f} "
              f"next_alloc[{alloc_str}]", flush=True)

    return all_records
