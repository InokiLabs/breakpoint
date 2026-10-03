"""Workstream 1c (T2): safety margins + per-axis attribution.

Two questions per failure family, answered from the space NOMINAL:

  * **Boundary / safety margin** -- the MINIMAL perturbation from nominal that triggers the
    failure, per axis. For each axis we line-search from nominal toward the family
    representative's theta (holding every other axis at nominal) and report the value where
    ``failure_score`` first crosses ``cfg.failure_threshold``. The human string ANDs the
    axes that actually gate the failure, e.g.::

        "stable until friction_floor < 0.35 AND latency_steps > 2.5"

  * **Attribution** -- which axes DRIVE the family, via local finite-difference sensitivity
    around the representative (perturb each axis +/- eps, measure the failure_score change),
    normalized to a per-axis importance that sums to 1.

CRITICAL testability (spec 1c): the boundary line-search + crossing/formatting and the
finite-difference normalization are PURE and take an injectable ``score_fn(thetas) ->
scores`` (numpy in, numpy out). They are unit-tested on CPU with a SYNTHETIC analytic score
function (no GPU). The evaluator-backed ``score_fn`` (``make_evaluator_score_fn``) is the
thin GPU wrapper, Colab-validated; it uses a FIXED key so scoring is deterministic, and it
batches all of a family's candidates into as few ``evaluate`` calls as possible (one per
boundary sweep, one per attribution sweep). All candidates are clipped to the space bounds,
so no out-of-bounds theta is ever proposed.
"""
from __future__ import annotations

import numpy as np


def _clip(vec, lower, upper):
    return np.minimum(np.maximum(vec, lower), upper)


def axis_line_candidates(nominal, target, axis, steps, lower=None, upper=None):
    """``steps`` thetas holding all axes at ``nominal`` and sweeping ``axis`` from nominal
    (frac 0) to ``target`` (frac 1). Clipped to ``[lower, upper]`` when given (nominal and
    target are already in-bounds, so this is a belt-and-braces guard).

    Returns ``(candidates (steps, D), fracs (steps,))``.
    """
    nominal = np.asarray(nominal, dtype=float)
    target = np.asarray(target, dtype=float)
    fracs = np.linspace(0.0, 1.0, int(steps))
    cands = np.tile(nominal, (len(fracs), 1))
    cands[:, axis] = nominal[axis] + fracs * (target[axis] - nominal[axis])
    if lower is not None and upper is not None:
        cands = _clip(cands, np.asarray(lower, float), np.asarray(upper, float))
    return cands, fracs


def first_crossing_index(scores, threshold):
    """Index of the first score ``>= threshold`` (the ``>=`` rule the search uses), or None."""
    scores = np.asarray(scores, dtype=float)
    hits = np.nonzero(scores >= float(threshold))[0]
    return int(hits[0]) if hits.size else None


def compute_boundary(nominal, target, lower, upper, names, threshold, score_fn, steps=11):
    """Per-axis boundary line-search from ``nominal`` toward ``target`` + a margin string.

    Batches ALL axes' candidates into a SINGLE ``score_fn`` call, evaluating the shared
    nominal vector exactly ONCE (the frac-0 row of every axis is the same nominal, so it is
    deduped rather than scored D times). For each axis that both moves (target != nominal)
    and crosses ``threshold`` at a non-nominal point, the boundary is the LAST-SAFE value
    (the last candidate still below threshold, ``cands[idx-1]``), so "stable until X < b"
    quotes a value that was actually still stable rather than the first failing one.

    Returns ``(margin: str, boundaries: dict)`` where ``boundaries[name]`` is
    ``{gates, boundary, direction, op, nominal, target}`` (JSON-safe).
    """
    nominal = np.asarray(nominal, float)
    target = np.asarray(target, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    threshold = float(threshold)
    D = len(names)
    nominal_clipped = _clip(nominal, lower, upper)

    # Per-axis PERTURBATION candidates only (drop the shared frac-0 nominal row); the nominal
    # is scored once and reused across axes.
    blocks = []
    for a in range(D):
        cands, fracs = axis_line_candidates(nominal, target, a, steps, lower, upper)
        blocks.append((a, cands[1:], fracs[1:]))          # perturbations (frac > 0)

    stacked = np.concatenate(
        [nominal_clipped[None, :]] + [c for _, c, _ in blocks], axis=0
    )
    scores = np.asarray(score_fn(stacked), dtype=float)
    nominal_score = float(scores[0])
    fails_at_nominal = nominal_score >= threshold

    boundaries = {}
    clauses = []
    off = 1                                                # index 0 is the shared nominal
    for a, cands, fracs in blocks:
        seg = scores[off:off + len(fracs)]
        off += len(fracs)
        name = names[a]
        moves = float(target[a]) != float(nominal[a])
        # Full per-axis view: nominal (index 0) followed by the perturbations, so crossing
        # indices match the original nominal->target sweep semantics.
        full_scores = np.concatenate([[nominal_score], seg])
        full_axis_vals = np.concatenate([[float(nominal_clipped[a])], np.asarray(cands)[:, a]])
        idx = first_crossing_index(full_scores, threshold)
        # gates iff moving THIS axis alone (a non-nominal perturbation, idx > 0) trips the
        # failure. idx == 0 means it already fails at nominal (handled separately).
        gates = bool(moves and idx is not None and idx > 0)
        info = {
            "gates": gates,
            "boundary": None,
            "direction": None,
            "op": None,
            "nominal": float(nominal[a]),
            "target": float(target[a]),
        }
        if gates:
            bval = float(full_axis_vals[idx - 1])         # LAST-SAFE value (still stable)
            direction = "high" if target[a] > nominal[a] else "low"
            op = ">" if direction == "high" else "<"
            info.update(boundary=bval, direction=direction, op=op)
            clauses.append(f"{name} {op} {bval:.3g}")
        boundaries[name] = info

    if fails_at_nominal:
        margin = "fails at nominal (no perturbation needed)"
    elif clauses:
        margin = "stable until " + " AND ".join(clauses)
    else:
        margin = "no single-axis boundary within bounds"
    return margin, boundaries


def compute_attribution(rep, lower, upper, names, score_fn, eps_frac=0.05):
    """Per-axis local sensitivity around ``rep`` via central finite-difference, normalized.

    For each axis, perturbs by ``+/- eps`` where ``eps = eps_frac * (upper - lower)``, clips
    to bounds, and estimates ``|d score / d axis|`` as ``|s(+) - s(-)| / (x(+) - x(-))``.
    Importances are the sensitivities normalized to sum to 1 (all-zero -> all-zero). Batches
    the ``2 * D`` candidates into a SINGLE ``score_fn`` call. Returns ``{name: importance}``.
    """
    rep = np.asarray(rep, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    D = len(names)
    eps = float(eps_frac) * (upper - lower)

    plus = np.tile(rep, (D, 1))
    minus = np.tile(rep, (D, 1))
    for a in range(D):
        plus[a, a] = min(rep[a] + eps[a], upper[a])
        minus[a, a] = max(rep[a] - eps[a], lower[a])

    stacked = np.concatenate([plus, minus], axis=0)          # (2D, D)
    scores = np.asarray(score_fn(stacked), dtype=float)
    sp, sm = scores[:D], scores[D:]

    sens = np.zeros(D)
    for a in range(D):
        delta = float(plus[a, a] - minus[a, a])
        sens[a] = abs(float(sp[a] - sm[a])) / delta if delta > 0.0 else 0.0

    total = float(sens.sum())
    imp = sens / total if total > 0.0 else np.zeros(D)
    return {names[a]: float(imp[a]) for a in range(D)}


# --------------------------------------------------------------------------- #
# GPU wrapper + per-family orchestration (Colab-run; jax deferred to call time)
# --------------------------------------------------------------------------- #
def make_evaluator_score_fn(evaluate, space, thresholds, horizon, key, objective=None):
    """Wrap the GPU ``evaluate`` as ``score_fn(thetas) -> scores`` using a FIXED ``key``.

    Deterministic: the same candidate batch always maps to the same scores (the batch key is
    fixed; the evaluator folds a per-row key from it, exactly as the search does). Not
    exercised by the CPU tests (jax import is deferred to call time); Colab-validated. T7:
    ``objective`` (the ACTIVE target's Objective, legged by default) drives scoring so the
    boundary/attribution are computed against the target's own failure_score.
    """
    from breakpoint_mjx.search import scores_from_trajectories

    def score_fn(thetas):
        import jax.numpy as jnp

        traj = evaluate(jnp.asarray(np.asarray(thetas, dtype=np.float32)), key)
        scores, _, _ = scores_from_trajectories(traj, thresholds, horizon, objective)
        return np.asarray(scores, dtype=float)

    return score_fn


def analyze_family(rep_theta, space, cfg, score_fn, steps=11, eps_frac=0.05):
    """Boundary + attribution for one family, given its representative's theta vector.

    Returns ``(margin, boundaries, attribution)``. Pure given ``score_fn``.
    """
    nominal = np.asarray(space.nominal_vector(), float)
    target = np.asarray(rep_theta, float)
    lower = np.asarray(space.lower(), float)
    upper = np.asarray(space.upper(), float)
    names = space.names()
    margin, boundaries = compute_boundary(
        nominal, target, lower, upper, names, float(cfg.failure_threshold), score_fn, steps
    )
    attribution = compute_attribution(target, lower, upper, names, score_fn, eps_frac)
    return margin, boundaries, attribution


def compute_margins(families, records_by_id, space, cfg, evaluate, key,
                    steps=11, eps_frac=0.05, objective=None):
    """BP_MARGINS stage: annotate each family with margin + boundaries + attribution.

    Scores candidate thetas through the GPU ``evaluate`` (fixed ``key``); uses
    ``cfg.thresholds`` (already per-target merged) + ``cfg.failure_threshold`` + the space
    bounds. T7: ``objective`` (legged by default) drives the scoring. Mutates the
    ``FailureFamily`` objects in place and returns them.
    """
    score_fn = make_evaluator_score_fn(evaluate, space, cfg.thresholds, cfg.horizon, key,
                                       objective)
    for f in families:
        rep = records_by_id.get(f.representative_id)
        if rep is None:
            continue
        margin, boundaries, attribution = analyze_family(
            rep.theta_vector, space, cfg, score_fn, steps, eps_frac
        )
        f.margin = margin
        f.boundaries = boundaries
        f.attribution = attribution
    return families
