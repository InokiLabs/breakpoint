"""Upgrade #4: a failure ATLAS via hand-rolled MAP-Elites.

CMA-ES / LLM search return the *worst* points; they do not map the *variety* of
ways a policy fails. MAP-Elites keeps the single best ("elite") solution per cell
of a 2D behavior grid, so the output is a MAP of distinct failure modes rather
than a pile of near-duplicate worst cases.

Grid (from ``failure.behavior_descriptor``):
  * axis 1 -- dominant failure mode (categorical, ``failure.FAILURE_MODES``)
  * axis 2 -- time-to-failure, normalized to [0,1] then binned into ``n_ttf_bins``.

Fitness = transfer-weighted failure score (``failure_score * transfer_likelihood``),
the same objective the rest of the engine maximizes, so each cell holds the most
*realistic* failure of that mode/timing.

Hand-rolled (~150 lines) over pulling in qdax: the archive is a small dict and the
loop is short; no new dependency. The archive insertion + binning is PURE numpy
(``_ttf_bin`` / ``_fitness`` / ``_insert``) and unit-tested; only the search loop
touches JAX/GPU (validated on Colab)."""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.objectives import LEGGED_OBJECTIVE
from breakpoint_mjx.search import _records, scores_from_trajectories

# Fallback grid/loop params if the search.yaml `atlas:` block is absent (back-compat).
_DEFAULTS = {"n_ttf_bins": 8, "iterations": 20, "batch_size": 256, "sigma": 0.15}


def _ttf_bin(ttf_norm: float, n_ttf_bins: int) -> int:
    """Map ttf_norm in [0,1] to an integer bin in [0, n_ttf_bins-1] (clamped). Pure."""
    b = int(float(ttf_norm) * (n_ttf_bins - 1))
    return max(0, min(b, n_ttf_bins - 1))


def _fitness(rec) -> float:
    """Transfer-weighted failure score for a FailureRecord (atlas elite criterion). Pure."""
    return float(rec.failure_score) * float(rec.transfer_score or 0.0)


def _insert(archive, descriptors, records, n_ttf_bins):
    """Place each record into archive[(mode_index, ttf_bin)], keeping the higher-fitness
    record per cell (best-per-cell). Mutates and returns ``archive``. Pure numpy."""
    for (mode_index, ttf_norm), rec in zip(descriptors, records):
        key = (int(mode_index), _ttf_bin(ttf_norm, n_ttf_bins))
        cur = archive.get(key)
        if cur is None or _fitness(rec) > _fitness(cur):
            archive[key] = rec
    return archive


def _max_weighted(archive) -> float:
    return max((_fitness(r) for r in archive.values()), default=0.0)


def _descriptors_for_batch(traj, thresholds, horizon, comps, objective=None):
    """Per-env (mode_index, ttf_norm) descriptors for a batched trajectory dict. Pure.

    ``comps`` is the per-env ``component_signals`` list already computed by
    ``scores_from_trajectories``; passing it in lets ``behavior_descriptor`` reuse it
    instead of recomputing ``component_signals`` (one pass per env, not two). T7: routes
    through the ACTIVE target's ``objective.behavior_descriptor`` (legged by default). N is
    read from any key so a manipulation traj works too."""
    obj = objective or LEGGED_OBJECTIVE
    n = int(np.asarray(next(iter(traj.values()))).shape[0])
    out = []
    for i in range(n):
        one = {k: np.asarray(v[i]) for k, v in traj.items()}
        out.append(obj.behavior_descriptor(one, thresholds, horizon, sig=comps[i]))
    return out


def map_elites_search(evaluate, space, cfg, key, objective=None):
    """Hand-rolled MAP-Elites over the 2D failure-behavior grid, GPU-batched via evaluate.

    Returns the archive's elites as a ``list[FailureRecord]`` (tag ``"atlas"``), each
    carrying ``batch_key`` + ``env_index`` (via ``search._records``) so the renderer can
    reproduce it bit-for-bit, exactly like the other searches. Loop params come from the
    ``atlas:`` block of ``cfg`` (``getattr(cfg, 'atlas', {})``) with sensible fallbacks.
    """
    import jax
    import jax.numpy as jnp
    from breakpoint_mjx.thetas import sample_uniform

    obj = objective or LEGGED_OBJECTIVE
    failure_modes = obj.failure_modes
    ac = dict(_DEFAULTS, **(getattr(cfg, "atlas", None) or {}))
    n_ttf_bins = int(ac["n_ttf_bins"])
    iterations = int(ac["iterations"])
    batch_size = max(1, int(ac["batch_size"]))  # >=1 so the seed batch is non-empty
    sigma = float(ac["sigma"])  # perturbation std as a fraction of each dim's span

    lo = np.array(space.lower()); hi = np.array(space.upper()); span = hi - lo
    rng = np.random.default_rng(int(cfg.seed))  # numpy stream for perturbation (mirrors llm_search)

    archive: dict = {}
    placed = 0  # running start_id so atlas record ids stay unique across batches

    def _eval_and_place(batch, ek):
        """Evaluate a (B,D) numpy batch on GPU, score + describe, insert best-per-cell."""
        nonlocal placed
        traj = evaluate(jnp.asarray(batch), ek)
        scores, sigs, comps = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, obj)
        descriptors = _descriptors_for_batch(traj, cfg.thresholds, cfg.horizon, comps, obj)
        recs = _records(space, batch, scores, sigs, comps, cfg.seed, "atlas", placed, ek)
        _insert(archive, descriptors, recs, n_ttf_bins)
        placed += len(recs)
        return scores

    # Seed: one uniform batch.
    key, sk, ek = jax.random.split(key, 3)
    seed_batch = np.asarray(sample_uniform(space, sk, batch_size))
    scores = _eval_and_place(seed_batch, ek)
    print(f"atlas seed: batch={batch_size} occupied={len(archive)}/{n_ttf_bins * len(failure_modes)} "
          f"max_score={float(scores.max()):.3f} max_weighted={_max_weighted(archive):.3f}",
          flush=True)

    # Iterate: sample parents from occupied cells, perturb in bound-normalized space.
    # After a non-empty seed batch every row lands in some cell, so the archive is
    # always occupied here (batch_size >= 1) -- there is no empty-archive case.
    for it in range(iterations):
        elites = list(archive.values())
        parents = rng.integers(0, len(elites), size=batch_size)
        # float32 to match sample_uniform's dtype, so evaluate isn't recompiled under x64.
        batch = np.empty((batch_size, len(lo)), dtype=np.float32)
        for j, pi in enumerate(parents):
            base = np.asarray(elites[pi].theta_vector, dtype=float)
            # Gaussian perturbation in normalized space (sigma * span), clip to bounds.
            batch[j] = np.clip(base + rng.normal(size=len(lo)) * sigma * span, lo, hi)
        key, ek = jax.random.split(key)
        scores = _eval_and_place(batch, ek)
        print(f"atlas iter {it}: occupied={len(archive)}/{n_ttf_bins * len(failure_modes)} "
              f"max_score={float(scores.max()):.3f} max_weighted={_max_weighted(archive):.3f}",
              flush=True)

    return list(archive.values())
