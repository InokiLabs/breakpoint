"""Noise-robust multi-restart CMA-ES: score a condition on several seeds, not one.

THE PROBLEM. ``multi_cmaes_search`` scores each candidate condition on ONE rollout seed.
Where falls are seed-dependent (near nominal, the failure probability of most conditions
is well below 1), a single lucky fall gives a high fitness and CMA-ES chases it; the
condition then fails to reproduce on fresh seeds. The benchmark measured exactly this:
near nominal every method found falls, none of them validated (0/339).

WHAT THIS DOES. Each candidate is rolled out on ``seeds_per_theta`` distinct seeds in the
same batched call and its search fitness is the MEAN failure score (times transfer
likelihood, as in ``multi_cmaes_search``). That estimates fall *probability*, the quantity
validation tests. Optional novelty sharing reuses ``multi_search.NoveltyArchive``.

COST. At a fixed rollout budget it tests ``seeds_per_theta`` times fewer conditions.
Every rollout is returned (theta repeated per seed) with its own true score, so budget
accounting and failure labels are per rollout exactly as for every other method.

Opt-in only; nothing calls this by default. ``multi_cmaes_search`` is unchanged.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.multi_search import (NoveltyArchive, _normalize, restart_means,
                                         split_scores)
from breakpoint_mjx.search import scores_from_trajectories
from breakpoint_mjx.transfer import transfer_likelihood


def robust_multi_cmaes_search(evaluate, space, cfg, restarts, seeds_per_theta,
                              objective=None, spread=1.0, novelty=False,
                              novelty_radius=0.6, novelty_strength=0.8, log=None):
    """-> (thetas (G*R*P*K, D), scores (same,)): every rollout, in evaluation order.

    One evaluate call per generation of restarts*popsize*seeds_per_theta rollouts; rows
    for one candidate are contiguous (candidate-major), each on a distinct rollout key
    because the batched evaluator draws one key per row.
    """
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import CMA_ES

    K = int(seeds_per_theta)
    if K < 1:
        raise ValueError("seeds_per_theta must be >= 1")
    D = len(space.names())
    lo = jnp.asarray(space.lower())
    hi = jnp.asarray(space.upper())
    popsize = cfg.popsize

    es = CMA_ES(population_size=popsize, solution=jnp.zeros(D))
    params = es.default_params
    means = restart_means(space, restarts, cfg.seed, None, spread)
    states = [es.init(jax.random.key(int(cfg.seed) + 1000 * r), jnp.asarray(means[r]), params)
              for r in range(restarts)]
    es_key = jax.random.key(int(cfg.seed))
    eval_key = jax.random.PRNGKey(int(cfg.seed) + 1)
    archive = NoveltyArchive(novelty_radius, novelty_strength) if novelty else None

    X_out, S_out = [], []
    for g in range(cfg.generations):
        pops, th = [], []
        for r in range(restarts):
            es_key, ask_k = jax.random.split(es_key, 2)
            pop, states[r] = es.ask(ask_k, states[r], params)
            pops.append(pop)
            th.append(lo + jax.nn.sigmoid(pop) * (hi - lo))
        thetas = np.asarray(jnp.concatenate(th, axis=0))          # (R*P, D)
        rows = np.repeat(thetas, K, axis=0)                         # (R*P*K, D)
        eval_key, ek = jax.random.split(eval_key, 2)
        traj = evaluate(jnp.asarray(rows, dtype=jnp.float32), ek)
        scores, _, _ = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, objective)
        scores = np.asarray(scores, dtype=float)
        mean_score = scores.reshape(-1, K).mean(axis=1)             # (R*P,)
        transfer = np.array([transfer_likelihood(space.vector_to_named(t), space)
                             for t in thetas])
        fit = mean_score * transfer
        u = _normalize(space, thetas)
        steer = fit * archive.penalty(u) if archive is not None else fit
        per = split_scores(steer, restarts, popsize)
        for r in range(restarts):
            es_key, tell_k = jax.random.split(es_key, 2)
            states[r], _ = es.tell(tell_k, pops[r], -jnp.asarray(per[r]), states[r], params)
        if archive is not None:
            # a region counts as found when the candidate fails on most of its seeds
            hits = u[mean_score >= max(cfg.failure_threshold, 0.5)]
            if hits.size:
                archive.add(hits)
        X_out.append(rows)
        S_out.append(scores)
        if log:
            log(f"robust gen {g}: rollouts={rows.shape[0]} max_mean={mean_score.max():.3f}")
    return np.concatenate(X_out), np.concatenate(S_out)
