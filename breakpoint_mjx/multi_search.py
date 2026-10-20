"""Parallel multi-restart CMA-ES: fill the GPU that single-start CMA-ES leaves idle.

THE PROBLEM. ``search.cmaes_search`` runs ``generations`` sequential GPU calls of
``popsize`` (256) worlds each. The evaluator is proven at 2048 worlds per batch, so
every one of those calls uses about an eighth of the device and the search phase is
bounded by GPU *calls*, not by GPU *work*.

WHAT THIS DOES. Run ``restarts`` INDEPENDENT CMA-ES instances, concatenate their
populations into ONE batched evaluate call, then scatter the scores back so each
instance updates on its own slice only. The GPU sees restarts*popsize (~2048) worlds
per call, every instance keeps its full ``generations`` of adaptation, and many
independent starts attack the failure mode this landscape actually has: one run
converging into a single failure family and never finding the others.

WHAT THE MEASUREMENT ACTUALLY SAID (5 seeds, every row at the same 2048 worlds per
call over 15 calls, i.e. identical cost; scripts/bench_search_scaling.py):

    single 1x128 (the old default)   1.8/6 modes  +-0.7
    1x2048  (just raise popsize)     4.8/6 modes  +-0.7
    8x256   restarts                 4.8/6 modes  +-0.7
    16x128  restarts                 5.8/6 modes  +-0.4

Two things not to gloss over. First, MOST of the gain over the old default comes from
simply FILLING THE BATCH: raising popsize to 2048 already nearly triples mode coverage,
and at 8 restarts the restart machinery buys exactly nothing over it -- they tie. The
extra mode and the halved seed variance come from going to MANY SMALL restarts (16),
so the lever is the NUMBER of independent starts, not restarts-versus-popsize as such.
Second, what raising popsize alone cannot buy is generations: at a fixed batch 1x2048
spends its entire budget on a single covariance adaptation, which is why it carries the
higher variance and why it does not improve further.

Record ids carry the restart index (``cmaes-r3-...``) so per-restart behaviour can be
split out of a single run afterwards.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.search import _records, scores_from_trajectories, theta_to_cmaes_mean
from breakpoint_mjx.transfer import joint_plausibility


def restart_means(space, restarts, seed, seed_theta=None, spread=1.0):
    """Diverse CMA-ES starting points, one per restart.

    Restart 0 keeps the informed start (the LLM-proposed seed theta if there is one,
    else the space centre) so multi-start never does WORSE than the single-start run
    it replaces. The rest are scattered so the instances explore different basins;
    identical starts would just run the same search eight times.
    """
    D = len(space.names())
    rng = np.random.default_rng(int(seed))
    means = np.zeros((restarts, D), dtype=float)
    if seed_theta is not None:
        means[0] = theta_to_cmaes_mean(space, seed_theta)
    for r in range(1, restarts):
        # Logit-space scatter: +-spread here is roughly the middle 70% of each
        # bounded dim, so restarts start well apart without pinning to the corners.
        means[r] = rng.normal(0.0, spread, size=D)
    return means


def split_scores(values, restarts, popsize):
    """Cut a concatenated (restarts*popsize,) result back into per-restart slices."""
    values = np.asarray(values)
    expected = restarts * popsize
    if values.shape[0] != expected:
        raise ValueError(
            f"expected {expected} rows ({restarts} restarts x {popsize}), got "
            f"{values.shape[0]} -- the batch was assembled inconsistently"
        )
    return [values[r * popsize:(r + 1) * popsize] for r in range(restarts)]


def _normalize(space, thetas):
    lo = np.array(space.lower())
    hi = np.array(space.upper())
    return (np.asarray(thetas) - lo) / np.maximum(hi - lo, 1e-9)


class NoveltyArchive:
    """Discovered failure regions, used to push later restarts somewhere new.

    Independent restarts are not automatically DIVERSE restarts: several can descend
    into the same basin and re-find the same failure family, which is why plain
    multi-start plateaus well short of the number of modes present. The archive keeps
    one entry per distinct discovered region (anything within ``radius`` of an existing
    entry is the same region) and the search shares fitness against it.

    The penalty applies ONLY to the fitness handed to CMA-ES. Every record keeps its
    true failure_score: this steers the search, it does not edit the results.
    """

    def __init__(self, radius=0.5, strength=0.6):
        self.radius = float(radius)
        self.strength = float(strength)
        self.entries = []          # normalized theta vectors

    def add(self, u_rows):
        """Add normalized rows, keeping only ones that open a NEW region."""
        added = 0
        for u in np.atleast_2d(np.asarray(u_rows, dtype=float)):
            if not self.entries or self._min_dist(u) > self.radius:
                self.entries.append(u.copy())
                added += 1
        return added

    def _min_dist(self, u):
        if not self.entries:
            return np.inf
        return float(np.linalg.norm(np.asarray(self.entries) - u[None, :], axis=1).min())

    def penalty(self, u_rows):
        """Multiplier in (0, 1] per row: 1 far from everything known, lower when crowded.

        Capped by ``strength`` so a crowded sample is discounted, never zeroed -- a
        zero would make CMA-ES blind inside an already-found basin and it could not
        refine the failure it is standing on.
        """
        u_rows = np.atleast_2d(np.asarray(u_rows, dtype=float))
        if not self.entries:
            return np.ones(u_rows.shape[0])
        arch = np.asarray(self.entries)
        d = np.linalg.norm(u_rows[:, None, :] - arch[None, :, :], axis=2).min(axis=1)
        crowd = np.clip(1.0 - d / max(self.radius, 1e-9), 0.0, 1.0)
        return 1.0 - self.strength * crowd


def multi_cmaes_search(evaluate, space, cfg, key, restarts=16, seed_theta=None,
                       objective=None, spread=1.0, novelty=False,
                       novelty_radius=0.6, novelty_strength=0.8):
    """Multi-restart CMA-ES. Returns list[FailureRecord] across all restarts.

    One ``evaluate`` call per generation covering every restart, so the number of
    sequential GPU calls is unchanged from single-start while the work per call rises
    by ``restarts``.

    PICK ``restarts`` SO THAT ``restarts * cfg.popsize`` FILLS THE BATCH (~2048 on the
    hardware this was measured on), AND PREFER MANY SMALL RESTARTS TO FEW LARGE ONES.
    Measured over 5 seeds on a landscape with 6 planted modes, every row costing the
    same 2048 worlds per call over 15 calls:

        single 1x128 (the old default)   1.8/6 modes  +-0.7
        1x2048  (just raise popsize)     4.8/6 modes  +-0.7
        8x256   restarts                 4.8/6 modes  +-0.7
        16x128  restarts                 5.8/6 modes  +-0.4   <- default

    Note the honest middle rows: at 8 restarts, splitting into restarts is worth
    NOTHING over simply raising popsize -- they tie exactly. The gain comes from going
    to 16 small restarts, which also has the lowest variance across seeds. So the
    lever is the NUMBER of independent starts, not restarts-versus-popsize per se.

    ``novelty`` defaults OFF because it did not pay off: at radii 0.5, 0.62 and 0.93
    it found the same 4/6 modes as plain multi-start and slightly worsened mean
    closest approach at 16 restarts (0.467 -> 0.495). It is kept as an option because
    it is cheap and a different landscape may reward it, but it is not on by default
    and it should not be turned on without re-measuring.
    """
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import CMA_ES

    D = len(space.names())
    lo = jnp.asarray(space.lower())
    hi = jnp.asarray(space.upper())
    popsize = cfg.popsize

    es = CMA_ES(population_size=popsize, solution=jnp.zeros(D))
    params = es.default_params
    means = restart_means(space, restarts, cfg.seed, seed_theta, spread)

    states = []
    for r in range(restarts):
        # Distinct evosax key per restart: same key would make the instances draw
        # identical populations and the restarts would be decorative.
        states.append(es.init(jax.random.key(int(cfg.seed) + 1000 * r),
                              jnp.asarray(means[r]), params))

    es_key = jax.random.key(int(cfg.seed))
    eval_key = jax.random.PRNGKey(int(cfg.seed) + 1)
    recs = []
    archive = (NoveltyArchive(novelty_radius, novelty_strength) if novelty else None)

    for g in range(cfg.generations):
        populations, thetas_all = [], []
        for r in range(restarts):
            es_key, ask_k = jax.random.split(es_key, 2)
            pop, states[r] = es.ask(ask_k, states[r], params)
            populations.append(pop)
            thetas_all.append(lo + jax.nn.sigmoid(pop) * (hi - lo))

        thetas = jnp.concatenate(thetas_all, axis=0)      # (restarts*popsize, D)
        eval_key, ek = jax.random.split(eval_key, 2)
        traj = evaluate(thetas, ek)                        # ONE batched GPU call
        scores, sigs, comps = scores_from_trajectories(
            traj, cfg.thresholds, cfg.horizon, objective)

        th_np = np.asarray(thetas)
        transfer = np.array([
            joint_plausibility(space.vector_to_named(th_np[i]), space)
            for i in range(th_np.shape[0])
        ])
        weighted = np.asarray(scores) * transfer

        # Novelty sharing: discount samples sitting in an already-discovered region so
        # restarts spread across basins instead of re-finding one family. Applied to
        # the SEARCH fitness only; the records below keep their true failure scores.
        u = _normalize(space, th_np)
        steer = weighted * archive.penalty(u) if archive is not None else weighted

        per_restart = split_scores(steer, restarts, popsize)
        for r in range(restarts):
            es_key, tell_k = jax.random.split(es_key, 2)
            states[r], _ = es.tell(tell_k, populations[r],
                                   -jnp.asarray(per_restart[r]), states[r], params)

        if archive is not None:
            hits = u[np.asarray(scores) >= cfg.failure_threshold]
            if hits.size:
                archive.add(hits)

        # Tag ids by restart so a single run can be split per restart later.
        start = len(recs)
        gen_recs = _records(space, thetas, scores, sigs, comps, cfg.seed, "cmaes",
                            start, ek)
        for i, rec in enumerate(gen_recs):
            rec.id = f"cmaes-r{i // popsize}-{start + i}"
        recs += gen_recs

        best = float(np.max(weighted))
        regions = len(archive.entries) if archive is not None else 0
        print(f"gen {g}: evals={th_np.shape[0]} max_score={float(np.max(scores)):.3f} "
              f"max_weighted={best:.3f} restarts={restarts} regions={regions}",
              flush=True)

    return recs
