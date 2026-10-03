"""Surrogate-guided ACTIVE search (BP_SEARCH=active): the M1+M2 learned-layer modules become a
search operator.

CMA-ES and MAP-Elites pay one expensive GPU rollout for every theta they try. This mode
closes the loop the acquisition module (M2) was built for:

  1. train a deep ENSEMBLE of failure surrogates (M1) on every labeled record so far,
  2. score a huge uniform candidate pool for FREE on the CPU,
  3. simulate only the most INFORMATIVE candidates -- worlds on the predicted failure
     boundary where the ensemble disagrees (boundary x epistemic-uncertainty, with a
     greedy diversity filter so the batch spans the boundary instead of clustering),
  4. fold the new labels back in and repeat.

Each round also keeps an EXPLORATION fraction of pure uniform rows, so the search can
escape the surrogate's own blind spots (a surrogate trained only on its own proposals
otherwise never learns about regions it wrongly predicts safe). Budget-parity with
cmaes_search: cfg.generations rounds x cfg.popsize evals.

The batch PROPOSAL logic (`propose_batch`) is pure numpy and unit-tested on CPU
(tests/test_active_search.py); only the outer loop touches jax (key handling + the GPU
evaluate), same discipline as llm_search.
"""
from __future__ import annotations

import numpy as np

from breakpoint_mjx.acquisition import sample_candidates, select_informative
from breakpoint_mjx.search import _records, scores_from_trajectories
from breakpoint_mjx.surrogate import Ensemble

# Below this many labeled records the ensemble has nothing real to say -- cold-start
# rounds are pure uniform exploration.
MIN_RECORDS_TO_FIT = 16


class _SeedRecord:
    """Minimal labeled point for surrogate warm-start (what propose_batch reads)."""

    __slots__ = ("theta_vector", "failure_score")

    def __init__(self, theta_vector, failure_score):
        self.theta_vector = theta_vector
        self.failure_score = failure_score


def transferable_seed_records(prior_records, space, *, max_records: int = 4096):
    """Map a PRIOR run's records (an earlier policy version, or another robot sharing the
    same theta dims) into surrogate seed data for the CURRENT space -- cross-policy
    warm start: every past run's labeled (theta -> failure_score) points warm-start
    the next policy's active search at zero GPU cost.

    Name-based and defensive (edge-case-enumeration decisions inline):
      * thetas are rebuilt FROM ``theta_named`` in the CURRENT space's dim order, so a
        prior run whose YAML listed dims in a different order still maps correctly
        (HANDLE); a record missing ``theta_named`` falls back to ``theta_vector`` only
        when its length matches D (HANDLE), else it is DROPPED (a mismatched-dim record
        cannot be interpreted).
      * a record missing ANY current dim name is DROPPED (REJECT-by-skip: filling with
        nominal would fabricate a label at a theta that was never simulated).
      * non-finite theta values or scores are DROPPED (they would poison the ensemble).
      * values are clipped to the CURRENT bounds (a prior version searched wider bounds
        still contributes boundary-adjacent evidence).
      * capped at ``max_records`` (newest-last order preserved; the cap keeps ensemble
        training seconds-fast as history accumulates).

    Returns a list of lightweight seed records (only ``theta_vector`` /
    ``failure_score``, the two fields ``propose_batch`` reads) -- prior records are
    TRAINING DATA for the surrogate, never returned as this run's results.
    """
    names = space.names()
    D = len(names)
    lo = np.asarray(space.lower(), dtype=np.float64)
    hi = np.asarray(space.upper(), dtype=np.float64)
    out = []
    for r in prior_records or []:
        named = getattr(r, "theta_named", None)
        if named:
            if not all(n in named for n in names):
                # Incomplete name set == a DIFFERENT theta space; its vector ordering
                # cannot be trusted either, so drop rather than misinterpret.
                continue
            vec = np.array([float(named[n]) for n in names], dtype=np.float64)
        else:
            vec = np.asarray(getattr(r, "theta_vector", None) or [], dtype=np.float64)
            if vec.shape != (D,):
                continue  # mismatched dims: uninterpretable in this space
        score = getattr(r, "failure_score", None)
        if score is None or not np.isfinite(score) or not np.isfinite(vec).all():
            continue
        vec = np.clip(vec, lo, hi)
        out.append(_SeedRecord([float(x) for x in vec], float(score)))
    # Keep the newest ``max_records``. NOT ``out[-cap:]`` -- ``cap == 0`` would make that
    # ``out[-0:] == out[:]`` (the WHOLE list), the opposite of a zero cap; max(0, ...)
    # yields [] for cap <= 0 and all rows when cap exceeds len(out).
    cap = int(max_records)
    return out[max(0, len(out) - cap):]


def propose_batch(records, space, n, rng, *, explore_frac=0.25, pool=8192,
                  ensemble_k=5, threshold=0.5, fit_kw=None):
    """Pure: propose the next ``(n, D) float32`` theta batch from the labeled ``records``.

    * Cold start (fewer than MIN_RECORDS_TO_FIT records) -> all-uniform batch.
    * Otherwise: train an Ensemble on (theta_vector -> failure_score), score a ``pool``-row
      uniform candidate pool, pick ``n - n_explore`` boundary-x-uncertainty rows via
      ``select_informative``, and fill ``n_explore = round(n * explore_frac)`` rows with
      fresh uniform draws (escape hatch from surrogate blind spots).

    ``rng`` is a numpy Generator (the caller owns the stream; passing the same generator
    twice advances it). ``fit_kw`` forwards to ``Surrogate.fit`` (tests shrink epochs).

    Returns ``(batch, info)`` where info = {"mode": "explore"|"acquire", "r2": float|None,
    "n_exploit": int} -- printed by the round loop so a silent cold-start is visible.
    """
    lo = np.asarray(space.lower(), dtype=np.float64)
    hi = np.asarray(space.upper(), dtype=np.float64)
    D = len(space.names())
    n = int(n)

    def uniform(m):
        return lo + rng.random((int(m), D)) * (hi - lo)

    if len(records) < MIN_RECORDS_TO_FIT:
        return uniform(n).astype(np.float32), {"mode": "explore", "r2": None, "n_exploit": 0}

    thetas = np.asarray([r.theta_vector for r in records], dtype=np.float64)
    scores = np.asarray([r.failure_score for r in records], dtype=np.float64)
    ens = Ensemble.train(thetas, scores, space, k=int(ensemble_k), **(fit_kw or {}))
    for m in ens.members:  # acquisition centers its boundary term on this threshold
        m.threshold = float(threshold)
    r2 = float(np.mean([m.r2(thetas, scores) for m in ens.members]))

    n_explore = int(round(n * float(np.clip(explore_frac, 0.0, 1.0))))
    n_exploit = n - n_explore
    rows = []
    if n_exploit > 0:
        cands = sample_candidates(space, int(pool), seed=int(rng.integers(2**31 - 1)))
        idx = select_informative(ens, cands, n_exploit)
        rows.append(cands[idx])
    if n_explore > 0:
        rows.append(uniform(n_explore))
    batch = np.clip(np.vstack(rows), lo, hi).astype(np.float32)
    return batch, {"mode": "acquire", "r2": r2, "n_exploit": n_exploit}


def active_search(evaluate, space, cfg, key, objective=None, *, rounds=None,
                  explore_frac=0.25, pool=8192, ensemble_k=5, seed_records=None):
    """Closed-loop surrogate-guided search. Returns list[FailureRecord] across all rounds.

    ``seed_records`` (e.g. run_pipeline's already-evaluated random-search records) warm-start
    the surrogate at zero extra GPU cost -- they inform the ensemble but are NOT returned
    (the caller already owns them). Budget: ``rounds or cfg.generations`` x cfg.popsize.
    """
    import jax
    import jax.numpy as jnp

    R = int(rounds or cfg.generations)
    N = int(cfg.popsize)
    rng = np.random.default_rng(int(cfg.seed))
    labeled = list(seed_records or [])
    out = []
    for r in range(R):
        batch, info = propose_batch(
            labeled, space, N, rng, explore_frac=explore_frac, pool=pool,
            ensemble_k=ensemble_k, threshold=cfg.failure_threshold,
        )
        key, ek = jax.random.split(key)
        traj = evaluate(jnp.asarray(batch), ek)
        scores, sigs, comps = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon,
                                                       objective)
        recs = _records(space, batch, scores, sigs, comps, cfg.seed, f"active-r{r}",
                        len(out), ek)
        out += recs
        labeled += recs
        n_fail = int((scores >= cfg.failure_threshold).sum())
        r2s = f"{info['r2']:.3f}" if info["r2"] is not None else "n/a"
        print(f"active round {r}: mode={info['mode']} exploit={info['n_exploit']}/{N} "
              f"surrogate_r2={r2s} max_score={scores.max():.3f} fails={n_fail}/{N}",
              flush=True)
    return out
