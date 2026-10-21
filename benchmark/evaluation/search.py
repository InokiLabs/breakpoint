"""Run one search method for one (configuration, regime, repetition) at a fixed budget.

uniform, lhs_boundary and directed are ``breakpoint_mjx.equal_budget.run_arm`` unchanged;
directed is pinned to the original novelty-off search (the engine's default turned novelty
on after this benchmark), so the published numbers reproduce. sobol is the one new
baseline. Every method evaluates through the same Recorder, the same
compiled evaluator (same batch shape) and spends exactly ``plan.budget`` rollouts.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from benchmark.baselines.samplers import sobol_points, sobol_rollout_key


def engine_plan(plan):
    from breakpoint_mjx.equal_budget import plan_directed

    return plan_directed(plan.budget, plan.restarts, plan.popsize)


def run_variant(v, recorder, space, cfg, plan, seed, objective):
    """A named directed-search variant (benchmark config ``method_variants``).

    novelty  : the engine's multi_cmaes_search with its existing opt-in novelty sharing
    restarts : the engine's multi_cmaes_search with a different restarts x popsize split
    robust   : breakpoint_mjx.robust_search (each condition scored on several seeds)
    """
    import jax

    from benchmark.evaluation.setup import variant_plan

    r, p, k, g = variant_plan(v, plan)
    vcfg = dataclasses.replace(cfg, seed=int(seed), popsize=p, generations=g)
    kind = v["kind"]
    if kind in ("novelty", "restarts"):
        from breakpoint_mjx.multi_search import multi_cmaes_search

        kw = {"novelty": False}      # the original search, as measured
        if kind == "novelty":
            kw = {"novelty": True, "novelty_radius": float(v.get("radius", 0.6)),
                  "novelty_strength": float(v.get("strength", 0.8))}
        recs = multi_cmaes_search(recorder, space, vcfg, jax.random.PRNGKey(int(seed)),
                                  restarts=r, seed_theta=None, objective=objective, **kw)
        X = np.asarray([x.theta_vector for x in recs], dtype=float)
    elif kind == "robust":
        from breakpoint_mjx.robust_search import robust_multi_cmaes_search

        X, _ = robust_multi_cmaes_search(
            recorder, space, vcfg, restarts=r, seeds_per_theta=k, objective=objective,
            novelty=bool(v.get("novelty", False)),
            novelty_radius=float(v.get("radius", 0.6)),
            novelty_strength=float(v.get("strength", 0.8)))
    else:
        raise ValueError(kind)
    return X, {"budget": plan.budget, "restarts": r, "popsize": p, "seeds_per_theta": k,
               "generations": g, "batch": r * p * k}


def run_method(method, recorder, space, cfg, plan, seed, objective, variant=None):
    """-> dict with the recorded table, timing and bookkeeping for exactly B rollouts."""
    from breakpoint_mjx.equal_budget import _eval_rows, run_arm

    import jax

    ep = engine_plan(plan)
    cfg = dataclasses.replace(cfg, popsize=ep.popsize, generations=ep.generations)
    recorder.reset()
    t0 = time.time()
    vplan = None
    if variant is not None:
        X, vplan = run_variant(variant, recorder, space, cfg, plan, seed, objective)
        evals = len(X)
    elif method in ("uniform", "lhs_boundary", "directed"):
        X, _scores, evals = run_arm(method, recorder, space, cfg, ep, seed,
                                    objective=objective,
                                    novelty=False if method == "directed" else None)
    elif method == "sobol":
        X = sobol_points(space, ep.budget, seed)
        _scores, evals = _eval_rows(recorder, X, ep.batch,
                                    jax.random.PRNGKey(sobol_rollout_key(seed)), cfg,
                                    objective)
    else:
        raise ValueError(f"unknown method {method!r}")
    wall = time.time() - t0
    tab = recorder.table()
    n = tab["theta"].shape[0]
    if evals != ep.budget or n != ep.budget:
        raise RuntimeError(f"{method}: recorded {n} rollouts / reported {evals}, "
                           f"budget {ep.budget}; methods must spend identical budgets")
    if not np.allclose(np.asarray(X, dtype=float), tab["theta"], atol=1e-5):
        raise RuntimeError(f"{method}: recorded thetas differ from the method's own; "
                           "the recorder is not seeing the evaluated rows")
    tm = recorder.timing()
    tm["method_wall_s"] = round(wall, 3)
    tm["overhead_s"] = round(wall - tm["eval_s"], 3)   # sampler / CMA-ES / bookkeeping
    if vplan is not None:
        return {"table": tab, "timing": tm, "evals": int(evals), "plan": vplan}
    return {"table": tab, "timing": tm, "evals": int(evals),
            "plan": {"budget": ep.budget, "restarts": ep.restarts, "popsize": ep.popsize,
                     "generations": ep.generations, "batch": ep.batch}}
