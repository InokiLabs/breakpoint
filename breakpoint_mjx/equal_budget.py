"""Equal-budget baseline benchmark: directed search vs standard practice, same rollouts.

THE QUESTION. Does the engine's directed search find failures that standard robustness
evaluation would miss, when both spend the SAME number of rollouts, and when "better" is
judged by metrics the search does NOT optimize?

Why this module exists: ``scripts/benchmark_search.py`` scores "plausible yield" with the
plausibility heuristic CMA-ES maximizes (score * ``transfer.joint_plausibility``),
so directed-beats-random on that metric is partly circular; and the pipeline's built-in
random population (pipeline.py, ``random_search`` with ``n_eval=popsize``) is one small
batch, not an equal budget.

ARMS (one evaluator, one theta space, one horizon, B rollouts per arm per seed):

  * ``uniform``  -- B iid uniform draws over the theta box. What "evaluate on N random
    DR configs" means in practice.
  * ``directed`` -- the engine's multi-restart CMA-ES (``multi_search.multi_cmaes_search``)
    with ``restarts * popsize * generations == B`` exactly (see ``plan_directed``). No LLM
    seed (restart 0 starts at the box centre) so the arm is deterministic and offline.
  * ``train_dr`` -- B draws from the policy's DECLARED operational / training-DR
    distribution (``riskcert.OperationalDist`` resolved by ``famrisk.resolve_odd`` from
    BP_RISK_DIST or a contract's ``operational:`` block). Skipped, with the reason logged,
    when none is declared: the plausible-band fallback is not a training distribution.
  * ``lhs_boundary`` -- a RoboGate-style baseline (arXiv 2603.22126: Latin-hypercube
    sampling plus boundary-focused sampling). Half the budget (rounded to whole batches)
    is a Latin hypercube over the box; the rest is spent around the stage-1 worlds whose
    failure score sits closest to the threshold, by Gaussian perturbation (5% of each
    dim's range). This is our implementation of the published description, not their
    code; it is here so the comparison includes a current competitor's search strategy,
    not only plain random sampling.

All arms evaluate in batches of ``restarts * popsize`` rows so they share ONE
compiled evaluator (a different batch shape would recompile the jitted rollout).

METRICS (per arm, per seed). None of them reads ``transfer_likelihood``:

  a. ``n_failures``        failure_score >= the target's failure threshold.
  b. ``closest_l2`` / ``closest_linf``  min distance from nominal over failures, in
     box-normalized units ((theta - lo) / (hi - lo)). Lower is better: the standard
     "most-likely failure" proxy in falsification (a failure near nominal is one a real
     deployment can hit). L2 is the raw Euclidean norm in the unit box (max sqrt(D)).
  c. ``n_in_band``         failures that pass plausibility.py's RULE checks: every
     ``impossible_reasons`` rule, plus its ``n_extreme <= MAX_EXTREME`` and
     ``n_outside <= max_outside_default`` count rules. plausibility.py's ``low_transfer``
     rule IS the transfer heuristic, so it is deliberately NOT applied here. Caveat: the
     two count rules read the same per-dim plausible bands the heuristic averages, so this
     metric is a different functional of shared inputs, not a fully independent one.
  d. ``n_modes`` / ``n_modes_in_band``  distinct direction-aware dominant-deviating dims
     among failures ("high latency_steps"), the transfer-free fingerprint from
     ``scripts/benchmark_search.py::_dominant_mode``.
  e. ``replay_confirmed`` / ``replay_n`` (optional, BP_BASELINE_REPLAY=1): the k closest
     failures per arm re-run on the independent C-MuJoCo full-collision replay
     (``crossval_engines.replay_family``, the pipeline's replay stage); the rate is
     reproduced / replayed.

Across seeds: mean, min, max and the per-seed values. Every output says "simulation only".

    python -m breakpoint_mjx.equal_budget --target go1 --budget 192 --seeds 2 \\
        --preset smoke --out out/baseline
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

BASIS_LABEL = "simulation only"
# Order matters: an arm's RNG stream is keyed by its index, so new arms are appended.
ARMS = ("uniform", "directed", "train_dr", "lhs_boundary")
BOUNDARY_SIGMA = 0.05       # lhs_boundary stage-2 perturbation, fraction of each range
METRIC_KEYS = ("n_failures", "failure_rate", "closest_l2", "closest_linf", "n_in_band",
               "n_modes", "n_modes_in_band", "replay_rate")


# --------------------------------------------------------------------------------------
# Pure metric functions (numpy + the ThetaSpace; no JAX)
# --------------------------------------------------------------------------------------

def _box(space):
    lo = np.asarray(space.lower(), dtype=float)
    hi = np.asarray(space.upper(), dtype=float)
    return lo, np.where(hi > lo, hi - lo, 1.0)


def box_normalize(X, space):
    """(N, D) thetas -> unit-box coordinates (theta - lo) / (hi - lo)."""
    lo, span = _box(space)
    return (np.atleast_2d(np.asarray(X, dtype=float)) - lo) / span


def nominal_distances(X, space):
    """(l2, linf) distance of each row from nominal, in box-normalized units."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    d = box_normalize(X, space) - box_normalize(space.nominal_vector(), space)[0]
    return np.sqrt((d ** 2).sum(axis=1)), np.abs(d).max(axis=1)


def closest_failure(X_fail, space):
    """Min L2 / L-inf nominal distance over failures; None values when there are none.

    The two minima are taken independently (the L2-closest and the L-inf-closest failure
    can be different rows); ``index`` is the L2-closest row."""
    l2, linf = nominal_distances(X_fail, space)
    if l2.size == 0:
        return {"l2": None, "linf": None, "index": None}
    i = int(np.argmin(l2))
    return {"l2": float(l2[i]), "linf": float(linf.min()), "index": i}


def in_band(theta_named, space, horizon=None):
    """True iff a world passes plausibility.py's rule checks WITHOUT its transfer floor.

    impossible_reasons (all rules) + the joint count rules (n_extreme, n_outside). The
    ``low_transfer`` rule is skipped because it is transfer_likelihood itself."""
    from breakpoint_mjx.plausibility import (EXTREME_FRAC, MAX_EXTREME, dim_excess,
                                             impossible_reasons, max_outside_default)

    if impossible_reasons(theta_named, space, horizon=horizon):
        return False
    ex = [dim_excess(float(theta_named.get(n, d.nominal)), d) for n, d in space.dims.items()]
    n_outside = sum(1 for e in ex if e > 0.0)
    n_extreme = sum(1 for e in ex if e >= EXTREME_FRAC)
    return n_extreme <= MAX_EXTREME and n_outside <= max_outside_default(space)


def in_band_mask(X, space, horizon=None):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.array([in_band(space.vector_to_named(x), space, horizon) for x in X],
                    dtype=bool).reshape(-1)


def dominant_modes(X, space):
    """Direction-aware dominant deviating dim per row ('high latency_steps').

    Same fingerprint as scripts/benchmark_search.py::_dominant_mode: |theta - nominal|
    over the bound span, argmax dim, signed by the side of nominal. Transfer-free."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 0:
        return []
    _, span = _box(space)
    nom = np.asarray(space.nominal_vector(), dtype=float)
    names = space.names()
    j = np.argmax(np.abs(X - nom) / span, axis=1)
    return [("high " if X[i, k] > nom[k] else "low ") + names[k] for i, k in enumerate(j)]


def arm_metrics(X, scores, space, threshold, horizon=None):
    """Every transfer-free metric for one arm's (thetas, failure_scores). Pure."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if X.shape[0] != scores.shape[0]:
        raise ValueError(f"{X.shape[0]} thetas vs {scores.shape[0]} scores")
    fail = scores >= float(threshold)
    Xf = X[fail]
    band = in_band_mask(Xf, space, horizon) if len(Xf) else np.zeros(0, dtype=bool)
    close = closest_failure(Xf, space)
    modes = sorted(set(dominant_modes(Xf, space)))
    modes_band = sorted(set(dominant_modes(Xf[band], space))) if band.any() else []
    return {
        "evals": int(X.shape[0]),
        "n_failures": int(fail.sum()),
        "failure_rate": round(float(fail.mean()) if len(fail) else 0.0, 4),
        "closest_l2": close["l2"],
        "closest_linf": close["linf"],
        "n_in_band": int(band.sum()),
        "n_modes": len(modes),
        "n_modes_in_band": len(modes_band),
        "modes": modes,
    }


def k_closest_failures(X, scores, space, threshold, k):
    """The k failures closest to nominal (L2, box units), closest first."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    fail = np.asarray(scores, dtype=float).reshape(-1) >= float(threshold)
    Xf = X[fail]
    if len(Xf) == 0 or k <= 0:
        return Xf[:0]
    l2, _ = nominal_distances(Xf, space)
    return Xf[np.argsort(l2, kind="stable")[:k]]


def aggregate(per_seed, keys=METRIC_KEYS):
    """{key: {mean, min, max, values, n_missing}} across seeds; None values skipped."""
    out = {}
    for k in keys:
        vals = [r.get(k) for r in per_seed]
        got = [float(v) for v in vals if v is not None]
        out[k] = {"mean": (float(np.mean(got)) if got else None),
                  "min": (float(np.min(got)) if got else None),
                  "max": (float(np.max(got)) if got else None),
                  "values": vals, "n_missing": len(vals) - len(got)}
    return out


# --------------------------------------------------------------------------------------
# Budget accounting
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DirectedPlan:
    budget: int
    restarts: int
    popsize: int
    generations: int

    @property
    def batch(self) -> int:
        return self.restarts * self.popsize

    def describe(self) -> str:
        return (f"{self.restarts} restarts x {self.popsize} popsize x {self.generations} "
                f"generations = {self.budget} rollouts ({self.batch} per batched call)")


def plan_directed(budget, restarts, popsize):
    """Split B into restarts * popsize * generations EXACTLY, or refuse.

    Rounding would hand one arm more rollouts than another, which is the flaw this
    benchmark exists to remove, so a non-divisible budget is an error naming valid ones."""
    budget, restarts, popsize = int(budget), int(restarts), int(popsize)
    if min(budget, restarts, popsize) <= 0:
        raise ValueError("budget, restarts and popsize must be positive")
    batch = restarts * popsize
    if budget % batch:
        lo = max(batch, budget // batch * batch)
        raise ValueError(
            f"budget {budget} is not a multiple of restarts*popsize = {restarts}*{popsize} "
            f"= {batch}; use {lo} or {lo + batch} (or change --restarts/--popsize)")
    return DirectedPlan(budget, restarts, popsize, budget // batch)


# --------------------------------------------------------------------------------------
# train_dr arm: the declared operational distribution (reuses famrisk.resolve_odd)
# --------------------------------------------------------------------------------------

def resolve_train_dr(space, environ=None):
    """(OperationalDist or None, source-or-skip-reason).

    Uses the same resolution as the per-family risk stage (BP_RISK_DIST, then the
    BP_OBS_SPEC contract's ``operational:`` block). The plausible-band fallback is NOT a
    declared training distribution, so it yields None and the arm is skipped."""
    from breakpoint_mjx.famrisk import resolve_odd

    env = os.environ if environ is None else environ
    p, src = resolve_odd(space, env)
    if src.startswith("plausible"):
        return None, ("no operational distribution declared (BP_RISK_DIST or a contract "
                      "`operational:` block); the plausible-band default is not the "
                      "policy's training DR, so the train_dr arm is skipped")
    return p, src


# --------------------------------------------------------------------------------------
# Runner (needs an evaluator; tests pass a synthetic one)
# --------------------------------------------------------------------------------------

def _eval_rows(evaluate, X, batch, key, cfg, objective):
    """Evaluate (N, D) rows in fixed-size batches (last one padded, padding discarded)
    so every call has the directed arm's shape and reuses its compile. Returns scores
    and the number of REAL rollouts counted against the budget."""
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.search import scores_from_trajectories

    X = np.asarray(X, dtype=np.float32)
    out = []
    for s in range(0, X.shape[0], batch):
        chunk = X[s:s + batch]
        n = chunk.shape[0]
        if n < batch:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch - n, axis=0)])
        key, ek = jax.random.split(key)
        traj = evaluate(jnp.asarray(chunk), ek)
        sc, _, _ = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, objective)
        out.append(np.asarray(sc)[:n])
    scores = np.concatenate(out) if out else np.zeros(0)
    return scores, int(X.shape[0])


def nominal_check(evaluate, space, cfg, plan, *, objective=None, seed=0):
    """One extra batch (outside every arm's budget) of the all-nominal world on distinct
    rollout keys: the fraction that fails. If the policy already fails at nominal, the
    distance-to-nominal metric is meaningless and the report must say so."""
    import jax

    X = np.tile(np.asarray(space.nominal_vector(), dtype=float), (plan.batch, 1))
    sc, n = _eval_rows(evaluate, X, plan.batch, jax.random.PRNGKey(int(seed) + 424_242),
                       cfg, objective)
    fails = int((sc >= cfg.failure_threshold).sum())
    return {"n": n, "failures": fails, "rate": round(fails / max(n, 1), 4),
            "max_score": round(float(sc.max()), 4) if n else None}


def run_arm(arm, evaluate, space, cfg, plan, seed, *, objective=None, train_dist=None,
            novelty=None, log=print):
    """One arm, one seed -> (thetas (B, D), scores (B,), evals). Exactly B rollouts.
    ``novelty`` (directed arm only) overrides multi_cmaes_search's default when set."""
    import dataclasses

    import jax

    B = plan.budget
    if arm == "directed":
        from breakpoint_mjx.multi_search import multi_cmaes_search

        dcfg = dataclasses.replace(cfg, seed=int(seed), popsize=plan.popsize,
                                   generations=plan.generations)
        recs = multi_cmaes_search(evaluate, space, dcfg, jax.random.PRNGKey(int(seed)),
                                  restarts=plan.restarts, seed_theta=None,
                                  objective=objective,
                                  **({} if novelty is None else {"novelty": bool(novelty)}))
        X = np.asarray([r.theta_vector for r in recs], dtype=float)
        sc = np.asarray([r.failure_score for r in recs], dtype=float)
        evals = len(recs)
    elif arm in ("uniform", "train_dr"):
        rng = np.random.default_rng([int(seed), ARMS.index(arm)])
        if arm == "uniform":
            lo = np.asarray(space.lower(), dtype=float)
            hi = np.asarray(space.upper(), dtype=float)
            X = lo + rng.random((B, len(lo))) * (hi - lo)
        else:
            if train_dist is None:
                raise ValueError("train_dr arm needs a declared distribution")
            X = np.asarray(train_dist.sample(B, rng), dtype=float)
        key = jax.random.PRNGKey(int(seed) * 7919 + 101 * (ARMS.index(arm) + 1))
        sc, evals = _eval_rows(evaluate, X, plan.batch, key, cfg, objective)
    elif arm == "lhs_boundary":
        X, sc, evals = _lhs_boundary(evaluate, space, cfg, plan, seed, objective)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if evals != B:
        raise RuntimeError(f"{arm} arm spent {evals} rollouts, budget is {B}")
    return X, sc, evals


def _lhs_boundary(evaluate, space, cfg, plan, seed, objective):
    """Latin hypercube, then boundary-focused refinement (see the module docstring)."""
    import jax
    from scipy.stats import qmc

    B, batch = plan.budget, plan.batch
    lo = np.asarray(space.lower(), dtype=float)
    hi = np.asarray(space.upper(), dtype=float)
    b1 = min(B, max(batch, (B // 2) // batch * batch))
    sampler = qmc.LatinHypercube(d=len(lo), seed=np.random.default_rng(
        [int(seed), ARMS.index("lhs_boundary")]))
    X1 = lo + sampler.random(b1) * (hi - lo)
    key = jax.random.PRNGKey(int(seed) * 7919 + 101 * (ARMS.index("lhs_boundary") + 1))
    k1, k2 = jax.random.split(key)
    s1, e1 = _eval_rows(evaluate, X1, batch, k1, cfg, objective)
    b2 = B - b1
    if b2 <= 0:
        return X1, s1, e1
    rng = np.random.default_rng([int(seed), ARMS.index("lhs_boundary"), 2])
    n_anchor = max(1, b2 // 4)
    anchors = X1[np.argsort(np.abs(s1 - cfg.failure_threshold))[:n_anchor]]
    X2 = anchors[np.arange(b2) % n_anchor] + rng.normal(
        0.0, BOUNDARY_SIGMA, (b2, len(lo))) * (hi - lo)
    X2 = np.clip(X2, lo, hi)
    s2, e2 = _eval_rows(evaluate, X2, batch, k2, cfg, objective)
    return np.concatenate([X1, X2]), np.concatenate([s1, s2]), e1 + e2


def run_benchmark(evaluate, space, cfg, plan, seeds, *, arms=ARMS, objective=None,
                  train_dist=None, train_reason=None, replay_fn=None, replay_k=4,
                  log=print):
    """All arms x seeds. ``replay_fn(X_closest) -> {"n", "confirmed", ...}`` is optional.

    Returns {"arms": {arm: {"per_seed": [...], "summary": {...}} | {"skipped": reason}}}."""
    result = {"arms": {}}
    for arm in arms:
        if arm == "train_dr" and train_dist is None:
            result["arms"][arm] = {"skipped": train_reason or "no declared distribution"}
            if log:
                log(f"[equal_budget] train_dr skipped: {train_reason}")
            continue
        per_seed = []
        for s in seeds:
            t0 = time.time()
            if log:
                log(f"[equal_budget] arm={arm} seed={s} budget={plan.budget} ...")
            X, sc, evals = run_arm(arm, evaluate, space, cfg, plan, s, objective=objective,
                                   train_dist=train_dist, log=log)
            m = arm_metrics(X, sc, space, cfg.failure_threshold, cfg.horizon)
            m["seed"] = int(s)
            m["replay_rate"] = None
            if replay_fn is not None:
                Xc = k_closest_failures(X, sc, space, cfg.failure_threshold, replay_k)
                try:
                    rp = replay_fn(Xc) if len(Xc) else {"n": 0, "confirmed": 0}
                except Exception as e:  # noqa: BLE001 -- a replay error never loses the arms
                    rp = {"n": 0, "confirmed": 0, "error": f"{type(e).__name__}: {e}"}
                    if log:
                        log(f"[equal_budget] replay failed for arm={arm} seed={s}: "
                            f"{rp['error']}")
                m["replay"] = rp
                m["replay_rate"] = (rp["confirmed"] / rp["n"]) if rp.get("n") else None
            m["wall_s"] = round(time.time() - t0, 1)
            if log:
                log(f"[equal_budget] arm={arm} seed={s}: failures={m['n_failures']}/"
                    f"{evals} closest_l2={m['closest_l2']} in_band={m['n_in_band']} "
                    f"modes={m['n_modes']} [{m['wall_s']} s]")
            per_seed.append(m)
        result["arms"][arm] = {"per_seed": per_seed, "summary": aggregate(per_seed)}
    return result


# --------------------------------------------------------------------------------------
# Replay confirmation (C-MuJoCo full-collision; the pipeline's replay machinery)
# --------------------------------------------------------------------------------------

def replay_confirmation(X, backend, space, *, k_seeds, score_fn, threshold, thresholds,
                        batch_key, n_after, max_mult=1):
    """Replay each theta row on ``k_seeds`` seeds via crossval_engines.replay_family.

    A raw failure theta is wrapped as a family whose "minimal cause" is every dim, so
    ``minimal_counterexample`` rebuilds exactly that theta. ``backend`` is anything with
    ``simulate(record)`` + a settable ``early_stop`` (MujocoCBackend, or a test fake)."""
    from breakpoint_mjx.crossval_engines import replay_family

    names = space.names()
    rows = []
    for i, x in enumerate(np.atleast_2d(np.asarray(X, dtype=float))):
        fam = SimpleNamespace(
            id=f"eb-{i}", cause_certified=True,
            minimal_cause=[{"dim": n, "value": float(v)} for n, v in zip(names, x)])
        rp, _, _ = replay_family(backend, space, fam, k=k_seeds, score_fn=score_fn,
                                 threshold=threshold, thresholds=thresholds,
                                 batch_key=batch_key, n_after=n_after, max_mult=max_mult)
        rows.append({"verdict": rp["verdict"], "k": rp["k"], "n": rp["n"],
                     "reproduces": rp["reproduces"]})
    return {"n": len(rows), "confirmed": sum(1 for r in rows if r["reproduces"]),
            "rows": rows}


def make_replay_fn(env, policy_fn, target, space, consts, cfg, base_model, *, k_seeds,
                   seed=0, max_mult=1):
    """(replay_fn, None) or (None, reason). Same gates as crossval_engines._replay_stage."""
    from breakpoint_mjx.crossval_engines import REPLAY_BATCH_KEY, score_summary_fn

    obj = getattr(target, "objective", None)
    if obj is not None and getattr(obj, "name", "legged") != "legged":
        return None, "the C-MuJoCo replay is legged-only"
    try:
        from breakpoint_mjx.backends.mujoco_c import MujocoCBackend, build_full_collision_model
    except Exception as e:  # noqa: BLE001
        return None, f"C-MuJoCo backend unavailable ({type(e).__name__}: {e})"
    if not MujocoCBackend.ready():
        return None, "mujoco not importable"
    if getattr(type(env), "_get_obs", None) is None:
        return None, f"{type(env).__name__} lacks the mujoco_playground hooks"
    saved = getattr(env, "_mjx_model", None)
    env._mjx_model = base_model
    try:
        model, info = build_full_collision_model(env, target)
    finally:
        env._mjx_model = saved
    if model is None:
        return None, f"no full-collision model: {info.get('reason')}"
    backend = MujocoCBackend(env, policy_fn, target, space, consts, cfg.horizon,
                             mj_model=model, jit_env=True)
    score_fn = score_summary_fn(cfg.thresholds, cfg.horizon, obj)
    n_after = int(round(1.0 / float(env.dt)))
    batch_key = [0, (REPLAY_BATCH_KEY + int(seed)) & 0xFFFFFFFF]

    def fn(X):
        # rollout_one rebinds env._mjx_model during the search; the C replay's reset reads
        # it, so restore the pristine model for the replay (as replay_stage does).
        prev = getattr(env, "_mjx_model", None)
        env._mjx_model = base_model
        try:
            out = replay_confirmation(X, backend, space, k_seeds=k_seeds, score_fn=score_fn,
                                      threshold=cfg.failure_threshold,
                                      thresholds=cfg.thresholds, batch_key=batch_key,
                                      n_after=n_after, max_mult=max_mult)
        finally:
            env._mjx_model = prev
        out["collision_model"] = info.get("source")
        return out

    return fn, None


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

_COLS = (("n_failures", "failures", "{:.1f}"),
         ("closest_l2", "closest L2 (lower=better)", "{:.3f}"),
         ("closest_linf", "closest Linf (lower=better)", "{:.3f}"),
         ("n_in_band", "in-band failures", "{:.1f}"),
         ("n_modes", "modes", "{:.1f}"),
         ("n_modes_in_band", "in-band modes", "{:.1f}"),
         ("replay_rate", "replay confirmed", "{:.2f}"))


def _cell(s, fmt):
    if s["mean"] is None:
        return "n/a"
    if s["min"] == s["max"]:
        body = fmt.format(s["mean"])
    else:
        body = f"{fmt.format(s['mean'])} [{fmt.format(s['min'])}, {fmt.format(s['max'])}]"
    return body + (f" ({s['n_missing']} seed(s) n/a)" if s["n_missing"] else "")


def basis_line(basis):
    return (f"{basis.get('label') or 'run'}: target {basis['target']}, B={basis['budget']} "
            f"rollouts per arm per seed, horizon {basis['horizon']} steps, "
            f"{len(basis['seeds'])} seed(s) {basis['seeds']}, {BASIS_LABEL}")


def to_markdown(result):
    b = result["basis"]
    lines = [f"# Equal-budget baseline ({b.get('label') or 'run'})", "",
             f"**Basis:** {basis_line(b)}.", "",
             f"Directed split: {b['directed_plan']}. Policy: {b['policy']}. "
             f"Failure threshold {b['failure_threshold']}.", ""]
    nc = b.get("nominal_check")
    if nc:
        lines += [f"Nominal check (one extra batch, outside every arm's budget): "
                  f"{nc['failures']}/{nc['n']} all-nominal rollouts fail"
                  + (" -- the policy fails at nominal, so distance-to-nominal is not "
                     "meaningful here." if nc["failures"] else "."), ""]
    lines += ["Cells: mean [min, max] across seeds. No metric uses the transfer-likelihood "
              "heuristic the directed search optimizes.", ""]
    cols = [c for c in _COLS if c[0] != "replay_rate" or b.get("replay")]
    lines.append("| arm | " + " | ".join(c[1] for c in cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for arm, r in result["arms"].items():
        if "skipped" in r:
            lines.append(f"| {arm} | " + " | ".join("skipped" for _ in cols) + " |")
            continue
        lines.append(f"| {arm} | " + " | ".join(_cell(r["summary"][k], f) for k, _, f in cols)
                     + " |")
    skipped = [(a, r["skipped"]) for a, r in result["arms"].items() if "skipped" in r]
    if skipped:
        lines.append("")
        lines += [f"- {a} skipped: {why}" for a, why in skipped]
    if b.get("replay") and b.get("replay_note"):
        lines += ["", f"Replay: {b['replay_note']}"]
    lines += ["", "Per-seed values are in equal_budget.json."]
    return "\n".join(lines) + "\n"


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "equal_budget.json")
    mp = os.path.join(out_dir, "equal_budget.md")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(result))
    return jp, mp


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser(add_help=True):
    ap = argparse.ArgumentParser(
        prog="breakpoint baseline", add_help=add_help,
        description="Equal-budget baseline: directed search vs uniform (vs the declared "
                    "training DR) at the same number of rollouts, on transfer-free metrics.")
    ap.add_argument("--target", default="go1")
    ap.add_argument("--budget", type=int, required=True,
                    help="rollouts per arm per seed (B); must equal restarts*popsize*gens")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0, help="first seed (seeds are seed0..)")
    ap.add_argument("--preset", default=None, choices=("smoke", "standard", "full"),
                    help="BP_BUDGET preset for horizon/popsize (default: BP_BUDGET or full)")
    ap.add_argument("--horizon", type=int, default=None, help="override rollout horizon")
    ap.add_argument("--restarts", type=int, default=None,
                    help="directed restarts (default: the preset's BUDGET_RESTARTS, else 16)")
    ap.add_argument("--popsize", type=int, default=None, help="directed popsize override")
    ap.add_argument("--policy", default=None,
                    help="policy spec as for `breakpoint run`; default for go1 = the "
                         "bundled demo policy")
    ap.add_argument("--obs-spec", default=None, help="policy contract YAML (BP_OBS_SPEC)")
    ap.add_argument("--risk-dist", default=None,
                    help="YAML with an `operational:` block (BP_RISK_DIST) for train_dr")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--replay", action="store_true",
                    help="C-MuJoCo replay of the k closest failures (also "
                         "BP_BASELINE_REPLAY=1)")
    ap.add_argument("--replay-k", type=int, default=4, help="failures replayed per arm/seed")
    ap.add_argument("--replay-seeds", type=int, default=None,
                    help="replay seeds per failure (default BP_REPLAY_K or 8)")
    ap.add_argument("--label", default=None,
                    help="basis label printed on every output, e.g. 'CPU smoke'")
    ap.add_argument("--out", default="out/equal_budget")
    return ap


def main(argv=None):
    import sys

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    return run(build_parser().parse_args(argv))


def run(a):
    """Run the benchmark from parsed args (shared by ``main`` and ``breakpoint baseline``)."""
    import dataclasses

    from breakpoint_mjx.accel import tune_for_gpu
    tune_for_gpu()
    import jax  # noqa: F401  (after the XLA flags)

    from breakpoint_mjx.config import (BUDGET_RESTARTS, load_search_config,
                                       load_theta_space, merge_target_thresholds)
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.targets import get_target

    env_vars = dict(os.environ)
    if a.preset:
        env_vars["BP_BUDGET"] = a.preset
    preset = (env_vars.get("BP_BUDGET") or "full").strip().lower()
    target = get_target(a.target)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml"),
                                                     env=env_vars), target)
    if a.horizon:
        cfg = dataclasses.replace(cfg, horizon=int(a.horizon))
    elif target.horizon is not None and not env_vars.get("BP_HORIZON"):
        cfg = dataclasses.replace(cfg, horizon=int(target.horizon))
    if a.popsize:
        cfg = dataclasses.replace(cfg, popsize=int(a.popsize))
    restarts = a.restarts or int(env_vars.get("BP_RESTARTS")
                                 or BUDGET_RESTARTS.get(preset, 16))
    plan = plan_directed(a.budget, restarts, cfg.popsize)
    cfg = dataclasses.replace(cfg, generations=plan.generations)
    space = load_theta_space(config_path(target.theta_space_path))

    if a.obs_spec:
        os.environ["BP_OBS_SPEC"] = os.path.abspath(a.obs_spec)
    if a.risk_dist:
        os.environ["BP_RISK_DIST"] = os.path.abspath(a.risk_dist)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"--arms: unknown {bad}; choose from {ARMS}")
    train_dist, train_src = (resolve_train_dr(space) if "train_dr" in arms
                             else (None, "not requested"))

    env = target.load_env()
    policy_spec = a.policy
    if policy_spec is None:
        if target.name != "go1":
            raise SystemExit("--policy is required for targets other than go1")
        from breakpoint_mjx.cli import demo_asset
        policy_spec = str(demo_asset())
        policy_label = "bundled Go1 demo policy (assets/demo/go1_policy.pkl)"
    else:
        policy_label = policy_spec
    from breakpoint_mjx.pipeline import _maybe_adapter_policy
    pargs = argparse.Namespace(policy=policy_spec, obs_norm=None, activation=None,
                               action_scale=None, action_bias=None, squash=False,
                               weight_layout=None, mean_dim=None)
    policy_fn = _maybe_adapter_policy(env, pargs, target=target)
    consts = target.resolve_constants(env)
    base_model = env.mjx_model
    evaluate = make_evaluator(env, policy_fn, space, cfg.horizon, target, consts)

    replay_on = a.replay or os.environ.get("BP_BASELINE_REPLAY") == "1"
    replay_fn, replay_note = None, None
    if replay_on:
        k_seeds = a.replay_seeds or int(os.environ.get("BP_REPLAY_K", "8"))
        replay_fn, why = make_replay_fn(env, policy_fn, target, space, consts, cfg,
                                        base_model, k_seeds=k_seeds, seed=a.seed0)
        replay_note = (f"{a.replay_k} closest failures per arm/seed, {k_seeds} seeds each, "
                       f"C-MuJoCo full-collision; rate = reproduced/replayed (CP lower "
                       f"bound >= the CMC bar)" if replay_fn else f"not run: {why}")
        print(f"[equal_budget] replay: {replay_note}")

    seeds = [a.seed0 + i for i in range(a.seeds)]
    basis = {"label": a.label, "target": target.name, "budget": plan.budget,
             "horizon": cfg.horizon, "seeds": seeds, "preset": preset,
             "directed_plan": plan.describe(),
             "directed_split": {"restarts": plan.restarts, "popsize": plan.popsize,
                                "generations": plan.generations},
             "directed_seed_theta": "none (restart 0 at box centre; no LLM seed)",
             "failure_threshold": cfg.failure_threshold, "policy": policy_label,
             "train_dr_source": train_src if train_dist is not None else None,
             "replay": bool(replay_fn), "replay_note": replay_note,
             "basis": BASIS_LABEL, "backend": jax.default_backend()}
    print(f"[equal_budget] {basis_line(basis)}; directed = {plan.describe()}")
    t0 = time.time()
    basis["nominal_check"] = nominal_check(evaluate, space, cfg, plan,
                                           objective=target.objective, seed=a.seed0)
    print(f"[equal_budget] nominal check (outside the budget): "
          f"{basis['nominal_check']['failures']}/{basis['nominal_check']['n']} nominal "
          f"rollouts fail")
    result = run_benchmark(evaluate, space, cfg, plan, seeds, arms=arms,
                           objective=target.objective, train_dist=train_dist,
                           train_reason=train_src, replay_fn=replay_fn, replay_k=a.replay_k)
    basis["wall_s"] = round(time.time() - t0, 1)
    result = {"basis": basis, **result}
    jp, mp = write_outputs(result, a.out)
    print(to_markdown(result))
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
