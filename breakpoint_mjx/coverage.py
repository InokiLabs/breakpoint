"""Campaign coverage for a multi-restart failure search: a uniform residual audit, plus
mode counts and (opt-in, UNCALIBRATED) stop rules.

MEASURED STATUS (SYNTHETIC planted landscapes, docs/coverage.md): the audit's 95% bound
held in 200/200 prefixes on each of two 50-run sets; every stop rule FAILED the
pre-registered false-complete <= 10% bar, so the default runs a fixed wave budget and
claims nothing about completeness.

WHAT A VALIDATION LEAD ASKS. "Are we done testing, and what did we miss?" The only
statistic the report had was a Wilson CI over SEARCH-SELECTED envs, which answers
neither. This module adds two numbers and a stopping rule, in this order of trust:

1. UNIFORM RESIDUAL AUDIT (the lead number, no mode-count estimator involved).
   Draw ``n`` worlds uniformly (scrambled Sobol) from the declared DR box and evaluate
   them once. A failing audit world is "covered" if it is CONNECTED to a discovered
   failure mode (see 2); otherwise it is "unseen". ``k_unseen / n`` with a one-sided
   Clopper-Pearson upper bound estimates P(a random world from the declared box fails
   in a region the search never reached). This is textbook Monte Carlo failure-
   probability estimation (Corso et al. JAIR 2021 cover it); nothing new is claimed
   for it. It is the only part free of adaptive-sampling bias.

2. MODES = CONNECTED COMPONENTS OF THE FAILING SET, SAMPLED BY RESTART INCUMBENTS.
   Each independent CMA-ES restart contributes ONE point: its incumbent (best-scoring
   sample), if that sample fails. Two incumbents are the same mode if the straight
   segment between them stays failing at ``n_probe`` interior probes; modes are the
   connected components of that graph (union-find, order independent). There is NO
   radius and nothing tied to a known K. The earlier idea (leader clustering of every
   failing sample at a fixed radius 0.30) was measured to produce 1118 "modes" on a
   6-mode planted landscape, because in 16-D the failing set is a shell wider than the
   radius. Segment connectivity is conservative in one direction only: a curved
   (non-convex) failure region can be SPLIT into several modes, never merged with a
   disjoint one. For the audit that means an unseen-mass bound that errs HIGH.
   In a mostly-failing box every incumbent is connected and the answer is "1 mode",
   which is the honest reading there ("one big failure regime").

3. RICHNESS + STOPPING RULES over those basins. This is the multistart stopping-rule
   setting of Boender & Rinnooy Kan (Math. Programming 37, 1987): independent local
   searches, distinct optima as a multinomial sample over basins, stop when few optima
   are expected to remain. We implement their closed-form posterior estimate
   ``w (N-1) / (N-w-2)`` as a BASELINE rule next to the Chao2 incidence estimator
   (Chao 1987) with Chao & Jost (2012) sample-coverage deficit, and bias-corrected
   Chao1 on basin counts. None of these estimators is new; this is a transplant of
   multistart basin-count stopping rules to adversarial DR failure search, shipped as a
   product feature. Note that with one incumbent per restart Chao2 and Chao1 differ only
   in the (m-1)/m factor and in what m counts.

SCOPE. Every number here is conditional on the declared DR box, the simulator, the
failure threshold and the segment-connectivity mode definition. It is a statement about
simulation, not the real robot, and it cannot certify the worst undiscovered region.

Pure numpy + stdlib. ``score_fn(thetas (N, D) ndarray) -> (N,) failure scores`` is the
only link to physics; ``make_score_fn`` builds it from a pipeline evaluator.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np

#: The rule is frozen here, before any planted-landscape run (see docs/coverage.md).
N_PROBE = 8
STOP_GAP = 0.5          # Chao2 - S_obs must fall below this (and BRK's w_est - w)
STOP_EPS = 0.05         # Chao & Jost new-mode probability per restart must fall below this

_UNIT_RE = re.compile(r"^cmaes-(?:w(\d+)-)?r(\d+)-")


# --------------------------------------------------------------------------- units

def restart_unit(rec_id):
    """``'cmaes-r3-40' -> (0, 3)``, ``'cmaes-w2-r3-40' -> (2, 3)``, else None."""
    m = _UNIT_RE.match(rec_id or "")
    if not m:
        return None
    return (int(m.group(1) or 0), int(m.group(2)))


def restart_incumbents(records, threshold, exclude_restart0=True):
    """Group records by (wave, restart) and keep each unit's best-scoring record.

    Returns ``(units, incumbents)``: ``units`` is every sampling unit seen (sorted),
    ``incumbents`` maps unit -> record for units whose best sample FAILS
    (``failure_score >= threshold``). Restart 0 is excluded by default because its start
    is the LLM seed / box centre, so it is not exchangeable with the scattered restarts.
    """
    best = {}
    for r in records:
        u = restart_unit(r.id)
        if u is None or (exclude_restart0 and u[1] == 0):
            continue
        if u not in best or r.failure_score > best[u].failure_score:
            best[u] = r
    units = sorted(best)
    inc = {u: best[u] for u in units if best[u].failure_score >= float(threshold)}
    return units, inc


# --------------------------------------------------------------------------- modes

def _probe_points(a, b, n_probe):
    t = np.arange(1, n_probe + 1, dtype=float) / (n_probe + 1)
    return a[None, :] + t[:, None] * (b - a)[None, :]


def segments_connected(score_fn, pairs_a, pairs_b, threshold, n_probe=N_PROBE):
    """For each pair (a_i, b_i) of failing theta vectors: does the straight segment
    between them stay failing at every one of ``n_probe`` interior probes?
    One batched ``score_fn`` call for all pairs."""
    pairs_a = np.atleast_2d(np.asarray(pairs_a, dtype=float))
    pairs_b = np.atleast_2d(np.asarray(pairs_b, dtype=float))
    if pairs_a.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    pts = np.concatenate([_probe_points(a, b, n_probe) for a, b in zip(pairs_a, pairs_b)])
    s = np.asarray(score_fn(pts), dtype=float).reshape(pairs_a.shape[0], n_probe)
    return (s >= float(threshold)).all(axis=1)


class ModeGraph:
    """Incumbents as nodes, failing segments as edges; modes = connected components.

    Pair results are cached, so adding a wave only evaluates the new pairs."""

    def __init__(self, score_fn, threshold, n_probe=N_PROBE):
        self.score_fn = score_fn
        self.threshold = float(threshold)
        self.n_probe = int(n_probe)
        self.keys, self.points = [], []
        self._edge = {}
        self.probe_evals = 0

    def add(self, key, theta):
        if key in self.keys:
            return
        self.keys.append(key)
        self.points.append(np.asarray(theta, dtype=float))

    def _ensure_edges(self):
        todo = [(i, j) for j in range(len(self.keys)) for i in range(j)
                if (self.keys[i], self.keys[j]) not in self._edge]
        if not todo:
            return
        P = np.asarray(self.points)
        ok = segments_connected(self.score_fn, P[[i for i, _ in todo]],
                                P[[j for _, j in todo]], self.threshold, self.n_probe)
        self.probe_evals += len(todo) * self.n_probe
        for (i, j), c in zip(todo, ok):
            self._edge[(self.keys[i], self.keys[j])] = bool(c)

    def labels(self, keys=None):
        """Component label per key (restricted to ``keys`` if given, default all)."""
        self._ensure_edges()
        keys = list(self.keys) if keys is None else list(keys)
        idx = {k: n for n, k in enumerate(keys)}
        parent = list(range(len(keys)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), c in self._edge.items():
            if c and a in idx and b in idx:
                ra, rb = find(idx[a]), find(idx[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        roots, out = {}, {}
        for k in keys:
            out[k] = roots.setdefault(find(idx[k]), len(roots))
        return out


# --------------------------------------------------------------------------- estimators

def chao2(incidence_counts, m, z=1.96):
    """Chao2 richness from per-mode incidence counts (units that detected each mode).

    ``m`` = number of sampling units. Returns S_obs, Q1, Q2, estimate, variance, the
    log-normal 95% CI (lower end never below S_obs) and the Chao & Jost (2012)
    sample-coverage deficit = estimated probability that the next unit's detection is a
    mode not yet seen. When Q1 == 0 the estimate equals S_obs and the CI is reported as
    the degenerate [S_obs, S_obs] (Chao2 has no information about unseen modes then).
    """
    q = np.asarray([int(c) for c in incidence_counts if int(c) > 0], dtype=int)
    S = int(q.size)
    Q1 = int((q == 1).sum())
    Q2 = int((q == 2).sum())
    U = int(q.sum())
    m = int(m)
    A = (m - 1) / m if m > 0 else 0.0
    if Q2 > 0:
        f0 = A * Q1 * Q1 / (2.0 * Q2)
        r = Q1 / Q2
        var = Q2 * (A / 2 * r ** 2 + A ** 2 * r ** 3 + A ** 2 / 4 * r ** 4)
    else:
        f0 = A * Q1 * (Q1 - 1) / 2.0
        est_tmp = S + f0
        var = (A * Q1 * (Q1 - 1) / 2.0 + A ** 2 * Q1 * (2 * Q1 - 1) ** 2 / 4.0
               - (A ** 2 * Q1 ** 4 / (4.0 * est_tmp) if est_tmp > 0 else 0.0))
    var = max(float(var), 0.0)
    est = S + f0
    if f0 > 0 and var > 0:
        K = math.exp(z * math.sqrt(math.log(1.0 + var / (f0 * f0))))
        ci = (S + f0 / K, S + f0 * K)
    else:
        ci = (float(S), float(est))
    denom = (m - 1) * Q1 + 2 * Q2
    new_p = (Q1 / U) * ((m - 1) * Q1 / denom) if (U > 0 and denom > 0) else 0.0
    return {"S_obs": S, "Q1": Q1, "Q2": Q2, "m": m, "U": U, "estimate": float(est),
            "var": var, "ci": [float(ci[0]), float(ci[1])], "new_mode_prob": float(new_p)}


def chao1_bc(abundances):
    """Bias-corrected Chao1 on basin hit counts (n = total hits)."""
    f = np.asarray([int(c) for c in abundances if int(c) > 0], dtype=int)
    n = int(f.sum())
    S = int(f.size)
    f1, f2 = int((f == 1).sum()), int((f == 2).sum())
    if n == 0:
        return {"S_obs": 0, "estimate": 0.0}
    return {"S_obs": S, "estimate": float(S + (n - 1) / n * f1 * (f1 - 1) / (2.0 * (f2 + 1)))}


def boender_rinnooy_kan(w, N):
    """Boender & Rinnooy Kan (1987) posterior estimates for N local searches that found
    w distinct optima (closed forms as usually quoted; valid for N >= w + 3):
    expected number of optima ``w (N-1) / (N-w-2)`` and expected relative size of the
    unobserved basins ``w (w+1) / (N (N-1))``. ``stop`` is the standard rule
    ``w_est - w < 0.5``; it can never fire before N >= w + 3."""
    w, N = int(w), int(N)
    if w == 0 or N < w + 3:
        return {"w": w, "N": N, "w_est": None, "unseen_basin_mass": None, "stop": False}
    w_est = w * (N - 1) / (N - w - 2)
    return {"w": w, "N": N, "w_est": float(w_est),
            "unseen_basin_mass": float(w * (w + 1) / (N * (N - 1))),
            "stop": bool(w_est - w < STOP_GAP)}


def _binom_cdf(k, n, p):
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    terms = [lg(n + 1) - lg(i + 1) - lg(n - i + 1) + i * lp + (n - i) * lq
             for i in range(0, k + 1)]
    mx = max(terms)
    return min(1.0, math.exp(mx) * sum(math.exp(t - mx) for t in terms))


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson upper bound on a binomial proportion:
    the p solving P(X <= k; n, p) = alpha. k == n -> 1.0; n == 0 -> 1.0."""
    k, n = int(k), int(n)
    if n <= 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


def clopper_pearson_interval(k, n, alpha=0.05):
    """Two-sided (1-alpha) exact interval."""
    k, n = int(k), int(n)
    if n <= 0:
        return (0.0, 1.0)
    hi = clopper_pearson_upper(k, n, alpha / 2)
    lo = 0.0 if k == 0 else 1.0 - clopper_pearson_upper(n - k, n, alpha / 2)
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- audit

def uniform_audit_thetas(space, n, seed=0):
    """``n`` scrambled-Sobol worlds from the declared box (plain uniform if scipy is
    missing). Returns (n, D) theta vectors."""
    lo = np.asarray(space.lower(), dtype=float)
    hi = np.asarray(space.upper(), dtype=float)
    try:
        from scipy.stats import qmc
        u = qmc.Sobol(d=lo.size, scramble=True, seed=int(seed)).random(int(n))
    except Exception:  # scipy is not a declared dependency
        u = np.random.default_rng(int(seed)).random((int(n), lo.size))
    return lo + u * (hi - lo)


def residual_audit(audit_thetas, audit_scores, graph, mode_labels, threshold,
                   space=None, alpha=0.05):
    """Classify every failing audit world as covered (segment-connected to the nearest
    incumbent of SOME discovered mode) or unseen. Returns counts, the failure-rate CI and
    the Clopper-Pearson upper bound on the unseen failure mass."""
    th = np.asarray(audit_thetas, dtype=float)
    sc = np.asarray(audit_scores, dtype=float)
    n = int(sc.size)
    fail = np.where(sc >= float(threshold))[0]
    k_fail = int(fail.size)
    unseen = np.ones(k_fail, dtype=bool)
    probe_evals = 0
    if k_fail and mode_labels:
        P = {k: p for k, p in zip(graph.keys, graph.points)}
        lo = np.asarray(space.lower()) if space is not None else 0.0
        span = (np.maximum(np.asarray(space.upper()) - lo, 1e-9)
                if space is not None else 1.0)
        modes = sorted(set(mode_labels.values()))
        members = {mm: np.asarray([P[k] for k, lab in mode_labels.items() if lab == mm])
                   for mm in modes}
        A, B, owner = [], [], []
        for idx, x in enumerate(th[fail]):
            for mm in modes:
                M = members[mm]
                d = np.linalg.norm((M - x[None, :]) / span, axis=1)
                A.append(x)
                B.append(M[int(np.argmin(d))])
                owner.append(idx)
        ok = segments_connected(graph.score_fn, np.asarray(A), np.asarray(B),
                                threshold, graph.n_probe)
        probe_evals = len(A) * graph.n_probe
        for o, c in zip(owner, ok):
            if c:
                unseen[o] = False
    k_unseen = int(unseen.sum())
    lo_f, hi_f = clopper_pearson_interval(k_fail, n, alpha)
    return {"n": n, "k_fail": k_fail, "fail_rate": (k_fail / n) if n else 0.0,
            "fail_ci": [lo_f, hi_f], "k_unseen": k_unseen,
            "unseen_rate": (k_unseen / n) if n else 0.0,
            "cp_upper": clopper_pearson_upper(k_unseen, n, alpha),
            "confidence": 1.0 - alpha, "probe_evals": int(probe_evals),
            "unseen_index": [int(fail[i]) for i in np.where(unseen)[0]]}


# --------------------------------------------------------------------------- certificate

def estimate_from_graph(graph, units, incumbents):
    """Richness estimates + all three stop rules for the units seen so far."""
    keys = [u for u in units if u in incumbents]
    labels = graph.labels(keys)
    counts = np.bincount(list(labels.values())) if labels else np.zeros(0, dtype=int)
    m = len(units)
    c2 = chao2(counts, m)
    brk = boender_rinnooy_kan(c2["S_obs"], len(keys))
    c1 = chao1_bc(counts)
    stop_chao2 = bool(c2["S_obs"] > 0 and c2["estimate"] - c2["S_obs"] < STOP_GAP
                      and c2["new_mode_prob"] < STOP_EPS)
    stop_chao1 = bool(c1["S_obs"] > 0 and c1["estimate"] - c1["S_obs"] < STOP_GAP)
    return {"units": m, "failing_units": len(keys), "chao2": c2, "chao1": c1, "brk": brk,
            "stop": {"chao2": stop_chao2, "brk": brk["stop"], "chao1": stop_chao1},
            "labels": {f"w{k[0]}-r{k[1]}": int(v) for k, v in labels.items()}}


def make_score_fn(evaluate, cfg, key, objective=None, batch=None):
    """Wrap a pipeline evaluator as ``score_fn(thetas) -> failure scores``.

    Uses ONE fixed key (deterministic, like margins/crossval) and pads every call to a
    multiple of ``batch`` so a jitted GPU evaluator is not recompiled per shape."""
    import jax.numpy as jnp

    from breakpoint_mjx.search import scores_from_trajectories

    batch = int(batch or cfg.n_eval)

    def score_fn(thetas):
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        out = []
        for s in range(0, thetas.shape[0], batch):
            chunk = thetas[s:s + batch]
            pad = batch - chunk.shape[0]
            if pad:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
            sc, _, _ = scores_from_trajectories(
                evaluate(jnp.asarray(chunk, dtype=jnp.float32), key),
                cfg.thresholds, cfg.horizon, objective)
            out.append(np.asarray(sc, dtype=float)[:batch - pad])
        return np.concatenate(out) if out else np.zeros(0)

    return score_fn


def run_until_saturated(evaluate, space, cfg, key, wave=16, max_waves=4, rule="none",
                        objective=None, seed_theta=None, score_fn=None, n_audit=2048,
                        audit_seed=0, search_fn=None):
    """Multi-restart CMA-ES in waves of ``wave`` independent restarts, then the uniform
    residual audit.

    ``rule`` defaults to 'none': run all ``max_waves`` and make NO completeness claim.
    'chao2' | 'brk' | 'chao1' stop early when that rule fires, but NONE of them passed the
    pre-registered SYNTHETIC safety bar (false-complete <= 10%): chao2 28% on seeds 0-9,
    brk 16% on fresh seeds 10-19 (docs/coverage.md). They are opt-in and labelled
    uncalibrated in the output. ``multi_search.py`` is unchanged: each wave is one
    ``multi_cmaes_search`` call with a distinct seed, and record ids are re-tagged
    ``cmaes-w{wave}-r{restart}-{n}`` so they stay unique and split by unit.

    Returns ``(records, coverage)``; ``coverage`` is JSON-safe."""
    if search_fn is None:
        from breakpoint_mjx.multi_search import multi_cmaes_search as search_fn
    if score_fn is None:
        import jax
        score_fn = make_score_fn(evaluate, cfg, jax.random.PRNGKey(int(cfg.seed) + 7),
                                 objective, batch=wave * cfg.popsize)
    graph = ModeGraph(score_fn, cfg.failure_threshold)
    records, history = [], []
    reason = ("fixed budget, no stop rule" if rule == "none"
              else "max_waves (stop rule did not fire)")
    est = None
    for w in range(int(max_waves)):
        wcfg = replace(cfg, seed=int(cfg.seed) + 100003 * w)
        recs = search_fn(evaluate, space, wcfg, key, restarts=wave,
                         seed_theta=seed_theta if w == 0 else None, objective=objective)
        for i, r in enumerate(recs):
            u = restart_unit(r.id)
            ridx = u[1] if u else 0
            r.id = f"cmaes-w{w}-r{ridx}-{len(records) + i}"
        records += recs
        units, inc = restart_incumbents(records, cfg.failure_threshold)
        for k in units:
            if k in inc:
                graph.add(k, inc[k].theta_vector)
        est = estimate_from_graph(graph, units, inc)
        c2 = est["chao2"]
        history.append({"wave": w, "units": est["units"], "S_obs": c2["S_obs"],
                        "chao2": c2["estimate"], "new_mode_prob": c2["new_mode_prob"],
                        "brk_w_est": est["brk"]["w_est"], "stop": est["stop"]})
        print(f"coverage wave {w}: units={est['units']} failing={est['failing_units']} "
              f"S_obs={c2['S_obs']} chao2={c2['estimate']:.2f} "
              f"new_mode_p={c2['new_mode_prob']:.3f} stop={est['stop']}", flush=True)
        if rule != "none" and est["stop"][rule]:
            reason = f"saturated ({rule})"
            break
    units, inc = restart_incumbents(records, cfg.failure_threshold)
    labels = graph.labels([u for u in units if u in inc])
    audit = None
    if n_audit:
        th = uniform_audit_thetas(space, n_audit, seed=audit_seed)
        audit = residual_audit(th, score_fn(th), graph, labels, cfg.failure_threshold,
                               space=space)
        audit.pop("unseen_index", None)
    coverage = certificate_dict(est, audit, rule, reason, len(history), wave, graph)
    coverage["history"] = history
    return records, coverage


def certificate_dict(est, audit, rule, reason, waves, wave, graph):
    c2 = est["chao2"] if est else chao2([], 0)
    return {
        "method": "segment-connected restart-incumbent basins; Chao2 + Boender-Rinnooy "
                  "Kan stop rules; uniform Sobol residual audit",
        "rule": rule, "stopped_reason": reason, "waves": int(waves),
        "restarts_per_wave": int(wave),
        "S_obs": c2["S_obs"], "chao2": c2["estimate"], "ci": c2["ci"],
        "Q1": c2["Q1"], "Q2": c2["Q2"], "m": c2["m"],
        "new_mode_prob": c2["new_mode_prob"],
        "brk": est["brk"] if est else None,
        "chao1": est["chao1"]["estimate"] if est else None,
        "audit": audit, "n_probe": graph.n_probe, "probe_evals": graph.probe_evals,
        "scope": "simulation only; conditional on the declared DR box, failure threshold "
                 "and segment-connectivity mode definition; a lower bound on mode count; "
                 "does not certify the worst undiscovered region",
    }


def coverage_sentence(cov):
    """One release-review sentence (plain text) for the report and the CLI. The audit
    leads; mode counts are stated as observed, and no completeness claim is made because
    no stop rule passed calibration (docs/coverage.md)."""
    if not cov:
        return ""
    a = cov.get("audit") or {}
    s = ""
    if a:
        s += (f"Uniform audit of {a['n']} random worlds from the declared DR box: "
              f"{a['k_fail']} fail; {a['k_unseen']} fail outside every found failure mode, "
              f"so the failure probability in modes the search never found is "
              f"<= {a['cp_upper']:.4f} ({int(round(a['confidence'] * 100))}% "
              f"Clopper-Pearson). ")
    s += (f"The search found {cov['S_obs']} connected failure mode(s) in {cov['waves']} "
          f"wave(s) of {cov['restarts_per_wave']} restarts ({cov['stopped_reason']}); "
          f"Chao2 estimate {cov['chao2']:.1f} is informational only and is not a "
          f"completeness claim (stop rules failed synthetic calibration).")
    return s + (" Conditional on the declared DR box, the failure threshold and the "
                "segment-connectivity mode definition; simulation only.")
