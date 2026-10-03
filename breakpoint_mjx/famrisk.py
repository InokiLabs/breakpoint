"""Cause-Stratified Operational Risk (CSOR): certified per-family deployment risk and
out-of-operating-domain (ODD) labelling from Certified Minimal Cause regions.

The question it answers. The search box is deliberately wide, so most uniformly drawn worlds
fail and a family can be an artifact of the chosen bounds. A buyer needs, per failure family,
"how often does THIS failure happen where I actually deploy?". riskcert.py answers that for
the policy as a whole (one number, opt-in via BP_RISK=1). This module answers it per family.

Method (own design; the pieces are classical, the strata are ours):

  (0) Region. CMC (mincause.py) certifies, for each family, a 1-minimal AND-region
      R_f = AND_{s in S} { theta_s >= t_s }  (or <= for a "low" clause)
      where t_s is the shrunk minimal-counterexample value. For a dim the simulator rounds
      (latency_steps) the region is taken in the effective integer: round(x) >= k is
      x >= k - 0.5. Dims outside S are unconstrained.
  (1) Exact stratum mass. The declared operational distribution p (riskcert.OperationalDist)
      is a product of independent per-dim marginals, so
      m_f = P_p(theta in R_f) = prod_{s in S} P_p(theta_s beyond t_s)
      is a product of 1-D tail CDFs (uniform: a length ratio; truncated normal: a ratio of
      erfc terms). No sampling error in the mass.
  (2) Fresh conditional rollouts. n_f worlds are drawn from p conditioned on R_f: each cause
      dim from its marginal truncated to the region (exact inverse CDF), every other dim from
      p. They are rolled out with the pipeline's own fail_fn and give a Clopper-Pearson
      interval [a_f, b_f] on the conditional failure rate at level delta / F (Bonferroni
      across the F families that get rollouts).
  (3) Per-family operational risk = m_f * [a_f, b_f]. Because m_f is exact and the rollouts
      are fresh draws that never touch the search's records, the interval is valid for the
      data-selected region (the region is fixed before these rollouts are drawn).
  (4) Classification. "outside_odd": m_f == 0 or m_f * b_f < BP_FAMRISK_OUT_EPS (1e-6).
      "in_domain_risk": m_f * a_f > BP_FAMRISK_MIN (1e-3). Otherwise "in_domain_low".
  (5) Residual. sum_f m_f * b_f upper-bounds P_p(fail AND theta in the union of the named
      regions) (union bound, simultaneous by Bonferroni). When BP_RISK=1 also ran under the
      SAME p, total_lo - sum_f m_f*b_f is a lower bound on the operational failure mass the
      named families do not explain (``residual``).

The ODD is chosen in order: the ``operational:`` block of the YAML at BP_RISK_DIST; else
the ``operational:`` block of the contract YAML at BP_OBS_SPEC; else each dim's plausible
band (riskcert.OperationalDist.from_space). Every number here is a SIMULATION probability
under the declared p, not a real-world failure rate.

Pipeline: ``run_stage`` is called at the end of ``mincause.run_stage`` (OFF by default,
BP_FAMRISK=1 enables, because the out-of-ODD label fails its safety bar in the unsafe
direction, see docs/famrisk.md Bar 3; BP_FAMRISK_N overrides the per-family budget). It never fails a run.
Pure numpy + stdlib.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from breakpoint_mjx.riskcert import OperationalDist, clopper_pearson

DELTA = 0.05
MIN_RISK = 1e-3        # BP_FAMRISK_MIN: m*a above this => "in-domain risk"
OUT_EPS = 1e-6         # m*b below this (or m == 0) => "outside operating domain"
N_PRESETS = {"smoke": 32, "standard": 256, "full": 1024}
SEED_BASE = 1_500_000_000   # far from the CMC seed range (seed0 + 1_000_003 * (i + 1) + ...)
SEED_STRIDE = 10_000_019
LABEL = ("SIMULATION probability under the declared operating domain p; not a real-world "
         "failure rate.")
CLASSES = ("in_domain_risk", "in_domain_low", "outside_odd")

_SQRT2 = math.sqrt(2.0)


@functools.lru_cache(maxsize=8192)
def cp_interval(k, n, delta):
    """Two-sided Clopper-Pearson (riskcert.clopper_pearson), cached, and computed on the
    smaller tail by the exact symmetry CP(k, n) = 1 - reversed CP(n - k, n) so a high
    failure count does not sum thousands of binomial terms."""
    k, n = int(k), int(n)
    if n > 0 and k > n - k:
        lo, hi = clopper_pearson(n - k, n, delta)
        return (1.0 - hi, 1.0 - lo)
    return clopper_pearson(k, n, delta)


# --------------------------------------------------------------------------- #
# Regions
# --------------------------------------------------------------------------- #
def region_from_cause(minimal_cause, names, integer_dims=frozenset()):
    """[{dim, idx, op, value, bound}] from a family's ``minimal_cause`` list.

    ``bound`` is the continuous threshold the simulator honours: the CMC value itself, or
    for a rounded dim the half-integer edge of the effective integer (round(x) >= k is
    x >= k - 0.5; round(x) <= k is x <= k + 0.5).
    """
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for c in minimal_cause or []:
        name, v, d = c["dim"], float(c["value"]), c["direction"]
        op = ">=" if d == "high" else "<="
        if name in integer_dims:
            k = int(np.round(v))
            bound = k - 0.5 if op == ">=" else k + 0.5
            shown = k
        else:
            bound, shown = v, v
        out.append({"dim": name, "idx": idx[name], "op": op, "value": shown,
                    "bound": float(bound)})
    return out


def relax_region(region, space, shrink_steps, names=None):
    """Move every threshold toward nominal by CMC's bisection resolution span / 2**steps,
    never past nominal. CMC's shrunk value is a certified FAILING point, so the true
    boundary lies between it and the last bisection point toward nominal, at most
    |theta* - nominal| / 2**steps <= span / 2**steps away."""
    names = names or space.names()
    out = []
    for c in region:
        d = space.dims[names[c["idx"]]]
        step = (d.bounds[1] - d.bounds[0]) / float(2 ** int(shrink_steps))
        if c["op"] == ">=":
            b = max(d.nominal, c["bound"] - step)
        else:
            b = min(d.nominal, c["bound"] + step)
        out.append({**c, "bound": float(b), "value": float(b)})
    return out


def region_string(region):
    if not region:
        return "whole operating domain (fails at nominal)"
    parts = []
    for c in region:
        v = c["value"]
        parts.append(f'{c["dim"]} {c["op"]} {v}' if isinstance(v, int)
                     else f'{c["dim"]} {c["op"]} {v:.4g}')
    return " AND ".join(parts)


def _interval(c):
    return (c["bound"], math.inf) if c["op"] == ">=" else (-math.inf, c["bound"])


# --------------------------------------------------------------------------- #
# Exact per-dim marginals of p
# --------------------------------------------------------------------------- #
def _q(z):
    """Upper normal tail 1 - Phi(z), accurate in both tails (erfc, no cancellation)."""
    return 0.5 * math.erfc(z / _SQRT2)


def _marginal(p: OperationalDist, j):
    """('uniform', lo, hi) or ('tn', lo, hi, mean, sd) for dim j, as p actually samples it
    (a truncnorm p uses DiagTruncNormal's clipped mean/sd, so we read those back)."""
    lo, hi = float(p.lo[j]), float(p.hi[j])
    if getattr(p, "_tn", None) is not None:
        return ("tn", lo, hi, float(p._tn.mean[j]), float(p._tn.sd[j]))
    return ("uniform", lo, hi)


def _tail_mass(marg, a, b):
    """P_p(a <= x_j <= b) for one marginal, exact."""
    lo, hi = marg[1], marg[2]
    a, b = max(a, lo), min(b, hi)
    if b <= a:
        return 0.0
    if marg[0] == "uniform":
        return (b - a) / (hi - lo)
    _, lo, hi, m, s = marg
    # P(a<=X<=b | lo<=X<=hi) with Q = upper tail; stable for far tails on either side.
    def between(u, v):
        zu, zv = (u - m) / s, (v - m) / s
        if zu >= 0:
            return _q(zu) - _q(zv)
        if zv <= 0:
            return _q(-zv) - _q(-zu)
        return 1.0 - _q(-zu) - _q(zv)
    den = between(lo, hi)
    return float(max(0.0, between(a, b)) / den) if den > 0 else 0.0


def stratum_mass(p: OperationalDist, region):
    """(m_f, [per-dim mass]) with m_f = prod of per-dim tail masses under p. Exact."""
    per = [_tail_mass(_marginal(p, c["idx"]), *_interval(c)) for c in region]
    m = 1.0
    for v in per:
        m *= v
    return float(m), [float(v) for v in per]


def _ppf_std(u):
    """Vectorized standard normal quantile by bisection on erfc (no scipy). Accurate for
    small u (the samplers below always call it on the tail that is small)."""
    u = np.asarray(u, float)
    lo, hi = np.full_like(u, -40.0), np.full_like(u, 40.0)
    erfc = np.frompyfunc(math.erfc, 1, 1)
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * erfc(-mid / _SQRT2).astype(float)   # accurate in the left tail
        go = cdf < u
        lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
    return 0.5 * (lo + hi)


def _sample_truncated(marg, a, b, n, rng):
    lo, hi = marg[1], marg[2]
    a, b = max(a, lo), min(b, hi)
    if marg[0] == "uniform":
        return a + rng.random(n) * (b - a)
    _, _, _, m, s = marg
    # Inverse CDF inside [a, b]; work in whichever tail keeps precision.
    if (a - m) / s >= 0:           # right tail: sample on the survival scale
        qa, qb = _q((a - m) / s), _q((b - m) / s)
        qs = qb + rng.random(n) * (qa - qb)
        return np.clip(m - s * _ppf_std(qs), a, b)
    pa, pb = _q(-(a - m) / s), _q(-(b - m) / s)   # Phi(a), Phi(b)
    ps = pa + rng.random(n) * (pb - pa)
    return np.clip(m + s * _ppf_std(ps), a, b)


def sample_stratum(p: OperationalDist, region, n, rng):
    """n draws from p conditioned on the region (exact; region must have positive mass)."""
    X = p.sample(int(n), rng)
    for c in region:
        a, b = _interval(c)
        X[:, c["idx"]] = _sample_truncated(_marginal(p, c["idx"]), a, b, int(n), rng)
    return X


def in_region(X, region):
    X = np.atleast_2d(X)
    ok = np.ones(len(X), bool)
    for c in region:
        col = X[:, c["idx"]]
        ok &= (col >= c["bound"]) if c["op"] == ">=" else (col <= c["bound"])
    return ok


# --------------------------------------------------------------------------- #
# Per-family risk + classification
# --------------------------------------------------------------------------- #
def classify(mass, risk_lo, risk_hi, min_risk=MIN_RISK, out_eps=OUT_EPS):
    if mass <= 0.0 or risk_hi < out_eps:
        return "outside_odd"
    if risk_lo > min_risk:
        return "in_domain_risk"
    return "in_domain_low"


def family_risk(fail_fn, p, region, n, *, delta_f=DELTA, rng=None, seed0=SEED_BASE,
                min_risk=MIN_RISK, out_eps=OUT_EPS, relaxed=None):
    """The odd_risk dict for one region. ``fail_fn(thetas, seeds) -> bool``.

    A zero-mass region spends no rollouts. ``relaxed`` (see ``relax_region``): when the
    certified region has zero mass under p but its boundary-relaxed version does not, the
    rollouts are spent on the relaxed region instead and ``region_basis`` says so; the
    interval is then P_p(fail AND theta in relaxed region), still valid because the
    relaxed region is fixed before the rollouts are drawn. This stops a family whose true
    boundary sits inside the ODD, but whose certified threshold overshot it by less than
    CMC's bisection resolution, from being called out-of-ODD."""
    rng = rng or np.random.default_rng(0)
    m, per = stratum_mass(p, region)
    basis = "certified"
    certified_mass = m
    if m == 0.0 and relaxed is not None:
        m_rel, per_rel = stratum_mass(p, relaxed)
        if m_rel > 0.0:
            region, m, per, basis = relaxed, m_rel, per_rel, "relaxed"
    k = nn = 0
    if m > 0.0 and n > 0:
        X = sample_stratum(p, region, n, rng)
        seeds = (int(seed0) + np.arange(len(X))) & 0x7FFFFFFF
        f = np.asarray(fail_fn(X, seeds), bool)
        k, nn = int(f.sum()), int(len(f))
    a, b = cp_interval(k, nn, float(delta_f)) if nn else (0.0, 1.0)
    lo, hi = m * a, m * b
    return {
        "region": [{kk: c[kk] for kk in ("dim", "op", "value")} for c in region],
        "region_str": region_string(region),
        "mass": m,
        "mass_per_dim": {c["dim"]: v for c, v in zip(region, per)},
        "region_basis": basis,
        "certified_region_mass": certified_mass,
        "k": k, "n": nn,
        "cond_rate": (k / nn) if nn else None,
        "cond_lo": float(a), "cond_hi": float(b),
        "risk": (m * k / nn) if nn else (0.0 if m == 0.0 else None),
        "risk_lo": float(lo), "risk_hi": float(hi),
        "confidence_family": 1.0 - delta_f,
        "class": classify(m, lo, hi, min_risk, out_eps),
        "label": "SIMULATION",
    }


# --------------------------------------------------------------------------- #
# ODD resolution + pipeline stage
# --------------------------------------------------------------------------- #
def _yaml_block(path):
    import yaml

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc.get("operational") if isinstance(doc, dict) else None


def resolve_odd(space, environ=None):
    """(OperationalDist, source string). BP_RISK_DIST > BP_OBS_SPEC contract > plausible."""
    import os

    env = os.environ if environ is None else environ
    for var, what in (("BP_RISK_DIST", "BP_RISK_DIST"), ("BP_OBS_SPEC", "contract")):
        path = env.get(var)
        if not path:
            continue
        try:
            block = _yaml_block(path)
        except Exception:  # an unreadable file falls through to the next source
            block = None
        if block:
            p = OperationalDist.from_space(space, block)
            return p, f"{what} operational: block ({path})"
    return OperationalDist.from_space(space, None), "plausible bands (theta_space.yaml)"


def settings(environ=None):
    """(enabled, n_per_family, min_risk, out_eps, delta)."""
    import os

    env = os.environ if environ is None else environ
    on = str(env.get("BP_FAMRISK", "0")).strip().lower() not in ("0", "false", "off", "no")
    budget = (env.get("BP_BUDGET") or "full").strip().lower()
    n = N_PRESETS.get(budget, N_PRESETS["full"])
    if env.get("BP_FAMRISK_N") not in (None, ""):
        n = int(env["BP_FAMRISK_N"])
        if n < 0:
            raise ValueError(f"BP_FAMRISK_N must be >= 0, got {env['BP_FAMRISK_N']!r}")
    min_risk = float(env.get("BP_FAMRISK_MIN") or MIN_RISK)
    out_eps = float(env.get("BP_FAMRISK_OUT_EPS") or OUT_EPS)
    delta = float(env.get("BP_FAMRISK_DELTA") or DELTA)
    return on, n, min_risk, out_eps, delta


def relax_settings(environ=None):
    """(relax on?, CMC shrink steps for this budget). BP_FAMRISK_RELAX=0 turns the
    boundary relaxation off (the certified-region-only ablation)."""
    import os

    from breakpoint_mjx.mincause import SHRINK_STEPS, stage_settings

    env = os.environ if environ is None else environ
    on = str(env.get("BP_FAMRISK_RELAX", "1")).strip().lower() not in ("0", "false", "off")
    try:
        steps = int(stage_settings(env)[1].get("shrink", SHRINK_STEPS))
    except ValueError:
        steps = SHRINK_STEPS
    return on, steps


def certified(f):
    return getattr(f, "cause_certified", None) is True and \
        getattr(f, "minimal_cause", None) is not None


def run_stage(families, space, fail_fn, environ=None, log=print, seed=0,
              integer_dims=None):
    """Set ``f.odd_risk`` on every CMC-certified family; return the summary dict.

    Never raises: any error returns {"status": "error", ...} and leaves families as they
    were. Families that were not certified keep ``odd_risk = None``.
    """
    try:
        on, n, min_risk, out_eps, delta = settings(environ)
    except ValueError as e:
        if log:
            log(f"per-family operational risk skipped ({e})")
        return {"status": "error", "reason": str(e), "label": LABEL}
    if not on:
        return {"status": "disabled", "reason": "BP_FAMRISK=0", "label": LABEL}
    try:
        from breakpoint_mjx.mincause import _integer_dims

        idims = _integer_dims(space) if integer_dims is None else frozenset(integer_dims)
        p, source = resolve_odd(space, environ)
        names = space.names()
        cert = [f for f in families if certified(f)]
        base = {"odd": p.spec(), "odd_source": source, "n_per_family": n, "delta": delta,
                "min_risk": min_risk, "out_eps": out_eps, "label": LABEL,
                "method": ("cause-stratified: exact stratum mass under p x Clopper-Pearson "
                           "on fresh conditional rollouts, Bonferroni across families")}
        if not cert:
            if log:
                log("per-family operational risk: no CMC-certified family at this budget")
            return {**base, "status": "no_certified", "n_certified": 0, "rollouts": 0,
                    "union_upper": 0.0, "counts": {c: 0 for c in CLASSES}}
        regions = [region_from_cause(f.minimal_cause, names, idims) for f in cert]
        relax_on, steps = relax_settings(environ)
        relaxed = [relax_region(r, space, steps, names) if relax_on else None
                   for r in regions]
        masses = [max(stratum_mass(p, r)[0],
                      stratum_mass(p, rr)[0] if rr is not None else 0.0)
                  for r, rr in zip(regions, relaxed)]
        n_roll = max(1, sum(1 for m in masses if m > 0.0))
        delta_f = delta / n_roll
        rng = np.random.default_rng(int(seed) + 424_242)
        total = 0
        for j, (f, r, rr) in enumerate(zip(cert, regions, relaxed)):
            f.odd_risk = family_risk(fail_fn, p, r, n, delta_f=delta_f, rng=rng,
                                     seed0=SEED_BASE + SEED_STRIDE * j, min_risk=min_risk,
                                     out_eps=out_eps, relaxed=rr)
            total += f.odd_risk["n"]
            if log:
                o = f.odd_risk
                tag = (" (certified region has zero mass in the ODD; measured on the region "
                       "relaxed by the CMC bisection resolution)"
                       if o["region_basis"] == "relaxed" else "")
                log(f"  ODD risk {f.id} [{o['class']}]: {o['region_str']}{tag}; mass "
                    f"{o['mass']:.3g}; {o['k']}/{o['n']} fail (CP {o['cond_lo']:.3f}-"
                    f"{o['cond_hi']:.3f}); risk {o['risk_lo']:.3g}-{o['risk_hi']:.3g} "
                    f"(SIMULATION)")
        counts = {c: sum(1 for f in cert if f.odd_risk["class"] == c) for c in CLASSES}
        union = float(sum(f.odd_risk["risk_hi"] for f in cert))
        if log:
            log(f"per-family operational risk (ODD: {source}): {counts['in_domain_risk']} "
                f"in-domain risk, {counts['in_domain_low']} in-domain low, "
                f"{counts['outside_odd']} outside the operating domain; union upper bound "
                f"{union:.3g}; {total} fresh rollouts")
        base["boundary_relaxation"] = ({"shrink_steps": steps} if relax_on else None)
        return {**base, "status": "ran", "n_certified": len(cert), "rollouts": total,
                "delta_family": delta_f, "union_upper": union, "counts": counts}
    except Exception as e:  # never fail the run over this stage
        for f in families:
            if hasattr(f, "odd_risk"):
                f.odd_risk = None
        if log:
            log(f"per-family operational risk skipped ({type(e).__name__}: {e})")
        return {"status": "error", "reason": f"{type(e).__name__}: {e}", "label": LABEL}


def same_dist(a: dict, b: dict) -> bool:
    keys = ("kind", "dims", "mean", "sd")
    return all(a.get(k) == b.get(k) for k in keys)


def residual(summary, operational_risk):
    """Compare the named families' union upper bound with riskcert's total (BP_RISK=1).

    Returns None when either is missing. ``comparable`` is False when the two used
    different operating domains, in which case no unexplained-mass bound is given.
    """
    if not summary or summary.get("status") != "ran" or not operational_risk:
        return None
    comparable = same_dist(summary.get("odd") or {}, operational_risk.get("dist") or {})
    u = float(summary.get("union_upper", 0.0))
    out = {"families_union_upper": u,
           "total_estimate": float(operational_risk["estimate"]),
           "total_lo": float(operational_risk["lo"]),
           "total_hi": float(operational_risk["hi"]),
           "comparable": bool(comparable)}
    if comparable:
        out["unexplained_mass_lower"] = max(0.0, out["total_lo"] - u)
        out["note"] = ("total_lo - union_upper lower-bounds the operational failure mass "
                       "outside every named family region (confidence >= 1 - 2*delta).")
    else:
        out["note"] = ("BP_RISK used a different operating domain than the per-family "
                       "stage, so the two are not subtracted.")
    return out
