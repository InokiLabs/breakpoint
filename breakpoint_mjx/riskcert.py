"""Operational failure probability with a finite-sample, anytime-valid confidence interval.

Every rate BreakPoint reported before this module is a rate among worlds the SEARCH chose.
Those worlds are adversarially selected, so "0.967 of evaluated envs failed" says nothing
about how often the policy fails where it is actually deployed. This module answers the
question an operator actually asks:

    under a declared deployment distribution p(theta) over the DR space, what is
    P_p(policy fails), with a confidence interval that is valid at any sample size?

What it is, stated plainly (this is an engineering contribution, not a new estimator):

  * The estimator is DOUBLY-ROBUST importance sampling with a betting confidence
    sequence. It is the off-policy estimator of Karampatziakis, Mineiro and Ramdas,
    "Off-Policy Confidence Sequences" (ICML 2021) and Waudby-Smith, Wu, Ramdas,
    Karampatziakis and Mineiro, "Anytime-valid off-policy inference for contextual
    bandits" (ACM/IMS JDS 2024), specialised to a failure indicator: logging policy =
    proposal q, target policy = p, reward = 1{fail}, reward model = surrogate g_hat.
    It is also the prediction-powered-inference rectifier (Angelopoulos et al. 2023)
    moved onto an importance-sampled proposal.
  * The proposal is a DEFENSIVE mixture (Owen and Zhou 2000)
        q = alpha * p + (1 - alpha) * sum_k pi_k * TN(mu_k, diag(s_k^2))
    whose components are diagonal truncated normals on p's support box. Diagonal means
    the truncation normaliser is a product of 1-D normal CDF differences, so q is EXACT
    and the weight bound w = p/q <= 1/alpha holds pointwise (unit-tested). A full
    covariance on a d=12 box has no closed-form normaliser, and an approximate one would
    silently break the bound the guarantee rests on, so it is deliberately not offered.
  * The CI is the hedged-capital betting confidence sequence of Waudby-Smith and Ramdas
    (JRSS-B 2023) for a bounded mean. Anytime-valid: the caller may stop at the first
    batch whose half-width is below a target and the coverage still holds.

What it is NOT: it is not guaranteed to be narrower than plain Monte Carlo. A
distribution-free CI for a mean bounded in a range of width 2/alpha cannot rule out an
unseen draw at the bound, so its half-width has a floor of roughly
(2/alpha) * log(1/delta) / N even with a PERFECT surrogate. Plain-MC Clopper-Pearson
shrinks like sqrt(p/N). So PP/DR-IS only wins once N * P_fail >> 1/alpha.
``dr_floor_halfwidth`` and ``crossover_n`` compute that floor so a caller can see, before
spending GPU, whether a budget can beat plain MC at all. The measurement in
docs/riskcert.md shows where that crossover sits on a SYNTHETIC landscape.

The probability this produces is a SIMULATION probability under the declared p. It is
not a real-world failure rate: sim-to-real calibration needs paired hardware runs.

Pure numpy + stdlib (no scipy), like the rest of the learned layer.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# --------------------------------------------------------------------------- #
# Diagonal truncated normal on a box (exact density, exact sampling)
# --------------------------------------------------------------------------- #

class DiagTruncNormal:
    """Product of independent 1-D normals truncated to ``[lo, hi]`` per dim.

    The normaliser is a product of 1-D ``Phi(b) - Phi(a)`` terms, so ``logpdf`` is exact.
    Sampling is exact per-dim rejection from the untruncated normal. ``sd`` is capped at
    the box width, which keeps the per-dim acceptance above ~0.34 for any mean inside
    the box (a mean outside the box is clipped into it first).
    """

    def __init__(self, mean, sd, lo, hi):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        width = np.maximum(self.hi - self.lo, 1e-12)
        self.mean = np.clip(np.asarray(mean, dtype=np.float64), self.lo, self.hi)
        self.sd = np.clip(np.asarray(sd, dtype=np.float64), 1e-6 * width, width)
        z = [(_ncdf((h - m) / s) - _ncdf((l - m) / s))
             for l, h, m, s in zip(self.lo, self.hi, self.mean, self.sd)]
        self.log_norm = float(np.sum(np.log(np.maximum(z, 1e-300))))

    def logpdf(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        u = (x - self.mean) / self.sd
        lp = -0.5 * (u ** 2).sum(1) - np.log(self.sd).sum() - len(self.sd) * _LOG_SQRT_2PI
        lp = lp - self.log_norm
        inside = np.all((x >= self.lo) & (x <= self.hi), axis=1)
        return np.where(inside, lp, -np.inf)

    def sample(self, n: int, rng) -> np.ndarray:
        d = len(self.mean)
        out = np.empty((n, d))
        for j in range(d):
            got = np.empty(0)
            while got.size < n:
                draw = rng.normal(self.mean[j], self.sd[j], size=2 * (n - got.size) + 8)
                draw = draw[(draw >= self.lo[j]) & (draw <= self.hi[j])]
                got = np.concatenate([got, draw])
            out[:, j] = got[:n]
        return out


# --------------------------------------------------------------------------- #
# (1) Operational distribution p(theta)
# --------------------------------------------------------------------------- #

@dataclass
class OperationalDist:
    """The operator-declared deployment distribution over the DR space.

    Independent per dim. ``kind="uniform"`` is uniform on ``[lo, hi]``;
    ``kind="truncnorm"`` is a diagonal truncated normal on ``[lo, hi]`` with ``mean``/``sd``.
    """

    names: list
    lo: np.ndarray
    hi: np.ndarray
    kind: str = "uniform"
    mean: np.ndarray | None = None
    sd: np.ndarray | None = None
    source: str = "explicit"
    _tn: DiagTruncNormal | None = field(default=None, repr=False)

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=np.float64)
        self.hi = np.asarray(self.hi, dtype=np.float64)
        if np.any(self.hi <= self.lo):
            raise ValueError("OperationalDist needs hi > lo on every dim")
        if self.kind == "truncnorm":
            if self.mean is None or self.sd is None:
                raise ValueError("truncnorm OperationalDist needs mean and sd")
            self._tn = DiagTruncNormal(self.mean, self.sd, self.lo, self.hi)
        elif self.kind != "uniform":
            raise ValueError(f"unknown OperationalDist kind {self.kind!r}")

    @property
    def dim(self) -> int:
        return len(self.lo)

    def sample(self, n: int, rng) -> np.ndarray:
        if self._tn is not None:
            return self._tn.sample(n, rng)
        return self.lo + rng.random((n, self.dim)) * (self.hi - self.lo)

    def logpdf(self, x) -> np.ndarray:
        if self._tn is not None:
            return self._tn.logpdf(x)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        inside = np.all((x >= self.lo) & (x <= self.hi), axis=1)
        return np.where(inside, -float(np.log(self.hi - self.lo).sum()), -np.inf)

    def spec(self) -> dict:
        d = {"kind": self.kind, "source": self.source,
             "dims": {n: [float(a), float(b)] for n, a, b in zip(self.names, self.lo, self.hi)}}
        if self.kind == "truncnorm":
            d["mean"] = {n: float(m) for n, m in zip(self.names, self._tn.mean)}
            d["sd"] = {n: float(s) for n, s in zip(self.names, self._tn.sd)}
        return d

    @classmethod
    def from_space(cls, space, block: dict | None = None) -> "OperationalDist":
        """Default: uniform on each ThetaDim's ``plausible`` band (config.py ThetaDim).

        ``block`` (e.g. an ``operational:`` YAML mapping) overrides per dim with
        ``{name: {"range": [lo, hi]}}`` or ``{name: {"mean": m, "sd": s}}``; a mean/sd on
        any dim makes the whole distribution a truncated normal (the other dims keep a
        flat-ish sd equal to their band width, which is NOT uniform, so give every dim a
        mean/sd if you want an exact truncnorm spec). Ranges are clipped to the search
        bounds, because the simulator is only defined there.
        """
        names = space.names()
        lo, hi, mean, sd = [], [], [], []
        any_normal = False
        for n in names:
            dim = space.dims[n]
            a, b = dim.plausible
            o = (block or {}).get(n) or {}
            if "range" in o:
                a, b = float(o["range"][0]), float(o["range"][1])
            a, b = max(a, dim.bounds[0]), min(b, dim.bounds[1])
            if b <= a:  # degenerate band: widen to the full search range
                a, b = dim.bounds
            lo.append(a)
            hi.append(b)
            if "mean" in o:
                any_normal = True
                mean.append(float(o["mean"]))
                sd.append(float(o.get("sd", (b - a) / 4.0)))
            else:
                mean.append(0.5 * (a + b))
                sd.append(b - a)
        src = "yaml" if block else "plausible"
        if any_normal:
            return cls(names, np.array(lo), np.array(hi), "truncnorm",
                       np.array(mean), np.array(sd), source=src)
        return cls(names, np.array(lo), np.array(hi), source=src)


# --------------------------------------------------------------------------- #
# (2) Defensive-mixture proposal q
# --------------------------------------------------------------------------- #

class DefensiveMixture:
    """q = alpha * p + (1 - alpha) * sum_k pi_k * DiagTruncNormal_k on p's support box.

    ``weight(x) = p(x)/q(x) <= 1/alpha`` exactly, for every x with p(x) > 0.
    ``sample`` also returns which draws came from the defensive p-component: those are
    EXACT draws from p, so a plain-MC baseline on them is free.
    """

    def __init__(self, p: OperationalDist, comps: list, pis, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.p = p
        self.comps = list(comps)
        pis = np.asarray(pis, dtype=np.float64) if len(self.comps) else np.zeros(0)
        self.pis = pis / pis.sum() if pis.size else pis
        self.alpha = float(alpha) if self.comps else 1.0

    @property
    def w_max(self) -> float:
        return 1.0 / self.alpha

    def logpdf(self, x) -> np.ndarray:
        lp = self.p.logpdf(x)
        if not self.comps:
            return lp
        terms = [np.log(self.alpha) + lp]
        for pi, c in zip(self.pis, self.comps):
            terms.append(np.log(max((1.0 - self.alpha) * pi, 1e-300)) + c.logpdf(x))
        return np.logaddexp.reduce(np.stack(terms, 0), axis=0)

    def weight(self, x) -> np.ndarray:
        lp = self.p.logpdf(x)
        lq = self.logpdf(x)
        w = np.exp(np.where(np.isfinite(lp), lp - lq, -np.inf))
        # Exact algebra gives w <= 1/alpha; clip the last ulp of float round-off only.
        return np.minimum(w, self.w_max)

    def sample(self, n: int, rng):
        """Returns ``(x, from_p)`` where ``from_p`` flags the defensive p-draws."""
        if not self.comps:
            return self.p.sample(n, rng), np.ones(n, dtype=bool)
        which = rng.choice(len(self.comps) + 1, size=n,
                           p=np.concatenate([[self.alpha], (1 - self.alpha) * self.pis]))
        x = np.empty((n, self.p.dim))
        for k in range(len(self.comps) + 1):
            idx = np.flatnonzero(which == k)
            if idx.size:
                x[idx] = (self.p.sample(idx.size, rng) if k == 0
                          else self.comps[k - 1].sample(idx.size, rng))
        return x, which == 0

    def spec(self) -> dict:
        return {"alpha": self.alpha, "n_components": len(self.comps),
                "pis": [float(v) for v in self.pis],
                "centers": [c.mean.tolist() for c in self.comps],
                "sds": [c.sd.tolist() for c in self.comps]}


def build_proposal(p: OperationalDist, centers, *, g_fn=None, unc_fn=None,
                   alpha: float = 0.3, sd_frac: float = 0.08, n_quad: int = 200_000,
                   fit: str = "moments", min_mass: float = 5.0,
                   rng=None) -> DefensiveMixture:
    """Defensive mixture seeded by the search's failure-mode centers.

    centers: (K, d) thetas, e.g. family ``mean_theta`` vectors or multi-start CMA-ES final
      means. Centers outside p's support are clipped into it (their failure mass under p
      is what matters).
    g_fn: surrogate failure prediction in [0, 1]. When given, pi_k is proportional to the
      p-weighted predicted failure mass in center k's Voronoi cell (``n_quad`` cheap p
      draws); a floor keeps every pi_k > 0. Without it, pi is uniform.
    unc_fn: optional surrogate epistemic std in [0, 1] (M2's ensemble std). A center whose
      cell is uncertain gets a proportionally wider component.
    sd_frac: component sd as a fraction of p's box width per dim (the floor under "moments").
    fit: "fixed" keeps each component at its search center with sd = sd_frac * width.
      "moments" (default, needs g_fn) moment-matches each component to the surrogate-
      predicted failure mass in its cell: mean and per-dim sd of the g-weighted p-draws,
      floored at sd_frac * width. This is one cross-entropy-method step on the surrogate.
      It matters because a DR failure set is usually wide along the axes that do not
      cause it (a latency failure happens at any friction); a component that is narrow
      there leaves most of the failing p-mass at w ~ 1/alpha, which measured WORSE than
      plain MC on the synthetic benchmark (docs/riskcert.md). Cells with less than
      ``min_mass`` predicted-failure mass keep the fixed component.
    """
    rng = rng or np.random.default_rng(0)
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64)) if len(centers) else \
        np.zeros((0, p.dim))
    if centers.shape[0] == 0:
        return DefensiveMixture(p, [], [], 1.0)
    width = p.hi - p.lo
    centers = np.clip(centers, p.lo, p.hi)
    pis = np.ones(len(centers))
    infl = np.ones(len(centers))
    if g_fn is not None or unc_fn is not None:
        xs = p.sample(n_quad, rng)
        un = (xs - p.lo) / width
        cn = (centers - p.lo) / width
        d2 = ((un[:, None, :] - cn[None, :, :]) ** 2).sum(-1)
        cell = d2.argmin(1)
        if g_fn is not None:
            g = np.clip(np.asarray(g_fn(xs), dtype=np.float64), 0.0, 1.0)
            mass = np.bincount(cell, weights=g, minlength=len(centers))
            pis = mass + 0.05 * max(mass.sum(), 1e-12) / len(centers)
        if unc_fn is not None:
            u = np.clip(np.asarray(unc_fn(xs), dtype=np.float64), 0.0, 1.0)
            cnt = np.maximum(np.bincount(cell, minlength=len(centers)), 1)
            infl = 1.0 + np.bincount(cell, weights=u, minlength=len(centers)) / cnt
    means = centers.copy()
    sds = sd_frac * width * infl[:, None]
    if fit == "moments" and g_fn is not None:
        for k in range(len(centers)):
            gk = g * (cell == k)
            if gk.sum() >= min_mass:
                mu = (gk[:, None] * xs).sum(0) / gk.sum()
                sd = np.sqrt((gk[:, None] * (xs - mu) ** 2).sum(0) / gk.sum())
                means[k] = mu
                sds[k] = np.maximum(sd * infl[k], sd_frac * width)
    elif fit not in ("fixed", "moments"):
        raise ValueError(f"unknown fit {fit!r}")
    comps = [DiagTruncNormal(m, sd, p.lo, p.hi) for m, sd in zip(means, sds)]
    return DefensiveMixture(p, comps, pis, alpha)


# --------------------------------------------------------------------------- #
# (4) Intervals: Clopper-Pearson, betting confidence sequence, floors
# --------------------------------------------------------------------------- #

def _binom_logpmf(k, n, p):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), stable for small k (the regime rare-event bounds live in)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    ks = np.arange(k + 1)
    lp = np.array([_binom_logpmf(int(j), n, p) for j in ks])
    m = lp.max()
    return float(min(1.0, math.exp(m) * np.exp(lp - m).sum()))


def _bisect(fn, lo, hi, iters=80):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, delta: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - delta) Clopper-Pearson interval for a binomial proportion."""
    k, n = int(k), int(n)
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p) >= delta / 2, 0.0, 1.0)
    hi = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) <= delta / 2, 0.0, 1.0)
    return (float(lo), float(hi))


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided (1 - delta) upper bound. k=0, n=299, delta=0.05 gives ~0.00997."""
    k, n = int(k), int(n)
    if n <= 0 or k >= n:
        return 1.0
    return float(_bisect(lambda p: _binom_cdf(k, n, p) <= delta, 0.0, 1.0))


def _predictable_lambdas(x: np.ndarray, delta_eff: float) -> np.ndarray:
    """WSR predictable plug-in bets (before truncation), from data strictly before t."""
    n = len(x)
    t = np.arange(1, n + 1, dtype=np.float64)
    mu_after = (0.5 + np.cumsum(x)) / (t + 1.0)                    # mu_hat_i (includes x_i)
    sq = np.concatenate([[0.0], np.cumsum((x - mu_after) ** 2)])[:-1]
    sig2_prev = (0.25 + sq) / t                                    # sigma^2_{t-1}
    return np.sqrt(2.0 * math.log(1.0 / delta_eff) / (sig2_prev * t * np.log1p(t)))


def _log_capital_max(x, lam, m, side: str, c: float) -> np.ndarray:
    """max over t of the log wealth for candidate means ``m`` (vector), chunked."""
    out = np.full(len(m), -np.inf)
    step = max(1, 2_000_000 // max(len(x), 1))
    for s in range(0, len(m), step):
        mm = m[s:s + step][None, :]
        if side == "+":
            l = np.minimum(lam[:, None], c / np.maximum(mm, 1e-12))
            f = 1.0 + l * (x[:, None] - mm)
        else:
            l = np.minimum(lam[:, None], c / np.maximum(1.0 - mm, 1e-12))
            f = 1.0 - l * (x[:, None] - mm)
        out[s:s + step] = np.cumsum(np.log(np.maximum(f, 1e-300)), axis=0).max(0)
    return out


def _cs_excluded(x, m, delta, c, side, theta):
    """Running-intersection exclusion of candidate means ``m`` for data ``x`` in [0,1]."""
    if side == "two":
        lam = _predictable_lambdas(x, delta / 2.0)
        lp = _log_capital_max(x, lam, m, "+", c)
        lm = _log_capital_max(x, lam, m, "-", c)
        # Hedged capital max(theta K+, (1-theta) K-); its running max is the max of each
        # side's running max, so this is the exact running-intersection test.
        cap = np.maximum(math.log(theta) + lp, math.log(1.0 - theta) + lm)
    elif side == "upper":   # H0: mean >= m is rejected when the data are small
        lam = _predictable_lambdas(x, delta)
        cap = _log_capital_max(x, lam, m, "-", c)
    else:                   # "lower"
        lam = _predictable_lambdas(x, delta)
        cap = _log_capital_max(x, lam, m, "+", c)
    return cap >= math.log(1.0 / delta)


def _unit_grid(n=41):
    g = np.concatenate([np.linspace(0.0, 1.0, n), np.geomspace(1e-9, 0.5, n),
                        1.0 - np.geomspace(1e-9, 0.5, n)])
    return np.unique(np.clip(g, 0.0, 1.0))


def betting_cs(x, delta: float = 0.05, *, side: str = "two", c: float = 0.5,
               theta: float = 0.5, grid=None, refine: int = 50) -> tuple[float, float]:
    """Waudby-Smith and Ramdas hedged-capital betting CS for the mean of ``x`` in [0, 1].

    Returns the running-intersection interval at the last sample, which is valid at ANY
    data-dependent stopping time. ``side="upper"`` gives a one-sided upper bound (the
    ``P(fail) <= eps`` gate) with lo=0; ``side="lower"`` the mirror.
    Bisection from the empirical mean refines each end point; a coarse grid backstops it.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    if x.size == 0:
        return (0.0, 1.0)

    def excl(v):
        return bool(_cs_excluded(x, np.array([v]), delta, c, side, theta)[0])

    # Anchor at the empirical mean (inside the CS except on a delta-probability event)
    # and bisect outward. A coarse grid then checks nothing outside [lo, hi] survives,
    # which guards the bisection against a non-interval set.
    m = _unit_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    anchor = float(x.mean())
    if excl(anchor):
        exc = _cs_excluded(x, m, delta, c, side, theta)
        keep = np.flatnonzero(~exc)
        if keep.size == 0:  # empty set (a delta-probability event): collapse to the mean
            return (anchor, anchor)
        anchor = float(m[keep[len(keep) // 2]])
    lo, hi = 0.0, 1.0
    if side != "upper" and excl(0.0):
        a, b = 0.0, anchor
        for _ in range(refine):
            mid = 0.5 * (a + b)
            a, b = (mid, b) if excl(mid) else (a, mid)
        lo = b
    if side != "lower" and excl(1.0):
        a, b = anchor, 1.0
        for _ in range(refine):
            mid = 0.5 * (a + b)
            a, b = (a, mid) if excl(mid) else (mid, b)
        hi = a
    out = m[(m < lo) | (m > hi)]
    if out.size:
        alive = out[~_cs_excluded(x, out, delta, c, side, theta)]
        if alive.size:
            lo, hi = min(lo, float(alive.min())), max(hi, float(alive.max()))
    return (lo, hi)


@functools.lru_cache(maxsize=512)
def _const_cs_halfwidth(n: int, delta: float) -> float:
    lo, hi = betting_cs(np.full(int(n), 0.5), delta, refine=30)
    return 0.5 * (hi - lo)


def dr_floor_halfwidth(n: int, alpha: float, delta: float = 0.05) -> float:
    """Best possible DR-IS betting half-width at budget ``n`` (PERFECT surrogate).

    With a perfect surrogate every rectifier term is exactly 0, i.e. the rescaled sample
    sits at the midpoint of a range of width 2/alpha. The betting CS still has to rule
    out one unseen draw at either bound, so its width is set by n and alpha alone. This
    is the floor no proposal or surrogate can beat.
    """
    return _const_cs_halfwidth(int(n), float(delta)) * (2.0 / alpha)


def crossover_n(p_fail: float, alpha: float, delta: float = 0.05,
                n_grid=None) -> int | None:
    """Smallest budget on ``n_grid`` at which the DR floor is below plain-MC CP.

    CP half-width is evaluated at the expected count k = round(n * p_fail). Returns None
    if no grid budget crosses. This is an OPTIMISTIC crossover for DR-IS (it assumes a
    perfect surrogate); a real surrogate crosses later. The floor falls like ~1/n and
    CP like ~1/sqrt(n), so "floor < CP" is monotone in n and a bisection over the grid
    finds the first crossing.
    """
    n_grid = n_grid or [int(v) for v in np.unique(np.geomspace(500, 500_000, 31).astype(int))]

    def wins(n):
        k = int(round(n * p_fail))
        a, b = clopper_pearson(k, n, delta)
        return dr_floor_halfwidth(n, alpha, delta) < 0.5 * (b - a)

    if not wins(n_grid[-1]):
        return None
    lo, hi = -1, len(n_grid) - 1   # invariant: wins(n_grid[hi]); not wins(n_grid[lo])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if wins(n_grid[mid]):
            hi = mid
        else:
            lo = mid
    return int(n_grid[hi])


# --------------------------------------------------------------------------- #
# (3) Estimators on fresh rollouts
# --------------------------------------------------------------------------- #

def failure_prob_link(score, threshold: float, band: float = 0.1) -> np.ndarray:
    """Map a surrogate failure SCORE to a [0, 1] control-variate value.

    0 below ``(1 - band) * threshold``, 1 above ``(1 + band) * threshold``, linear between.
    This is a heuristic link, not a calibrated probability. It replaces the older
    ``clip(s / (2 * threshold))`` for this use because that one is ~0.2-0.4 across the
    safe bulk, which feeds large residuals into the rectifier. Validity of the CI does
    not depend on this choice; only its width does.
    """
    s = np.asarray(score, dtype=np.float64) / float(threshold)
    return np.clip((s - (1.0 - band)) / (2.0 * band), 0.0, 1.0)


def surrogate_mean_bound(g_p: np.ndarray, delta: float) -> tuple[float, float]:
    """E_p[g_hat] +/- an empirical-Bernstein radius (Maurer and Pontil 2009), g in [0,1].

    The surrogate term is computed on many cheap p-draws, but "many" is not infinite: at
    P_fail ~ 1e-3 a Hoeffding radius on 1e6 draws would already be comparable to the
    target, so the variance-adaptive bound is used and its delta is paid for.
    """
    g = np.clip(np.asarray(g_p, dtype=np.float64), 0.0, 1.0)
    n = len(g)
    mean = float(g.mean())
    if n < 2:
        return mean, 1.0
    var = float(g.var(ddof=1))
    lg = math.log(4.0 / delta)  # two-sided
    rad = math.sqrt(2.0 * var * lg / n) + 7.0 * lg / (3.0 * (n - 1))
    return mean, rad


def estimate_from_samples(f, w, g_q=None, c_hat: float = 0.0, c_rad: float = 0.0, *,
                          alpha: float, delta: float = 0.05, method: str = "dr") -> dict:
    """CI for P_p(fail) from FRESH draws theta_i ~ q with labels f_i and weights w_i.

    method:
      "is"      : X = alpha * w * f in [0, 1]   (no surrogate; one-sided-friendly range)
      "dr"      : X = c + w (f - g) in [c - 1/alpha, c + 1/alpha]; the surrogate enters
                  only as a control variate, so a wrong surrogate cannot bias it. The
                  c_hat uncertainty (c_rad, spent at delta/2) is added to the interval.
      "is_wald" / "dr_wald": the same point estimates with an ASYMPTOTIC normal CI. Not
                  finite-sample valid; reported only to show what the guarantee costs.
    """
    f = np.asarray(f, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(f)
    wmax = 1.0 / alpha
    if method.startswith("is"):
        y = w * f
        est = float(y.mean())
        if method == "is_wald":
            h = 1.96 * float(y.std(ddof=1)) / math.sqrt(n) if n > 1 else wmax
            return {"estimate": est, "lo": max(0.0, est - h), "hi": est + h}
        lo, hi = betting_cs(y / wmax, delta)
        return {"estimate": est, "lo": lo * wmax, "hi": hi * wmax}
    g_q = np.zeros(n) if g_q is None else np.clip(np.asarray(g_q, dtype=np.float64), 0, 1)
    r = w * (f - g_q)                     # rectifier, in [-1/alpha, 1/alpha]
    est = float(c_hat + r.mean())
    if method == "dr_wald":
        h = 1.96 * float(r.std(ddof=1)) / math.sqrt(n) if n > 1 else wmax
        return {"estimate": est, "lo": max(0.0, est - h), "hi": min(1.0, est + h)}
    d_cs = delta / 2.0 if c_rad > 0 else delta
    lo, hi = betting_cs((r + wmax) / (2 * wmax), d_cs)
    lo = lo * 2 * wmax - wmax + c_hat - c_rad
    hi = hi * 2 * wmax - wmax + c_hat + c_rad
    return {"estimate": est, "lo": max(0.0, lo), "hi": min(1.0, hi)}


def operational_risk(evaluate_fail, p: OperationalDist, q: DefensiveMixture, *,
                     g_fn=None, n_max: int = 8192, batch: int = 512,
                     target_halfwidth: float | None = None, delta: float = 0.05,
                     n_quad: int = 1_000_000, rng=None, c_override=None) -> dict:
    """The operational_risk block: sequential DR-IS with an anytime-valid betting CS.

    evaluate_fail(thetas (n,d) ndarray) -> bool (n,) runs FRESH simulator rollouts. The
    search's own records are never reused: they were adaptively selected, and reusing
    them would void the guarantee. Draws continue in ``batch``-sized calls until the CI
    half-width is <= ``target_halfwidth`` or ``n_max`` is spent; stopping on the CI is
    allowed because the CS is anytime-valid.

    The block always carries a plain-MC Clopper-Pearson baseline computed on the
    defensive p-draws inside the same rollouts (exact p samples, so a valid but
    smaller-N baseline, labelled with its own n).

    ``c_override=(mean, radius)`` reuses an already-computed surrogate-term bound (the
    benchmark does this to avoid recomputing the same quadrature every repetition).
    """
    rng = rng or np.random.default_rng(0)
    alpha = q.alpha
    if g_fn is not None:
        if c_override is not None:
            c_hat, c_rad = (float(v) for v in c_override)
        else:
            c_hat, c_rad = surrogate_mean_bound(g_fn(p.sample(n_quad, rng)), delta / 2.0)
        method = "dr"
    else:
        c_hat, c_rad, method = 0.0, 0.0, "is"
    xs, fs, ws, gs, from_p = [], [], [], [], []
    stopped = "budget"
    res = {"estimate": float("nan"), "lo": 0.0, "hi": 1.0}
    while sum(len(v) for v in fs) < n_max:
        k = min(batch, n_max - sum(len(v) for v in fs))
        x, fp = q.sample(k, rng)
        f = np.asarray(evaluate_fail(x), dtype=bool)
        xs.append(x); fs.append(f); ws.append(q.weight(x)); from_p.append(fp)
        gs.append(np.clip(g_fn(x), 0, 1) if g_fn is not None else np.zeros(k))
        F, W, G = np.concatenate(fs), np.concatenate(ws), np.concatenate(gs)
        res = estimate_from_samples(F, W, G, c_hat, c_rad, alpha=alpha, delta=delta,
                                    method=method)
        if target_halfwidth is not None and 0.5 * (res["hi"] - res["lo"]) <= target_halfwidth:
            stopped = "target_halfwidth"
            break
    F, W, FP = np.concatenate(fs), np.concatenate(ws), np.concatenate(from_p)
    n = len(F)
    ess = float(W.sum() ** 2 / max((W ** 2).sum(), 1e-300))
    k_p, n_p = int(F[FP].sum()), int(FP.sum())
    cp_lo, cp_hi = clopper_pearson(k_p, n_p, delta) if n_p else (0.0, 1.0)
    return {
        "dist": p.spec(),
        "estimate": float(res["estimate"]),
        "lo": float(res["lo"]),
        "hi": float(res["hi"]),
        "confidence": 1.0 - delta,
        "method": ("doubly-robust IS + betting confidence sequence (anytime-valid)"
                   if method == "dr" else "IS + betting confidence sequence (anytime-valid)"),
        "n_rollouts": n,
        "n_failures": int(F.sum()),
        "stopped": stopped,
        "alpha": float(alpha),
        "w_max_bound": float(1.0 / alpha),
        "w_max_observed": float(W.max()) if n else 0.0,
        "ess": round(ess, 2),
        "surrogate_term": ({"mean": c_hat, "radius": c_rad, "n_quad": int(n_quad)}
                           if method == "dr" else None),
        "dr_floor_halfwidth": dr_floor_halfwidth(n, alpha, delta),
        "plain_mc_baseline": {"k": k_p, "n": n_p, "rate": (k_p / n_p) if n_p else None,
                              "lo": cp_lo, "hi": cp_hi,
                              "method": "Clopper-Pearson on the defensive p-draws"},
        "proposal": q.spec(),
        "caveat": ("SIMULATION probability under the declared operational distribution; "
                   "not a real-world failure rate."),
    }


# --------------------------------------------------------------------------- #
# (5) Simulator-backed box certificate
# --------------------------------------------------------------------------- #

def certify_box(evaluate_fail, box_lo, box_hi, n: int = 299, *, eps: float = 0.01,
                delta: float = 0.05, rng=None) -> dict:
    """Upper-bound the failure rate inside a box from ``n`` FRESH simulator rollouts.

    Uniform draws in the box; Clopper-Pearson one-sided upper bound at 1 - delta.
    ``certified`` iff that bound is <= eps. 0 failures in 299 draws bounds the rate at
    ~0.00997 (the "rule of three" identity), so n=299 is the smallest budget that can
    certify eps=0.01 at 95%.
    """
    rng = rng or np.random.default_rng(0)
    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    x = lo + rng.random((int(n), len(lo))) * np.maximum(hi - lo, 0.0)
    f = np.asarray(evaluate_fail(x), dtype=bool)
    k = int(f.sum())
    ub = clopper_pearson_upper(k, int(n), delta)
    return {"n": int(n), "failures": k, "upper_bound": ub, "confidence": 1.0 - delta,
            "eps": float(eps), "certified": bool(ub <= eps),
            "method": "Clopper-Pearson upper bound on fresh simulator rollouts"}


# --------------------------------------------------------------------------- #
# Pipeline stage (jax-free; run_pipeline supplies the simulator closure)
# --------------------------------------------------------------------------- #

def centers_from_run(families, records, space, threshold: float, k: int = 16) -> np.ndarray:
    """Proposal centers from what the search already produced.

    Family ``mean_theta`` vectors first (the clustered failure modes), then the
    highest-scoring failing records, deduplicated at 0.1 normalised distance, up to k.
    """
    names = space.names()
    lo = np.asarray(space.lower(), dtype=np.float64)
    width = np.maximum(np.asarray(space.upper(), dtype=np.float64) - lo, 1e-12)
    cands = []
    for f in families or []:
        mt = getattr(f, "mean_theta", None)
        if mt and all(n in mt for n in names):
            cands.append(np.array([mt[n] for n in names], dtype=np.float64))
    fails = sorted((r for r in records or [] if r.failure_score >= threshold),
                   key=lambda r: -r.failure_score)
    cands += [np.asarray(r.theta_vector, dtype=np.float64) for r in fails]
    out = []
    for c in cands:
        if all(np.linalg.norm((c - o) / width) > 0.1 for o in out):
            out.append(c)
        if len(out) >= k:
            break
    return np.array(out) if out else np.zeros((0, len(names)))


def run_stage(evaluate_fail, space, families, records, threshold: float, *,
              surrogate=None, envelope=None, dist_block: dict | None = None,
              n_max: int = 8192, batch: int = 512, target_halfwidth: float | None = 0.005,
              alpha: float = 0.3, delta: float = 0.05, n_quad: int = 1_000_000,
              certify_n: int = 299, certify_eps: float = 0.01, seed: int = 0,
              mc_reference_n: int = 0) -> dict:
    """The BP_RISK stage: operational_risk block (+ envelope simulator certificate).

    ``surrogate`` (optional) is anything with ``predict_score``; it enters ONLY as the
    control variate, through ``failure_prob_link``. ``envelope`` (optional) is the M3
    ``safety_envelope`` dict; when given, its box is certified with fresh rollouts.
    """
    rng = np.random.default_rng(seed)
    p = OperationalDist.from_space(space, dist_block)
    g_fn = None
    if surrogate is not None:
        def g_fn(x):
            return failure_prob_link(surrogate.predict_score(x), threshold)
    centers = centers_from_run(families, records, space, threshold)
    q = build_proposal(p, centers, g_fn=g_fn, alpha=alpha,
                       n_quad=min(n_quad, 100_000), rng=rng)
    block = operational_risk(evaluate_fail, p, q, g_fn=g_fn, n_max=n_max, batch=batch,
                             target_halfwidth=target_halfwidth, delta=delta,
                             n_quad=n_quad, rng=rng)
    block["label"] = "operational (declared p), not search-selected"
    if mc_reference_n > 0:
        f_ref = np.zeros(0, dtype=bool)
        while len(f_ref) < mc_reference_n:
            k = min(batch, mc_reference_n - len(f_ref))
            f_ref = np.concatenate([f_ref, np.asarray(evaluate_fail(p.sample(k, rng)), bool)])
        kr = int(f_ref.sum())
        lo_r, hi_r = clopper_pearson(kr, len(f_ref), delta)
        block["plain_mc_reference"] = {"k": kr, "n": int(len(f_ref)), "rate": kr / len(f_ref),
                                       "lo": lo_r, "hi": hi_r,
                                       "ci_covers_reference_rate": bool(block["lo"] <= kr / len(f_ref)
                                                               <= block["hi"])}
    out = {"operational_risk": block}
    if envelope is not None:
        from breakpoint_mjx.envelope import certify_envelope
        out["envelope_certificate"] = certify_envelope(
            envelope, space, evaluate_fail, n=certify_n, eps=certify_eps, delta=delta,
            seed=seed + 1)
    return out
