"""Certified Minimal Cause (CMC): name and merge failure families by the minimal
necessary-and-sufficient perturbation from nominal.

Why this exists. ``clustering.build_families`` names a family by the dims whose family MEAN
deviates most from the mean of ALL failures, with all 16 domain-randomization dims perturbed
at once. That produced names that contradict the displayed values (an "underweight trunk"
family whose representative carries +1.11 kg). CMC instead asks the simulator which of the
representative's perturbations are actually needed:

  (i)   Candidate set C = dims where the representative theta* differs from nominal by more
        than 1% of the dim's span.
  (ii)  Batched delta debugging (ddmin, Zeller & Hildebrandt 2002) toward nominal. A
        configuration is a subset S of dims kept at theta*, every other dim at nominal. Each
        round builds ALL candidate reductions at once (every S minus one dim, plus the chunks
        and chunk-complements of the current partition) and scores them in ONE ``fail_fn``
        call per seed round. Seed certification (for flaky noise-driven rollouts) is
        SEQUENTIAL: seeds come in rounds of K/2 (pooled counts, one batched call per round,
        up to ``max_seed_mult * K`` seeds) and a configuration stops as soon as it crosses
        an exact binomial group-sequential boundary on either side of 0.5, with overall
        one-sided error <= alpha = 0.1 (``look_schedule``). A reduction "still fails" only
        when it certifies FAIL. The loop ends when no
        reduction certifies, so S is 1-minimal: reverting any one dim of S stops the
        certified failure.
  (iii) Necessity check (in the spirit of DeCaF counterfactuals): theta* with S reverted to
        nominal and all other dims kept must NOT certify as failing.
  (iv)  Magnitude shrink: every s in S is bisected between nominal and theta*[s] (6 steps,
        batched across S), giving a minimal counterexample theta_min and its normalized
        L-inf distance from nominal.
  (v)   Label = conjunction over S via ``naming.label_from_cause`` with direction from
        sign(theta_min[s] - nominal[s]), never from a failure mean, plus an assertion that
        every label direction agrees with the displayed representative values.
  (vi)  Families whose (S, directions) coincide are merged.

The core (``certified_minimal_cause``) is PURE: it takes an injectable
``fail_fn(thetas (N, D), seeds (N,)) -> bool (N,)`` so it is unit-tested on CPU with
analytic functions. ``make_evaluator_fail_fn`` is the thin GPU adapter over the pipeline's
``evaluate`` (same pattern as ``margins.make_evaluator_score_fn``) and is Colab-validated
only. With ``k_seeds=1`` the certification is disabled (plain ddmin on single outcomes);
that is the ablation reported in docs/mincause.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CAND_FRAC = 0.01      # (i) a dim is a candidate if |theta* - nominal| > 1% of span
ALPHA = 0.10          # one-sided Clopper-Pearson level
FAIL_BAR = 0.5        # a config "fails" if CP lower bound on its failure fraction >= this
SHRINK_STEPS = 6

# Dims the rollout ROUNDS before use (rollout.py: jnp.round(latency_steps)
# to an integer action delay). ThetaSpace has no integer flag, so they are listed here; a
# space may override with an ``integer_dims`` attribute. For these dims the shrunk value is
# reported as the effective integer threshold (0.516 steps runs as a 1-step delay).
INTEGER_DIMS = frozenset({"latency_steps"})


def _integer_dims(space):
    dims = getattr(space, "integer_dims", None)
    return frozenset(dims) if dims is not None else INTEGER_DIMS


def effective_threshold(name, value, integer_dims=INTEGER_DIMS):
    """The value the simulator actually applies: ``round`` (half to even, as jnp.round)
    for integer dims, unchanged otherwise."""
    if name in integer_dims:
        return int(np.round(value))
    return value


# --------------------------------------------------------------------------- #
# Clopper-Pearson bounds (pure stdlib: exact binomial tail + bisection, no scipy)
# --------------------------------------------------------------------------- #
def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k, n + 1)))


def cp_lower(k, n, alpha=ALPHA):
    """One-sided (1 - alpha) Clopper-Pearson LOWER bound on p given k successes in n."""
    if k <= 0 or n <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    lo, hi = 0.0, 1.0
    for _ in range(60):                       # P(X >= k | p) increasing in p; solve = alpha
        mid = 0.5 * (lo + hi)
        if _binom_sf(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_upper(k, n, alpha=ALPHA):
    """One-sided (1 - alpha) Clopper-Pearson UPPER bound on p given k successes in n."""
    if n <= 0:
        return 1.0
    return 1.0 - cp_lower(n - k, n, alpha)


def decide(k, n, k_seeds, alpha=ALPHA, bar=FAIL_BAR):
    """'fail' | 'pass' | 'undecided' for a configuration seen failing k of n times.

    ``k_seeds == 1`` is the uncertified ablation: a single observed failure is a failure.
    """
    if k_seeds <= 1:
        return "fail" if k >= 1 else "pass"
    if cp_lower(k, n, alpha) >= bar:
        return "fail"
    if cp_upper(k, n, alpha) < bar:
        return "pass"
    return "undecided"


# --------------------------------------------------------------------------- #
# The certifier: batched, pooled, SEQUENTIAL seed certification of many configs
# --------------------------------------------------------------------------- #
def _binom_pmf_row(n, q):
    return [math.comb(n, j) * q ** j * (1.0 - q) ** (n - j) for j in range(n + 1)]


def look_schedule(k_seeds, max_n, alpha=ALPHA, bar=FAIL_BAR):
    """Exact binomial group-sequential boundaries for sequential seed certification.

    Seeds are added in rounds of ``h = K // 2`` up to ``max_n``. At look r (n_r seeds, k
    failures) a configuration is certified FAIL when ``k >= c_r`` and PASS when
    ``k <= d_r``; otherwise it gets another round. The boundaries are computed by exact
    dynamic programming over the binomial paths at the least favorable failure rate
    q = ``bar``: at every look c_r is the loosest threshold that keeps the cumulative
    probability of a false FAIL (over all looks so far, counting only paths still
    undecided) at or below ``alpha``, and the same for d_r and a false PASS. This is an
    alpha-spending design that spends alpha as early as it can; the overall one-sided error
    is <= ``alpha`` on each side for every q on the wrong side of the bar (the crossing
    probability is monotone in q by a path-coupling argument). It is a standard
    group-sequential test, not claimed as novel.

    Looks where no still-undecided path can reach either boundary are dropped (merged into
    the next round), so no batched call is spent where nothing can be decided (for K=4,
    max 8 the look at n=6 is unreachable and the schedule is n = 4, 8). Returns
    ``(looks, fail_c, pass_d)``, parallel lists. ``k_seeds == 1`` is the uncertified
    ablation: one look at n=1 with c=1, d=0.

    Note: the round-1 certifier used the single-look CP test at full alpha on EVERY
    top-up wave, so its actual overall false-FAIL rate at q=0.5 was ~0.118 for K=8 and
    4 waves (computed exactly by ``scripts/bench_mincause.py --oc``), above its nominal 0.1.
    """
    k = max(1, int(k_seeds))
    max_n = max(1, int(max_n))
    if k <= 1:
        return [1], [1], [0]
    h = max(1, k // 2)
    grid = list(range(h, max_n + 1, h))
    if not grid or grid[-1] != max_n:
        grid.append(max_n)
    state = {0: 1.0}
    prev, cum_f, cum_p = 0, 0.0, 0.0
    looks, cs, ds = [], [], []
    for n in grid:
        step = _binom_pmf_row(n - prev, bar)
        dist = {}
        for kk, pk in state.items():
            for j, pj in enumerate(step):
                dist[kk + j] = dist.get(kk + j, 0.0) + pk * pj
        c, tail = n + 1, 0.0
        for cc in range(n, -1, -1):
            t2 = tail + dist.get(cc, 0.0)
            if cum_f + t2 > alpha + 1e-12:
                break
            c, tail = cc, t2
        d, head = -1, 0.0
        for dd in range(0, c):
            h2 = head + dist.get(dd, 0.0)
            if cum_p + h2 > alpha + 1e-12:
                break
            d, head = dd, h2
        cum_f += tail
        cum_p += head
        state = {kk: pk for kk, pk in dist.items() if d < kk < c}
        prev = n
        # A look is kept only if some still-undecided path can actually stop there (nonzero
        # stopping probability; the binomial support is the same for every 0 < q < 1).
        if tail > 0.0 or head > 0.0 or n == max_n:
            looks.append(n)
            cs.append(c)
            ds.append(d)
    return looks, cs, ds


def operating_characteristics(looks, cs, ds, q):
    """Exact (P(FAIL), P(PASS), P(undecided), E[seeds]) of a schedule at true rate q."""
    state = {0: 1.0}
    prev = 0
    pf = pp = en = 0.0
    for n, c, d in zip(looks, cs, ds):
        step = _binom_pmf_row(n - prev, q)
        dist = {}
        for kk, pk in state.items():
            for j, pj in enumerate(step):
                dist[kk + j] = dist.get(kk + j, 0.0) + pk * pj
        state = {}
        for kk, pk in dist.items():
            if kk >= c:
                pf += pk
                en += pk * n
            elif kk <= d:
                pp += pk
                en += pk * n
            else:
                state[kk] = pk
        prev = n
    und = sum(state.values())
    return pf, pp, und, en + und * prev


class _Certifier:
    """Scores batches of configurations through ONE fail_fn call per round, adding seeds
    to a configuration only while its Clopper-Pearson interval still straddles the bar
    (early-decisive sequential certification with exact group-sequential boundaries; see
    ``look_schedule``). Not claimed as novel.

    ``sequential=False`` restores the round-1 behavior (K seeds per wave, full alpha at
    every wave); it exists only for the benchmark comparison.
    """

    def __init__(self, fail_fn, k_seeds=8, max_seed_mult=4, alpha=ALPHA, seed0=0,
                 sequential=True, on_call=None):
        self.fail_fn = fail_fn
        self.k = max(1, int(k_seeds))
        self.max_n = self.k * (max(1, int(max_seed_mult)) if self.k > 1 else 1)
        self.alpha = alpha
        self.sequential = bool(sequential)
        if self.sequential:
            self.looks, self.fail_c, self.pass_d = look_schedule(self.k, self.max_n, alpha)
        else:
            self.looks = list(range(self.k, self.max_n + 1, self.k))
            self.fail_c = self.pass_d = None
        self._seed = int(seed0)
        self.sims = 0
        self.calls = 0
        self.phase = ""
        self.on_call = on_call

    def _run(self, thetas, reps):
        thetas = np.asarray(thetas, float)
        rows = np.repeat(thetas, reps, axis=0)
        seeds = np.arange(self._seed, self._seed + len(rows), dtype=np.int64)
        self._seed += len(rows)
        out = np.asarray(self.fail_fn(rows, seeds), dtype=bool).reshape(len(thetas), reps)
        self.sims += len(rows)
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.phase, len(rows), self.sims)
        return out.sum(axis=1)

    def _decide(self, k, n, look):
        if self.k <= 1:
            return "fail" if k >= 1 else "pass"
        if self.fail_c is None:                 # round-1 fixed-K: full alpha every wave
            return decide(k, n, self.k, self.alpha)
        if k >= self.fail_c[look]:
            return "fail"
        if k <= self.pass_d[look]:
            return "pass"
        return "undecided"

    def certify(self, thetas):
        """Returns (decisions list[str], k array, n array) for each row of ``thetas``."""
        thetas = np.asarray(thetas, float)
        m = len(thetas)
        if m == 0:
            return [], np.zeros(0, int), np.zeros(0, int)
        k = np.zeros(m, dtype=int)
        n = np.zeros(m, dtype=int)
        dec = ["undecided"] * m
        open_ = list(range(m))
        for look, target in enumerate(self.looks):
            if not open_:
                break
            reps = target - (self.looks[look - 1] if look else 0)
            k_more = self._run(thetas[open_], reps)
            for j, i in enumerate(open_):
                k[i] += int(k_more[j])
                n[i] += reps
                dec[i] = self._decide(k[i], n[i], look)
            open_ = [i for i in open_ if dec[i] == "undecided"]
        return dec, k, n


# --------------------------------------------------------------------------- #
# Pure core
# --------------------------------------------------------------------------- #
@dataclass
class CauseResult:
    certified: bool                 # theta* (restricted to C) certified as failing
    cause: list                     # 1-minimal S as dim indices, ordered by distance desc
    necessary: bool | None          # S reverted (others kept) no longer fails; None = undecided
    theta_min: np.ndarray | None    # minimal counterexample (S shrunk, rest nominal)
    min_distance: float | None      # normalized L-inf distance of theta_min from nominal
    directions: list = field(default_factory=list)   # "high"/"low" per entry of cause
    k: int = 0                      # failures observed at theta_min certification
    n: int = 0                      # seeds used at theta_min certification
    cp_lo: float = 0.0              # CP lower bound at theta_min
    rep_k: int = 0                  # failures observed at the representative
    rep_n: int = 0
    sims: int = 0                   # total rollouts spent
    calls: int = 0                  # total fail_fn calls (GPU batches)
    rounds: int = 0                 # ddmin rounds
    shrunk: bool = True             # False if the jointly-shrunk point failed to certify
    final_certified: bool = True    # False if even the unshrunk S did not re-certify (flaky)


def _config(nominal, theta, subset):
    x = nominal.copy()
    if subset:
        idx = list(subset)
        x[idx] = theta[idx]
    return x


def _chunks(seq, n):
    n = max(1, min(n, len(seq)))
    return [list(c) for c in np.array_split(np.asarray(seq, dtype=int), n) if len(c)]


def certified_minimal_cause(theta_star, nominal, lower, upper, fail_fn, k_seeds=8,
                            max_seed_mult=4, alpha=ALPHA, shrink_steps=SHRINK_STEPS,
                            cand_frac=CAND_FRAC, seed0=0, max_rounds=64, sequential=True,
                            progress=None):
    """Certified minimal cause of the failure at ``theta_star`` (see module docstring).

    ``progress(phase, rows, cumulative_rollouts)`` is called after every batched
    ``fail_fn`` call (console progress in the pipeline). ``sequential=False`` is the
    round-1 fixed-K certifier, kept for the benchmark comparison only.
    """
    theta = np.asarray(theta_star, float)
    nominal = np.asarray(nominal, float)
    span = np.maximum(np.asarray(upper, float) - np.asarray(lower, float), 1e-12)
    cert = _Certifier(fail_fn, k_seeds, max_seed_mult, alpha, seed0,
                      sequential=sequential, on_call=progress)

    dev = np.abs(theta - nominal) / span
    C = [int(i) for i in np.nonzero(dev > cand_frac)[0]]
    C_all = [int(i) for i in np.nonzero(dev > 0)[0]]

    # Is theta* (and theta* restricted to C) a certified failure at all?
    cert.phase = "representative"
    dec, kk, nn = cert.certify(np.stack([theta, _config(nominal, theta, C)]))
    res = CauseResult(certified=False, cause=[], necessary=None, theta_min=None,
                      min_distance=None, rep_k=int(kk[0]), rep_n=int(nn[0]))
    if dec[0] != "fail":
        res.sims, res.calls = cert.sims, cert.calls
        return res
    S = C if dec[1] == "fail" else C_all
    res.certified = True

    # (ii) batched ddmin toward nominal
    gran = 2
    rounds = 0
    while S and rounds < max_rounds:
        rounds += 1
        cert.phase = f"ddmin round {rounds} (|S|={len(S)})"
        cands, seen = [], set()

        def _add(sub):
            key = tuple(sorted(sub))
            if len(key) < len(S) and key not in seen:
                seen.add(key)
                cands.append(list(key))

        for d in S:
            _add([x for x in S if x != d])
        for g in sorted({gran, 2}):
            for ch in _chunks(S, g):
                _add(ch)
                _add([x for x in S if x not in ch])
        if not cands:
            break
        dec, kk, nn = cert.certify(np.stack([_config(nominal, theta, c) for c in cands]))
        failing = [(len(c), -kk[i] / nn[i], i) for i, c in enumerate(cands) if dec[i] == "fail"]
        if failing:
            failing.sort()
            S = cands[failing[0][2]]
            gran = max(gran - 1, 2)
        else:
            # Every single-dim reversion of S was in this batch and none certified as still
            # failing, so S is 1-minimal. (Finer chunking cannot beat that test.)
            break
    res.rounds = rounds

    # (iii) necessity: revert S only, keep every other dim at theta*
    x_nec = theta.copy()
    if S:
        x_nec[S] = nominal[S]
    # (iv) magnitude shrink by batched bisection between nominal (safe) and theta* (fail).
    # Every step bisects all dims of S at once, each relative to the current point ``cand``.
    # Combining several individually-certified moves is SPECULATIVE (interactions, flaky
    # seeds), so the combined point is re-certified inside the next step's batch; if it
    # does not certify, that step is rolled back and only the single largest move, which
    # was certified on its own relative to a verified point, is kept. ``verified`` is thus
    # always a certified failing point and theta_min never reports an uncertified value.
    lo_v = nominal.copy()
    hi_v = theta.copy()
    verified = _config(nominal, theta, S)
    cand, cand_checked = verified.copy(), True
    fallback = None                            # (point, lo, hi) of the safe single move
    rows = [x_nec]
    dec_nec = None
    for step in range(int(shrink_steps) if S else 0):
        cert.phase = f"shrink step {step + 1}/{int(shrink_steps)}"
        mids = []
        for s in S:
            x = cand.copy()
            x[s] = 0.5 * (lo_v[s] + hi_v[s])
            mids.append(x)
        extra = rows + ([] if cand_checked else [cand])
        dec, kk, nn = cert.certify(np.stack(mids + extra))
        if rows:
            dec_nec = dec[len(mids)]
            rows = []
        if not cand_checked:
            if dec[-1] != "fail":              # speculative combination broke: roll back
                cand, lo_v, hi_v = fallback
                verified, cand_checked = cand.copy(), True
                continue
            verified = cand.copy()
        snap = (verified.copy(), lo_v.copy(), hi_v.copy())
        moved = []
        for j, s in enumerate(S):
            mid = mids[j][s]
            if dec[j] == "fail":
                moved.append((abs(hi_v[s] - mid) / span[s], j, s, mid))
                hi_v[s] = mid
            else:
                lo_v[s] = mid
        cand = _config(nominal, hi_v, S)
        cand_checked = len(moved) <= 1         # zero/one move: already certified as a row
        if cand_checked:
            verified = cand.copy()
        else:
            _, j, s, mid = max(moved)
            f_lo, f_hi = snap[1].copy(), snap[2].copy()
            f_hi[s] = mid
            for j2, s2 in enumerate(S):        # certified "pass" rows are valid vs verified
                if dec[j2] != "fail":
                    f_lo[s2] = mids[j2][s2]
            fallback = (mids[j].copy(), f_lo, f_hi)
    if rows:  # no shrink steps ran (empty S or shrink_steps == 0)
        cert.phase = "necessity"
        dec, _, _ = cert.certify(np.stack(rows))
        dec_nec = dec[0]
    res.necessary = {"pass": True, "fail": False}.get(dec_nec) if S else None

    theta_min = cand
    cert.phase = "final check"
    dec, kk, nn = cert.certify(theta_min[None, :])
    if dec[0] != "fail" and not cand_checked:  # last speculative combination: fall back
        theta_min = fallback[0]
        dec, kk, nn = cert.certify(theta_min[None, :])
    if dec[0] != "fail":                       # flaky re-check: report the unshrunk S
        theta_min = _config(nominal, theta, S)
        dec, kk, nn = cert.certify(theta_min[None, :])
    res.shrunk = bool(S) and not np.allclose(theta_min, _config(nominal, theta, S))
    res.final_certified = dec[0] == "fail"
    res.k, res.n = int(kk[0]), int(nn[0])
    res.cp_lo = cp_lower(res.k, res.n, alpha)
    dist = np.abs(theta_min - nominal) / span
    order = sorted(S, key=lambda s: -dist[s])
    res.cause = [int(s) for s in order]
    res.directions = ["high" if theta_min[s] > nominal[s] else "low" for s in order]
    res.theta_min = theta_min
    res.min_distance = float(dist[order].max()) if order else 0.0
    res.sims, res.calls = cert.sims, cert.calls
    return res


# --------------------------------------------------------------------------- #
# Rendering + family integration (pure)
# --------------------------------------------------------------------------- #
def check_label_consistency(res, names, nominal, displayed):
    """Assert every clause direction agrees with theta_min AND with the displayed
    representative values (``displayed``: {name: value}). Raises AssertionError."""
    for s, d in zip(res.cause, res.directions):
        want = "high" if res.theta_min[s] > nominal[s] else "low"
        assert d == want, f"{names[s]}: label says {d}, theta_min says {want}"
        shown = float(displayed[names[s]])
        disp = "high" if shown > nominal[s] else "low"
        assert d == disp, f"{names[s]}: label says {d}, displayed value {shown:.4g} is {disp}"


def margin_string(res, names, integer_dims=INTEGER_DIMS):
    if not res.certified:
        return (f"minimal cause not certified: representative failed {res.rep_k}/"
                f"{res.rep_n} seeds (CP lower {cp_lower(res.rep_k, res.rep_n):.2f} < 0.5)")
    if not res.cause:
        return (f"fails at nominal (no perturbation needed); certified {res.k}/{res.n} "
                f"seeds (CP lower {res.cp_lo:.2f})")
    clauses = []
    for s, d in zip(res.cause, res.directions):
        op = ">=" if d == "high" else "<="
        v = effective_threshold(names[s], res.theta_min[s], integer_dims)
        clauses.append(f"{names[s]} {op} {v}" if isinstance(v, int)
                       else f"{names[s]} {op} {v:.3g}")
    nec = {True: "yes", False: "no", None: "undecided"}[res.necessary]
    extra = "" if res.shrunk or not res.cause else " (magnitudes not shrunk)"
    cert = (f"certified {res.k}/{res.n} seeds (CP lower {res.cp_lo:.2f})"
            if res.final_certified else
            f"final re-check failed only {res.k}/{res.n} seeds, NOT re-certified (flaky)")
    return (f"fails when {' AND '.join(clauses)}; {cert}; necessary: {nec}; distance from "
            f"nominal {res.min_distance:.2f} of span{extra}")


def cause_key(res):
    if not res.certified:
        return None
    return tuple(sorted(zip(res.cause, res.directions)))


def merge_families(families, keys, records_by_id=None):
    """Merge families whose certified cause key coincides (None keys never merge).

    The kept family is the one with the best (lowest) priority_rank; member ids are
    concatenated, mean scores are member-weighted, and member records' family_id is
    repointed when ``records_by_id`` is given. Returns the new list (priority order).
    """
    groups, order = {}, []
    for f, k in zip(families, keys):
        gk = ("solo", f.id) if k is None else k
        if gk not in groups:
            groups[gk] = []
            order.append(gk)
        groups[gk].append(f)
    out = []
    for gk in order:
        fs = sorted(groups[gk], key=lambda f: f.priority_rank)
        keep = fs[0]
        if len(fs) > 1:
            sizes = np.array([len(f.member_ids) for f in fs], float)
            w = sizes / max(sizes.sum(), 1.0)
            keep.mean_failure_score = float(sum(wi * f.mean_failure_score for wi, f in zip(w, fs)))
            keep.mean_transfer_score = float(
                sum(wi * f.mean_transfer_score for wi, f in zip(w, fs)))
            if all(f.mean_theta for f in fs):
                keep.mean_theta = {n: float(sum(wi * f.mean_theta[n] for wi, f in zip(w, fs)))
                                   for n in keep.mean_theta}
            keep.member_ids = [m for f in fs for m in f.member_ids]
            keep.merged_from = [f.id for f in fs]
            if records_by_id is not None:
                for m in keep.member_ids:
                    r = records_by_id.get(m)
                    if r is not None:
                        r.family_id = keep.id
        out.append(keep)
    out.sort(key=lambda f: f.priority_rank)
    return out


def _dedupe_names(families):
    from collections import Counter
    counts, seen = Counter(f.name for f in families), {}
    for f in families:
        if counts[f.name] > 1:
            seen[f.name] = seen.get(f.name, 0) + 1
            f.name = f"{f.name} ({seen[f.name]})"


class _CapReached(Exception):
    """Raised inside the counted fail_fn when the next batch would exceed the rollout cap."""


def _fmt_s(sec):
    sec = max(0.0, float(sec))
    return f"{sec:.0f}s" if sec < 90 else f"{sec / 60:.1f}min"


class _Progress:
    """One flushed console line per batched fail_fn call: family i/n, phase, rows,
    cumulative rollouts, elapsed, and an ETA from the measured per-rollout rate."""

    def __init__(self, n_families, cap, prior_per_family, log, clock):
        self.n, self.cap, self.prior = max(1, n_families), cap, prior_per_family
        self.log, self.clock = log, clock
        self.t0 = clock()
        self.total = 0
        self.done_sims = []            # rollouts spent by each finished family
        self.fam_i, self.fam_start = 0, 0
        self.first = None                  # (elapsed, rollouts) after the first call

    def start_family(self, i):
        self.fam_i, self.fam_start = i, self.total

    def end_family(self):
        self.done_sims.append(self.total - self.fam_start)

    def expected_total(self):
        # A family that stops early (uncertified representative) says little about the
        # next one, so the per-family estimate never drops below the prior.
        per = max(float(self.prior), float(np.mean(self.done_sims)) if self.done_sims else 0.0)
        cur = self.total - self.fam_start
        left = max(self.n - len(self.done_sims) - 1, 0)
        est = self.total + max(per - cur, 0.1 * per) + per * left
        return min(est, self.cap) if self.cap else est

    def __call__(self, phase, rows, fam_sims):
        self.total = self.fam_start + fam_sims
        now = self.clock()
        el = now - self.t0
        if self.first is None:                 # first call carries the JIT compile
            self.first = (el, self.total)
            eta_s = "pending (first batch includes JIT compile)"
        else:
            rate = (el - self.first[0]) / max(self.total - self.first[1], 1)
            eta_s = "~" + _fmt_s(rate * max(self.expected_total() - self.total, 0.0))
        cap = f"/{self.cap}" if self.cap else ""
        if self.log:
            self.log(f"  CMC family {self.fam_i + 1}/{self.n} | {phase} | {rows} rows | "
                     f"{self.total}{cap} rollouts | {_fmt_s(el)} elapsed | ETA {eta_s}")


def apply_mincause(families, records_by_id, space, fail_fn, k_seeds=8, max_seed_mult=4,
                   seed0=0, log=print, top_n=None, rollout_cap=None, budget_label=None,
                   clock=None, shrink_steps=SHRINK_STEPS):
    """Run CMC on each family representative, rename/annotate, merge. Returns new list.

    ``top_n``: only the ``top_n`` families by ``priority_rank`` get CMC. ``rollout_cap``: a
    HARD total over all families (never exceeded); a family whose next batch would cross
    it is abandoned mid-way. Families that do not get CMC keep their clustering names and a
    ``margin`` saying why (``CMC skipped at <budget> budget``); they never merge.
    """
    import time

    from breakpoint_mjx.naming import label_from_cause

    names = space.names()
    idims = _integer_dims(space)
    nominal = np.asarray(space.nominal_vector(), float)
    lower, upper = np.asarray(space.lower(), float), np.asarray(space.upper(), float)
    ranked = sorted(range(len(families)), key=lambda j: families[j].priority_rank)
    chosen = set(ranked if top_n is None else ranked[:max(0, int(top_n))])
    tag = f"{budget_label} budget" if budget_label else "this budget"
    used = {"n": 0}

    def counted(X, seeds):
        if rollout_cap is not None and used["n"] + len(X) > rollout_cap:
            raise _CapReached()
        used["n"] += len(X)
        return fail_fn(X, seeds)

    prior = (rollout_cap / max(len(chosen), 1)) if rollout_cap else 100 * max(1, k_seeds)
    prog = _Progress(len(chosen), rollout_cap, prior, log, clock or time.monotonic)
    keys = []
    done = 0
    for i, f in enumerate(families):
        rep = records_by_id.get(f.representative_id)
        if i not in chosen:
            f.margin = (f"CMC skipped at {tag} (only the top {len(chosen)} families by "
                        f"priority are certified; clustering name kept)")
            keys.append(None)
            if log:
                log(f"  CMC {f.id}: skipped at {tag} (not in the top {len(chosen)})")
            continue
        if rep is None:
            keys.append(None)
            continue
        prog.start_family(done)
        try:
            res = certified_minimal_cause(rep.theta_vector, nominal, lower, upper, counted,
                                          k_seeds=k_seeds, max_seed_mult=max_seed_mult,
                                          seed0=seed0 + 1_000_003 * (i + 1), progress=prog,
                                          shrink_steps=shrink_steps)
        except _CapReached:
            prog.end_family()
            done += 1
            f.margin = (f"CMC skipped at {tag} (rollout cap {rollout_cap} reached; "
                        f"clustering name kept)")
            keys.append(None)
            if log:
                log(f"  CMC {f.id}: stopped, rollout cap {rollout_cap} reached "
                    f"({used['n']} used); clustering name kept")
            continue
        prog.end_family()
        done += 1
        f.cause_certified = bool(res.certified)
        f.margin = margin_string(res, names, idims)
        if res.certified:
            check_label_consistency(res, names, nominal, f.triggering_parameters)
            f.minimal_cause = [
                {"dim": names[s], "value": float(res.theta_min[s]), "direction": d}
                for s, d in zip(res.cause, res.directions)
            ]
            f.min_distance = res.min_distance
            f.cause_necessary = res.necessary
            f.name = label_from_cause([(names[s], d) for s, d in zip(res.cause, res.directions)])
        if log:
            log(f"  CMC {f.id}: {f.margin}  [{res.sims} rollouts, {res.calls} batches]")
        keys.append(cause_key(res))
    merged = merge_families(families, keys, records_by_id)
    _dedupe_names(merged)
    return merged


# --------------------------------------------------------------------------- #
# GPU adapter (Colab-run; jax deferred to call time)
# --------------------------------------------------------------------------- #
def default_max_batch(environ=None, backend=None):
    """Rows per evaluate call: ``BP_MINCAUSE_MAXBATCH`` if set, else 256 on CPU and 1024 on
    an accelerator (``jax.default_backend()``); 64 on CPU at ``BP_BUDGET=smoke``. The CPU
    cap bounds the peak memory of one batched MJX rollout on a laptop."""
    import os

    env = os.environ if environ is None else environ
    v = env.get("BP_MINCAUSE_MAXBATCH")
    if v not in (None, ""):
        n = int(v)
        if n <= 0:
            raise ValueError(f"BP_MINCAUSE_MAXBATCH must be a positive integer, got {v!r}")
        return n
    if backend is None:
        try:
            import jax
            backend = jax.default_backend()
        except Exception:
            backend = "cpu"
    if str(backend).lower() != "cpu":
        return 1024
    # smoke on CPU: one padded shape (64, the search's own batch) so CMC pays one compile
    smoke = str(env.get("BP_BUDGET", "")).strip().lower() == "smoke"
    return 64 if smoke else 256


def bucket_sizes(max_batch):
    """The FIXED padded batch shapes (at most 3: max/4, max/2, max, none below 64 unless
    max is), so the jitted evaluator compiles at most three shapes over a CMC run."""
    mb = max(1, int(max_batch))
    floor = min(64, mb)                       # never pad below 64 rows unless max is smaller
    return sorted({max(floor, mb // 4), max(floor, mb // 2), mb})


def _bucket(n, sizes):
    return next(b for b in sizes if b >= n)


def make_evaluator_fail_fn(evaluate, space, thresholds, horizon, failure_threshold, base_key,
                           objective=None, max_batch=None, log=None):
    """Wrap the pipeline ``evaluate(thetas, key)`` as ``fail_fn(thetas, seeds) -> bool``.

    Row i of an evaluate call is rolled out under fold_in(batch_key, i), so K replicated rows
    already see K different noise keys. The batch key is fold_in(base_key, seeds[start]);
    the certifier hands out fresh, increasing seeds per call, so every call is new noise and
    the whole CMC run is reproducible from ``base_key``. Rows are split into chunks of at
    most ``max_batch`` (``default_max_batch()`` when None) and each chunk is padded (with
    nominal) to one of ``bucket_sizes(max_batch)``, so the jit compiles at most three
    shapes. The first use of each shape is logged (a CPU compile can take tens of seconds);
    ``log=None`` prints with flush, ``log=False`` is silent.
    """
    from breakpoint_mjx.search import scores_from_trajectories

    if log is None:
        log = _flush_print

    nominal = np.asarray(space.nominal_vector(), np.float32)
    max_batch = default_max_batch() if max_batch is None else int(max_batch)
    sizes = bucket_sizes(max_batch)
    seen_shapes = set()

    def fail_fn(thetas, seeds):
        import jax
        import jax.numpy as jnp

        thetas = np.asarray(thetas, np.float32)
        out = np.zeros(len(thetas), bool)
        for start in range(0, len(thetas), max_batch):
            chunk = thetas[start:start + max_batch]
            b = _bucket(len(chunk), sizes)
            if b not in seen_shapes:
                seen_shapes.add(b)
                if log:
                    log(f"  CMC: first batch of {b} rows (may include a one-time JIT compile)")
            pad = np.tile(nominal, (b - len(chunk), 1))
            batch = np.concatenate([chunk, pad], axis=0) if len(pad) else chunk
            key = jax.random.fold_in(base_key, int(seeds[start]) & 0x7FFFFFFF)
            traj = evaluate(jnp.asarray(batch), key)
            scores, _, _ = scores_from_trajectories(traj, thresholds, horizon, objective)
            out[start:start + len(chunk)] = (
                np.asarray(scores, float)[:len(chunk)] >= float(failure_threshold))
        return out

    return fail_fn


# --------------------------------------------------------------------------- #
# Pipeline stage (env-gated; ON by default)
# --------------------------------------------------------------------------- #
# CMC size per BP_BUDGET preset (cli._budget_env exports BP_BUDGET). smoke is sized for a
# CPU laptop: K=4, top-2 families and a hard rollout cap, so the stage finishes in minutes
# instead of the 21-25 min an uncapped K=8 run took on a Windows CPU box in round 1.
# BP_MINCAUSE_K / _MAXMULT / _TOP / _CAP override any preset field ("all" = no limit).
CMC_PRESETS = {
    "smoke": {"k": 4, "mult": 2, "top": 2, "cap": 600, "shrink": 3},
    "standard": {"k": 8, "mult": 3, "top": 4, "cap": None, "shrink": SHRINK_STEPS},
    "full": {"k": 8, "mult": 4, "top": None, "cap": None, "shrink": SHRINK_STEPS},
}


def stage_settings(environ=None):
    """(budget_name, {k, mult, top, cap}) from BP_BUDGET + the BP_MINCAUSE_* overrides."""
    import os

    env = os.environ if environ is None else environ
    name = (env.get("BP_BUDGET") or "full").strip().lower()
    cfg = dict(CMC_PRESETS.get(name, CMC_PRESETS["full"]))
    for var, key in (("BP_MINCAUSE_K", "k"), ("BP_MINCAUSE_MAXMULT", "mult"),
                     ("BP_MINCAUSE_TOP", "top"), ("BP_MINCAUSE_CAP", "cap")):
        v = env.get(var)
        if v in (None, ""):
            continue
        if key in ("top", "cap") and str(v).strip().lower() in ("all", "none"):
            cfg[key] = None
        else:
            n = int(v)
            if n <= 0:
                raise ValueError(f"{var} must be a positive integer or 'all', got {v!r}")
            cfg[key] = n
    return name, cfg


def _flush_print(msg):
    print(msg, flush=True)


def run_stage(families, records_by_id, space, make_fail_fn, environ=None, log=print,
              famrisk_out=None):
    """The pipeline's CMC stage. ON by default; ``BP_MINCAUSE=0`` disables it.

    ``make_fail_fn()`` is called lazily (only when the stage runs) so the disabled path never
    builds the GPU adapter. Size comes from ``BP_BUDGET`` via ``CMC_PRESETS``; the knobs
    ``BP_MINCAUSE_K`` (seeds per configuration), ``BP_MINCAUSE_MAXMULT`` (pooled cap as a
    multiple of K), ``BP_MINCAUSE_TOP`` (families certified, by priority) and
    ``BP_MINCAUSE_CAP`` (hard total rollouts) override it. Every line is flushed, which
    also drains anything the pipeline printed before. Never fails a run: on any exception
    the ORIGINAL families are returned untouched (the stage works on a deep copy) and the
    error is logged. Returns ``(families, ran: bool)``.

    After CMC, the per-family operational-risk stage (``famrisk.run_stage``, OFF by default,
    ``BP_FAMRISK=1`` enables) sets ``odd_risk`` on every certified family using the SAME
    fail_fn; its summary is written into ``famrisk_out`` (a dict) when one is passed. That
    stage has its own guard, so an error there never discards the CMC result.
    """
    def _note(summary):
        if famrisk_out is not None:
            famrisk_out.clear()
            famrisk_out.update(summary)

    import copy
    import os

    env = os.environ if environ is None else environ
    if log is print:
        log = _flush_print
    if str(env.get("BP_MINCAUSE", "1")).strip().lower() in ("0", "false", "off", "no"):
        return families, False
    if not families:
        return families, False
    try:
        budget, st = stage_settings(env)
    except ValueError as e:
        if log:
            log(f"CMC stage skipped ({e}); keeping clustering names")
        return families, False
    k, mult, top, cap = st["k"], st["mult"], st["top"], st["cap"]
    shrink = st.get("shrink", SHRINK_STEPS)
    if log:
        n_run = len(families) if top is None else min(top, len(families))
        log(f"certified minimal cause (CMC), budget={budget}: K={k} seeds/config in "
            f"sequential rounds, pooled up to {k * mult}; {n_run}/{len(families)} families"
            + (f"; hard cap {cap} rollouts" if cap else "") + " ...")
    try:
        work = copy.deepcopy(families)
        fail_fn = make_fail_fn()
        out = apply_mincause(work, records_by_id, space, fail_fn, k_seeds=k,
                             max_seed_mult=mult, log=log, top_n=top, rollout_cap=cap,
                             shrink_steps=shrink,
                             budget_label=budget)
    except Exception as e:  # never fail the run over CMC
        if log:
            log(f"CMC stage skipped ({type(e).__name__}: {e}); keeping clustering names")
        _note({"status": "cmc_not_run", "reason": f"CMC failed: {type(e).__name__}"})
        return families, False
    try:
        from breakpoint_mjx.famrisk import run_stage as _famrisk_stage
        _note(_famrisk_stage(out, space, fail_fn, environ=env, log=log))
    except Exception as e:  # noqa: BLE001 -- famrisk.run_stage already guards; belt+braces
        _note({"status": "error", "reason": f"{type(e).__name__}: {e}"})
    return out, True
