"""Workstream 2a (T3): failure REGRESSION testing -- "unit tests for robustness".

BreakPoint discovers failure families of a frozen policy. This module turns those
families into a persistent, per-policy-version FAILURE ATLAS and DIFFS versions, so a
new failure family introduced by checkpoint N is caught like a failing test (spec
section "2a. Failure regression testing (the wedge)").

The hard part is IDENTITY across runs. KMeans cluster indices (``fam-<k>``) are not
stable -- the same failure mode can land in cluster 0 one run and cluster 2 the next,
so raw ids cannot be diffed. Instead each family gets a STABLE *fingerprint* derived
from its discriminative signature -- the family-mean dominant deviating ``(dim,
direction)`` pairs (persisted by ``clustering.build_families`` as ``dominant_dims``, the
exact quantity ``name_family`` labels on) plus a coarse centroid of the family mean
theta. Two runs that rediscover the same failure mode produce matching-or-similar
fingerprints, so ``match_families`` / ``diff_atlases`` can line them up across checkpoints.

Matching is deliberately biased AGAINST the dangerous false-negative (a genuinely
worsened family slipping through the CI gate as "resolved + new-noncritical"): it matches
on the PRIMARY dominant axis plus signature-set overlap and a centroid check, uses a
GLOBAL-greedy (order-independent) assignment, does a second lenient pass to catch a
family that drifted while intensifying, and a final ``diff_atlases`` safety net re-emits
any NEW family that shares a RESOLVED family's primary failure axis as WORSENED.

Everything here is PURE -- jax-free AND sklearn-free (numpy only, via ``naming``): the
fingerprints, matching, the diff, the JSON persistence, and the CI gate. So the whole
regression workflow (and the CI ``--families`` path) is dependency-light; the *search* it
wraps (scripts/regression.py --run) is the only Colab/GPU piece.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from breakpoint_mjx.config import ThetaSpace
from breakpoint_mjx.naming import _label  # sklearn-free label helper
from breakpoint_mjx.schemas import FailureFamily
from breakpoint_mjx.transfer import SEVERITY_RANK, severity_from_score

# --------------------------------------------------------------------------- #
# Tunables (all overridable per call / per CLI flag)
# --------------------------------------------------------------------------- #
# Number of dominant deviating dims that make up a family's fingerprint SIGNATURE.
DEFAULT_N_DIMS = 2
# A dim only enters the signature if its family-mean deviates from its reference by at
# least this fraction of its bound span -- keeps near-reference (noise) axes out.
DEFAULT_DEV_FLOOR = 0.05
# Tight (identity) match: two fingerprints must be within this RMS normalized-centroid
# distance over their shared dominant axes (~= "average per-axis deviation" there).
DEFAULT_TOL = 0.15
# Tight match also requires signature-set Jaccard overlap >= this (so gaining/losing ONE
# secondary dominant dim does not break identity), on top of primary-axis agreement.
DEFAULT_JACCARD = 0.5
# Lenient (drift-catch) pass: after the tight pass, still-unmatched old<->new that agree
# on the PRIMARY (dim, direction) and are within this looser primary-axis distance are
# matched, so a family that INTENSIFIED and drifted is worsened, never resolved+new.
DEFAULT_LENIENT_TOL = 0.30
# WORSENED is declared when a matched family got worse by more than this epsilon on any
# monitored quantity (guards against float noise flipping unchanged->worsened).
DEFAULT_WORSEN_EPS = 1e-6

# CI-gate thresholds (the numbers are configurable; the gate itself is on by default).
DEFAULT_CRITICAL_SEVERITY = "high"       # a NEW family at or above this severity fails CI
DEFAULT_CRITICAL_VALIDITY = 0.8          # ...or a NEW family with measured validity >= this
DEFAULT_WORSEN_FAILURE_DELTA = 0.10      # matched family whose mean_failure_score rose > this
DEFAULT_WORSEN_VALIDITY_DELTA = 0.10     # ...or whose measured validity dropped > this

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Fingerprint (stable, cluster-index-independent family identity)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilyFingerprint:
    """Stable identity of a failure family, independent of its ``fam-<k>`` cluster index.

    signature : the dominant deviating ``(dim, direction)`` pairs (top-N by normalized
                deviation), the discriminative quantity clustering names on. Signature[0]
                is the PRIMARY failure axis -- the anchor identity survives across runs.
    centroid  : the family (mean) theta as a bound-normalized [0,1] vector, in ``dims``
                order (nominal fallback for any missing dim). Used for tolerant matching.
    dims      : the space dim names aligned to ``centroid``.
    """

    signature: tuple[tuple[str, str], ...]
    centroid: tuple[float, ...]
    dims: tuple[str, ...]

    @property
    def primary(self):
        """The primary dominant ``(dim, direction)`` (the anchor axis), or None if mixed."""
        return self.signature[0] if self.signature else None

    @property
    def label(self) -> str:
        """Human-readable label (reuses clustering's direction-aware labels)."""
        return " + ".join(_label(d, dr) for d, dr in self.signature) or "mixed failure"

    @property
    def key(self) -> str:
        """A compact, stable string id: the signature dims/directions + a COARSE bucket of
        each signature dim's centroid, so mild-vs-severe on the same axis differ but small
        run-to-run churn does not. Suitable as a dict key / log line."""
        if not self.signature:
            return "mixed"
        idx = {n: i for i, n in enumerate(self.dims)}
        parts = []
        for d, dr in self.signature:
            v = self.centroid[idx[d]] if d in idx else 0.0
            parts.append(f"{d}:{dr}:{_bucket(v)}")
        return "|".join(parts)

    def __str__(self) -> str:  # so ``str(fingerprint)`` == its key (spec allows a str API)
        return self.key


def _bucket(v: float, n_levels: int = 5) -> int:
    """Coarse-quantize a [0,1] value into ``n_levels`` buckets (stable under small churn)."""
    v = min(max(float(v), 0.0), 1.0)
    return min(int(v * n_levels), n_levels - 1)


def _deviating_dims(theta_named: dict, space: ThetaSpace):
    """[(dev, name, direction), ...] sorted by descending normalized deviation from NOMINAL.

    Fallback path when a family carries no persisted ``dominant_dims`` (e.g. hand-built or
    pre-T3). Mirrors ``naming._ranked_dims`` (same span-normalized measure) against the
    space nominal; missing dims are skipped.
    """
    out = []
    for name, dim in space.dims.items():
        if name not in theta_named:
            continue
        span = max(dim.bounds[1] - dim.bounds[0], 1e-9)
        val = float(theta_named[name])
        dev = abs(val - dim.nominal) / span
        out.append((dev, name, "high" if val > dim.nominal else "low"))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def _signature_from_dominant_dims(dominant_dims, dev_floor, n_dims):
    """Signature from a persisted ``dominant_dims`` ([[dev, name, direction], ...]), robust
    to JSON round-trip (lists) and re-sorted defensively by deviation descending."""
    ranked = sorted(
        ((float(d[0]), d[1], d[2]) for d in dominant_dims), key=lambda t: t[0], reverse=True
    )
    return tuple((n, dr) for dev, n, dr in ranked if dev >= dev_floor)[: max(0, int(n_dims))]


def family_fingerprint(
    family: FailureFamily,
    space: ThetaSpace,
    n_dims: int = DEFAULT_N_DIMS,
    dev_floor: float = DEFAULT_DEV_FLOOR,
) -> FamilyFingerprint:
    """Stable fingerprint for ``family`` from its discriminative signature + centroid.

    Prefers the family-MEAN quantities persisted by ``clustering.build_families``
    (``dominant_dims`` for the signature -- matching the displayed name -- and ``mean_theta``
    for the centroid), so the fingerprint reflects the family, not the jittery
    representative row. Falls back to ``triggering_parameters`` (vs nominal) when those are
    absent (pre-T3 or hand-built families).
    """
    dominant_dims = getattr(family, "dominant_dims", None)
    if dominant_dims:
        signature = _signature_from_dominant_dims(dominant_dims, dev_floor, n_dims)
    else:
        dev = _deviating_dims(family.triggering_parameters or {}, space)
        signature = tuple(
            (name, direction) for d, name, direction in dev if d >= dev_floor
        )[: max(0, int(n_dims))]

    src = getattr(family, "mean_theta", None) or (family.triggering_parameters or {})
    names = space.names()
    lo, hi = space.lower(), space.upper()
    centroid = []
    for i, name in enumerate(names):
        val = float(src.get(name, space.dims[name].nominal))
        centroid.append((val - lo[i]) / (hi[i] - lo[i] + 1e-9))
    return FamilyFingerprint(
        signature=signature, centroid=tuple(centroid), dims=tuple(names)
    )


def _union_sig_dims(a: FamilyFingerprint, b: FamilyFingerprint):
    """Ordered union of the dims appearing in either fingerprint's signature."""
    seen = []
    for d, _ in a.signature:
        if d not in seen:
            seen.append(d)
    for d, _ in b.signature:
        if d not in seen:
            seen.append(d)
    return seen


def _centroid_distance(a: FamilyFingerprint, b: FamilyFingerprint, dims=None) -> float:
    """RMS distance between two normalized centroids over ``dims`` (default: all shared)."""
    ia = {n: i for i, n in enumerate(a.dims)}
    ib = {n: i for i, n in enumerate(b.dims)}
    use = [n for n in (dims or []) if n in ia and n in ib]
    if not use:  # no (shared) signature dims -> fall back to all shared dims
        use = [n for n in a.dims if n in ib]
    if not use:
        return float("inf")
    s = sum((a.centroid[ia[n]] - b.centroid[ib[n]]) ** 2 for n in use)
    return math.sqrt(s / len(use))


def _jaccard(sa: set, sb: set) -> float:
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def _tight_match(a, b, tol, jaccard_thresh) -> bool:
    """Identity match: same PRIMARY dominant axis, signature-set Jaccard >= threshold, and
    centroid within ``tol`` over the shared dominant axes."""
    pa, pb = a.primary, b.primary
    if pa is None or pb is None:
        # Two "mixed" (no-dominant-dim) families: match only if both mixed and close.
        return pa is None and pb is None and _centroid_distance(a, b) <= tol
    if pa != pb:
        return False
    if _jaccard(set(a.signature), set(b.signature)) < jaccard_thresh:
        return False
    return _centroid_distance(a, b, dims=_union_sig_dims(a, b)) <= tol


def _lenient_match(a, b, lenient_tol) -> bool:
    """Drift-catch match: same PRIMARY dominant axis and within a looser distance ON THAT
    axis -- pulls back a family that intensified and drifted before it is mislabeled
    resolved+new."""
    pa, pb = a.primary, b.primary
    if pa is None or pb is None or pa != pb:
        return False
    return _centroid_distance(a, b, dims=[pa[0]]) <= lenient_tol


def fingerprints_match(
    a: FamilyFingerprint,
    b: FamilyFingerprint,
    tol: float = DEFAULT_TOL,
    jaccard_thresh: float = DEFAULT_JACCARD,
) -> bool:
    """True iff ``a`` and ``b`` are the SAME family under the tight (identity) criterion."""
    return _tight_match(a, b, tol, jaccard_thresh)


# --------------------------------------------------------------------------- #
# Matching families across two atlases (global-greedy, order-independent)
# --------------------------------------------------------------------------- #
def _candidate_pairs(old_fp, new_fp, predicate, dist_fn):
    cands = []
    for i, (_of, ofp) in enumerate(old_fp):
        for j, (_nf, nfp) in enumerate(new_fp):
            if predicate(ofp, nfp):
                cands.append((dist_fn(ofp, nfp), i, j))
    cands.sort(key=lambda t: (t[0], t[1], t[2]))  # (distance, old_index, new_index)
    return cands


def _assign(cands, used_old, used_new):
    """Global-greedy assignment: take candidate pairs in (distance, i, j) order, skipping
    already-claimed rows/cols. Deterministic and independent of input family order."""
    out = []
    for _d, i, j in cands:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        out.append((i, j))
    return out


def match_families(
    old_families,
    new_families,
    space: ThetaSpace,
    tol: float = DEFAULT_TOL,
    n_dims: int = DEFAULT_N_DIMS,
    dev_floor: float = DEFAULT_DEV_FLOOR,
    jaccard_thresh: float = DEFAULT_JACCARD,
    lenient_tol: float = DEFAULT_LENIENT_TOL,
):
    """Pair old vs new families by fingerprint, biased against false-negative "resolved".

    Returns ``(old|None, new|None)`` tuples: ``(old, new)`` for a match, ``(old, None)``
    for an old-only family (candidate RESOLVED), ``(None, new)`` for a new-only family
    (candidate NEW).

    Two passes, both GLOBAL-greedy (assign the globally closest valid pair first, so the
    result is independent of family order and one old family cannot "steal" the match a
    closer pair needs): (1) tight identity match; (2) a lenient primary-axis pass over
    whatever is still unmatched, to catch an intensifying family that drifted.
    """
    old_fp = [(f, family_fingerprint(f, space, n_dims, dev_floor)) for f in old_families]
    new_fp = [(f, family_fingerprint(f, space, n_dims, dev_floor)) for f in new_families]

    used_old, used_new = set(), set()
    tight = _candidate_pairs(
        old_fp, new_fp,
        predicate=lambda a, b: _tight_match(a, b, tol, jaccard_thresh),
        dist_fn=lambda a, b: _centroid_distance(a, b, dims=_union_sig_dims(a, b)),
    )
    matched = _assign(tight, used_old, used_new)
    lenient = _candidate_pairs(
        old_fp, new_fp,
        predicate=lambda a, b: _lenient_match(a, b, lenient_tol),
        dist_fn=lambda a, b: _centroid_distance(a, b, dims=[a.primary[0]]),
    )
    matched += _assign(lenient, used_old, used_new)

    pairs = [(old_fp[i][0], new_fp[j][0]) for i, j in matched]
    pairs += [(old_fp[i][0], None) for i in range(len(old_fp)) if i not in used_old]
    pairs += [(None, new_fp[j][0]) for j in range(len(new_fp)) if j not in used_new]
    return pairs


# --------------------------------------------------------------------------- #
# Diff (new / resolved / worsened / unchanged with deltas)
# --------------------------------------------------------------------------- #
def family_severity(f: FailureFamily) -> str:
    """A family's severity bucket, derived from its mean failure score (families do not
    carry a severity field; the score is the family-level analogue of a record's)."""
    return severity_from_score(f.mean_failure_score)


def _severity_rank(f: FailureFamily) -> int:
    return SEVERITY_RANK[family_severity(f)]


def _family_summary(f: FailureFamily, space: ThetaSpace) -> dict:
    return {
        "family": f,
        "id": f.id,
        "name": f.name,
        "fingerprint": family_fingerprint(f, space).key,
        "severity": family_severity(f),
        "mean_failure_score": f.mean_failure_score,
        "mean_transfer_score": f.mean_transfer_score,
        "validity": f.validity,
        "members": len(f.member_ids),
    }


def _matched_entry(old, new, space, worsen_eps, extra_reason=None) -> dict:
    d_fail = new.mean_failure_score - old.mean_failure_score
    d_transfer = new.mean_transfer_score - old.mean_transfer_score
    d_sev = _severity_rank(new) - _severity_rank(old)
    d_validity = None
    if old.validity is not None and new.validity is not None:
        d_validity = new.validity - old.validity
    d_members = len(new.member_ids) - len(old.member_ids)

    reasons = []
    if d_sev > 0:
        reasons.append(f"severity {family_severity(old)} -> {family_severity(new)}")
    if d_fail > worsen_eps:
        reasons.append(f"mean_failure_score +{d_fail:.3f}")
    if d_validity is not None and d_validity < -worsen_eps:
        reasons.append(f"validity {d_validity:+.3f}")
    if extra_reason and reasons:
        reasons.append(extra_reason)

    return {
        "old": old,
        "new": new,
        "old_id": old.id,
        "new_id": new.id,
        "name": new.name,
        "old_name": old.name,
        "fingerprint": family_fingerprint(new, space).key,
        "worsened": bool(reasons),
        "reasons": reasons,
        "deltas": {
            "mean_failure_score": d_fail,
            "mean_transfer_score": d_transfer,
            "severity_rank": d_sev,
            "old_severity": family_severity(old),
            "new_severity": family_severity(new),
            "validity": d_validity,
            "members": d_members,
        },
    }


def _reconcile_primary_drift(out: dict, space: ThetaSpace, worsen_eps: float):
    """Safety net against the dangerous resolved+new false negative.

    Any NEW family that shares its PRIMARY dominant ``(dim, direction)`` with a RESOLVED
    family is really the same failure axis re-measured (it drifted past the tolerant match),
    NOT an unrelated pair. Re-emit it as a matched entry (WORSENED when the deltas show it
    got worse, UNCHANGED otherwise) so an intensifying-and-drifting family can never sail
    through as "resolved + new-noncritical".
    """
    fp_cache = {}

    def _fp(f):
        if id(f) not in fp_cache:
            fp_cache[id(f)] = family_fingerprint(f, space)
        return fp_cache[id(f)]

    for new_entry in list(out["new"]):
        nf = new_entry["family"]
        n_primary = _fp(nf).primary
        if n_primary is None:
            continue
        best, best_d = None, None
        for res_entry in out["resolved"]:
            of = res_entry["family"]
            if _fp(of).primary != n_primary:
                continue
            d = _centroid_distance(_fp(of), _fp(nf), dims=[n_primary[0]])
            if best_d is None or d < best_d:
                best, best_d = res_entry, d
        if best is not None:
            out["new"].remove(new_entry)
            out["resolved"].remove(best)
            entry = _matched_entry(
                best["family"], nf, space, worsen_eps,
                extra_reason="fingerprint drift on shared primary axis",
            )
            out["worsened" if entry["worsened"] else "unchanged"].append(entry)


def diff_atlases(
    old_families,
    new_families,
    space: ThetaSpace,
    tol: float = DEFAULT_TOL,
    worsen_eps: float = DEFAULT_WORSEN_EPS,
) -> dict:
    """Diff two atlases into ``{"new", "resolved", "worsened", "unchanged"}``.

    * NEW       -- a family in ``new`` with no fingerprint match in ``old``.
    * RESOLVED  -- a family in ``old`` with no fingerprint match in ``new``.
    * WORSENED  -- a matched family that got worse: higher severity, higher failure score,
                   or (when both sides measured it) lower validity. Carries the deltas.
    * UNCHANGED -- a matched family that did not get worse. Also carries the deltas.

    A final safety net (``_reconcile_primary_drift``) reclassifies any NEW family that
    shares a RESOLVED family's primary failure axis as a matched (worsened/unchanged) pair.
    """
    pairs = match_families(old_families, new_families, space, tol=tol)
    out = {"new": [], "resolved": [], "worsened": [], "unchanged": []}
    for old, new in pairs:
        if old is None:
            out["new"].append(_family_summary(new, space))
        elif new is None:
            out["resolved"].append(_family_summary(old, space))
        else:
            entry = _matched_entry(old, new, space, worsen_eps)
            out["worsened" if entry["worsened"] else "unchanged"].append(entry)
    _reconcile_primary_drift(out, space, worsen_eps)
    return out


def format_diff(diff: dict, baseline_version=None, version=None) -> str:
    """Render a diff as a readable multi-line string for CI logs."""
    header = "BreakPoint regression diff"
    if version is not None:
        header += f"  version={version}"
    if baseline_version is not None:
        header += f"  baseline={baseline_version}"
    lines = [
        header,
        f"  NEW: {len(diff['new'])}   RESOLVED: {len(diff['resolved'])}   "
        f"WORSENED: {len(diff['worsened'])}   UNCHANGED: {len(diff['unchanged'])}",
    ]
    for e in diff["new"]:
        v = f", validity {e['validity']:.2f}" if e["validity"] is not None else ""
        lines.append(
            f"  [NEW]       {e['name']}  (severity {e['severity']}, "
            f"score {e['mean_failure_score']:.3f}{v})  fp={e['fingerprint']}"
        )
    for e in diff["resolved"]:
        lines.append(
            f"  [RESOLVED]  {e['name']}  (was severity {e['severity']}, "
            f"score {e['mean_failure_score']:.3f})"
        )
    for e in diff["worsened"]:
        lines.append(f"  [WORSENED]  {e['name']}  ({'; '.join(e['reasons'])})")
    for e in diff["unchanged"]:
        d = e["deltas"]
        lines.append(
            f"  [UNCHANGED] {e['name']}  (dscore {d['mean_failure_score']:+.3f})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CI gate (exit-code logic)
# --------------------------------------------------------------------------- #
def evaluate_gate(
    diff: dict,
    *,
    critical_severity: str = DEFAULT_CRITICAL_SEVERITY,
    critical_validity: float = DEFAULT_CRITICAL_VALIDITY,
    worsen_failure_delta: float = DEFAULT_WORSEN_FAILURE_DELTA,
    worsen_validity_delta: float = DEFAULT_WORSEN_VALIDITY_DELTA,
    fail_on_new_critical: bool = True,
    fail_on_worsened: bool = True,
):
    """CI gate: return ``(exit_code, violations)``.

    Non-zero (fails CI) when a NEW family is "critical" -- at/above ``critical_severity``,
    or (when measured) validity >= ``critical_validity`` -- OR an existing family WORSENED
    past a threshold (mean_failure_score up > ``worsen_failure_delta``, validity down >
    ``worsen_validity_delta``, or its severity rose to at/above ``critical_severity``).

    Both gate halves are toggleable; the numeric thresholds are all configurable.
    """
    violations = []
    crit_rank = SEVERITY_RANK[critical_severity]

    if fail_on_new_critical:
        for e in diff.get("new", []):
            sev, val = e["severity"], e.get("validity")
            if SEVERITY_RANK[sev] >= crit_rank:
                violations.append(
                    f"NEW critical family '{e['name']}' (severity {sev}, "
                    f"score {e['mean_failure_score']:.3f})"
                )
            elif val is not None and val >= critical_validity:
                violations.append(
                    f"NEW high-validity family '{e['name']}' "
                    f"(validity {val:.2f} >= {critical_validity:.2f})"
                )

    if fail_on_worsened:
        for e in diff.get("worsened", []):
            d = e["deltas"]
            if d["mean_failure_score"] > worsen_failure_delta:
                violations.append(
                    f"WORSENED '{e['name']}': mean_failure_score "
                    f"+{d['mean_failure_score']:.3f} > {worsen_failure_delta:.3f}"
                )
            if d["validity"] is not None and (-d["validity"]) > worsen_validity_delta:
                violations.append(
                    f"WORSENED '{e['name']}': validity {d['validity']:+.3f} "
                    f"(drop > {worsen_validity_delta:.3f})"
                )
            if d["severity_rank"] > 0 and SEVERITY_RANK[d["new_severity"]] >= crit_rank:
                violations.append(
                    f"WORSENED '{e['name']}': severity {d['old_severity']} -> "
                    f"{d['new_severity']}"
                )

    return (1 if violations else 0), violations


# --------------------------------------------------------------------------- #
# Broken-search guard (Nice): a silently-empty/collapsed atlas must not read green
# --------------------------------------------------------------------------- #
def broken_search_warnings(
    new_families, old_families, *, drop_fraction: float = 0.5
) -> list[str]:
    """Loud warnings when a NEW atlas looks like a broken search rather than a clean policy.

    * 0 families while a baseline HAD families -> almost certainly a broken/misconfigured
      search (an env crash, a bad checkpoint path), not a suddenly-flawless policy.
    * a sharp family-count drop (> ``drop_fraction``) vs the baseline -> suspicious.

    Returned strings are surfaced by the CLI; the empty-atlas case is escalated to a gate
    violation there so it fails CI instead of passing as "all resolved".
    """
    warnings = []
    n_new, n_old = len(new_families), len(old_families)
    if n_new == 0 and n_old > 0:
        warnings.append(
            f"new atlas has 0 families but the baseline had {n_old} -- likely a BROKEN "
            f"search (crash / bad checkpoint), not a fixed policy"
        )
    elif n_old > 0 and n_new < n_old * (1.0 - drop_fraction):
        warnings.append(
            f"new atlas family count dropped sharply ({n_old} -> {n_new}) -- verify the "
            f"search ran fully"
        )
    return warnings


# --------------------------------------------------------------------------- #
# Persistence: per-version atlas store under <BP_OUT>/<target>/atlas_history/
# --------------------------------------------------------------------------- #
_SAFE_VERSION = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_version(version) -> str:
    """Make a version id filesystem-safe (checkpoint names can contain slashes etc.)."""
    v = _SAFE_VERSION.sub("_", str(version).strip())
    return v or "unknown"


def _now_iso() -> str:
    """Wall-clock timestamp for the meta blob ONLY (human display). Version identity and
    baseline ordering never depend on this -- they use the explicit id + stored sequence."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_atlas(families, meta, path) -> str:
    """Write an atlas (families + a meta/version blob) to ``path`` as JSON."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "meta": dict(meta or {}),
        "families": [asdict(f) for f in families],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def load_atlas(path):
    """Load an atlas JSON back into ``(families, meta)``."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    families = [FailureFamily(**f) for f in payload.get("families", [])]
    meta = payload.get("meta", {}) or {}
    return families, meta


def atlas_history_dir(bp_out, target) -> str:
    """The per-target version store: ``<bp_out>/<target>/atlas_history``."""
    return os.path.join(bp_out, target, "atlas_history")


def version_path(history_dir, version) -> str:
    """Path of one version's atlas file inside ``history_dir``."""
    return os.path.join(history_dir, f"{_sanitize_version(version)}.json")


def list_versions(history_dir):
    """List stored versions under ``history_dir``, oldest -> newest.

    Ordering is by the stored ``meta.sequence`` (assigned at save time, so it never
    depends on wall-clock), falling back to file mtime only for atlases that predate the
    sequence field. Each entry: ``{version, path, sequence, mtime, gate_passed}``.
    """
    if not os.path.isdir(history_dir):
        return []
    out = []
    for fn in sorted(os.listdir(history_dir)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(history_dir, fn)
        try:
            with open(p, encoding="utf-8") as fh:
                payload = json.load(fh)
            meta = payload.get("meta", {}) or {}
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "version": meta.get("version") or fn[:-5],
                "path": p,
                "sequence": meta.get("sequence"),
                "mtime": os.path.getmtime(p),
                "gate_passed": meta.get("gate_passed"),
            }
        )
    out.sort(key=lambda d: (d["sequence"] if d["sequence"] is not None else -1.0, d["mtime"]))
    return out


def latest_version(history_dir, exclude=None, green_only=False):
    """The most recent stored version (by sequence), or None.

    ``exclude`` (a version id) is skipped -- used so a run does not pick itself as its own
    baseline. ``green_only`` restricts to versions whose stored ``gate_passed`` is True
    (baseline-poisoning guard: prefer the last version that actually PASSED the gate).
    """
    ex = _sanitize_version(exclude) if exclude is not None else None
    for d in reversed(list_versions(history_dir)):
        if ex is not None and _sanitize_version(d["version"]) == ex:
            continue
        if green_only and not d.get("gate_passed"):
            continue
        return d
    return None


def resolve_baseline(history_dir, exclude=None):
    """Pick the default baseline: the most recent GREEN (gate_passed) version, falling back
    to the most recent version of any status if none is green. Avoids baseline poisoning --
    a regressed checkpoint does not become the accepted baseline, so a sustained regression
    keeps alarming instead of alarming once. Returns a version entry dict or None."""
    return latest_version(history_dir, exclude=exclude, green_only=True) or latest_version(
        history_dir, exclude=exclude
    )


def save_version(history_dir, version, families, meta=None) -> str:
    """Save ``families`` as ``version`` under ``history_dir``, stamping a stable sequence.

    The sequence is ``1 + max(existing sequences)`` so newer versions sort after older
    ones WITHOUT relying on wall-clock. Re-saving an existing version keeps its original
    sequence (so a re-run of an old checkpoint does not jump to "latest").
    """
    os.makedirs(history_dir, exist_ok=True)
    existing = list_versions(history_dir)
    key = _sanitize_version(version)
    by_ver = {_sanitize_version(d["version"]): d for d in existing}
    if key in by_ver and by_ver[key]["sequence"] is not None:
        seq = by_ver[key]["sequence"]
    else:
        seqs = [d["sequence"] for d in existing if d["sequence"] is not None]
        seq = (max(seqs) + 1) if seqs else (len(existing) + 1)

    full_meta = dict(meta or {})
    full_meta.setdefault("version", str(version))
    full_meta["sequence"] = seq
    full_meta.setdefault("saved_at", _now_iso())
    return save_atlas(families, full_meta, version_path(history_dir, version))


def load_version(history_dir, version):
    """Load one stored version's ``(families, meta)`` by id."""
    return load_atlas(version_path(history_dir, version))
