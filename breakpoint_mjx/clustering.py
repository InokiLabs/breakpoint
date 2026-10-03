"""Cluster discovered failures into families (KMeans over rollout-signature +
normalized theta), name each family by its dominant deviating dimension, and
build ranked FailureFamily objects. Pure (numpy + scikit-learn)."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from breakpoint_mjx.config import ThetaSpace
from breakpoint_mjx.transfer import priority_key
from breakpoint_mjx.schemas import FailureFamily
# Discriminative naming/ranking lives in the sklearn-free ``naming`` module so lightweight
# consumers (regression) can reuse it without importing scikit-learn. Re-exported here so
# clustering's public surface (and existing imports) are unchanged.
from breakpoint_mjx.naming import (  # noqa: F401
    _LABELS,
    _label,
    _ranked_dims,
    _reference,
    family_mean_theta,
)


def _features(records, space: ThetaSpace):
    sigs = np.array([r.rollout_signature for r in records], dtype=float)
    thetas = np.array([r.theta_vector for r in records], dtype=float)
    lo = np.array(space.lower()); hi = np.array(space.upper())
    norm_theta = (thetas - lo) / (hi - lo + 1e-9)
    return np.concatenate([sigs, norm_theta], axis=1)


def cluster_failures(records, space: ThetaSpace, k: int):
    feats = StandardScaler().fit_transform(_features(records, space))
    k = max(1, min(k, len(feats)))
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feats)


def name_family(member_records, space: ThetaSpace, all_records=None) -> str:
    """Name by the dim whose family-mean deviates most from a reference (the global
    failure mean if `all_records` given, else nominal), with direction."""
    ranked = _ranked_dims(member_records, _reference(all_records, space), space)
    if not ranked:
        return "mixed failure"
    _, name, direction = ranked[0]
    return _label(name, direction)


def build_families(records, space: ThetaSpace, k: int):
    """Rank all records by priority, cluster into k families, return
    (families sorted by priority, records with priority_rank + family_id set).
    Names are made unique: when two families share a primary label, the secondary
    distinguishing parameter is appended."""
    from collections import Counter
    ranked = sorted(
        records,
        key=lambda r: priority_key(r.severity, r.transfer_score or 0.0, r.failure_score),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        r.priority_rank = i
    labels = cluster_failures(records, space, k)
    ref = _reference(records, space)

    groups = []  # (lab, members, ranked_dims)
    for lab in sorted(set(int(x) for x in labels)):
        members = [r for r, l in zip(records, labels) if int(l) == lab]
        groups.append((lab, members, _ranked_dims(members, ref, space)))

    def _label_at_depth(rd, depth):
        if not rd:
            return "mixed failure"
        return " + ".join(_label(rd[i][1], rd[i][2]) for i in range(min(depth, len(rd))))

    # Deepen each family's label (primary, +secondary, +tertiary) until names are unique,
    # but CAP the concatenation at MAX_NAME_DEPTH clauses -- an active/boundary search finds
    # families that overlap on many axes, and without a cap the disambiguation loop grows
    # unreadable 6-way "A + B + C + D + E + F" names. Past the cap, the numeric suffix below
    # (e.g. "latency gait collapse (2)") guarantees uniqueness instead.
    MAX_NAME_DEPTH = 3
    depth = [1] * len(groups)
    for _ in range(len(space.names())):
        names = [_label_at_depth(rd, depth[i]) for i, (_, _, rd) in enumerate(groups)]
        dups = {n for n, c in Counter(names).items() if c > 1}
        if not dups:
            break
        progressed = False
        for i, (_, _, rd) in enumerate(groups):
            if names[i] in dups and depth[i] < min(len(rd), MAX_NAME_DEPTH):
                depth[i] += 1
                progressed = True
        if not progressed:  # every duplicate is already at the cap -> numeric suffix handles it
            break
    names = [_label_at_depth(rd, depth[i]) for i, (_, _, rd) in enumerate(groups)]
    seen, final_counts = {}, Counter(names)
    for i in range(len(names)):
        if final_counts[names[i]] > 1:
            seen[names[i]] = seen.get(names[i], 0) + 1
            names[i] = f"{names[i]} ({seen[names[i]]})"

    families = []
    for (lab, members, rd), nm in zip(groups, names):
        rep = min(members, key=lambda r: r.priority_rank)  # rank 1 = highest priority
        fid = f"fam-{lab}"
        for m in members:
            m.family_id = fid
        families.append(FailureFamily(
            id=fid,
            name=nm,
            member_ids=[m.id for m in members],
            representative_id=rep.id,
            triggering_parameters=rep.theta_named,
            mean_failure_score=float(np.mean([m.failure_score for m in members])),
            mean_transfer_score=float(np.mean([m.transfer_score or 0.0 for m in members])),
            priority_rank=min(m.priority_rank for m in members),
            # T3 regression fingerprinting: persist the family-MEAN discriminative signature
            # (the exact ranked dominant dims the displayed NAME is derived from, `rd`) and
            # the family mean theta, so regression.family_fingerprint keys off the family
            # mean -- matching the name and independent of the representative-row jitter.
            dominant_dims=[[float(dev), n, d] for dev, n, d in rd],
            mean_theta=family_mean_theta(members, space),
        ))
    families.sort(key=lambda f: f.priority_rank)
    return families, ranked
