"""Experiment B: re-evaluate selected failure conditions on fresh, independent seeds.

Each candidate theta is run ``n`` times through the same evaluator with rollout keys from
a stream (base ``VALIDATION_KEY_BASE``) that no search method ever uses. A rollout counts
as a failure only if it is a physically valid fall; invalid rollouts are counted and their
reasons recorded but never counted as failures.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from benchmark.analysis.metrics import clopper_pearson, cp_lower_one_sided, is_validated

VALIDATION_KEY_BASE = 9_000_000


def validate(recorder, thetas, n, batch, key_index):
    """-> list of per-candidate dicts, in the order of ``thetas``.

    ``key_index`` must be unique per call (e.g. a run index) so validation streams never
    repeat across runs."""
    import jax

    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    m = thetas.shape[0]
    if m == 0:
        return []
    rows = np.repeat(thetas, n, axis=0)                     # candidate-major
    total = rows.shape[0]
    pad = (-total) % batch
    if pad:
        rows = np.concatenate([rows, np.repeat(rows[-1:], pad, axis=0)])
    recorder.reset()
    for b, s in enumerate(range(0, rows.shape[0], batch)):
        key = jax.random.fold_in(jax.random.PRNGKey(VALIDATION_KEY_BASE + int(key_index)), b)
        recorder(rows[s:s + batch].astype(np.float32), key)
    tab = recorder.table()
    out = []
    for c in range(m):
        sl = slice(c * n, (c + 1) * n)
        fail = tab["failure"][sl]
        inv = tab["invalid"][sl]
        k = int(fail.sum())
        lo, hi = clopper_pearson(k, n)
        reasons = Counter(r for r, v in zip(tab["invalid_reason"][sl], inv) if v > 0.5 and r)
        ttf = tab["ttf"][sl][fail > 0.5]
        out.append({
            "k": k, "n": int(n), "p": k / n, "ci95": [round(lo, 4), round(hi, 4)],
            "cp_lower_one_sided": round(cp_lower_one_sided(k, n), 4),
            "validated": bool(is_validated(k, n)),
            "n_invalid": int(inv.sum()), "invalid_reasons": dict(reasons),
            "n_fall_any": int((tab["fall"][sl] > 0.5).sum()),
            "median_ttf": float(np.median(ttf)) if ttf.size else None,
        })
    return out
