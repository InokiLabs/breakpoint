"""Serialize a finished run into the machine-readable ``families.json`` that the web
push CLI (scripts/push_run.py -> BreakPoint Web) consumes.

This is the SAME data ``report.html`` renders, in the ingest wire shape (schema_version
1.0). Pure: no jax, no filesystem, no network -- unit-tested on CPU. Kept in lock-step
with the web app's ``lib/ingest/schema.ts`` zod schema; bump SCHEMA_VERSION on a
breaking change.
"""
from __future__ import annotations

from breakpoint_mjx.stats import severity_probability_rank
from breakpoint_mjx.transfer import severity_from_score

SCHEMA_VERSION = "1.0"


def _family_dict(fam, sev_prob: float) -> dict:
    return {
        "name": fam.name,
        "fingerprint_key": getattr(fam, "fingerprint_key", None),
        "severity": severity_from_score(fam.mean_failure_score),
        "mean_failure_score": float(fam.mean_failure_score),
        "mean_transfer_score": float(fam.mean_transfer_score),
        "member_count": len(fam.member_ids),
        "priority_rank": int(fam.priority_rank),
        "sev_prob_score": float(sev_prob),
        "representative_theta": {k: float(v) for k, v in fam.triggering_parameters.items()},
        "validity": None if fam.validity is None else float(fam.validity),
        "reproduction": fam.reproduction,
        "margin": fam.margin,
        "boundaries": fam.boundaries,
        "attribution": fam.attribution,
        # Certified Minimal Cause (mincause.py); None when the stage did not run.
        "minimal_cause": getattr(fam, "minimal_cause", None),
        "min_distance": getattr(fam, "min_distance", None),
        "cause_certified": getattr(fam, "cause_certified", None),
        "cause_necessary": getattr(fam, "cause_necessary", None),
        # Cause-Stratified Operational Risk (famrisk.py); None when not computed.
        "odd_risk": getattr(fam, "odd_risk", None),
    }


def build_families_json(
    *,
    target,
    space,
    families,
    n_total: int,
    n_failures: int,
    failure_rate_ci,
    search_mode: str,
    horizon: int,
    engine_commit: str | None = None,
    operational_risk: dict | None = None,
) -> dict:
    """Assemble the families.json payload. ``failure_rate_ci`` = (rate, lo, hi).

    ``stats.failure_rate`` is the rate among SEARCH-EVALUATED worlds (adversarially
    selected), labelled by ``stats.failure_rate_basis``. The deployment probability,
    when BP_RISK ran, is the separate ``operational_risk`` block (riskcert.py).
    """
    rate, lo, hi = (float(x) for x in failure_rate_ci)
    ranked = severity_probability_rank(families, n_total=n_total)
    sev_prob_by_name = {r["name"]: r["sev_prob"] for r in ranked}
    severity_primary = sorted(
        (
            {"name": f.name, "mean_failure_score": float(f.mean_failure_score)}
            for f in families
        ),
        key=lambda r: r["mean_failure_score"],
        reverse=True,
    )
    out = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "search_mode": search_mode,
            "horizon": int(horizon),
            "theta_dim": len(space.names()),
            "engine_commit": engine_commit,
            "target": target.name,
        },
        "stats": {
            "n_total": int(n_total),
            "failure_rate": rate,
            "failure_rate_basis": "search_selected",
            "ci_lo": lo,
            "ci_hi": hi,
            "expected_risk_ranking": [
                {"name": r["name"], "sev_prob_score": r["sev_prob"]} for r in ranked
            ],
            "severity_primary_ranking": severity_primary,
        },
        "families": [_family_dict(f, sev_prob_by_name.get(f.name, 0.0)) for f in families],
    }
    if operational_risk is not None:
        out["operational_risk"] = operational_risk
    return out


def engine_commit_short() -> str | None:
    """`git rev-parse --short HEAD`, or None if unavailable (impure; call at the edge)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
