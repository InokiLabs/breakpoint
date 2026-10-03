"""Cross-robot failure corpus: the asset that compounds, and its privacy contract.

Every consented run contributes the ENVIRONMENT CONDITIONS under which a policy
failed, never the policy or the robot. That corpus is what makes each new robot
cheaper to break than the last (the surrogate already opens at R^2 0.83 on the
fourth robot because earlier robots' failures seed it), and it cannot be copied
without re-running the years of search that produced it.

WHAT LEAVES THE MACHINE (the whole list, enforced by ALLOWED_* below):
  * theta_named        the physical conditions (friction, payload, latency, ...)
  * failure_score, component_signals, severity, transfer, validity
  * family fingerprint key
  * robot CLASS (quadruped / humanoid / arm), DOF count, coarse mass bucket
  * engine version, search mode, horizon

WHAT NEVER LEAVES:
  * policy weights or checkpoints, in any form
  * URDF / MJCF / meshes / robot name / model number
  * rollout trajectories or observations
  * file paths, hostnames, org identity, customer name

The redactor BUILDS the payload from an allow-list rather than deleting keys from
the record. That direction matters: a field added to FailureRecord later cannot
silently start shipping, and `test_corpus.py` pins it.

This is NOT the same path as webpush/push_run.py. That sends a customer their OWN
full results (including trajectories) into their OWN account. This is the
anonymous aggregate. Keeping them separate is deliberate; do not merge them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from breakpoint_mjx.tier import config_home

#: bump when the payload shape changes in a way a user should re-consent to.
MANIFEST_VERSION = 1

ALLOWED_RECORD_KEYS = (
    "theta_named", "failure_score", "component_signals",
    "severity", "transfer_score", "validity", "family_id",
)
ALLOWED_META_KEYS = (
    "robot_class", "dof", "mass_bucket", "engine_version",
    "search_mode", "horizon", "manifest_version",
)

ROBOT_CLASSES = ("quadruped", "humanoid", "arm", "wheeled", "other")


def mass_bucket(total_mass_kg) -> str:
    """Coarse size band, so a robot cannot be identified by an exact mass."""
    m = float(total_mass_kg)
    if m < 5:
        return "<5kg"
    if m < 20:
        return "5-20kg"
    if m < 60:
        return "20-60kg"
    return ">60kg"


def redact_record(record) -> dict:
    """Whitelist-build one corpus row from a FailureRecord (or a dict)."""
    get = record.get if isinstance(record, dict) else (lambda k, d=None: getattr(record, k, d))
    out = {}
    for k in ALLOWED_RECORD_KEYS:
        v = get(k, None)
        if v is None:
            continue
        if k == "theta_named":
            out[k] = {str(dk): float(dv) for dk, dv in dict(v).items()}
        elif k == "component_signals":
            out[k] = {str(dk): float(dv) for dk, dv in dict(v).items()}
        elif k in ("failure_score", "transfer_score", "validity"):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def build_payload(records, *, robot_class, dof, total_mass_kg, engine_version,
                  search_mode="unknown", horizon=None) -> dict:
    """The exact object that would be uploaded. Inspect it with `breakpoint corpus preview`."""
    if robot_class not in ROBOT_CLASSES:
        raise ValueError(f"robot_class must be one of {ROBOT_CLASSES}, got {robot_class!r}")
    meta = {
        "robot_class": robot_class,
        "dof": int(dof),
        "mass_bucket": mass_bucket(total_mass_kg),
        "engine_version": str(engine_version),
        "search_mode": str(search_mode),
        "horizon": None if horizon is None else int(horizon),
        "manifest_version": MANIFEST_VERSION,
    }
    extra = set(meta) - set(ALLOWED_META_KEYS)
    if extra:      # belt and braces: the allow-list is the spec, not the dict literal
        raise AssertionError(f"meta contains non-whitelisted keys: {sorted(extra)}")
    return {"meta": meta, "findings": [redact_record(r) for r in records]}


# --------------------------------------------------------------------------
# consent
# --------------------------------------------------------------------------

def consent_path(home=None) -> Path:
    return config_home(home) / "consent.json"


def read_consent(home=None):
    """True / False if the user has decided, None if never asked (or stale manifest)."""
    p = consent_path(home)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if int(d.get("manifest_version", -1)) != MANIFEST_VERSION:
        return None      # the deal changed; ask again rather than assume the old yes
    return bool(d["contribute"]) if "contribute" in d else None


def write_consent(contribute: bool, home=None) -> Path:
    p = consent_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "contribute": bool(contribute),
        "manifest_version": MANIFEST_VERSION,
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }), encoding="utf-8")
    return p


CONSENT_PROMPT = """\
Contribute anonymous failure data to the BreakPoint corpus?

  SENT:     the physical conditions a failure happened under (friction, payload,
            latency, ...), its scores, robot class / DOF / size band.
  NEVER:    your policy, your URDF or meshes, your robot's name, trajectories,
            file paths, or anything identifying you or your organisation.

It makes every future robot faster to test, including yours. You can change this
any time with `breakpoint corpus consent yes|no`, and preview the exact payload
with `breakpoint corpus preview`. Answering no leaves every feature working."""


def outbox_dir(home=None) -> Path:
    return config_home(home) / "outbox"


def stage(payload: dict, home=None, name=None) -> Path:
    """Write a payload to the local outbox.

    Staged, not sent: there is no corpus ingest endpoint yet, and writing a fake
    upload would be worse than writing none. `breakpoint corpus export` hands the
    user the same file so they can see precisely what would go.
    """
    d = outbox_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name or f"corpus-{time.strftime('%Y%m%d-%H%M%S')}.json")
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
