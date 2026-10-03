"""Editions for the installable BreakPoint CLI.

    OPEN SOURCE (Apache-2.0)   FIND and TEST failures: search, report, replay,
                               cross-engine checks, ratings, CI gate, self-signed
                               attestations. Free for any use, no caps.
    COMMERCIAL EDITION         FIX them: hardening / robust retraining, Inoki
                               certification, private runs (no corpus contribution).

Hardening and certification are not part of the open-source engine. This module only
tells a user which edition they are on and where the commercial features live.

ON ENFORCEMENT, HONESTLY: the key check is offline and format-only. It is a pointer
to the commercial edition, not DRM; the commercial code itself is not in the
open-source distribution.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FREE = "free"
OEM = "oem"

#: feature -> tiers permitted to use it.
FEATURES = {
    "search": (FREE, OEM),
    "report": (FREE, OEM),
    "replay": (FREE, OEM),
    "crossengine": (FREE, OEM),
    "harden": (OEM,),
    "certify": (OEM,),
    "private": (OEM,),      # run without contributing to the shared corpus
}

_KEY_PREFIX = "bp_oem_"
_BODY_LEN = 28
_SUM_LEN = 4
_SALT = b"breakpoint-oem-v1"


class LicenseRequired(RuntimeError):
    """Raised when a FREE-tier user invokes a paid feature."""


def config_home(home=None) -> Path:
    """Where the CLI keeps licence + consent. Override with BREAKPOINT_HOME."""
    if home is not None:
        return Path(home)
    env = os.environ.get("BREAKPOINT_HOME")
    return Path(env) if env else Path.home() / ".breakpoint"


def _checksum(body: str) -> str:
    return hashlib.sha256(body.encode() + _SALT).hexdigest()[:_SUM_LEN]


def mint_key(body: str) -> str:
    """Build a well-formed key from a 28-char body (used by tests and by us)."""
    body = body.lower()
    if len(body) != _BODY_LEN or any(c not in "0123456789abcdef" for c in body):
        raise ValueError(f"key body must be {_BODY_LEN} hex chars, got {body!r}")
    return f"{_KEY_PREFIX}{body}{_checksum(body)}"


def key_is_valid(key) -> bool:
    """Format + checksum only. Catches typos, not forgery (see module docstring)."""
    if not isinstance(key, str) or not key.startswith(_KEY_PREFIX):
        return False
    rest = key[len(_KEY_PREFIX):]
    if len(rest) != _BODY_LEN + _SUM_LEN:
        return False
    body, given = rest[:_BODY_LEN], rest[_BODY_LEN:]
    if any(c not in "0123456789abcdef" for c in body + given):
        return False
    return _checksum(body) == given


def license_path(home=None) -> Path:
    return config_home(home) / "license.json"


def save_key(key: str, home=None) -> Path:
    """Persist a licence key. Raises ValueError on a malformed key."""
    if not key_is_valid(key):
        raise ValueError(
            "that licence key is not well formed (check for a typo or a missing "
            "character). Keys look like bp_oem_<32 hex chars>."
        )
    p = license_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"key": key}), encoding="utf-8")
    return p


def load_key(env=None, home=None):
    """Key from BREAKPOINT_LICENSE_KEY, else the saved file, else None."""
    env = os.environ if env is None else env
    key = env.get("BREAKPOINT_LICENSE_KEY")
    if key:
        return key.strip()
    p = license_path(home)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("key")
        except (json.JSONDecodeError, OSError):
            return None      # a corrupt file means FREE, never a crash
    return None


def resolve_tier(env=None, home=None) -> str:
    return OEM if key_is_valid(load_key(env, home)) else FREE


def allows(feature: str, tier: str) -> bool:
    if feature not in FEATURES:
        raise KeyError(f"unknown feature {feature!r}; known: {sorted(FEATURES)}")
    return tier in FEATURES[feature]


def require(feature: str, tier=None, env=None, home=None) -> str:
    """Return the tier, or raise LicenseRequired with an actionable message."""
    tier = tier or resolve_tier(env, home)
    if allows(feature, tier):
        return tier
    raise LicenseRequired(
        f"'{feature}' is part of the BreakPoint commercial edition; you are on the "
        f"open-source edition.\n"
        f"The open-source edition finds and reports failures without limits. Hardening "
        f"a policy and issuing an Inoki certificate are commercial-edition features.\n"
        f"Already have a key?  breakpoint activate <key>\n"
        f"Need one?             https://breakpoint-inoki.vercel.app/#demo"
    )


def describe(tier=None, env=None, home=None) -> dict:
    """Everything `breakpoint license` prints, as data."""
    tier = tier or resolve_tier(env, home)
    return {
        "tier": tier,
        "unlocked": sorted(f for f in FEATURES if allows(f, tier)),
        "locked": sorted(f for f in FEATURES if not allows(f, tier)),
        "usage": ("open-source edition (Apache-2.0)" if tier == FREE
                  else "commercial edition under your BreakPoint licence agreement"),
    }
