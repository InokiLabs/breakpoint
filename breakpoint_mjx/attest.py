"""Signed, verifiable BreakPoint reports (ed25519).

A report (see the shared contract `breakpoint.report/v1`) is signed over its
CANONICAL JSON: UTF-8, keys sorted at every level, separators (",", ":"), with the
top-level `attestation` field removed. The signature covers those exact bytes, and
`payload_sha256` is the sha256 of the same bytes, so anyone holding the signer's
public key can check that the report was produced by that signer and not edited,
without ever seeing the policy weights.

    attestation = {"alg": "ed25519", "key_id": "<16 hex>", "signer": "self | inoki",
                   "payload_sha256": "<hex>", "signature": "<base64>"}

Key files are PEM: the private key is unencrypted PKCS8, the public key is
SubjectPublicKeyInfo (both readable by `openssl pkey`). `key_id` is the first 16 hex
characters of sha256 over the raw 32-byte public key.

The crypto backend is the `cryptography` package, an optional extra
(`pip install "breakpoint-mjx[attest]"`). It is imported lazily so the rest of the
CLI works without it; any signing call without it raises `AttestError` with the
install command.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ALG = "ed25519"
SIGNERS = ("self", "inoki")
KEY_ID_HEX = 16
INSTALL_HINT = ('signing needs the `cryptography` package: '
                'pip install "breakpoint-mjx[attest]"  (or: pip install cryptography)')


class AttestError(Exception):
    """A signing / key-handling problem the CLI reports without a traceback."""


def _crypto():
    """Return the ed25519 + serialization modules, or raise a clear AttestError."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise AttestError(INSTALL_HINT) from e
    return ed25519, serialization, InvalidSignature


# --------------------------------------------------------------------------
# canonical form
# --------------------------------------------------------------------------

def canonical_bytes(report: dict) -> bytes:
    """The exact bytes that are hashed and signed."""
    if not isinstance(report, dict):
        raise AttestError("a report must be a JSON object")
    body = {k: v for k, v in report.items() if k != "attestation"}
    try:
        text = json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        raise AttestError(f"report is not canonical-JSON serializable: {e}") from e
    return text.encode("utf-8")


def payload_sha256(report: dict) -> str:
    return hashlib.sha256(canonical_bytes(report)).hexdigest()


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------

def _raw_public(pub) -> bytes:
    _, serialization, _ = _crypto()
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(pub) -> str:
    """Short fingerprint of a public key object."""
    return hashlib.sha256(_raw_public(pub)).hexdigest()[:KEY_ID_HEX]


def generate_keypair():
    ed25519, _, _ = _crypto()
    priv = ed25519.Ed25519PrivateKey.generate()
    return priv, priv.public_key()


def private_pem(priv) -> bytes:
    _, s, _ = _crypto()
    return priv.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption())


def public_pem(pub) -> bytes:
    _, s, _ = _crypto()
    return pub.public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo)


def load_private_key(path):
    ed25519, s, _ = _crypto()
    p = Path(path)
    if not p.is_file():
        raise AttestError(f"private key not found: {p}")
    try:
        key = s.load_pem_private_key(p.read_bytes(), password=None)
    except (ValueError, TypeError) as e:
        raise AttestError(f"{p} is not an unencrypted PEM private key") from e
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise AttestError(f"{p} is not an ed25519 private key")
    return key


def load_public_key(path):
    ed25519, s, _ = _crypto()
    p = Path(path)
    if not p.is_file():
        raise AttestError(f"public key not found: {p}")
    try:
        key = s.load_pem_public_key(p.read_bytes())
    except (ValueError, TypeError) as e:
        raise AttestError(f"{p} is not a PEM public key") from e
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise AttestError(f"{p} is not an ed25519 public key")
    return key


def enclosing_repo(path) -> Path | None:
    """The nearest ancestor of `path` holding a `.git` entry (dir, or file for a
    worktree / submodule), or None. Works on paths that do not exist yet."""
    p = Path(path).expanduser().resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists():
            return d
    return None


def _write_exclusive(path: Path, data: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, mode)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    try:  # os.open's mode is masked by umask; on Windows chmod only toggles read-only
        os.chmod(path, mode)
    except OSError:
        pass


def write_keypair(out_dir, name: str = "breakpoint") -> tuple[Path, Path, str]:
    """Generate a keypair into `out_dir` as <name>.key (0600) and <name>.pub.

    Refuses a directory inside any git work tree, and refuses to overwrite."""
    out = Path(out_dir).expanduser().resolve()
    repo = enclosing_repo(out)
    if repo is not None:
        raise AttestError(
            f"refusing to write a private key inside a git work tree ({repo}). "
            "Keep signing keys outside every repository, e.g. ~/.breakpoint/keys")
    if not name or any(c in name for c in "/\\") or name.startswith("."):
        raise AttestError(f"bad key name: {name!r}")
    priv_p, pub_p = out / f"{name}.key", out / f"{name}.pub"
    for p in (priv_p, pub_p):
        if p.exists():
            raise AttestError(f"{p} already exists; refusing to overwrite a key")
    priv, pub = generate_keypair()
    out.mkdir(parents=True, exist_ok=True)
    _write_exclusive(priv_p, private_pem(priv), 0o600)
    _write_exclusive(pub_p, public_pem(pub), 0o644)
    return priv_p, pub_p, key_id(pub)


# --------------------------------------------------------------------------
# sign / verify
# --------------------------------------------------------------------------

def sign_report(report: dict, priv, signer: str = "self") -> dict:
    """Return a copy of `report` with a fresh attestation block.

    Any existing attestation is dropped first (it is never part of the payload),
    so re-signing replaces it rather than nesting it."""
    if signer not in SIGNERS:
        raise AttestError(f"signer must be one of {SIGNERS}, got {signer!r}")
    out = copy.deepcopy(report)
    out.pop("attestation", None)
    msg = canonical_bytes(out)
    out["attestation"] = {
        "alg": ALG,
        "key_id": key_id(priv.public_key()),
        "signer": signer,
        "payload_sha256": hashlib.sha256(msg).hexdigest(),
        "signature": base64.b64encode(priv.sign(msg)).decode("ascii"),
    }
    return out


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    signer: str | None = None
    key_id: str | None = None


def verify_report(report: dict, pub) -> VerifyResult:
    """Check the attestation against `pub`. Never raises for a bad report."""
    _, _, InvalidSignature = _crypto()
    if not isinstance(report, dict):
        return VerifyResult(False, "report is not a JSON object")
    att = report.get("attestation")
    if not isinstance(att, dict):
        return VerifyResult(False, "report is not signed (no attestation block)")
    signer, kid = att.get("signer"), att.get("key_id")
    if att.get("alg") != ALG:
        return VerifyResult(False, f"unsupported alg {att.get('alg')!r}", signer, kid)
    if signer not in SIGNERS:
        return VerifyResult(False, f"unknown signer {signer!r}", signer, kid)
    expected_kid = key_id(pub)
    if kid != expected_kid:
        return VerifyResult(False, f"signed by key {kid!r}, not the given key "
                                   f"{expected_kid!r}", signer, kid)
    try:
        msg = canonical_bytes(report)
    except AttestError as e:
        return VerifyResult(False, str(e), signer, kid)
    if att.get("payload_sha256") != hashlib.sha256(msg).hexdigest():
        return VerifyResult(False, "payload hash mismatch: the report was edited "
                                   "after signing", signer, kid)
    try:
        sig = base64.b64decode(att.get("signature") or "", validate=True)
    except (ValueError, TypeError):
        return VerifyResult(False, "signature is not valid base64", signer, kid)
    try:
        pub.verify(sig, msg)
    except InvalidSignature:
        return VerifyResult(False, "bad signature", signer, kid)
    # The signer label sits in the (unsigned) attestation block; bind it to the signed
    # run.runner field so it cannot be relabelled without invalidating the report.
    run = report.get("run")
    runner = run.get("runner") if isinstance(run, dict) else None
    if runner is not None and runner != signer:
        return VerifyResult(False, f"signer {signer!r} disagrees with the signed "
                                   f"run.runner {runner!r}", signer, kid)
    return VerifyResult(True, f"signed by {signer} key {kid}", signer, kid)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_policy_file(report: dict, policy_path) -> VerifyResult:
    """Does report.policy.sha256 match a local checkpoint file?"""
    p = Path(policy_path)
    if not p.is_file():
        return VerifyResult(False, f"policy file not found: {p}")
    pol = report.get("policy") if isinstance(report, dict) else None
    claimed = pol.get("sha256") if isinstance(pol, dict) else None
    if not claimed:
        return VerifyResult(False, "report has no policy.sha256")
    actual = file_sha256(p)
    if actual != claimed:
        return VerifyResult(False, f"policy sha256 mismatch: report {claimed}, "
                                   f"file {actual}")
    return VerifyResult(True, "policy sha256 matches the local file")


def _no_duplicates(pairs):
    """Refuse duplicate keys. Python keeps the last one, other parsers the first, so a
    report with duplicates could display different content from what was verified."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise AttestError(f"duplicate key {k!r} in report JSON")
        out[k] = v
    return out


def load_report(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise AttestError(f"report not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as e:
        raise AttestError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise AttestError(f"{p} is not a JSON object")
    return data


def save_report(report: dict, path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
