"""Signed, verifiable reports (breakpoint_mjx.attest + `breakpoint keys/sign/verify`).

CPU-pure: no jax, no mujoco. Needs the optional `cryptography` package; the one test
that checks the missing-library message runs regardless.
"""
import builtins
import copy
import json

import pytest

from breakpoint_mjx import attest, cli

try:
    import cryptography  # noqa: F401
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


@pytest.fixture(autouse=True)
def _needs_crypto(request):
    if not HAVE_CRYPTO and request.node.get_closest_marker("no_crypto_needed") is None:
        pytest.skip("attest extra (cryptography) not installed")


def _report():
    return {
        "schema": "breakpoint.report/v1",
        "kind": "rating",
        "battery": {"id": "legged-quadruped-v1", "sha256": "ab" * 32},
        "engine": {"version": "0.1.0", "commit": None},
        "robot": "go1",
        "policy": {"name": "demo", "source": "local", "sha256": "cd" * 32},
        "run": {"date": "2026-10-03T00:00:00Z", "backend": "mjx-cpu",
                "seeds": [0, 1, 2, 3, 4, 5, 6, 7], "runner": "self"},
        "cells": [
            {"id": "nominal", "n": 8, "falls": 0, "fall_rate": 0.0,
             "ci95": [0.0, 0.37], "passed": True},
            {"id": "lag-dropout", "n": 8, "falls": 3, "fall_rate": 0.375,
             "ci95": [0.09, 0.76], "passed": False},
        ],
        "summary": {"cells_passed": 1, "cells_total": 2, "label": "1/2 cells passed"},
        "gate": None,
        "attestation": None,
    }


@pytest.fixture(scope="module")
def keypair():
    if not HAVE_CRYPTO:
        pytest.skip("attest extra (cryptography) not installed")
    return attest.generate_keypair()


@pytest.fixture
def signed(keypair):
    return attest.sign_report(_report(), keypair[0], signer="self")


def test_sign_then_verify_passes(signed, keypair):
    res = attest.verify_report(signed, keypair[1])
    assert res.ok, res.reason
    a = signed["attestation"]
    assert a["alg"] == "ed25519" and a["signer"] == "self"
    assert a["key_id"] == attest.key_id(keypair[1]) and len(a["key_id"]) == 16
    assert a["payload_sha256"] == attest.payload_sha256(signed)


def test_canonical_form_matches_contract():
    r = {"b": 1, "a": {"y": "é", "x": [1, 2]}, "attestation": {"anything": 1}}
    assert attest.canonical_bytes(r) == '{"a":{"x":[1,2],"y":"é"},"b":1}'.encode("utf-8")


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("robot", "spot"),
    lambda r: r["summary"].__setitem__("cells_passed", 2),
    lambda r: r["cells"][1].__setitem__("passed", True),
    lambda r: r["cells"][1]["ci95"].__setitem__(1, 0.5),
    lambda r: r["run"]["seeds"].append(8),
    lambda r: r["policy"].__setitem__("sha256", "ee" * 32),
    lambda r: r.__setitem__("extra", 1),
    lambda r: r.pop("gate"),
    lambda r: r["cells"].reverse(),
], ids=["top", "nested", "list-item", "deep-list", "append", "policy-hash",
        "added-key", "removed-key", "list-order"])
def test_tampering_any_field_fails(signed, keypair, mutate):
    t = copy.deepcopy(signed)
    mutate(t)
    res = attest.verify_report(t, keypair[1])
    assert not res.ok
    assert "edited" in res.reason


def test_tampering_with_hash_recomputed_fails_on_signature(signed, keypair):
    t = copy.deepcopy(signed)
    t["summary"]["label"] = "2/2 cells passed"
    t["attestation"]["payload_sha256"] = attest.payload_sha256(t)
    res = attest.verify_report(t, keypair[1])
    assert not res.ok and res.reason == "bad signature"


def _reorder(obj):
    """Reverse key order at every level (list order is data and is kept)."""
    if isinstance(obj, dict):
        return {k: _reorder(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_reorder(v) for v in obj]
    return obj


def test_reordering_keys_still_verifies(signed, keypair, tmp_path):
    re = _reorder(signed)
    assert list(re) != list(signed)
    assert attest.verify_report(re, keypair[1]).ok
    # and through a file written with different whitespace
    p = tmp_path / "r.json"
    p.write_text(json.dumps(re, indent=4), encoding="utf-8")
    assert attest.verify_report(attest.load_report(p), keypair[1]).ok


def test_wrong_key_fails(signed):
    _, other_pub = attest.generate_keypair()
    res = attest.verify_report(signed, other_pub)
    assert not res.ok and "not the given key" in res.reason


def test_wrong_key_with_forged_key_id_fails_on_signature(signed):
    _, other_pub = attest.generate_keypair()
    t = copy.deepcopy(signed)
    t["attestation"]["key_id"] = attest.key_id(other_pub)
    res = attest.verify_report(t, other_pub)
    assert not res.ok and res.reason == "bad signature"


def test_existing_attestation_is_replaced_not_nested(keypair):
    r = _report()
    r["attestation"] = {"alg": "ed25519", "key_id": "stale", "signer": "self",
                        "payload_sha256": "0" * 64, "signature": "AAAA"}
    s1 = attest.sign_report(r, keypair[0])
    s2 = attest.sign_report(s1, keypair[0])
    for s in (s1, s2):
        assert set(s["attestation"]) == {"alg", "key_id", "signer",
                                         "payload_sha256", "signature"}
        assert "attestation" not in json.dumps(s["attestation"])
        assert attest.verify_report(s, keypair[1]).ok
    assert s1["attestation"]["payload_sha256"] == s2["attestation"]["payload_sha256"]
    assert r["attestation"]["key_id"] == "stale"  # input not mutated


def test_unsigned_report_fails(keypair):
    res = attest.verify_report(_report(), keypair[1])
    assert not res.ok and "not signed" in res.reason


def test_signer_relabel_fails(signed, keypair):
    t = copy.deepcopy(signed)
    t["attestation"]["signer"] = "inoki"
    res = attest.verify_report(t, keypair[1])
    assert not res.ok and "run.runner" in res.reason


def test_bad_signer_rejected_at_sign_time(keypair):
    with pytest.raises(attest.AttestError):
        attest.sign_report(_report(), keypair[0], signer="mallory")


def test_private_key_inside_repo_is_refused(tmp_path):
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(attest.AttestError, match="git work tree"):
        attest.write_keypair(repo / "keys" / "deep")
    assert not (repo / "keys").exists()


def test_private_key_inside_worktree_gitfile_is_refused(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /elsewhere\n")
    with pytest.raises(attest.AttestError, match="git work tree"):
        attest.write_keypair(wt)


def test_this_repo_is_refused():
    from pathlib import Path
    here = Path(attest.__file__).resolve().parent
    if attest.enclosing_repo(here) is None:
        pytest.skip("not running from a git checkout")
    with pytest.raises(attest.AttestError):
        attest.write_keypair(here / "keys")


def test_keypair_roundtrip_and_no_overwrite(tmp_path):
    priv_p, pub_p, kid = attest.write_keypair(tmp_path / "k", "team")
    priv = attest.load_private_key(priv_p)
    pub = attest.load_public_key(pub_p)
    assert attest.key_id(pub) == kid == attest.key_id(priv.public_key())
    with pytest.raises(attest.AttestError, match="already exists"):
        attest.write_keypair(tmp_path / "k", "team")


@pytest.mark.no_crypto_needed
def test_missing_crypto_lib_gives_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("No module named 'cryptography'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(attest.AttestError, match=r"breakpoint-mjx\[attest\]"):
        attest.generate_keypair()
    rc = cli.main(["keys", "new", "--out", "unused-dir-never-created"])
    assert rc == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_keys_sign_verify_end_to_end(tmp_path, capsys):
    kd = tmp_path / "keys"
    assert cli.main(["keys", "new", "--out", str(kd)]) == 0
    out = capsys.readouterr().out
    priv_text = (kd / "breakpoint.key").read_text()
    assert "PRIVATE KEY" in priv_text
    assert priv_text.strip().splitlines()[1] not in out  # private key never printed

    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(_report()), encoding="utf-8")
    assert cli.main(["sign", str(rp), "--key", str(kd / "breakpoint.key")]) == 0
    capsys.readouterr()
    assert cli.main(["verify", str(rp), "--pubkey", str(kd / "breakpoint.pub")]) == 0
    assert capsys.readouterr().out.startswith("PASS")

    data = json.loads(rp.read_text(encoding="utf-8"))
    data["cells"][0]["falls"] = 1
    rp.write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(["verify", str(rp), "--pubkey", str(kd / "breakpoint.pub")]) == 1
    assert capsys.readouterr().out.startswith("FAIL: payload hash mismatch")


def test_cli_verify_policy_flag(tmp_path, capsys, keypair):
    ckpt = tmp_path / "w.pkl"
    ckpt.write_bytes(b"weights")
    r = _report()
    r["policy"]["sha256"] = attest.file_sha256(ckpt)
    kd = tmp_path / "keys"
    priv_p, pub_p, _ = attest.write_keypair(kd)
    rp = tmp_path / "r.json"
    attest.save_report(attest.sign_report(r, attest.load_private_key(priv_p)), rp)
    assert cli.main(["verify", str(rp), "--pubkey", str(pub_p),
                     "--policy", str(ckpt)]) == 0
    assert "policy sha256 matches" in capsys.readouterr().out
    other = tmp_path / "x.pkl"
    other.write_bytes(b"other weights")
    assert cli.main(["verify", str(rp), "--pubkey", str(pub_p),
                     "--policy", str(other)]) == 1
    assert "policy sha256 mismatch" in capsys.readouterr().out


def test_cli_sign_bad_report_is_clean_error(tmp_path, capsys):
    kd = tmp_path / "keys"
    priv_p, _, _ = attest.write_keypair(kd)
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert cli.main(["sign", str(bad), "--key", str(priv_p)]) == 2
    assert "not a JSON object" in capsys.readouterr().err


@pytest.mark.no_crypto_needed
def test_load_report_refuses_duplicate_keys(tmp_path):
    # Python keeps the last duplicate, other parsers the first: refuse, never guess
    p = tmp_path / "dup.json"
    p.write_text('{"kind": "rating", "robot": "go1", "robot": "g1"}', encoding="utf-8")
    with pytest.raises(attest.AttestError, match="duplicate key 'robot'"):
        attest.load_report(p)
