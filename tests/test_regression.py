"""CPU unit tests for T3 failure-regression testing (breakpoint_mjx/regression.py)
and the scripts/regression.py CI entrypoint.

No jax/GPU: fingerprints, matching, the diff, persistence, and the CI gate are all pure.
Families are constructed directly (no search). The go1 theta space is loaded for real
dim names / nominals / bounds so fingerprints exercise the actual normalization.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.schemas import FailureFamily, FailureRecord
from breakpoint_mjx import regression
from breakpoint_mjx.regression import (
    broken_search_warnings,
    diff_atlases,
    evaluate_gate,
    family_fingerprint,
    fingerprints_match,
    load_atlas,
    match_families,
    resolve_baseline,
    save_atlas,
    save_version,
    latest_version,
    load_version,
    list_versions,
)

from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


def _nominal(space):
    return {n: space.dims[n].nominal for n in space.names()}


def _family(space, overrides, *, fid="fam-0", name="x", fail=0.6, transfer=0.5,
            members=3, validity=None, rep=None):
    """Build a FailureFamily whose triggering_parameters = nominal + overrides."""
    tp = _nominal(space)
    tp.update(overrides)
    member_ids = [f"{fid}-m{i}" for i in range(members)]
    return FailureFamily(
        id=fid,
        name=name,
        member_ids=member_ids,
        representative_id=rep or member_ids[0],
        triggering_parameters=tp,
        mean_failure_score=fail,
        mean_transfer_score=transfer,
        priority_rank=1,
        validity=validity,
    )


# --------------------------------------------------------------------------- #
# Fingerprint stability
# --------------------------------------------------------------------------- #
def test_fingerprint_same_family_same_key(space):
    fam = _family(space, {"friction_floor": 0.18}, fid="fam-0")
    fam_reindexed = _family(space, {"friction_floor": 0.18}, fid="fam-3")  # churned cluster idx
    a = family_fingerprint(fam, space)
    b = family_fingerprint(fam_reindexed, space)
    # Identity is independent of the fam-<k> id.
    assert a.key == b.key
    assert a.signature == b.signature
    assert a.signature[0] == ("friction_floor", "low")


def test_fingerprint_different_dominant_dim_differs(space):
    slip = _family(space, {"friction_floor": 0.18})
    latency = _family(space, {"latency_steps": 3.5})
    a = family_fingerprint(slip, space)
    b = family_fingerprint(latency, space)
    assert a.key != b.key
    assert set(a.signature) != set(b.signature)
    assert not fingerprints_match(a, b)


def test_fingerprint_similar_family_matches(space):
    a = family_fingerprint(_family(space, {"friction_floor": 0.18}), space)
    b = family_fingerprint(_family(space, {"friction_floor": 0.22}), space)  # slightly different
    assert set(a.signature) == set(b.signature)
    assert fingerprints_match(a, b)


def test_fingerprint_str_is_key(space):
    fp = family_fingerprint(_family(space, {"friction_floor": 0.18}), space)
    assert str(fp) == fp.key
    assert isinstance(fp.key, str) and fp.key


def test_fingerprint_pure_nominal_is_mixed(space):
    fp = family_fingerprint(_family(space, {}), space)  # no deviation from nominal
    assert fp.signature == ()
    assert fp.key == "mixed"


# --------------------------------------------------------------------------- #
# match_families
# --------------------------------------------------------------------------- #
def test_match_families_matched(space):
    old = [_family(space, {"friction_floor": 0.18}, fid="fam-0")]
    new = [_family(space, {"friction_floor": 0.20}, fid="fam-2")]  # same mode, churned idx
    pairs = match_families(old, new, space)
    assert len(pairs) == 1
    o, n = pairs[0]
    assert o is old[0] and n is new[0]


def test_match_families_unmatched_both_sides(space):
    old = [_family(space, {"friction_floor": 0.18}, fid="fam-0")]     # resolved
    new = [_family(space, {"latency_steps": 3.5}, fid="fam-1")]       # new mode
    pairs = match_families(old, new, space)
    matched = [(o, n) for (o, n) in pairs if o and n]
    only_old = [(o, n) for (o, n) in pairs if o and not n]
    only_new = [(o, n) for (o, n) in pairs if n and not o]
    assert not matched
    assert len(only_old) == 1 and only_old[0][0] is old[0]
    assert len(only_new) == 1 and only_new[0][1] is new[0]


def test_match_families_tolerant_centroid(space):
    old = [_family(space, {"friction_floor": 0.18})]
    # Same dominant dim+direction but far apart on that axis -> beyond tol -> no match.
    far = [_family(space, {"friction_floor": 0.60})]
    pairs = match_families(old, far, space, tol=regression.DEFAULT_TOL)
    assert all(not (o and n) for (o, n) in pairs)  # nothing matched
    # A generous tol pulls them back together.
    pairs_loose = match_families(old, far, space, tol=1.0)
    assert any(o and n for (o, n) in pairs_loose)


# --------------------------------------------------------------------------- #
# diff_atlases
# --------------------------------------------------------------------------- #
def test_diff_atlases_categories(space):
    # old atlas: a slip family and a latency family
    slip_old = _family(space, {"friction_floor": 0.18}, fid="fam-0", name="slip",
                       fail=0.55, members=3)
    latency_old = _family(space, {"latency_steps": 3.5}, fid="fam-1", name="latency",
                          fail=0.6, members=3)
    # new atlas: slip UNCHANGED (same-ish), latency WORSENED (higher score), a NEW payload
    slip_new = _family(space, {"friction_floor": 0.19}, fid="fam-2", name="slip",
                       fail=0.55, members=3)
    latency_new = _family(space, {"latency_steps": 3.6}, fid="fam-0", name="latency",
                          fail=0.85, members=6)  # score jumped -> worsened
    payload_new = _family(space, {"torso_mass_add_kg": 5.5}, fid="fam-3", name="payload",
                          fail=0.7, members=4)
    # latency_old resolved? No -- it matches latency_new. friction resolved? No -- matches.
    # Add an OLD-only family to trigger RESOLVED.
    com_old = _family(space, {"com_offset_x_m": 0.07}, fid="fam-9", name="com-shift",
                      fail=0.5, members=2)

    diff = diff_atlases([slip_old, latency_old, com_old],
                        [slip_new, latency_new, payload_new], space)

    names = lambda cat: sorted(
        (e.get("name") for e in diff[cat])
    )
    assert names("new") == ["payload"]
    assert names("resolved") == ["com-shift"]
    assert names("worsened") == ["latency"]
    assert names("unchanged") == ["slip"]

    # Deltas present on the worsened entry.
    w = diff["worsened"][0]
    assert w["deltas"]["mean_failure_score"] == pytest.approx(0.25, abs=1e-6)
    assert w["deltas"]["members"] == 3
    assert w["reasons"]


def test_diff_worsened_on_lower_validity(space):
    old = [_family(space, {"friction_floor": 0.18}, fail=0.6, validity=0.9)]
    new = [_family(space, {"friction_floor": 0.18}, fail=0.6, validity=0.5)]  # validity dropped
    diff = diff_atlases(old, new, space)
    assert len(diff["worsened"]) == 1
    assert diff["worsened"][0]["deltas"]["validity"] == pytest.approx(-0.4, abs=1e-6)


# --------------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------------- #
def test_save_load_atlas_roundtrip(space, tmp_path):
    fams = [
        _family(space, {"friction_floor": 0.18}, fid="fam-0", name="slip", validity=0.8),
        _family(space, {"latency_steps": 3.5}, fid="fam-1", name="latency"),
    ]
    path = tmp_path / "atlas.json"
    save_atlas(fams, {"version": "ckpt-1"}, str(path))
    loaded, meta = load_atlas(str(path))
    assert meta["version"] == "ckpt-1"
    assert len(loaded) == 2
    assert [f.name for f in loaded] == ["slip", "latency"]
    assert loaded[0].triggering_parameters == fams[0].triggering_parameters
    assert loaded[0].validity == 0.8
    # Fingerprints survive the round-trip.
    assert family_fingerprint(loaded[0], space).key == family_fingerprint(fams[0], space).key


def test_version_store_ordering_and_baseline(space, tmp_path):
    hist = str(tmp_path / "atlas_history")
    save_version(hist, "ckpt-1", [_family(space, {"friction_floor": 0.18})])
    save_version(hist, "ckpt-2", [_family(space, {"latency_steps": 3.5})])
    versions = list_versions(hist)
    assert [v["version"] for v in versions] == ["ckpt-1", "ckpt-2"]
    # latest excluding the version being written is the true prior version.
    assert latest_version(hist, exclude="ckpt-2")["version"] == "ckpt-1"
    assert latest_version(hist)["version"] == "ckpt-2"
    fams, meta = load_version(hist, "ckpt-1")
    assert meta["version"] == "ckpt-1" and meta["sequence"] == 1
    assert len(fams) == 1


def test_save_version_no_wallclock_dependence(space, tmp_path):
    """Ordering must come from the stored sequence, not mtime -- a re-saved older version
    keeps its sequence and does not become 'latest'."""
    hist = str(tmp_path / "hist")
    save_version(hist, "ckpt-1", [_family(space, {"friction_floor": 0.18})])
    save_version(hist, "ckpt-2", [_family(space, {"latency_steps": 3.5})])
    # Re-save ckpt-1 (as if re-run); it must retain sequence 1, stay the prior of ckpt-2.
    save_version(hist, "ckpt-1", [_family(space, {"friction_floor": 0.17})])
    _, meta1 = load_version(hist, "ckpt-1")
    assert meta1["sequence"] == 1
    assert latest_version(hist)["version"] == "ckpt-2"


# --------------------------------------------------------------------------- #
# CI gate exit-code logic
# --------------------------------------------------------------------------- #
def test_gate_clean_diff_exit_zero(space):
    old = [_family(space, {"friction_floor": 0.18}, fail=0.55)]
    new = [_family(space, {"friction_floor": 0.18}, fail=0.55)]
    diff = diff_atlases(old, new, space)
    code, violations = evaluate_gate(diff)
    assert code == 0 and not violations


def test_gate_new_critical_exit_nonzero(space):
    old = [_family(space, {"friction_floor": 0.18}, fail=0.55)]
    # A brand-new HIGH-severity family (score 0.85 -> "high").
    new = [
        _family(space, {"friction_floor": 0.18}, fail=0.55),
        _family(space, {"latency_steps": 3.9}, name="latency-collapse", fail=0.95),
    ]
    diff = diff_atlases(old, new, space)
    code, violations = evaluate_gate(diff)
    assert code == 1
    assert any("NEW critical" in v for v in violations)


def test_gate_new_noncritical_exit_zero(space):
    old = [_family(space, {"friction_floor": 0.18}, fail=0.55)]
    # New family but LOW severity (score 0.30 < high/critical) and no validity signal.
    new = [
        _family(space, {"friction_floor": 0.18}, fail=0.55),
        _family(space, {"latency_steps": 3.9}, name="mild", fail=0.30),
    ]
    diff = diff_atlases(old, new, space)
    code, violations = evaluate_gate(diff)
    assert code == 0 and not violations


def test_gate_worsened_past_threshold_exit_nonzero(space):
    old = [_family(space, {"friction_floor": 0.18}, fail=0.55)]
    new = [_family(space, {"friction_floor": 0.18}, fail=0.80)]  # +0.25 > default 0.10
    diff = diff_atlases(old, new, space)
    code, violations = evaluate_gate(diff)
    assert code == 1
    assert any("WORSENED" in v for v in violations)


def test_gate_new_high_validity_exit_nonzero(space):
    old = []
    new = [_family(space, {"friction_floor": 0.18}, fail=0.30, validity=0.9)]  # low sev, high validity
    diff = diff_atlases(old, new, space)
    code, violations = evaluate_gate(diff)
    assert code == 1
    assert any("high-validity" in v for v in violations)


# --------------------------------------------------------------------------- #
# scripts/regression.py end-to-end (CPU: --families path, no GPU)
# --------------------------------------------------------------------------- #
def _write_atlas(space, path, families):
    save_atlas(families, {}, str(path))


def test_script_no_baseline_exits_zero_and_establishes(space, tmp_path, capsys):
    from scripts import regression as cli

    atlas = tmp_path / "fams1.json"
    _write_atlas(space, atlas, [_family(space, {"friction_floor": 0.18}, name="slip", fail=0.6)])
    bp_out = str(tmp_path / "out")
    code = cli.main(["--target", "go1", "--version", "ckpt-1",
                     "--families", str(atlas), "--bp-out", bp_out])
    assert code == 0
    assert "baseline" in capsys.readouterr().out.lower()
    # The version was stored.
    hist = regression.atlas_history_dir(bp_out, "go1")
    assert latest_version(hist)["version"] == "ckpt-1"


def test_script_new_critical_exits_nonzero(space, tmp_path):
    from scripts import regression as cli

    bp_out = str(tmp_path / "out")
    base = tmp_path / "base.json"
    _write_atlas(space, base, [_family(space, {"friction_floor": 0.18}, name="slip", fail=0.55)])
    assert cli.main(["--target", "go1", "--version", "ckpt-1",
                     "--families", str(base), "--bp-out", bp_out]) == 0

    nxt = tmp_path / "next.json"
    _write_atlas(space, nxt, [
        _family(space, {"friction_floor": 0.18}, name="slip", fail=0.55),
        _family(space, {"latency_steps": 3.9}, name="latency-collapse", fail=0.95),
    ])
    code = cli.main(["--target", "go1", "--version", "ckpt-2",
                     "--families", str(nxt), "--bp-out", bp_out])
    assert code == 1


def test_script_clean_rerun_exits_zero(space, tmp_path):
    from scripts import regression as cli

    bp_out = str(tmp_path / "out")
    fams = [_family(space, {"friction_floor": 0.18}, name="slip", fail=0.55)]
    a1 = tmp_path / "a1.json"
    a2 = tmp_path / "a2.json"
    _write_atlas(space, a1, fams)
    _write_atlas(space, a2, fams)
    assert cli.main(["--target", "go1", "--version", "ckpt-1",
                     "--families", str(a1), "--bp-out", bp_out]) == 0
    assert cli.main(["--target", "go1", "--version", "ckpt-2",
                     "--families", str(a2), "--bp-out", bp_out]) == 0


# --------------------------------------------------------------------------- #
# Coordinator fixes: CI-gate false-negatives must be closed
# --------------------------------------------------------------------------- #
def test_worsened_family_that_gains_secondary_dim_is_worsened(space):
    """Fix #1: a family that WORSENS (score up past the delta) while its signature drifts
    (gains a secondary dominant dim) must classify as WORSENED and trip the gate -- never
    resolved+new-noncritical."""
    old = [_family(space, {"friction_floor": 0.18}, name="slip", fail=0.55)]
    # friction stays the primary axis (dev 0.59) but a secondary dominant dim appears.
    new = [_family(space, {"friction_floor": 0.18, "com_offset_x_m": 0.05},
                   name="slip+com", fail=0.66)]  # +0.11 > default 0.10
    diff = diff_atlases(old, new, space)
    assert len(diff["worsened"]) == 1, diff
    assert not diff["new"] and not diff["resolved"]
    code, violations = evaluate_gate(diff)
    assert code == 1
    assert any("WORSENED" in v for v in violations)


def test_worsened_family_that_drifts_far_on_axis_is_worsened(space):
    """Fix #1: a family that intensifies AND drifts far along its dominant axis (past the
    tolerant match) must still be WORSENED via the diff safety net, not resolved+new."""
    old = [_family(space, {"friction_floor": 0.20}, name="slip", fail=0.55)]
    new = [_family(space, {"friction_floor": 0.65}, name="slip-drift", fail=0.66)]  # far + worse
    # Matching alone does not pair them (too far for tight/lenient)...
    pairs = match_families(old, new, space)
    assert all(not (o and n) for (o, n) in pairs)
    # ...but the diff safety net reconciles the shared primary axis into WORSENED.
    diff = diff_atlases(old, new, space)
    assert len(diff["worsened"]) == 1, diff
    assert not diff["new"] and not diff["resolved"]
    code, violations = evaluate_gate(diff)
    assert code == 1


def test_match_families_global_greedy_is_order_independent(space):
    """Fix #2: global-greedy assignment pairs BOTH families where a naive per-old greedy
    (process O1 first -> O1 steals N2, dropping O2) would mismatch.

    All four share primary (friction, low); only centroids differ. O2 is close ONLY to N2;
    O1 is close to both but closest to N2. Naive would take O1->N2 and drop O2->N2; global
    greedy assigns the globally-closest (O2,N2) first, leaving O1->N1.
    """
    O1 = _family(space, {"friction_floor": 0.60}, fid="fam-0", name="O1")
    O2 = _family(space, {"friction_floor": 0.66}, fid="fam-1", name="O2")
    N1 = _family(space, {"friction_floor": 0.47}, fid="fam-9", name="N1")
    N2 = _family(space, {"friction_floor": 0.67}, fid="fam-7", name="N2")
    pairs = match_families([O1, O2], [N1, N2], space)
    matched = {o.name: n.name for (o, n) in pairs if o and n}
    assert matched == {"O1": "N1", "O2": "N2"}, pairs
    assert all(o and n for (o, n) in pairs)  # nothing dropped


def test_diff_safety_net_reclassifies_new_sharing_resolved_primary(space):
    """Fix #1 (safety net): even a family that IMPROVED but drifted far is reconciled (not
    resolved+new); improvement classifies as unchanged and does not trip the gate."""
    old = [_family(space, {"friction_floor": 0.20}, name="slip", fail=0.75)]
    new = [_family(space, {"friction_floor": 0.65}, name="slip-better", fail=0.60)]  # improved
    diff = diff_atlases(old, new, space)
    assert not diff["new"] and not diff["resolved"]
    assert len(diff["unchanged"]) == 1  # improved -> unchanged, not worsened
    code, _ = evaluate_gate(diff)
    assert code == 0


# --------------------------------------------------------------------------- #
# Fix #3: baseline poisoning -- diff against the last GREEN baseline
# --------------------------------------------------------------------------- #
def test_resolve_baseline_prefers_green(space, tmp_path):
    hist = str(tmp_path / "hist")
    save_version(hist, "ckpt-1", [_family(space, {"friction_floor": 0.18})],
                 {"gate_passed": True})
    save_version(hist, "ckpt-2", [_family(space, {"friction_floor": 0.18})],
                 {"gate_passed": False})
    # Most-recent is ckpt-2 (red); resolve_baseline must return the last GREEN one, ckpt-1.
    assert latest_version(hist)["version"] == "ckpt-2"
    assert resolve_baseline(hist, exclude="ckpt-3")["version"] == "ckpt-1"


def test_script_sustained_regression_trips_gate_twice(space, tmp_path):
    """Fix #3: two consecutive regressed checkpoints BOTH trip the gate, because the second
    diffs against the last GREEN baseline (ckpt-1), not the poisoned red ckpt-2."""
    from scripts import regression as cli

    bp_out = str(tmp_path / "out")
    clean = tmp_path / "clean.json"
    regressed = tmp_path / "regressed.json"
    _write_atlas(space, clean, [_family(space, {"friction_floor": 0.18}, name="slip", fail=0.55)])
    _write_atlas(space, regressed, [
        _family(space, {"friction_floor": 0.18}, name="slip", fail=0.55),
        _family(space, {"latency_steps": 3.9}, name="latency-collapse", fail=0.95),
    ])
    assert cli.main(["--target", "go1", "--version", "ckpt-1",
                     "--families", str(clean), "--bp-out", bp_out]) == 0
    assert cli.main(["--target", "go1", "--version", "ckpt-2",
                     "--families", str(regressed), "--bp-out", bp_out]) == 1
    # ckpt-3 still regressed vs the GREEN baseline (ckpt-1), so it must ALSO trip.
    assert cli.main(["--target", "go1", "--version", "ckpt-3",
                     "--families", str(regressed), "--bp-out", bp_out]) == 1
    hist = regression.atlas_history_dir(bp_out, "go1")
    assert resolve_baseline(hist, exclude="ckpt-4")["version"] == "ckpt-1"  # last green


# --------------------------------------------------------------------------- #
# Fix #4: fingerprint uses the persisted family-mean signature (matches the name)
# --------------------------------------------------------------------------- #
def _rec(space, theta_named, sig_val, fail, rid):
    return FailureRecord(
        theta_vector=[theta_named[n] for n in space.names()],
        theta_named=theta_named,
        failure_score=fail,
        component_signals={"fall": 0.0},
        rollout_signature=[sig_val] * 6,
        seed=0,
        severity="high",
        transfer_score=0.5,
        id=rid,
    )


def test_build_families_persists_mean_signature_and_fingerprint_uses_it(space):
    from breakpoint_mjx.clustering import build_families
    from breakpoint_mjx.naming import _label

    nom = _nominal(space)
    low_fric = dict(nom); low_fric["friction_floor"] = 0.20
    hi_lat = dict(nom); hi_lat["latency_steps"] = 3.8
    records = (
        [_rec(space, dict(low_fric), 0.0, 0.8, f"a-{i}") for i in range(6)]
        + [_rec(space, dict(hi_lat), 5.0, 0.85, f"b-{i}") for i in range(6)]
    )
    families, _ = build_families(records, space, k=2)
    assert len(families) == 2
    for f in families:
        # The mean-based discriminative signature is persisted at build time...
        assert f.dominant_dims is not None and f.mean_theta is not None
        fp = family_fingerprint(f, space)
        # ...and the fingerprint keys off it, so its primary label matches the displayed name
        # (fix #4: fingerprint tracks the family MEAN / name, not the representative row).
        assert fp.primary is not None
        assert _label(fp.primary[0], fp.primary[1]) in f.name


# --------------------------------------------------------------------------- #
# Fix #5: importing regression must NOT pull in sklearn (or jax)
# --------------------------------------------------------------------------- #
def test_regression_import_is_sklearn_and_jax_free():
    code = (
        "import sys, breakpoint_mjx.regression, breakpoint_mjx.naming; "
        "assert 'sklearn' not in sys.modules, 'sklearn imported'; "
        "assert 'jax' not in sys.modules, 'jax imported'"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# Fix #6: a silently-broken (empty/collapsed) search must not read green
# --------------------------------------------------------------------------- #
def test_broken_search_warnings(space):
    fam = [_family(space, {"friction_floor": 0.18})]
    assert broken_search_warnings([], fam)                     # 0 families vs a real baseline
    assert not broken_search_warnings(fam, fam)                # steady -> no warning
    assert broken_search_warnings(fam[:1], fam * 4)            # sharp drop 4 -> 1


def test_script_empty_atlas_trips_gate(space, tmp_path):
    from scripts import regression as cli

    bp_out = str(tmp_path / "out")
    base = tmp_path / "base.json"
    empty = tmp_path / "empty.json"
    _write_atlas(space, base, [_family(space, {"friction_floor": 0.18}, name="slip", fail=0.55)])
    _write_atlas(space, empty, [])  # broken search: no families
    assert cli.main(["--target", "go1", "--version", "ckpt-1",
                     "--families", str(base), "--bp-out", bp_out]) == 0
    assert cli.main(["--target", "go1", "--version", "ckpt-2",
                     "--families", str(empty), "--bp-out", bp_out]) == 1
