"""CPU-pure tests for the installable CLI: tiering, corpus privacy, command wiring.

No jax, no mujoco, no env loading -- these run in the normal suite on any box.
"""
import json

import pytest

from breakpoint_mjx import cli, corpus, tier


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated BREAKPOINT_HOME so tests never touch the developer's real config."""
    monkeypatch.setenv("BREAKPOINT_HOME", str(tmp_path))
    monkeypatch.delenv("BREAKPOINT_LICENSE_KEY", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# tier
# ---------------------------------------------------------------------------

def test_minted_key_validates_and_typo_does_not():
    k = tier.mint_key("0123456789abcdef0123456789ab")
    assert tier.key_is_valid(k)
    assert not tier.key_is_valid(k[:-1] + ("0" if k[-1] != "0" else "1"))
    assert not tier.key_is_valid("bp_oem_short")
    assert not tier.key_is_valid(None)
    assert not tier.key_is_valid("totally-made-up")


def test_mint_rejects_bad_body():
    with pytest.raises(ValueError):
        tier.mint_key("nothex" * 4)


def test_default_tier_is_free_and_key_upgrades_it(home):
    assert tier.resolve_tier() == tier.FREE
    tier.save_key(tier.mint_key("a" * 28))
    assert tier.resolve_tier() == tier.OEM


def test_env_key_beats_saved_file(home, monkeypatch):
    tier.save_key(tier.mint_key("a" * 28))
    monkeypatch.setenv("BREAKPOINT_LICENSE_KEY", "garbage")
    assert tier.resolve_tier() == tier.FREE  # env wins, and garbage is not a licence


def test_corrupt_license_file_degrades_to_free(home):
    tier.license_path().parent.mkdir(parents=True, exist_ok=True)
    tier.license_path().write_text("{not json", encoding="utf-8")
    assert tier.load_key() is None
    assert tier.resolve_tier() == tier.FREE  # never a crash on a bad config


def test_save_key_rejects_malformed(home):
    with pytest.raises(ValueError):
        tier.save_key("bp_oem_nope")


def test_free_finds_but_cannot_fix(home):
    assert tier.allows("search", tier.FREE)
    assert tier.allows("report", tier.FREE)
    assert not tier.allows("harden", tier.FREE)
    assert not tier.allows("certify", tier.FREE)
    with pytest.raises(tier.LicenseRequired) as e:
        tier.require("harden")
    assert "activate" in str(e.value)  # the message must say what to do next
    tier.save_key(tier.mint_key("b" * 28))
    assert tier.require("harden") == tier.OEM


def test_unknown_feature_is_an_error_not_a_silent_allow():
    with pytest.raises(KeyError):
        tier.allows("teleport", tier.OEM)


# ---------------------------------------------------------------------------
# corpus privacy -- the contract that matters
# ---------------------------------------------------------------------------

FORBIDDEN = ("rollout_signature", "theta_vector", "seed", "batch_key", "env_index",
             "checkpoint", "urdf", "policy")


def test_redaction_is_an_allow_list_so_new_fields_cannot_leak():
    """A field added to FailureRecord later must NOT start shipping by itself."""
    rec = {
        "theta_named": {"friction_floor": 0.4},
        "failure_score": 0.9,
        "component_signals": {"fall": 1.0},
        "severity": "critical",
        # everything below must be dropped
        "rollout_signature": [0.1] * 30,
        "theta_vector": [1.0, 2.0],
        "seed": 7,
        "batch_key": [1, 2],
        "env_index": 3,
        "policy_weights": [0.5],
        "urdf_path": "C:/secret/robot.urdf",
        "a_field_invented_next_year": "should not ship",
    }
    out = corpus.redact_record(rec)
    assert set(out) <= set(corpus.ALLOWED_RECORD_KEYS)
    assert "a_field_invented_next_year" not in out
    blob = json.dumps(out)
    for f in FORBIDDEN:
        assert f not in blob
    assert "secret" not in blob


def test_payload_contains_no_identifying_metadata():
    payload = corpus.build_payload(
        [{"theta_named": {"friction_floor": 0.4}, "failure_score": 0.9}],
        robot_class="quadruped", dof=12, total_mass_kg=12.743,
        engine_version="0.1.0", search_mode="agentic", horizon=500)
    assert set(payload["meta"]) <= set(corpus.ALLOWED_META_KEYS)
    assert payload["meta"]["mass_bucket"] == "5-20kg"   # bucketed, not exact
    assert "12.743" not in json.dumps(payload)


def test_bad_robot_class_is_rejected():
    with pytest.raises(ValueError):
        corpus.build_payload([], robot_class="mech", dof=1, total_mass_kg=1,
                             engine_version="0")


@pytest.mark.parametrize("kg,bucket", [(1, "<5kg"), (4.9, "<5kg"), (5, "5-20kg"),
                                       (19.9, "5-20kg"), (20, "20-60kg"),
                                       (59.9, "20-60kg"), (60, ">60kg"), (500, ">60kg")])
def test_mass_buckets(kg, bucket):
    assert corpus.mass_bucket(kg) == bucket


def test_consent_roundtrip_and_stale_manifest_re_asks(home, monkeypatch):
    assert corpus.read_consent() is None          # never asked
    corpus.write_consent(True)
    assert corpus.read_consent() is True
    corpus.write_consent(False)
    assert corpus.read_consent() is False

    # If the deal changes, a previous "yes" must not carry over silently.
    monkeypatch.setattr(corpus, "MANIFEST_VERSION", corpus.MANIFEST_VERSION + 1)
    assert corpus.read_consent() is None


def test_stage_writes_the_exact_payload(home):
    payload = corpus.build_payload(
        [{"theta_named": {"friction_floor": 0.4}, "failure_score": 0.9}],
        robot_class="humanoid", dof=23, total_mass_kg=35.0, engine_version="0.1.0")
    p = corpus.stage(payload, name="x.json")
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == payload


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_every_subcommand_parses():
    p = cli.build_parser()
    for argv in (["doctor"], ["license"], ["activate", "k"],
                 ["run", "--robot", "go1"], ["harden", "--robot", "go1"],
                 ["corpus", "status"], ["corpus", "consent", "yes"],
                 ["corpus", "preview"], ["corpus", "export", "--failures", "f.json"]):
        assert p.parse_args(argv).func is not None


def test_contribute_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["run", "--robot", "go1", "--contribute", "--no-contribute"])


def test_license_and_corpus_status_succeed(home, capsys):
    assert cli.main(["license"]) == 0
    assert cli.main(["corpus", "status"]) == 0
    assert "free" in capsys.readouterr().out


def test_harden_on_free_tier_exits_3_with_guidance(home, capsys):
    assert cli.main(["harden", "--robot", "go1"]) == 3
    err = capsys.readouterr().err
    assert "commercial" in err and "activate" in err


def test_run_rejects_unknown_robot_with_the_list(home, capsys):
    rc = cli.main(["run", "--robot", "definitely-not-a-robot", "--no-contribute"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown robot" in err and "go1" in err  # tells them what IS available


def test_run_rejects_missing_urdf_file(home, capsys):
    rc = cli.main(["run", "--robot", "no/such/robot.urdf", "--no-contribute"])
    assert rc == 2
    assert "no such robot description" in capsys.readouterr().err


def test_corpus_preview_shows_payload_without_forbidden_fields(home, capsys):
    assert cli.main(["corpus", "preview"]) == 0
    out = capsys.readouterr().out
    assert "NEVER" in out  # the promise is printed, not buried in docs

    # Check the PAYLOAD, not the prose: the promise text legitimately contains the
    # words "policy" and "URDF" while saying they are never sent.
    payload = json.loads(out[out.index("{"):])
    blob = json.dumps(payload)
    for f in FORBIDDEN:
        assert f not in blob
    assert set(payload["meta"]) <= set(corpus.ALLOWED_META_KEYS)
    for finding in payload["findings"]:
        assert set(finding) <= set(corpus.ALLOWED_RECORD_KEYS)


def test_non_interactive_run_never_opts_in_by_default(home, monkeypatch):
    """A CI or scripted run must default to NOT contributing."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli._resolve_consent(None) is False


def test_explicit_consent_is_persisted(home):
    assert cli._resolve_consent(True) is True
    assert corpus.read_consent() is True


def test_run_preflights_a_missing_policy_before_loading_anything(home, tmp_path, capsys):
    """The first-run failure mode must be an actionable message, not a traceback
    after a slow environment load."""
    rc = cli.main(["run", "--robot", "go1", "--out", str(tmp_path), "--no-contribute"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no policy to test" in err
    assert "--policy onnx:" in err and "scripts/train.py" in err


def test_preflight_is_skipped_when_a_policy_is_supplied(tmp_path):
    args = cli.build_parser().parse_args(
        ["run", "--robot", "go1", "--out", str(tmp_path), "--policy", "onnx:/p.onnx"])
    assert cli._missing_policy(args) is None


def test_preflight_passes_when_the_checkpoint_exists(tmp_path):
    from breakpoint_mjx.targets import get_target

    ckpt = tmp_path / "go1" / get_target("go1").checkpoint_name
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not a real checkpoint, but present")
    args = cli.build_parser().parse_args(
        ["run", "--robot", "go1", "--out", str(tmp_path)])
    assert cli._missing_policy(args) is None


def test_urdf_run_requires_an_explicit_policy(home, capsys, tmp_path):
    """A BYO robot has no checkpoint anywhere, so --policy is mandatory there."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='r'/>", encoding="utf-8")
    rc = cli.main(["run", "--robot", str(urdf), "--no-contribute"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "needs a policy" in err
    assert "ObsSpec" in err  # warns the obs layout is ours, not their policy's


def test_urdf_run_is_dispatched_as_urdf_not_target(tmp_path):
    """The pipeline must be invoked with --urdf; --target would reject a path."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='r'/>", encoding="utf-8")
    args = cli.build_parser().parse_args(
        ["run", "--robot", str(urdf), "--policy", "params:/w.npz"])
    assert cli._is_robot_file(args.robot)
    assert cli._missing_policy(args) is None
