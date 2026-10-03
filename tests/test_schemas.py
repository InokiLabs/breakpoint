import json
from breakpoint_mjx.schemas import FailureRecord, FailureFamily, BreakPointReport

def test_failure_record_roundtrips_json():
    r = FailureRecord(
        theta_vector=[1.0, 0.0], theta_named={"friction_scale": 1.0, "payload_mass_kg": 0.0},
        failure_score=0.9, component_signals={"fall": 1.0, "time_to_failure": 42.0},
        rollout_signature=[0.1, 0.2, 0.3], seed=7, severity="high",
        transfer_score=0.8, priority_rank=1,
    )
    s = r.to_json()
    back = FailureRecord.from_json(s)
    assert back == r
    assert json.loads(s)["seed"] == 7


def test_failure_record_roundtrips_batch_key_and_env_index():
    """Upgrade #6: batch_key (two uint32 ints) + env_index survive JSON round-trip."""
    r = FailureRecord(
        theta_vector=[1.0, 0.0], theta_named={"a": 1.0, "b": 0.0},
        failure_score=0.9, component_signals={"fall": 1.0},
        rollout_signature=[0.1], seed=7,
        batch_key=[123456789, 987654321], env_index=42,
    )
    s = r.to_json()
    assert json.loads(s)["batch_key"] == [123456789, 987654321]
    assert json.loads(s)["env_index"] == 42
    back = FailureRecord.from_json(s)
    assert back == r
    assert back.batch_key == [123456789, 987654321]
    assert back.env_index == 42


def test_failure_record_legacy_json_without_new_fields():
    """Records serialized before Upgrade #6 (no batch_key/env_index) still load,
    defaulting the new fields to None."""
    legacy = json.dumps({
        "theta_vector": [1.0, 0.0], "theta_named": {"a": 1.0, "b": 0.0},
        "failure_score": 0.9, "component_signals": {"fall": 1.0},
        "rollout_signature": [0.1], "seed": 7, "severity": "high",
        "transfer_score": 0.8, "priority_rank": 1, "family_id": None, "id": "x-0",
    })
    back = FailureRecord.from_json(legacy)
    assert back.batch_key is None
    assert back.env_index is None
    assert back.id == "x-0"

def test_report_collects_families_and_serializes():
    fam = FailureFamily(
        id="fam-0", name="low-friction slip", member_ids=["f1", "f2"],
        representative_id="f1", triggering_parameters={"friction_scale": 0.35},
        mean_failure_score=0.8, mean_transfer_score=0.7, priority_rank=1,
    )
    rep = BreakPointReport(
        run_id="r1", policy_descriptor="go1_joystick",
        comparison={"random_max": 0.6, "directed_max": 0.92},
        families=[fam], mp4_paths={"fam-0": "out/fam-0.mp4"},
    )
    s = rep.to_json()
    assert "low-friction slip" in s
    assert BreakPointReport.from_json(s).families[0].member_ids == ["f1", "f2"]
