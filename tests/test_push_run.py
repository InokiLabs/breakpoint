"""CPU tests for the web push path: the families.json payload builder (webpush) and
the stdlib push CLI's payload assembly (push_run). No jax, no network."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from breakpoint_mjx.schemas import FailureFamily
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.webpush import SCHEMA_VERSION, build_families_json
from breakpoint_mjx.targets import get_target
import push_run


from breakpoint_mjx.paths import config_path

def _family(name, score, transfer, rank, **kw):
    return FailureFamily(
        id=f"fam-{rank}", name=name, member_ids=["a", "b", "c"],
        representative_id="a", triggering_parameters={"friction_floor": 0.2},
        mean_failure_score=score, mean_transfer_score=transfer, priority_rank=rank, **kw,
    )


def _space():
    root = Path(__file__).resolve().parents[1]
    return load_theta_space(config_path(get_target("go1").theta_space_path))


def test_build_families_json_shape_and_versions():
    space = _space()
    fams = [
        _family("low-friction slip", 1.02, 0.91, 1, validity=0.83,
                reproduction=[True, True, False], margin="stable until friction_floor < 0.35",
                attribution={"friction_floor": 0.6}),
        _family("sensor-noise drift", 0.58, 0.64, 2),  # validity/margin None -> null path
    ]
    payload = build_families_json(
        target=get_target("go1"), space=space, families=fams,
        n_total=1000, n_failures=170, failure_rate_ci=(0.17, 0.14, 0.20),
        search_mode="cmaes", horizon=500, engine_commit="abc1234",
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run"]["theta_dim"] == len(space.names())
    assert payload["run"]["search_mode"] == "cmaes"
    assert payload["stats"]["failure_rate"] == 0.17
    assert payload["stats"]["ci_lo"] == 0.14 and payload["stats"]["ci_hi"] == 0.20
    assert len(payload["stats"]["expected_risk_ranking"]) == 2
    f0 = payload["families"][0]
    assert f0["name"] == "low-friction slip"
    assert f0["member_count"] == 3
    assert f0["severity"] in ("critical", "high", "medium", "low")
    assert isinstance(f0["sev_prob_score"], float)
    assert payload["families"][1]["validity"] is None
    # the whole thing must be JSON-serializable (no FailureFamily objects leaked)
    json.dumps(payload)


def test_push_run_build_payload_with_and_without_optionals(tmp_path):
    fam = {
        "schema_version": "1.0",
        "run": {"search_mode": "cmaes", "horizon": 500, "theta_dim": 16, "engine_commit": "abc"},
        "stats": {"n_total": 10, "failure_rate": 0.2, "ci_lo": 0.1, "ci_hi": 0.3,
                  "expected_risk_ranking": [], "severity_primary_ranking": []},
        "families": [{"id": "fam-0", "name": "low-friction slip", "severity": "critical",
                      "mean_failure_score": 1.0, "mean_transfer_score": 0.9, "member_count": 3,
                      "priority_rank": 1, "sev_prob_score": 0.4,
                      "representative_theta": {"friction_floor": 0.2},
                      "validity": None, "reproduction": None, "margin": None,
                      "boundaries": None, "attribution": None}],
    }
    (tmp_path / "families.json").write_text(json.dumps(fam))
    (tmp_path / "fam-0.mp4").write_bytes(b"\x00" * 16)

    p = push_run.build_payload(str(tmp_path), "6f1f2d4e-9c1b-4d2a-8e57-0a3b1c2d3e4f", "ckpt-1")
    assert p["run_id"].startswith("6f1f")
    assert p["version_label"] == "ckpt-1"
    names = [a["name"] for a in p["artifacts"]]
    assert "failures.json.gz" in names
    assert "video:low-friction slip" in names
    assert "regression" not in p and "hardening" not in p

    # with a regression.json present, it's attached
    (tmp_path / "regression.json").write_text(json.dumps({"gate_passed": False}))
    p2 = push_run.build_payload(str(tmp_path), "6f1f2d4e-9c1b-4d2a-8e57-0a3b1c2d3e4f", "ckpt-1")
    assert p2["regression"] == {"gate_passed": False}


def test_push_run_no_video_artifact_when_absent(tmp_path):
    fam = {"schema_version": "1.0", "run": {}, "stats": {},
           "families": [{"id": "fam-0", "name": "no-video family"}]}
    (tmp_path / "families.json").write_text(json.dumps(fam))
    p = push_run.build_payload(str(tmp_path), "6f1f2d4e-9c1b-4d2a-8e57-0a3b1c2d3e4f", "v")
    assert [a["name"] for a in p["artifacts"]] == ["failures.json.gz"]
