"""Defects vs coverage gaps (breakpoint_mjx/training_gap.py) -- pure, no JAX."""
from types import SimpleNamespace

import pytest
import yaml

from breakpoint_mjx import training_gap as tg
from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.paths import config_path
from breakpoint_mjx.report import _training_html, _training_row

DEMO_DR = config_path("configs/go1/theta_space.yaml").parents[2] / "assets" / "demo" / \
    "go1_training_dr.yaml"


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def trained(space):
    return tg.training_ranges(space, yaml.safe_load(DEMO_DR.read_text())["training_dr"])


def _fam(fid="fam-0", cause=None, theta=None, dominant=None, mean=None):
    return SimpleNamespace(id=fid, minimal_cause=cause, triggering_parameters=theta or {},
                           dominant_dims=dominant, mean_theta=mean)


def test_unlisted_dims_are_trained_at_nominal_only(space, trained):
    assert trained["friction_floor"] == (0.4, 1.0)
    assert trained["latency_steps"] == (0.0, 0.0)
    assert trained["kp_scale"] == (1.0, 1.0)


def test_latency_failure_on_the_demo_policy_is_a_coverage_gap(space, trained):
    c = tg.classify(_fam(cause=[{"dim": "latency_steps", "value": 2.0,
                                 "direction": "high"}]), space, trained)
    assert c["verdict"] == "coverage_gap" and c["cause_source"] == "minimal_cause"
    (row,) = c["outside"]
    assert row["dim"] == "latency_steps" and row["trained"] == [0.0, 0.0]
    assert row["excess"] == pytest.approx(0.5) and row["beyond_plausible"] is False


def test_failure_inside_the_trained_ranges_is_a_defect(space, trained):
    c = tg.classify(_fam(cause=[{"dim": "friction_floor", "value": 0.45},
                                {"dim": "torso_mass_add_kg", "value": 0.9}]), space, trained)
    assert c == {"verdict": "defect", "outside": [], "cause_source": "minimal_cause",
                 "provisional": False}


def test_only_cause_dims_count_not_every_jittered_dim(space, trained):
    """A minimal cause inside training is a defect even if the representative world also
    moved (irrelevant) untrained dims a little."""
    fam = _fam(cause=[{"dim": "friction_floor", "value": 0.42}],
               theta={"friction_floor": 0.42, "kp_scale": 1.3, "latency_steps": 3.0})
    assert tg.classify(fam, space, trained)["verdict"] == "defect"


def test_without_a_cause_the_representatives_moved_dims_are_used(space, trained):
    nom = {n: d.nominal for n, d in space.dims.items()}
    theta = dict(nom, latency_steps=3.0, kp_scale=1.01)   # kp moved < 5% of its span
    c = tg.classify(_fam(theta=theta), space, trained)
    assert c["cause_source"] == "representative" and c["provisional"] is True
    assert [r["dim"] for r in c["outside"]] == ["latency_steps"]


def test_without_a_cause_only_the_top_dominant_dims_are_judged(space, trained):
    """A search representative moves almost every dim; judging it whole blamed every axis
    (a smoke demo run's patch listed 14). Only the family's top dominant dims count, at
    the family-mean values, and push timing is never an axis of its own."""
    nom = {n: d.nominal for n, d in space.dims.items()}
    mean = dict(nom, torso_mass_add_kg=2.6, push_step_frac=0.6, friction_floor=0.6,
                com_offset_x_m=0.02, latency_steps=1.0)
    dominant = [[0.4, "torso_mass_add_kg", "high"], [0.3, "push_step_frac", "high"],
                [0.2, "friction_floor", "low"], [0.1, "com_offset_x_m", "high"],
                [0.05, "latency_steps", "high"]]
    c = tg.classify(_fam(dominant=dominant, mean=mean), space, trained)
    assert c["cause_source"] == "dominant_dims" and c["provisional"] is True
    assert [r["dim"] for r in c["outside"]] == ["torso_mass_add_kg"]   # top 3, no timing


def test_qualifier_dims_never_count(space, trained):
    c = tg.classify(_fam(cause=[{"dim": "push_step_frac", "value": 0.7},
                                {"dim": "push_mag", "value": 0.8}]), space, trained)
    assert [r["dim"] for r in c["outside"]] == ["push_mag"]


def test_no_training_dr_means_unknown(space):
    c = tg.classify(_fam(cause=[{"dim": "latency_steps", "value": 2.0}]), space, None)
    assert c["verdict"] == "unknown"


def test_patch_covers_the_failure_with_headroom_and_caps_at_plausible(space, trained):
    classified = {
        "fam-0": tg.classify(_fam("fam-0", [{"dim": "latency_steps", "value": 2.0}]),
                             space, trained),
        "fam-1": tg.classify(_fam("fam-1", [{"dim": "latency_steps", "value": 1.0}]),
                             space, trained),
        "fam-2": tg.classify(_fam("fam-2", [{"dim": "friction_floor", "value": 0.2}]),
                             space, trained),
    }
    patch = tg.suggest_patch(classified, space, trained)
    lat = patch["latency_steps"]
    # 2.0 + 10% headroom = 2.2, capped at the plausible top (2.0)
    assert lat["range"] == [0.0, 2.0] and lat["capped_at_plausible"] is True
    assert lat["was"] == [0.0, 0.0] and lat["families"] == ["fam-0", "fam-1"]
    # friction 0.2 is beyond the plausible band [0.4, 1.0]: the patch never widens past it
    fr = patch["friction_floor"]
    assert fr["range"] == [0.4, 1.0] and fr["capped_at_plausible"] is True
    assert "friction_range" in fr["hint"]


def test_patch_yaml_round_trips_as_a_training_dr_block(space, trained):
    classified = {"fam-0": tg.classify(_fam(cause=[{"dim": "push_mag", "value": 0.6}]),
                                       space, trained)}
    y = tg.patch_yaml(tg.suggest_patch(classified, space, trained))
    block = yaml.safe_load(y)["training_dr"]
    assert block == {"push_mag": {"range": [0.0, 0.66]}}
    assert "push_by_setting_velocity" in y


def test_annotate_reads_bp_train_dr_and_sets_each_family(space, tmp_path):
    fams = [_fam("fam-0", [{"dim": "latency_steps", "value": 2.0}]),
            _fam("fam-1", [{"dim": "friction_floor", "value": 0.5}])]
    out = tg.annotate(fams, space, environ={"BP_TRAIN_DR": str(DEMO_DR)})
    assert out["counts"] == {"coverage_gap": 1, "defect": 1}
    assert out["source"].startswith("BP_TRAIN_DR")
    assert fams[0].training_coverage["verdict"] == "coverage_gap"
    assert fams[1].training_coverage["verdict"] == "defect"
    assert set(out["patch"]) == {"latency_steps"}


def test_annotate_falls_back_to_the_contract_operational_block(space, tmp_path):
    contract = tmp_path / "contract.yaml"
    contract.write_text("operational:\n  latency_steps: {range: [0, 2]}\n")
    fams = [_fam("fam-0", [{"dim": "latency_steps", "value": 2.0}])]
    out = tg.annotate(fams, space, environ={"BP_OBS_SPEC": str(contract)})
    assert out["counts"] == {"defect": 1}


def test_annotate_without_any_source_leaves_families_unknown(space):
    fams = [_fam("fam-0", [{"dim": "latency_steps", "value": 2.0}])]
    out = tg.annotate(fams, space, environ={})
    assert out["source"] is None and out["counts"] == {"unknown": 1} and out["patch"] == {}


def test_report_renders_the_rows_and_the_patch(space):
    fams = [_fam("fam-0", [{"dim": "latency_steps", "value": 2.0}]),
            _fam("fam-1", [{"dim": "friction_floor", "value": 0.5}])]
    summary = tg.annotate(fams, space, environ={"BP_TRAIN_DR": str(DEMO_DR)})
    assert "coverage gap" in _training_row(fams[0]) and "latency_steps" in _training_row(fams[0])
    assert "defect" in _training_row(fams[1])
    html = _training_html(summary)
    assert "Defects vs coverage gaps" in html and "training_dr:" in html
    assert "Not classified" in _training_html(tg.annotate(fams, space, environ={}))


def test_latency_patch_is_whole_control_steps(space, trained):
    classified = {"fam-0": tg.classify(_fam(cause=[{"dim": "latency_steps", "value": 1.4}]),
                                       space, trained)}
    assert tg.suggest_patch(classified, space, trained)["latency_steps"]["range"] == [0, 2]

