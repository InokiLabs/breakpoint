from pathlib import Path
from breakpoint_mjx.config import (
    load_theta_space,
    load_search_config,
    merge_target_thresholds,
    SearchConfig,
)
from breakpoint_mjx.targets import get_target
from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]

# 16 dims after T6: the original 11 (9 model + obs_noise_scale + latency_steps) plus the
# richer failure axes -- 4 rollout (obs_dropout_prob, obs_bias_scale, push_mag,
# push_step_frac) + 1 model (floor_slope_rad).
N_DIMS = 16

def test_theta_space_has_sixteen_dims_with_valid_bounds():
    ts = load_theta_space(config_path("go1/theta_space.yaml"))
    assert len(ts.dims) == N_DIMS
    for name, d in ts.dims.items():
        lo, hi = d.bounds
        assert lo < hi, name
        assert lo <= d.nominal <= hi, name
        assert d.kind in ("model", "rollout"), name
        p_lo, p_hi = d.plausible
        assert p_lo >= lo and p_hi <= hi, f"{name}: plausible outside bounds"

def test_theta_space_vector_roundtrip():
    ts = load_theta_space(config_path("go1/theta_space.yaml"))
    names = ts.names()
    assert names[0] == "friction_floor"          # order is stable = YAML order
    nominal_vec = ts.nominal_vector()
    assert len(nominal_vec) == N_DIMS
    named = ts.vector_to_named(nominal_vec)
    assert named["torso_mass_add_kg"] == 0.0


# T6 richer failure axes: the new dims exist in ALL THREE targets with the right kinds,
# and all-nominal is the "no adversary" vector (so the space still degrades to nominal
# physics + no perturbations).
_NEW_ROLLOUT = ("obs_dropout_prob", "obs_bias_scale", "push_mag", "push_step_frac")
_NEW_MODEL = ("floor_slope_rad",)

def test_richer_axes_present_with_correct_kinds_in_all_targets():
    for tgt in ("go1", "barkour", "g1"):
        ts = load_theta_space(config_path(f"{tgt}/theta_space.yaml"))
        for name in _NEW_ROLLOUT:
            assert name in ts.dims, (tgt, name)
            assert ts.dims[name].kind == "rollout", (tgt, name)
            assert ts.dims[name].nominal == 0.0, (tgt, name)
        for name in _NEW_MODEL:
            assert name in ts.dims, (tgt, name)
            assert ts.dims[name].kind == "model", (tgt, name)
            assert ts.dims[name].nominal == 0.0, (tgt, name)

def test_floor_slope_bounds_are_signed_and_symmetric():
    ts = load_theta_space(config_path("go1/theta_space.yaml"))
    lo, hi = ts.dims["floor_slope_rad"].bounds
    assert lo < 0.0 < hi          # signed tilt (uphill / downhill)
    assert lo == -hi              # symmetric about flat
    assert ts.dims["push_mag"].bounds[0] == 0.0   # push magnitude is non-negative

def test_search_config_loads():
    sc = load_search_config(config_path("search.yaml"))
    assert sc.n_eval >= 1024
    assert sc.generations > 0 and sc.popsize > 0
    assert 0.0 < sc.failure_threshold < 1.0
    assert sc.thresholds["trunk_height_min"] > 0
    assert sc.thresholds["up_proj_min"] > 0
    assert sc.thresholds["tracking_err_max"] > 0
    assert sc.thresholds["ang_vel_max"] > 0
    assert len(sc.thresholds) == 4


def test_search_config_reads_atlas_block():
    """The atlas: block in search.yaml is read into SearchConfig.atlas."""
    sc = load_search_config(config_path("search.yaml"))
    assert sc.atlas["n_ttf_bins"] >= 2
    assert sc.atlas["iterations"] >= 1
    assert sc.atlas["batch_size"] >= 1


def test_search_config_atlas_defaults_empty_when_absent(tmp_path):
    """A config with no atlas: block still loads; atlas defaults to {} (back-compat).

    atlas.py supplies its own per-key fallbacks, so an empty dict is sufficient here."""
    yaml_text = (config_path("search.yaml")).read_text()
    # Drop the atlas block (everything from the `atlas:` line onward is the last block).
    trimmed = yaml_text.split("atlas:")[0]
    p = tmp_path / "search_no_atlas.yaml"
    p.write_text(trimmed)
    sc = load_search_config(p)
    assert sc.atlas == {}
    assert isinstance(sc, SearchConfig)


# --------------------------------------------------------------------------- #
# Per-target threshold merge (Part 2): target overrides win, absent leaves defaults
# --------------------------------------------------------------------------- #

def test_merge_thresholds_absent_override_leaves_defaults_unchanged():
    """A target with thresholds=None (go1) leaves cfg.thresholds exactly as search.yaml."""
    cfg = load_search_config(config_path("search.yaml"))
    merged = merge_target_thresholds(cfg, get_target("go1"))
    assert merged.thresholds == cfg.thresholds
    # frozen dataclass -> a replaced copy, original untouched.
    assert merged is not cfg or merged.thresholds == cfg.thresholds


def test_merge_thresholds_target_override_wins():
    """The g1 humanoid override for trunk_height_min wins over the search.yaml default,
    while every other (non-overridden) threshold key keeps the default value."""
    cfg = load_search_config(config_path("search.yaml"))
    g1 = get_target("g1")
    merged = merge_target_thresholds(cfg, g1)
    override = g1.thresholds["trunk_height_min"]
    assert override != cfg.thresholds["trunk_height_min"]   # the override is meaningful
    assert merged.thresholds["trunk_height_min"] == override
    # Non-overridden keys are untouched.
    for k, v in cfg.thresholds.items():
        if k == "trunk_height_min":
            continue
        assert merged.thresholds[k] == v
    # The merge does not mutate the search.yaml-loaded defaults in place.
    assert cfg.thresholds["trunk_height_min"] != override
