"""BP_BUDGET presets + single-knob overrides on load_search_config (CPU-pure)."""
import pytest
import yaml

from breakpoint_mjx import search
from breakpoint_mjx.config import (BUDGET_PRESETS, BUDGET_RESTARTS, apply_budget,
                                   budget_overrides, load_search_config)
from breakpoint_mjx.paths import config_path

YAML = config_path("search.yaml")


def _raw():
    return yaml.safe_load(YAML.read_text())


def test_full_equals_yaml_and_is_the_default():
    raw = _raw()
    for env in ({}, {"BP_BUDGET": "full"}, {"BP_BUDGET": ""}):
        cfg = load_search_config(YAML, env=env)
        assert (cfg.horizon, cfg.n_eval, cfg.generations, cfg.popsize) == (
            raw["horizon"], raw["n_eval"], raw["generations"], raw["popsize"])
        assert cfg.atlas == raw["atlas"]


def test_smoke_preset_values():
    cfg = load_search_config(YAML, env={"BP_BUDGET": "smoke"})
    assert (cfg.n_eval, cfg.popsize, cfg.generations, cfg.horizon) == (64, 32, 3, 150)
    assert cfg.atlas["batch_size"] == 32 and cfg.atlas["iterations"] == 3
    # untouched atlas keys survive the partial override
    assert cfg.atlas["sigma"] == _raw()["atlas"]["sigma"]
    # thresholds / seed / families are not budget knobs
    assert cfg.thresholds == load_search_config(YAML, env={}).thresholds
    assert cfg.seed == _raw()["seed"]


def test_standard_preset_values():
    cfg = load_search_config(YAML, env={"BP_BUDGET": "Standard"})  # case-insensitive
    assert (cfg.n_eval, cfg.popsize, cfg.generations, cfg.horizon) == (512, 128, 10, 300)


def test_single_overrides_win_over_preset():
    cfg = load_search_config(YAML, env={"BP_BUDGET": "smoke", "BP_HORIZON": "80",
                                        "BP_N_EVAL": "16", "BP_GENERATIONS": "2"})
    assert (cfg.horizon, cfg.n_eval, cfg.generations, cfg.popsize) == (80, 16, 2, 32)


def test_single_override_without_preset():
    raw = _raw()
    cfg = load_search_config(YAML, env={"BP_HORIZON": "200"})
    assert cfg.horizon == 200 and cfg.generations == raw["generations"]


@pytest.mark.parametrize("env", [{"BP_BUDGET": "huge"}, {"BP_HORIZON": "0"},
                                 {"BP_N_EVAL": "-4"}])
def test_bad_values_raise(env):
    with pytest.raises(ValueError):
        budget_overrides(env)


def test_apply_budget_does_not_mutate_input():
    raw = _raw()
    before = yaml.safe_dump(raw)
    apply_budget(raw, {"BP_BUDGET": "smoke"})
    assert yaml.safe_dump(raw) == before


def test_presets_shrink_monotonically():
    s, st = BUDGET_PRESETS["smoke"], BUDGET_PRESETS["standard"]
    for k in ("n_eval", "popsize", "generations", "horizon"):
        assert s[k] < st[k] < _raw()[k]
    assert BUDGET_RESTARTS["smoke"] < BUDGET_RESTARTS["standard"]


def test_os_environ_is_read_when_env_is_none(monkeypatch):
    monkeypatch.setenv("BP_BUDGET", "smoke")
    assert load_search_config(YAML).horizon == 150
    monkeypatch.delenv("BP_BUDGET")
    assert load_search_config(YAML).horizon == _raw()["horizon"]


def test_progress_line_eta():
    line = search.progress_line("cmaes gen", 2, 4, t0=100.0, now=110.0)
    assert line == "cmaes gen 2/4  elapsed 10s  eta 10s"
    assert "eta ?" in search.progress_line("x", 0, 4, t0=0.0, now=1.0)
