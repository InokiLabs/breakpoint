"""Typed loaders for the YAML configs. Pure stdlib + pyyaml."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ThetaDim:
    name: str
    kind: str                  # "model" | "rollout"
    bounds: tuple[float, float]
    nominal: float
    plausible: tuple[float, float]


@dataclass
class ThetaSpace:
    dims: dict[str, ThetaDim]  # insertion-ordered = vector order

    def names(self) -> list[str]:
        return list(self.dims.keys())

    def nominal_vector(self) -> list[float]:
        return [d.nominal for d in self.dims.values()]

    def lower(self) -> list[float]:
        return [d.bounds[0] for d in self.dims.values()]

    def upper(self) -> list[float]:
        return [d.bounds[1] for d in self.dims.values()]

    def vector_to_named(self, vec) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.names(), vec)}


@dataclass(frozen=True)
class SearchConfig:
    horizon: int
    n_eval: int
    generations: int
    popsize: int
    seed: int
    failure_threshold: float
    n_families: int
    thresholds: dict[str, float]
    # Upgrade #4 (failure atlas / MAP-Elites). Optional block; defaults to {} so configs
    # written before the atlas existed still load. atlas.py applies its own fallbacks per key.
    atlas: dict = field(default_factory=dict)


def load_theta_space(path: str | Path) -> ThetaSpace:
    raw = yaml.safe_load(Path(path).read_text())
    dims: dict[str, ThetaDim] = {}
    for name, d in raw["dims"].items():
        dims[name] = ThetaDim(
            name=name,
            kind=d["kind"],
            bounds=(float(d["bounds"][0]), float(d["bounds"][1])),
            nominal=float(d["nominal"]),
            plausible=(float(d["plausible"][0]), float(d["plausible"][1])),
        )
    return ThetaSpace(dims=dims)


# Budget presets (BP_BUDGET). search.yaml is sized for a GPU (2048 worlds x 500 steps x
# 25 generations); on a laptop CPU that is hours. A preset overrides only the knobs that
# set wall-clock, so the SAME pipeline runs end to end at a size the machine can finish.
# "full" is search.yaml unchanged and is the default, so nothing shifts unless asked.
# smoke is a plumbing/onboarding run: 3 generations cannot adapt a 16-dim covariance,
# so its families are a preview, not a converged search result.
BUDGET_PRESETS: dict[str, dict] = {
    "smoke": {"n_eval": 64, "popsize": 32, "generations": 3, "horizon": 150,
              "atlas": {"batch_size": 32, "iterations": 3}},
    "standard": {"n_eval": 512, "popsize": 128, "generations": 10, "horizon": 300,
                 "atlas": {"batch_size": 128, "iterations": 10}},
    "full": {},
}

# Per-preset restart count for the multi-restart / coverage modes, which read BP_RESTARTS
# from the environment (pipeline.py), not from search.yaml. Exported by `breakpoint run
# --budget` unless the user already set BP_RESTARTS. "full" keeps the pipeline default.
BUDGET_RESTARTS: dict[str, int] = {"smoke": 2, "standard": 4}

_SINGLE_OVERRIDES = (("BP_HORIZON", "horizon"), ("BP_N_EVAL", "n_eval"),
                     ("BP_GENERATIONS", "generations"))


def budget_overrides(env=None) -> dict:
    """Raw-config overrides implied by BP_BUDGET + BP_HORIZON / BP_N_EVAL / BP_GENERATIONS.

    Pure (reads ``env``, default ``os.environ``). The preset applies first, then any single
    override wins over it. An unknown preset name raises rather than silently running a
    full-size search the user did not ask for.
    """
    import os

    env = os.environ if env is None else env
    name = (env.get("BP_BUDGET") or "full").strip().lower()
    if name not in BUDGET_PRESETS:
        raise ValueError(f"BP_BUDGET={name!r} is not one of {sorted(BUDGET_PRESETS)}")
    out = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in BUDGET_PRESETS[name].items()}
    for var, key in _SINGLE_OVERRIDES:
        v = env.get(var)
        if v not in (None, ""):
            n = int(v)
            if n <= 0:
                raise ValueError(f"{var} must be a positive integer, got {v!r}")
            out[key] = n
    return out


def apply_budget(raw: dict, env=None) -> dict:
    """Return a copy of a parsed search.yaml dict with the budget overrides applied."""
    raw = dict(raw)
    for k, v in budget_overrides(env).items():
        if k == "atlas":
            raw["atlas"] = {**(raw.get("atlas") or {}), **v}
        else:
            raw[k] = v
    return raw


def load_search_config(path: str | Path, env=None) -> SearchConfig:
    """Load search.yaml, then apply BP_BUDGET / single overrides from ``env`` (os.environ)."""
    raw = apply_budget(yaml.safe_load(Path(path).read_text()), env)
    return SearchConfig(
        horizon=int(raw["horizon"]),
        n_eval=int(raw["n_eval"]),
        generations=int(raw["generations"]),
        popsize=int(raw["popsize"]),
        seed=int(raw["seed"]),
        failure_threshold=float(raw["failure_threshold"]),
        n_families=int(raw["n_families"]),
        thresholds={k: float(v) for k, v in raw["thresholds"].items()},
        atlas=dict(raw.get("atlas") or {}),
    )


def merge_target_thresholds(cfg: SearchConfig, target) -> SearchConfig:
    """Return ``cfg`` with thresholds = ``{**cfg.thresholds, **(target.thresholds or {})}``.

    Per-target overrides win over the search.yaml defaults; a target with no overrides
    (``thresholds`` None) leaves the defaults unchanged. ``SearchConfig`` is frozen, so
    this returns a replaced copy; calling it once in run_pipeline means every consumer
    of ``cfg.thresholds`` (scores_from_trajectories / atlas / behavior_descriptor) sees
    the per-robot values without further plumbing.

    Import-safe: takes the target by duck-typing (only reads ``.thresholds``), so it
    pulls in neither the targets package nor jax -- the merge stays CPU-testable.
    """
    merged = {**cfg.thresholds, **(getattr(target, "thresholds", None) or {})}
    return dataclasses.replace(cfg, thresholds=merged)
