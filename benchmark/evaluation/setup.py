"""Benchmark configuration: policy configurations, bound regimes, search plans.

Nothing here changes how the engine loads policies or runs rollouts; it only resolves a
benchmark config (YAML) into the engine objects the existing modules expect.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from breakpoint_mjx.config import ThetaDim, ThetaSpace

REPO = Path(__file__).resolve().parents[2]
REGIMES = ("near_nominal", "conservative", "broad")
NEAR_NOMINAL_FRACTION = 0.25   # near_nominal band = nominal + 25% of the way to each plausible edge
METHODS = ("uniform", "sobol", "lhs_boundary", "directed")
GATE_BUDGETS = ("smoke", "standard")   # breakpoint_mjx.gate.BUDGETS
METHOD_LABELS = {
    "uniform": "Uniform random",
    "sobol": "Sobol (space-filling)",
    "lhs_boundary": "LHS + boundary (RoboGate-style)",
    "directed": "BreakPoint directed",
}
# Variants of the directed search, declared per config under ``method_variants``. Each is a
# NEW named method; the original ``directed`` is never altered and is always reported.
VARIANT_KINDS = ("novelty", "restarts", "robust")


@dataclass(frozen=True)
class PolicyConfig:
    """One robot-policy configuration under test."""
    name: str
    robot: str
    checkpoint: str
    obs_spec: str | None = None
    note: str = ""
    independent_policy: bool = True     # False = same weights as another config

    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint.split(":", 1)[-1])
        return p if p.is_absolute() else REPO / p

    def obs_spec_path(self) -> Path | None:
        if not self.obs_spec:
            return None
        p = Path(self.obs_spec)
        return p if p.is_absolute() else REPO / p


@dataclass(frozen=True)
class SearchPlan:
    budget: int
    restarts: int
    popsize: int

    @property
    def batch(self) -> int:
        return self.restarts * self.popsize


@dataclass
class BenchConfig:
    name: str
    policies: list[PolicyConfig]
    regimes: list[str]
    methods: list[str]
    repetitions: int
    horizon: int
    plan: SearchPlan
    cluster_eps: float = 0.10
    cluster_eps_sensitivity: list[float] = field(default_factory=lambda: [0.05, 0.2, 0.3])
    validation_n: int = 20
    validation_first_k: int = 3
    validation_leaders: int = 8
    replay_per_run: int = 2
    replay_seeds: int = 4
    regression: dict = field(default_factory=dict)
    method_variants: dict = field(default_factory=dict)
    rep_offset: int = 0            # first repetition index (tuning runs use disjoint seeds)
    source: str = ""

    def reps(self) -> range:
        return range(self.rep_offset, self.rep_offset + self.repetitions)

    def label(self, method: str) -> str:
        v = self.method_variants.get(method)
        return v.get("label", method) if v else METHOD_LABELS[method]

    def sha256(self) -> str:
        return hashlib.sha256(Path(self.source).read_bytes()).hexdigest() if self.source else ""


def load_config(path: str | Path) -> BenchConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    plan = SearchPlan(**raw["search_plan"])
    if plan.budget % plan.batch:
        raise ValueError(f"budget {plan.budget} must be a multiple of restarts*popsize "
                         f"= {plan.batch} so every method spends exactly the same rollouts")
    regimes = list(raw.get("regimes", REGIMES))
    methods = list(raw.get("methods", METHODS))
    for r in regimes:
        if r not in REGIMES:
            raise ValueError(f"unknown regime {r!r}; choose from {REGIMES}")
    variants = dict(raw.get("method_variants") or {})
    for name, v in variants.items():
        if name in METHODS:
            raise ValueError(f"variant {name!r} shadows a built-in method")
        if v.get("kind") not in VARIANT_KINDS:
            raise ValueError(f"variant {name!r}: kind must be one of {VARIANT_KINDS}")
        _check_variant_budget(name, v, plan)
    for m in methods:
        if m not in METHODS and m not in variants:
            raise ValueError(f"unknown method {m!r}; choose from {METHODS} or a variant")
    gb = (raw.get("regression") or {}).get("budget", "standard")
    if gb not in GATE_BUDGETS:
        raise ValueError(f"regression budget {gb!r}: the gate accepts {GATE_BUDGETS}")
    v = raw.get("validation", {})
    rp = raw.get("replay", {})
    return BenchConfig(
        name=raw["name"],
        policies=[PolicyConfig(**p) for p in raw["policies"]],
        regimes=regimes, methods=methods,
        repetitions=int(raw["repetitions"]), horizon=int(raw["horizon"]), plan=plan,
        cluster_eps=float(raw.get("cluster_eps", 0.10)),
        cluster_eps_sensitivity=[float(x) for x in raw.get("cluster_eps_sensitivity",
                                                           [0.05, 0.2, 0.3])],
        validation_n=int(v.get("n", 20)), validation_first_k=int(v.get("first_k", 3)),
        validation_leaders=int(v.get("leaders", 8)),
        replay_per_run=int(rp.get("per_run", 2)), replay_seeds=int(rp.get("seeds", 4)),
        regression=raw.get("regression", {}), method_variants=variants,
        rep_offset=int(raw.get("rep_offset", 0)), source=str(path))


def variant_plan(v: dict, plan: SearchPlan) -> tuple[int, int, int, int]:
    """(restarts, popsize, seeds_per_theta, generations) for a variant at the shared budget."""
    r = int(v.get("restarts", plan.restarts))
    p = int(v.get("popsize", plan.popsize))
    k = int(v.get("seeds_per_theta", 1)) if v.get("kind") == "robust" else 1
    per_gen = r * p * k
    return r, p, k, plan.budget // per_gen


def _check_variant_budget(name, v, plan):
    r, p, k, g = variant_plan(v, plan)
    if g < 1 or g * r * p * k != plan.budget:
        raise ValueError(f"variant {name!r}: restarts*popsize*seeds_per_theta = {r * p * k} "
                         f"must divide the budget {plan.budget} exactly")


def regime_space(space: ThetaSpace, regime: str) -> ThetaSpace:
    """The search box for a regime.

    ``broad``: the stress-test ``bounds`` unchanged. ``conservative``: each dim's
    ``plausible`` deployment band becomes its bounds (nominal clipped into the band).
    ``near_nominal``: each plausible band shrunk toward nominal to NEAR_NOMINAL_FRACTION of
    its distance on each side (a well-controlled deployment; failures should be rare).
    The plausible band itself is kept so the engine's plausibility weighting is unchanged.
    """
    if regime == "broad":
        return space
    if regime not in ("conservative", "near_nominal"):
        raise ValueError(regime)
    dims = {}
    for n, d in space.dims.items():
        lo, hi = d.plausible
        if not hi > lo:
            raise ValueError(f"{n}: degenerate plausible band {d.plausible}")
        nom = min(max(d.nominal, lo), hi)
        if regime == "near_nominal":
            f = NEAR_NOMINAL_FRACTION
            lo, hi = nom - f * (nom - lo), nom + f * (hi - nom)
            if not hi > lo:      # nominal sits at both edges: keep a sliver so the box is valid
                hi = lo + 1e-6
        dims[n] = dataclasses.replace(d, bounds=(float(lo), float(hi)), nominal=float(nom))
    return ThetaSpace(dims=dims)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id(policy: str, regime: str, method: str, rep: int) -> str:
    return f"{policy}__{regime}__{method}__rep{rep}"


__all__ = ["BenchConfig", "PolicyConfig", "SearchPlan", "load_config", "regime_space",
           "sha256_file", "run_id", "REGIMES", "METHODS", "METHOD_LABELS", "ThetaDim"]
