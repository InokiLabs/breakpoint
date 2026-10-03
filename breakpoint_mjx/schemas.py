"""Data model for discovered failures, families, and the report. Pure stdlib."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class FailureRecord:
    theta_vector: list[float]
    theta_named: dict[str, float]
    failure_score: float
    component_signals: dict[str, float]
    rollout_signature: list[float]
    seed: int
    severity: str = "low"
    transfer_score: float | None = None
    priority_rank: int | None = None
    family_id: str | None = None
    id: str | None = None
    # Upgrade #6 (exact reproduction): the batch PRNGKey passed to evaluate(), as
    # two JSON-safe uint32 ints, plus the row index within that batch. render_failure
    # rebuilds the exact eval key via fold_in(uint32(batch_key), env_index). Default
    # None so records written before this change still deserialize.
    batch_key: list[int] | None = None
    env_index: int | None = None
    # Workstream 1a (T1, cross-config validity): MEASURED reproduction evidence, filled
    # in by crossval.validate_failures for the validated top-N. validity is the fraction
    # of independent physics configs -- the nominal reference + K MJX solver/contact
    # variants (+ any independent backend such as C-MuJoCo) -- under which this failure
    # still scores >= the search failure_threshold (i.e. still reproduces). reproduction
    # is the per-config boolean vector, in the same order as the configs passed to
    # validate_failures. Both default None so (a) records written before T1 still
    # deserialize with the same defaults discipline Upgrade #6 used, and (b) an
    # un-validated record (BP_VALIDATE off, or outside the validated top-N) keeps None
    # and the report falls back to the plausibility transfer_score.
    validity: float | None = None
    reproduction: list[bool] | None = None
    # Autolytic gain (breakpoint_mjx/autolytic.py): excess actuator work this world
    # cost the policy relative to a same-seed UNPERTURBED twin, divided by the energy
    # the perturbation injected. <=1 means the perturbation did the work (a shove);
    # >>1 means a small input recruited the controller into destroying itself, which
    # is a controller defect rather than a big push. None when not measured (the stage
    # is opt-in and costs a second rollout) or when the perturbation injected no
    # mechanical energy, e.g. a pure sensor attack -- see autolytic.classify.
    autolytic_gain: float | None = None
    autolytic_class: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "FailureRecord":
        return cls(**json.loads(s))


@dataclass
class FailureFamily:
    id: str
    name: str
    member_ids: list[str]
    representative_id: str
    triggering_parameters: dict[str, float]
    mean_failure_score: float
    mean_transfer_score: float
    priority_rank: int
    # T1 (cross-config validity): the family representative's MEASURED validity (fraction
    # of physics configs under which it reproduces) + the per-config reproduction vector,
    # copied from the validated representative record by crossval.annotate_families. None
    # until BP_VALIDATE annotates the family. Defaults keep old reports/JSON deserializing
    # (BreakPointReport.from_json reconstructs families via FailureFamily(**f)).
    validity: float | None = None
    reproduction: list[bool] | None = None
    # T2 (Workstream 1c, safety margins + attribution): filled by the BP_MARGINS stage
    # (margins.compute_margins). margin -- a human-readable boundary string ("stable until
    # friction_floor < 0.35 AND latency_steps > 2.5"); boundaries -- per-axis
    # {gates, boundary, direction, op, nominal, target}; attribution -- {axis_name:
    # importance in [0,1] summing to 1}. All default None so records/reports written before
    # T2 still deserialize (same back-compat discipline as T1's validity/Upgrade #6), and an
    # un-annotated family (BP_MARGINS off) simply omits these lines in the report.
    margin: str | None = None
    boundaries: dict | None = None
    attribution: dict | None = None
    # T3 (Workstream 2a, failure-regression fingerprinting): the family-MEAN discriminative
    # signature persisted by clustering.build_families. dominant_dims -- the ranked deviating
    # dims the displayed NAME is built from, as [[dev, name, direction], ...] (desc by dev);
    # mean_theta -- the family's per-dim mean theta. regression.family_fingerprint keys off
    # these (matches the name, independent of representative-row jitter) and falls back to
    # triggering_parameters when absent. Both default None so families/reports written before
    # T3 still deserialize (same back-compat discipline as T1/T2's optional fields).
    dominant_dims: list | None = None
    mean_theta: dict | None = None
    # Certified Minimal Cause (breakpoint_mjx/mincause.py, on by default, BP_MINCAUSE=0
    # disables): minimal_cause -- the 1-minimal set of dims that must be perturbed for the
    # failure to reproduce, as [{dim, value, direction}, ...] where value is the shrunk
    # minimal-counterexample value; min_distance -- its normalized L-inf distance from
    # nominal (fraction of span); cause_certified -- the representative itself certified as
    # failing across K seeds (Clopper-Pearson); cause_necessary -- reverting only the cause
    # (all other dims kept) stops the certified failure; merged_from -- ids of families
    # merged into this one because their certified causes coincided. All default None so
    # families/reports written before CMC still deserialize.
    minimal_cause: list | None = None
    min_distance: float | None = None
    cause_certified: bool | None = None
    cause_necessary: bool | None = None
    merged_from: list | None = None
    # Skeptic-grade evidence (crossval_engines.replay_stage, on by default, BP_REPLAY=0
    # disables). replay -- the independent C-MuJoCo FULL-collision replay of the CMC
    # minimal counterexample: {engine, collision_model, k, n, cp_lo, bar, reproduces,
    # min_floor_height, fall_steps, nominal_twin: {fell, min_floor_height}}; None when not
    # run. evidence_videos -- {"counterexample": mp4, "nominal": mp4} rendered side by side
    # (same seed, all other dims nominal, clipped 1 s after the fall). mjx_min_floor_height
    # -- lowest torso height above the floor in the MJX (search-scene) representative
    # rollout; the report flags a value below -0.02 m (a fallen trunk sinking through a
    # feet-only floor). All default None so older families still deserialize.
    replay: dict | None = None
    evidence_videos: dict | None = None
    mjx_min_floor_height: float | None = None
    # Cause-Stratified Operational Risk (breakpoint_mjx/famrisk.py, OFF by default,
    # BP_FAMRISK=1 enables): for a CMC-certified family, the exact mass of its minimal-cause
    # region under the declared operating domain p, the Clopper-Pearson interval on its
    # conditional failure rate from fresh rollouts, their product (risk_lo, risk_hi) and a
    # class in {in_domain_risk, in_domain_low, outside_odd}. SIMULATION probabilities.
    # None when the stage did not run or the family was not certified.
    odd_risk: dict | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def family_status(f) -> str:
    """'confirmed' | 'not_confirmed_outside_mjx' | 'unconfirmed' for a family.

    unconfirmed    -- the Certified Minimal Cause stage ran and could NOT certify the
                      representative under seed noise (cause_certified is False).
    not_confirmed_outside_mjx -- certified in MJX, but the independent C-MuJoCo
                      full-collision replay of its minimal counterexample did not clear the
                      same certification bar (verdict not_reproduced OR inconclusive).
    confirmed      -- everything else, including runs where CMC / replay did not run
                      (cause_certified None), which keep the pre-CMC behaviour.
    """
    if getattr(f, "cause_certified", None) is False:
        return "unconfirmed"
    rp = getattr(f, "replay", None) or {}
    if rp.get("reproduces") is False:
        return "not_confirmed_outside_mjx"
    return "confirmed"


def uncertified_name(f, max_dims=3) -> str:
    """'uncertified: dim_a, dim_b' from the family's clustering signature (no causal
    direction words, since nothing about the cause was certified)."""
    dims = [d[1] for d in (getattr(f, "dominant_dims", None) or [])
            if isinstance(d, (list, tuple)) and len(d) >= 2][:max_dims]
    if not dims:
        dims = list((getattr(f, "triggering_parameters", None) or {}).keys())[:max_dims]
    return "uncertified: " + (", ".join(dims) if dims else "no dominant dims")


@dataclass
class BreakPointReport:
    run_id: str
    policy_descriptor: str
    comparison: dict[str, float]
    families: list[FailureFamily] = field(default_factory=list)
    mp4_paths: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "BreakPointReport":
        d = json.loads(s)
        return cls(
            run_id=d["run_id"],
            policy_descriptor=d["policy_descriptor"],
            comparison=d["comparison"],
            families=[FailureFamily(**f) for f in d["families"]],
            mp4_paths=d["mp4_paths"],
        )
