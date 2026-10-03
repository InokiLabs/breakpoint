"""Workstream 1a (T1): cross-config VALIDITY -- turn the transfer claim from a
hand-weighted plausibility heuristic into a MEASUREMENT.

The idea (spec section "1a. Cross-config agreement"): a real failure of a frozen
policy should survive a change of *physics config*, while a numerical artifact (a
failure that only exists because the solver was loose or the timestep coarse)
vanishes when the solver tightens. So we re-run each candidate failure's EXACT
rollout -- same theta, same recorded seed -- under K independent physics configs and
define::

    validity = fraction of configs (incl. the nominal reference) under which the
               failure still reproduces (failure_score >= the search threshold)

and rank/label by validity. `transfer_likelihood` (plausibility) is kept as a prior,
but validity is the primary, measured evidence (see `combined_transfer`).

Two evidence sources (spec):
  * **MJX config variation (primary, cheap):** `default_physics_configs()` builds K
    variants of the base MJX model by perturbing `model.opt` (solver Newton vs CG,
    `iterations`, `ls_iterations`, `timestep`) and contact softness (`geom_solref`).
    Re-evaluated via the existing `rollout_one` (see `backends.MjxBackend`).
  * **Independent C-MuJoCo backend (stronger):** a genuinely different implementation
    (`backends.MujocoCBackend`, host CPU) contributes ONE extra entry to the
    reproduction vector. Optional + import-guarded (see `backends.make_mujoco_c_backend`).

Import-safety: this module is authored on a CPU-only box (no jax/mujoco). The pure
helpers below -- config-variant building and the validity aggregation -- import ONLY
numpy + stdlib and are unit-tested locally. The re-sim orchestrator `validate_failures`
imports jax / the backends INSIDE the function (like the rest of the codebase), so
importing `crossval` never requires a GPU install.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from breakpoint_mjx.transfer import priority_key

# MJX constraint-solver types (mjtSolver enum): CG=1, Newton=2. MJX supports only these
# two (PGS=0 is NOT implemented in MJX), so those are the only solvers a config may set
# -- a PGS config would pass a naive check then blow up at runtime, so it is rejected by
# is_valid() rather than allowed through.
# VERIFY (Colab): confirm the running MJX build accepts these ints for opt.solver; if an
# older build lacks Newton, drop `newton_tight` to CG-only variants.
SOLVER_CG, SOLVER_NEWTON = 1, 2


# --------------------------------------------------------------------------- #
# Physics-config variants (pure: no jax needed to *describe* a config)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhysicsConfig:
    """A declarative perturbation of the base MJX model's solver / integration / contact.

    A ``None`` field means "leave the base model's value unchanged"; the ``*_scale``
    fields default to 1.0 (no-op). ``apply_physics_config`` turns this into an actual
    (modified) MJX model. ``reference=True`` marks the nominal config (the one that
    reproduces the ORIGINAL scored physics); exactly one config in a set should set it.

    Fields
    ------
    name          : short label shown per-config in the report.
    solver        : mjtSolver int (SOLVER_CG / SOLVER_NEWTON) or None to keep base.
    iterations    : major solver iterations, or None to keep base.
    ls_iterations : line-search iterations, or None to keep base.
    timestep_scale: multiply base ``opt.timestep`` (smaller = finer/more accurate).
    solref_scale  : multiply ``geom_solref[:, 0]`` (contact time constant; larger =
                    softer contact). The cheap, robust contact-softness knob.
    reference     : True for the nominal reference config.
    """

    name: str
    solver: Optional[int] = None
    iterations: Optional[int] = None
    ls_iterations: Optional[int] = None
    timestep_scale: float = 1.0
    solref_scale: float = 1.0
    reference: bool = False

    def is_noop(self) -> bool:
        """True if this config leaves the base model bit-for-bit unchanged."""
        return (
            self.solver is None
            and self.iterations is None
            and self.ls_iterations is None
            and self.timestep_scale == 1.0
            and self.solref_scale == 1.0
        )

    def is_valid(self) -> bool:
        """Sanity: positive scales, positive iteration counts, a known solver."""
        if self.timestep_scale <= 0.0 or self.solref_scale <= 0.0:
            return False
        if self.iterations is not None and self.iterations <= 0:
            return False
        if self.ls_iterations is not None and self.ls_iterations <= 0:
            return False
        # Only the two MJX-supported solvers; PGS (0) is rejected (MJX can't run it).
        if self.solver is not None and self.solver not in (SOLVER_CG, SOLVER_NEWTON):
            return False
        return True


def default_physics_configs() -> list[PhysicsConfig]:
    """A small default set: the nominal reference + K=6 independent variants.

    The variants span "looser than the search" to "tighter than the search" so that a
    failure surviving all of them is robust to numerical detail, not an artifact of one
    solver setting:

      * ``newton_tight``   -- Newton solver with many iterations (tightest constraint
                              resolution). An artifact of a loose solve dies here.
      * ``cg_solver``      -- a genuinely different algorithm (CG) at modest iterations.
      * ``fine_timestep``  -- half the timestep (more accurate integration).
      * ``coarse_timestep``-- double the timestep (a blow-up here still reads as a
                              failure, which is intentional -- coarse physics is a real
                              deployment risk, not a false positive to discard).
      * ``soft_contacts``  -- softer contact (2x ``geom_solref`` time constant).
      * ``stiff_contacts`` -- stiffer contact (0.5x ``geom_solref`` time constant).

    # VERIFY (Colab): the absolute iteration counts below are reasonable MuJoCo values,
    # not the base model's exact ones (we don't read the base here). If a variant is a
    # no-op relative to base (e.g. base already uses Newton@50), it simply agrees with
    # nominal -- harmless for validity. Tune counts once you print base opt on Colab.
    """
    return [
        PhysicsConfig("nominal", reference=True),
        PhysicsConfig("newton_tight", solver=SOLVER_NEWTON, iterations=50, ls_iterations=20),
        PhysicsConfig("cg_solver", solver=SOLVER_CG, iterations=20, ls_iterations=10),
        PhysicsConfig("fine_timestep", timestep_scale=0.5),
        PhysicsConfig("coarse_timestep", timestep_scale=2.0),
        PhysicsConfig("soft_contacts", solref_scale=2.0),
        PhysicsConfig("stiff_contacts", solref_scale=0.5),
    ]


def build_config_variants(configs=None) -> list[PhysicsConfig]:
    """Validate + return a config set (defaults to ``default_physics_configs()``).

    Guarantees the invariants the re-sim + report rely on: names are unique, every
    config is valid (positive scales / iteration counts / a known solver), and there is
    exactly one nominal reference. Raises ValueError otherwise.
    """
    configs = list(configs) if configs is not None else default_physics_configs()
    if not configs:
        raise ValueError("need at least one physics config")
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"physics config names must be unique, got {names}")
    bad = [c.name for c in configs if not c.is_valid()]
    if bad:
        raise ValueError(f"invalid physics configs: {bad}")
    n_ref = sum(1 for c in configs if c.reference)
    if n_ref != 1:
        raise ValueError(f"expected exactly one reference config, got {n_ref}")
    return configs


def apply_physics_config(base_model, config: PhysicsConfig):
    """Return a copy of ``base_model`` with ``config``'s solver/integration/contact edits.

    Uses the MJX model's ``tree_replace`` (the same API the codebase already uses for
    top-level fields, e.g. ``base_model.tree_replace({field: ...})``) with dotted paths
    for the nested ``opt`` fields, and the ``.at[...].multiply`` update for the top-level
    ``geom_solref`` array.

    # VERIFY (Colab): mjx.Model.tree_replace accepts dotted nested paths like
    # "opt.timestep" / "opt.solver". If a given MJX build does not, use the flax-struct
    # replace instead:
    #     model = base_model.replace(opt=base_model.opt.replace(
    #         solver=config.solver, iterations=config.iterations,
    #         ls_iterations=config.ls_iterations, timestep=new_timestep))
    # VERIFY (Colab): changing opt.solver (a small int) between configs forces an XLA
    # recompile of the re-sim; that is expected and bounded (top-N * K re-sims).
    # VERIFY (Colab): geom_solimp (contact impedance) is the other contact-softness knob
    # the spec names; solref (time constant) is used here as the robust default. Add a
    # solimp variant once base solimp is confirmed on Colab.
    """
    if config.is_noop():
        return base_model

    updates = {}
    if config.solver is not None:
        updates["opt.solver"] = config.solver
    if config.iterations is not None:
        updates["opt.iterations"] = config.iterations
    if config.ls_iterations is not None:
        updates["opt.ls_iterations"] = config.ls_iterations
    if config.timestep_scale != 1.0:
        updates["opt.timestep"] = base_model.opt.timestep * config.timestep_scale

    model = base_model.tree_replace(updates) if updates else base_model

    if config.solref_scale != 1.0:
        model = model.tree_replace(
            {"geom_solref": model.geom_solref.at[:, 0].multiply(config.solref_scale)}
        )
    return model


# --------------------------------------------------------------------------- #
# Validity aggregation (pure numpy/stdlib -- fully CPU-unit-tested)
# --------------------------------------------------------------------------- #
def reproduces(score: float, threshold: float) -> bool:
    """A failure "reproduces" under a config iff its re-scored failure_score is still
    at or above the SAME threshold the search used to call it a failure."""
    return bool(float(score) >= float(threshold))


def reproduction_vector(scores, threshold: float) -> list[bool]:
    """Per-config reproduction booleans from per-config re-scored failure_scores."""
    return [reproduces(s, threshold) for s in scores]


def validity_score(reproduction) -> float:
    """validity = fraction of configs under which the failure reproduced (0..1).

    Empty vector -> 0.0 (no evidence). This is the primary, measured evidence number.
    """
    rep = list(reproduction)
    if not rep:
        return 0.0
    return float(sum(1 for r in rep if r) / len(rep))


def combined_transfer(
    validity: Optional[float],
    plausibility: Optional[float],
    validity_weight: float = 0.75,
) -> float:
    """Combined transfer/evidence key: MEASURED validity dominates, plausibility is a prior.

    ``transfer_likelihood`` (plausibility, 0..1) answers "is this theta physically
    plausible?"; validity answers "does the failure actually survive independent
    physics?". We keep both but let the measurement lead:

        combined = w * validity + (1 - w) * plausibility        (validity measured)
        combined = plausibility                                 (validity is None)

    Default ``w = 0.75`` so a family that reproduces under most configs ranks above one
    that is merely plausible, while plausibility still breaks ties among un-reproduced
    (or not-yet-validated) failures. When validity is None (un-validated record), the
    report/ranking degrade cleanly to the original plausibility-only behavior.
    """
    p = float(plausibility) if plausibility is not None else 0.0
    if validity is None:
        return p
    w = float(validity_weight)
    return float(w * float(validity) + (1.0 - w) * p)


def select_top(records, top_n: Optional[int]):
    """Top-N records by the same priority key the clustering/report uses, so we validate
    the failures the report actually surfaces. top_n=None -> all records (ranked)."""
    ranked = sorted(
        records,
        key=lambda r: priority_key(
            r.severity, r.transfer_score or 0.0, r.failure_score
        ),
        reverse=True,
    )
    if top_n is None:
        return ranked
    return ranked[: max(0, int(top_n))]


def annotate_families(families, validated_records):
    """Copy each family's validated REPRESENTATIVE validity/reproduction onto the family.

    validate_failures annotates the representative FailureRecords; this lifts that
    measurement to the FailureFamily objects the report renders. Families whose
    representative was not validated keep validity=None (report falls back to plausibility).
    """
    by_id = {r.id: r for r in validated_records}
    for f in families:
        r = by_id.get(f.representative_id)
        if r is not None and r.validity is not None:
            f.validity = r.validity
            f.reproduction = r.reproduction
    return families


def rank_families_by_validity(families):
    """Order families by MEASURED validity (desc), then by the original priority_rank.

    Un-validated families (validity=None) sort to a sentinel of -1.0, so when NO family
    is validated the order is exactly the original priority order (report unchanged).
    """
    return sorted(
        families,
        key=lambda f: (
            -(f.validity if f.validity is not None else -1.0),
            f.priority_rank,
        ),
    )


# --------------------------------------------------------------------------- #
# Orchestrator (Colab-run: imports jax + backends lazily; VERIFY-noted)
# --------------------------------------------------------------------------- #
def validate_failures(
    target,
    space,
    records,
    policy_fn,
    cfg,
    key,
    *,
    env,
    consts,
    base_model,
    configs=None,
    top_n: Optional[int] = 8,
    extra_backends=None,
):
    """Re-sim the top-N failures under K physics configs; annotate validity + reproduction.

    For each selected failure we re-run its EXACT rollout -- reusing the recorded
    (batch_key, env_index) so the ONLY thing that changes across configs is the physics,
    not the seed (spec: "same theta, same key") -- under every physics config via the
    MJX backend, plus one re-sim per independent ``extra_backends`` entry (e.g. the
    C-MuJoCo backend). RAISES ValueError if a selected record predates exact-repro
    (Upgrade #6: no batch_key/env_index) -- its original scored rollout can't be
    reconstructed, so it is refused rather than validated against a fabricated key. A
    config "reproduces" the failure iff the re-scored failure_score is still >=
    ``cfg.failure_threshold``. Each record gets:

        record.reproduction = [bool per config, then per extra backend]
        record.validity     = fraction of those that reproduced

    Returns ``(validated_records, config_names)`` where config_names is the label vector
    aligned with each record.reproduction (config names followed by backend names), for
    the report's per-config display.

    ``key`` is accepted for signature stability but is intentionally NOT used for the
    re-sim: each failure is replayed under its OWN recorded (batch_key, env_index), which
    is the whole point of the measurement (physics varies, seed fixed).

    Cost note: this is OFF by default (BP_VALIDATE=1) -- it is top_n * (K configs +
    backends) extra rollouts. Keep top_n small (family representatives).

    # VERIFY (Colab): with the nominal config, the re-scored failure_score should match
    # the record's stored failure_score (same theta + same reconstructed key + base
    # model) -- a good sanity check that the reconstruction is exact.
    """
    from breakpoint_mjx.backends.mjx_backend import MjxBackend
    from breakpoint_mjx.objectives import LEGGED_OBJECTIVE

    configs = build_config_variants(configs)
    threshold = float(cfg.failure_threshold)
    thresholds = cfg.thresholds
    horizon = int(cfg.horizon)
    # T7: score re-sims with the ACTIVE target's objective (legged by default), so a
    # manipulation failure is re-scored by its own component_signals/failure_score.
    objective = getattr(target, "objective", None) or LEGGED_OBJECTIVE
    component_signals = objective.component_signals
    failure_score = objective.failure_score

    selected = select_top(records, top_n)
    mjx = MjxBackend(
        env=env,
        policy_fn=policy_fn,
        target=target,
        space=space,
        consts=consts,
        base_model=base_model,
        horizon=horizon,
    )

    config_names = [c.name for c in configs]
    backend_names = [b.name for b in (extra_backends or [])]

    for rec in selected:
        scores = []
        for c in configs:
            model = apply_physics_config(base_model, c)
            # Each backend reconstructs the failure's OWN recorded key (physics varies,
            # seed fixed); it raises on a legacy record rather than fabricating one.
            summary = mjx.simulate(rec, model=model)
            scores.append(failure_score(component_signals(summary, thresholds), horizon))
        for backend in extra_backends or []:
            summary = backend.simulate(rec)
            scores.append(failure_score(component_signals(summary, thresholds), horizon))
        rec.reproduction = reproduction_vector(scores, threshold)
        rec.validity = validity_score(rec.reproduction)

    return selected, config_names + backend_names
