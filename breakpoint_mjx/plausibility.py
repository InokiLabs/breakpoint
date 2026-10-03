"""Plausibility gate for LLM-proposed failure worlds. Pure stdlib + the ThetaSpace.

WHY THIS EXISTS
---------------
A customer dismisses a failure report the moment it reads "your robot falls on a
frictionless ice rink with a broken IMU and 6 kg bolted on". That world is a failure of
the *simulator*, not of the policy -- and every implausible proposal also burns GPU
budget that could have found a real defect. ``llm_search._parse_configs`` only CLIPS to
``bounds``; clipping keeps a proposal inside the box but says nothing about whether the
corner of the box it lands in is a machine anyone could build.

THREE SEPARABLE LAYERS, increasing strictness. A caller picks how far to go.

  1. ``impossible_reasons``  -- hard reject, never simulate. Physically incoherent or
     degenerate: NaN, negative friction, non-positive total mass, a latency the action
     ring buffer cannot express, a fraction outside [0, 1]. Each rule is derived from the
     space (or the model, when the caller passes real masses) rather than hardcoded, and
     every reason string names the rule, the dim and the offending value.

  2. ``implausibility``      -- within bounds but not credible hardware. Combines the
     EXISTING ``transfer.transfer_likelihood`` (depth of violation, per-dim bands) with
     two JOINT notions the per-dim bands cannot express:
        * n_outside  -- how many axes sit outside their plausible band at all (breadth),
        * n_extreme  -- how many axes are pinned NEAR THEIR WORST HARDWARE EXTREME.
     n_extreme is the one that catches the ice-rink report: three axes at their hard
     bounds and thirteen at nominal scores transfer 0.81 and n_outside 3, which every
     per-dim check happily waves through. It is the simultaneity that is incredible.

  3. ``gate`` (+ repair)     -- the batch-level entry point. REPAIR IS PREFERRED OVER
     REJECTION: pull the worst offenders back just far enough to pass while PROTECTING
     the axes the proposal is most about, so the LLM's intent (which axes it wanted to
     attack) survives and the LLM call is not wasted. A proposal that would need more
     than a third of its dims rewritten is not a repairable near-miss -- it is a
     different world -- so it is rejected as an unsurvivable corner.

CALIBRATION (measured, not guessed)
-----------------------------------
Against the engine's own shipped findings in ``media/demo/demo_measurements.json`` --
four representative thetas from real GPU runs, the exact latency+dropout+slope worlds the
product reports:

    condition 0/1/2 : transfer 1.000, n_outside 0, n_extreme 0
    condition 3     : transfer 0.849, n_outside 4, n_extreme 0  (max per-dim excess 0.758)
    all 16 axes pinned to their worst bound: transfer 0.063, n_outside 16, n_extreme 16

Every default below is placed with visible daylight between the real findings and the
degenerate corner. A gate tuned tighter would reject exactly the interesting worlds and
quietly turn the LLM proposer into a nominal-config generator -- see
``tests/test_plausibility.py::test_real_reported_failures_pass_the_gate``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from breakpoint_mjx.config import ThetaSpace
from breakpoint_mjx.transfer import _dim_plausibility, transfer_likelihood

# --------------------------------------------------------------------------------------
# Defaults. Each one is justified against the calibration numbers in the module docstring.
# --------------------------------------------------------------------------------------

#: Minimum ``transfer_likelihood``. The weakest REAL reported failure scores 0.849; the
#: all-axes-pinned corner scores 0.063. 0.55 sits ~0.30 below every real finding and well
#: above the corner. Read directly: at most ~45% of the space's total plausibility budget
#: may be spent, i.e. roughly seven of sixteen axes fully pinned to hardware limits.
MIN_TRANSFER = 0.55

#: Fraction of the beyond-band region past which a dim counts as pinned at its worst
#: hardware extreme. 1.0 == exactly at the hard bound. The most extreme axis in any real
#: reported failure measures 0.758, so 0.85 leaves that finding clear headroom while still
#: meaning "essentially the worst hardware this space admits".
EXTREME_FRAC = 0.85

#: How many axes may sit at that extreme SIMULTANEOUSLY. This is the unsurvivable-corner
#: budget and the rule that catches "ice rink + broken IMU + 6 kg payload" (n_extreme 3).
#: Two compounding worst-case axes is already a strong adversarial claim; a third is a
#: tail event no fleet operator will accept as their bug. Deliberately generous in
#: practice because exceeding it triggers REPAIR, not rejection: the two dominant attacked
#: axes stay pinned and only the surplus is pulled back.
MAX_EXTREME = 2

#: Breadth cap on axes outside their plausible band, as a fraction of the space's
#: dimension count (ceil). 0.45 -> 8 of go1's 16 dims, 5 of panda's 10. Real reported
#: failures use 4 of 16, so this admits twice the breadth the engine actually reports.
MAX_OUTSIDE_FRAC = 0.45

#: Repair budget: the fraction of dims ``repair`` may rewrite. Beyond it the proposal is
#: not a near-miss to be nudged but a different world, and is rejected. 1/3 -> 5 of 16.
MAX_REPAIR_FRAC = 1.0 / 3.0

#: Relative slack when comparing against hard bounds, so float round-tripping through
#: JSON does not manufacture an out_of_bounds reason.
_TOL = 1e-9


def max_outside_default(space: ThetaSpace) -> int:
    """Breadth cap for ``space``: at least 4 axes, else ceil(MAX_OUTSIDE_FRAC * n_dims)."""
    return max(4, math.ceil(MAX_OUTSIDE_FRAC * len(space.dims)))


def max_repair_default(space: ThetaSpace) -> int:
    """Repair budget for ``space``: at least 1 dim, else floor(MAX_REPAIR_FRAC * n_dims)."""
    return max(1, int(MAX_REPAIR_FRAC * len(space.dims)))


# --------------------------------------------------------------------------------------
# Layer 1: IMPOSSIBLE
# --------------------------------------------------------------------------------------
# Semantic classifiers. Name-based on purpose: the theta spaces (go1 / g1 / barkour /
# panda / URDF-derived) share a naming convention, so a rule written once covers robots
# this file has never seen, and a space that invents a new dim simply skips the rule
# rather than crashing.

_MASS_ADD_RE = re.compile(r"mass.*add|add.*mass", re.I)


def _is_mass_add(name: str) -> bool:
    return bool(_MASS_ADD_RE.search(name))


def _is_mass_scale(name: str) -> bool:
    return "mass" in name.lower() and name.lower().endswith("_scale")


def _is_friction(name: str) -> bool:
    return "friction" in name.lower()


def _is_unit_interval(name: str) -> bool:
    """Probabilities and fractions: ``*_prob`` / ``*_frac`` are [0, 1] by definition."""
    n = name.lower()
    return n.endswith("_prob") or n.endswith("_frac")


def _is_latency(name: str) -> bool:
    return "latency" in name.lower()


def _is_multiplicative_scale(name: str, dim) -> bool:
    """A ``*_scale`` dim whose NOMINAL is positive is a multiplier on a physical property.

    Derived, not listed: ``kp_scale``/``joint_armature_scale`` have nominal 1.0 (multiply
    an existing gain/inertia -> zero or negative is singular or non-physical), whereas
    ``obs_noise_scale`` has nominal 0.0 and is a magnitude, for which zero is the correct
    noise-free world. The nominal is what separates them, so no per-robot list is needed.
    """
    return name.lower().endswith("_scale") and dim.nominal > 0.0


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def impossible_reasons(theta_named, space: ThetaSpace, horizon=None,
                       body_masses=None) -> list[str]:
    """Reasons ``theta_named`` describes a world that cannot physically exist.

    A non-empty result means HARD REJECT: never spend a simulation slot on it. The
    trajectory such a world produces is meaningless, not a policy failure.

    Args:
        theta_named: dim name -> value. Missing dims are treated as their nominal (the
            same convention ``_parse_configs`` uses), so a partial proposal is judged on
            what it actually says.
        space: the ThetaSpace whose ``bounds``/``nominal`` define the physical envelope.
        horizon: rollout length in control steps. When given, a latency at or beyond it
            is impossible -- the action ring buffer cannot express a delay the episode
            never reaches, so the policy's commands would never be applied at all.
        body_masses: optional {mass-add dim name -> base body mass in kg}, from the real
            MJX model. Supplying it makes the non-positive-total-mass rule bite on the
            actual robot instead of only on the space's declared lower bound.

    Returns:
        A list of human-readable reasons, each naming the rule, the dim and the value.
        Empty means "coherent enough to simulate" (NOT "plausible" -- that is layer 2).
    """
    reasons: list[str] = []
    masses = dict(body_masses or {})

    for name, dim in space.dims.items():
        raw = theta_named.get(name, dim.nominal)

        # R1 non_finite. Checked first and `continue`d: every later rule compares numbers,
        # and NaN silently wins comparisons. Note this rule genuinely bites downstream of
        # _parse_configs, whose min(max(...)) clip PROPAGATES NaN rather than removing it.
        if not _finite(raw):
            reasons.append(f"non_finite: {name}={raw!r} is not a finite number")
            continue
        v = float(raw)
        lo, hi = dim.bounds

        # R2 out_of_bounds. The declared bounds ARE the space's statement of what hardware
        # can exist; outside them the model perturbation itself is undefined.
        span = max(abs(hi - lo), 1.0)
        if v < lo - _TOL * span or v > hi + _TOL * span:
            reasons.append(
                f"out_of_bounds: {name}={v:g} outside its physical bounds [{lo:g}, {hi:g}]")

        # R3 negative_friction. A negative Coulomb coefficient (or a negative multiplier
        # on one) is a contact that ADDS energy: the simulator will produce motion no real
        # surface can. Applies to absolute coefficients and to *_friction_scale alike.
        if _is_friction(name) and v < 0.0:
            reasons.append(
                f"negative_friction: {name}={v:g} < 0; a negative friction coefficient is "
                "an energy-generating contact, not a slippery floor")

        # R4 nonpositive_multiplicative_scale. Zero or negative multiplier on an existing
        # gain / inertia / friction-loss: a massless or sign-inverted actuator.
        if _is_multiplicative_scale(name, dim) and v <= 0.0:
            reasons.append(
                f"nonpositive_scale: {name}={v:g} <= 0; a multiplicative scale on a "
                f"physical property (nominal {dim.nominal:g}) must stay positive")

        # R5 nonpositive_total_mass, two forms.
        if _is_mass_scale(name) and v <= 0.0:
            reasons.append(
                f"nonpositive_mass: {name}={v:g} <= 0 scales a body mass to zero or below")
        if _is_mass_add(name) and name in masses:
            base = float(masses[name])
            if base + v <= 0.0:
                reasons.append(
                    f"nonpositive_mass: {name}={v:g} added to a {base:g} kg body gives a "
                    f"total mass of {base + v:g} kg (<= 0)")

        # R6 unit_interval. A probability or a phase fraction outside [0, 1] is not a
        # severe world, it is a meaningless one (push_step_frac indexes into the episode).
        if _is_unit_interval(name) and not (0.0 <= v <= 1.0):
            reasons.append(
                f"unit_interval: {name}={v:g} outside [0, 1]; it is a "
                f"{'probability' if name.lower().endswith('_prob') else 'fraction'}")

        # R7 negative_magnitude. Dims that are pure magnitudes by construction (nominal ==
        # lower bound == 0: noise stddev, dropout rate, push speed, latency) have no
        # meaningful negative side -- a negative delay is time travel.
        if v < 0.0 and dim.nominal == 0.0 and lo == 0.0:
            reasons.append(
                f"negative_magnitude: {name}={v:g} < 0; this dim is a non-negative "
                "magnitude (nominal 0 at its lower bound)")

        # R8 latency_exceeds_horizon. The rollout's action ring buffer is indexed by the
        # delay; a delay at or past the episode length means no commanded action is ever
        # applied, so the run measures the initial buffer, not the policy.
        if _is_latency(name) and horizon is not None and round(v) >= int(horizon):
            reasons.append(
                f"latency_exceeds_horizon: {name}={v:g} steps >= rollout horizon "
                f"{int(horizon)}; the action buffer cannot express this delay")

    return reasons


# --------------------------------------------------------------------------------------
# Layer 2: IMPLAUSIBLE
# --------------------------------------------------------------------------------------


def dim_excess(value: float, dim) -> float:
    """How far outside its plausible band a value sits, as a fraction of the room left.

    0.0 == inside the band; 1.0 == at the hard bound, i.e. the worst hardware the space
    admits on this axis. Exactly ``1 - transfer._dim_plausibility``, so this layer and the
    existing transfer heuristic cannot drift apart.
    """
    return 1.0 - _dim_plausibility(value, dim.plausible, dim.bounds)


def implausibility(theta_named, space: ThetaSpace) -> dict:
    """Score how credible ``theta_named`` is as real hardware. No thresholds applied here.

    Returns a dict with:
        transfer   -- ``transfer.transfer_likelihood`` (the existing per-dim heuristic).
        n_outside  -- how many dims sit outside their plausible band at all (breadth).
        n_extreme  -- how many dims are pinned at/past ``EXTREME_FRAC`` of the way to
                      their hard bound (the joint "unsurvivable corner" signal).
        max_excess -- the single worst axis's excess.
        outside    -- [{name, value, plausible, bounds, excess}], worst-first.
        worst      -- dim names, worst-first (the axes the proposal is most "about").
    """
    filled = {n: float(theta_named.get(n, d.nominal)) for n, d in space.dims.items()}
    rows = []
    for name, dim in space.dims.items():
        e = dim_excess(filled[name], dim)
        if e > 0.0:
            rows.append({"name": name, "value": filled[name], "plausible": dim.plausible,
                         "bounds": dim.bounds, "excess": e})
    rows.sort(key=lambda r: r["excess"], reverse=True)
    return {
        "transfer": transfer_likelihood(filled, space),
        "n_outside": len(rows),
        "n_extreme": sum(1 for r in rows if r["excess"] >= EXTREME_FRAC),
        "max_excess": rows[0]["excess"] if rows else 0.0,
        "outside": rows,
        "worst": [r["name"] for r in rows],
    }


def implausible_reasons(theta_named, space: ThetaSpace, *, min_transfer=MIN_TRANSFER,
                        max_outside=None, max_extreme=MAX_EXTREME) -> list[str]:
    """Reasons ``theta_named`` is within bounds yet not credible hardware. Empty = fine."""
    if max_outside is None:
        max_outside = max_outside_default(space)
    m = implausibility(theta_named, space)
    reasons: list[str] = []
    if m["transfer"] < min_transfer:
        reasons.append(
            f"low_transfer: transfer_likelihood {m['transfer']:.3f} < {min_transfer:.2f} "
            f"({m['n_outside']} of {len(space.dims)} dims outside their realistic band)")
    if m["n_extreme"] > max_extreme:
        pinned = ", ".join(
            f"{r['name']}={r['value']:g}" for r in m["outside"][:m["n_extreme"]])
        reasons.append(
            f"unsurvivable_corner: {m['n_extreme']} dims pinned at their worst hardware "
            f"extreme simultaneously (budget {max_extreme}): {pinned}")
    if m["n_outside"] > max_outside:
        reasons.append(
            f"too_many_implausible_dims: {m['n_outside']} dims outside their realistic "
            f"band (budget {max_outside}): {', '.join(m['worst'][:max_outside + 1])}...")
    return reasons


# --------------------------------------------------------------------------------------
# Layer 3: REPAIR
# --------------------------------------------------------------------------------------


def _value_at_excess(dim, value: float, target_excess: float) -> float:
    """The value on the SAME side of the plausible band as ``value`` with ``target_excess``.

    Keeps the direction of attack (a proposal that wanted a slippery floor still gets a
    slippery floor, just a buildable one) -- only the magnitude is walked back.
    """
    lo_p, hi_p = dim.plausible
    lo_b, hi_b = dim.bounds
    t = min(max(target_excess, 0.0), 1.0)
    if value < lo_p:
        return lo_p - t * max(lo_p - lo_b, 0.0)
    if value > hi_p:
        return hi_p + t * max(hi_b - hi_p, 0.0)
    return value


@dataclass(frozen=True)
class Change:
    """One dim rewritten by ``repair``."""
    dim: str
    was: float
    now: float
    rule: str          # which implausibility rule forced this change

    def __str__(self) -> str:
        return f"{self.dim} {self.was:g}->{self.now:g} ({self.rule})"


@dataclass(frozen=True)
class RepairResult:
    ok: bool                          # did the repaired config end up passing?
    cfg: dict                         # repaired config (== input when nothing changed)
    changes: list[Change]
    reasons: list[str]                # the violations repair was asked to fix
    give_up: str | None = None        # why repair was abandoned, when ok is False


def repair(theta_named, space: ThetaSpace, *, min_transfer=MIN_TRANSFER, max_outside=None,
           max_extreme=MAX_EXTREME, max_changes=None, protect=None) -> RepairResult:
    """Pull an implausible config back into credibility while keeping its attack intent.

    The rule that makes repair worth preferring over rejection: the axes the proposal
    pushed HARDEST are what the LLM was trying to say. So the top ``protect`` offenders
    (default ``max_extreme``, i.e. the two dominant attacked axes) are held fixed, and
    everything else is walked back -- extremes first, then the cheapest band violations,
    then the worst remaining offenders until transfer clears its floor.

    ``max_changes`` (default ``max_repair_default(space)``, one third of the dims) is what
    separates a near-miss from a different world. A proposal needing more rewrites than
    that is abandoned: rewriting most of a config is not honouring the LLM's intent, it is
    inventing a config and pretending the LLM asked for it.

    Returns a RepairResult; ``ok is False`` means the caller should reject.
    """
    if max_outside is None:
        max_outside = max_outside_default(space)
    if max_changes is None:
        max_changes = max_repair_default(space)
    if protect is None:
        protect = max_extreme

    cfg = {n: float(theta_named.get(n, d.nominal)) for n, d in space.dims.items()}
    reasons = implausible_reasons(cfg, space, min_transfer=min_transfer,
                                  max_outside=max_outside, max_extreme=max_extreme)
    if not reasons:
        return RepairResult(True, cfg, [], [])

    # The protected set is fixed up front from the ORIGINAL proposal, so a long repair
    # cannot drift into rewriting the very axes it set out to preserve.
    ranked = implausibility(cfg, space)["worst"]
    protected = set(ranked[:max(0, int(protect))])
    changes: list[Change] = []

    def _set(name, new, rule):
        changes.append(Change(name, cfg[name], float(new), rule))
        cfg[name] = float(new)

    def _candidates():
        """Offending, unprotected, not-yet-repaired dims, worst-first."""
        done = {c.dim for c in changes}
        return [r for r in implausibility(cfg, space)["outside"]
                if r["name"] not in protected and r["name"] not in done]

    # Pass 1 -- the unsurvivable corner. Surplus extreme axes get walked back to just
    # inside the extreme threshold (still heavily attacked, no longer at the hard bound).
    m = implausibility(cfg, space)
    if m["n_extreme"] > max_extreme:
        target = EXTREME_FRAC * 0.9
        for r in [c for c in _candidates() if c["excess"] >= EXTREME_FRAC]:
            if implausibility(cfg, space)["n_extreme"] <= max_extreme:
                break
            if len(changes) >= max_changes:
                break
            _set(r["name"], _value_at_excess(space.dims[r["name"]], r["value"], target),
                 "unsurvivable_corner")

    # Pass 2 -- breadth. Snap the MILDEST offenders fully back inside their band: they
    # carry the least attack intent per dim removed, so this is the cheapest way to buy
    # breadth back.
    while implausibility(cfg, space)["n_outside"] > max_outside and len(changes) < max_changes:
        cands = _candidates()
        if not cands:
            break
        r = cands[-1]
        _set(r["name"], _value_at_excess(space.dims[r["name"]], r["value"], 0.0),
             "too_many_implausible_dims")

    # Pass 3 -- depth. Worst offender first, since it buys the most transfer per dim.
    while (implausibility(cfg, space)["transfer"] < min_transfer
           and len(changes) < max_changes):
        cands = _candidates()
        if not cands:
            break
        r = cands[0]
        _set(r["name"], _value_at_excess(space.dims[r["name"]], r["value"], 0.0),
             "low_transfer")

    left = implausible_reasons(cfg, space, min_transfer=min_transfer,
                               max_outside=max_outside, max_extreme=max_extreme)
    if left:
        return RepairResult(
            False, cfg, changes, reasons,
            give_up=(f"unrepairable: a {max_changes}-dim repair budget was not enough "
                     f"({len(changes)} dims changed); still {'; '.join(left)}"))
    return RepairResult(True, cfg, changes, reasons)


# --------------------------------------------------------------------------------------
# The batch-level gate
# --------------------------------------------------------------------------------------


#: Tie-break order for the one-line summary, most diagnostic first. Without it a plain
#: alphabetical sort buries ``unsurvivable_corner`` -- the rule that actually explains a
#: rejection -- behind ``low_transfer``/``too_many_implausible_dims``, which are its
#: symptoms, and ``unrepairable``, which is only a consequence and is ranked last.
_RULE_ORDER = {
    "non_finite": 0, "out_of_bounds": 1, "negative_friction": 2, "nonpositive_scale": 3,
    "nonpositive_mass": 4, "unit_interval": 5, "negative_magnitude": 6,
    "latency_exceeds_horizon": 7,
    "unsurvivable_corner": 10, "low_transfer": 11, "too_many_implausible_dims": 12,
    "unrepairable": 90,
}


@dataclass(frozen=True)
class Repaired:
    original: dict
    cfg: dict
    changes: list[Change]
    reasons: list[str]


@dataclass(frozen=True)
class Rejected:
    cfg: dict
    reasons: list[str]
    kind: str          # "impossible" | "implausible"


@dataclass(frozen=True)
class GateResult:
    """What the gate did to one round of proposals.

    Invariant (asserted in the tests): ``len(kept) + len(repaired) + len(rejected)``
    equals the number of configs handed in.
    """
    kept: list[dict] = field(default_factory=list)          # passed untouched
    repaired: list[Repaired] = field(default_factory=list)  # passed after being nudged
    rejected: list[Rejected] = field(default_factory=list)  # never simulated
    counts: dict = field(default_factory=dict)

    @property
    def accepted(self) -> list[dict]:
        """Everything worth simulating, in input order-ish: kept first, then repaired."""
        return list(self.kept) + [r.cfg for r in self.repaired]

    def summary(self) -> str:
        """One operator-facing line. A SILENT GATE IS A BAD GATE: if the LLM is producing
        junk, that has to be visible in the run log, not buried in a return value."""
        s = (f"gate: {len(self.kept)} kept, {len(self.repaired)} repaired, "
             f"{len(self.rejected)} rejected")
        if self.rejected:
            # Tally the RULE names (the token before the first colon), most common first,
            # and show at most three so one pathological round cannot flood the log.
            tally: dict[str, int] = {}
            for r in self.rejected:
                for reason in r.reasons or ["unspecified"]:
                    rule = reason.split(":")[0]
                    tally[rule] = tally.get(rule, 0) + 1
            top = sorted(tally.items(),
                         key=lambda kv: (-kv[1], _RULE_ORDER.get(kv[0], 50), kv[0]))[:3]
            s += " (" + ", ".join(f"{k} x{v}" if v > 1 else k for k, v in top) + ")"
        return s


def gate(cfgs, space: ThetaSpace, *, min_transfer=MIN_TRANSFER, max_outside=None,
         max_extreme=MAX_EXTREME, repair_enabled=True, horizon=None, body_masses=None,
         max_changes=None) -> GateResult:
    """Filter a round of proposals into kept / repaired / rejected.

    Layer 1 (impossible) is unconditional -- those worlds are never simulated and are
    never repaired, because "repairing" an incoherent world means inventing a different
    one. Layer 2 (implausible) is repaired when ``repair_enabled`` (the default) and
    rejected otherwise.

    ``horizon`` and ``body_masses`` are forwarded to ``impossible_reasons``; both are
    optional, and omitting them only makes the hard layer more permissive, never wrong.
    """
    if max_outside is None:
        max_outside = max_outside_default(space)
    res = GateResult(kept=[], repaired=[], rejected=[], counts={})
    n_impossible = n_implausible = 0

    for raw in cfgs:
        cfg = {n: raw.get(n, d.nominal) for n, d in space.dims.items()}
        hard = impossible_reasons(cfg, space, horizon=horizon, body_masses=body_masses)
        if hard:
            res.rejected.append(Rejected(cfg, hard, "impossible"))
            n_impossible += 1
            continue
        cfg = {k: float(v) for k, v in cfg.items()}
        soft = implausible_reasons(cfg, space, min_transfer=min_transfer,
                                   max_outside=max_outside, max_extreme=max_extreme)
        if not soft:
            res.kept.append(cfg)
            continue
        n_implausible += 1
        if not repair_enabled:
            res.rejected.append(Rejected(cfg, soft, "implausible"))
            continue
        rep = repair(cfg, space, min_transfer=min_transfer, max_outside=max_outside,
                     max_extreme=max_extreme, max_changes=max_changes)
        if rep.ok:
            res.repaired.append(Repaired(cfg, rep.cfg, rep.changes, rep.reasons))
        else:
            res.rejected.append(
                Rejected(cfg, soft + [rep.give_up or "unrepairable"], "implausible"))

    res.counts.update(
        input=len(res.kept) + len(res.repaired) + len(res.rejected),
        kept=len(res.kept), repaired=len(res.repaired), rejected=len(res.rejected),
        impossible=n_impossible, implausible=n_implausible,
        changed_dims=sum(len(r.changes) for r in res.repaired),
    )
    return res
