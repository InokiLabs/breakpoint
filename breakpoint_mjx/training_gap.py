"""Bugs vs coverage gaps: did the policy fail inside the conditions it was trained on?

A failure on an axis the policy never saw in training (latency on a policy trained with no
delay, a push on one trained without pushes) is a COVERAGE GAP: expected, and the fix is
a randomization term, not a code change. A failure inside the trained ranges is a real
DEFECT of the policy, which is the finding a team cannot get from reading its own config.
Reports today mix the two, so a fragile demo policy's "failure families" are mostly a list
of randomization terms nobody added.

This module classifies each family against the declared TRAINING randomization and, for
coverage gaps, proposes the range to add. Pure numpy + yaml; no JAX.

What is classified. When the Certified Minimal Cause stage certified the family, its
``minimal_cause`` is the set of dims the failure needs, at their shrunk values, so a dim
the cause does not include cannot make a family a "gap". Without a certified cause (a
smoke budget certifies few families) the verdict is PROVISIONAL and uses the family's top
``FALLBACK_TOP`` dominant dims (the ones its name is built from) at the family-mean
values; a search representative moves almost every dim, so judging it whole would blame
every axis at once. Qualifier dims (``push_step_frac``: when a push lands) are never an
axis of their own.

Where the training randomization comes from, first match wins:
  1. ``BP_TRAIN_DR``: a YAML file with a ``training_dr:`` (or ``operational:``) block;
  2. ``BP_OBS_SPEC``: the policy contract's ``training_dr:`` block, else its
     ``operational:`` block when the contract was written by ``breakpoint init --from``
     (the importer maps the trainer's domain randomization into it);
  3. nothing: families are left unclassified (``verdict: unknown``).
Block format: ``{dim: {range: [lo, hi]}}``. A dim the block does not list was not
randomized, so the policy trained at its nominal value only.
"""
from __future__ import annotations

import math
import os

import yaml

#: Fraction of a dim's search span a value may sit outside the trained range and still
#: count as trained (float round-off, CMC shrink granularity).
TOL_FRAC = 0.02
#: Without a minimal cause, dims the representative moves less than this fraction of their
#: span from nominal are ignored.
MIN_DEV_FRAC = 0.05
#: Dims used for a family without a certified minimal cause.
FALLBACK_TOP = 3
#: Dims that only qualify another dim (push timing means nothing without a push).
QUALIFIER_DIMS = frozenset({"push_step_frac"})
#: Dims a trainer configures in whole units (control steps of delay): patch ends round out.
INTEGER_DIMS = frozenset({"latency_steps"})
#: A suggested range covers the failing value plus this fraction of the gap beyond it,
#: so the retrained policy sees the failing world inside, not on the edge of, training.
PATCH_HEADROOM = 0.10

# Where each dim lives in the frameworks teams train with, for the patch's hints. A dim
# without an entry gets a generic hint.
FRAMEWORK_HINTS = {
    "friction_floor": "legged_gym domain_rand.friction_range; Isaac Lab "
                      "randomize_rigid_body_material static_friction_range",
    "torso_mass_add_kg": "legged_gym domain_rand.added_mass_range; Isaac Lab "
                         "randomize_rigid_body_mass (operation add, base body)",
    "com_offset_x_m": "Isaac Lab randomize_rigid_body_com com_range.x",
    "com_offset_y_m": "Isaac Lab randomize_rigid_body_com com_range.y",
    "com_offset_z_m": "Isaac Lab randomize_rigid_body_com com_range.z",
    "push_mag": "legged_gym domain_rand.max_push_vel_xy (push_robots); Isaac Lab "
                "push_by_setting_velocity velocity_range",
    "latency_steps": "action delay in control steps (e.g. Isaac Lab DelayedPDActuator "
                     "min/max_delay, or a delay buffer in your env step)",
    "kp_scale": "PD stiffness randomization (Isaac Lab randomize_actuator_gains "
                "stiffness_distribution_params, operation scale)",
    "kd_scale": "PD damping randomization (Isaac Lab randomize_actuator_gains "
                "damping_distribution_params, operation scale)",
    "joint_frictionloss_scale": "Isaac Lab randomize_joint_parameters "
                                "friction_distribution_params (operation scale)",
    "joint_armature_scale": "Isaac Lab randomize_joint_parameters "
                            "armature_distribution_params (operation scale)",
    "obs_noise_scale": "observation noise (Isaac Lab ObsTerm noise / your obs noise scale)",
    "obs_bias_scale": "a per-episode constant bias on the sensor observations",
    "obs_dropout_prob": "per-step sensor dropout with sample-and-hold",
    "floor_slope_rad": "terrain: train on slopes (Isaac Lab terrain generator)",
}


def _block_from_yaml(path):
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        return None
    for key in ("training_dr", "operational"):
        if isinstance(doc.get(key), dict) and doc[key]:
            return doc[key]
    return None


def load_training_dr(space, environ=None):
    """``({dim: (lo, hi)}, source)`` for every dim of ``space``, or ``(None, reason)``.

    Dims the block does not list are pinned at their nominal value (trained, but only
    there). Ranges are clipped to the search bounds."""
    env = os.environ if environ is None else environ
    block, source = None, None
    for var in ("BP_TRAIN_DR", "BP_OBS_SPEC"):
        path = env.get(var)
        if not path:
            continue
        try:
            block = _block_from_yaml(path)
        except (OSError, yaml.YAMLError):
            block = None
        if block:
            source = f"{var} ({path})"
            break
    if not block:
        return None, ("no training randomization declared (set BP_TRAIN_DR, or a contract "
                      "from `breakpoint init --from ...`)")
    return training_ranges(space, block), source


def training_ranges(space, block):
    """``{dim: (lo, hi)}`` from a ``{dim: {range: [lo, hi]}}`` block (see module doc)."""
    out = {}
    for name, dim in space.dims.items():
        o = (block or {}).get(name) or {}
        if "range" in o:
            lo, hi = float(o["range"][0]), float(o["range"][1])
            lo, hi = max(min(lo, hi), dim.bounds[0]), min(max(lo, hi), dim.bounds[1])
        else:
            lo = hi = float(dim.nominal)
        out[name] = (lo, hi)
    return out


def _span(dim):
    return max(float(dim.bounds[1]) - float(dim.bounds[0]), 1e-12)


def cause_dims(family, space):
    """``([(dim, value)], source)`` the classification looks at (see module doc)."""
    def usable(name):
        return name in space.dims and name not in QUALIFIER_DIMS

    mc = getattr(family, "minimal_cause", None)
    if mc:
        return ([(c["dim"], float(c["value"])) for c in mc if usable(c.get("dim"))],
                "minimal_cause")
    dd = getattr(family, "dominant_dims", None) or []
    mt = getattr(family, "mean_theta", None) or {}
    if dd and mt:
        names = [n for _dev, n, _dir in dd if usable(n) and n in mt]
        return [(n, float(mt[n])) for n in names[:FALLBACK_TOP]], "dominant_dims"
    theta = getattr(family, "triggering_parameters", None) or {}
    moved = []
    for name, dim in space.dims.items():
        if usable(name) and name in theta:
            dev = abs(float(theta[name]) - float(dim.nominal)) / _span(dim)
            if dev > MIN_DEV_FRAC:
                moved.append((dev, name, float(theta[name])))
    moved.sort(reverse=True)
    return [(n, v) for _d, n, v in moved[:FALLBACK_TOP]], "representative"


def classify(family, space, trained) -> dict:
    """``{verdict, outside, cause_source}`` for one family.

    verdict ``defect``: every cause dim sits inside the trained ranges, so the policy
    fails in conditions it was trained on. ``coverage_gap``: at least one cause dim is
    outside; ``outside`` lists them, largest excess first, with the value, the trained
    range and the excess as a fraction of the dim's span. ``unknown``: no training
    randomization declared (``trained`` is None). ``provisional`` is True when the
    family had no certified minimal cause."""
    dims, src = cause_dims(family, space)
    prov = src != "minimal_cause"
    if trained is None:
        return {"verdict": "unknown", "outside": [], "cause_source": src,
                "provisional": prov}
    rows = []
    for name, v in dims:
        lo, hi = trained[name]
        dim = space.dims[name]
        excess = max(lo - v, v - hi, 0.0) / _span(dim)
        if excess > TOL_FRAC:
            rows.append({"dim": name, "value": v, "trained": [lo, hi],
                         "excess": round(excess, 4),
                         "beyond_plausible": not (dim.plausible[0] <= v <= dim.plausible[1])})
    rows.sort(key=lambda r: r["excess"], reverse=True)
    return {"verdict": "coverage_gap" if rows else "defect", "outside": rows,
            "cause_source": src, "provisional": prov}


def suggest_patch(classified, space, trained) -> dict:
    """``{dim: {"range": [lo, hi], "was": [lo, hi], "hint": str, "families": [...]}}``.

    Each coverage-gap dim widens its trained range to cover the farthest failing value
    across families plus ``PATCH_HEADROOM`` of the gap, clipped to the dim's PLAUSIBLE
    band: training on hardware no one will deploy only buys a slower, more conservative
    policy. A gap beyond the plausible band is still reported (``capped``), so the team
    decides whether its deployment really reaches it."""
    out = {}
    for fid, c in classified.items():
        for r in c.get("outside", []):
            name, v = r["dim"], r["value"]
            dim = space.dims[name]
            lo, hi = out.get(name, {}).get("range") or list(trained[name])
            if v < lo:
                lo = v - PATCH_HEADROOM * (lo - v)
            elif v > hi:
                hi = v + PATCH_HEADROOM * (v - hi)
            p_lo, p_hi = float(dim.plausible[0]), float(dim.plausible[1])
            capped = lo < p_lo or hi > p_hi
            lo, hi = max(lo, min(p_lo, trained[name][0])), min(hi, max(p_hi, trained[name][1]))
            ent = out.setdefault(name, {"was": list(trained[name]), "families": [],
                                        "hint": FRAMEWORK_HINTS.get(
                                            name, "add this dim to your randomization")})
            if name in INTEGER_DIMS:
                lo, hi = math.floor(lo + 1e-9), math.ceil(hi - 1e-9)
            ent["range"] = [round(lo, 4), round(hi, 4)]
            ent["capped_at_plausible"] = bool(ent.get("capped_at_plausible") or capped)
            if fid not in ent["families"]:
                ent["families"].append(fid)
    return out


def annotate(families, space, environ=None) -> dict:
    """Classify every family in place (``family.training_coverage``) and return the
    run-level summary: source, counts per verdict and the suggested patch."""
    trained, source = load_training_dr(space, environ)
    classified = {}
    for f in families:
        c = classify(f, space, trained)
        f.training_coverage = c
        classified[f.id] = c
    counts = {}
    for c in classified.values():
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    if trained is None:
        return {"source": None, "reason": source, "counts": counts, "patch": {}}
    return {"source": source, "counts": counts,
            "patch": suggest_patch(classified, space, trained)}


def patch_yaml(patch) -> str:
    """The suggested patch as a ``training_dr:`` YAML block (the BP_TRAIN_DR format), one
    comment per dim naming where it lives in Isaac Lab / legged_gym."""
    if not patch:
        return ""
    lines = ["training_dr:"]
    for name, ent in patch.items():
        lo, hi = ent["range"]
        was = ent["was"]
        lines.append(f"  # {ent['hint']}")
        lines.append(f"  # was [{was[0]:g}, {was[1]:g}]; covers "
                     f"{', '.join(ent['families'])}"
                     + ("; capped at the plausible band" if ent["capped_at_plausible"]
                        else ""))
        lines.append(f"  {name}: {{range: [{lo:g}, {hi:g}]}}")
    return "\n".join(lines) + "\n"
