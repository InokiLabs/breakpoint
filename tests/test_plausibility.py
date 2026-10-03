"""Tests for the plausibility gate. CPU-pure: no JAX, no LLM, no GPU, no network.

The gate exists to stop simulation budget going to worlds a customer would dismiss
("your robot falls on a frictionless ice rink with a broken IMU and 6 kg bolted on").
The risk in building it is the OPPOSITE failure: a gate tuned too tight rejects exactly
the edge-of-plausible worlds BreakPoint exists to find, and quietly turns the LLM
proposer into a nominal-config generator. So the anchor test here is not "junk is
rejected" -- it is ``test_real_reported_failures_pass_the_gate``, which replays the
engine's OWN shipped findings from media/demo/demo_measurements.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from breakpoint_mjx.config import load_theta_space
from breakpoint_mjx.llm_search import _build_batch, gated_proposals
from breakpoint_mjx.paths import config_path
from breakpoint_mjx.plausibility import (
    EXTREME_FRAC,
    MAX_EXTREME,
    MIN_TRANSFER,
    dim_excess,
    gate,
    impossible_reasons,
    implausibility,
    max_outside_default,
    max_repair_default,
    repair,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 500  # configs/search.yaml


@pytest.fixture(scope="module")
def space():
    return load_theta_space(config_path("go1/theta_space.yaml"))


@pytest.fixture(scope="module")
def panda_space():
    return load_theta_space(config_path("panda/theta_space.yaml"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def nominal(space, **over):
    cfg = {n: float(d.nominal) for n, d in space.dims.items()}
    cfg.update({k: float(v) for k, v in over.items()})
    return cfg


def pinned_worst(space):
    """Every dim slammed to whichever hard bound is FURTHEST from nominal."""
    out = {}
    for n, d in space.dims.items():
        lo, hi = d.bounds
        out[n] = lo if abs(lo - d.nominal) > abs(hi - d.nominal) else hi
    return out


def realistic_adversarial(space):
    """A world HARDER than anything the engine has actually reported, that must still pass.

    The shipped findings in media/demo/demo_measurements.json turn out to sit almost
    entirely INSIDE the plausible bands (conditions 0-2 score transfer 1.000), so they
    alone would not prove the gate tolerates band violations. This config deliberately
    goes further: seven axes pushed OUTSIDE their realistic band at once -- worn bearings,
    a slippery-but-real floor, a heavy payload, an 11-degree ramp, a hard shove, a laggy
    control loop and a flaky sensor. Every value is individually a machine you could
    build and a site you could walk into; none is at a hardware extreme. This is precisely
    the edge-of-plausible region BreakPoint exists to search, so a gate that rejects it is
    broken by definition.

    The seed values follow demo condition 3 (kp 0.73, kd 1.36, frictionloss 2.55,
    slope 0.176 -- four axes outside their bands in a REAL reported failure), extended.
    """
    return nominal(
        space,
        latency_steps=3.0,            # 1.5x the plausible ceiling: a laggy control loop
        obs_dropout_prob=0.30,        # a flaky sensor link, not a dead IMU
        floor_slope_rad=-0.15,        # ~8.6 deg downhill ramp
        friction_floor=0.28,          # wet painted concrete, not ice
        torso_mass_add_kg=3.5,        # an over-spec payload
        push_mag=1.5,                 # a hard shove
        kp_scale=0.75,                # de-tuned / worn actuators
        push_step_frac=0.47, com_offset_x_m=-0.048, obs_noise_scale=0.038,
        joint_frictionloss_scale=1.25,
    )


def _walk_thetas(obj):
    """Every theta-looking dict inside the demo measurements JSON."""
    if isinstance(obj, dict):
        if "friction_floor" in obj and "latency_steps" in obj:
            return [obj]
        return [t for v in obj.values() for t in _walk_thetas(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _walk_thetas(v)]
    return []


def demo_thetas():
    p = ROOT / "media" / "demo" / "demo_measurements.json"
    if not p.exists():
        return []
    return _walk_thetas(json.loads(p.read_text()))


# ===========================================================================
# Layer 1: IMPOSSIBLE -- each rule fires with a useful message ...
# ===========================================================================

def test_sane_config_has_no_impossible_reasons(space):
    """... and none of them fire on a sane config. This is the guard on all of the below:
    a rule that fires on nominal would silently kill every run."""
    assert impossible_reasons(nominal(space), space, horizon=HORIZON) == []
    assert impossible_reasons(realistic_adversarial(space), space, horizon=HORIZON) == []
    for th in demo_thetas():
        assert impossible_reasons(th, space, horizon=HORIZON) == []


def test_impossible_non_finite(space):
    """NaN survives ``_parse_configs``: min(max(nan, lo), hi) returns nan in Python, so
    the existing clip-to-bounds defence does NOT remove it. This rule is the only thing
    between a NaN proposal and a NaN trajectory."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        rs = impossible_reasons(nominal(space, friction_floor=bad), space)
        assert any(r.startswith("non_finite:") for r in rs), (bad, rs)
        assert "friction_floor" in rs[0]

    # And confirm the premise: the clip really does pass NaN through.
    d = space.dims["friction_floor"]
    clipped = min(max(float("nan"), d.bounds[0]), d.bounds[1])
    assert math.isnan(clipped), "clip-to-bounds unexpectedly removed NaN"


def test_impossible_out_of_bounds(space):
    rs = impossible_reasons(nominal(space, friction_floor=6.2), space)
    assert any(r.startswith("out_of_bounds:") for r in rs)
    assert "friction_floor=6.2" in rs[0] and "[0.15, 1.2]" in rs[0]


def test_impossible_negative_friction(space):
    """A negative Coulomb coefficient is an energy-generating contact, not a slippery
    floor -- the resulting fall is a simulator artefact."""
    rs = impossible_reasons(nominal(space, friction_floor=-0.3), space)
    assert any(r.startswith("negative_friction:") for r in rs)
    assert any("energy-generating" in r for r in rs)


def test_impossible_nonpositive_scale(space):
    """A multiplicative scale (nominal > 0) at or below zero is a singular actuator."""
    rs = impossible_reasons(nominal(space, joint_armature_scale=0.0), space)
    assert any(r.startswith("nonpositive_scale:") for r in rs)

    # ... but a MAGNITUDE dim whose nominal is 0 must NOT trip it: a noise-free world is
    # the correct nominal world, not a degenerate one. This is the rule's whole subtlety.
    assert impossible_reasons(nominal(space, obs_noise_scale=0.0), space) == []
    assert impossible_reasons(nominal(space, obs_bias_scale=0.0), space) == []


def test_nonpositive_scale_stands_alone_from_the_bounds_rule(space):
    """In the shipped spaces every ``*_scale`` bound is already positive, so the rule only
    ever co-fires with out_of_bounds there. It has to hold on its own for a space whose
    author (or the URDF-derived generator) admits zero -- test that directly."""
    from breakpoint_mjx.config import ThetaDim, ThetaSpace
    sloppy = ThetaSpace(dims={"kp_scale": ThetaDim(
        "kp_scale", "model", bounds=(-1.0, 2.0), nominal=1.0, plausible=(0.85, 1.15))})
    rs = impossible_reasons({"kp_scale": 0.0}, sloppy)
    assert [r.split(":")[0] for r in rs] == ["nonpositive_scale"], rs
    rs = impossible_reasons({"kp_scale": -0.5}, sloppy)
    assert [r.split(":")[0] for r in rs] == ["nonpositive_scale"], rs
    assert impossible_reasons({"kp_scale": 0.6}, sloppy) == []


def test_impossible_nonpositive_total_mass(space, panda_space):
    """A payload that takes total mass to zero or below."""
    # Derived from the real model when the caller supplies body masses.
    rs = impossible_reasons(nominal(space, torso_mass_add_kg=-0.9), space,
                            body_masses={"torso_mass_add_kg": 0.5})
    assert any(r.startswith("nonpositive_mass:") for r in rs)
    assert "-0.4 kg" in rs[0], rs
    # A real Go1 trunk is ~5 kg, so the same -0.9 kg payload is fine there.
    assert impossible_reasons(nominal(space, torso_mass_add_kg=-0.9), space,
                              body_masses={"torso_mass_add_kg": 5.0}) == []
    # The *_mass_scale form needs no external mass to be provably degenerate.
    rs = impossible_reasons(nominal(panda_space, object_mass_scale=0.0), panda_space)
    assert any(r.startswith("nonpositive_mass:") for r in rs), rs


def test_impossible_unit_interval(space):
    """push_step_frac indexes into the episode; outside [0, 1] it is meaningless, not
    severe. Same for a dropout probability."""
    rs = impossible_reasons(nominal(space, push_step_frac=1.7), space)
    unit = [r for r in rs if r.startswith("unit_interval:")]
    assert unit and "fraction" in unit[0] and "push_step_frac=1.7" in unit[0]
    rs = impossible_reasons(nominal(space, obs_dropout_prob=-0.2), space)
    unit = [r for r in rs if r.startswith("unit_interval:")]
    assert unit and "probability" in unit[0]


def test_impossible_negative_magnitude(space):
    rs = impossible_reasons(nominal(space, push_mag=-1.0), space)
    assert any(r.startswith("negative_magnitude:") for r in rs)


def test_impossible_latency_beyond_horizon(space):
    """The action ring buffer cannot express a delay the episode never reaches."""
    rs = impossible_reasons(nominal(space, latency_steps=4.0), space, horizon=3)
    assert any(r.startswith("latency_exceeds_horizon:") for r in rs)
    assert "horizon 3" in [r for r in rs if r.startswith("latency_exceeds")][0]
    # Same value, a real horizon -> perfectly expressible.
    assert impossible_reasons(nominal(space, latency_steps=4.0), space,
                              horizon=HORIZON) == []
    # And with no horizon supplied the rule simply does not apply (never wrong, only
    # more permissive).
    assert impossible_reasons(nominal(space, latency_steps=4.0), space) == []


def test_impossible_reasons_name_the_rule_the_dim_and_the_value(space):
    """Every message must be actionable on its own in a run log."""
    bad = nominal(space, friction_floor=-0.5, push_step_frac=3.0, push_mag=-2.0)
    rs = impossible_reasons(bad, space, horizon=HORIZON)
    assert len(rs) >= 3
    for r in rs:
        assert ":" in r, r
        assert any(d in r for d in space.names()), r
        assert any(ch.isdigit() for ch in r.split(":", 1)[1]), r


# ===========================================================================
# THE ANTI-OVER-TIGHTENING TEST. If this ever fails, the gate has eaten the product.
# ===========================================================================

def test_real_reported_failures_pass_the_gate(space):
    """The engine's OWN shipped findings must survive the gate untouched.

    media/demo/demo_measurements.json holds representative thetas from real GPU runs --
    the latency + dropout + slope worlds the product reports to customers. If the gate
    would have rejected or even repaired any of them, the gate is wrong, not the finding.
    """
    thetas = demo_thetas()
    assert thetas, "demo measurements missing; this test is the gate's safety anchor"
    res = gate(thetas, space, horizon=HORIZON)
    assert res.counts["rejected"] == 0, [r.reasons for r in res.rejected]
    assert res.counts["repaired"] == 0, [
        [str(c) for c in r.changes] for r in res.repaired]
    assert res.counts["kept"] == len(thetas)

    # And with visible margin, not by a hair: the weakest real finding must sit clearly
    # above every threshold, so ordinary run-to-run variation cannot start rejecting them.
    worst = min(implausibility(t, space)["transfer"] for t in thetas)
    assert worst >= MIN_TRANSFER + 0.20, (
        f"weakest real finding transfer={worst:.3f} is uncomfortably close to the "
        f"MIN_TRANSFER={MIN_TRANSFER} floor")
    assert max(implausibility(t, space)["max_excess"] for t in thetas) < EXTREME_FRAC


def test_hand_built_adversarial_world_passes(space):
    """A world harder than anything the engine has reported must still be KEPT untouched.

    This is the test that stops the gate quietly becoming a nominal-config generator:
    seven axes outside their realistic band simultaneously, and it goes through unchanged.
    """
    adv = realistic_adversarial(space)
    m = implausibility(adv, space)
    res = gate([adv], space, horizon=HORIZON)
    assert res.counts["kept"] == 1, (m, [r.reasons for r in res.rejected],
                                     [[str(c) for c in r.changes] for r in res.repaired])
    # It really is adversarial, not a nominal config in disguise: strictly harsher on
    # every count than the worst REAL reported failure, which uses 4 of 16 axes.
    assert m["n_outside"] >= 7, m
    assert m["transfer"] < min(implausibility(t, space)["transfer"] for t in demo_thetas())
    assert adv["latency_steps"] > space.dims["latency_steps"].plausible[1]
    # ... yet still comfortably clear of every threshold, not scraping past one.
    assert m["transfer"] >= MIN_TRANSFER + 0.15, m
    assert m["max_excess"] < EXTREME_FRAC and m["n_extreme"] == 0


def test_gate_does_not_collapse_to_a_nominal_generator(space):
    """A spread of edge-of-plausible worlds (each axis walked to its band edge in turn)
    must pass essentially in full. If the gate started rejecting these, the LLM proposer
    would degenerate into re-proposing nominal."""
    cfgs = []
    for name, d in space.dims.items():
        lo_p, hi_p = d.plausible
        for edge in (lo_p, hi_p):
            cfgs.append(nominal(space, **{name: edge}))
    res = gate(cfgs, space, horizon=HORIZON)
    assert res.counts["kept"] == len(cfgs), [r.reasons for r in res.rejected]


# ===========================================================================
# Layer 2: IMPLAUSIBLE / the unsurvivable corner
# ===========================================================================

def test_all_axes_pinned_is_rejected_as_an_unsurvivable_corner(space):
    """Every axis at its worst hard bound at once: the classic uninteresting failure.

    Note this world is IN BOUNDS -- clip-to-bounds, the entire pre-existing defence,
    waves it straight through. Rejection here is the joint layer earning its keep.
    """
    worst = pinned_worst(space)
    assert impossible_reasons(worst, space, horizon=HORIZON) == [], \
        "the pinned corner is in bounds; it must be caught by layer 2, not layer 1"
    res = gate([worst], space, horizon=HORIZON)
    assert res.counts["rejected"] == 1
    reasons = res.rejected[0].reasons
    assert any(r.startswith("unsurvivable_corner:") for r in reasons), reasons
    assert any(r.startswith("low_transfer:") for r in reasons), reasons
    assert any(r.startswith("unrepairable:") for r in reasons), reasons


def test_the_ice_rink_report_is_caught_by_the_joint_rule_alone(space):
    """"Frictionless ice rink + broken IMU + 6 kg bolted on": THE dismissible report.

    Its transfer is 0.81 and only 3 of 16 dims are outside their band -- both per-dim
    style checks pass it happily. Only the simultaneity of three axes at their absolute
    hardware extremes catches it, which is exactly why n_extreme exists.
    """
    ice = nominal(space, friction_floor=0.15, obs_dropout_prob=0.50, torso_mass_add_kg=6.0)
    m = implausibility(ice, space)
    assert m["transfer"] > MIN_TRANSFER, "per-dim transfer alone would let this through"
    assert m["n_outside"] <= max_outside_default(space), "breadth alone would let it through"
    assert m["n_extreme"] == 3 > MAX_EXTREME

    res = gate([ice], space, horizon=HORIZON, repair_enabled=False)
    assert res.counts["rejected"] == 1
    assert any(r.startswith("unsurvivable_corner:")
               for r in res.rejected[0].reasons), res.rejected[0].reasons


def test_implausibility_reports_worst_dims_in_order(space):
    cfg = nominal(space, floor_slope_rad=0.20, kp_scale=1.20, obs_noise_scale=0.05)
    m = implausibility(cfg, space)
    assert m["n_outside"] == 3
    assert m["worst"][0] == "floor_slope_rad"  # excess 1.0, the hard bound
    excesses = [r["excess"] for r in m["outside"]]
    assert excesses == sorted(excesses, reverse=True)
    assert m["max_excess"] == pytest.approx(1.0)
    assert dim_excess(0.0, space.dims["floor_slope_rad"]) == 0.0


# ===========================================================================
# Layer 3: REPAIR
# ===========================================================================

def test_repair_passes_while_keeping_the_dominant_attacked_axis_attacked(space):
    """Repair must not neuter the proposal.

    Four axes at their hardware extremes: the two the proposal is most about stay pinned,
    and only the surplus is walked back. What comes out is still an attack on the same
    axes, just one a customer would accept.
    """
    cfg = nominal(space, latency_steps=4.0, obs_dropout_prob=0.5, floor_slope_rad=-0.20,
                  friction_floor=0.15)
    before = implausibility(cfg, space)
    assert before["n_extreme"] == 4 > MAX_EXTREME
    dominant = before["worst"][0]

    rep = repair(cfg, space)
    assert rep.ok, rep.give_up
    assert rep.changes, "repair reported success without changing anything"
    # The repaired world is credible ...
    assert gate([rep.cfg], space, horizon=HORIZON).counts["kept"] == 1
    # ... the dominant attacked axis was NOT touched ...
    assert dominant not in {c.dim for c in rep.changes}
    assert rep.cfg[dominant] == cfg[dominant]
    assert implausibility(rep.cfg, space)["outside"][0]["name"] == dominant
    # ... and every axis that WAS walked back kept its direction of attack: a proposal
    # that wanted a slippery floor still gets a slippery floor.
    for c in rep.changes:
        d = space.dims[c.dim]
        assert (c.was - d.nominal) * (c.now - d.nominal) >= 0, c
        assert abs(c.now - d.nominal) <= abs(c.was - d.nominal) + 1e-9, c
        assert d.bounds[0] - 1e-9 <= c.now <= d.bounds[1] + 1e-9, c


def test_repair_is_a_no_op_on_an_already_plausible_config(space):
    rep = repair(realistic_adversarial(space), space)
    assert rep.ok and rep.changes == [] and rep.reasons == []


def test_repair_gives_up_rather_than_inventing_a_different_world(space):
    """Rewriting most of a config is not honouring the LLM's intent -- it is fabricating
    a proposal. Past the repair budget, repair must decline."""
    rep = repair(pinned_worst(space), space)
    assert not rep.ok
    assert rep.give_up and rep.give_up.startswith("unrepairable:")
    assert len(rep.changes) <= max_repair_default(space)


def test_repair_changes_are_reported_and_replayable(space):
    cfg = nominal(space, latency_steps=4.0, obs_dropout_prob=0.5, floor_slope_rad=-0.20,
                  obs_bias_scale=0.10, obs_noise_scale=0.10)
    res = gate([cfg], space, horizon=HORIZON)
    assert res.counts["repaired"] == 1
    r = res.repaired[0]
    assert r.original == cfg and r.cfg != cfg
    replayed = dict(r.original)
    for c in r.changes:
        assert replayed[c.dim] == pytest.approx(c.was)
        assert c.rule in {"unsurvivable_corner", "too_many_implausible_dims",
                          "low_transfer"}
        replayed[c.dim] = c.now
    assert replayed == r.cfg, "the change list must fully explain the repaired config"
    assert "->" in str(r.changes[0])


# ===========================================================================
# The gate as a whole
# ===========================================================================

def test_gate_counts_add_up(space):
    cfgs = [
        nominal(space),                                                # kept
        realistic_adversarial(space),                                  # kept
        nominal(space, friction_floor=float("nan")),                   # impossible
        nominal(space, push_step_frac=4.0),                            # impossible
        nominal(space, friction_floor=0.15, obs_dropout_prob=0.5,
                torso_mass_add_kg=6.0),                                # repaired
        pinned_worst(space),                                           # rejected
    ]
    res = gate(cfgs, space, horizon=HORIZON)
    c = res.counts
    assert c["kept"] + c["repaired"] + c["rejected"] == len(cfgs) == c["input"]
    assert c["kept"] == 2 and c["repaired"] == 1 and c["rejected"] == 3
    assert c["impossible"] == 2
    assert len(res.accepted) == c["kept"] + c["repaired"]
    assert all(set(a) == set(space.names()) for a in res.accepted)


def test_gate_never_repairs_an_impossible_world(space):
    """An incoherent world has no nearby coherent one worth guessing at."""
    res = gate([nominal(space, friction_floor=float("nan"))], space, repair_enabled=True)
    assert res.counts["rejected"] == 1 and res.rejected[0].kind == "impossible"
    assert res.counts["repaired"] == 0


def test_gate_repair_disabled_rejects_instead(space):
    ice = nominal(space, friction_floor=0.15, obs_dropout_prob=0.50, torso_mass_add_kg=6.0)
    assert gate([ice], space, horizon=HORIZON).counts["repaired"] == 1
    off = gate([ice], space, horizon=HORIZON, repair_enabled=False)
    assert off.counts["rejected"] == 1 and off.rejected[0].kind == "implausible"


def test_gate_fills_missing_dims_from_nominal(space):
    """Proposals may arrive partial (the same convention _parse_configs uses)."""
    res = gate([{"latency_steps": 1.9}], space, horizon=HORIZON)
    assert res.counts["kept"] == 1
    assert res.kept[0]["kp_scale"] == pytest.approx(space.dims["kp_scale"].nominal)


def test_gate_summary_is_one_informative_line(space):
    res = gate([nominal(space), pinned_worst(space)], space, horizon=HORIZON)
    s = res.summary()
    assert s.count("\n") == 0
    assert "1 kept" in s and "1 rejected" in s and "unsurvivable_corner" in s
    assert gate([], space).summary() == "gate: 0 kept, 0 repaired, 0 rejected"


def test_gate_works_on_a_manipulation_space_it_has_never_seen(space, panda_space):
    """Nothing in the gate is go1-specific: the rules read the space, not a robot list."""
    assert gate([nominal(panda_space)], panda_space, horizon=HORIZON).counts["kept"] == 1
    assert gate([pinned_worst(panda_space)], panda_space,
                horizon=HORIZON).counts["rejected"] == 1


# ===========================================================================
# Downstream: an all-rejected round must NOT shrink the evaluation budget
# ===========================================================================

def test_all_rejected_round_still_yields_a_full_batch(space, capsys):
    """If a round's proposals are all junk the batch must still be filled from random
    padding. A short batch would silently change the eval budget and make two runs
    incomparable -- a far worse failure than simulating a few random worlds."""
    junk = [pinned_worst(space) for _ in range(6)]
    accepted = gated_proposals(space, junk, horizon=HORIZON)
    assert accepted == []

    N = 64
    batch = _build_batch(space, accepted, N, np.random.default_rng(0))
    assert batch.shape == (N, len(space.dims))
    lo, hi = np.array(space.lower()), np.array(space.upper())
    assert np.all(batch >= lo - 1e-9) and np.all(batch <= hi + 1e-9)

    # ... and the operator was told, on one line, that it happened.
    out = capsys.readouterr().out
    assert "gate: 0 kept, 0 repaired, 6 rejected" in out
    assert "unsurvivable_corner" in out


def test_gated_proposals_prints_and_passes_good_worlds_through(space, capsys):
    good = [realistic_adversarial(space), nominal(space)]
    accepted = gated_proposals(space, good, horizon=HORIZON)
    assert accepted == good
    assert "gate: 2 kept, 0 repaired, 0 rejected" in capsys.readouterr().out


def test_gated_proposals_label_is_used_for_per_role_lines(space, capsys):
    gated_proposals(space, [nominal(space)], horizon=HORIZON, label="gate[timing]")
    assert "gate[timing]: 1 kept" in capsys.readouterr().out


# ===========================================================================
# QUANTIFICATION. A self-contained corpus of LLM-style proposals -- realistic ones,
# borderline ones and deliberately absurd ones -- run through the gate, with the
# per-proposal verdict printed. Run `pytest tests/test_plausibility.py -k report -s`
# to read the table; the assertions below pin the numbers so a threshold change that
# alters the gate's behaviour cannot land silently.
# ===========================================================================

def proposal_corpus(space):
    """(label, expected_verdict, cfg) for a plausible round of LLM output."""
    c = []
    for i, t in enumerate(demo_thetas()):
        c.append((f"real-report-{i}", "kept", {**nominal(space), **t}))
    c += [
        ("nominal", "kept", nominal(space)),
        ("hand-built-adversarial", "kept", realistic_adversarial(space)),
        ("slippery+laggy", "kept",
         nominal(space, friction_floor=0.35, latency_steps=2.5)),
        ("payload+com-shift", "kept",
         nominal(space, torso_mass_add_kg=3.0, com_offset_x_m=0.06, com_offset_y_m=0.05)),
        ("degraded-perception", "kept",
         nominal(space, obs_dropout_prob=0.28, obs_noise_scale=0.06, obs_bias_scale=0.06)),
        ("ramp+shove", "kept",
         nominal(space, floor_slope_rad=0.16, push_mag=1.6, push_step_frac=0.30)),

        # Borderline: real attack intent, one axis too many at a hardware extreme.
        ("ice-rink+dead-imu+6kg", "repaired",
         nominal(space, friction_floor=0.15, obs_dropout_prob=0.50,
                 torso_mass_add_kg=6.0)),
        ("everything-timing-maxed", "repaired",
         nominal(space, latency_steps=4.0, push_mag=2.0, obs_dropout_prob=0.50,
                 floor_slope_rad=-0.20)),
        ("all-actuators-maxed", "repaired",
         nominal(space, kp_scale=0.50, kd_scale=1.50, joint_armature_scale=2.0,
                 joint_frictionloss_scale=3.0)),
        ("broad-but-shallow", "repaired",
         nominal(space, friction_floor=0.36, torso_mass_add_kg=2.6, com_offset_x_m=0.062,
                 com_offset_y_m=0.062, kp_scale=0.80, kd_scale=1.22, com_offset_z_m=0.036,
                 joint_frictionloss_scale=1.55, joint_armature_scale=1.25,
                 obs_noise_scale=0.055)),

        # Absurd.
        ("every-axis-pinned", "rejected", pinned_worst(space)),
        ("negative-friction", "rejected", nominal(space, friction_floor=-0.5)),
        ("nan-payload", "rejected", nominal(space, torso_mass_add_kg=float("nan"))),
        ("phase-frac-7", "rejected", nominal(space, push_step_frac=7.0)),
        ("latency-600-steps", "rejected", nominal(space, latency_steps=600.0)),
        ("zero-armature", "rejected", nominal(space, joint_armature_scale=0.0)),
    ]
    return c


def test_gate_report_on_a_realistic_llm_proposal_set(space):
    corpus = proposal_corpus(space)
    res = gate([cfg for _, _, cfg in corpus], space, horizon=HORIZON)

    def verdict_of(cfg):
        """Per-proposal verdict. The gate is order-independent (it judges each config on
        its own), so gating one config alone gives the same answer as its slot in the
        batch -- which the aggregate assertions at the bottom re-confirm."""
        one = gate([cfg], space, horizon=HORIZON)
        return "kept" if one.kept else "repaired" if one.repaired else "rejected"

    order = [(label, expected, verdict_of(cfg)) for label, expected, cfg in corpus]

    print("\n" + "=" * 78)
    print(f"PLAUSIBILITY GATE REPORT  ({len(corpus)} proposals, go1, horizon {HORIZON})")
    print(f"defaults: min_transfer={MIN_TRANSFER}  max_extreme={MAX_EXTREME}  "
          f"extreme_frac={EXTREME_FRAC}  max_outside={max_outside_default(space)}  "
          f"max_repair={max_repair_default(space)}")
    print("=" * 78)
    for label, expected, got in order:
        cfg = dict(next(c for l, _, c in corpus if l == label))
        try:
            m = implausibility(cfg, space)
            stats = (f"transfer={m['transfer']:.3f} outside={m['n_outside']:2d} "
                     f"extreme={m['n_extreme']:2d}")
        except Exception:
            stats = "transfer=n/a (non-finite)"
        print(f"  {got.upper():<9} {label:<24} {stats}")
    print("-" * 78)
    for r in res.repaired:
        print(f"  REPAIRED  changes: {', '.join(str(c) for c in r.changes)}")
    for r in res.rejected:
        print(f"  REJECTED  {r.kind:<11} {r.reasons[0][:96]}")
    print("-" * 78)
    print(f"  {res.summary()}")
    print(f"  counts: {res.counts}")
    print("=" * 78)

    mismatches = [(l, e, g) for l, e, g in order if e != g]
    assert not mismatches, mismatches
    c = res.counts
    assert c["kept"] + c["repaired"] + c["rejected"] == len(corpus)
    assert (c["kept"], c["repaired"], c["rejected"]) == (10, 4, 6)
    assert c["impossible"] == 5, "five proposals are physically incoherent, not merely absurd"
    # Repair is the workhorse: of the 10 proposals that failed layer 2, 4 were salvaged
    # with a handful of dim edits rather than costing the whole LLM call.
    assert c["changed_dims"] <= 4 * max_repair_default(space)


def test_gate_opts_are_honoured(space):
    """An operator can loosen or tighten the gate without editing the module."""
    ice = nominal(space, friction_floor=0.15, obs_dropout_prob=0.50, torso_mass_add_kg=6.0)
    assert gated_proposals(space, [ice], horizon=HORIZON,
                           gate_opts={"max_extreme": 3}) == [ice]
    assert gated_proposals(space, [realistic_adversarial(space)], horizon=HORIZON,
                           gate_opts={"min_transfer": 0.999, "repair_enabled": False}) == []
