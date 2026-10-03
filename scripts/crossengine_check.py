"""Run the cross-ENGINE agreement layer for real, across independent SIMULATORS.

crossval.py measures cross-CONFIG validity (MJX solver/timestep variants). This script
exercises the layer above it -- ``crossval_engines`` -- across genuinely different
SIMULATORS: MJX, C MuJoCo, and Genesis.

It synthesizes a spread of thetas from the target's theta_space bounds (friction,
payload, latency, slope, push, perception noise), replays each one through every engine
``run_cross_engine`` can instantiate, prints the agreement report and writes it to JSON.

    # a mujoco_playground target -> engines {mjx, mujoco_c}
    python scripts/crossengine_check.py --records 10 --horizon 60 --out out/crossengine.json

    # a bring-your-own-robot URDF target -> engines {mjx, genesis}
    python scripts/crossengine_check.py --urdf tests/fixtures/urdf/quadruped.urdf \
        --records 12 --horizon 30 --out out/crossengine_urdf.json

WHICH ENGINES RUN IS TARGET-DEPENDENT, and the report says which did not and why.
``MujocoCBackend`` reaches parity by shadowing a mujoco_playground env's own
``_get_obs`` and swapping ``mjx_env.step``; a URDF-derived ``URDFEnv`` has neither hook,
so it is skipped there. ``GenesisBackend`` is the mirror image: it can only run where
BreakPoint owns the observation/reset/command definition, i.e. a URDF target, so it is
skipped for Playground. Every run therefore has TWO engines, and the skip lines say
which third one was not available.

WHAT THE NUMBERS MEAN. This validates the PLUMBING end-to-end: that two independent
engines can be handed the same recorded failure and produce comparable, separately
scored verdicts, and that where they disagree the disagreement is surfaced. It is NOT a
scientific result. With a RANDOM-weight policy (the default here -- no trained checkpoint
needed) the robot falls in almost every world, so reproduction rates near 1.0 say
"the replay ran", not "these failures are real". The scientific claim -- how often a
failure found against a TRAINED policy in MJX survives an independent engine -- needs
that trained policy, and is a GPU run.

    --checkpoint <dir>   use a trained policy from <dir> instead of random weights.

MEMORY NOTE (CPU boxes): each eager MJX re-sim compiles the whole scan body. On a small
commit limit prepend
``XLA_FLAGS="--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"``
-- codegen effort only, not semantics. Genesis additionally pays a one-off taichi
kernel compile per distinct floor SLOPE (the plane's orientation is fixed at
``scene.build()``), which is minutes on a cold box -- keep ``--records`` modest.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from breakpoint_mjx.paths import config_path

ROOT = Path(__file__).resolve().parents[1]

# Theta spreads: one axis pushed toward a bound per record, so the batch covers the
# distinct failure axes (contact, inertia, timing, geometry, perception, disturbance)
# rather than 10 samples of the same corner. Values are inside configs/*/theta_space.yaml
# bounds; any dim a space does not declare is dropped by _record, and any value outside a
# DERIVED (URDF) space's bounds is clipped to it by _records.
SPREAD = [
    {},                                                            # nominal reference
    {"friction_floor": 0.18},                                      # slick floor
    {"friction_floor": 1.15},                                      # grippy floor
    {"torso_mass_add_kg": 5.0, "com_offset_x_m": 0.06},            # heavy, nose-forward payload
    {"com_offset_y_m": -0.07, "com_offset_z_m": 0.04},             # lateral + raised CoM
    {"latency_steps": 4.0},                                        # max actuation delay
    {"kp_scale": 0.55, "kd_scale": 1.45},                          # soft, over-damped joints
    {"joint_frictionloss_scale": 2.8, "joint_armature_scale": 1.9},  # stiff drivetrain
    {"floor_slope_rad": 0.18},                                     # ~10 deg uphill tilt
    {"floor_slope_rad": -0.18, "friction_floor": 0.30},            # downhill AND slick
    {"obs_noise_scale": 0.09, "obs_dropout_prob": 0.4, "obs_bias_scale": 0.08},  # perception
    {"push_mag": 1.9, "push_step_frac": 0.35, "latency_steps": 2.0},  # shove mid-gait
]


def _obs_dim(env):
    """Observation width for either env flavour.

    A mujoco_playground env reports ``{"state": (D,), ...}``; ``URDFEnv`` reports a plain
    int. Read by shape rather than assuming, so one script drives both.
    """
    size = env.observation_size
    if isinstance(size, dict):
        return int(size["state"][0])
    return int(size)


def _policy(env, checkpoint, target, seed=0):
    """A deterministic policy_fn: a trained checkpoint if given, else RANDOM weights.

    Parity/plumbing does not depend on the policy being good, and a random-weight MLP
    keeps this runnable with no checkpoint -- but see the module docstring on what the
    resulting numbers do and do not mean.
    """
    if checkpoint:
        from pathlib import Path

        from breakpoint_mjx.policy import make_policy_fn

        # Accept EITHER the directory holding the checkpoint or the .pkl itself.
        # make_policy_fn joins out_dir with the target's checkpoint_name, so passing
        # the file produced ".../go1_policy.pkl/go1_policy.pkl" and a NotADirectoryError.
        # Pointing at the file is the natural thing to do, so handle it rather than
        # making the caller know the convention.
        p = Path(checkpoint)
        out_dir = str(p.parent) if p.is_file() else str(p)
        if p.is_file() and p.name != target.checkpoint_name:
            raise SystemExit(
                f"checkpoint file is named {p.name!r} but target {target.name!r} "
                f"expects {target.checkpoint_name!r}. Rename it, or pass the "
                f"directory and let the target pick its own file."
            )
        return make_policy_fn(out_dir=out_dir, target=target), "trained"

    import numpy as np

    from breakpoint_mjx.adapters import from_params

    rng = np.random.default_rng(seed)
    obs_dim = _obs_dim(env)
    act_dim = int(env.action_size)
    layers = [
        (rng.normal(size=(obs_dim, 32)) * 0.15, np.zeros(32)),
        (rng.normal(size=(32, act_dim)) * 0.15, np.zeros(act_dim)),
    ]
    return from_params(layers, activation="tanh").policy_fn, "random-weight MLP"


def _records(space, n, seed=0):
    """Synthesize FailureRecords over the SPREAD, each with its own exact-repro key.

    Every record carries a distinct (batch_key, env_index), because the engines replay a
    record's OWN recorded key -- that is what makes "same theta, same seed, different
    engine" the only difference between the two runs.

    SPREAD values are written for the Go1 space. A URDF target's space is DERIVED from
    the measured robot (``urdf_target.derive_theta_space``), so a hand-written value can
    fall outside its bounds -- e.g. +5 kg on an 8.2 kg quadruped. Values are therefore
    CLIPPED to each dim's bounds rather than used raw: a theta outside the searched box
    is not a failure the search could have found, so replaying one would not be evidence
    about anything the atlas contains.
    """
    from breakpoint_mjx.schemas import FailureRecord

    names = space.names()
    out = []
    for i, overrides in enumerate(SPREAD[:n]):
        theta = {nm: space.dims[nm].nominal for nm in names}
        for k, v in overrides.items():
            if k in theta:
                lo, hi = space.dims[k].bounds
                theta[k] = float(min(max(float(v), float(lo)), float(hi)))
        label = "-".join(sorted(overrides)) or "nominal"
        out.append(FailureRecord(
            id=f"x{i:02d}", family_id=label,
            theta_vector=[theta[nm] for nm in names], theta_named=theta,
            failure_score=1.0, component_signals={}, rollout_signature=[],
            seed=seed + i, batch_key=[seed, 100 + i], env_index=i,
        ))
    return out


def _load_target(args):
    """The Target under test: a registry name, or a URDF imported on the spot.

    ``--urdf`` routes through ``make_urdf_target``, which measures the robot, derives its
    theta space and writes it next to the output. That is the only kind of target the
    Genesis backend can run, because it is the only kind whose observation / reset /
    command definition BreakPoint owns on both sides.
    """
    if args.urdf:
        from breakpoint_mjx.urdf_target import make_urdf_target

        name = args.name or Path(args.urdf).stem
        ts = Path(args.out).parent / f"bp_{name}_theta_space.yaml"
        ts.parent.mkdir(parents=True, exist_ok=True)
        target = make_urdf_target(name, args.urdf, theta_space_path=ts)
        return target, str(target.theta_space_path)

    from breakpoint_mjx.targets import get_target

    target = get_target(args.target)
    return target, config_path(target.theta_space_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="go1",
                    help="registry target name (ignored when --urdf is given)")
    ap.add_argument("--urdf", default=None,
                    help="path to a flat .urdf; imports it as a target (enables Genesis)")
    ap.add_argument("--name", default=None,
                    help="name for the --urdf target (default: the URDF file stem)")
    ap.add_argument("--records", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="failure_score at/above which an engine 'reproduces' a failure")
    ap.add_argument("--checkpoint", default=None,
                    help="dir holding a trained policy (default: random weights)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("BP_OUT", "out"),
                                                  "crossengine.json"))
    args = ap.parse_args()

    from breakpoint_mjx.config import load_theta_space
    from breakpoint_mjx.crossval_engines import resolve_thresholds, run_cross_engine

    target, ts_path = _load_target(args)
    env = target.load_env()
    consts = target.resolve_constants(env)
    space = load_theta_space(ts_path)
    base_model = env.mjx_model          # pristine; captured before any re-sim
    policy_fn, policy_kind = _policy(env, args.checkpoint, target)
    records = _records(space, args.records)

    label = f"urdf:{args.urdf}" if args.urdf else args.target
    print(f"target={target.name} ({label}) horizon={args.horizon} policy={policy_kind}")

    t0 = time.time()
    report, matrix = run_cross_engine(
        records, env, policy_fn, target, space, consts, base_model, args.horizon,
        threshold=args.threshold, thresholds=resolve_thresholds(target),
        return_matrix=True,
    )
    print("engines:", report.engines)
    for name, reason in report.skipped:   # printed so a 1-engine report never passes
        print(f"  SKIPPED {name}: {reason}")   # for cross-engine evidence
    if not report.is_cross_engine:
        print("\nWARNING: fewer than 2 engines produced a verdict -- this is NOT "
              "cross-engine evidence.")
    print(f"\nreplayed {len(records)} records x {len(report.engines)} engines "
          f"in {time.time() - t0:.1f}s")

    print("\nper-record scores (score / reproduced):")
    by_id = {r.id: r for r in records}
    for rid, row in matrix.items():
        cells = "  ".join(
            f"{c.engine}={c.error if c.error else f'{c.score:.3f}/{int(c.reproduced)}'}"
            for c in row
        )
        print(f"  {rid} [{by_id[rid].family_id}]  {cells}")

    print("\nreproduction rate:", {k: round(v, 3) for k, v in
                                   report.reproduction_rate.items()})
    print("error rate:       ", {k: round(v, 3) for k, v in report.error_rate.items()})
    print("pairwise agreement:", {k: round(v, 3) for k, v in
                                  report.pairwise_agreement.items()})
    print(f"disagreements: {len(report.disagreements)}")
    for d in report.disagreements:
        print(f"  {d['id']} [{d['family']}] verdicts={d['verdicts']} "
              f"scores={ {k: round(v, 3) for k, v in d['scores'].items()} }")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report.to_json())
    payload["policy"] = policy_kind
    payload["target"] = target.name
    payload["target_source"] = label
    payload["theta_space_path"] = str(ts_path)
    payload["horizon"] = args.horizon
    payload["skipped_engines"] = [list(s) for s in report.skipped]
    payload["per_record"] = {
        rid: {c.engine: {"score": c.score, "reproduced": c.reproduced, "error": c.error}
              for c in row}
        for rid, row in matrix.items()
    }
    payload["theta_spread"] = {r.id: r.family_id for r in records}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
