"""Nominal-behavior preflight for a policy: does it walk BEFORE we try to break it?

A bring-your-own policy with the wrong observation order, the wrong normalization, or the
wrong action scale does not raise anything: the dimension checks pass, the rollout runs,
and the search then "discovers" that the robot falls everywhere. That report is garbage,
and it looks exactly like a real one. This check runs the policy in N nominal worlds
(every theta at its nominal value, only the reset key varies), and refuses to go on if it
cannot stand and track at nominal:

    breakpoint check-policy --robot go1 --policy params:./weights.npz
    python -m breakpoint_mjx.check_policy --robot go1 --policy ./go1_policy.pkl
    breakpoint check-policy --robot go1 --policy ./policy.onnx --obs-spec contract.yaml
    breakpoint check-policy --robot go1 --policy ./policy.onnx --infer-contract

Fails loudly (exit 3) when the nominal fall rate exceeds 0.25 or the mean tracking error
exceeds ``thresholds.tracking_err_max``, and prints a diagnosis: per-obs-dim statistics of
the obs the policy actually received compared with the policy's own normalization
statistics (when it has them), the action saturation fraction, a time-to-fall histogram,
and the likely causes.

The env / target / policy are built the same way pipeline.py builds them (its BYO adapter
helper is imported, not copied), so a policy that passes here is the policy the search
will run. The diagnosis itself (``diagnose`` / ``format_diagnosis``) is pure numpy and is
unit-tested on synthetic signals.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXIT_FAIL = 3
FALL_RATE_MAX = 0.25
SAT_LEVEL = 0.98          # |action| at or above this counts as saturated (tanh heads)
SAT_FRAC_BAD = 0.30       # more than this fraction of pre-fall actions saturated -> flag
ZERO_ACTION = 0.02        # mean |action| below this -> the policy is barely acting
Z_BAD = 3.0               # obs dim mean more than this many policy-stds from its norm mean
STD_RATIO_BAD = 5.0       # obs dim spread more than this many times the norm std
ADAPTER_KINDS = ("onnx", "params", "npz", "weights")


# --------------------------------------------------------------------------
# pure diagnosis
# --------------------------------------------------------------------------

def _valid_mask(ttf, n, h):
    """(N, H) bool: steps strictly before each env's first fall (all steps if it never fell).

    Statistics after a fall describe a robot lying on the floor, which says nothing about
    whether the policy was fed the right inputs, so they are excluded.
    """
    steps = np.arange(h)[None, :]
    lim = np.clip(np.asarray(ttf, dtype=float).reshape(n, 1), 1, h)
    return steps < lim


def obs_dim_stats(obs, mask, norm_mean=None, norm_std=None):
    """Per-dim mean/std of the observed obs (masked) and, if given, the z-offset and the
    std ratio against the policy's own normalization statistics. Returns a list of dicts."""
    obs = np.asarray(obs, dtype=float)
    d = obs.shape[-1]
    flat = obs[mask] if mask is not None else obs.reshape(-1, d)
    if flat.size == 0:
        flat = obs.reshape(-1, d)
    flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    rows = []
    for i in range(d):
        row = {"dim": i, "mean": float(mu[i]), "std": float(sd[i])}
        if norm_mean is not None and norm_std is not None:
            ns = max(float(norm_std[i]), 1e-6)
            row["norm_mean"] = float(norm_mean[i])
            row["norm_std"] = float(norm_std[i])
            row["z"] = abs(float(mu[i]) - float(norm_mean[i])) / ns
            row["std_ratio"] = float(sd[i]) / ns
            row["flag"] = bool(row["z"] > Z_BAD or row["std_ratio"] > STD_RATIO_BAD)
        rows.append(row)
    return rows


def ttf_histogram(ttf_fallen, horizon, bins=5):
    """Counts of time-to-fall over ``bins`` equal slices of [0, horizon]."""
    edges = np.linspace(0, horizon, bins + 1)
    counts, _ = np.histogram(np.asarray(ttf_fallen, dtype=float), bins=edges)
    return [(int(edges[i]), int(edges[i + 1]), int(counts[i])) for i in range(bins)]


def diagnose(signals, obs, actions, thresholds, horizon, obs_norm=None,
             fall_rate_max=FALL_RATE_MAX):
    """Decide pass/fail and explain it. Pure numpy.

    signals : list of per-rollout component_signals dicts (needs ``fall``,
              ``time_to_failure``; ``tracking_error`` when the objective has one).
    obs     : (N, H, D) the obs vector the policy actually received, pre-normalization.
    actions : (N, H, A) the raw actions it emitted.
    obs_norm: optional (mean, std) the policy normalizes its obs with.
    """
    n = len(signals)
    obs = np.asarray(obs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    h = obs.shape[1]
    falls = np.array([float(s.get("fall", 0.0)) for s in signals])
    ttf = np.array([float(s.get("time_to_failure", h)) for s in signals])
    trk = [s["tracking_error"] for s in signals if "tracking_error" in s]
    fall_rate = float(falls.mean()) if n else 0.0
    track = float(np.mean(trk)) if trk else None
    track_max = float(thresholds.get("tracking_err_max", math.inf))

    mask = _valid_mask(np.where(falls > 0, ttf, h), n, h)
    act = np.nan_to_num(actions[mask] if mask.any() else actions.reshape(-1, actions.shape[-1]))
    sat_frac = float((np.abs(act) >= SAT_LEVEL).mean()) if act.size else 0.0
    act_abs = float(np.abs(act).mean()) if act.size else 0.0
    nm, ns = (obs_norm if obs_norm is not None else (None, None))
    rows = obs_dim_stats(obs, mask, nm, ns)
    flagged = [r for r in rows if r.get("flag")]
    fallen_ttf = ttf[falls > 0]
    hist = ttf_histogram(fallen_ttf, h)

    reasons = []
    if fall_rate > fall_rate_max:
        reasons.append(f"nominal fall rate {fall_rate:.2f} > {fall_rate_max:.2f}")
    if track is not None and track > track_max:
        reasons.append(f"mean tracking error {track:.3f} > tracking_err_max {track_max:.3f}")
    passed = not reasons

    causes = []
    if not passed:
        if obs_norm is not None and flagged:
            dims = ", ".join(str(r["dim"]) for r in sorted(flagged, key=lambda r: -r["z"])[:8])
            causes.append(
                f"OBSERVATION MISMATCH: {len(flagged)}/{len(rows)} obs dims sit far outside "
                f"the policy's own normalization statistics (worst dims: {dims}). Either the "
                f"obs ORDER differs from the one the policy was trained on, or the "
                f"normalization (--obs-norm mean/std) belongs to a different export.")
        if obs_norm is None:
            causes.append(
                "NO OBS NORMALIZATION is configured. If the policy was trained on "
                "normalized observations, pass --obs-norm <npz with mean, std>.")
        if sat_frac > SAT_FRAC_BAD:
            causes.append(
                f"ACTION SCALE: {sat_frac:.0%} of pre-fall actions are saturated "
                f"(|a| >= {SAT_LEVEL}). The action scale is likely too large, or the head "
                f"needs --squash / --mean-dim.")
        if act_abs < ZERO_ACTION:
            causes.append(
                f"ACTION SCALE: mean |action| is {act_abs:.4f}, the policy is barely "
                f"acting. The action scale may be too small, or the wrong output is read.")
        if len(fallen_ttf) and float(np.median(fallen_ttf)) < 0.2 * h:
            causes.append(
                f"EARLY FALLS: median time-to-fall {np.median(fallen_ttf):.0f}/{h} steps. "
                f"Falling almost immediately usually means wrong inputs or a wrong action "
                f"mapping, not a weak policy.")
        if track is not None and track > track_max and fall_rate <= fall_rate_max:
            causes.append(
                "STANDS BUT DOES NOT TRACK: the velocity-command channels may sit at a "
                "different place in the obs vector, or the policy was trained for a "
                "different command range.")
        causes.append(
            "OBS ORDER: BreakPoint feeds the env's obs['state'] in Playground's layout. If "
            "your policy was trained with a different ordering, remap it with an ObsSpec "
            "index list (breakpoint_mjx.adapters.ObsSpec).")

    return {
        "passed": passed, "reasons": reasons, "causes": causes,
        "n": n, "horizon": h, "fall_rate": fall_rate, "tracking_error": track,
        "tracking_err_max": track_max, "fall_rate_max": fall_rate_max,
        "sat_frac": sat_frac, "action_abs_mean": act_abs,
        "obs_dims": rows, "flagged_dims": [r["dim"] for r in flagged],
        "has_obs_norm": obs_norm is not None, "ttf_hist": hist,
    }


def format_diagnosis(d, max_rows=12):
    """Human-readable report of a ``diagnose`` result."""
    L = []
    verdict = "PASS" if d["passed"] else "FAIL"
    trk = "n/a" if d["tracking_error"] is None else f"{d['tracking_error']:.3f}"
    L.append(f"check-policy: {verdict}  ({d['n']} nominal rollouts x {d['horizon']} steps)")
    L.append(f"  fall rate        {d['fall_rate']:.2f}   (max {d['fall_rate_max']:.2f})")
    L.append(f"  tracking error   {trk}   (max {d['tracking_err_max']:.3f})")
    L.append(f"  action |a| mean  {d['action_abs_mean']:.3f}   saturated "
             f"{d['sat_frac']:.1%} of pre-fall steps")
    L.append("  time-to-fall histogram (steps: count of fallen rollouts)")
    for lo, hi, c in d["ttf_hist"]:
        L.append(f"    {lo:>4}-{hi:<4} {'#' * c}{c if c else ''}")
    rows = d["obs_dims"]
    if d["has_obs_norm"]:
        L.append(f"  obs vs the policy's normalization ({len(d['flagged_dims'])}/{len(rows)} "
                 f"dims flagged; showing the worst {min(max_rows, len(rows))})")
        L.append("    dim     obs_mean   obs_std   norm_mean  norm_std     z   std_ratio")
        for r in sorted(rows, key=lambda r: -r["z"])[:max_rows]:
            L.append(f"    {r['dim']:>3}{' *' if r['flag'] else '  '} {r['mean']:>10.3f}"
                     f" {r['std']:>9.3f} {r['norm_mean']:>10.3f} {r['norm_std']:>9.3f}"
                     f" {r['z']:>6.1f} {r['std_ratio']:>9.2f}")
    else:
        L.append("  obs stats (policy has no normalization statistics to compare against;"
                 f" showing the {min(max_rows, len(rows))} largest-magnitude dims)")
        L.append("    dim     obs_mean   obs_std")
        for r in sorted(rows, key=lambda r: -abs(r["mean"]))[:max_rows]:
            L.append(f"    {r['dim']:>3}   {r['mean']:>10.3f} {r['std']:>9.3f}")
    if not d["passed"]:
        L.append("  why it failed: " + "; ".join(d["reasons"]))
        L.append("  likely causes:")
        for c in d["causes"]:
            L.append(f"    - {c}")
        L.append("  A search on a policy that cannot walk at nominal reports that it falls "
                 "everywhere, which is not a finding. Fix the export, then re-run.")
    return "\n".join(L)


# --------------------------------------------------------------------------
# simulator side (jax / mujoco imported lazily)
# --------------------------------------------------------------------------

def is_adapter_spec(spec):
    kind, sep, _ = str(spec).partition(":")
    return bool(sep) and kind.lower() in ADAPTER_KINDS


def _brax_obs_norm(params):
    """(mean, std) of obs['state'] from a brax (normalizer, policy, value) params tuple."""
    try:
        norm = params[0]
        mean, std = norm.mean, norm.std
        if isinstance(mean, dict):
            mean, std = mean["state"], std["state"]
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    except Exception:  # noqa: BLE001 - diagnosis is best-effort, never block the check
        return None


def _obs_map(sp):
    """The contract part of an ObsSpec (index select + per-dim affine, NOT the policy's own
    normalization) as a numpy function, so the diagnosis compares what the policy
    actually received against its normalization statistics."""
    idx = None if sp.indices is None else np.asarray(sp.indices)
    off = None if sp.offset is None else np.asarray(sp.offset, dtype=float)
    sc = None if sp.scale is None else np.asarray(sp.scale, dtype=float)
    if idx is None and off is None and sc is None:
        return None

    def f(obs):
        x = obs[..., idx] if idx is not None else obs
        if off is not None:
            x = x - off
        if sc is not None:
            x = x * sc
        return x

    return f


def load_policy(target, env, spec, opts=None):
    """Return (policy_fn, obs_norm, obs_map, label) built exactly as pipeline.py would.

    ``onnx:`` / ``params:`` specs, and ANY spec with a contract (``--obs-spec``), go
    through pipeline._maybe_adapter_policy (the BYO path; a brax .pkl is weight-imported
    there so its contract can be applied). Anything else is a brax params pickle
    (optionally prefixed ``brax:``) rebuilt with policy.make_policy_fn, the default path.
    ``obs_map`` is a numpy function applying the contract's index/affine map, or None.
    """
    opts = opts or {}
    obs_spec = opts.get("obs_spec")
    if is_adapter_spec(spec) or obs_spec:
        from breakpoint_mjx.pipeline import _maybe_adapter_policy

        ns = SimpleNamespace(policy=spec, obs_norm=opts.get("obs_norm"),
                             activation=opts.get("activation"),
                             action_scale=opts.get("action_scale"), action_bias=None,
                             squash=bool(opts.get("squash")), weight_layout=None,
                             mean_dim=opts.get("mean_dim"))
        prev = os.environ.get("BP_OBS_SPEC")
        if obs_spec:
            # the pipeline reads the contract from BP_OBS_SPEC; set it the same way here
            os.environ["BP_OBS_SPEC"] = str(Path(obs_spec).resolve())
        try:
            fn = _maybe_adapter_policy(env, ns)
        finally:
            if obs_spec:
                if prev is None:
                    os.environ.pop("BP_OBS_SPEC", None)
                else:
                    os.environ["BP_OBS_SPEC"] = prev
        mlp = getattr(fn, "__self__", None)
        norm, obs_map = None, None
        if mlp is not None and getattr(mlp, "obs_spec", None) is not None:
            sp = mlp.obs_spec
            if sp.mean is not None and sp.std is not None:
                norm = (np.asarray(sp.mean, dtype=float), np.asarray(sp.std, dtype=float))
            obs_map = _obs_map(sp)
        label = f"BYO adapter {spec}" + (f" via contract {obs_spec}" if obs_spec else "")
        return fn, norm, obs_map, label

    from breakpoint_mjx.policy import make_policy_fn

    path = str(spec)[5:] if str(spec).lower().startswith("brax:") else str(spec)
    if not Path(path).exists():
        raise FileNotFoundError(f"no policy file at {path}")
    params = pickle.loads(Path(path).read_bytes())
    fn = make_policy_fn(params=params, target=target)
    return fn, _brax_obs_norm(params), None, f"brax params {path}"


def run_check(robot, spec, n=16, horizon=150, seed=0, opts=None, verbose=True, ctx=None):
    """Build target/env/policy, run ``n`` nominal rollouts, return the diagnosis dict.

    ``ctx`` (a dict, optional) receives the ``target`` and ``env`` so a follow-up
    (``--infer-contract``) reuses them instead of loading the env twice."""
    from breakpoint_mjx.accel import tune_for_gpu

    tune_for_gpu()
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.config import (load_search_config, load_theta_space,
                                       merge_target_thresholds)
    from breakpoint_mjx.evaluator import make_demo_collector
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.targets import get_target

    t0 = time.monotonic()
    target = get_target(robot)
    cfg = merge_target_thresholds(load_search_config(config_path("search.yaml")), target)
    space = load_theta_space(config_path(target.theta_space_path))
    env = (ctx or {}).get("env") or _load_env(target, (opts or {}).get("obs_spec"))
    if ctx is not None:
        ctx.update(target=target, env=env)
    policy_fn, norm, obs_map, label = load_policy(target, env, spec, opts)
    if verbose:
        print(f"check-policy: {target.name} with {label}; {n} nominal rollouts x "
              f"{horizon} steps (first call compiles, ~1-3 min on CPU) ...", flush=True)
    consts = target.resolve_constants(env)
    collect = make_demo_collector(env, policy_fn, space, horizon, target, consts)
    thetas = jnp.tile(jnp.asarray(space.nominal_vector())[None, :], (n, 1))
    base_model = env._mjx_model
    try:
        traj = collect(thetas, jax.random.PRNGKey(seed))
    finally:
        # rollout_one binds the per-world model with ``env._mjx_model = model`` inside the
        # trace, which leaves a batched tracer on the env. Restore the concrete model so a
        # follow-up on the same env (--infer-contract via ctx) does not hit a leaked tracer.
        env._mjx_model = base_model
    host = {k: np.asarray(v) for k, v in traj.items()}
    obs = host.pop("obs_state")
    actions = host.pop("action")
    if obs_map is not None:
        obs = obs_map(obs)
    signals = [target.objective.component_signals({k: v[i] for k, v in host.items()},
                                                  cfg.thresholds) for i in range(n)]
    d = diagnose(signals, obs, actions, cfg.thresholds, horizon, obs_norm=norm)
    d["seconds"] = time.monotonic() - t0
    d["policy"] = label
    return d


# --------------------------------------------------------------------------
# Actuation: load the env at the contract's gains, gate unhonored mismatches, and the
# sim-to-sim gap check (contract gains vs the target's native gains)
# --------------------------------------------------------------------------

GAP_SURV_SHIFT = 0.10     # survival rates this far apart -> "nominal shift"
GAP_TRACK_RATIO = 1.25    # tracking errors this many times apart -> "nominal shift"
HONOR_TARGETS = ("go1",)  # targets whose load_env honors an actuation block


def _load_env(target, obs_spec=None, mode="contract", log=print):
    """``target.load_env()`` with the actuation of the contract at ``obs_spec``.

    mode ``"contract"``: the contract's gains when its block says ``honor: true`` (what the
    pipeline does through ``BP_OBS_SPEC``); ``"native"``: the target's own gains;
    ``"force"``: the contract's gains even when ``honor: false`` (the gap check's twin).
    The env is always loaded native first and the block applied explicitly, so the result
    does not depend on whatever ``BP_OBS_SPEC`` happens to be set in this process.
    """
    from breakpoint_mjx.adapters.actuation import (honoring_enabled, maybe_apply_from_env,
                                                   parse_actuation)

    user_on = honoring_enabled()
    prev = os.environ.get("BP_HONOR_ACTUATION")
    os.environ["BP_HONOR_ACTUATION"] = "0"
    try:
        env = target.load_env()
    finally:
        if prev is None:
            os.environ.pop("BP_HONOR_ACTUATION", None)
        else:
            os.environ["BP_HONOR_ACTUATION"] = prev
    if mode == "native" or not obs_spec or not Path(obs_spec).exists():
        return env
    if mode == "contract" and not user_on:
        return env
    block = parse_actuation(obs_spec)
    if block is None:
        return env
    if target.name not in HONOR_TARGETS:
        if block["honor"] or mode == "force":
            log(f"actuation: honoring a contract's PD gains is implemented for "
                f"{', '.join(HONOR_TARGETS)} only; {target.name} keeps its own gains")
        return env
    maybe_apply_from_env(env, target.name, spec_path=obs_spec, force=(mode == "force"),
                         log=log)
    return env


def gain_gate(obs_spec, robot, allow=False):
    """(level, detail) when the contract's gains differ from the target's; ``fatal`` for a
    >20% mismatch with ``honor: false`` and no override. ``None`` when nothing to say."""
    from breakpoint_mjx.adapters.actuation import gain_gate_row, parse_actuation
    from breakpoint_mjx.adapters.obs_spec import target_interface

    if not obs_spec or not Path(obs_spec).exists():
        return None
    block = parse_actuation(obs_spec)
    if block is None:
        return None
    try:
        iface = target_interface(robot)
    except KeyError:
        return None
    return gain_gate_row(block, iface.kp, iface.kd, allow=allow)


def gap_verdict(d_contract, d_native, honor=True):
    """Compare two ``diagnose`` results: the policy at its contract's gains vs at the
    target's native gains. Pure; returns the verdict dict ``format_gap`` prints."""
    def arm(d):
        return {"survival": 1.0 - float(d["fall_rate"]), "fall_rate": float(d["fall_rate"]),
                "tracking_error": d["tracking_error"], "passed": bool(d["passed"]),
                "n": int(d["n"])}

    c, nat = arm(d_contract), arm(d_native)
    ds = c["survival"] - nat["survival"]
    tc, tn = c["tracking_error"], nat["tracking_error"]
    ratio = None
    if tc is not None and tn is not None and min(tc, tn) > 0:
        ratio = max(tc, tn) / min(tc, tn)
    shift = abs(ds) >= GAP_SURV_SHIFT or (ratio is not None and ratio >= GAP_TRACK_RATIO)
    refuse, msg = False, ""
    if c["passed"] and nat["passed"]:
        verdict = "NOMINAL SHIFT" if shift else "no nominal shift"
    elif c["passed"] or nat["passed"]:
        refuse = True
        where = ("its contract's (training) gains" if c["passed"]
                 else "the target's native gains")
        other = ("the target's native gains" if c["passed"]
                 else "its contract's (training) gains")
        verdict = "WALKS AT ONE GAIN SET ONLY"
        msg = (f"the policy walks at {where} but not at {other}. Nominal behaviour hinges "
               f"on the PD gains, so a failure search would measure the gain choice, not "
               f"the policy. Refused; pass --allow-gain-shift to search anyway (the run "
               f"uses {'the contract' if honor else 'the native'} gains).")
    else:
        refuse = True
        verdict = "WALKS AT NEITHER"
        msg = ("the policy does not walk at nominal with either gain set; the problem is "
               "not (only) the gains. See the diagnoses above.")
    return {"contract": c, "native": nat, "survival_delta": ds, "tracking_ratio": ratio,
            "verdict": verdict, "shift": bool(shift), "refuse": refuse, "message": msg,
            "honor": bool(honor)}


def format_gap(g, gains_c="contract", gains_n="native"):
    """Human-readable gap-check result (both arms, the delta and the verdict)."""
    L = ["sim-to-sim gap check (same policy, same seeds, only the PD gains differ):"]
    for nm, a, kp in (("contract gains", g["contract"], gains_c),
                      ("native gains  ", g["native"], gains_n)):
        trk = "n/a" if a["tracking_error"] is None else f"{a['tracking_error']:.3f}"
        L.append(f"  {nm} ({kp}): survival {a['survival']:.3f} "
                 f"({round(a['survival'] * a['n'])}/{a['n']})  tracking error {trk}  "
                 f"{'walks' if a['passed'] else 'DOES NOT WALK'}")
    rt = "n/a" if g["tracking_ratio"] is None else f"{g['tracking_ratio']:.2f}x"
    L.append(f"  survival delta {g['survival_delta']:+.3f}, tracking ratio {rt} -> "
             f"verdict: {g['verdict']}")
    if g["message"]:
        L.append(f"  {g['message']}")
    return "\n".join(L)


def run_gap_check(args, opts):
    """Nominal rollouts at the contract's gains vs the target's native gains. Returns the
    exit code: EXIT_FAIL when refused or when the run's own arm does not walk."""
    from breakpoint_mjx.adapters.actuation import _fmt, parse_actuation
    from breakpoint_mjx.targets import get_target

    spec = opts.get("obs_spec")
    block = parse_actuation(spec) if spec else None
    if block is None:
        print("check-policy: --gap-check needs a contract with an actuation: block "
              "(breakpoint init writes one from the training config)", file=sys.stderr)
        return 2
    target = get_target(args.robot)
    n = int(getattr(args, "gap_n", None) or 32)
    env_n = _load_env(target, spec, mode="native")
    env_c = _load_env(target, spec, mode="force")
    rec = getattr(env_c, "_bp_actuation", None) or {}
    gains_c = (f"kp {_fmt(rec['kp'])} kd {_fmt(rec['kd'])}" if rec
               else f"kp {block['kp']} kd {block.get('kd')}")
    gains_n = (f"kp {_fmt(rec['native_kp'])} kd {_fmt(rec['native_kd'])}" if rec
               else "native")
    out = {}
    for nm, env in (("contract", env_c), ("native", env_n)):
        print(f"\ngap check: {nm} gains, {n} nominal rollouts ...", flush=True)
        d = run_check(args.robot, args.policy, n=n, horizon=args.horizon, seed=args.seed,
                      opts=opts, ctx={"env": env})
        print(format_diagnosis(d), flush=True)
        out[nm] = d
    g = gap_verdict(out["contract"], out["native"], honor=block["honor"])
    print("\n" + format_gap(g, gains_c, gains_n), flush=True)
    run_arm = out["contract"] if block["honor"] else out["native"]
    if g["refuse"] and not (getattr(args, "allow_gain_shift", False) and run_arm["passed"]):
        return EXIT_FAIL
    return 0 if run_arm["passed"] else EXIT_FAIL


def _carry_extras(prior_path, out_path):
    """Copy the prior contract's actuation / operational blocks into a contract that
    --infer-contract wrote (BCI identifies the obs/action map only)."""
    import yaml

    from breakpoint_mjx.adapters.obs_spec import EXTRA_BLOCKS, _read_yaml

    if not prior_path or not Path(prior_path).exists() or not Path(out_path).exists():
        return
    prior = _read_yaml(prior_path)
    extra = {k: prior[k] for k in EXTRA_BLOCKS if prior.get(k)}
    if not extra:
        return
    d = _read_yaml(out_path)
    d.update(extra)
    Path(out_path).write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None),
                              encoding="utf-8")


# --------------------------------------------------------------------------
# --infer-contract: Behavioral Contract Identification (adapters/contract_id.py)
# --------------------------------------------------------------------------

def _raw_policy(spec, opts):
    """The policy WITHOUT any contract: its own weights, normalization and head."""
    from breakpoint_mjx.adapters.importers import build_policy

    prev = os.environ.pop("BP_OBS_SPEC", None)
    try:
        return build_policy(
            spec, obs_norm_path=opts.get("obs_norm"), activation=opts.get("activation"),
            action_scale=opts.get("action_scale"), squash=bool(opts.get("squash")),
            mean_dim=opts.get("mean_dim"))
    finally:
        if prev is not None:
            os.environ["BP_OBS_SPEC"] = prev


def infer_contract(args, opts, ctx):
    """Run BCI, print the contract diff, write the YAML. Returns (exit_code, path|None)."""
    from breakpoint_mjx.adapters.contract_id import contract_diff, identify_contract
    from breakpoint_mjx.adapters.obs_spec import (build_contract, load_contract,
                                                  target_interface)

    target = ctx.get("target")
    env = ctx.get("env")
    if target is None:
        from breakpoint_mjx.targets import get_target

        target = get_target(args.robot)
        env = _load_env(target, opts.get("obs_spec"))
    try:
        iface = target_interface(target.name)
    except KeyError as e:
        print(f"infer-contract: {e.args[0]}", file=sys.stderr)
        return 2, None
    policy = _raw_policy(args.policy, opts)
    prior = load_contract(opts["obs_spec"]) if opts.get("obs_spec") else None
    budget = int(getattr(args, "bci_budget", None) or 1024)
    print(f"\ninfer-contract: identifying the policy's obs/action contract on "
          f"{target.name} (budget {budget} nominal rollouts; minutes on a CPU) ...",
          flush=True)
    res = identify_contract(policy, target, env, prior_contract=prior, budget=budget,
                            seed=int(args.seed), verbose=True)
    declared = prior if prior is not None else build_contract(
        {"robot": iface.robot, "obs": list(iface.layout)}, robot=iface)
    ev = res.evidence
    b = ev["best"]
    print(f"infer-contract: {res.status.upper()} after {res.rollouts} rollouts "
          f"({res.seconds:.0f}s)")
    print(f"  best:      {b['falls']}/{b['n']} falls (95% CP upper "
          f"{b['fall_cp_upper']:.3f}), {b['success']}/{b['n']} walk and track")
    if "runner_up" in ev:
        r = ev["runner_up"]
        print(f"  runner-up: {r['falls']}/{r['n']} falls, {r['success']}/{r['n']} walk and "
              f"track (a different term/joint structure)")
    print("  contract vs declared:")
    for row in contract_diff(declared, res.contract):
        print(f"    {row}")
    out = Path(getattr(args, "contract_out", None) or "contract.yaml")
    if res.status == "accepted":
        res.contract.to_yaml(out, include_normalization=False)
        _carry_extras(opts.get("obs_spec"), out)
        print(f"  wrote {out}. Use it: breakpoint run --robot {args.robot} --policy "
              f"{args.policy} --obs-spec {out}")
        return 0, out
    if res.status == "ambiguous":
        p1 = out.with_name(out.stem + ".candidate-1" + out.suffix)
        p2 = out.with_name(out.stem + ".candidate-2" + out.suffix)
        res.contract.to_yaml(p1, include_normalization=False)
        _carry_extras(opts.get("obs_spec"), p1)
        if res.runner_up is not None:
            res.runner_up.contract(iface).to_yaml(p2, include_normalization=False)
            _carry_extras(opts.get("obs_spec"), p2)
        print(f"  AMBIGUOUS: the evidence does not separate the top two contracts, so "
              f"none was chosen. Both are written ({p1}, {p2}); pick the one that "
              f"matches your training config and pass it as --obs-spec.")
        return EXIT_FAIL, None
    print("  FAILED: no hypothesis walks at nominal within the budget. The policy may "
          "need an obs term BreakPoint does not produce, or different gains or robot.")
    return EXIT_FAIL, None


def add_args(ap):
    ap.add_argument("--robot", required=True, help="registered robot name (go1, spot, ...)")
    ap.add_argument("--policy", required=True,
                    help="params:/w.npz, onnx:/p.onnx, or a brax params .pkl path")
    ap.add_argument("--n", type=int, default=16, help="nominal rollouts (default 16)")
    ap.add_argument("--horizon", type=int, default=150, help="steps per rollout (default 150)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-norm", dest="obs_norm", default=None,
                    help="BYO: npz with mean/std (also BP_OBS_NORM)")
    ap.add_argument("--action-scale", dest="action_scale", type=float, default=None,
                    help="BYO: scalar action scale (also BP_ACTION_SCALE)")
    ap.add_argument("--squash", action="store_true", help="BYO: tanh-squash the output")
    ap.add_argument("--activation", default=None,
                    help="BYO MLP activation tanh|relu|elu|silu")
    ap.add_argument("--mean-dim", dest="mean_dim", type=int, default=None)
    ap.add_argument("--obs-spec", dest="obs_spec", default=None, metavar="YAML",
                    help="the policy's obs/action contract (breakpoint init writes one; "
                         "also BP_OBS_SPEC)")
    ap.add_argument("--infer-contract", dest="infer_contract", action="store_true",
                    help="if the policy fails at nominal, identify its obs/action contract "
                         "(Behavioral Contract Identification) and write it as YAML")
    ap.add_argument("--contract-out", dest="contract_out", default="contract.yaml",
                    help="where --infer-contract writes the contract (default contract.yaml)")
    ap.add_argument("--bci-budget", dest="bci_budget", type=int, default=1024,
                    help="nominal rollouts --infer-contract may spend (default 1024)")
    ap.add_argument("--gap-check", dest="gap_check", action="store_true",
                    help="sim-to-sim gap check: nominal rollouts at the contract's "
                         "actuation gains vs the target's native gains")
    ap.add_argument("--gap-n", dest="gap_n", type=int, default=32,
                    help="nominal rollouts per arm of --gap-check (default 32)")
    ap.add_argument("--allow-gain-mismatch", dest="allow_gain_mismatch",
                    action="store_true",
                    help="accept a >20%% PD-gain mismatch with actuation.honor: false "
                         "(also BP_ALLOW_GAIN_MISMATCH=1)")
    ap.add_argument("--allow-gain-shift", dest="allow_gain_shift", action="store_true",
                    help="search a policy that walks at only one of the two gain sets")
    return ap


def check(args):
    """Run the check for parsed ``args``; print the diagnosis; return the exit code."""
    opts = {k: getattr(args, k, None)
            for k in ("obs_norm", "action_scale", "squash", "activation", "mean_dim",
                      "obs_spec")}
    if opts["obs_spec"] is None and os.environ.get("BP_OBS_SPEC"):
        opts["obs_spec"] = os.environ["BP_OBS_SPEC"]
    allow = bool(getattr(args, "allow_gain_mismatch", False)) or os.environ.get(
        "BP_ALLOW_GAIN_MISMATCH", "").strip().lower() in ("1", "true", "yes", "on")
    try:
        row = gain_gate(opts["obs_spec"], args.robot, allow=allow)
    except ValueError as e:
        print(f"check-policy: bad contract: {e}", file=sys.stderr)
        return 2
    if row is not None:
        print(f"check-policy: {row[0].upper()}  actuation  {row[1]}", flush=True)
        if row[0] == "fatal":
            return 2
    if getattr(args, "gap_check", False):
        try:
            return run_gap_check(args, opts)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"check-policy: cannot load the policy: {e}", file=sys.stderr)
            return 2
    ctx = {}
    try:
        d = run_check(args.robot, args.policy, n=args.n, horizon=args.horizon,
                      seed=args.seed, opts=opts, ctx=ctx)
    except SystemExit as e:  # pipeline's adapter raises SystemExit(str) on dim mismatch
        if isinstance(e.code, str):
            print(f"check-policy: FAIL\n  {e.code}", file=sys.stderr)
            if getattr(args, "infer_contract", False):
                return infer_contract(args, opts, ctx)[0]
            return EXIT_FAIL
        raise
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"check-policy: cannot load the policy: {e}", file=sys.stderr)
        return 2
    print(format_diagnosis(d), flush=True)
    print(f"  ({d['seconds']:.0f}s)")
    if d["passed"]:
        return 0
    if not getattr(args, "infer_contract", False):
        return EXIT_FAIL
    rc, path = infer_contract(args, opts, ctx)
    if path is None:
        return rc
    print(f"\ncheck-policy: re-checking through {path} ...", flush=True)
    d = run_check(args.robot, args.policy, n=args.n, horizon=args.horizon,
                  seed=args.seed, opts=dict(opts, obs_spec=str(path)), ctx=ctx)
    print(format_diagnosis(d), flush=True)
    return 0 if d["passed"] else EXIT_FAIL


def main(argv=None):
    ap = add_args(argparse.ArgumentParser(
        prog="breakpoint check-policy",
        description="Refuse to search a policy that cannot walk at nominal."))
    return check(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
