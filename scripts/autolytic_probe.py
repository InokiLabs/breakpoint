"""Autolytic probe: did the perturbation break the policy, or did the policy break itself?

Runs each theta TWICE under the same key -- once perturbed ("infected") and once with
every perturbation channel at nominal ("twin") -- and reports the gain
G / E_in, where G is the excess actuator work the perturbed host spent relative to its
twin and E_in is the energy the perturbation injected (see breakpoint_mjx/autolytic.py).

  python scripts/autolytic_probe.py --target go1 --n 16 --horizon 200

With no checkpoint present it uses a random-weight MLP: that validates the measurement
plumbing end to end, NOT the science. The scientific claim (what fraction of REAL
discovered failures are host-amplified) needs a trained policy and is a GPU run.

SCOPE (honest): E_in is mechanical, so gain is defined for MECHANICAL perturbations
(push, slope, mass, CoM). A pure sensor attack (noise/bias/dropout) injects no
mechanical energy, so its denominator is ~0 and the gain is reported as unmeasured
rather than as a spuriously huge number -- pricing sensor attacks needs a different
denominator and is not attempted here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakpoint_mjx.autolytic import (  # noqa: E402
    classify, gain_window, summarize_batch,
)
from breakpoint_mjx.config import load_search_config, load_theta_space  # noqa: E402
from breakpoint_mjx.targets import get_target  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

MECHANICAL = ("push_mag", "push_step_frac", "floor_slope_rad", "torso_mass_add_kg",
              "com_offset_x_m", "com_offset_y_m", "com_offset_z_m", "friction_floor")


def _random_policy(obs_dim, act_dim, seed=0):
    from breakpoint_mjx.adapters import from_params

    rng = np.random.default_rng(seed)
    return from_params([
        (rng.normal(0, 0.05, (obs_dim, 32)), np.zeros(32)),
        (rng.normal(0, 0.05, (32, act_dim)), np.zeros(act_dim)),
    ])


def build_thetas(space, n, rng):
    """Mechanical-perturbation thetas: pushes and slopes at varied strengths."""
    names = space.names()
    nominal = np.array(space.nominal_vector(), dtype=float)
    rows = np.tile(nominal, (n, 1))
    for i in range(n):
        frac = (i + 1) / n
        for dim, lo_hi in (("push_mag", None), ("push_step_frac", (0.2, 0.8)),
                           ("floor_slope_rad", None)):
            if dim not in names:
                continue
            j = names.index(dim)
            d = space.dims[dim]
            if dim == "push_step_frac":
                rows[i, j] = rng.uniform(*lo_hi)
            elif dim == "push_mag":
                rows[i, j] = d.bounds[0] + frac * (d.bounds[1] - d.bounds[0])
            else:
                rows[i, j] = d.nominal + (frac - 0.5) * (d.bounds[1] - d.nominal)
    return rows


def twin_of(space, thetas):
    """The same worlds with every ROLLOUT perturbation off (the uninfected control).

    Model dims stay as-is: the twin must be the SAME robot in the SAME world, differing
    only by the attack. Zeroing model dims too would compare different robots.
    """
    names = space.names()
    twin = np.array(thetas, dtype=float, copy=True)
    for k, d in space.dims.items():
        if d.kind == "rollout":
            twin[:, names.index(k)] = d.nominal
    return twin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="go1")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--window", type=int, default=75, help="steps before collapse to integrate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.evaluator import make_evaluator

    target = get_target(args.target)
    env = target.load_env()
    consts = target.resolve_constants(env)
    space = load_theta_space(target.theta_space_path)
    cfg = load_search_config(config_path("search.yaml"))

    obs_dim = int(env.observation_size["state"][0])
    policy = _random_policy(obs_dim, env.action_size)
    print(f"[probe] {args.target}: random-weight policy, N={args.n}, H={args.horizon}")

    rng = np.random.default_rng(0)
    thetas = build_thetas(space, args.n, rng)
    twins = twin_of(space, thetas)

    # collect_energy rides on the shared rollout, so infected and twin are scored by
    # the identical code path under the identical key.
    def _make_eval():
        from breakpoint_mjx.rollout import rollout_one
        from breakpoint_mjx.thetas import apply_thetas
        from breakpoint_mjx.rollout import env_key

        base_model = env.mjx_model
        summarize = lambda st: target.objective.summarize(st, consts)  # noqa: E731

        def one(model, rollout_i, key):
            return rollout_one(env, policy.policy_fn, model, rollout_i, key,
                               args.horizon, summarize, collect_energy=True)

        def _ev(th, key):
            batched, in_axes, rollout = apply_thetas(target, space, base_model, th, consts)
            keys = jax.vmap(lambda i: env_key(key, i))(jnp.arange(th.shape[0]))
            return jax.vmap(one, in_axes=(in_axes, {k: 0 for k in rollout}, 0))(
                batched, rollout, keys)

        return jax.jit(_ev)

    evaluate = _make_eval()
    key = jax.random.PRNGKey(0)
    inf = evaluate(jnp.asarray(thetas), key)
    twin = evaluate(jnp.asarray(twins), key)

    h = np.asarray(inf["trunk_height"])
    up = np.asarray(inf["up_proj"])
    fallen = (h < cfg.thresholds["trunk_height_min"]) | (up < cfg.thresholds["up_proj_min"])
    w_inf = np.asarray(inf["self_work"])
    w_twin = np.asarray(twin["self_work"])
    p_in = np.asarray(inf["input_power"])

    gains, rows = [], []
    for i in range(len(thetas)):
        g, G, E_in, (lo, hi) = gain_window(
            w_inf[i], w_twin[i], p_in[i], fallen_flags=fallen[i], window_steps=args.window)
        gains.append(g)
        rows.append({
            "i": i, "gain": g, "G": G, "E_in": E_in, "window": [lo, hi],
            "fell": bool(fallen[i].any()), "class": classify(g),
            "push_mag": float(thetas[i][space.names().index("push_mag")]),
        })

    summary = summarize_batch(gains)
    print(f"\n{'i':>3} {'push':>6} {'fell':>5} {'E_in':>10} {'G':>12} {'gain':>9}  class")
    for r in rows:
        gs = "  n/a" if r["gain"] is None else f"{r['gain']:9.2f}"
        print(f"{r['i']:>3} {r['push_mag']:6.2f} {str(r['fell']):>5} "
              f"{r['E_in']:10.4f} {r['G']:12.3f} {gs}  {r['class']}")
    print(f"\n[summary] {json.dumps(summary, indent=2)}")

    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
        print("wrote", args.out)
    return rows, summary


if __name__ == "__main__":
    main()
