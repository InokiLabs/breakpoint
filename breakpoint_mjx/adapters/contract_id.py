"""Behavioral Contract Identification (BCI): infer a frozen policy's obs/action interface.

An outside team's policy (Isaac Lab, legged_gym, unitree_rl_gym, RSL-RL) was trained on
a specific observation vector: which terms, in which order, with which per-term scales,
joint order and default pose, and a specific action scale. Get any of it wrong and the
policy still runs; it just falls, and a failure search on it reports noise. Today the
fix is a hand-written mapping (Isaac Lab's sim-to-sim transfer uses a hand-listed YAML of
joint names per backend; unitree_rl_gym deploy configs hand-list scales and default
angles). BCI identifies the contract from the checkpoint and the simulator, with no
training, and refuses to guess when the evidence does not separate two contracts.

The contract is DATA (index arrays and per-dim affine coefficients, see
``obs_spec.PolicyContract``), so candidate contracts ``vmap`` together: one batched
rollout call evaluates many hypotheses at once.

Method
------
Stage A, moment matching (no policy rollouts). Roll the target env under zero action for
a short warmup and record per-channel mean/std of every obs channel. Where the checkpoint
ships its own running mean/std (brax, RSL-RL, Isaac Lab all do), match the policy's input
slots to named target terms: enumerate how the policy's input splits into term-sized
slots, assign terms to slots within each dimension group (Hungarian assignment on a
moment-matching cost, then a ranked list of the next-best assignments), and estimate each
term's scale in closed form from the std ratio. Joint permutations are restricted to the
leg-symmetry group of the robot (leg-major vs joint-type-major layouts x every leg order:
48 orders for a quadruped, not 12! = 479,001,600). Without checkpoint statistics the
first-layer weight-column norms stand in as a weak prior on input magnitude.

Closed-loop moment re-estimation. A standing robot says little about the statistics of
joint velocities or actions while walking. So once any candidate walks, its pre-fall
closed-loop obs statistics replace the standing warmup as the reference and Stage A
re-ranks every hypothesis against them. Scales of dynamic terms and the action scale are
re-estimated the same way (the last_action slot pins the action scale, because the
policy sees its own raw output there and the env reports the scaled action).

Stage B, racing. The top hypotheses run as ONE batched rollout call on common reset
keys; each is scored by survival and command tracking; successive halving keeps the
better half each round with more rollouts per survivor. Acceptance is a Clopper-Pearson
rule: the winner is accepted only if the one-sided 95% upper bound on its nominal fall
rate is <= ``accept_fall_ub`` (0.2) AND its success rate (no fall and tracking error
within the target threshold) beats the runner-up's with CP-separated bounds. Otherwise
the result is ``ambiguous`` and both contracts are reported; nothing is silently picked.

Import-safety: numpy only at module import; jax is imported inside the rollout builder.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from breakpoint_mjx.adapters.obs_spec import (
    JOINT_TERMS,
    PolicyContract,
    build_contract,
    target_interface,
)

VEC_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity", "commands")
EXOGENOUS = ("projected_gravity", "commands")   # set by physics / the env, not the policy
EXO_WEIGHT = 10.0
ACCEPT_FALL_UB = 0.2
DELTA = 0.05
STD_FLOOR = 1e-3


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def cp_upper(k, n, delta=DELTA):
    """One-sided (1 - delta) Clopper-Pearson upper bound on a binomial rate."""
    from breakpoint_mjx.riskcert import clopper_pearson_upper

    return clopper_pearson_upper(int(k), int(n), delta)


def cp_lower(k, n, delta=DELTA):
    """One-sided (1 - delta) Clopper-Pearson lower bound on a binomial rate."""
    if int(k) <= 0:
        return 0.0
    return 1.0 - cp_upper(int(n) - int(k), int(n), delta)


def masked_moments(obs, mask):
    """Per-channel mean/std of ``obs`` (N, H, D) over the steps where ``mask`` (N, H)."""
    obs = np.asarray(obs, dtype=float)
    d = obs.shape[-1]
    flat = obs[mask] if mask is not None and mask.any() else obs.reshape(-1, d)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.mean(axis=0), flat.std(axis=0), int(flat.shape[0])


# --------------------------------------------------------------------------- #
# The leg-symmetry group of joint orders
# --------------------------------------------------------------------------- #
def _split_joint(name):
    """``FR_thigh_joint`` -> (``FR``, ``thigh``)."""
    s = str(name)
    if s.lower().endswith("_joint"):
        s = s[: -len("_joint")]
    leg, _, jt = s.partition("_")
    return leg, jt


def symmetry_orders(iface):
    """Every joint order in the robot's leg-symmetry group, as policy-order name lists.

    Legs are permuted as whole units (so a hip is always a hip), and the joints are laid
    out either leg-major (FR_hip, FR_thigh, FR_calf, FL_hip, ...) or joint-type-major
    (FR_hip, FL_hip, RR_hip, RL_hip, FR_thigh, ...). That covers the orders the big
    frameworks actually produce (Playground/MuJoCo menagerie leg-major FR,FL,RR,RL;
    legged_gym leg-major FL,FR,RL,RR; Isaac Lab joint-type-major FL,FR,RL,RR) plus every
    left/right and front/rear swap: 2 x 4! = 48 orders for a quadruped.
    """
    names = list(iface.joint_names)
    parts = [_split_joint(n) for n in names]
    legs = list(dict.fromkeys(p[0] for p in parts))
    types = list(dict.fromkeys(p[1] for p in parts))
    lookup = {p: n for p, n in zip(parts, names)}
    if len(legs) * len(types) != len(names) or len(lookup) != len(names):
        return [names]          # not a regular leg x joint-type grid: identity only
    orders, seen = [], set()
    for lp in itertools.permutations(legs):
        for layout in ("leg", "type"):
            if layout == "leg":
                o = [lookup[(lg, t)] for lg in lp for t in types]
            else:
                o = [lookup[(lg, t)] for t in types for lg in lp]
            if tuple(o) not in seen:
                seen.add(tuple(o))
                orders.append(o)
    return orders


def perm_of(order, iface):
    """Env actuator index of each joint in ``order``."""
    idx = {n: i for i, n in enumerate(iface.joint_names)}
    return np.asarray([idx[n] for n in order], dtype=int)


# --------------------------------------------------------------------------- #
# Hypotheses
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    """One candidate contract, in declarative form plus the Stage A cost."""

    terms: tuple            # canonical term names in policy order
    scales: dict            # term -> scalar or per-channel list
    order: tuple            # joint names in policy order
    action_scale: float     # policy action scale, rad per unit
    cost: float = 0.0
    origin: str = "stageA"

    def spec(self, robot):
        obs = []
        for t in self.terms:
            e = {"term": t}
            s = self.scales.get(t, 1.0)
            if not (np.isscalar(s) and float(s) == 1.0):
                e["scale"] = (float(s) if np.isscalar(s)
                              else [float(x) for x in np.asarray(s).reshape(-1)])
            obs.append(e)
        return {"version": 1, "robot": robot, "joint_names": list(self.order),
                "action": {"scale": float(self.action_scale)}, "obs": obs}

    def contract(self, iface):
        return build_contract(self.spec(iface.robot), robot=iface)

    def key(self):
        """Discrete identity: term order + joint order (scales are continuous)."""
        return (self.terms, tuple(self.order))


def term_dims(iface):
    return {t: iface.layout[t][1] for t in iface.layout}


def segmentations(iface, in_dim, terms=None):
    """Every ordered split of ``in_dim`` into distinct target terms whose dims add up.

    Returns a list of term-dimension sequences, e.g. (3, 3, 3, 12, 12, 12, 3). For Go1's
    48-dim policy that is the C(7, 3) = 35 ways to interleave four 3-dim and three
    12-dim terms. Term subsets are allowed when ``in_dim`` is smaller than the full layout
    (a 45-dim unitree_rl_gym policy has no linear-velocity term).
    """
    dims = term_dims(iface)
    names = list(terms or dims)
    out = set()
    for r in range(1, len(names) + 1):
        for sub in itertools.combinations(names, r):
            if sum(dims[t] for t in sub) != in_dim:
                continue
            ms = tuple(sorted(dims[t] for t in sub))
            for seq in set(itertools.permutations(ms)):
                out.add((seq, tuple(sorted(sub))))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Stage A: moment matching
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    """Per-channel env statistics (target obs space) used as the matching reference."""

    mean: np.ndarray
    std: np.ndarray
    source: str = "warmup"
    act_std: object = None   # per-actuator std of the applied env action (closed loop)
    warm_mean: object = None  # the standing warmup, kept for the exogenous terms
    warm_std: object = None

    @property
    def closed_loop(self):
        return self.act_std is not None


def _block_cost(mu_p, sd_p, mu_e, sd_e, fixed_scale=None):
    """Moment-matching cost of explaining policy slot stats by env channels x a scale.

    Returns ``(cost, scale)``. The scale is the closed-form geometric mean of the std
    ratios, signed by the means when they are decisive. The cost is the spread of the
    per-channel log std ratios (a wrong channel profile) plus the mean mismatch in units
    of the policy's own std.
    """
    sd_p = np.maximum(np.asarray(sd_p, float), STD_FLOOR)
    sd_e = np.maximum(np.asarray(sd_e, float), STD_FLOOR)
    mu_p = np.asarray(mu_p, float)
    mu_e = np.asarray(mu_e, float)
    lr = np.log(sd_p) - np.log(sd_e)
    decisive = (np.abs(mu_e) > 2 * sd_e) & (np.abs(mu_p) > 2 * sd_p)
    if fixed_scale is None and decisive.any():
        # a channel with a decisive mean (gravity's z) pins the scale by least squares on
        # the means; its std is only sensor noise
        c = float(np.sum(mu_p[decisive] * mu_e[decisive]) / np.sum(mu_e[decisive] ** 2))
        c = c if abs(c) > 1e-6 else 1.0
    elif fixed_scale is None:
        mag = float(np.exp(np.mean(lr)))
        dot = float(np.sum(mu_p * mu_e))
        sgn = -1.0 if dot < 0 and abs(dot) > 0.5 * float(np.sum(sd_p * sd_e)) else 1.0
        c = sgn * mag
    else:
        c = float(np.mean(np.asarray(fixed_scale, float)))
    spread = float(np.mean((lr - math.log(abs(c))) ** 2))
    mean_err = float(np.mean(np.minimum(((mu_p - c * mu_e) / sd_p) ** 2, 25.0)))
    return spread + 0.5 * mean_err, c


def _vec_signs(mu_p, sd_p, mu_e, sd_e, c):
    """Per-channel sign flips for a 3-vector term (gravity/velocity frame conventions).

    A channel flips only when its mean is decisive on both sides (|mean| > 2 std), so
    noise can never introduce a flip."""
    s = np.ones(len(mu_p))
    for i in range(len(mu_p)):
        if abs(mu_p[i]) > 2 * sd_p[i] and abs(mu_e[i]) > 2 * max(sd_e[i], STD_FLOOR):
            s[i] = 1.0 if np.sign(mu_p[i]) == np.sign(c * mu_e[i]) else -1.0
    return s


def _type_consistency(mu_p, sd_p, types):
    """Leg-order-free evidence for a joint LAYOUT: channels of the same joint type (all
    hips, all calves) should look alike in the policy's own statistics, and the types
    should differ. Returns the within-type share of the variance of two features
    (log std and mean/std), in [0, 1]; low = this layout fits."""
    sd = np.maximum(np.asarray(sd_p, float), STD_FLOOR)
    feats = [np.log(sd), np.clip(np.asarray(mu_p, float) / sd, -5, 5)]
    types = np.asarray(types)
    out = []
    for f in feats:
        tot = float(np.sum((f - f.mean()) ** 2))
        if tot < 1e-9:
            out.append(1.0)
            continue
        within = sum(float(np.sum((f[types == t] - f[types == t].mean()) ** 2))
                     for t in np.unique(types))
        out.append(within / tot)
    return float(np.mean(out))


def stage_a(iface, in_dim, pol_mean, pol_std, ref, *, top_k=288, orders=None,
            action_scales=None, fixed=None):
    """Rank contract hypotheses by moment matching. Zero policy rollouts.

    ``pol_mean``/``pol_std`` are the policy's expected input statistics (the checkpoint
    normalizer, or a weak prior); ``ref`` is the env reference. The cost of a hypothesis
    sums, per slot:

    * 3-vector terms: ``_block_cost`` against ``ref`` (gravity and the command channels
      match a standing warmup exactly; body velocities only once ``ref`` is closed-loop).
    * joint terms: the layout's ``_type_consistency`` (which slot channels are hips,
      thighs, calves); with a closed-loop ``ref`` also ``_block_cost`` per channel
      (this is what orders the legs); ``joint_vel`` additionally pays for non-zero
      means (a joint velocity averages zero over a gait).

    Scales: every term's scale comes from the closed-form std ratio against ``ref``,
    except that against a standing warmup the scales of the dynamic terms (body and
    joint velocities, joint positions) are set to 1: a standing robot's statistics are
    sensor noise, and a scale fitted to noise is worse than the unit prior. The action
    scale is pinned by the last_action slot once ``ref`` is closed-loop, else a coarse
    log grid (``action_scales``). Hypotheses tie on leg order when nothing separates
    them; they are all returned and racing separates them.
    """
    pol_mean = np.asarray(pol_mean, float)
    pol_std = np.asarray(pol_std, float)
    orders = orders or symmetry_orders(iface)
    perms = [perm_of(o, iface) for o in orders]
    types = [_split_joint(n)[1] for n in iface.joint_names]
    dims = term_dims(iface)
    fixed = dict(fixed or {})
    closed = ref.closed_loop
    cache = {}

    def jcost(term, start, pi):
        k = (term, start, pi)
        if k in cache:
            return cache[k]
        s0, d = iface.layout[term]
        mp, sp = pol_mean[start:start + d], pol_std[start:start + d]
        c = _type_consistency(mp, sp, [types[e] for e in perms[pi]])
        scale = 1.0
        if term == "joint_vel":
            c += 0.5 * float(np.mean(np.minimum((mp / np.maximum(sp, STD_FLOOR)) ** 2, 4)))
        if closed:
            ch = np.arange(s0, s0 + d)[perms[pi]]
            if term == "last_action":
                me = np.zeros(d)
                se = np.asarray(ref.act_std)[perms[pi]]
                bc, _ = _block_cost(mp - mp.mean(), sp, me, se)
                c += bc
            else:
                bc, scale = _block_cost(mp, sp, ref.mean[ch], ref.std[ch], fixed.get(term))
                c += bc
        cache[k] = (c, scale)
        return cache[k]

    def vcost(term, start):
        k = (term, start)
        if k in cache:
            return cache[k]
        s0, d = iface.layout[term]
        mp, sp = pol_mean[start:start + d], pol_std[start:start + d]
        rm, rs = ref.mean[s0:s0 + d], ref.std[s0:s0 + d]
        if term in EXOGENOUS and ref.warm_mean is not None:
            # gravity and the command are not driven by the policy: the standing warmup
            # is their exact reference, whatever the gait looks like
            rm, rs = ref.warm_mean[s0:s0 + d], ref.warm_std[s0:s0 + d]
        c, scale = _block_cost(mp, sp, rm, rs, fixed.get(term))
        if term in EXOGENOUS:
            # an exact reference: a mismatch here is evidence, not train/test shift, so it
            # weighs more than a dynamic term's (whose training statistics include the
            # training perturbations and never match a nominal gait closely)
            c *= EXO_WEIGHT
        if not closed and term in ("base_lin_vel", "base_ang_vel"):
            # standing: the body velocities are sensor noise; only the mean terms speak
            c = 0.5 * float(np.mean(np.minimum((mp / np.maximum(sp, STD_FLOOR)) ** 2, 4)))
            scale = 1.0
        sg = _vec_signs(mp, sp, rm, rs, scale)
        sc = (abs(scale) * sg).tolist() if np.any(sg < 0) else abs(scale)
        cache[k] = (c, sc)
        return cache[k]

    structures = []
    for seq, subset in segmentations(iface, in_dim):
        starts = np.concatenate([[0], np.cumsum(seq)[:-1]]).astype(int)
        slots_by_dim = {}
        for si, d in enumerate(seq):
            slots_by_dim.setdefault(d, []).append(si)
        terms_by_dim = {}
        for t in subset:
            terms_by_dim.setdefault(dims[t], []).append(t)
        # vector-term assignment is independent of the joint order
        vec_opts = []
        for d, slots in slots_by_dim.items():
            if not any(t in VEC_TERMS for t in terms_by_dim[d]):
                continue
            opts = []
            for assign in itertools.permutations(terms_by_dim[d]):
                opts.append((sum(vcost(t, int(starts[s]))[0] for t, s in zip(assign, slots)),
                             dict(zip(slots, assign))))
            opts.sort(key=lambda z: z[0])
            vec_opts.append(opts[:4])
        joint_groups = [(d, slots) for d, slots in slots_by_dim.items()
                        if all(t in JOINT_TERMS for t in terms_by_dim[d])]
        for pi in range(len(perms)):
            jopts_all = []
            for d, slots in joint_groups:
                opts = []
                for assign in itertools.permutations(terms_by_dim[d]):
                    opts.append((sum(jcost(t, int(starts[s]), pi)[0]
                                     for t, s in zip(assign, slots)), dict(zip(slots, assign))))
                opts.sort(key=lambda z: z[0])
                jopts_all.append(opts[:3])
            for combo in itertools.product(*vec_opts, *jopts_all):
                total = sum(c for c, _ in combo)
                slot_term = {}
                for _, m in combo:
                    slot_term.update(m)
                terms = tuple(slot_term[i] for i in range(len(seq)))
                structures.append((total, terms, tuple(int(x) for x in starts), pi))
    structures.sort(key=lambda z: z[0])

    hyps, seen = [], set()
    for total, terms, starts, pi in structures:
        if (terms, pi) in seen:
            continue
        seen.add((terms, pi))
        scales = {}
        for t, st in zip(terms, starts):
            if t in VEC_TERMS:
                scales[t] = vcost(t, st)[1]
            elif t != "last_action":
                scales[t] = jcost(t, st, pi)[1]
        for t, v in fixed.items():
            scales[t] = v
        for a_s in _action_scale_candidates(iface, terms, starts, pi, perms, pol_std, ref,
                                            action_scales):
            hyps.append(Hypothesis(terms=terms, scales=dict(scales), order=tuple(orders[pi]),
                                   action_scale=a_s, cost=total,
                                   origin="stageA:" + ref.source))
        if len(hyps) >= top_k:
            break
    return hyps[:top_k]


def _action_scale_candidates(iface, terms, starts, pi, perms, pol_std, ref, given):
    """Action-scale hypotheses (rad per unit of the policy's raw output).

    The policy sees its own raw output in its last_action slot while the env reports the
    applied, scaled action; so with closed-loop statistics the ratio of the two stds pins
    the scale (continuous, not a grid). A standing warmup has no action statistics, so
    the fallback is a coarse log2 grid around the target's own scale.
    """
    if given:
        return list(given)
    if "last_action" in terms and ref.closed_loop:
        st = int(starts[list(terms).index("last_action")])
        d = iface.layout["last_action"][1]
        env_sd = np.maximum(np.asarray(ref.act_std)[perms[pi]], STD_FLOOR)
        pol_sd = np.maximum(pol_std[st:st + d], STD_FLOOR)
        ratio = float(np.exp(np.mean(np.log(env_sd) - np.log(pol_sd))))
        return [ratio * iface.action_scale]
    return [iface.action_scale * f for f in (1.0, 0.5, 2.0)]


# --------------------------------------------------------------------------- #
# Stage B: batched closed-loop racing (jax; compiled once per target/horizon/batch)
# --------------------------------------------------------------------------- #
def policy_core(policy):
    """The policy's own math on a flat input vector, contract excluded, as DATA.

    Returns ``(apply, weights, meta)`` where ``apply(weights, x, xp) -> y`` is pure and
    ``weights`` is a pytree of arrays (MLP layers, normalization, head scale/bias). The
    weights are an ARGUMENT of the jitted racer, so one compile serves every policy with
    the same layer shapes. ``meta`` has ``in_dim``, ``norm`` ((mean, std) or None) and
    ``first_w``.

    For an ``MLPPolicy`` the core is its normalization, MLP and head (mean slice, squash,
    policy-side scale/bias). Index maps, per-term scales and the action permutation are
    what BCI identifies, so they are NOT part of the core.
    """
    from breakpoint_mjx.adapters.mlp import mlp_forward

    sp = policy.obs_spec
    d = int(policy.in_dim)
    mean = np.zeros(d) if sp is None or sp.mean is None else np.asarray(sp.mean, float)
    std = np.ones(d) if sp is None or sp.std is None else np.asarray(sp.std, float)
    has_norm = sp is not None and sp.mean is not None and sp.std is not None
    ot = policy.out_transform
    mean_dim = None if ot is None or ot.mean_dim is None else int(ot.mean_dim)
    squash = False if ot is None or not ot.squash else (
        ot.squash if isinstance(ot.squash, str) else "tanh")
    a = int(policy.action_size)
    hs = np.broadcast_to(np.asarray(1.0 if ot is None else ot.scale, float), (a,)).copy()
    hb = np.broadcast_to(np.asarray(0.0 if ot is None else ot.bias, float), (a,)).copy()
    act = policy.activation
    weights = {"layers": [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                          for W, b in policy.params],
               "mean": mean.astype(np.float32), "std": std.astype(np.float32),
               "hs": hs.astype(np.float32), "hb": hb.astype(np.float32)}

    def apply(w, x, xp):
        x = (x - w["mean"]) / w["std"]
        y = mlp_forward(w["layers"], x, act, xp)
        if mean_dim is not None:
            y = y[..., :mean_dim]
        if squash == "tanh":
            y = xp.tanh(y)
        elif squash == "sigmoid":
            y = 1.0 / (1.0 + xp.exp(-y))
        return y * w["hs"] + w["hb"]

    meta = {"in_dim": d, "norm": (mean, std) if has_norm else None,
            "first_w": np.asarray(policy.params[0][0]),
            "signature": (act, mean_dim, squash,
                          tuple(np.asarray(W).shape for W, _ in policy.params))}
    return apply, weights, meta


def input_stats(in_dim, norm, first_w):
    """The policy's expected input statistics for Stage A.

    With checkpoint normalization: its mean/std exactly. Without: a weak prior from the
    first-layer weights (a network trained on an input of typical magnitude m tends to
    learn column norms of order 1/m), with zero means.
    """
    if norm is not None:
        return np.asarray(norm[0], float), np.asarray(norm[1], float), "checkpoint"
    w = np.linalg.norm(np.asarray(first_w, float), axis=1)
    pos = w[w > 0]
    w = np.where(w > 0, w, np.median(pos) if pos.size else 1.0)
    return np.zeros(in_dim), 1.0 / w, "first-layer weight norms (weak prior)"


class Racer:
    """Batched nominal rollouts of many contracts at once, on common reset keys.

    ``race(contracts, n, seed, weights)`` rolls every contract ``n`` times: all
    ``len(contracts) * n`` rollouts go through ONE jitted ``vmap`` over (contract arrays,
    reset key), chunked into a fixed batch, with the policy weights as an argument, so it
    compiles once per (target, horizon, batch, layer shapes).
    """

    def __init__(self, env, target, apply_fn, horizon=150, batch=64, thresholds=None,
                 signature=None):
        import jax
        import jax.numpy as jnp

        self.env, self.target = env, target
        self.horizon, self.batch = int(horizon), int(batch)
        self.signature = signature
        consts = target.resolve_constants(env)
        summarize = target.objective.summarize
        if thresholds is None:
            from breakpoint_mjx.config import load_search_config, merge_target_thresholds
            from breakpoint_mjx.paths import config_path

            thresholds = merge_target_thresholds(
                load_search_config(config_path("search.yaml")), target).thresholds
        self.thresholds = thresholds
        self.rollouts = 0
        self.seconds = 0.0
        H = self.horizon

        def one(w, arrs, key):
            obs_idx, off, sc, aidx, asc, aoff = arrs
            st = env.reset(key)

            def step(st, _):
                src = jnp.asarray(st.obs["state"])
                x = (src[obs_idx] - off) * sc
                y = apply_fn(w, x, jnp)
                a = y[aidx] * asc + aoff
                st2 = env.step(st, a)
                return st2, (summarize(st2, consts), src, a)

            _, out = jax.lax.scan(step, st, None, length=H)
            return out

        def batched(w, arrs, keys):
            with jax.default_matmul_precision("highest"):
                return jax.vmap(one, in_axes=(None, 0, 0))(w, arrs, keys)

        self._fn = jax.jit(batched)
        self._jax, self._jnp = jax, jnp

    def race(self, contracts, n, seed, weights):
        """Roll each contract ``n`` times (common reset keys across contracts)."""
        jax, jnp = self._jax, self._jnp
        t0 = time.monotonic()
        m = len(contracts)
        arrs = [tuple(np.asarray(a, np.float32) if j not in (0, 3) else np.asarray(a, np.int32)
                      for j, a in enumerate(c.arrays())) for c in contracts]
        rows = [(ci, r) for ci in range(m) for r in range(n)]
        rkeys = np.asarray(jax.random.split(jax.random.PRNGKey(seed), max(n, 1)))
        outs = []
        for lo in range(0, len(rows), self.batch):
            chunk = rows[lo:lo + self.batch]
            chunk_p = chunk + [chunk[-1]] * (self.batch - len(chunk))
            stacked = tuple(jnp.asarray(np.stack([arrs[ci][j] for ci, _ in chunk_p]))
                            for j in range(6))
            keys = jnp.asarray(np.stack([rkeys[r] for _, r in chunk_p]))
            tr, src, act = self._fn(weights, stacked, keys)
            k = len(chunk)
            outs.append(({kk: np.asarray(v)[:k] for kk, v in tr.items()},
                         np.asarray(src)[:k], np.asarray(act)[:k]))
        tr = {k: np.concatenate([o[0][k] for o in outs]) for k in outs[0][0]}
        src = np.concatenate([o[1] for o in outs])
        act = np.concatenate([o[2] for o in outs])
        self.rollouts += len(rows)
        self.seconds += time.monotonic() - t0
        comp = self.target.objective.component_signals
        trk_max = float(self.thresholds.get("tracking_err_max", 1e9))
        res = []
        for ci in range(m):
            sel = [i for i, (c, _) in enumerate(rows) if c == ci]
            sig = [comp({k: v[i] for k, v in tr.items()}, self.thresholds) for i in sel]
            falls = np.array([s["fall"] for s in sig])
            trk = np.array([s["tracking_error"] for s in sig])
            pre = np.array([_pre_fall_tracking({k: v[i] for k, v in tr.items()}, s)
                            for i, s in zip(sel, sig)])
            res.append({"n": len(sel), "falls": int(falls.sum()), "H": self.horizon,
                        "ttf": np.array([s["time_to_failure"] for s in sig]),
                        "tracking": trk, "trk_pre": pre,
                        "success": int(((falls == 0) & (trk <= trk_max)).sum()),
                        "src": src[sel], "act": act[sel]})
        return res


def score(r):
    """Racing score (higher = better): minus the mean per-rollout loss ``_loss``."""
    return -_loss(r, r.get("H", float(np.max(r["ttf"]))))


def closed_loop_reference(results, warm=None, min_steps=300):
    """Pre-fall obs/action statistics pooled from the rollouts that stayed up longest.

    The closed-loop moment re-estimation step: env channels are physical, so their
    statistics under ANY contract that walks are a valid reference for re-matching
    every hypothesis. Returns None if the candidates did not stay up long enough."""
    best = sorted(results, key=score, reverse=True)[:4]
    srcs, acts, masks = [], [], []
    for r in best:
        n, h = r["src"].shape[:2]
        lim = np.clip(r.get("src_ttf", r["ttf"]), 0, h).reshape(n, 1)
        mask = np.arange(h)[None, :] < lim
        mask[:, :10] = False        # the reset transient is not the gait
        srcs.append(r["src"])
        acts.append(r["act"])
        masks.append(mask)
    mask = np.concatenate(masks)
    if mask.sum() < min_steps:
        return None
    mu, sd, _ = masked_moments(np.concatenate(srcs), mask)
    _, asd, _ = masked_moments(np.concatenate(acts), mask)
    return Reference(mean=mu, std=sd, source="closed-loop", act_std=asd,
                     warm_mean=None if warm is None else warm.mean,
                     warm_std=None if warm is None else warm.std)


@dataclass
class BCIResult:
    status: str                       # "accepted" | "ambiguous" | "failed"
    contract: object = None           # PolicyContract of the best hypothesis
    hypothesis: object = None
    runner_up: object = None          # best hypothesis with a different discrete structure
    evidence: dict = field(default_factory=dict)
    rollouts: int = 0
    seconds: float = 0.0
    log: list = field(default_factory=list)

    @property
    def accepted(self):
        return self.status == "accepted"


def _hkey(h):
    s = tuple((t, tuple(np.round(np.log2(np.abs(np.asarray(h.scales.get(t, 1.0), float))
                                          .reshape(-1)), 1)))
              for t in h.terms)
    return (h.key(), s, round(math.log2(h.action_scale), 1))


def _dedupe(hyps):
    out, seen = [], set()
    for h in hyps:
        k = _hkey(h)
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def random_hypotheses(iface, in_dim, n, rng, scale_grid=(0.5, 1.0, 2.0),
                      action_grid=(0.5, 1.0, 2.0)):
    """Uniform samples from the same hypothesis space BCI searches (the baseline).

    Term order uniform over the orderings of a dimension-consistent term set, joint order
    uniform over the leg-symmetry group, each non-action term scale uniform on
    ``scale_grid``, action scale uniform on ``action_grid`` x the target's scale."""
    orders = symmetry_orders(iface)
    segs = segmentations(iface, in_dim)
    subsets = sorted({sub for _, sub in segs})
    out = []
    for _ in range(n):
        sub = list(subsets[rng.integers(len(subsets))])
        terms = tuple(sub[i] for i in rng.permutation(len(sub)))
        scales = {t: float(rng.choice(scale_grid)) for t in terms if t != "last_action"}
        out.append(Hypothesis(terms=terms, scales=scales,
                              order=tuple(orders[rng.integers(len(orders))]),
                              action_scale=float(rng.choice(action_grid)) * iface.action_scale,
                              origin="random"))
    return out


def identify_contract(policy, target, env=None, *, prior_contract=None, budget=1024,
                      horizon=150, batch=64, n_first=288, n_final=32, refine_reserve=224,
                      seed=0, racer=None, reference=None, method="bci", thresholds=None,
                      verbose=False):
    """Identify ``policy``'s contract on ``target`` (see the module docstring).

    ``policy`` is an ``MLPPolicy`` WITHOUT a contract applied (its own normalization and
    head only). ``prior_contract`` (a ``PolicyContract``, e.g. from ``breakpoint init``)
    joins the first racing round. ``budget`` caps racing rollouts (the warmup is counted
    too). ``method="random"`` swaps Stage A and the re-estimation for uniform samples of
    the same hypothesis space under the same racing and acceptance rule (the baseline).
    Returns a ``BCIResult``; ``status == "accepted"`` only under the Clopper-Pearson rule.
    """
    t0 = time.monotonic()
    iface = target_interface(target.name)
    log = []

    def say(msg):
        log.append(msg)
        if verbose:
            print(f"  [bci] {msg}", flush=True)

    apply_fn, weights, meta = policy_core(policy)
    in_dim = meta["in_dim"]
    pmu, psd, src = input_stats(in_dim, meta["norm"], meta["first_w"])
    say(f"policy input dim {in_dim}; input statistics from {src}")
    if env is None:
        env = target.load_env()
    if racer is None or racer.signature != meta["signature"]:
        racer = Racer(env, target, apply_fn, horizon=horizon, batch=batch,
                      thresholds=thresholds, signature=meta["signature"])
    r0 = racer.rollouts
    rng = np.random.default_rng(seed)

    def used():
        return racer.rollouts - r0

    # --- Stage A0: reset statistics (env resets only, zero policy rollouts) ---------
    if method == "bci":
        if reference is None:
            reference = reset_reference(env, n=256, seed=seed + 991)
            say("reference: obs statistics of 256 env resets (no policy rollouts)")
        k1 = min(n_first, max(8, (budget - refine_reserve) // 2))
        hyps = stage_a(iface, in_dim, pmu, psd, reference, top_k=k1)
        say(f"stage A: {len(hyps)} hypotheses (best cost {hyps[0].cost:.3f})")
    else:
        k1 = min(n_first, max(8, (budget - refine_reserve) // 2))
        hyps = random_hypotheses(iface, in_dim, k1, rng)
        say(f"random: {len(hyps)} uniform hypotheses")
    if prior_contract is not None:
        hyps = [_hyp_from_contract(prior_contract, iface, "prior")] + hyps
    hyps = _dedupe(hyps)

    pool = {}
    all_h = []

    def run(cands, n, rseed):
        res = racer.race([h.contract(iface) for h in cands], n=n, seed=rseed,
                         weights=weights)
        for h, r in zip(cands, res):
            key = id(h)
            if key not in pool:
                all_h.append(h)
            pool[key] = _merge(pool.get(key), r)

    # Round 1: every Stage A hypothesis once.
    run(hyps, 1, seed + 1)
    cands = hyps
    rnd, n_per, reest = 1, 2, 0
    search_budget = budget - refine_reserve
    while used() < search_budget:
        ranked = sorted(all_h, key=lambda h: score(pool[id(h)]), reverse=True)
        top = pool[id(ranked[0])]
        say(f"round {rnd}: {len(cands)} hyps; best {top['success']}/{top['n']} ok, "
            f"{top['falls']}/{top['n']} falls; used {used()}/{budget}")
        rnd += 1
        fresh = []
        if method == "bci" and reest < 3:
            ref2 = closed_loop_reference([pool[id(h)] for h in ranked[:8]], warm=reference)
            if ref2 is not None:
                reest += 1
                fresh = stage_a(iface, in_dim, pmu, psd, ref2, top_k=16)
                for h in fresh:
                    h.origin = f"re-estimated#{reest}"
                say(f"  closed-loop re-estimation #{reest}: {len(fresh)} hypotheses")
            # per-hypothesis EM: each leading hypothesis re-fits its own scales to the
            # closed-loop statistics of its own rollouts
            kids = [k for h in ranked[:8]
                    for k in [self_reestimate(h, pool[id(h)], iface, pmu, psd)] if k]
            fresh = fresh + kids
            if kids:
                say(f"  self re-estimation: {len(kids)} children")
        elif method == "random" and rnd <= 4:
            fresh = random_hypotheses(iface, in_dim, 16, rng)
        keep = 16 if rnd <= 2 else (8 if rnd <= 4 else (4 if rnd <= 5 else 2))
        cands = _dedupe(ranked[:keep] + fresh)
        if not fresh and len(cands) <= 2:
            break
        n_eff = min(n_per, max(1, (search_budget - used()) // max(len(cands), 1)))
        if (search_budget - used()) < len(cands):
            break
        run(cands, n_eff, seed + 100 + rnd)
        n_per = min(2 * n_per, 16)

    # Closed-loop scale refinement: local racing in log-scale around the leader.
    ranked = sorted(all_h, key=lambda h: score(pool[id(h)]), reverse=True)
    leader = ranked[0]
    for factors in ((0.5, 2.0), (2 ** -0.5, 2 ** 0.5)):
        if budget - used() < refine_reserve // 2 + 2 * n_final:
            break
        variants = _scale_variants(leader, factors)
        run([leader] + variants, 4, seed + 5000 + len(pool))
        cur = _loss(pool[id(leader)], horizon)
        better = [v for v in variants if _loss(pool[id(v)], horizon) < cur - 0.02]
        if not better:
            continue
        # combine every coordinate that improved on its own, then check the combination
        combo = _combine(leader, better)
        run([leader, combo], 8, seed + 6000 + len(pool))
        best_v = min(better + [combo], key=lambda v: _loss(pool[id(v)], horizon))
        if _loss(pool[id(best_v)], horizon) < _loss(pool[id(leader)], horizon):
            say(f"  refined scales ({factors[0]:.2f}/{factors[1]:.2f} steps): "
                f"loss {cur:.3f} -> {_loss(pool[id(best_v)], horizon):.3f}")
            best_v.origin = "refined"
            leader = best_v

    # Final: the best and the best DIFFERENT structure, topped up to n_final each.
    ranked = sorted(all_h, key=lambda h: score(pool[id(h)]), reverse=True)
    best = leader if pool[id(leader)]["n"] >= 4 else ranked[0]
    runner = next((h for h in ranked[1:] if h.key() != best.key()), None)
    finals = [h for h in (best, runner) if h is not None]
    need = max(0, n_final - min(pool[id(h)]["n"] for h in finals))
    if need:
        run(finals, need, seed + 7777)
    rb = pool[id(best)]
    ev = {"best": _summ(rb)}
    ub = cp_upper(rb["falls"], rb["n"])
    ev["best"]["fall_cp_upper"] = ub
    sep = True
    if runner is not None:
        rr = pool[id(runner)]
        ev["runner_up"] = _summ(rr)
        lo_b = cp_lower(rb["success"], rb["n"])
        hi_r = cp_upper(rr["success"], rr["n"])
        ev["best"]["success_cp_lower"] = lo_b
        ev["runner_up"]["success_cp_upper"] = hi_r
        sep = lo_b > hi_r
    if ub <= ACCEPT_FALL_UB and sep:
        status = "accepted"
    elif ub <= ACCEPT_FALL_UB:
        status = "ambiguous"
    else:
        status = "failed"
    say(f"decision: {status} (fall CP upper {ub:.3f} on {rb['n']} rollouts"
        + (f"; success lower {ev['best'].get('success_cp_lower', 0):.2f} vs runner-up "
           f"upper {ev['runner_up']['success_cp_upper']:.2f})" if runner is not None else ")"))
    return BCIResult(status=status, contract=best.contract(iface), hypothesis=best,
                     runner_up=runner, evidence=ev, rollouts=used(),
                     seconds=time.monotonic() - t0, log=log)


def _loss(r, horizon):
    """Mean per-rollout racing loss (lower = better): command-tracking error over the
    steps BEFORE any fall (clipped at 3; 3 if it fell within 20 steps) plus 0.5 per fall.

    Tracking is measured pre-fall on purpose. A hypothesis with the right structure but
    not-yet-right scales walks, tracks, and sometimes falls late; a wrong one with a tiny
    action scale stands still and never falls. Charging the post-fall steps (or a large
    fall penalty) ranks the stander first, which is the one attractor racing must avoid.
    Acceptance does not use this loss: it is decided on falls and success counts."""
    ttf = np.asarray(r["ttf"], float)
    fell = ttf < horizon
    trk = np.minimum(np.asarray(r.get("trk_pre", r["tracking"]), float), 3.0)
    return float(np.mean(trk + 0.5 * fell))


def _pre_fall_tracking(traj, sig, min_steps=20):
    """Mean velocity-tracking error over the steps before the first fall."""
    ttf = int(sig["time_to_failure"])
    if ttf < min_steps:
        return 3.0
    if not all(k in traj for k in ("lin_vel", "ang_vel", "command")):
        return float(sig["tracking_error"])
    from breakpoint_mjx.failure import _tracking_err_series

    return float(np.mean(_tracking_err_series(traj)[:ttf]))


def _scaled(v, f):
    a = np.asarray(v, float) * f
    return float(a) if a.ndim == 0 else a.tolist()


def _scale_variants(h, factors):
    """One-coordinate moves of ``h`` in log-scale: each non-gravity term's scale and the
    action scale times each factor. Gravity's scale is pinned by its mean exactly."""
    out = []
    for t in h.terms:
        if t in ("last_action", "projected_gravity"):
            continue
        for f in factors:
            sc = dict(h.scales)
            sc[t] = _scaled(sc.get(t, 1.0), f)
            out.append(Hypothesis(terms=h.terms, scales=sc, order=h.order,
                                  action_scale=h.action_scale, cost=h.cost,
                                  origin=f"scale {t} x{f:.2f}"))
    for f in factors:
        out.append(Hypothesis(terms=h.terms, scales=dict(h.scales), order=h.order,
                              action_scale=h.action_scale * f, cost=h.cost,
                              origin=f"action x{f:.2f}"))
    return out


def self_reestimate(h, r, iface, pol_mean, pol_std, min_steps=150):
    """EM step for ONE hypothesis: refit its dynamic-term scales and its action scale to
    the pre-fall closed-loop statistics of its own rollouts (``scale = policy std / env
    std`` per term; the action scale from the last_action slot). Gravity and the command
    keep their exact reset-based scales. Returns a child ``Hypothesis`` or None."""
    src, act = r.get("src"), r.get("act")
    if src is None or len(src) == 0:
        return None
    n, hh = src.shape[:2]
    lim = np.clip(r.get("src_ttf", r["ttf"]), 0, hh).reshape(n, 1)
    mask = np.arange(hh)[None, :] < lim
    mask[:, :10] = False
    if mask.sum() < min_steps:
        return None
    mu, sd, _ = masked_moments(src, mask)
    _, asd, _ = masked_moments(act, mask)
    perm = perm_of(h.order, iface)
    sc = dict(h.scales)
    a_s = h.action_scale
    st = 0
    for t in h.terms:
        s0, d = iface.layout[t]
        ch = np.arange(s0, s0 + d)
        if t in JOINT_TERMS:
            ch = ch[perm]
        psd_t = np.maximum(pol_std[st:st + d], STD_FLOOR)
        if t == "last_action":
            ratio = float(np.exp(np.mean(np.log(np.maximum(asd[perm], STD_FLOOR))
                                         - np.log(psd_t))))
            a_s = ratio * iface.action_scale
        elif t not in EXOGENOUS:
            c = float(np.exp(np.mean(np.log(psd_t) - np.log(np.maximum(sd[ch], STD_FLOOR)))))
            old = np.asarray(sc.get(t, 1.0), float)
            sc[t] = _scaled(np.sign(old) if old.ndim else 1.0, c) if old.ndim else c
        st += d
    return Hypothesis(terms=h.terms, scales=sc, order=h.order, action_scale=a_s,
                      cost=h.cost, origin="self re-estimated")


def _combine(h, better):
    """Apply every improving one-coordinate move of ``h`` at once (last move per coordinate
    wins when two factors of the same coordinate both improved)."""
    sc, a = dict(h.scales), h.action_scale
    for v in better:
        if v.action_scale != h.action_scale:
            a = v.action_scale
        for t in h.terms:
            if not np.allclose(np.asarray(v.scales.get(t, 1.0), float),
                               np.asarray(h.scales.get(t, 1.0), float)):
                sc[t] = v.scales[t]
    return Hypothesis(terms=h.terms, scales=sc, order=h.order, action_scale=a, cost=h.cost,
                      origin="combined")


def _summ(r):
    return {"n": int(r["n"]), "falls": int(r["falls"]), "success": int(r["success"]),
            "mean_ttf": float(np.mean(r["ttf"])),
            "mean_tracking": float(np.mean(r["tracking"]))}


def _merge(prev, r):
    if prev is None:
        return dict(r)
    return {"n": prev["n"] + r["n"], "falls": prev["falls"] + r["falls"], "H": r.get("H"),
            "success": prev["success"] + r["success"],
            "ttf": np.concatenate([prev["ttf"], r["ttf"]]),
            "tracking": np.concatenate([prev["tracking"], r["tracking"]]),
            "trk_pre": np.concatenate([prev["trk_pre"], r["trk_pre"]]),
            "src": np.concatenate([prev["src"], r["src"]])[-16:],
            "act": np.concatenate([prev["act"], r["act"]])[-16:],
            "src_ttf": np.concatenate([prev.get("src_ttf", prev["ttf"]), r["ttf"]])[-16:]}


_RESET_FNS = {}


def reset_reference(env, n=256, seed=0):
    """Per-channel obs statistics over ``n`` env resets: the exact reference for the
    exogenous channels (gravity, the sampled velocity command). No policy is run."""
    import jax

    fn = _RESET_FNS.get(id(env))
    if fn is None:
        fn = _RESET_FNS[id(env)] = jax.jit(jax.vmap(env.reset))
    st = fn(jax.random.split(jax.random.PRNGKey(seed), n))
    s = np.asarray(st.obs["state"], dtype=float)
    mu, sd = s.mean(axis=0), s.std(axis=0)
    return Reference(mean=mu, std=sd, source="env resets", warm_mean=mu, warm_std=sd)


def _hyp_from_contract(c, iface, origin):
    terms = tuple(t["term"] for t in c.terms)
    scales = {t["term"]: t.get("scale", 1.0) for t in c.terms}
    order = tuple(c.policy_joint_names) if c.policy_joint_names else tuple(iface.joint_names)
    a = (c.action or {}).get("scale", iface.action_scale)
    a = float(np.mean(np.asarray(a, float)))
    return Hypothesis(terms=terms, scales=scales, order=order, action_scale=a,
                      cost=float("nan"), origin=origin)


def fold_contract(policy, contract):
    """A policy that EXPECTS ``contract``, built from one that runs on the target's native
    layout (identity contract). Exact, by folding the contract into the weights.

    The native policy computes ``h = ((x_t - mu) / sd) @ W0``. Under the contract its
    input slot ``j`` carries ``x_s[j] = (x_t[e_j] - off_j) * c_j``, so the folded policy
    uses ``mu_s[j] = c_j (mu[e_j] - off_j)``, ``sd_s[j] = |c_j| sd[e_j]`` and
    ``W0_s[j] = sign(c_j) W0[e_j]``. On the output side its head is permuted into the
    contract's joint order and divided by the contract's per-joint action ratio, so that
    driving it through ``contract`` reproduces the native policy's actions exactly. Used
    to make synthetic scrambles of a REAL trained policy (scripts/bench_contract.py) and
    in the unit tests.
    """
    from breakpoint_mjx.adapters.mlp import MLPPolicy, OutputTransform
    from breakpoint_mjx.adapters.obs_spec import ObsSpec

    sp = policy.obs_spec
    d = policy.in_dim
    mu = np.zeros(d) if sp is None or sp.mean is None else np.asarray(sp.mean, float)
    sd = np.ones(d) if sp is None or sp.std is None else np.asarray(sp.std, float)
    e = np.asarray(contract.obs_idx)
    c = np.asarray(contract.obs_scale, float)
    off = np.asarray(contract.obs_offset, float)
    if len(e) != d:
        raise ValueError(f"contract has {len(e)} obs channels, policy takes {d}")
    params = [(np.array(W, float), np.array(b, float)) for W, b in policy.params]
    W0, b0 = params[0]
    params[0] = (W0[e] * np.sign(c)[:, None], b0)
    mu_s = c * (mu[e] - off)
    sd_s = np.abs(c) * sd[e]
    ot = policy.out_transform or OutputTransform()
    a = contract.action_dim
    act_idx = np.asarray(contract.act_idx)
    p2e = np.empty(a, dtype=int)
    p2e[act_idx] = np.arange(a)          # policy joint k drives env actuator p2e[k]
    if np.any(np.asarray(contract.act_offset) != 0):
        raise ValueError("fold_contract supports contracts without action offsets")
    Wl, bl = params[-1]
    md = int(ot.mean_dim) if ot.mean_dim is not None else Wl.shape[1]
    cols = np.arange(Wl.shape[1])
    cols[:md] = cols[:md][p2e]            # head column k <- native column p2e[k]
    params[-1] = (Wl[:, cols], bl[cols])
    ratio = np.asarray(contract.act_scale, float)[p2e]   # per policy joint
    base_s = np.broadcast_to(np.asarray(ot.scale, float), (a,))[p2e]
    base_b = np.broadcast_to(np.asarray(ot.bias, float), (a,))[p2e]
    new_ot = OutputTransform(squash=ot.squash, mean_dim=ot.mean_dim,
                             scale=base_s / ratio, bias=base_b / ratio)
    return MLPPolicy(params=params, activation=policy.activation,
                     obs_spec=ObsSpec(keys=sp.keys if sp is not None else ("state",),
                                      mean=mu_s, std=sd_s),
                     out_transform=new_ot)


def contract_diff(declared, found):
    """Human-readable differences between two ``PolicyContract``s (declared vs found)."""
    rows = []
    dt = [t["term"] for t in (declared.terms if declared else [])]
    ft = [t["term"] for t in found.terms]
    if dt != ft:
        rows.append(f"obs term order: {dt or '(target layout)'} -> {ft}")
    ds = {t["term"]: t.get("scale", 1.0) for t in (declared.terms if declared else [])}
    for t in found.terms:
        a, b = ds.get(t["term"], 1.0), t.get("scale", 1.0)
        if not np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0.1):
            rows.append(f"scale[{t['term']}]: {a} -> {_fmt(b)}")
    dj = list(declared.policy_joint_names) if declared and declared.policy_joint_names else None
    if dj != list(found.policy_joint_names):
        rows.append(f"joint order: {dj or '(target order)'} -> {list(found.policy_joint_names)}")
    da = (declared.action or {}).get("scale") if declared else None
    fa = (found.action or {}).get("scale")
    if da is None or not np.allclose(np.asarray(da, float), np.asarray(fa, float), rtol=0.1):
        rows.append(f"action scale: {da if da is not None else '(target)'} -> {_fmt(fa)}")
    return rows or ["(no difference)"]


def _fmt(v):
    a = np.asarray(v, float).reshape(-1)
    return f"{a[0]:.3g}" if a.size == 1 else "[" + ", ".join(f"{x:.3g}" for x in a) + "]"
