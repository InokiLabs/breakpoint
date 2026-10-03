"""Behavioral Contract Identification (adapters/contract_id.py).

Pure-numpy parts (symmetry group, segmentations, Clopper-Pearson, folding, Stage A on
synthetic statistics, the contract diff) always run. The racing parts run on a tiny
synthetic jax "robot" (``ToyEnv``): its obs follows Go1's 48-channel layout, and it
"falls" when the applied action strays from what a hidden native linear controller would
do. That makes identification CPU-light (seconds) while exercising the real racer,
acceptance rule and refusal path. The real-policy numbers are in
scripts/bench_contract.py / docs/contract_results.json.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from breakpoint_mjx.adapters.contract_id import (
    Hypothesis,
    Reference,
    _type_consistency,
    contract_diff,
    cp_lower,
    cp_upper,
    fold_contract,
    perm_of,
    policy_core,
    segmentations,
    stage_a,
    symmetry_orders,
)
from breakpoint_mjx.adapters.importers import (
    ISAACLAB_QUADRUPED_ORDER,
    LEGGED_GYM_QUADRUPED_ORDER,
    apply_contract,
    from_params,
)
from breakpoint_mjx.adapters.mlp import OutputTransform
from breakpoint_mjx.adapters.obs_spec import GO1_INTERFACE as G
from breakpoint_mjx.adapters.obs_spec import ObsSpec, build_contract

TERMS = list(G.layout)


# --------------------------------------------------------------------------- #
# numpy-only
# --------------------------------------------------------------------------- #
def test_symmetry_group_is_small_and_contains_the_framework_orders():
    orders = symmetry_orders(G)
    assert len(orders) == 48                           # 2 layouts x 4! leg orders
    as_t = {tuple(o) for o in orders}
    assert tuple(G.joint_names) in as_t                # Playground / menagerie
    assert ISAACLAB_QUADRUPED_ORDER in as_t            # Isaac Lab joint-type-major
    assert LEGGED_GYM_QUADRUPED_ORDER in as_t          # legged_gym leg-major FL,FR,RL,RR
    for o in orders:                                   # every order is a permutation
        assert sorted(perm_of(o, G)) == list(range(12))


def test_segmentations_enumerate_term_orders():
    segs = segmentations(G, 48)
    assert len(segs) == 35                             # C(7,3) interleavings of 3s and 12s
    assert all(sum(seq) == 48 for seq, _ in segs)
    subs45 = {sub for _, sub in segmentations(G, 45)}  # one 3-dim term dropped
    assert len(subs45) == 4 and all(len(s) == 6 for s in subs45)


def test_clopper_pearson_bounds():
    assert cp_upper(0, 32) == pytest.approx(1 - 0.05 ** (1 / 32), abs=1e-6)
    assert cp_upper(0, 16) < 0.2 < cp_upper(1, 16)     # the acceptance arithmetic
    assert cp_lower(0, 10) == 0.0
    assert 0.0 < cp_lower(30, 32) < 30 / 32 < cp_upper(30, 32) < 1.0


def _linear_native(rng, in_dim=48, act=12):
    W = rng.normal(0, 0.3, (in_dim, act))
    b = rng.normal(0, 0.1, act)
    mean = rng.normal(0, 0.5, in_dim)
    std = rng.uniform(0.2, 2.0, in_dim)
    return from_params([(W, b)], activation="identity",
                       obs_spec=ObsSpec(mean=mean, std=std),
                       out_transform=OutputTransform(squash="tanh"))


def _scramble(order_idx=17, scales=None, action=1.0):
    terms = ("commands", "joint_vel", "projected_gravity", "last_action", "base_ang_vel",
             "joint_pos_rel", "base_lin_vel")
    return Hypothesis(terms=terms, scales=scales or {"commands": 2.0, "base_ang_vel": 0.5},
                      order=tuple(symmetry_orders(G)[order_idx]), action_scale=action)


def test_fold_contract_is_exact_through_the_contract():
    rng = np.random.default_rng(0)
    native = _linear_native(rng)
    c = _scramble().contract(G)
    folded = fold_contract(native, c)
    driven = apply_contract(folded, c)
    for _ in range(5):
        o = {"state": rng.normal(size=48)}
        np.testing.assert_allclose(driven.act_np(o), native.act_np(o), atol=1e-12)
    # and the folded policy's core is exactly what the racer runs
    apply, w, meta = policy_core(folded)
    o = rng.normal(size=48)
    x = (o[c.obs_idx] - c.obs_offset) * c.obs_scale
    y = apply(w, x, np)
    np.testing.assert_allclose(y[c.act_idx] * c.act_scale + c.act_offset,
                               native.act_np({"state": o}), atol=1e-5)


def test_type_consistency_prefers_the_true_layout():
    types = [n.split("_")[1] for n in G.joint_names]    # hip/thigh/calf
    by_type = {"hip": 0.1, "thigh": 0.3, "calf": 0.9}
    sd = np.array([by_type[t] for t in types])
    mu = np.array([0.0 if t != "calf" else 0.5 for t in types])
    right = _type_consistency(mu, sd, types)
    wrong = _type_consistency(mu, sd, [types[i] for i in perm_of(ISAACLAB_QUADRUPED_ORDER, G)])
    assert right < 0.05 < wrong


def _synthetic_policy_stats(truth, rng):
    """Policy-side input stats a checkpoint would carry for contract ``truth``, from
    synthetic 'training' statistics in the target layout."""
    types = [n.split("_")[1] for n in G.joint_names]
    f = np.array([{"hip": 0.5, "thigh": 1.0, "calf": 2.0}[t] for t in types])
    mean = np.zeros(48)
    std = np.ones(48)
    std[0:3] = [0.5, 0.25, 0.15]
    std[3:6] = [0.6, 0.5, 0.7]
    mean[6:9], std[6:9] = [0, 0, -1.0], [0.03, 0.03, 0.03]
    std[9:21], mean[9:21] = 0.1 * f, 0.05 * (f == 2.0)
    std[21:33] = 2.5 * f
    std[33:45], mean[33:45] = 0.3 * f, 0.4 * (f == 2.0)
    std[45:48] = [0.85, 0.45, 0.65]
    c = truth.contract(G)
    pm = c.obs_scale * (mean[c.obs_idx] - c.obs_offset)
    ps = np.abs(c.obs_scale) * std[c.obs_idx]
    return pm, ps, Reference(mean=mean, std=std, source="env resets",
                             warm_mean=mean, warm_std=std)


def test_stage_a_keeps_the_true_structure_in_the_racing_set():
    rng = np.random.default_rng(1)
    truth = _scramble(scales={"projected_gravity": 2.0})
    pm, ps, ref = _synthetic_policy_stats(truth, rng)
    hyps = stage_a(G, 48, pm, ps, ref, top_k=288)
    keys = [h.key() for h in hyps]
    assert truth.key() in keys
    h = hyps[keys.index(truth.key())]
    assert h.scales["projected_gravity"] == pytest.approx(2.0, rel=1e-6)  # pinned by the mean
    assert h.scales["commands"] == pytest.approx(1.0, rel=1e-6)            # exact reference
    # gravity and commands never land in the wrong slot among the ranked set
    for x in hyps[:50]:
        assert x.terms.index("projected_gravity") == truth.terms.index("projected_gravity")


def test_contract_diff_names_what_changed():
    declared = build_contract({"robot": "go1", "obs": TERMS}, robot=G)
    found = _scramble().contract(G)
    rows = "\n".join(contract_diff(declared, found))
    assert "obs term order" in rows and "joint order" in rows and "scale[commands]" in rows
    assert contract_diff(found, found) == ["(no difference)"]


# --------------------------------------------------------------------------- #
# racing on a synthetic jax robot
# --------------------------------------------------------------------------- #
jax = pytest.importorskip("jax")
jnp = jax.numpy

TYPES = np.array([{"hip": 0.5, "thigh": 1.0, "calf": 2.0}[n.split("_")[1]]
                  for n in G.joint_names])


class ToyState(NamedTuple):
    obs: dict
    x: jax.Array       # latent AR(1) signals for the 45 non-action channels
    cmd: jax.Array
    last: jax.Array
    err: jax.Array
    key: jax.Array


class ToyEnv:
    """Obs in Go1's layout; the 'robot' falls when the applied action strays from a hidden
    native controller ``a* = tanh(W x_native + b)``."""

    action_size = 12

    def __init__(self, W, b):
        self.W, self.b = jnp.asarray(W), jnp.asarray(b)
        std = np.ones(48)
        mean = np.zeros(48)
        std[0:3], std[3:6] = [0.5, 0.25, 0.15], [0.6, 0.5, 0.7]
        mean[6:9], std[6:9] = [0, 0, -1.0], [0.03, 0.03, 0.03]
        std[9:21], mean[9:21] = 0.1 * TYPES, 0.05 * (TYPES == 2.0)
        std[21:33] = 2.5 * TYPES
        self.mean, self.std = jnp.asarray(mean), jnp.asarray(std)
        self.cmd_std = jnp.asarray([0.85, 0.45, 0.65])

    def _obs(self, x, cmd, last):
        base = self.mean + self.std * jnp.concatenate([x, jnp.zeros(15)])
        s = jnp.concatenate([base[:33], last, cmd])
        return {"state": s}

    def reset(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.normal(k1, (33,))
        cmd = self.cmd_std * jax.random.normal(k2, (3,))
        last = jnp.zeros(12)
        return ToyState(self._obs(x, cmd, last), x, cmd, last, jnp.zeros(()), k3)

    def ideal(self, obs):
        return jnp.tanh(obs["state"] @ self.W + self.b)

    def step(self, st, a):
        star = self.ideal(st.obs)
        err = jnp.mean(jnp.abs(a - star)) / (jnp.mean(jnp.abs(star)) + 1e-6)
        k1, k2 = jax.random.split(st.key)
        x = 0.9 * st.x + np.sqrt(1 - 0.81) * jax.random.normal(k1, (33,))
        return ToyState(self._obs(x, st.cmd, a), x, st.cmd, a, err, k2)


class _Obj:
    @staticmethod
    def summarize(st, consts):
        return {"err": st.err}

    @staticmethod
    def component_signals(traj, thr):
        e = np.asarray(traj["err"])
        bad = e > 0.5
        fall = float(bad.any())
        return {"fall": fall, "time_to_failure": float(np.argmax(bad)) if fall else float(len(e)),
                "tracking_error": float(e.mean())}


class ToyTarget:
    name = "go1"
    objective = _Obj()

    @staticmethod
    def resolve_constants(env):
        return None


def _native_and_env(seed=0, zero=False):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (48, 12)) / np.sqrt(48) * TYPES[None, :]
    W[33:45] *= 0.1                              # weak dependence on the last action
    b = rng.normal(0, 0.1, 12)
    if zero:
        W[:] = 0.0
    env = ToyEnv(W, b)
    # the "checkpoint" normalization: the native obs statistics under its own contract
    mean = np.asarray(env.mean).copy()
    std = np.asarray(env.std).copy()
    mean[45:48], std[45:48] = 0.0, np.asarray(env.cmd_std)
    std[33:45] = 0.4 * TYPES
    native = from_params([(W * std[:, None], b + mean @ W)], activation="identity",
                         obs_spec=ObsSpec(mean=mean, std=std),
                         out_transform=OutputTransform(squash="tanh"))
    return native, env


THR = {"tracking_err_max": 0.3}


def test_native_contract_walks_on_the_toy():
    from breakpoint_mjx.adapters.contract_id import Racer

    native, env = _native_and_env()
    apply, w, meta = policy_core(native)
    racer = Racer(env, ToyTarget(), apply, horizon=30, batch=32, thresholds=THR,
                  signature=meta["signature"])
    ident = build_contract({"robot": "go1", "obs": TERMS}, robot=G)
    wrong = _scramble().contract(G)
    r_ok, r_bad = racer.race([ident, wrong], n=8, seed=0, weights=w)
    assert r_ok["falls"] == 0 and r_ok["success"] == 8
    assert r_bad["falls"] == 8


def test_bci_identifies_a_scrambled_toy_policy():
    from breakpoint_mjx.adapters.contract_id import identify_contract

    native, env = _native_and_env()
    truth = Hypothesis(terms=("base_ang_vel", "projected_gravity", "commands",
                              "joint_pos_rel", "joint_vel", "last_action", "base_lin_vel"),
                       scales={"joint_vel": 0.5}, order=ISAACLAB_QUADRUPED_ORDER,
                       action_scale=G.action_scale)
    policy = fold_contract(native, truth.contract(G))
    res = identify_contract(policy, ToyTarget(), env, budget=700, horizon=30, batch=64,
                            thresholds=THR, seed=3)
    assert res.status == "accepted", res.log
    assert res.hypothesis.key() == truth.key(), res.log
    assert res.evidence["best"]["falls"] == 0


def test_bci_refuses_when_the_evidence_cannot_separate_contracts():
    """A policy that ignores its inputs behaves identically under every contract: the
    rule must say 'ambiguous', not pick one."""
    from breakpoint_mjx.adapters.contract_id import identify_contract

    native, env = _native_and_env(zero=True)
    res = identify_contract(native, ToyTarget(), env, budget=400, horizon=20, batch=64,
                            thresholds=THR, seed=1)
    assert res.status == "ambiguous", res.log
    assert res.runner_up is not None and res.runner_up.key() != res.hypothesis.key()
