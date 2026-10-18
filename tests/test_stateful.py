"""Stateful policies (breakpoint_mjx/adapters/stateful.py): observation history and
recurrent actors, in numpy, in jax, through the contract, and inside the real rollout."""
from __future__ import annotations

import numpy as np
import pytest

from breakpoint_mjx.adapters.mlp import MLPPolicy
from breakpoint_mjx.adapters.obs_spec import ObsSpec, build_contract
from breakpoint_mjx.adapters.stateful import (HistoryPolicy, RecurrentPolicy, gru_cell,
                                              is_stateful, lstm_cell, parse_history,
                                              recurrent_from_state_dict, stack_history)

rng = np.random.default_rng(0)


def _mlp(in_dim, out_dim, keys=("_history",), hidden=16, activation="tanh"):
    params = [(rng.standard_normal((in_dim, hidden)) * 0.3, rng.standard_normal(hidden) * 0.1),
              (rng.standard_normal((hidden, out_dim)) * 0.3, np.zeros(out_dim))]
    return MLPPolicy(params=params, activation=activation,
                     obs_spec=ObsSpec(keys=keys, expected_dim=in_dim))


# ---------------------------------------------------------------- parsing + layout

def test_parse_history():
    assert parse_history(None) is None
    assert parse_history(1) is None
    assert parse_history({"length": 1}) is None
    assert parse_history(3) == {"length": 3, "layout": "term", "fill": "first",
                                "order": "oldest_first"}
    assert parse_history({"length": 5, "layout": "frame", "fill": "zeros"})["fill"] == "zeros"
    for bad in ({"length": 0}, {"length": 2.5}, {"length": 3, "layout": "x"},
                {"length": 3, "fill": "x"}, {"length": 3, "order": "x"}):
        with pytest.raises(ValueError):
            parse_history(bad)


def test_stack_history_layouts():
    # 3 frames, two terms of widths 2 and 1: frame t = [a_t, a_t', b_t]
    buf = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)   # row 0 oldest
    assert stack_history(buf, "frame", "oldest_first", (2, 1), np).tolist() == \
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert stack_history(buf, "frame", "newest_first", (2, 1), np).tolist() == \
        [7, 8, 9, 4, 5, 6, 1, 2, 3]
    # Isaac Lab flatten_history_dim: term a's 3 frames, then term b's 3 frames
    assert stack_history(buf, "term", "oldest_first", (2, 1), np).tolist() == \
        [1, 2, 4, 5, 7, 8, 3, 6, 9]


# ---------------------------------------------------------------- history policy

def _frame_obs(t, d=4):
    return {"state": np.arange(d, dtype=float) + 10.0 * t}


def test_history_policy_numpy_matches_manual_stacking():
    H, D = 3, 4
    head = _mlp(H * D, 2)
    hp = HistoryPolicy(frame_spec=ObsSpec(expected_dim=D), mlp=head, length=H,
                       layout="frame", fill="first")
    seq = [_frame_obs(t) for t in range(5)]
    got = hp.rollout_np(seq)
    frames = [seq[0]["state"]] * H                 # filled with the first frame
    want = []
    for o in seq:
        frames = frames[1:] + [o["state"]]
        want.append(head.act_np({"_history": np.concatenate(frames)}))
    np.testing.assert_allclose(got, np.stack(want), rtol=1e-12)


def test_history_fill_zeros_differs_from_first():
    H, D = 3, 4
    head = _mlp(H * D, 2)
    a = HistoryPolicy(ObsSpec(expected_dim=D), head, H, "frame", "first")
    b = HistoryPolicy(ObsSpec(expected_dim=D), head, H, "frame", "zeros")
    seq = [_frame_obs(t) for t in range(4)]
    ra, rb = a.rollout_np(seq), b.rollout_np(seq)
    assert not np.allclose(ra[0], rb[0])           # step 0: history differs
    np.testing.assert_allclose(ra[-1], rb[-1])     # after H steps it is the same


def test_history_policy_rejects_wrong_widths():
    with pytest.raises(ValueError, match="first layer takes"):
        HistoryPolicy(ObsSpec(expected_dim=4), _mlp(10, 2), 3, "frame")
    with pytest.raises(ValueError, match="per-term widths"):
        HistoryPolicy(ObsSpec(expected_dim=4), _mlp(12, 2), 3, "term", term_dims=(1, 1))


def test_history_policy_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    H, D = 4, 4
    hp = HistoryPolicy(ObsSpec(expected_dim=D), _mlp(H * D, 3), H, "term",
                       term_dims=(3, 1))
    seq = [_frame_obs(t) for t in range(6)]
    want = hp.rollout_np(seq)
    step = jax.jit(lambda o, c: hp(o, None, c))
    carry = hp.init_carry({"state": jnp.asarray(seq[0]["state"])})
    got = []
    for o in seq:
        a, carry = step({"state": jnp.asarray(o["state"])}, carry)
        got.append(np.asarray(a))
    np.testing.assert_allclose(np.stack(got), want, rtol=1e-5, atol=1e-6)
    assert is_stateful(hp)
    with pytest.raises(TypeError, match="stateful"):
        hp({"state": jnp.zeros(D)}, None)


# ---------------------------------------------------------------- contract

QUAD_CONTRACT = {"robot": "go1", "obs": [{"term": "base_ang_vel"},
                                         {"term": "projected_gravity"}]}


def test_contract_history_round_trips_and_records_term_widths():
    c = build_contract({**QUAD_CONTRACT, "history": {"length": 5}}, robot="go1")
    assert c.history == {"length": 5, "layout": "term", "fill": "first",
                         "order": "oldest_first"}
    assert c.term_dims == (3, 3)
    assert c.to_dict()["history"]["length"] == 5
    assert build_contract(QUAD_CONTRACT, robot="go1").history is None


def test_contract_history_builds_a_history_policy():
    from breakpoint_mjx.adapters.importers import apply_contract

    c = build_contract({**QUAD_CONTRACT, "history": {"length": 5}}, robot="go1")
    base = _mlp(30, 12, keys=("state",))
    hp = apply_contract(base, c)
    assert isinstance(hp, HistoryPolicy)
    assert hp.length == 5 and hp.frame_dim == 6 and hp.in_dim == 30
    # without the history block the same checkpoint is refused, and the error says why
    with pytest.raises(ValueError, match="history: block"):
        apply_contract(base, build_contract(QUAD_CONTRACT, robot="go1"))


def test_contract_per_frame_normalization_is_tiled_into_the_stacked_layout():
    from breakpoint_mjx.adapters.importers import apply_contract

    mean = np.arange(6, dtype=float)
    c = build_contract({**QUAD_CONTRACT, "history": {"length": 2},
                        "normalization": {"mean": mean.tolist(), "std": [1.0] * 6}},
                       robot="go1")
    hp = apply_contract(_mlp(12, 12, keys=("state",)), c)
    # term layout: term 0 (3 wide) x 2 frames, then term 1 x 2 frames
    assert hp.mlp.obs_spec.mean.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]


# ---------------------------------------------------------------- recurrent

def _ref_lstm(x, h, c, w_ih, w_hh, b_ih, b_hh):
    """torch.nn.LSTM's documented equations, written out gate by gate."""
    n = h.shape[0]
    sig = lambda v: 1 / (1 + np.exp(-v))   # noqa: E731
    W_ii, W_if, W_ig, W_io = (w_ih[k * n:(k + 1) * n] for k in range(4))
    W_hi, W_hf, W_hg, W_ho = (w_hh[k * n:(k + 1) * n] for k in range(4))
    bi = b_ih[:n] + b_hh[:n]
    bf = b_ih[n:2 * n] + b_hh[n:2 * n]
    bg = b_ih[2 * n:3 * n] + b_hh[2 * n:3 * n]
    bo = b_ih[3 * n:] + b_hh[3 * n:]
    i = sig(W_ii @ x + W_hi @ h + bi)
    f = sig(W_if @ x + W_hf @ h + bf)
    g = np.tanh(W_ig @ x + W_hg @ h + bg)
    o = sig(W_io @ x + W_ho @ h + bo)
    c2 = f * c + i * g
    return o * np.tanh(c2), c2


def _ref_gru(x, h, w_ih, w_hh, b_ih, b_hh):
    n = h.shape[0]
    sig = lambda v: 1 / (1 + np.exp(-v))   # noqa: E731
    r = sig(w_ih[:n] @ x + b_ih[:n] + w_hh[:n] @ h + b_hh[:n])
    z = sig(w_ih[n:2 * n] @ x + b_ih[n:2 * n] + w_hh[n:2 * n] @ h + b_hh[n:2 * n])
    nn = np.tanh(w_ih[2 * n:] @ x + b_ih[2 * n:] + r * (w_hh[2 * n:] @ h + b_hh[2 * n:]))
    return (1 - z) * nn + z * h


def test_lstm_and_gru_cells_match_the_torch_equations():
    I, H = 5, 4
    x, h, c = rng.standard_normal(I), rng.standard_normal(H), rng.standard_normal(H)
    w = [rng.standard_normal(s) for s in ((4 * H, I), (4 * H, H), (4 * H,), (4 * H,))]
    h2, c2 = lstm_cell(x, h, c, *w, np)
    rh, rc = _ref_lstm(x, h, c, *w)
    np.testing.assert_allclose(h2, rh, rtol=1e-12)
    np.testing.assert_allclose(c2, rc, rtol=1e-12)
    w = [rng.standard_normal(s) for s in ((3 * H, I), (3 * H, H), (3 * H,), (3 * H,))]
    np.testing.assert_allclose(gru_cell(x, h, *w, np), _ref_gru(x, h, *w), rtol=1e-12)


def _rsl_state_dict(kind="lstm", obs=6, hidden=8, layers=2, act=3):
    g = 4 if kind == "lstm" else 3
    sd = {}
    for k in range(layers):
        sd[f"memory_a.rnn.weight_ih_l{k}"] = rng.standard_normal(
            (g * hidden, obs if k == 0 else hidden)).astype(np.float32) * 0.3
        sd[f"memory_a.rnn.weight_hh_l{k}"] = rng.standard_normal(
            (g * hidden, hidden)).astype(np.float32) * 0.3
        sd[f"memory_a.rnn.bias_ih_l{k}"] = rng.standard_normal(g * hidden).astype(np.float32)
        sd[f"memory_a.rnn.bias_hh_l{k}"] = rng.standard_normal(g * hidden).astype(np.float32)
    sd["actor.0.weight"] = rng.standard_normal((16, hidden)).astype(np.float32) * 0.3
    sd["actor.0.bias"] = np.zeros(16, np.float32)
    sd["actor.2.weight"] = rng.standard_normal((act, 16)).astype(np.float32) * 0.3
    sd["actor.2.bias"] = np.zeros(act, np.float32)
    sd["critic.0.weight"] = np.zeros((4, 4), np.float32)      # ignored
    return sd


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_recurrent_from_rsl_rl_state_dict_has_memory(kind):
    sd = _rsl_state_dict(kind)
    pol = recurrent_from_state_dict(sd, obs_spec=ObsSpec(expected_dim=6))
    assert pol.kind == kind and pol.hidden == 8 and len(pol.layers) == 2
    assert pol.in_dim == 6 and pol.action_size == 3
    same = [{"state": np.ones(6)}] * 4
    out = pol.rollout_np(same)
    assert not np.allclose(out[0], out[3])      # same input, different action: memory


def test_recurrent_state_dict_errors():
    with pytest.raises(ValueError, match="no recurrent weights"):
        recurrent_from_state_dict({"actor.0.weight": np.zeros((2, 2))},
                                  obs_spec=ObsSpec())
    with pytest.raises(ValueError, match="are a lstm"):
        recurrent_from_state_dict(_rsl_state_dict("lstm"), kind="gru",
                                  obs_spec=ObsSpec(expected_dim=6))


def test_recurrent_jax_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    pol = recurrent_from_state_dict(_rsl_state_dict("lstm"), obs_spec=ObsSpec(expected_dim=6))
    seq = [{"state": rng.standard_normal(6)} for _ in range(5)]
    want = pol.rollout_np(seq)
    step = jax.jit(lambda o, c: pol(o, None, c))
    carry = pol.init_carry(None)
    got = []
    for o in seq:
        a, carry = step({"state": jnp.asarray(o["state"], dtype=jnp.float32)}, carry)
        got.append(np.asarray(a))
    np.testing.assert_allclose(np.stack(got), want, rtol=1e-4, atol=1e-5)


def test_build_policy_detects_a_recurrent_npz(tmp_path):
    from breakpoint_mjx.adapters import build_policy

    p = tmp_path / "rnn.npz"
    np.savez(p, **_rsl_state_dict("gru"))
    with pytest.warns(UserWarning):                       # no norm / action transform set
        pol = build_policy(f"params:{p}", expected_dim=6)
    assert isinstance(pol, RecurrentPolicy) and pol.kind == "gru"
    assert pol.policy_fn is pol


# ---------------------------------------------------------------- inside the rollout

def test_history_policy_reproduces_the_stateless_policy_in_the_real_rollout():
    """A history policy whose first layer reads ONLY the newest frame is mathematically the
    stateless policy. Rolled out through the real evaluator (MJX, Go1, perturbed worlds,
    vmap + scan), it must give the same trajectories: this pins the carry plumbing, the
    frame order and the per-env state end to end."""
    pytest.importorskip("mujoco_playground")
    import jax
    import jax.numpy as jnp

    from breakpoint_mjx.adapters.importers import from_brax_pkl
    from breakpoint_mjx.check_policy import _load_env
    from breakpoint_mjx.config import load_theta_space
    from breakpoint_mjx.evaluator import make_evaluator
    from breakpoint_mjx.paths import config_path
    from breakpoint_mjx.targets import get_target

    target = get_target("go1")
    env = _load_env(target, None)
    space = load_theta_space(config_path(target.theta_space_path))
    base = from_brax_pkl("breakpoint_mjx/assets/demo/go1_policy.pkl")
    H, D = 3, base.in_dim
    (W0, b0), rest = base.params[0], base.params[1:]
    W_hist = np.zeros((H * D, W0.shape[1]), dtype=W0.dtype)
    W_hist[(H - 1) * D:] = W0                                 # frame layout: newest last
    head = MLPPolicy(params=[(W_hist, b0)] + list(rest), activation=base.activation,
                     obs_spec=ObsSpec(keys=("_history",),
                                      mean=np.tile(base.obs_spec.mean, H),
                                      std=np.tile(base.obs_spec.std, H),
                                      expected_dim=H * D),
                     out_transform=base.out_transform)
    hp = HistoryPolicy(frame_spec=ObsSpec(keys=base.obs_spec.keys, expected_dim=D),
                       mlp=head, length=H, layout="frame")

    consts = target.resolve_constants(env)
    nom = np.asarray(space.nominal_vector(), np.float32)
    X = np.tile(nom, (4, 1))
    names = space.names()
    X[1, names.index("obs_noise_scale")] = 0.03            # per-step noise
    X[2, names.index("latency_steps")] = 2                 # latency buffer
    X[3, names.index("obs_bias_scale")] = 0.04             # per-episode bias in the carry
    key = jax.random.PRNGKey(0)
    model0 = env._mjx_model
    try:
        a = make_evaluator(env, base.policy_fn, space, 40, target, consts)(jnp.asarray(X), key)
        env._mjx_model = model0
        b = make_evaluator(env, hp, space, 40, target, consts)(jnp.asarray(X), key)
    finally:
        env._mjx_model = model0
    for k in a:
        np.testing.assert_allclose(np.asarray(a[k]), np.asarray(b[k]), rtol=1e-4,
                                   atol=1e-4, err_msg=k)
