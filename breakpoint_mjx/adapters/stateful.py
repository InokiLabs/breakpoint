"""Stateful policies: observation history and recurrent (LSTM / GRU) actors.

Most deployed legged policies are not a bare MLP on one observation. Isaac Lab and
legged_gym policies are commonly trained on a STACK of the last H observations, and
RSL-RL's ``ActorCriticRecurrent`` puts an LSTM or GRU in front of the MLP. Both need
memory that survives from one control step to the next, per environment.

The shared rollout (breakpoint_mjx/rollout.py) supports that through one protocol:

    policy.init_carry(obs)            -> carry       (a pytree, built at reset)
    policy(obs, key, carry)           -> (action, carry)

``rollout_one`` keeps the carry in its ``lax.scan`` state, so it is per-env under the
evaluator's vmap and replays bit-exactly like everything else. A plain stateless
``policy_fn(obs, key) -> (action, extras)`` is untouched: the rollout checks for
``init_carry`` at trace time and takes exactly the old code path when it is absent.

Like ``MLPPolicy``, the math is written against an array module ``xp`` (numpy in tests,
jax.numpy on device), so the same forward is CPU-testable without jax.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from breakpoint_mjx.adapters.mlp import MLPPolicy

HISTORY_LAYOUTS = ("frame", "term")
HISTORY_FILLS = ("first", "zeros")
HISTORY_ORDERS = ("oldest_first", "newest_first")


def is_stateful(policy_fn) -> bool:
    """True when ``policy_fn`` follows the stateful protocol (has ``init_carry``)."""
    return callable(getattr(policy_fn, "init_carry", None))


def parse_history(raw):
    """Normalize a contract ``history:`` block to a dict, or None for no history.

    Accepts an int (the length) or ``{length, layout, fill, order}``:

    * ``length``  frames kept (>= 1; 1 means no history).
    * ``layout``  ``frame``: whole frames concatenated, ``[o_{t-H+1} ... o_t]`` (legged_gym,
                  unitree_rl_gym); ``term``: each obs term's own history concatenated,
                  then the terms in contract order (Isaac Lab's ``history_length`` with
                  ``flatten_history_dim=True``). Default ``term`` for Isaac Lab parity.
    * ``fill``    what the history holds before H steps have passed: ``first`` repeats
                  the first observation (Isaac Lab's CircularBuffer) or ``zeros``.
    * ``order``   ``oldest_first`` (default) or ``newest_first`` within each block.
    """
    if raw is None or raw is False:
        return None
    d = {"length": raw} if isinstance(raw, int) and not isinstance(raw, bool) else dict(raw)
    length = d.get("length", 1)
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError(f"history.length must be an integer >= 1, got {length!r}")
    if length == 1:
        return None
    layout = d.get("layout", "term")
    fill = d.get("fill", "first")
    order = d.get("order", "oldest_first")
    if layout not in HISTORY_LAYOUTS:
        raise ValueError(f"history.layout must be one of {HISTORY_LAYOUTS}, got {layout!r}")
    if fill not in HISTORY_FILLS:
        raise ValueError(f"history.fill must be one of {HISTORY_FILLS}, got {fill!r}")
    if order not in HISTORY_ORDERS:
        raise ValueError(f"history.order must be one of {HISTORY_ORDERS}, got {order!r}")
    return {"length": length, "layout": layout, "fill": fill, "order": order}


def stack_history(buf, layout, order, term_dims, xp):
    """Flatten a (H, D) history buffer (row 0 oldest) into the policy's input vector."""
    if order == "newest_first":
        buf = buf[::-1]
    if layout == "frame":
        return xp.reshape(buf, (-1,))
    parts, s = [], 0
    for d in term_dims:
        parts.append(xp.reshape(buf[:, s:s + d], (-1,)))   # this term's H frames
        s += d
    return xp.concatenate(parts)


@dataclass(eq=False)        # identity hash: jax.jit and the backend caches key on it
class HistoryPolicy:
    """An MLP trained on a stack of the last ``length`` observation frames.

    ``frame_spec`` maps the env obs to ONE frame (contract map, no normalization);
    ``mlp`` is the weight-imported network whose input is the stacked vector, with the
    policy's own normalization over that whole vector (how RSL-RL's empirical normalizer
    sees it). ``term_dims`` are the per-term widths of a frame (needed for ``term``
    layout)."""

    frame_spec: object
    mlp: MLPPolicy
    length: int
    layout: str = "term"
    fill: str = "first"
    order: str = "oldest_first"
    term_dims: tuple = ()
    contract: object = None
    frame_dim: int = field(default=0, init=False)

    def __post_init__(self):
        self.frame_dim = int(self.frame_spec.output_dim())
        if self.layout == "term":
            if not self.term_dims or sum(self.term_dims) != self.frame_dim:
                raise ValueError(f"term layout needs per-term widths summing to the frame "
                                 f"width {self.frame_dim}; got {list(self.term_dims)}")
        want = self.length * self.frame_dim
        if self.mlp.in_dim != want:
            raise ValueError(
                f"the policy's first layer takes {self.mlp.in_dim} inputs but a history "
                f"of {self.length} x {self.frame_dim}-wide frames is {want}; check "
                f"history.length and the obs terms")

    # the same surface MLPPolicy exposes to the loaders
    @property
    def in_dim(self):
        return self.mlp.in_dim

    @property
    def action_size(self):
        return self.mlp.action_size

    @property
    def obs_spec(self):
        return self.frame_spec

    @property
    def activation(self):
        return self.mlp.activation

    @property
    def policy_fn(self):
        return self

    def _frame(self, obs, xp):
        return self.frame_spec.transform(obs, xp)

    def _init(self, obs, xp):
        f = self._frame(obs, xp)
        if self.fill == "zeros":
            return xp.zeros((self.length, self.frame_dim), dtype=f.dtype)
        return xp.tile(f[None, :], (self.length, 1))

    def _step(self, obs, buf, xp):
        f = self._frame(obs, xp)
        buf = xp.concatenate([buf[1:], f[None, :]], axis=0)       # drop oldest, push newest
        x = stack_history(buf, self.layout, self.order, self.term_dims, xp)
        return self.mlp._run({"_history": x}, xp), buf

    def init_carry(self, obs):
        import jax.numpy as jnp

        return self._init(obs, jnp)

    def __call__(self, obs, key=None, carry=None):
        import jax
        import jax.numpy as jnp

        if carry is None:
            raise TypeError("HistoryPolicy is stateful: call it as policy(obs, key, carry) "
                            "with carry from init_carry(obs)")
        with jax.default_matmul_precision("highest"):
            return self._step(obs, carry, jnp)

    def rollout_np(self, obs_seq):
        """Actions for a sequence of obs dicts, in numpy (CPU tests). Mirrors the rollout:
        the carry is built from the first obs, then every obs (the first included) is
        pushed before acting."""
        buf = self._init(obs_seq[0], np)
        out = []
        for o in obs_seq:
            a, buf = self._step(o, buf, np)
            out.append(a)
        return np.stack(out)


# --------------------------------------------------------------------------------------
# recurrent cores (torch gate conventions, so an RSL-RL state_dict maps over directly)
# --------------------------------------------------------------------------------------

def _sigmoid(x, xp):
    return 1.0 / (1.0 + xp.exp(-x))


def lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh, xp):
    """One torch.nn.LSTM step. Weights are torch-shaped: w_ih (4H, in), gates i, f, g, o."""
    gates = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh
    n = h.shape[-1]
    i = _sigmoid(gates[..., 0:n], xp)
    f = _sigmoid(gates[..., n:2 * n], xp)
    g = xp.tanh(gates[..., 2 * n:3 * n])
    o = _sigmoid(gates[..., 3 * n:4 * n], xp)
    c = f * c + i * g
    return o * xp.tanh(c), c


def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh, xp):
    """One torch.nn.GRU step. Gates r, z, n (torch order)."""
    n = h.shape[-1]
    gi = x @ w_ih.T + b_ih
    gh = h @ w_hh.T + b_hh
    r = _sigmoid(gi[..., 0:n] + gh[..., 0:n], xp)
    z = _sigmoid(gi[..., n:2 * n] + gh[..., n:2 * n], xp)
    cand = xp.tanh(gi[..., 2 * n:3 * n] + r * gh[..., 2 * n:3 * n])
    return (1.0 - z) * cand + z * h


@dataclass(eq=False)
class RecurrentPolicy:
    """An RSL-RL-style recurrent actor: obs -> (normalize) -> LSTM/GRU stack -> MLP.

    ``layers`` is a list of ``(w_ih, w_hh, b_ih, b_hh)`` per recurrent layer in torch
    shapes; ``mlp`` is the actor head on the last layer's hidden state. The carry is
    ``(h, c)`` stacked over layers for an LSTM, ``h`` for a GRU, zeros at reset (RSL-RL
    resets hidden state on episode start)."""

    kind: str
    layers: list
    mlp: MLPPolicy
    obs_spec: object
    contract: object = None
    hidden: int = field(default=0, init=False)

    def __post_init__(self):
        if self.kind not in ("lstm", "gru"):
            raise ValueError(f"recurrent kind must be lstm or gru, got {self.kind!r}")
        if not self.layers:
            raise ValueError("a recurrent policy needs at least one layer")
        gates = 4 if self.kind == "lstm" else 3
        self.layers = [tuple(np.asarray(a, dtype=np.float32) for a in L)
                       for L in self.layers]
        self.hidden = int(self.layers[0][1].shape[1])
        for li, (w_ih, w_hh, b_ih, b_hh) in enumerate(self.layers):
            if w_hh.shape != (gates * self.hidden, self.hidden):
                raise ValueError(f"layer {li}: weight_hh shape {w_hh.shape} is not "
                                 f"({gates}*{self.hidden}, {self.hidden}) for a {self.kind}")
            if w_ih.shape[0] != gates * self.hidden or b_ih.shape != (gates * self.hidden,):
                raise ValueError(f"layer {li}: weight_ih/bias shapes do not match a "
                                 f"{self.kind} of hidden size {self.hidden}")
            if li and w_ih.shape[1] != self.hidden:
                raise ValueError(f"layer {li}: input width {w_ih.shape[1]} != hidden "
                                 f"{self.hidden}")
        if self.mlp.in_dim != self.hidden:
            raise ValueError(f"actor head takes {self.mlp.in_dim} inputs but the "
                             f"{self.kind} hidden size is {self.hidden}")
        od = self.obs_spec.output_dim()
        if od is not None and od != self.layers[0][0].shape[1]:
            raise ValueError(f"obs maps to {od} channels but the {self.kind} takes "
                             f"{self.layers[0][0].shape[1]}")

    @property
    def in_dim(self):
        return int(self.layers[0][0].shape[1])

    @property
    def action_size(self):
        return self.mlp.action_size

    @property
    def activation(self):
        return self.mlp.activation

    @property
    def policy_fn(self):
        return self

    def _zeros(self, xp):
        z = xp.zeros((len(self.layers), self.hidden), dtype=np.float32)
        return (z, z) if self.kind == "lstm" else z

    def _step(self, obs, carry, xp):
        x = self.obs_spec.transform(obs, xp)
        hs, cs = [], []
        for li, (w_ih, w_hh, b_ih, b_hh) in enumerate(self.layers):
            w_ih, w_hh = xp.asarray(w_ih), xp.asarray(w_hh)
            b_ih, b_hh = xp.asarray(b_ih), xp.asarray(b_hh)
            if self.kind == "lstm":
                h, c = lstm_cell(x, carry[0][li], carry[1][li], w_ih, w_hh, b_ih, b_hh, xp)
                cs.append(c)
            else:
                h = gru_cell(x, carry[li], w_ih, w_hh, b_ih, b_hh, xp)
            hs.append(h)
            x = h
        a = self.mlp._run({"_hidden": x}, xp)
        new = (xp.stack(hs), xp.stack(cs)) if self.kind == "lstm" else xp.stack(hs)
        return a, new

    def init_carry(self, obs):
        import jax.numpy as jnp

        return self._zeros(jnp)

    def __call__(self, obs, key=None, carry=None):
        import jax
        import jax.numpy as jnp

        if carry is None:
            raise TypeError("RecurrentPolicy is stateful: call it as policy(obs, key, "
                            "carry) with carry from init_carry(obs)")
        with jax.default_matmul_precision("highest"):
            return self._step(obs, carry, jnp)

    def rollout_np(self, obs_seq):
        carry = self._zeros(np)
        out = []
        for o in obs_seq:
            a, carry = self._step(o, carry, np)
            out.append(a)
        return np.stack(out)


def recurrent_from_state_dict(sd, *, kind=None, obs_spec, activation="elu",
                              out_transform=None, actor_prefix="actor.",
                              rnn_prefix="memory_a.rnn."):
    """Build a ``RecurrentPolicy`` from an RSL-RL ``ActorCriticRecurrent`` state_dict
    (a mapping of names to arrays, e.g. ``np.load`` of a dumped .npz).

    Reads ``memory_a.rnn.weight_ih_l{k}`` / ``weight_hh_l{k}`` / ``bias_ih_l{k}`` /
    ``bias_hh_l{k}`` for each layer and the ``actor.{i}.weight`` / ``actor.{i}.bias``
    Linear layers (torch ``(out, in)``). ``kind`` is inferred from the gate count when
    omitted (4H rows: LSTM, 3H: GRU)."""
    from breakpoint_mjx.adapters.importers import from_params

    layers, k = [], 0
    while f"{rnn_prefix}weight_ih_l{k}" in sd:
        layers.append(tuple(np.asarray(sd[f"{rnn_prefix}{n}_l{k}"])
                            for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")))
        k += 1
    if not layers:
        raise ValueError(f"no recurrent weights under {rnn_prefix!r} (expected "
                         f"{rnn_prefix}weight_ih_l0); is this an ActorCriticRecurrent?")
    hidden = layers[0][1].shape[1]
    rows = layers[0][1].shape[0] // hidden
    inferred = {4: "lstm", 3: "gru"}.get(rows)
    if inferred is None:
        raise ValueError(f"cannot tell LSTM from GRU: weight_hh has {rows}x hidden rows")
    if kind is not None and kind != inferred:
        raise ValueError(f"kind={kind!r} but the weights are a {inferred}")
    idx = sorted({int(n[len(actor_prefix):].split(".")[0]) for n in sd
                  if n.startswith(actor_prefix) and n.endswith(".weight")})
    if not idx:
        raise ValueError(f"no actor Linear layers under {actor_prefix!r}")
    pairs = [(np.asarray(sd[f"{actor_prefix}{i}.weight"]),
              np.asarray(sd[f"{actor_prefix}{i}.bias"])) for i in idx]
    from breakpoint_mjx.adapters.obs_spec import ObsSpec

    head = from_params(pairs, activation=activation, weight_layout="out_in",
                       obs_spec=ObsSpec(keys=("_hidden",), expected_dim=hidden),
                       out_transform=out_transform)
    return RecurrentPolicy(kind=inferred, layers=layers, mlp=head, obs_spec=obs_spec)
