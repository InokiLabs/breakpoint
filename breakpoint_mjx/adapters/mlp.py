"""Array-module-agnostic MLP forward + a JAX-vmappable ``MLPPolicy`` (T4, workstream 2b).

The evaluator runs the policy INSIDE ``jax.vmap`` on GPU (``rollout_one`` calls
``policy_fn(obs, key)`` under ``lax.scan`` under ``vmap``), so a foreign ONNX/PyTorch
callable cannot be dropped in -- it can't run inside a jax trace. The solution is
WEIGHT-IMPORT: export the foreign policy's weights and rebuild the forward pass here so it
runs on-device at full parallelism.

``mlp_forward`` is written against an abstract array module ``xp`` (``numpy`` in tests,
``jax.numpy`` in production) so the EXACT same math is CPU-testable without jax and runs
on-device unchanged. ``MLPPolicy.policy_fn`` fixes ``xp = jax.numpy`` (imported lazily so
this module imports with no jax present) and exposes the
``policy_fn(obs, key) -> (action, extras)`` contract the evaluator expects.

Scope: deterministic MLP control policies (the common RSL-RL / Isaac Lab / torch / ONNX
export). For a Gaussian-policy export we reconstruct the MEAN (pre-tanh) action head, which
is what a deterministic eval uses; stochastic sampling, recurrent state, and vision/CNN
nets are out of scope (see module docstring in ``adapters/__init__``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from breakpoint_mjx.adapters.obs_spec import ObsSpec


# --------------------------------------------------------------------------- #
# Activations -- array-module-agnostic (same op on numpy and jax.numpy)
# --------------------------------------------------------------------------- #
def _tanh(x, xp):
    return xp.tanh(x)


def _relu(x, xp):
    return xp.maximum(x, 0)


def _elu(x, xp):
    # ELU(alpha=1): x for x>0 else exp(x)-1. expm1 is exact + present on numpy and jnp.
    return xp.where(x > 0, x, xp.expm1(x))


def _silu(x, xp):
    # SiLU / swish: x * sigmoid(x). brax's PPO networks (so every Playground policy) default
    # to it, which is what lets a brax checkpoint weight-import as an MLPPolicy.
    return x / (1.0 + xp.exp(-x))


def _identity(x, xp):
    return x


ACTIVATIONS = {
    "tanh": _tanh,
    "relu": _relu,
    "elu": _elu,
    "silu": _silu,
    "swish": _silu,
    "identity": _identity,
    "linear": _identity,
    "none": _identity,
}


def resolve_activation(activation):
    """Accept a callable ``act(x, xp)`` or a name in ``ACTIVATIONS`` (case-insensitive)."""
    if callable(activation):
        return activation
    key = str(activation).lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; known: {sorted(ACTIVATIONS)} "
            f"or pass a callable act(x, xp)"
        )
    return ACTIVATIONS[key]


# --------------------------------------------------------------------------- #
# The forward -- one place, two array modules
# --------------------------------------------------------------------------- #
def mlp_forward(params, x, activation="tanh", xp=np):
    """Feed-forward MLP: ``h = act(h @ W + b)`` per hidden layer, LINEAR last layer.

    Parameters
    ----------
    params : list[(W, b)]
        Ordered layers. ``W`` is in the CANONICAL ``(in_dim, out_dim)`` orientation so the
        layer is ``h @ W + b`` (this is the flax "kernel" convention; a torch
        ``nn.Linear`` weight is ``(out, in)`` and is transposed to canonical form by the
        importers). ``b`` may be ``None`` (treated as zero).
    x : array
        Input vector, shape ``(in_dim,)`` (or batched ``(..., in_dim)``).
    activation : str | callable
        Hidden-layer activation, applied after every layer EXCEPT the last (the output is a
        linear action head -- the mean for a Gaussian-policy export).
    xp : module
        Array module: ``numpy`` for CPU tests, ``jax.numpy`` in production. The math is
        identical either way; ``W``/``b`` are moved into ``xp`` via ``xp.asarray`` so plain
        numpy weights run on-device under jax.

    Returns
    -------
    action : array, shape ``(out_dim,)`` -- the deterministic action (pre output-transform).
    """
    act = resolve_activation(activation)
    n = len(params)
    if n == 0:
        raise ValueError("mlp_forward: params is empty")
    h = x
    for i, (W, b) in enumerate(params):
        W = xp.asarray(W)
        h = h @ W
        if b is not None:
            h = h + xp.asarray(b)
        if i < n - 1:
            h = act(h, xp)
    return h


# --------------------------------------------------------------------------- #
# Output transform -- deterministic action post-processing
# --------------------------------------------------------------------------- #
@dataclass
class OutputTransform:
    """Optional post-processing of the raw MLP head into an env action.

    Applied as ``action = scale * f(head[:mean_dim]) + bias`` where ``f`` is the squash
    function (identity when ``squash`` is falsy). Covers the common export conventions:

    * raw-action policies (RSL-RL / Isaac Lab PPO actors): default (no squash, unit scale).
    * tanh-squashed policies (SAC-style): ``squash=True`` (or ``"tanh"``) + ``scale``/``bias``
      mapping the squashed output to the env action range.
    * sigmoid-squashed heads: ``squash="sigmoid"``.

    ``squash`` accepts ``True`` (== ``"tanh"``), ``"tanh"``, ``"sigmoid"``, or falsy (none).
    ``mean_dim`` handles a final layer that emits a concatenated ``[mean, log_std]`` head: it
    keeps the first ``mean_dim`` entries (the mean) for deterministic eval.

    ``perm`` maps the policy's joint order onto the env's actuator order: env actuator ``j``
    receives policy output ``perm[j]``. It is applied after the squash and before
    ``scale``/``bias``, so a per-joint ``scale``/``bias`` array is in ENV actuator order.
    This is how a policy trained with Isaac Lab's joint-type-major order (all hips, then all
    thighs, then all calves) drives a Playground robot whose actuators are leg-major.
    """

    squash: object = False
    scale: object = 1.0
    bias: object = 0.0
    mean_dim: object = None
    perm: object = None

    def __post_init__(self):
        if self.perm is not None:
            self.perm = tuple(int(i) for i in self.perm)
            if sorted(self.perm) != list(range(len(self.perm))):
                raise ValueError(
                    f"OutputTransform.perm must be a permutation of 0..{len(self.perm) - 1}; "
                    f"got {list(self.perm)}"
                )

    def apply(self, a, xp):
        if self.mean_dim is not None:
            a = a[..., : int(self.mean_dim)]
        if self.squash:
            fn = self.squash if isinstance(self.squash, str) else "tanh"
            if fn == "tanh":
                a = xp.tanh(a)
            elif fn == "sigmoid":
                a = 1.0 / (1.0 + xp.exp(-a))
            else:
                raise ValueError(
                    f"unknown squash {self.squash!r}; use True/'tanh'/'sigmoid'/False"
                )
        if self.perm is not None:
            # a static numpy index array works under numpy and inside a jax trace alike
            # (jnp.take rejects a python list)
            a = a[..., np.asarray(self.perm, dtype=int)]
        return a * xp.asarray(self.scale) + xp.asarray(self.bias)


# --------------------------------------------------------------------------- #
# The policy object
# --------------------------------------------------------------------------- #
def _normalize_layer_pair(layer):
    """Coerce one ``(W, b)`` layer to numpy arrays; fill a missing bias with zeros."""
    if not (isinstance(layer, (tuple, list)) and len(layer) == 2):
        raise ValueError(
            "each MLPPolicy layer must be a (W, b) pair; got "
            f"{type(layer).__name__} of length "
            f"{len(layer) if hasattr(layer, '__len__') else 'n/a'}"
        )
    W, b = layer
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError(f"layer weight must be 2-D (in, out); got shape {W.shape}")
    b = np.zeros(W.shape[1]) if b is None else np.asarray(b).reshape(-1)
    if b.shape[0] != W.shape[1]:
        raise ValueError(
            f"layer bias length {b.shape[0]} != weight out-dim {W.shape[1]}"
        )
    return (W, b)


@dataclass
class MLPPolicy:
    """A weight-imported MLP policy exposing the evaluator's ``policy_fn`` contract.

    Holds plain numpy weights (import-safe + picklable); ``policy_fn`` runs the forward in
    ``jax.numpy`` so it is pure jax and vmappable on-device. ``act_np`` runs the identical
    forward in numpy for CPU tests.

    ``params`` are ``[(W, b), ...]`` in CANONICAL ``(in, out)`` orientation. Build one via
    ``adapters.from_params`` / ``from_onnx`` rather than constructing directly (they handle
    weight-layout transposition and wire ``obs_spec.expected_dim`` to the first-layer input
    dim so an obs/policy mismatch fails fast).
    """

    params: object
    activation: object = "tanh"
    obs_spec: object = None
    out_transform: object = None
    in_dim: int = field(default=0, init=False)
    action_dim: int = field(default=0, init=False)

    def __post_init__(self):
        self.params = [_normalize_layer_pair(l) for l in self.params]
        if not self.params:
            raise ValueError("MLPPolicy needs at least one layer")
        # Verify layer shapes chain: out of layer i == in of layer i+1.
        for i in range(len(self.params) - 1):
            out_i = self.params[i][0].shape[1]
            in_next = self.params[i + 1][0].shape[0]
            if out_i != in_next:
                raise ValueError(
                    f"layer {i} out-dim {out_i} != layer {i + 1} in-dim {in_next}; "
                    f"layers are not a valid MLP chain"
                )
        self.in_dim = int(self.params[0][0].shape[0])
        self.action_dim = int(self.params[-1][0].shape[1])
        if self.out_transform is not None and self.out_transform.mean_dim is not None:
            self.action_dim = int(self.out_transform.mean_dim)

        if self.obs_spec is None:
            self.obs_spec = ObsSpec(expected_dim=self.in_dim)
        # Cross-check the obs shim's mapped length against the MLP's input dim.
        od = self.obs_spec.output_dim()
        if od is not None and od != self.in_dim:
            raise ValueError(
                f"ObsSpec maps to length {od} but the MLP's first layer expects "
                f"{self.in_dim}; fix ObsSpec.keys/indices/normalization"
            )
        # Pin expected_dim to the MLP input so the runtime obs-length guard is always active
        # (re-validates any mean/std lengths against in_dim too).
        if self.obs_spec.expected_dim is None:
            self.obs_spec = self.obs_spec.with_expected_dim(self.in_dim)

    @property
    def action_size(self):
        """Action vector length -- must equal ``env.action_size`` for the evaluator."""
        return self.action_dim

    def _run(self, obs, xp):
        x = self.obs_spec.transform(obs, xp)
        a = mlp_forward(self.params, x, self.activation, xp)
        if self.out_transform is not None:
            a = self.out_transform.apply(a, xp)
        return a

    def act_np(self, obs):
        """Deterministic forward in numpy (for CPU tests; no jax needed)."""
        return self._run(obs, np)

    def policy_fn(self, obs, key=None):
        """The evaluator contract: ``(obs_dict, key) -> (action, extras)``.

        Pure jax: ``jax.numpy`` is imported lazily (so this module imports with no jax) and
        every op inside is a jnp op, so this runs inside the evaluator's ``lax.scan`` /
        ``vmap``. Deterministic -- ``key`` is accepted (the evaluator always passes one) and
        ignored, because a control-policy eval uses the mean action with no sampling.

        On-device precision: the forward runs at float32 (JAX's default), so "CPU-green"
        (float64) tests do not by themselves certify the on-device numbers. The matmuls are
        pinned to the highest precision (``default_matmul_precision("highest")``) so TF32 on
        Ampere+ GPUs cannot silently degrade fidelity vs the source policy. The math is
        otherwise identical to the numpy path.
        """
        import jax
        import jax.numpy as jnp

        with jax.default_matmul_precision("highest"):
            a = self._run(obs, jnp)
        return a, {}
