"""BYO-policy adapters (T4, workstream 2b): weight-import a foreign policy into JAX.

BreakPoint tests FROZEN policies, and its evaluator runs the policy INSIDE ``jax.vmap`` on
GPU (``rollout_one`` -> ``policy_fn(obs, key)`` under ``lax.scan`` under ``vmap``). A foreign
ONNX/PyTorch callable therefore cannot be dropped in -- it can't run inside a jax trace. The
adapter's job is WEIGHT-IMPORT: read the foreign policy's weights and rebuild the forward
pass in JAX so it runs on-device at full parallelism.

Public API
----------
* ``mlp_forward(params, x, activation, xp)`` -- array-module-agnostic MLP forward (numpy in
  tests, jax.numpy in production; identical math).
* ``MLPPolicy`` -- holds numpy weights + an ``ObsSpec`` + an optional ``OutputTransform`` and
  exposes ``policy_fn(obs, key) -> (action, extras)`` (pure jax, vmappable).
* ``ObsSpec`` -- the observation shim (which key(s), ordering, per-dim mean/std).
* ``OutputTransform`` -- deterministic action post-processing (tanh squash / scale / bias).
* ``from_params`` / ``from_npz`` / ``from_onnx`` -- weight importers.
* ``build_policy(spec, ...)`` -- the ``"onnx:<path>"`` / ``"params:<path>"`` / brax ``.pkl``
  pipeline loader; ``BP_OBS_SPEC`` (a contract YAML) applies an obs/action contract.
* ``PolicyContract`` / ``build_contract`` / ``load_contract`` -- a policy's obs/action
  interface as declarative YAML resolved to index arrays (obs_spec.py).
* ``from_isaaclab_cfg`` / ``from_rsl_rl_run`` / ``from_unitree_rl_gym`` -- training-config
  importers that emit a contract plus a table of fields BreakPoint cannot honor.
* ``actuation`` -- the contract's ``actuation:`` block (the policy's training PD gains),
  applied to the Go1 target's mj + MJX models by ``targets/go1.load_env``.
* ``contract_id.identify_contract`` -- Behavioral Contract Identification (imported
  explicitly; it needs jax + the target env).

Import-safety: this package imports with NO jax and NO onnx installed. ``jax.numpy`` is
imported lazily inside ``MLPPolicy.policy_fn``; ``onnx`` inside ``from_onnx``. So the
importers/shim/forward are all CPU-testable, and the jax forward is exercised only on the
GPU (Colab).

Scope (documented, not enforced): deterministic MLP control policies. Stochastic sampling,
recurrent (RNN/LSTM) state, and vision/CNN/transformer/VLA nets are unsupported here -- a
harder later tail. For a Gaussian-policy export the MEAN (pre-tanh) head is used, which is
what a deterministic eval takes.
"""
from __future__ import annotations

from breakpoint_mjx.adapters.mlp import (
    ACTIVATIONS,
    MLPPolicy,
    OutputTransform,
    mlp_forward,
    resolve_activation,
)
from breakpoint_mjx.adapters.obs_spec import (
    PolicyContract,
    TargetInterface,
    ObsSpec,
    build_contract,
    load_contract,
    target_interface,
)
from breakpoint_mjx.adapters.actuation import (
    actuation_block,
    apply_actuation,
    parse_actuation,
)
from breakpoint_mjx.adapters.importers import (
    apply_contract,
    build_policy,
    from_brax_params,
    from_brax_pkl,
    from_isaaclab_cfg,
    from_npz,
    from_onnx,
    from_params,
    from_rsl_rl_run,
    from_unitree_rl_gym,
    parse_policy_spec,
)

__all__ = [
    "ACTIVATIONS",
    "MLPPolicy",
    "ObsSpec",
    "OutputTransform",
    "mlp_forward",
    "resolve_activation",
    "from_params",
    "from_npz",
    "from_onnx",
    "build_policy",
    "parse_policy_spec",
    "from_brax_params",
    "from_brax_pkl",
    "apply_contract",
    "PolicyContract",
    "TargetInterface",
    "build_contract",
    "load_contract",
    "target_interface",
    "from_isaaclab_cfg",
    "from_rsl_rl_run",
    "from_unitree_rl_gym",
    "actuation_block",
    "apply_actuation",
    "parse_actuation",
]
