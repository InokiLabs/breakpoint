"""Theta space sampling and MJX model batching utilities.

Generic across robots via the Target abstraction (Upgrade #1): apply_thetas vmaps
target.build_fields over the model-dim theta columns and builds in_axes / the
batched model from target.model_fields (the validated single-source-of-truth
pattern from the Phase 2 Colab proof-of-concept, now per target).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from breakpoint_mjx.config import ThetaSpace
from breakpoint_mjx.targets.go1 import GO1 as _GO1

# Backward-compat alias: the Go1 model fields, sourced from the go1 target so it
# can never drift from apply_thetas / render (which use target.model_fields).
MODEL_FIELDS = _GO1.model_fields


def sample_uniform(space: ThetaSpace, key: jax.Array, n: int) -> jax.Array:
    """Sample n parameter vectors uniformly within bounds.

    Returns (n, D) float32 array where D = len(space.dims).
    """
    lo = jnp.array(space.lower(), dtype=jnp.float32)
    hi = jnp.array(space.upper(), dtype=jnp.float32)
    return jax.random.uniform(key, shape=(n, len(space.dims)), minval=lo, maxval=hi)


def split_named(space: ThetaSpace, thetas: jax.Array) -> dict[str, jax.Array]:
    """Split (n, D) theta matrix into {name: column (n,)} dict.

    Column order matches YAML insertion order in theta_space.yaml.
    """
    return {name: thetas[:, i] for i, name in enumerate(space.names())}


def model_constants(env) -> tuple[int, int]:
    """Backward-compat shim for the Go1 (floor_geom_id, torso_body_id).

    Prefer ``target.resolve_constants(env)``; this delegates to the go1 target so
    older Colab cells keep working.
    """
    return _GO1.resolve_constants(env)


def apply_thetas(target, space: ThetaSpace, base_model, thetas: jax.Array, consts):
    """Batch base_model across N theta rows via target.build_fields.

    Implements the validated pattern from the Phase 2 Colab proof-of-concept: a
    jax.vmap inner fn builds per-row modified model fields, which are then stacked
    via tree_replace. in_axes is built from target.model_fields so it can't drift
    from the batched tree_replace.

    Parameters
    ----------
    target     : the active Target (provides build_fields + model_fields).
    space      : ThetaSpace (model + rollout dims as in the target's theta_space.yaml).
    base_model : env.mjx_model (the captured base MJX model from load_env()).
    thetas     : (N, D) float32 theta matrix from sample_uniform().
    consts     : target-specific constants from target.resolve_constants(env);
                 by convention (floor_geom_id, torso_body_id).

    Returns
    -------
    batched_model : MJX model with N-leading-dim arrays on the modified fields.
    in_axes       : matching pytree with 0 on those fields, None everywhere else.
    rollout       : {name: (N,)} for each kind == "rollout" dim the space declares
                    (e.g. obs noise/bias/dropout, latency, gait-push -- read by name in
                    rollout_one, which degrades gracefully for any absent dim).
    """
    col = split_named(space, thetas)

    # The model-dim columns (kind == "model"), keyed by name -- vmapped per row.
    model_names = [n for n in space.names() if space.dims[n].kind == "model"]
    model_cols = {n: col[n] for n in model_names}

    # Build per-row modified field arrays via vmap over the N theta rows. base_model
    # and consts are closed over (broadcast, not mapped); only model_cols is mapped.
    batched_fields = jax.vmap(
        lambda scalars: target.build_fields(scalars, base_model, consts)
    )(model_cols)

    # in_axes: all None (broadcast), then mark the modified fields as axis 0.
    # Built from target.model_fields so this dict and the batched tree_replace
    # below can't drift.
    in_axes = jax.tree_util.tree_map(lambda x: None, base_model)
    in_axes = in_axes.tree_replace({f: 0 for f in target.model_fields})

    # batched_model: base with the modified fields replaced by N-row stacked arrays.
    batched = base_model.tree_replace(
        {f: batched_fields[f] for f in target.model_fields}
    )

    # Pass through whatever rollout dims the space declares (kind == "rollout"),
    # keyed by name. The recognized dims are obs_noise_scale and latency_steps;
    # rollout_one reads them by name and degrades gracefully if one is absent.
    rollout = {
        n: col[n] for n in space.names() if space.dims[n].kind == "rollout"  # each (N,)
    }

    return batched, in_axes, rollout
