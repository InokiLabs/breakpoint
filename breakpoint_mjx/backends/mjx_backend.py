"""MJX re-simulation backend (T1 cross-config validity, primary/cheap path).

A thin wrapper over the SAME MJX rollout the evaluator/renderer use (`rollout_one`),
so a re-sim under a config-varied model reproduces the scored failure bit-for-bit
except for the physics config being tested. This is the "config variation" evidence
source in the spec; the independent-implementation evidence is `MujocoCBackend`.

Import-safety: jax / apply_thetas / rollout_one are imported INSIDE `simulate` (they
pull in jax), so importing this module on a CPU-only box is safe. `crossval` imports
this lazily inside `validate_failures`.
"""
from __future__ import annotations

import numpy as np


class MjxBackend:
    """Re-simulate a single recorded failure via `rollout_one` on a given MJX model.

    Mirrors `render.render_failure`'s squeeze discipline exactly: apply the failure's
    theta to the (config-varied) base model with N=1, squeeze row 0 to a single-env
    model, reconstruct the exact evaluator key, and run the shared `rollout_one` (no
    state collection). Returns the per-step summary dict (numpy) so it feeds straight
    into `failure.component_signals` / `failure_score`.
    """

    name = "mjx"

    def __init__(self, env, policy_fn, target, space, consts, base_model, horizon):
        self.env = env
        self.policy_fn = policy_fn
        self.target = target
        self.space = space
        self.consts = consts  # opaque per-target consts blob (interpreted by the objective)
        self.base_model = base_model
        self.horizon = int(horizon)

    def _key(self, record):
        """Reconstruct the EXACT evaluator key: fold_in(uint32(batch_key), env_index).

        RAISES (mirrors render_failure) for a legacy record that predates exact-repro
        (Upgrade #6): there is no way to reconstruct the original scored rollout's key, so
        validating against a fabricated key would silently score the WRONG rollout. The
        guard is checked BEFORE importing jax so it is CPU-testable and never masks a
        missing-jax error with a spurious success.
        """
        if record.batch_key is None or record.env_index is None:
            raise ValueError(
                "record predates exact-repro (Upgrade #6); re-run the search to validate"
            )
        import jax
        import jax.numpy as jnp

        return jax.random.fold_in(
            jnp.asarray(record.batch_key, dtype=jnp.uint32), int(record.env_index)
        )

    def simulate(self, record, model=None, collect_obs_act=False):
        """Re-sim `record` on `model` (default: nominal base). Returns summary dict (numpy).

        ``collect_obs_act`` forwards rollout_one's demonstration-collection flag, adding
        the per-step ``obs_state`` (what the policy SAW, post-adversary) and ``action``
        (raw, pre-latency) to the summary. Off by default so the scoring path is
        untouched; the cross-engine parity test turns it on to compare the two engines'
        step-0 observation and action.
        """
        # Reconstruct the exact recorded key FIRST -- raises early (no jax needed for the
        # guard) on a legacy record rather than silently validating the wrong rollout.
        key = self._key(record)

        import jax.numpy as jnp

        from breakpoint_mjx.rollout import rollout_one
        from breakpoint_mjx.thetas import apply_thetas

        base = model if model is not None else self.base_model
        # T7: per-target summary hook bound to consts (opaque per-target blob), so the
        # re-sim summarizes exactly like the evaluator that scored the record.
        summarize = lambda state: self.target.objective.summarize(state, self.consts)  # noqa: E731

        thetas = jnp.asarray(np.asarray(record.theta_vector, dtype=float)[None, :])  # (1, D)
        batched, _, rollout = apply_thetas(self.target, self.space, base, thetas, self.consts)
        # apply_thetas batches exactly target.model_fields (leading dim n=1); squeeze
        # just those to a single-env model, exactly like render_failure.
        model_i = batched.tree_replace(
            {f: getattr(batched, f)[0] for f in self.target.model_fields}
        )
        rollout_i = {k: v[0] for k, v in rollout.items()}

        summary = rollout_one(
            self.env, self.policy_fn, model_i, rollout_i, key, self.horizon,
            summarize, collect_states=False, collect_obs_act=collect_obs_act,
        )
        return {k: np.asarray(v) for k, v in summary.items()}
