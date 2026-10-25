"""Cross-engine replay of validated failures (C MuJoCo, full-body collision scene).

Thin wrapper over ``breakpoint_mjx.equal_budget.make_replay_fn`` (the pipeline's replay
machinery and reproduction rule). Agreement between two simulators is evidence that a
failure is not an artifact of one contact model; it is NOT evidence of real-world transfer.
"""
from __future__ import annotations


def make_replayer(env, policy_fn, target, full_space, consts, cfg, base_model, k_seeds,
                  seed=0):
    """(fn(X) -> {"n", "confirmed", "rows", ...}, None) or (None, reason).

    Always built on the FULL theta space: conservative-regime thetas are valid points of
    it (plausible bands lie inside the stress-test bounds)."""
    from breakpoint_mjx.equal_budget import make_replay_fn

    return make_replay_fn(env, policy_fn, target, full_space, consts, cfg, base_model,
                          k_seeds=k_seeds, seed=seed)
