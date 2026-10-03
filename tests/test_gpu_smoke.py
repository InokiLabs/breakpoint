"""GPU-path smoke tests (Upgrade #7a): exercise the JAX/MJX path on TINY sizes.

Covers, on N=4 envs x horizon=8: apply_thetas shapes, the evaluator contract,
CMA-ES one generation, MAP-Elites one iteration, the Upgrade #6 determinism
invariant (vmapped eval row i == direct rollout_one with env_key(key, i)), and the
state-stacking + bit-exact-repro property of rollout_one(collect_states=True).

SKIP semantics (so `pytest` stays green on a CPU-only laptop, never a collection
error): the module first does importorskip on jax + mujoco_playground -- if either is
missing it shows SKIPPED at collection. Then it skips at module level unless a JAX GPU
device is present OR BP_RUN_GPU=1 is set. On Colab run:

    BP_RUN_GPU=1 python -m pytest tests/test_gpu_smoke.py -q

This file cannot run on the dev box (no jax/GPU); it is validated by reading + the
Colab smoke cell appended to colab/breakpoint_mjx.ipynb.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

# --- skip ladder ----------------------------------------------------------- #
# No jax / no mujoco_playground -> SKIPPED at import (importorskip raises a
# module-level Skipped), so a CPU box never errors on collection.
jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_playground")

# Then require a real GPU device, or an explicit opt-in via BP_RUN_GPU.
if not any(getattr(d, "platform", None) == "gpu" for d in jax.devices()) and not os.environ.get(
    "BP_RUN_GPU"
):
    pytest.skip(
        "no JAX GPU device and BP_RUN_GPU not set (set BP_RUN_GPU=1 to force on Colab)",
        allow_module_level=True,
    )

import jax.numpy as jnp  # noqa: E402  (only safe to import once jax is known present)
import numpy as np  # noqa: E402

from breakpoint_mjx.atlas import map_elites_search  # noqa: E402
from breakpoint_mjx.config import SearchConfig, load_theta_space  # noqa: E402
from breakpoint_mjx.evaluator import make_evaluator  # noqa: E402
from breakpoint_mjx.policy import make_policy_fn  # noqa: E402
from breakpoint_mjx.rollout import env_key, rollout_one  # noqa: E402
from breakpoint_mjx.search import cmaes_search  # noqa: E402
from breakpoint_mjx.targets import get_target  # noqa: E402
from breakpoint_mjx.thetas import apply_thetas, sample_uniform  # noqa: E402
from breakpoint_mjx.paths import config_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 4          # tiny batch / popsize / atlas batch (one XLA compile shared by all)
HORIZON = 8    # tiny rollout
_THRESHOLDS = {
    "trunk_height_min": 0.18,
    "up_proj_min": 0.5,
    "tracking_err_max": 0.5,
    "ang_vel_max": 8.0,
}
_CONTRACT_KEYS = ("trunk_height", "up_proj", "lin_vel", "ang_vel", "command")


@pytest.fixture(scope="module")
def fx():
    """Tiny shared fixture: go1 target + env + consts + space + deterministic policy
    + ONE evaluator. base_model and evaluate are captured BEFORE any evaluate() call
    (which mutates env._mjx_model), so apply_thetas in the tests below squeezes the
    pristine original model -- the same discipline render_failure uses."""
    target = get_target("go1")
    env = target.load_env()
    consts = target.resolve_constants(env)
    space = load_theta_space(config_path(target.theta_space_path))
    base_model = env.mjx_model  # pristine ORIGINAL; captured before evaluate mutates it

    # Deterministic policy. Prefer a trained checkpoint (exercises the real
    # make_policy_fn plumbing -- the normal Colab path); else a deterministic
    # zero-action policy. Either satisfies policy_fn(obs, key) -> (action, extras)
    # and is reproducible. These tests check plumbing/determinism, not policy quality.
    # train.py / run_pipeline.py namespace BP_OUT by target, so the trained policy
    # lives under <BP_OUT>/<target>/; look there (else fall back to a fresh policy).
    bp_out = os.path.join(os.environ.get("BP_OUT", str(ROOT / "out")), target.name)
    ckpt = Path(bp_out) / target.checkpoint_name
    if ckpt.exists():
        policy_fn = make_policy_fn(out_dir=bp_out, target=target)
    else:
        action_size = env.action_size

        def policy_fn(obs, key):  # deterministic: ignores obs + key
            return jnp.zeros((action_size,)), {}

    evaluate = make_evaluator(env, policy_fn, space, HORIZON, target, consts)
    return types.SimpleNamespace(
        target=target,
        env=env,
        consts=consts,
        space=space,
        base_model=base_model,
        policy_fn=policy_fn,
        evaluate=evaluate,
    )


def _thetas(fx, seed):
    return sample_uniform(fx.space, jax.random.PRNGKey(seed), N)


def _row_model(fx, thetas, i):
    """apply_thetas, then squeeze row i into a single-env (model_i, rollout_i) for a
    direct rollout_one -- mirrors render.render_failure's squeeze, and uses the
    captured pristine base_model (never env.mjx_model, which evaluate has mutated)."""
    batched, _, rollout = apply_thetas(fx.target, fx.space, fx.base_model, thetas, fx.consts)
    model_i = batched.tree_replace({f: getattr(batched, f)[i] for f in fx.target.model_fields})
    rollout_i = {k: v[i] for k, v in rollout.items()}
    return model_i, rollout_i


def _tiny_cfg(**over):
    base = dict(
        horizon=HORIZON,
        n_eval=N,
        generations=1,
        # popsize=N=4 on the 16-dim space is a deliberately small/rank-deficient CMA-ES
        # population: these tests assert record shape/count + reproducibility fields,
        # not search quality, so a tiny population keeps the run fast (one XLA compile).
        popsize=N,
        seed=0,
        failure_threshold=0.5,
        n_families=2,
        thresholds=dict(_THRESHOLDS),
        atlas={"n_ttf_bins": 2, "iterations": 1, "batch_size": N, "sigma": 0.15},
    )
    base.update(over)
    return SearchConfig(**base)


# (a) apply_thetas: batched model fields have leading dim N; rollout dict has the
#     space's rollout dims, each shape (N,).
def test_apply_thetas_shapes(fx):
    thetas = _thetas(fx, seed=0)
    batched, _, rollout = apply_thetas(fx.target, fx.space, fx.base_model, thetas, fx.consts)
    for f in fx.target.model_fields:
        leaf = getattr(batched, f)
        assert leaf.shape[0] == N, (f, leaf.shape)
    rollout_dims = [n for n in fx.space.names() if fx.space.dims[n].kind == "rollout"]
    assert set(rollout.keys()) == set(rollout_dims)
    for k, v in rollout.items():
        assert v.shape == (N,), (k, v.shape)


# (b) evaluator contract: dict with the 5 contract keys + shapes (N, H, ...).
def test_evaluator_contract(fx):
    thetas = _thetas(fx, seed=1)
    traj = fx.evaluate(thetas, jax.random.PRNGKey(0))
    for k in _CONTRACT_KEYS:
        assert k in traj, k
    assert traj["trunk_height"].shape == (N, HORIZON)
    assert traj["up_proj"].shape == (N, HORIZON)
    assert traj["lin_vel"].shape == (N, HORIZON, 3)
    assert traj["ang_vel"].shape == (N, HORIZON, 3)
    assert traj["command"].shape == (N, HORIZON, 3)


# (c) CMA-ES one generation -> FailureRecords with batch_key + env_index set.
def test_cmaes_one_generation(fx):
    cfg = _tiny_cfg()
    # The `key` arg here is vestigial -- cmaes_search derives its evosax/eval streams
    # from cfg.seed -- but it's part of the signature, so pass a throwaway PRNGKey.
    recs = cmaes_search(fx.evaluate, fx.space, cfg, jax.random.PRNGKey(0), seed_theta=None)
    assert len(recs) == cfg.popsize
    for r in recs:
        assert r.batch_key is not None and len(r.batch_key) == 2
        assert r.env_index is not None


# (d) MAP-Elites one iteration -> FailureRecords with batch_key + env_index set.
def test_map_elites_one_iteration(fx):
    cfg = _tiny_cfg()
    recs = map_elites_search(fx.evaluate, fx.space, cfg, jax.random.PRNGKey(0))
    assert len(recs) >= 1
    for r in recs:
        assert r.batch_key is not None and len(r.batch_key) == 2
        assert r.env_index is not None


# (e) #6 determinism invariant: vmapped eval row i == a direct rollout_one rebuilt
#     from the same batch key via env_key(key, i) on the squeezed row-i model.
def test_determinism_vmap_row_equals_direct_rollout(fx):
    thetas = _thetas(fx, seed=2)
    key = jax.random.PRNGKey(7)
    traj = fx.evaluate(thetas, key)
    i = N - 2
    model_i, rollout_i = _row_model(fx, thetas, i)
    # T7: rollout_one takes the target's summary hook (bound to consts), not torso_body_id.
    summarize = lambda st: fx.target.objective.summarize(st, fx.consts)  # noqa: E731
    summary_i = rollout_one(
        fx.env, fx.policy_fn, model_i, rollout_i, env_key(key, i), HORIZON,
        summarize, collect_states=False,
    )
    for k in _CONTRACT_KEYS:
        # The two paths share rollout_one / keys / model, so they are algebraically
        # identical -- BUT one runs in a jit-vmap BATCHED lane and the other eager
        # SINGLE-lane, and XLA compiles those with different float op orderings / FMA
        # fusion (GPU-dependent). That reassociation compounds over the MJX contact
        # steps to ~1e-3 (observed max abs ~8e-4, rel ~1% on newer jaxlib/T4). This is
        # NOT a correctness property: the determinism that matters -- identical inputs on
        # the SAME lane -> bit-identical -- is asserted at rtol=1e-6 in
        # test_state_stacking_and_bit_exact_repro below. Here we only guard that the two
        # lanes describe the SAME trajectory, so the tolerance is set well above the
        # reassociation floor yet far below a real key/model bug (which diverges
        # chaotically to O(100%) within a few steps, caught easily). Observed floors:
        # ~1% rel on T4/jaxlib-0.8.3 GPU, ~5% rel on CPU jax (same version) -- contact
        # dynamics amplify reassociation differently per backend, hence the margin.
        np.testing.assert_allclose(
            np.asarray(summary_i[k]), np.asarray(traj[k][i]),
            rtol=8e-2, atol=2e-2,
            err_msg=f"vmapped eval row {i} != direct rollout_one for {k!r}",
        )


# (f) state-stacking + bit-exact repro: collect_states=True returns (summary, states)
#     whose summary equals the collect_states=False summary for the SAME key (state
#     collection does not perturb numerics), with len(states) == horizon + 1.
def test_state_stacking_and_bit_exact_repro(fx):
    thetas = _thetas(fx, seed=3)
    model_i, rollout_i = _row_model(fx, thetas, 0)
    # T7: rollout_one takes the target's summary hook (bound to consts), not torso_body_id.
    summarize = lambda st: fx.target.objective.summarize(st, fx.consts)  # noqa: E731
    key = env_key(jax.random.PRNGKey(11), 0)
    summary = rollout_one(
        fx.env, fx.policy_fn, model_i, rollout_i, key, HORIZON, summarize,
        collect_states=False,
    )
    summary2, states = rollout_one(
        fx.env, fx.policy_fn, model_i, rollout_i, key, HORIZON, summarize,
        collect_states=True,
    )
    for k in _CONTRACT_KEYS:
        # These two summaries are expected to be genuinely identical: it's the same
        # ops on the same scan -- collect_states only adds an extra (discarded) scan
        # output -- so the small tolerance here is just a safety margin, not slack.
        np.testing.assert_allclose(
            np.asarray(summary2[k]), np.asarray(summary[k]),
            rtol=1e-6, atol=1e-6,
            err_msg=f"collect_states perturbed {k!r}",
        )
    assert len(states) == HORIZON + 1


def _push_family(fx):
    """A fake discovered failure that concentrates on a rollout-dim (push + obs-noise)
    region, for the hardening-wrapper smoke."""
    from breakpoint_mjx.schemas import FailureFamily

    tp = {n: fx.space.dims[n].nominal for n in fx.space.names()}
    tp["push_mag"] = 1.5
    tp["obs_noise_scale"] = 0.08
    tp["latency_steps"] = 3.0
    return FailureFamily(
        id="f0", name="push+noise", member_ids=["m0"], representative_id="m0",
        triggering_parameters=tp, mean_failure_score=0.6, mean_transfer_score=0.5,
        priority_rank=1, validity=None,
    )


# (g) full-space hardening WRAPPER: the perturbation training wrapper (hardening_env)
#     resets/steps on MJX, preserves obs structure, threads its per-episode adversary
#     state under info['bp_harden'], vmaps over N (the shape brax training uses), and is
#     deterministic given a key. This is the Colab verification the CPU box cannot do.
def test_perturbation_wrapper_reset_step_vmap(fx):
    # hardening_env is commercial-edition only; skip in the open-source engine.
    make_perturbation_wrapper = pytest.importorskip(
        "breakpoint_mjx.hardening_env").make_perturbation_wrapper
    from breakpoint_mjx.rollout import _obs_state

    wenv = make_perturbation_wrapper(
        fx.env, fx.space, [_push_family(fx)], 1.0,
        episode_length=HORIZON, n_pool=64, seed=0,
    )
    assert wenv is not fx.env                    # go1 has rollout dims -> genuinely wrapped
    assert wenv.action_size == fx.env.action_size  # delegation intact

    key = jax.random.PRNGKey(0)
    st = wenv.reset(key)
    assert "bp_harden" in st.info
    base_shape = _obs_state(fx.env.reset(key).obs).shape
    assert _obs_state(st.obs).shape == base_shape          # obs structure preserved

    act = jnp.zeros((fx.env.action_size,))
    st2 = wenv.step(st, act)
    assert _obs_state(st2.obs).shape == base_shape
    assert int(st2.info["bp_harden"]["step"]) == 1         # counter advanced

    # Determinism: same key + same action -> identical perturbed obs (byte-repro like eval).
    a = wenv.step(wenv.reset(key), act)
    b = wenv.step(wenv.reset(key), act)
    np.testing.assert_allclose(
        np.asarray(_obs_state(a.obs)), np.asarray(_obs_state(b.obs)),
        rtol=1e-6, atol=1e-6, err_msg="perturbation wrapper is nondeterministic given a key",
    )

    # Vmapped reset over N keys (the training shape): leading dim N on the obs + adversary.
    batched = jax.vmap(wenv.reset)(jax.random.split(key, N))
    assert _obs_state(batched.obs).shape[0] == N
    assert batched.info["bp_harden"]["vec"].shape[0] == N


# (h2) surrogate-guided ACTIVE search: one cold-start round on GPU -> exactly popsize
#      records with reproducibility fields (the propose -> evaluate -> records wiring;
#      the acquisition math itself is CPU-tested in test_active_search.py).
def test_active_search_one_round(fx):
    from breakpoint_mjx.active_search import active_search

    cfg = _tiny_cfg()
    recs = active_search(fx.evaluate, fx.space, cfg, jax.random.PRNGKey(0),
                         rounds=1, pool=256, ensemble_k=2)
    assert len(recs) == cfg.popsize
    for r in recs:
        assert r.batch_key is not None and len(r.batch_key) == 2
        assert r.env_index is not None


# (h) LLM-guided search closed loop on GPU with the RANDOM fallback (no API key): one
#     round -> exactly popsize FailureRecords with reproducibility fields, exercising the
#     propose -> build_batch -> evaluate -> score -> records path end-to-end.
def test_llm_guided_search_random_fallback(fx, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from breakpoint_mjx.llm_search import llm_guided_search

    cfg = _tiny_cfg()
    recs = llm_guided_search(
        fx.evaluate, fx.space, cfg, jax.random.PRNGKey(0),
        rounds=1, n_proposals=3, robot_name="go1",
    )
    assert len(recs) == cfg.popsize
    for r in recs:
        assert r.batch_key is not None and len(r.batch_key) == 2
        assert r.env_index is not None
