"""End-to-end BreakPoint: search -> rank+cluster -> render top of each family ->
static report. Tests a frozen (already-trained) policy; GPU recommended.

  breakpoint run --robot go1 --policy onnx:./policy.onnx
  python -m breakpoint_mjx.pipeline [--target go1]

NOTE: the per-env key scheme changed (Upgrade #6: row i = fold_in(batch_key, i)),
so any failures.json written before this change is INVALID -- delete it and re-run.
NOTE: T6 added richer failure axes (obs_dropout_prob / obs_bias_scale / push_mag /
push_step_frac rollout dims + floor_slope_rad model dim), so the theta dimension D
changed too; any failures.json / report.html / *.mp4 from before T6 is likewise INVALID
(theta vectors have a different length) -- delete them and re-search from scratch."""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

# GPU XLA flags must be set BEFORE jax is imported (no-op on CPU).
from breakpoint_mjx.accel import tune_for_gpu  # noqa: E402
tune_for_gpu()

import jax
import jax.numpy as jnp
import numpy as np

from breakpoint_mjx.clustering import build_families
from breakpoint_mjx.crossval_engines import replay_stage
from breakpoint_mjx.config import (
    load_search_config,
    load_theta_space,
    merge_target_thresholds,
)
from breakpoint_mjx.evaluator import make_evaluator
from breakpoint_mjx.explain import explain_failure
from breakpoint_mjx.policy import make_policy_fn
from breakpoint_mjx.render import render_failure
from breakpoint_mjx.report import write_report
from breakpoint_mjx.schemas import BreakPointReport
from breakpoint_mjx.llm_seed import propose_seed
from breakpoint_mjx.search import cmaes_search, load_failures, random_search, save_failures
from breakpoint_mjx.targets import get_target
from breakpoint_mjx.paths import config_path



# Contact model of the SEARCH scene, stated in every report (physics-fidelity disclosure).
SEARCH_COLLISION_MODELS = {
    "Go1JoystickFlatTerrain": (
        "feet-only (Playground scene_mjx_feetonly_flat_terrain: only the four foot "
        "spheres collide with the floor, so a fallen trunk can sink through it)"),
}


def _top20w(records):
    ws = np.sort(
        np.array([r.failure_score * (r.transfer_score or 0.0) for r in records])
    )[::-1][:20]
    return float(ws.mean()) if len(ws) else 0.0


def _obs_state_dim(env):
    """Length of env obs['state'] whether observation_size is a dict or a scalar/tuple."""
    def _flat(v):
        return int(v[-1]) if isinstance(v, (tuple, list)) else int(v)

    obs_size = env.observation_size
    return _flat(obs_size["state"]) if isinstance(obs_size, dict) else _flat(obs_size)


def _maybe_adapter_policy(env, args, target=None):
    """Build a BYO adapter policy_fn from ``--policy`` / ``BP_POLICY``, else return None.

    A spec is ``kind:path`` ONLY when ``kind`` is one of ``check_policy.ADAPTER_KINDS``
    (onnx / params / npz / weights). Anything else, including a Windows path such as
    ``C:/runs/go1_policy.pkl`` (whose drive letter looks like a ``kind:`` prefix) or a
    ``brax:``-prefixed path, is a brax params pickle rebuilt for ``target``, unless a
    contract is set (``BP_OBS_SPEC``): then the pickle is weight-imported by
    ``adapters.build_policy`` so the contract's obs/action map applies on top.

    Weight-IMPORTS a foreign MLP policy (ONNX or numpy params) into a JAX forward that runs
    INSIDE the evaluator's ``vmap`` exactly like the brax policy_fn -- the only supported way
    to bring a non-brax policy, since a raw ONNX/PyTorch callable can't run in a jax trace.

    The user must match their exported policy's obs normalization (``--obs-norm`` npz with
    ``mean``/``std``) and action post-processing (``--action-scale`` / ``--squash`` /
    ``--mean-dim``); a mismatch yields wrong actions, not an error, so the two shape checks
    below are the only automatic guards. Every option also has a ``BP_*`` env fallback so the
    pipeline runs with a single ``BP_POLICY=... python scripts/run_pipeline.py``.
    """
    spec = args.policy or os.environ.get("BP_POLICY")
    if not spec:
        return None
    from breakpoint_mjx.check_policy import is_adapter_spec

    # A contract (BP_OBS_SPEC) needs the weight-imported forward, so a brax .pkl with a
    # contract goes through build_policy below; without one it is rebuilt natively.
    if not is_adapter_spec(spec) and not os.environ.get("BP_OBS_SPEC"):
        import pickle

        path = spec[5:] if spec.lower().startswith("brax:") else spec
        if not os.path.exists(path):
            raise SystemExit(f"--policy: no brax params pickle at {path} (adapter specs "
                             f"look like onnx:/path/model.onnx or params:/path/w.npz)")
        with open(path, "rb") as fh:
            params = pickle.load(fh)
        print(f"using brax params pickle: {path}")
        return make_policy_fn(params=params, target=target)
    from breakpoint_mjx.adapters import build_policy

    def opt(cli, envvar, cast=str, default=None):
        if cli is not None:
            return cli
        v = os.environ.get(envvar)
        return cast(v) if v is not None else default

    mlp = build_policy(
        spec,
        obs_norm_path=opt(args.obs_norm, "BP_OBS_NORM"),
        activation=opt(args.activation, "BP_ACTIVATION"),
        action_scale=opt(args.action_scale, "BP_ACTION_SCALE", float),
        action_bias=opt(args.action_bias, "BP_ACTION_BIAS", float),
        squash=bool(args.squash) or os.environ.get("BP_SQUASH") == "1",
        weight_layout=opt(args.weight_layout, "BP_WEIGHT_LAYOUT", default="out_in"),
        mean_dim=opt(args.mean_dim, "BP_MEAN_DIM", int),
    )

    # The evaluator's latency buffer is sized to env.action_size, so a wrong-width action
    # breaks the rollout (not just the scores) -- fail fast with a clear message.
    if mlp.action_size != env.action_size:
        raise SystemExit(
            f"BYO policy action dim {mlp.action_size} != env.action_size "
            f"{env.action_size}; check the export (or --mean-dim for a [mean, log_std] head)"
        )
    state_dim = _obs_state_dim(env)
    if mlp.obs_spec.indices is None and mlp.in_dim != state_dim:
        raise SystemExit(
            f"BYO policy expects obs dim {mlp.in_dim} but env obs['state'] is {state_dim}; "
            f"supply an obs index remap or point --obs-norm/--policy at the matching export"
        )
    print(
        f"using BYO adapter policy: {spec} "
        f"(in={mlp.in_dim}, act={mlp.action_size}, activation={mlp.activation})"
    )
    return mlp.policy_fn


def main(argv=None):
    try:  # progress lines must reach a piped log (Colab / CI) as they happen
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001 -- a replaced stdout may not support it
        pass
    ap = argparse.ArgumentParser(description="BreakPoint end-to-end pipeline.")
    ap.add_argument("--target", default="go1", help="registered target name (default: go1)")
    ap.add_argument("--urdf", default=None,
                    help="path to a ROS2/URDF robot description; builds the target from "
                         "the model itself instead of a registered Playground env. The "
                         "run dir is namespaced by the URDF stem, and --target is ignored.")
    # BYO (bring-your-own) non-brax policy adapter (T4). Default (no --policy/BP_POLICY) keeps
    # the brax policy_fn unchanged. See adapters/ + the README "Bring your own policy" section.
    ap.add_argument(
        "--policy", default=None,
        help="BYO policy adapter: 'onnx:/path/model.onnx' or 'params:/path/weights.npz', "
             "or a plain path to a brax params .pkl (also BP_POLICY). Omit to use "
             "<BP_OUT>/<target>/<checkpoint> trained for the target.",
    )
    ap.add_argument(
        "--obs-norm", default=None,
        help="Path to an .npz with 'mean'/'std' for the BYO policy's obs normalization "
             "(also BP_OBS_NORM).",
    )
    ap.add_argument(
        "--activation", default=None,
        help="BYO MLP hidden activation tanh|relu|elu; ONNX auto-detects (also BP_ACTIVATION).",
    )
    ap.add_argument(
        "--action-scale", default=None, type=float,
        help="Scalar action scale applied to the BYO policy output (also BP_ACTION_SCALE).",
    )
    ap.add_argument(
        "--action-bias", default=None, type=float,
        help="Scalar action bias added after scaling: action = scale*f(mean) + bias "
             "(also BP_ACTION_BIAS).",
    )
    ap.add_argument(
        "--squash", action="store_true",
        help="tanh-squash the BYO policy output before scaling (also BP_SQUASH=1).",
    )
    ap.add_argument(
        "--weight-layout", default=None, choices=["in_out", "out_in"],
        help="params: weight orientation; torch=out_in (default), flax=in_out "
             "(also BP_WEIGHT_LAYOUT).",
    )
    ap.add_argument(
        "--mean-dim", default=None, type=int,
        help="If the BYO head emits [mean, log_std], keep the first --mean-dim entries "
             "(also BP_MEAN_DIM).",
    )
    args = ap.parse_args(argv)

    if args.urdf:
        # BYO robot: derive the whole target (model, theta space, thresholds,
        # constants) from the URDF. The derived theta space is written next to the
        # run so the exact bounds a run used stay with its artifacts.
        from breakpoint_mjx.urdf_target import format_report, make_urdf_target

        urdf = Path(args.urdf).resolve()
        name = urdf.stem
        space_out = Path(os.environ.get("BP_OUT", "out")) / name / "theta_space.yaml"
        space_out.parent.mkdir(parents=True, exist_ok=True)
        target = make_urdf_target(name, str(urdf), theta_space_path=str(space_out))
        print(format_report(getattr(target.load_env, "report", {})), flush=True)
    else:
        target = get_target(args.target)
    # Namespace the run dir by target so go1 and barkour runs don't collide
    # (shared failures.json / fam-*.mp4 / report.html / checkpoints). BP_OUT stays
    # the parent root; every artifact for this run lives under <BP_OUT>/<target>/.
    OUT = os.path.join(os.environ.get("BP_OUT", "out"), target.name)
    os.makedirs(OUT, exist_ok=True)
    cfg = load_search_config(config_path("search.yaml"))
    # Apply per-target failure-threshold overrides (e.g. a humanoid's taller fall
    # height) ON TOP of the search.yaml defaults. SearchConfig is frozen, so this
    # returns a replaced copy; doing it here means every downstream consumer of
    # cfg.thresholds (scores_from_trajectories / atlas / behavior_descriptor) sees the
    # per-robot values. go1/barkour declare no overrides, so cfg is unchanged for them.
    cfg = merge_target_thresholds(cfg, target)
    # Apply the per-target rollout horizon override (env steps) ON TOP of the search.yaml
    # default, the same replaced-copy pattern as the thresholds merge. A target trained on a
    # shorter episode (e.g. PandaPickCube's 150 vs the 500-step locomotion default) sets
    # target.horizon so the rollout length + the time-to-failure normalization match its
    # training; every downstream consumer (evaluator / scoring / margins / crossval / render)
    # reads cfg.horizon, so setting it once here propagates everywhere. Legged targets
    # (horizon None) are unchanged.
    if target.horizon is not None:
        cfg = dataclasses.replace(cfg, horizon=int(target.horizon))
    space = load_theta_space(config_path(target.theta_space_path))
    env = target.load_env()
    # BYO adapter policy (T4) if --policy/BP_POLICY is set; else the brax default (unchanged).
    policy_fn = _maybe_adapter_policy(env, args, target=target)
    if policy_fn is None:
        policy_fn = make_policy_fn(out_dir=OUT, target=target)
    consts = target.resolve_constants(env)
    base_model = env.mjx_model  # capture ORIGINAL before evaluator/renders mutate env._mjx_model
    evaluate = make_evaluator(env, policy_fn, space, cfg.horizon, target, consts)

    fpath = os.path.join(OUT, "failures.json")
    coverage = None  # set only by BP_SEARCH=multi + BP_COVERAGE=1 on a fresh search
    if os.path.exists(fpath):
        print(f"loading existing {fpath} (delete it to force a fresh search)")
        cpath = os.path.join(OUT, "coverage.json")
        if os.path.exists(cpath):
            with open(cpath) as _fh:
                coverage = json.load(_fh)
        all_records = load_failures(fpath)
        rr = [r for r in all_records if (r.id or "").startswith("random")]
        # "directed" = any non-random search (cmaes / llm / atlas / active) so the
        # comparison stays meaningful when reloading a saved directed-mode run.
        cr = [r for r in all_records
              if (r.id or "").startswith(("cmaes", "llm", "atlas", "active", "agentic"))]
    else:
        key = jax.random.PRNGKey(cfg.seed)
        _, rk, ck = jax.random.split(key, 3)
        random_cfg = dataclasses.replace(cfg, n_eval=cfg.popsize)
        # T7: the ACTIVE target's failure objective (legged by default) drives scoring in
        # every search mode + margins, so a manipulation target scores against task
        # success / dropped object / joint-limit instead of "stay upright".
        objective = target.objective
        print("random search ...")
        rr = random_search(evaluate, space, random_cfg, rk, objective=objective)
        search_mode = os.environ.get("BP_SEARCH", "cmaes").lower()
        if search_mode == "multi":
            from breakpoint_mjx.multi_search import multi_cmaes_search
            restarts = int(os.environ.get("BP_RESTARTS", "16"))
            # Seed the FIRST restart with the LLM's proposal, exactly as the cmaes
            # branch does. That is what makes multi-start provably no worse than the
            # single-start run it replaces: restart 0 is the informed start and the
            # other 15 explore. (This was computed only in the cmaes branch before,
            # so multi crashed on an unbound name the moment it ran.)
            seed_theta = propose_seed(space, robot_name=target.env_name)
            if seed_theta:
                print("LLM seed:", {k: round(v, 3) for k, v in seed_theta.items()})
            else:
                print("LLM seed unavailable; restart 0 starts from box center")
            if os.environ.get("BP_COVERAGE") == "1":
                # Campaign coverage: BP_COVERAGE_WAVES waves of `restarts` independent
                # restarts, then a uniform Sobol residual audit of BP_COVERAGE_AUDIT worlds.
                # No early stop by default: every stop rule failed the synthetic
                # false-complete bar (BP_COVERAGE_RULE=chao2|brk|chao1 opts in anyway). Extra GPU cost: the extra waves, the
                # audit batch and the segment probes (see docs/coverage.md).
                from breakpoint_mjx.coverage import coverage_sentence, run_until_saturated
                print(f"BP_COVERAGE=1: waves of {restarts} restarts until saturated ...")
                cr, coverage = run_until_saturated(
                    evaluate, space, cfg, ck, wave=restarts,
                    max_waves=int(os.environ.get("BP_COVERAGE_WAVES", "4")),
                    rule=os.environ.get("BP_COVERAGE_RULE", "none"),
                    objective=objective, seed_theta=seed_theta,
                    n_audit=int(os.environ.get("BP_COVERAGE_AUDIT", "2048")))
                print("coverage:", coverage_sentence(coverage))
            else:
                print(f"multi-restart CMA-ES ({restarts} parallel restarts, one batched "
                      f"eval per generation) ...")
                cr = multi_cmaes_search(evaluate, space, cfg, ck, restarts=restarts,
                                        seed_theta=seed_theta, objective=objective)
        elif search_mode == "agentic":
            from breakpoint_mjx.agentic_search import agentic_search
            print("agentic multi-proposer search ...")
            cr = agentic_search(evaluate, space, cfg, ck, robot_name=target.env_name,
                                objective=objective)
        elif search_mode == "llm":
            from breakpoint_mjx.llm_search import llm_guided_search
            print("LLM-guided search ...")
            cr = llm_guided_search(evaluate, space, cfg, ck, robot_name=target.env_name,
                                   objective=objective)
        elif search_mode == "atlas":
            from breakpoint_mjx.atlas import map_elites_search
            print("MAP-Elites failure atlas ...")
            cr = map_elites_search(evaluate, space, cfg, ck, objective=objective)
        elif search_mode == "active":
            from breakpoint_mjx.active_search import active_search, transferable_seed_records
            print("surrogate-guided active search ...")
            # The already-evaluated random records warm-start the ensemble for free
            # (they inform the surrogate but active_search does not return them).
            seed_recs = list(rr)
            # Cross-policy warm start: BP_PRIOR=path/to/failures.json (an earlier policy
            # version's run) additionally seeds the surrogate -- name-mapped into this
            # space, clipped to bounds, non-transferable records dropped.
            prior_path = os.environ.get("BP_PRIOR")
            if prior_path:
                if os.path.exists(prior_path):
                    # A malformed / old-schema prior must NOT abort the run (the search
                    # has already spent GPU) -- the warm-start is an optional bonus, so
                    # print-and-skip on any load/parse error, same as the not-found branch.
                    try:
                        prior = load_failures(prior_path)
                        xfer = transferable_seed_records(prior, space)
                        print(f"BP_PRIOR: warm-starting surrogate from {len(xfer)}/{len(prior)} "
                              f"transferable prior records ({prior_path})")
                        seed_recs += xfer
                    except Exception as _e:
                        print(f"BP_PRIOR unreadable ({type(_e).__name__}: {_e}); continuing "
                              f"without warm-start: {prior_path}")
                else:
                    print(f"BP_PRIOR set but not found, ignoring: {prior_path}")
            cr = active_search(evaluate, space, cfg, ck, objective=objective,
                               seed_records=seed_recs)
        else:
            seed_theta = propose_seed(space, robot_name=target.env_name)
            if seed_theta:
                print("LLM seed:", {k: round(v, 3) for k, v in seed_theta.items()})
            else:
                print("LLM seed unavailable; CMA-ES starts from box center")
            cr = cmaes_search(evaluate, space, cfg, ck, seed_theta=seed_theta,
                              objective=objective)
        all_records = cr + rr
        save_failures(all_records, cfg.failure_threshold, OUT, name="failures.json")
        if coverage is not None:
            with open(os.path.join(OUT, "coverage.json"), "w") as _fh:
                json.dump(coverage, _fh)
    # NOTE: for BP_SEARCH=atlas, cr is the curated best-per-cell elite set (<= grid size),
    # so directed_top20 is over an already-maxed population -- not strictly apples-to-apples
    # with random's raw per-eval scores.
    comparison = {"random_top20": _top20w(rr), "directed_top20": _top20w(cr)}

    failures = [r for r in all_records if r.failure_score >= cfg.failure_threshold]

    # Crash-proof stages (stagecache.py). cluster -> CMC -> replay each checkpoint to
    # <OUT>/stage_<name>.json (atomic), keyed by failures.json + the budget knobs, so a
    # native crash later (render, report) resumes here instead of redoing CMC.
    from breakpoint_mjx import stagecache
    stage_key = stagecache.run_key(fpath, _stage_budget(cfg))
    resumed, stage_d = stagecache.load_latest(OUT, stage_key)
    if resumed:
        print(f"resuming from {resumed} ({stagecache.stage_path(OUT, resumed)}); "
              f"delete the stage_*.json files to recompute", flush=True)
        families = stagecache.families_from_dicts(stage_d["families"])
        print("families:", [(f.priority_rank, f.name, len(f.member_ids)) for f in families])
    else:
        families, _ = build_families(failures, space, cfg.n_families)
        print("families:", [(f.priority_rank, f.name, len(f.member_ids)) for f in families])
        stagecache.save_stage(OUT, "cluster", stage_key, families)
    stagecache.crash_point("cluster")

    # Certified Minimal Cause (breakpoint_mjx/mincause.py). ON by default; BP_MINCAUSE=0
    # disables. For each family representative, batched seed-certified delta debugging toward
    # nominal finds the 1-minimal set of perturbations that still reproduces the failure,
    # checks it is necessary, shrinks its magnitudes to a minimal counterexample, and then
    # RENAMES the family by that cause (directions from the counterexample, never from a
    # failure mean) and merges families whose causes coincide. Runs before validity so
    # BP_VALIDATE annotates the merged families. Cost: roughly 0.5-1.5k rollouts per family
    # at K=8 (BP_MINCAUSE_K), in a few dozen batched evaluate calls.
    #
    # The same stage then runs Cause-Stratified Operational Risk (famrisk.py; OFF by default,
    # BP_FAMRISK=1 enables; the out-of-ODD label is unsafe-direction, docs/famrisk.md Bar 3): each certified cause region's EXACT mass under the declared
    # operating domain times a Clopper-Pearson interval from fresh conditional rollouts
    # (BP_FAMRISK_N per family; smoke 32 / standard 256 / full 1024), and an in-domain vs
    # outside-the-operating-domain label per family. Its summary is checkpointed with the
    # CMC stage (and carried into the replay checkpoint) so a resumed run keeps it.
    famrisk_summary = {}
    if resumed in ("cmc", "replay"):
        famrisk_summary = dict(stage_d["extra"].get("famrisk") or {})
    else:
        from breakpoint_mjx.mincause import make_evaluator_fail_fn, run_stage as _cmc_stage
        families, _cmc_ran = _cmc_stage(
            families, {r.id: r for r in failures}, space,
            lambda: make_evaluator_fail_fn(evaluate, space, cfg.thresholds, cfg.horizon,
                                           cfg.failure_threshold,
                                           jax.random.PRNGKey(cfg.seed + 5_000_011),
                                           objective=target.objective),
            famrisk_out=famrisk_summary)
        if _cmc_ran:
            # Merges leave gaps (#1, #3, #111...): renumber 1..n in the merged priority
            # order, and strip the causal name from any family CMC could not certify.
            from breakpoint_mjx.schemas import uncertified_name
            for i, f in enumerate(sorted(families, key=lambda f: f.priority_rank), 1):
                f.priority_rank = i
                if f.cause_certified is False and not f.name.startswith("uncertified:"):
                    f.name = uncertified_name(f)
            families.sort(key=lambda f: f.priority_rank)
            print("families (CMC):", [(f.priority_rank, f.name, len(f.member_ids))
                                      for f in families])
        if not famrisk_summary:
            famrisk_summary = {"status": "cmc_not_run",
                               "reason": "the certified-minimal-cause stage did not run"}
        stagecache.save_stage(OUT, "cmc", stage_key, families,
                              extra={"cmc_ran": _cmc_ran, "famrisk": famrisk_summary})
    if not famrisk_summary:
        famrisk_summary = {"status": "cmc_not_run",
                           "reason": "the certified-minimal-cause stage did not run"}
    stagecache.crash_point("cmc")

    # Skeptic-grade evidence (crossval_engines.replay_stage). ON by default; BP_REPLAY=0
    # disables. Each certified minimal counterexample (only the cause dims moved off
    # nominal) is replayed on BP_REPLAY_K=8 seeds in C MuJoCo on a FULL-collision variant
    # of the search scene, plus a same-seed nominal twin. A family below the CMC bar there
    # is demoted in the report to "did not reproduce outside MJX" (or inconclusive at this
    # seed budget; BP_REPLAY_MAXMULT pools more seeds like CMC does). CPU only, jitted env:
    # tens of seconds per family. The verdicts are evidence and are checkpointed; the
    # evidence VIDEOS (seed 0, clipped 1 s after the fall) are rendered later, after the
    # report has been written once, so a render crash cannot cost the verdicts.
    if resumed == "replay":
        replay_summary = stage_d["extra"].get("replay_summary") or {}
    else:
        try:
            replay_summary = replay_stage(
                families, env=env, policy_fn=policy_fn, target=target, space=space,
                consts=consts, horizon=cfg.horizon, thresholds=cfg.thresholds,
                threshold=cfg.failure_threshold, out_dir=OUT, seed=cfg.seed,
                base_model=base_model, render=False)
            stagecache.save_stage(OUT, "replay", stage_key, families,
                                  extra={"replay_summary": replay_summary,
                                         "famrisk": famrisk_summary})
        except Exception as e:  # noqa: BLE001 -- e.g. mujoco.FatalError: Could not
            # allocate memory while building the full-collision model on a full box.
            # Report without the replay rather than lose the run; no stage_replay is
            # written, so the next run retries the replay (CMC stays checkpointed).
            for f in families:
                f.replay = None
            replay_summary = {"status": "unavailable",
                              "reason": f"the replay failed on this machine "
                                        f"({type(e).__name__}: {e}); rerun the same "
                                        f"command to retry it (CMC is kept)"}
    stagecache.crash_point("replay")
    if replay_summary.get("status") == "ran":
        print(f"independent replay (C-MuJoCo full-collision, "
              f"{replay_summary['collision_model'].get('source')}): "
              f"{replay_summary['n_reproduced']}/{replay_summary['n_replayed']} families "
              f"reproduced, {replay_summary['n_demoted']} not confirmed outside MJX "
              f"({replay_summary['n_inconclusive']} of those inconclusive at this seed "
              f"budget) "
              f"[{replay_summary['seconds']} s]")
    else:
        print(f"independent replay not run: {replay_summary.get('reason')}")

    # Workstream 1a (T1): cross-config VALIDITY. OFF by default (extra GPU cost); enable
    # with BP_VALIDATE=1. Re-sims each family's representative failure under K independent
    # physics configs (+ an optional independent C-MuJoCo backend via BP_CBACKEND=1) reusing
    # the recorded key (physics varies, seed fixed), then annotates families with MEASURED
    # validity. The report (render_html) is the single place that ORDERS families by
    # validity, so we do not re-sort here. base_model/env/consts were captured above,
    # before evaluate mutated env._mjx_model.
    # Autolytic gain: did the perturbation break the policy, or did the policy break
    # ITSELF? OFF by default (BP_AUTOLYTIC=1) because it re-runs each measured world a
    # second time as an unperturbed twin. Annotates the top-N records with the ratio of
    # excess actuator work to injected energy, which separates "we shoved it over" from
    # "a small input recruited the controller into destroying itself" -- the latter is a
    # controller defect and the far more valuable finding for a customer.
    if os.environ.get("BP_AUTOLYTIC") == "1":
        from breakpoint_mjx.autolytic import measure_records
        from breakpoint_mjx.rollout import env_key, rollout_one
        from breakpoint_mjx.thetas import apply_thetas

        top_n = int(os.environ.get("BP_AUTOLYTIC_TOP", "32"))
        print(f"BP_AUTOLYTIC=1: twin-world gain on the top {top_n} failures ...")
        _summarize = lambda st: target.objective.summarize(st, consts)  # noqa: E731

        def _one_energy(model, rollout_i, k):
            return rollout_one(env, policy_fn, model, rollout_i, k, cfg.horizon,
                               _summarize, collect_energy=True)

        def _evaluate_energy(th, k):
            batched, in_axes, rollout = apply_thetas(target, space, base_model, th, consts)
            keys = jax.vmap(lambda i: env_key(k, i))(jnp.arange(th.shape[0]))
            return jax.vmap(_one_energy,
                            in_axes=(in_axes, {kk: 0 for kk in rollout}, 0))(
                                batched, rollout, keys)

        autolytic_summary = measure_records(
            all_records, jax.jit(_evaluate_energy), space, cfg.thresholds, cfg.horizon,
            top_n=top_n)
        print("autolytic:", json.dumps(autolytic_summary))
        save_failures(all_records, cfg.failure_threshold, OUT, name="failures.json")

    validity_configs = None
    if os.environ.get("BP_VALIDATE") == "1":
        from breakpoint_mjx.crossval import (
            annotate_families,
            default_physics_configs,
            validate_failures,
        )
        print("BP_VALIDATE=1: cross-config validity over family representatives ...")
        configs = default_physics_configs()
        extra_backends = []
        if os.environ.get("BP_CBACKEND") == "1":
            # Independent C-MuJoCo backend (stronger evidence). OPTIONAL: make_mujoco_c_backend
            # returns None unless the backend is READY (mujoco importable AND its reset/obs/
            # command parity ported -- measured in tests/test_mujoco_c_parity.py), so a box
            # without the bindings degrades to MJX-only validity instead of erroring.
            from breakpoint_mjx.backends import make_mujoco_c_backend
            cb = make_mujoco_c_backend(env=env, policy_fn=policy_fn, target=target,
                                       space=space, consts=consts, horizon=cfg.horizon)
            if cb is not None:
                extra_backends.append(cb)
                print("  + independent C-MuJoCo backend enabled")
            else:
                print("  BP_CBACKEND=1 but the C-MuJoCo backend is unavailable "
                      "(mujoco missing, READY off, or a non-legged target); "
                      "using MJX configs only")
        reps = [next(r for r in failures if r.id == f.representative_id) for f in families]
        _, validity_configs = validate_failures(
            target, space, reps, policy_fn, cfg, jax.random.PRNGKey(cfg.seed),
            env=env, consts=consts, base_model=base_model, configs=configs,
            top_n=len(reps), extra_backends=extra_backends,
        )
        annotate_families(families, reps)
        print("validity:", [(f.name, round(f.validity, 3) if f.validity is not None else None)
                            for f in families])

    # Workstream 1c (T2): safety margins + per-axis attribution. OFF by default (extra GPU
    # cost, like BP_VALIDATE); enable with BP_MARGINS=1. Line-searches from the space nominal
    # toward each family representative to find the minimal per-axis perturbation that trips
    # the failure (the boundary/margin string) and finite-differences the failure_score around
    # the representative for per-axis attribution. Scores candidates through the SAME evaluate
    # with a FIXED key (deterministic); all candidates are bounds-clipped. Annotates families.
    if os.environ.get("BP_MARGINS") == "1":
        from breakpoint_mjx.margins import compute_margins
        print("BP_MARGINS=1: safety margins + attribution over family representatives ...")
        records_by_id = {r.id: r for r in failures}
        _cmc_margin = {f.id: f.margin for f in families if f.minimal_cause}
        compute_margins(families, records_by_id, space, cfg, evaluate,
                        jax.random.PRNGKey(cfg.seed), objective=target.objective)
        for f in families:  # keep the CMC cause in front; the single-axis sweep follows it
            if f.id in _cmc_margin:
                f.margin = f"{_cmc_margin[f.id]}. Single-axis sweep: {f.margin}"
        print("margins:", [(f.name, f.margin) for f in families])

    expl = {}
    nominal_named = {n: float(v) for n, v in zip(space.names(), space.nominal_vector())}
    for f in families:
        rep = next(r for r in failures if r.id == f.representative_id)
        e = explain_failure(rep.theta_named, rep.component_signals, rep.transfer_score,
                            robot_name=target.env_name)
        if e:
            expl[f.id] = e

    # mp4_paths is filled in by the render phase at the very end: the report is written
    # once BEFORE any video is rendered, so a native render crash still leaves it.
    mp4s = {}
    report = BreakPointReport(
        run_id="run-0",
        policy_descriptor=target.env_name,
        comparison=comparison,
        families=families,
        mp4_paths=mp4s,
    )
    # Workstream 1b (T2): always-on statistical rigor -- overall failure rate with a Wilson
    # CI over ALL evaluated envs + the severity x probability ranking of families. Pure numpy,
    # cheap, additive (independent of validity/margins). n_total = every evaluated env, so the
    # family member-fractions share the overall failure rate's scale.
    from breakpoint_mjx.stats import compute_report_stats
    stats = compute_report_stats(
        families, n_total=len(all_records), n_failures=len(failures),
        threshold=cfg.failure_threshold, coverage=coverage,
        # without records the physical-validity screen never runs: simulator artifacts
        # stay in every denominator and the report's validity line is blank
        records=all_records,
    )
    # This rate is over the worlds the SEARCH chose (adversarially selected), so it is
    # NOT a deployment failure probability. BP_RISK=1 below produces that one.
    print("failure rate among search-evaluated worlds (adversarially selected): "
          f'{stats["overall"]["rate"]:.3f} '
          f'(95% CI {stats["overall"]["lo"]:.3f}-{stats["overall"]["hi"]:.3f}) '
          f'over {stats["overall"]["n"]} envs')
    # BP_EMBED_VIDEO=0 keeps report.html small by referencing mp4s as external
    # files instead of base64-embedding them (useful once a hosted report is
    # served from object storage rather than emailed around as a single file).
    embed = os.environ.get("BP_EMBED_VIDEO", "1").lower() not in ("0", "false")

    # Machine-readable mirror of the report for the web push CLI (scripts/push_run.py).
    from breakpoint_mjx.webpush import build_families_json, engine_commit_short
    ov = stats["overall"]
    families_payload = build_families_json(
        target=target, space=space, families=families,
        n_total=len(all_records), n_failures=len(failures),
        failure_rate_ci=(ov["rate"], ov["lo"], ov["hi"]),
        search_mode=os.environ.get("BP_SEARCH", "cmaes"),
        horizon=cfg.horizon, engine_commit=engine_commit_short(),
    )
    # Learned layer (M1/M3/M4): train the pure-numpy failure surrogate on this run's labeled
    # records, derive the machine-verified safety envelope, and the cross-policy failure
    # fingerprint. All CPU, seconds, additive; degrades gracefully. Disable with BP_NO_LEARNED=1.
    sur = None  # the learned surrogate, reused by BP_RISK as a control variate if trained
    surrogate_fit = None
    # Nominal pass count (BP_NOMINAL_N seeds at the space nominal, default 16) through
    # the SAME evaluator + scoring, so every "no safe box" / failure-rate line in the
    # report sits next to how the policy does with nothing perturbed. One batched call.
    nominal_check = None
    n_nom = int(os.environ.get("BP_NOMINAL_N", "16"))
    if n_nom > 0:
        try:
            from breakpoint_mjx.mincause import make_evaluator_fail_fn
            _nom_fail = make_evaluator_fail_fn(
                evaluate, space, cfg.thresholds, cfg.horizon, cfg.failure_threshold,
                jax.random.PRNGKey(cfg.seed + 11_000_027), objective=target.objective,
                log=False)  # its own line below; the helper's "CMC:" label would mislead
            _nf = _nom_fail(np.tile(np.asarray(space.nominal_vector(), np.float32),
                                    (n_nom, 1)), np.arange(n_nom))
            nominal_check = {"n": n_nom, "fails": int(np.asarray(_nf).sum())}
            families_payload["nominal_check"] = nominal_check
            print(f"nominal check: {n_nom - nominal_check['fails']}/{n_nom} seeds pass "
                  f"at the space nominal")
        except Exception as _e:  # noqa: BLE001 -- never fail the run over this line
            print("nominal check skipped:", _e)
    if os.environ.get("BP_NO_LEARNED") != "1":
        try:
            from breakpoint_mjx.surrogate import train_from_records
            from breakpoint_mjx.envelope import safety_envelope
            from breakpoint_mjx.embedding import run_embedding
            # The DISPLAYED surrogate + what-if explorer train on ALL records: dense coverage
            # of the failure regions is what the interactive predictor wants.
            sur = train_from_records(all_records, space, epochs=300)
            sur.threshold = float(cfg.failure_threshold)
            # Fit quality is 5-fold CROSS-VALIDATED, never in-sample (a 64x64 MLP
            # memorizes a few hundred points to R2 ~1.0), and hidden with the reason when
            # the labels are too few or too one-sided to mean anything (surrogate.py).
            from breakpoint_mjx.surrogate import r2_report
            surrogate_fit = r2_report(all_records, space, cfg.failure_threshold, epochs=150)
            families_payload["surrogate"] = sur.to_dict()
            families_payload["surrogate_r2"] = surrogate_fit["r2"]
            families_payload["surrogate_fit"] = surrogate_fit
            # The safety ENVELOPE is a CERTIFICATE, so its surrogate must be trained on a
            # REPRESENTATIVE sample. A failure-concentrated search (cmaes/atlas/active/llm)
            # oversamples failures, biasing a surrogate toward "unsafe everywhere" and
            # collapsing the envelope to the nominal point. Train the envelope's surrogate on
            # the UNIFORM random-search records only (rr); fall back to all records if too
            # few uniform points exist to fit.
            uniform_records = [r for r in all_records if (r.id or "").startswith("random")]
            env_records = uniform_records if len(uniform_records) >= 64 else all_records
            env_sur = train_from_records(env_records, space, epochs=300)
            env_sur.threshold = float(cfg.failure_threshold)
            families_payload["envelope"] = safety_envelope(
                env_sur, space, tol=cfg.failure_threshold)
            families_payload["envelope"]["surrogate_sample"] = (
                "uniform" if env_records is uniform_records else "all")
            from breakpoint_mjx.envelope import annotate_label_balance
            n_env_fail = sum(1 for r in env_records
                             if r.failure_score >= cfg.failure_threshold)
            annotate_label_balance(families_payload["envelope"], len(env_records),
                                   n_env_fail, nominal_check=nominal_check)
            families_payload["embedding"] = run_embedding(families, space, ov["rate"]).tolist()
            fit_txt = (f"{surrogate_fit['method']} R2={surrogate_fit['r2']:.3f}"
                       if surrogate_fit["r2"] is not None else surrogate_fit["hidden_reason"])
            print(f"trained failure surrogate ({fit_txt}) + safety envelope: "
                  f'{families_payload["envelope"]["summary_long"]}')
        except Exception as _e:  # never fail the run over the learned layer
            print("learned layer (surrogate/envelope/embedding) skipped:", _e)

    # Coverage certificate goes in OUTSIDE the learned-layer try/except above, so a failure here
    # is loud instead of swallowed.
    if coverage is not None:
        families_payload["coverage"] = coverage

    # BP_RISK=1: operational failure probability under a DECLARED deployment distribution
    # (default: uniform on each ThetaDim's plausible band) with an anytime-valid 95% CI,
    # from FRESH rollouts drawn from a defensive mixture seeded by this run's failure
    # modes; the learned surrogate (if any) is only a control variate. Also certifies the M3
    # envelope box with simulator rollouts. Costs up to BP_RISK_N extra rollouts.
    operational_risk = None
    if os.environ.get("BP_RISK") == "1":
        try:
            from breakpoint_mjx.riskcert import run_stage
            from breakpoint_mjx.search import scores_from_trajectories
            _risk_key = [jax.random.PRNGKey(cfg.seed + 7_000_003)]

            def evaluate_fail(thetas_np):
                _risk_key[0], ek = jax.random.split(_risk_key[0])
                traj = evaluate(jnp.asarray(thetas_np, dtype=jnp.float32), ek)
                sc, _, _ = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon,
                                                    target.objective)
                return np.asarray(sc) >= cfg.failure_threshold

            dist_block = None
            if os.environ.get("BP_RISK_DIST"):
                import yaml as _yaml
                with open(os.environ["BP_RISK_DIST"]) as _fh:
                    dist_block = (_yaml.safe_load(_fh) or {}).get("operational")
            tol = os.environ.get("BP_RISK_TOL", "0.005")
            risk = run_stage(
                evaluate_fail, space, families, all_records, cfg.failure_threshold,
                surrogate=sur, envelope=families_payload.get("envelope"),
                dist_block=dist_block,
                n_max=int(os.environ.get("BP_RISK_N", "8192")),
                batch=int(os.environ.get("BP_RISK_BATCH", "512")),
                target_halfwidth=(float(tol) if tol not in ("", "0") else None),
                alpha=float(os.environ.get("BP_RISK_ALPHA", "0.3")), seed=cfg.seed,
                mc_reference_n=int(os.environ.get("BP_RISK_MC_N", "0")))
            operational_risk = risk["operational_risk"]
            families_payload["operational_risk"] = operational_risk
            if "envelope_certificate" in risk and families_payload.get("envelope"):
                families_payload["envelope"]["simulator_certificate"] =                     risk["envelope_certificate"]
            o = operational_risk
            print(f"operational failure probability (declared p, SIMULATION): "
                  f'{o["estimate"]:.4f} (95% anytime CI {o["lo"]:.4f}-{o["hi"]:.4f}) '
                  f'from {o["n_rollouts"]} fresh rollouts; plain-MC baseline on the '
                  f'{o["plain_mc_baseline"]["n"]} p-draws: '
                  f'{o["plain_mc_baseline"]["lo"]:.4f}-{o["plain_mc_baseline"]["hi"]:.4f}')
        except Exception as _e:  # never fail the run over the risk stage
            print("operational risk stage (BP_RISK) skipped:", _e)

    # Per-family operational risk (famrisk.py) vs the whole-policy BP_RISK total.
    from breakpoint_mjx.famrisk import residual as _famrisk_residual
    _res = _famrisk_residual(famrisk_summary, operational_risk)
    if _res is not None:
        famrisk_summary["residual"] = _res
    families_payload["family_operational_risk"] = famrisk_summary

    evidence = {
        "search_collision_model": SEARCH_COLLISION_MODELS.get(target.env_name),
        "replay": replay_summary,
        "surrogate_fit": surrogate_fit,
        "nominal_check": nominal_check,
    }
    families_payload["evidence"] = evidence
    report_path = os.path.join(OUT, "report.html")
    families_path = os.path.join(OUT, "families.json")

    def _emit(video_state, note=None):
        """Write report.html + families.json atomically. ``video_state`` is recorded in
        families.json ("rendering", "skipped_lowmem", "done", "partial"); ``note`` is
        appended to the explanation line of every family without a video, so the page
        says why the slot is empty (report.py is unchanged; the state travels through
        ``explanations``)."""
        from breakpoint_mjx.schemas import family_status
        for fd, f in zip(families_payload["families"], families):
            fd["status"] = family_status(f)
            fd["replay"] = f.replay
            fd["evidence_videos"] = f.evidence_videos
            fd["mjx_min_floor_height"] = f.mjx_min_floor_height
        families_payload["videos"] = {"state": video_state, "rendered": sorted(mp4s)}
        ex = dict(expl)
        if note:
            for f in families:
                if f.id not in mp4s:
                    ex[f.id] = (ex.get(f.id, "") + " " + note).strip()
        tmp = os.path.join(OUT, ".report.tmp.html")
        write_report(report, tmp, explanations=ex,
                     embed=embed, validity_configs=validity_configs, stats=stats,
                     operational_risk=operational_risk,
                     envelope=families_payload.get("envelope"), evidence=evidence,
                     nominal=nominal_named, famrisk=famrisk_summary)
        stagecache.replace_atomic(tmp, report_path)
        stagecache.write_json_atomic(families_path, families_payload)

    lowmem, lowmem_why = stagecache.lowmem_enabled()
    if lowmem:
        _emit("skipped_lowmem", f"[video: not rendered, low-memory mode ({lowmem_why})]")
        print(f"low-memory mode ({lowmem_why}): mp4 rendering skipped")
    else:
        _emit("rendering", "[video: rendering...]")
        print("wrote", report_path, "(evidence complete; videos rendering next)")
    stagecache.crash_point("render")

    if not lowmem:
        failed = _render_videos(
            families, failures, mp4s, env=env, policy_fn=policy_fn, target=target,
            space=space, consts=consts, cfg=cfg, base_model=base_model, out_dir=OUT)
        _emit("partial" if failed else "done",
              "[video: render failed on this machine; the evidence above is complete]"
              if failed else None)
    print("wrote", report_path)
    print("wrote", families_path)


def _stage_budget(cfg):
    """Everything besides failures.json that changes what cluster/CMC/replay compute."""
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("BP_MINCAUSE", "BP_REPLAY", "BP_FAMRISK"))}
    return {"budget": os.environ.get("BP_BUDGET"), "n_families": cfg.n_families,
            "seed": cfg.seed, "horizon": cfg.horizon,
            "failure_threshold": cfg.failure_threshold,
            "policy": os.environ.get("BP_POLICY"), "obs_spec": os.environ.get("BP_OBS_SPEC"),
            "env": env}


def _render_videos(families, failures, mp4s, *, env, policy_fn, target, space, consts, cfg,
                   base_model, out_dir):
    """Render phase, strictly after the report exists. One family at a time: the XLA
    compile cache is dropped once before the first render and Python garbage before each
    (per-family cache clearing is opt-in, ``BP_RENDER_CLEAR=1``), since this is where a
    CPU box runs out of memory. A Python-level failure (MemoryError, renderer missing)
    skips that family's video and is counted; returns the number of failures."""
    import copy

    from breakpoint_mjx.render import release_memory

    failed = 0
    replay_env = dict(os.environ, BP_REPLAY_K="1", BP_REPLAY_MAXMULT="1")
    release_memory()          # drop the search/CMC compiles once, before any render
    for f in families:
        if release_memory(per_family=True):
            print("BP_RENDER_CLEAR=1: cleared the JAX compile cache before this render "
                  "(slower, lower peak)", flush=True)
        rep = next(r for r in failures if r.id == f.representative_id)
        path = os.path.join(out_dir, f"{f.id}.mp4")
        try:
            # Always re-rendered (not reused): CMC merges/renumbering can change which
            # world a fam-N id points at, and the render also measures the
            # physics-fidelity flag.
            _, rinfo = render_failure(env, policy_fn, target, space, rep.theta_vector,
                                      rep.batch_key, rep.env_index, cfg.horizon, path,
                                      consts, base_model=base_model, camera=target.camera,
                                      thresholds=cfg.thresholds, clip_after_fall_s=1.0,
                                      return_info=True)
        except Exception as e:  # noqa: BLE001 -- the report already exists; keep going
            failed += 1
            print(f"render {f.id} failed ({type(e).__name__}: {e}); the report keeps its "
                  f"evidence without this video", flush=True)
            continue
        f.mjx_min_floor_height = rinfo["min_floor_height"]
        mfh = rinfo["min_floor_height"]
        print("rendered", f.id, f.name,
              "" if mfh is None else f"(MJX min torso height above floor {mfh:.3f} m)",
              flush=True)
        mp4s[f.id] = f"{f.id}.mp4"   # relative filename (portable)
        # Evidence pair from the independent replay: re-run seed 0 only (the seed the
        # K-seed verdict renders from) on a COPY, so the checkpointed verdict is untouched.
        if f.replay and f.cause_certified and f.minimal_cause:
            try:
                fc = copy.deepcopy(f)
                replay_stage([fc], env=env, policy_fn=policy_fn, target=target, space=space,
                             consts=consts, horizon=cfg.horizon, thresholds=cfg.thresholds,
                             threshold=cfg.failure_threshold, out_dir=out_dir,
                             seed=cfg.seed, base_model=base_model, render=True,
                             environ=replay_env, log=None)
                if fc.evidence_videos and fc.evidence_videos.get("counterexample"):
                    f.evidence_videos = fc.evidence_videos
            except Exception as e:  # noqa: BLE001
                print(f"replay video {f.id} failed ({type(e).__name__}: {e})", flush=True)
        # No independent replay video: still show the minimal counterexample vs its
        # nominal twin, from the MJX search scene, same seed.
        if f.evidence_videos is None and f.cause_certified and f.minimal_cause:
            try:
                from breakpoint_mjx.crossval_engines import minimal_counterexample
                ce = minimal_counterexample(f, space)
                ce_vec = [ce[n] for n in space.names()]
                _, ci = render_failure(env, policy_fn, target, space, ce_vec, rep.batch_key,
                                       rep.env_index, cfg.horizon,
                                       os.path.join(out_dir, f"{f.id}-mincause.mp4"),
                                       consts, base_model=base_model, camera=target.camera,
                                       thresholds=cfg.thresholds, clip_after_fall_s=1.0,
                                       return_info=True)
                render_failure(env, policy_fn, target, space, space.nominal_vector(),
                               rep.batch_key, rep.env_index, cfg.horizon,
                               os.path.join(out_dir, f"{f.id}-nominal.mp4"), consts,
                               base_model=base_model, camera=target.camera,
                               max_steps=ci["n_steps"])
                f.evidence_videos = {"counterexample": f"{f.id}-mincause.mp4",
                                     "nominal": f"{f.id}-nominal.mp4", "source": "mjx"}
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"counterexample video {f.id} failed ({type(e).__name__}: {e})",
                      flush=True)
    return failed

if __name__ == "__main__":
    main()
