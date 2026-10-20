"""Phase 4: random + CMA-ES adversarial search maximizing plausibility-weighted
(``transfer.joint_plausibility``) failure score, with persistence. The pure helper turns a batched trajectory into
per-env scores; the search loops call the (jit'd) evaluate once per batch/generation."""
from __future__ import annotations
import json, os, time
import numpy as np

from breakpoint_mjx.objectives import LEGGED_OBJECTIVE
from breakpoint_mjx.transfer import joint_plausibility, severity_from_score
from breakpoint_mjx.schemas import FailureRecord


def theta_to_cmaes_mean(space, seed_theta):
    """Inverse of the lo + sigmoid(x)*(hi-lo) squash: convert a named theta to the
    CMA-ES initial mean x0 (logit of the bound-normalized value)."""
    lo = np.array(space.lower()); hi = np.array(space.upper())
    x0 = np.zeros(len(space.names()))
    for i, n in enumerate(space.names()):
        v = float(seed_theta.get(n, space.dims[n].nominal))
        u = (v - lo[i]) / (hi[i] - lo[i] + 1e-9)
        u = min(max(u, 1e-3), 1.0 - 1e-3)
        x0[i] = np.log(u / (1.0 - u))
    return x0


def scores_from_trajectories(traj, thresholds, horizon, objective=None):
    """traj: batched dict (N,H,...). Returns (failure_scores (N,), signatures list, comps list).

    T7: routes through the ACTIVE target's ``objective`` (component_signals / failure_score
    / rollout_signature). ``objective=None`` uses the wrapped LEGGED objective, whose hooks
    ARE the original ``failure.*`` functions -- so the legged path is byte-identical to the
    pre-T7 direct calls. N is taken from any key so a manipulation traj (no "trunk_height")
    works too."""
    obj = objective or LEGGED_OBJECTIVE
    # Pull each (N,H,...) array off the device ONCE, then slice numpy views per row.
    # Converting inside the loop cost one device->host transfer per row PER KEY
    # (N x K transfers per generation, ~50 generations per search).
    host = {k: np.asarray(v) for k, v in traj.items()}
    n = int(next(iter(host.values())).shape[0])
    scores = np.zeros(n); sigs = []; comps = []
    for i in range(n):
        one = {k: v[i] for k, v in host.items()}
        c = obj.component_signals(one, thresholds)
        scores[i] = obj.failure_score(c, horizon)
        sigs.append(obj.rollout_signature(one)); comps.append(c)
    return scores, sigs, comps


def _records(space, thetas, scores, sigs, comps, seed, tag, start_id, batch_key):
    """Build FailureRecords for one evaluated batch.

    batch_key is the raw legacy PRNGKey actually passed to that evaluate(...) call;
    it is stored as two JSON-safe uint32 ints and each row gets env_index = i (its
    index WITHIN this batch), so render_failure can rebuild the exact eval key via
    fold_in(uint32(batch_key), env_index)."""
    bk = [int(x) for x in np.asarray(batch_key).reshape(-1)[:2]]
    recs = []
    thetas = np.asarray(thetas)
    for i in range(len(scores)):
        named = space.vector_to_named(thetas[i])
        tr = joint_plausibility(named, space)
        recs.append(FailureRecord(
            theta_vector=[float(x) for x in thetas[i]],
            theta_named={k: float(v) for k, v in named.items()},
            failure_score=float(scores[i]),
            component_signals={k: float(v) for k, v in comps[i].items()},
            rollout_signature=[float(x) for x in sigs[i]],
            seed=int(seed),
            transfer_score=float(tr),
            severity=severity_from_score(float(scores[i])),
            id=f"{tag}-{start_id + i}",
            batch_key=bk,
            env_index=i,
        ))
    return recs


def progress_line(label, done, total, t0, now=None):
    """``"<label> k/n  elapsed 12s  eta 30s"``. Pure (pass ``now`` in tests).

    ETA is a straight-line extrapolation of the mean per-step time so far. The first
    step includes the jit compile, so early ETAs over-estimate; that is the safe side for
    someone deciding whether to wait.
    """
    now = time.monotonic() if now is None else now
    el = max(0.0, now - t0)
    eta = el / done * (total - done) if done > 0 else float("nan")
    eta_s = "?" if eta != eta else f"{eta:.0f}s"
    return f"{label} {done}/{total}  elapsed {el:.0f}s  eta {eta_s}"


def random_search(evaluate, space, cfg, key, objective=None):
    import jax
    from breakpoint_mjx.thetas import sample_uniform
    budget = cfg.generations * cfg.popsize
    batch = cfg.n_eval
    n_batches = max(1, budget // batch)
    recs = []
    t0 = time.monotonic()
    for b in range(n_batches):
        key, sk, ek = jax.random.split(key, 3)
        thetas = sample_uniform(space, sk, batch)
        traj = evaluate(thetas, ek)
        scores, sigs, comps = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, objective)
        recs += _records(space, thetas, scores, sigs, comps, cfg.seed, "random", len(recs), ek)
        print(f"{progress_line('random batch', b + 1, n_batches, t0)}  "
              f"max_score={scores.max():.3f}", flush=True)
    return recs


def cmaes_search(evaluate, space, cfg, key, seed_theta=None, objective=None):
    """evosax 0.2.x CMA-ES maximizing transfer-weighted failure score.

    evosax internals use typed keys (jax.random.key); evaluate keeps a separate
    legacy PRNGKey stream so it shares one compiled binary with random_search.
    """
    import jax, jax.numpy as jnp
    from evosax.algorithms import CMA_ES
    D = len(space.names())
    lo = jnp.asarray(space.lower()); hi = jnp.asarray(space.upper())
    es = CMA_ES(population_size=cfg.popsize, solution=jnp.zeros(D))
    params = es.default_params
    es_key = jax.random.key(int(cfg.seed))            # typed key for evosax
    eval_key = jax.random.PRNGKey(int(cfg.seed) + 1)  # legacy key for evaluate
    # VERIFY on Colab: print gen-0 mean theta for seeded vs unseeded -- they should differ.
    # If evosax's init ignores the mean arg, set it via strategy params instead:
    #   params = params.replace(mean=init_mean)  (evosax >= 0.2 uses flax replace)
    init_mean = jnp.asarray(theta_to_cmaes_mean(space, seed_theta)) if seed_theta else jnp.zeros(D)
    state = es.init(es_key, init_mean, params)
    recs = []
    t0 = time.monotonic()
    for g in range(cfg.generations):
        es_key, ask_k, tell_k = jax.random.split(es_key, 3)
        eval_key, ek = jax.random.split(eval_key, 2)
        population, state = es.ask(ask_k, state, params)      # (popsize, D) in R
        thetas = lo + jax.nn.sigmoid(population) * (hi - lo)  # squash to bounds
        traj = evaluate(thetas, ek)
        scores, sigs, comps = scores_from_trajectories(traj, cfg.thresholds, cfg.horizon, objective)
        named = [space.vector_to_named(np.asarray(thetas)[i]) for i in range(len(scores))]
        transfer = np.array([joint_plausibility(nm, space) for nm in named])
        weighted = scores * transfer                          # transfer-weighted objective
        state, _ = es.tell(tell_k, population, -jnp.asarray(weighted), state, params)  # minimizes
        recs += _records(space, thetas, scores, sigs, comps, cfg.seed, "cmaes", len(recs), ek)
        print(f"gen {g}: max_score={scores.max():.3f} max_weighted={weighted.max():.3f} "
              f"mean_transfer={transfer.mean():.3f}  "
              f"[{progress_line('cmaes gen', g + 1, cfg.generations, t0)}]", flush=True)
    return recs


def save_failures(records, threshold, out_dir, name="failures.json"):
    failures = [r for r in records if r.failure_score >= threshold]
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump([json.loads(r.to_json()) for r in records], f)
    print(f"{len(failures)}/{len(records)} above threshold; wrote {path}")
    return failures, path


def load_failures(path):
    """Load a JSON list of FailureRecord dicts back into FailureRecord objects."""
    with open(path) as f:
        data = json.load(f)
    return [FailureRecord(**d) for d in data]
