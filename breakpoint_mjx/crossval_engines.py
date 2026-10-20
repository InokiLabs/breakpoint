"""Cross-SIMULATOR validation: replay verified failures across independent physics
engines and measure where they agree (T-crossengine, the layer above crossval.py).

crossval.py already measures cross-CONFIG validity (MJX solver/timestep variants +
optionally the C-MuJoCo backend). This module generalizes that to an ENGINE
AGREEMENT MATRIX across genuinely different simulators (MJX, C MuJoCo, Genesis, and
future PyBullet/Drake backends): per failure, which engines reproduce it; per
engine pair, how often they agree; and the disagreement list — the "survives in
engine A, dies in engine B" evidence that no single-simulator vendor can produce.

WHICH ENGINES CAN RUN IS TARGET-DEPENDENT, and the report says so rather than
quietly shrinking. A mujoco_playground target gets {mjx, mujoco_c} (Genesis cannot
call Playground's reset/obs/command); a URDF-derived target gets {mjx, genesis} (the
C backend's parity hooks are Playground contract points that URDFEnv does not have).
``EngineReport.skipped`` carries the reason for every engine that did not run and
``EngineReport.is_cross_engine`` is False when fewer than two engines produced a
verdict — check it before quoting an agreement number, because a one-engine report
has an EMPTY ``pairwise_agreement`` that reads exactly like "nothing disagreed".

``run_cross_engine`` is the one-call entry point; ``available_engines`` /
``replay_matrix`` / ``agreement_report`` are the pieces it composes.

Engines are anything with ``.name`` and ``.simulate(record) -> summary`` (the
rollout_one summary contract shared by breakpoint_mjx.backends). Aggregation here
is pure numpy/stdlib and unit-tested on CPU; scoring goes through the same
component_signals -> objective.failure_score path the search uses, so an engine
"reproduces" a failure by exactly the criterion that discovered it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from breakpoint_mjx.crossval import reproduces


@dataclass
class EngineResult:
    engine: str
    score: float | None = None       # re-scored failure_score, None if errored
    reproduced: bool | None = None   # None if errored
    error: str | None = None


@dataclass
class EngineReport:
    n_records: int
    threshold: float
    engines: list[str]
    reproduction_rate: dict[str, float]      # engine -> fraction reproduced (of runs)
    error_rate: dict[str, float]             # engine -> fraction errored
    pairwise_agreement: dict[str, float]     # "A|B" -> agree fraction (both ran)
    disagreements: list[dict] = field(default_factory=list)
    # Engines that did NOT run, with the reason. Carried ON the report (not just printed
    # by a script) so a consumer can always tell a 1-engine run from cross-engine
    # evidence -- ``is_cross_engine`` is the check every caller should make.
    skipped: list[list] = field(default_factory=list)

    @property
    def is_cross_engine(self) -> bool:
        """True only if at least TWO engines actually produced a verdict.

        A report with one engine is a re-sim, not cross-engine evidence; reading
        ``pairwise_agreement`` from such a report would silently return ``{}`` and could
        be mistaken for "no disagreements found".
        """
        return len([e for e in self.engines if self.error_rate.get(e, 0.0) < 1.0]) >= 2

    def to_json(self, path=None):
        d = {**self.__dict__, "is_cross_engine": self.is_cross_engine}
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
        return d


def score_summary_fn(thresholds, horizon, objective=None):
    """Default scorer: summary -> failure_score via the SAME path the search used."""
    from breakpoint_mjx.failure import component_signals, failure_score as legged_fs

    fs = objective.failure_score if objective is not None else legged_fs
    comps = objective.component_signals if objective is not None else component_signals

    def score(summary):
        return float(fs(comps(summary, thresholds), horizon))

    return score


def replay_matrix(records, engines, threshold, score_fn):
    """Replay every record on every engine. Returns {record.id: [EngineResult, ...]}.

    An engine exception is captured per-cell (recorded, never raised) so one broken
    backend cannot void a whole report; errored cells are excluded from agreement
    denominators but surfaced in error_rate — no silent skips.
    """
    matrix = {}
    for rec in records:
        row = []
        for eng in engines:
            try:
                s = score_fn(eng.simulate(rec))
                row.append(EngineResult(eng.name, s, reproduces(s, threshold)))
            except Exception as e:  # noqa: BLE001 -- per-cell capture is the contract
                row.append(EngineResult(eng.name, error=f"{type(e).__name__}: {e}"))
        matrix[rec.id or f"idx{len(matrix)}"] = row
    return matrix


def agreement_report(matrix, threshold, records=None, skipped=None):
    """Aggregate a replay matrix into the engine-agreement report.

    ``skipped`` (``[(name, reason), ...]``) is carried onto the report so a consumer can
    see which engines were NOT part of the agreement numbers, not just which were.
    """
    skipped = [list(s) for s in (skipped or [])]
    if not matrix:
        return EngineReport(0, threshold, [], {}, {}, {}, skipped=skipped)
    engines = [r.engine for r in next(iter(matrix.values()))]
    fam = {r.id: r.family_id for r in records or [] if r.id}

    ran = {e: 0 for e in engines}
    rep = {e: 0 for e in engines}
    err = {e: 0 for e in engines}
    agree = {}
    both = {}
    disagreements = []

    for rid, row in matrix.items():
        by = {r.engine: r for r in row}
        for e in engines:
            if by[e].error:
                err[e] += 1
            else:
                ran[e] += 1
                rep[e] += int(by[e].reproduced)
        for i, a in enumerate(engines):
            for b in engines[i + 1:]:
                if by[a].error or by[b].error:
                    continue
                k = f"{a}|{b}"
                both[k] = both.get(k, 0) + 1
                agree[k] = agree.get(k, 0) + int(by[a].reproduced == by[b].reproduced)
        verdicts = {r.engine: r.reproduced for r in row if not r.error}
        if len(set(verdicts.values())) > 1:
            disagreements.append({
                "id": rid,
                "family": fam.get(rid),
                "scores": {r.engine: r.score for r in row if not r.error},
                "verdicts": verdicts,
            })

    n = len(matrix)
    return EngineReport(
        n_records=n,
        threshold=threshold,
        engines=engines,
        reproduction_rate={e: (rep[e] / ran[e] if ran[e] else 0.0) for e in engines},
        error_rate={e: err[e] / n for e in engines},
        pairwise_agreement={k: agree[k] / both[k] for k in both},
        disagreements=disagreements,
        skipped=skipped,
    )


def available_engines(env, policy_fn, target, space, consts, base_model, horizon,
                      mj_model=None):
    """Instantiate every engine backend that is importable AND parity-ready here.

    Returns (engines, skipped) where skipped is [(name, reason)] — callers print
    skipped so a 1-engine report never masquerades as cross-engine evidence.
    """
    engines, skipped = [], []
    from breakpoint_mjx.backends.mjx_backend import MjxBackend

    mjx = MjxBackend(env, policy_fn, target, space, consts, base_model, horizon)
    mjx.name = "mjx"
    engines.append(mjx)

    from breakpoint_mjx.backends.mujoco_c import MujocoCBackend
    if not MujocoCBackend.available():
        skipped.append(("mujoco_c", "mujoco not importable"))
    elif not MujocoCBackend.ready():
        skipped.append(("mujoco_c", "reset/obs parity not ported (READY=False)"))
    elif getattr(type(env), "_get_obs", None) is None:
        # The C backend reaches obs parity by shadowing the env's own ``_get_obs`` and by
        # swapping ``mjx_env.step`` -- both are mujoco_playground contract points. A
        # URDF-derived env (urdf_target.URDFEnv) names its observation builder ``_obs``
        # and calls ``mjx.step`` directly, so neither hook fires. Skipping with a reason
        # here is honest; letting it run would produce one NotImplementedError per record
        # and an error_rate of 1.0 that says nothing.
        skipped.append((
            "mujoco_c",
            f"{type(env).__name__} does not expose the mujoco_playground hooks the C "
            f"backend needs (_get_obs + physics via mjx_env.step)",
        ))
    else:
        c = MujocoCBackend(env, policy_fn, target, space, consts, horizon,
                           mj_model=mj_model)
        c.name = "mujoco_c"
        engines.append(c)

    try:
        from breakpoint_mjx.backends.genesis_backend import GenesisBackend

        # One helper owns every reason Genesis can be unavailable (not importable, not
        # parity-ready, or a Playground target it deliberately refuses), so the skip line
        # in the report always says WHICH of those it was.
        reason = GenesisBackend.skip_reason(env)
        if reason:
            skipped.append(("genesis", reason))
        else:
            g = GenesisBackend(env, policy_fn, target, space, consts, horizon,
                               mj_model=mj_model)
            g.name = "genesis"
            engines.append(g)
    except ImportError:
        skipped.append(("genesis", "backend module absent"))

    return engines, skipped


def resolve_thresholds(target):
    """The failure thresholds the search would have used for ``target``.

    Same resolution run_pipeline does -- ``configs/search.yaml`` defaults with the
    target's own overrides on top -- so an engine "reproduces" a failure by exactly the
    criterion that discovered it, including a URDF target's derived
    ``trunk_height_min``.
    """
    from breakpoint_mjx.config import load_search_config, merge_target_thresholds
    from breakpoint_mjx.paths import config_path

    cfg = load_search_config(config_path("search.yaml"))
    return dict(merge_target_thresholds(cfg, target).thresholds)


def run_cross_engine(records, env, policy_fn, target, space, consts, base_model, horizon,
                     *, threshold=0.5, thresholds=None, mj_model=None, out_path=None,
                     return_matrix=False):
    """Replay ``records`` on every available engine and return the ``EngineReport``.

    The one-call entry point for the cross-SIMULATOR layer: instantiate the engines that
    are importable AND parity-ready here, replay every record on each, re-score each
    replay through the target's OWN ``component_signals -> failure_score`` path, and
    aggregate into agreement / disagreement numbers.

    Parameters
    ----------
    records   : iterable of ``FailureRecord`` (each needs ``batch_key``/``env_index`` --
                a legacy record is refused per-cell rather than replayed off a fabricated
                key).
    env, policy_fn, target, space, consts, base_model, horizon
                the same objects the evaluator ran with. ``base_model`` is the PRISTINE
                ``env.mjx_model``, captured before any re-sim.
    threshold : failure_score at/above which an engine "reproduces" a failure. Use the
                SAME threshold the search used.
    thresholds: component thresholds for scoring; defaults to ``resolve_thresholds(target)``.
    mj_model  : optional host ``mujoco.MjModel`` for the host-CPU backends (defaults to
                the env's own).
    out_path  : write the report JSON here as well as returning it.

    Returns
    -------
    ``EngineReport`` (or ``(EngineReport, matrix)`` with ``return_matrix=True``). The
    report carries ``skipped`` and ``is_cross_engine``: a caller MUST check the latter
    before quoting agreement numbers, because a single-engine report has an empty
    ``pairwise_agreement`` that reads like "nothing disagreed".
    """
    records = list(records)
    engines, skipped = available_engines(
        env, policy_fn, target, space, consts, base_model, horizon, mj_model=mj_model
    )
    score_fn = score_summary_fn(
        thresholds if thresholds is not None else resolve_thresholds(target),
        horizon, getattr(target, "objective", None),
    )
    matrix = replay_matrix(records, engines, threshold, score_fn)
    report = agreement_report(matrix, threshold, records, skipped=skipped)
    if out_path:
        report.to_json(out_path)
    return (report, matrix) if return_matrix else report


# --------------------------------------------------------------------------- #
# Skeptic-grade evidence: independent full-collision replay of each certified
# minimal counterexample (pipeline stage, ON by default, BP_REPLAY=0 disables)
# --------------------------------------------------------------------------- #
REPLAY_K = 8
REPLAY_BATCH_KEY = 9_000_001       # replay seeds are fold_in([0, this + cfg.seed], i)


def minimal_counterexample(family, space):
    """The CMC minimal counterexample as a full theta dict: every dim nominal except the
    certified cause dims, which take their shrunk counterexample values. None when the
    family has no certified cause."""
    if getattr(family, "cause_certified", None) is not True or family.minimal_cause is None:
        return None
    theta = {n: float(v) for n, v in zip(space.names(), space.nominal_vector())}
    for c in family.minimal_cause:
        theta[c["dim"]] = float(c["value"])
    return theta


def _replay_record(space, theta_named, batch_key, env_index, rid):
    from breakpoint_mjx.schemas import FailureRecord

    names = space.names()
    return FailureRecord(
        theta_vector=[float(theta_named[n]) for n in names], theta_named=dict(theta_named),
        failure_score=0.0, component_signals={}, rollout_signature=[], seed=int(env_index),
        batch_key=[int(b) for b in batch_key], env_index=int(env_index), id=rid)


def stop_after_fall(thresholds, n_after, max_steps=None):
    """A fresh ``early_stop(t, summary)`` callback: stop ``n_after`` steps after the first
    fallen step, or at ``max_steps``. After a fall the replay verdict cannot change (a
    physically valid fall scores >= 0.7 and the validity gate only reads steps up to the
    fall), so this saves CPU without changing any reproduce/not-reproduce decision."""
    import numpy as np

    state = {"fall": None}

    def stop(t, s):
        if state["fall"] is None:
            h = float(np.asarray(s["floor_height"] if "floor_height" in s
                                 else s["trunk_height"]))
            up = float(np.asarray(s["up_proj"]))
            if (not np.isfinite(h) or not np.isfinite(up)
                    or h < float(thresholds["trunk_height_min"])
                    or up < float(thresholds["up_proj_min"])):
                state["fall"] = t
        if state["fall"] is not None and t >= state["fall"] + n_after:
            return True
        return max_steps is not None and t + 1 >= int(max_steps)

    return stop


def replay_family(backend, space, family, *, k, score_fn, threshold, thresholds, batch_key,
                  n_after, alpha=None, bar=None, collect=False, max_mult=1):
    """Replay one family's minimal counterexample on ``k`` seeds (+ one nominal twin).

    With the default ``max_mult=1`` there is one look at ``k`` seeds and the verdict uses
    the SAME Clopper-Pearson rule as CMC (``mincause.decide``): ``reproduced`` when the
    lower bound clears the bar, ``not_reproduced`` when the upper bound is below it, else
    ``inconclusive``; the replay stops as soon as the bar is out of reach. With
    ``max_mult > 1`` an undecided family is topped up in rounds up to ``k * max_mult``
    seeds and decided on the exact group-sequential boundaries of
    ``mincause.look_schedule`` (the same certifier CMC uses), so repeated looks do not
    inflate the error rate past ``alpha``: a full-alpha CP test at every wave would.
    Only ``reproduced`` keeps a family in the ranked list. Returns ``(replay_dict,
    ce_states, twin_states)``; the state lists are None unless ``collect``. ``backend``
    is anything with ``simulate(record, collect_states=...)`` and a settable
    ``early_stop``, so this is CPU-unit-testable with a fake."""
    from breakpoint_mjx.mincause import ALPHA, FAIL_BAR, cp_lower, cp_upper, look_schedule
    from breakpoint_mjx.failure import fall_step, min_floor_height

    alpha = ALPHA if alpha is None else alpha
    bar = FAIL_BAR if bar is None else bar
    ce = minimal_counterexample(family, space)
    if ce is None:
        return None, None, None
    need = next((j for j in range(k + 1) if cp_lower(j, k, alpha) >= bar), k + 1)
    hits, n, falls, floors, ce_states = 0, 0, [], [], None
    n_max = k * max(1, int(max_mult))
    sequential = max_mult > 1 and k > 1
    if sequential:
        looks, fail_c, pass_d = look_schedule(k, n_max, alpha, bar)
        look_at = {nn: r for r, nn in enumerate(looks)}
    seq_verdict = None
    for i in range(n_max):
        backend.early_stop = stop_after_fall(thresholds, n_after)
        rec = _replay_record(space, ce, batch_key, i, f"replay-{family.id}-{i}")
        want = collect and i == 0
        out = backend.simulate(rec, collect_states=True) if want else backend.simulate(rec)
        summ, states = out if want else (out, None)
        if want:
            ce_states = states
        n += 1
        hits += int(float(score_fn(summ)) >= threshold)
        falls.append(fall_step(summ, thresholds))
        floors.append(min_floor_height(summ))
        if not sequential and (n - hits) > (k - need):
            break                          # 'reproduced' is out of reach within k seeds
        if sequential and n in look_at:
            r = look_at[n]
            if hits >= fail_c[r]:
                seq_verdict = "reproduced"
            elif hits <= pass_d[r]:
                seq_verdict = "not_reproduced"
            if seq_verdict is not None:
                break                      # decided on the group-sequential boundary
    twin, twin_states = None, None
    if family.minimal_cause:               # an empty cause means it fails AT nominal
        cap = (falls[0] + 1 + n_after) if falls and falls[0] is not None else None
        backend.early_stop = stop_after_fall(thresholds, n_after, max_steps=cap)
        nom = {nm: float(v) for nm, v in zip(space.names(), space.nominal_vector())}
        rec = _replay_record(space, nom, batch_key, 0, f"replay-{family.id}-nominal")
        out = backend.simulate(rec, collect_states=True) if collect else backend.simulate(rec)
        summ, twin_states = out if collect else (out, None)
        tf = fall_step(summ, thresholds)
        twin = {"fell": tf is not None, "fall_step": tf,
                "min_floor_height": min_floor_height(summ), "seed": 0,
                "steps": int(len(next(iter(summ.values()))))}
    backend.early_stop = None
    lo = cp_lower(hits, n, alpha)
    hi = cp_upper(hits, n, alpha)
    if sequential:
        verdict = seq_verdict or "inconclusive"
    else:
        verdict = ("reproduced" if lo >= bar else "not_reproduced" if hi < bar
                   else "inconclusive")
    fl = [x for x in floors if x is not None]
    return {
        "k": int(hits), "n": int(n), "k_planned": int(k), "cp_lo": round(float(lo), 4),
        "cp_hi": round(float(hi), 4), "verdict": verdict,
        "bar": float(bar), "alpha": float(alpha),
        "reproduces": verdict == "reproduced",
        "decision_rule": "group-sequential" if sequential else "clopper-pearson",
        "fall_steps": falls, "min_floor_height": (min(fl) if fl else None),
        "counterexample": ce, "nominal_twin": twin,
    }, ce_states, twin_states


def replay_stage(families, *, env, policy_fn, target, space, consts, horizon, thresholds,
                 threshold, out_dir=None, seed=0, environ=None, render=True, log=print,
                 base_model=None):
    """The pipeline's independent-replay stage. ON by default; ``BP_REPLAY=0`` disables.

    For every family with a certified minimal cause, replays the minimal counterexample
    on ``BP_REPLAY_K`` (default 8) seeds in C MuJoCo on a FULL-collision variant of the
    search scene (``backends.mujoco_c.build_full_collision_model``), plus one nominal twin
    on seed 0, and annotates ``family.replay``. With ``render`` and ``out_dir`` it also
    writes ``<fam>-mincause.mp4`` / ``<fam>-nominal.mp4`` from those replays (clipped 1 s
    after the fall) into ``family.evidence_videos``.

    Never fails a run: every problem becomes ``status`` + ``reason`` in the returned
    summary so the report can say it rather than silently fall back.

    ``base_model`` is the PRISTINE ``env.mjx_model`` captured before any evaluation:
    ``rollout_one`` rebinds ``env._mjx_model`` to a vmapped tracer during the search, and
    the env's own reset (which the C replay runs) reads it, so it is restored for the
    duration of the stage and put back afterwards.
    """
    saved = getattr(env, "_mjx_model", None)
    swap = base_model is not None and hasattr(env, "_mjx_model")
    if swap:
        env._mjx_model = base_model
    try:
        return _replay_stage(families, env=env, policy_fn=policy_fn, target=target,
                             space=space, consts=consts, horizon=horizon,
                             thresholds=thresholds, threshold=threshold, out_dir=out_dir,
                             seed=seed, environ=environ, render=render, log=log)
    finally:
        if swap:
            env._mjx_model = saved


def _replay_stage(families, *, env, policy_fn, target, space, consts, horizon, thresholds,
                  threshold, out_dir, seed, environ, render, log):
    import os
    import time

    from breakpoint_mjx.mincause import ALPHA, FAIL_BAR

    env_vars = os.environ if environ is None else environ
    if str(env_vars.get("BP_REPLAY", "1")).strip().lower() in ("0", "false", "off", "no"):
        return {"status": "disabled", "reason": "BP_REPLAY=0"}
    todo = [f for f in families if minimal_counterexample(f, space) is not None]
    if not todo:
        return {"status": "skipped", "reason": "no family has a certified minimal cause"}
    obj = getattr(target, "objective", None)
    if obj is not None and getattr(obj, "name", "legged") != "legged":
        return {"status": "unavailable", "reason": "the C-MuJoCo replay is legged-only"}
    from breakpoint_mjx.backends.mujoco_c import MujocoCBackend, build_full_collision_model

    if not MujocoCBackend.ready():
        return {"status": "unavailable", "reason": "mujoco not importable"}
    if getattr(type(env), "_get_obs", None) is None:
        return {"status": "unavailable",
                "reason": f"{type(env).__name__} lacks the mujoco_playground hooks"}
    t0 = time.time()
    model, info = build_full_collision_model(env, target)
    if model is None:
        return {"status": "unavailable", "reason": info.get("reason"), "collision_model": info}
    k = int(env_vars.get("BP_REPLAY_K", str(REPLAY_K)))
    mult = int(env_vars.get("BP_REPLAY_MAXMULT", "1"))
    backend = MujocoCBackend(env, policy_fn, target, space, consts, horizon, mj_model=model,
                             jit_env=True)
    backend.name = "mujoco_c_fullcollision"
    score_fn = score_summary_fn(thresholds, horizon, obj)
    n_after = int(round(1.0 / float(env.dt)))
    batch_key = [0, (REPLAY_BATCH_KEY + int(seed)) & 0xFFFFFFFF]
    camera = getattr(target, "camera", "track")
    out = {"status": "ran", "engine": "mujoco_c", "collision_model": info, "k": k,
           "bar": FAIL_BAR, "alpha": ALPHA, "families": {}}
    for f in todo:
        try:
            rp, ce_states, twin_states = replay_family(
                backend, space, f, k=k, score_fn=score_fn, threshold=threshold,
                thresholds=thresholds, batch_key=batch_key, n_after=n_after,
                collect=bool(render and out_dir), max_mult=mult)
        except Exception as e:  # noqa: BLE001 -- one family's replay never kills the run
            out["families"][f.id] = {"error": f"{type(e).__name__}: {e}"}
            if log:
                log(f"  replay {f.id}: error {type(e).__name__}: {e}")
            continue
        rp.update(engine="mujoco_c", collision_model=info.get("source"),
                  scene=info.get("scene"))
        f.replay = rp
        out["families"][f.id] = {kk: rp[kk] for kk in ("k", "n", "cp_lo", "cp_hi",
                                                       "verdict", "reproduces")}
        if render and out_dir and ce_states:
            from breakpoint_mjx.render import render_states

            vids = {}
            try:
                vids["counterexample"] = os.path.basename(render_states(
                    env, ce_states, os.path.join(out_dir, f"{f.id}-mincause.mp4"),
                    camera=camera, mj_model=model))
                if twin_states:
                    vids["nominal"] = os.path.basename(render_states(
                        env, twin_states, os.path.join(out_dir, f"{f.id}-nominal.mp4"),
                        camera=camera, mj_model=model))
                vids["source"] = "c_mujoco_full_collision_replay"
                f.evidence_videos = vids
            except Exception as e:  # noqa: BLE001 -- no display / renderer on this box
                out["families"][f.id]["render_error"] = f"{type(e).__name__}: {e}"
        if log:
            log(f"  replay {f.id} ({f.name}): reproduces in C-MuJoCo full-collision "
                f"{rp['k']}/{rp['n']} (CP lower {rp['cp_lo']:.2f}, bar {rp['bar']}) -> "
                f"{rp['verdict'].upper().replace('_', ' ')}")
    ran = [f for f in todo if f.replay]
    out["n_replayed"] = len(ran)
    out["n_reproduced"] = sum(1 for f in ran if f.replay["reproduces"])
    out["n_demoted"] = len(ran) - out["n_reproduced"]
    out["n_inconclusive"] = sum(1 for f in ran if f.replay["verdict"] == "inconclusive")
    out["max_mult"] = mult
    out["seconds"] = round(time.time() - t0, 1)
    return out
