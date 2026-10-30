"""BreakPoint AR4 Stress Lab: the local web app (``breakpoint arm serve``).

One process: a Flask app for the UI and JSON API, a single job thread (one heavy job at
a time) and a shared process pool that every job fans its simulations out to. Everything
the user creates lives in the workspace directory (default ``~/.breakpoint/ar4``):

    programs/*.yaml        uploaded robot programs
    logs/*.csv             robot logs (cmd vs feedback, optional TCP)
    world_models/*.json    fitted world models
    studies/*.json         job results (search, envelope, endurance, calibration)
"""
from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import yaml

from breakpoint_mjx.arm.robots import ROBOTS, get_robot
from breakpoint_mjx.arm import report as rep
from breakpoint_mjx.arm import world_model as WM
from breakpoint_mjx.arm.program import (Kinematics, build_reference, builtin_programs,
                                        load_program, parse_program)
from breakpoint_mjx.arm.search import (BUDGETS, Runner, endurance, envelope, minimal_cause,
                                       run_search, stress_space)
from breakpoint_mjx.arm.sim import CONTROL_MODES, MODE_LABEL, simulate, taught_offsets

WEB = Path(__file__).parent / "web"
_NAME = re.compile(r"^[A-Za-z0-9_.\- ]{1,80}$")


def _safe(name: str) -> str:
    name = str(name).strip()
    if not _NAME.match(name) or ".." in name or "/" in name:
        raise ValueError(f"bad name {name!r}")
    return name


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


class Lab:
    """Workspace + job queue. The Flask routes are thin wrappers around this."""

    def __init__(self, workspace: Path, workers: int | None = None):
        self.ws = Path(workspace).expanduser()
        for d in ("programs", "logs", "world_models", "studies"):
            (self.ws / d).mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self.runner = Runner(workers)
        self.runner.__enter__()
        self.kin = {}
        self.active_wm: str | None = None
        self._wm_cache = {}
        cfg = self.ws / "settings.json"
        if cfg.exists():
            self.active_wm = json.loads(cfg.read_text()).get("active_world_model")
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ resources
    def program(self, name: str):
        name = _safe(name)
        p = self.ws / "programs" / f"{name}.yaml"
        return load_program(p if p.exists() else name)

    def programs(self) -> list[dict]:
        out = []
        for n in builtin_programs():
            p = load_program(n)
            out.append({"id": n, "name": p.name, "builtin": True, "description": p.description,
                        "robot": p.model, "robot_name": get_robot(p.model).name})
        for f in sorted((self.ws / "programs").glob("*.yaml")):
            try:
                p = load_program(f)
                out.append({"id": f.stem, "name": p.name, "builtin": False,
                            "description": p.description, "robot": p.model,
                            "robot_name": get_robot(p.model).name})
            except Exception as e:                       # a broken upload stays visible
                out.append({"id": f.stem, "name": f.stem, "builtin": False, "error": str(e)})
        return out

    def save_program(self, text: str) -> str:
        raw = yaml.safe_load(text)
        prog = parse_program(raw)
        build_reference(prog, {}, rate_hz=200.0, kin=self._kin(prog.model))
        stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", prog.name.lower()).strip("_") or "program"
        (self.ws / "programs" / f"{stem}.yaml").write_text(text)
        return stem

    def _kin(self, model):
        if model not in self.kin:
            self.kin[model] = Kinematics(model)
        return self.kin[model]

    def world_model(self, name=None):
        name = name if name is not None else self.active_wm
        if not name:
            return None
        if name not in self._wm_cache:
            f = self.ws / "world_models" / f"{_safe(name)}.json"
            self._wm_cache[name] = WM.WorldModel.from_json(f.read_text())
        return self._wm_cache[name]

    def set_active_wm(self, name):
        if name:
            self.world_model(name)
        self.active_wm = name or None
        (self.ws / "settings.json").write_text(json.dumps({"active_world_model": self.active_wm}))

    def world_models(self):
        out = []
        for f in sorted((self.ws / "world_models").glob("*.json")):
            d = json.loads(f.read_text())
            r = d.get("report", {})
            out.append({"id": f.stem, "params": d.get("params"), "fitted_at": r.get("fitted_at"),
                        "log": r.get("log", {}).get("meta", {}).get("source"),
                        "before": r.get("before"), "after": r.get("after")})
        return out

    def logs(self):
        out = []
        for f in sorted((self.ws / "logs").glob("*.csv")):
            try:
                out.append({"id": f.stem, **WM.log_from_csv(f.read_text()).summary()})
            except Exception as e:
                out.append({"id": f.stem, "error": str(e)})
        return out

    def studies(self):
        out = []
        for f in sorted((self.ws / "studies").glob("*.json"), key=lambda p: -p.stat().st_mtime)[:40]:
            d = json.loads(f.read_text())
            out.append({"id": f.stem, "kind": d.get("kind"), "program": d.get("program_id"),
                        "created": d.get("created"), "summary": d.get("summary")})
        return out

    def study(self, sid):
        return json.loads((self.ws / "studies" / f"{_safe(sid)}.json").read_text())

    def _save_study(self, kind, program_id, result, summary, wm_name):
        sid = time.strftime("%Y%m%d-%H%M%S-") + kind + "-" + uuid.uuid4().hex[:4]
        doc = {"id": sid, "kind": kind, "program_id": program_id, "world_model": wm_name,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary,
               "result": result}
        (self.ws / "studies" / f"{sid}.json").write_text(json.dumps(_jsonable(doc)))
        return sid

    # ------------------------------------------------------------------ previews
    def preview(self, program_id):
        prog = self.program(program_id)
        kin = self._kin(prog.model)
        ref = build_reference(prog, {}, rate_hz=1000.0, kin=kin)
        step = 40
        tcp = [kin.fk(q)[0].tolist() for q in ref.q[::step]]
        marks = []
        for (i, lab, c) in ref.measure:
            marks.append({"label": lab, "cycle": c + 1, "t": i * 1e-3,
                          "xyz": kin.fk(ref.q[i])[0].tolist()})
        robot = get_robot(prog.model)
        return {"program": prog.to_dict(), "id": program_id, "duration_s": float(ref.t[-1]),
                "robot": robot.info(), "mesh_dir": rep.mesh_dir(robot.id),
                "issues": ref.issues, "tcp_path": tcp, "measure": marks,
                "q": np.rad2deg(ref.q[::step]).round(3).tolist(),
                "t": ref.t[::step].round(3).tolist(),
                "min_manipulability": ref.min_manip, "peak_speed_ratio": ref.peak_speed_ratio,
                "yaml": Path(prog.source).read_text() if prog.source else ""}

    def replay(self, program_id, theta=None, use_wm=True, cycles=None):
        prog = self.program(program_id)
        wm = self.world_model() if use_wm else None
        st = dict(theta or {})
        if wm is not None and "world_model_scale" not in st:
            st = {**wm.nominal_stress(), **st}
        if cycles:
            st["cycles"] = int(cycles)
        taught = self.runner.taught(prog, wm) if prog.fixed_reference is None else None
        r = simulate(prog, st, world_model=wm, kin=self._kin(prog.model), taught=taught)
        return rep.compact_replay(r, kin=self._kin(prog.model))

    # ------------------------------------------------------------------ jobs
    def submit(self, kind, params) -> str:
        jid = uuid.uuid4().hex[:10]
        self.jobs[jid] = {"id": jid, "kind": kind, "status": "queued", "done": 0, "total": 1,
                          "note": "queued", "params": params, "created": time.time()}
        self.q.put(jid)
        return jid

    def _progress(self, job):
        def cb(done, total, note):
            job.update(done=int(done), total=int(total), note=str(note))
        return cb

    def _loop(self):
        while True:
            jid = self.q.get()
            job = self.jobs[jid]
            job["status"] = "running"
            job["started"] = time.time()
            try:
                job["result_id"] = getattr(self, f"_job_{job['kind']}")(job, **job["params"])
                job["status"] = "done"
            except Exception as e:
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc()[-2000:]
            job["finished"] = time.time()

    def _job_search(self, job, program, budget="quick", use_wm=True, seed=0):
        prog = self.program(program)
        wm = self.world_model() if use_wm else None
        out = run_search(prog, budget=budget, seed=int(seed), world_model=wm,
                         runner=self.runner, progress=self._progress(job))
        summary = {"worlds": out["n_worlds"], "failed": out["n_failed"],
                   "families": len(out["families"]), "seconds": out["seconds"],
                   "budget": budget}
        return self._save_study("search", program, out, summary, self.active_wm if wm else None)

    def _job_mincause(self, job, program, theta, use_wm=True, study=None, family=None):
        prog = self.program(program)
        wm = self.world_model() if use_wm else None
        space = stress_space(prog, world_model=wm)
        full = {**{n: d.nominal for n, d in space.dims.items()}, **theta}
        job.update(note="isolating the cause", total=1)
        mc = minimal_cause(prog, full, space, world_model=wm, runner=self.runner)
        if study and family:                       # attach to the search it came from
            f = self.ws / "studies" / f"{_safe(study)}.json"
            doc = json.loads(f.read_text())
            for fam in doc["result"]["families"]:
                if fam["key"] == family:
                    fam["minimal_cause"] = _jsonable(mc)
            f.write_text(json.dumps(doc))
        return self._save_study("mincause", program, {"theta": full, "cause": mc},
                                {"axes": mc.get("axes")}, self.active_wm if wm else None)

    def _job_envelope(self, job, program, x="payload_kg", y="accel_scale", n=7, use_wm=True,
                      base=None):
        prog = self.program(program)
        wm = self.world_model() if use_wm else None
        job.update(total=int(n) * int(n), note="sweeping")
        out = envelope(prog, x, y, nx=int(n), ny=int(n), base=base, world_model=wm,
                       runner=self.runner, progress=self._progress(job))
        fails = sum(c["failed"] for c in out["cells"])
        return self._save_study("envelope", program, out,
                                {"x": x, "y": y, "cells": len(out["cells"]), "failed": fails},
                                self.active_wm if wm else None)

    def _job_endurance(self, job, program, cycles=10, theta=None, use_wm=True):
        prog = self.program(program)
        wm = self.world_model() if use_wm else None
        st = {**(wm.nominal_stress() if wm else {}), **(theta or {})}
        job.update(note=f"running {cycles} cycles", total=1)
        taught = self.runner.taught(prog, wm)
        from breakpoint_mjx.arm.sim import simulate as _sim
        r = _sim(prog, {**st, "cycles": int(cycles)}, world_model=wm, kin=self._kin(prog.model),
                 taught=taught, keep_trace=False)
        per = {}
        for p in r.get("taught_points", []):
            per.setdefault(p["label"], []).append(p)
        out = {"cycles": int(cycles), "theta": st, "per_point": per, "metrics": r["metrics"],
               "events": r["events"], "failed": r["failed"], "dominant": r["dominant"]}
        return self._save_study("endurance", program, out,
                                {"cycles": int(cycles), "drift_mm": r["metrics"].get("repeat_drift_mm")},
                                self.active_wm if wm else None)

    def _job_calibrate(self, job, log, fit_payload=False, name=None):
        lg = WM.log_from_csv((self.ws / "logs" / f"{_safe(log)}.csv").read_text())
        model = str(lg.meta.get("model", "mk5"))
        wm = WM.fit(lg, model=model, fit_payload=bool(fit_payload), runner=self.runner,
                    progress=self._progress(job))
        name = _safe(name or f"wm-{log}")
        (self.ws / "world_models" / f"{name}.json").write_text(wm.to_json())
        self._wm_cache.pop(name, None)
        self.set_active_wm(name)
        r = wm.report
        return self._save_study("calibration", None, {"world_model": name, "report": r},
                                {"before": r["before"], "after": r["after"], "params": wm.params},
                                name)

    def synthetic_log(self, program_id):
        prog = self.program(program_id)
        lg = WM.synthetic_log(prog)
        name = f"synthetic-{program_id}"
        (self.ws / "logs" / f"{name}.csv").write_text(WM.log_to_csv(lg))
        return name


def create_app(workspace="~/.breakpoint/ar4", workers=None):
    from flask import Flask, Response, abort, jsonify, request, send_from_directory

    lab = Lab(Path(workspace), workers)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.lab = lab

    def ok(x):
        return jsonify(_jsonable(x))

    @app.errorhandler(ValueError)
    def bad(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(FileNotFoundError)
    def missing(e):
        return jsonify({"error": f"not found: {e}"}), 404

    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

    @app.get("/static/<path:p>")
    def static_files(p):
        return send_from_directory(WEB, p)

    @app.get("/api/state")
    def state():
        return ok({"robots": {r.id: r.info() for r in ROBOTS.values() if r.id != "ar4_mk3"},
                   "programs": lab.programs(),
                   "world_models": lab.world_models(), "active_world_model": lab.active_wm,
                   "logs": lab.logs(), "studies": lab.studies(), "budgets": BUDGETS,
                   "modes": MODE_LABEL, "control_modes": list(CONTROL_MODES),
                   "spaces": {pid: list(stress_space(lab.program(pid)).dims)
                              for pid in [p["id"] for p in lab.programs() if not p.get("error")]},
                   "workers": lab.runner.workers, "workspace": str(lab.ws)})

    @app.get("/api/program/<pid>")
    def program(pid):
        return ok(lab.preview(pid))

    @app.post("/api/program")
    def upload_program():
        return ok({"id": lab.save_program(request.get_json(force=True)["yaml"])})

    @app.post("/api/replay")
    def replay():
        b = request.get_json(force=True)
        return ok(lab.replay(b["program"], b.get("theta"), b.get("use_wm", True), b.get("cycles")))

    @app.post("/api/jobs")
    def submit():
        b = request.get_json(force=True)
        kind = b.pop("kind")
        if kind not in ("search", "mincause", "envelope", "endurance", "calibrate"):
            raise ValueError(f"unknown job kind {kind!r}")
        return ok({"id": lab.submit(kind, b)})

    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = lab.jobs.get(jid)
        if j is None:
            abort(404)
        return ok({k: v for k, v in j.items() if k != "params"})

    @app.get("/api/study/<sid>")
    def study(sid):
        return ok(lab.study(sid))

    @app.post("/api/logs")
    def upload_log():
        b = request.get_json(force=True)
        text = b["csv"]
        WM.log_from_csv(text)                       # validate before saving
        name = _safe(re.sub(r"\.csv$", "", b.get("name") or "log"))
        (lab.ws / "logs" / f"{name}.csv").write_text(text)
        return ok({"id": name})

    @app.post("/api/logs/synthetic")
    def synth():
        return ok({"id": lab.synthetic_log(request.get_json(force=True)["program"])})

    @app.get("/api/logs/<lid>.csv")
    def get_log(lid):
        return Response((lab.ws / "logs" / f"{_safe(lid)}.csv").read_text(), mimetype="text/csv")

    @app.post("/api/world_model/active")
    def set_wm():
        lab.set_active_wm(request.get_json(force=True).get("id"))
        return ok({"active_world_model": lab.active_wm})

    @app.get("/api/world_model/<wid>")
    def get_wm(wid):
        return ok(json.loads((lab.ws / "world_models" / f"{_safe(wid)}.json").read_text()))

    @app.get("/api/report/<sid>.json")
    def report_json(sid):
        doc = rep.build_report(lab, sid)
        return Response(json.dumps(_jsonable(doc), indent=1), mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename={sid}.json"})

    @app.get("/api/report/<sid>.html")
    def report_html(sid):
        html = rep.export_html(lab, [sid])
        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": f"attachment; filename=ar4-{sid}.html"})

    return app


def serve(workspace="~/.breakpoint/ar4", host="127.0.0.1", port=8740, workers=None,
          open_browser=True):
    app = create_app(workspace, workers)
    url = f"http://{host}:{port}/"
    print(f"BreakPoint AR4 Stress Lab on {url}  (workspace {app.lab.ws}, "
          f"{app.lab.runner.workers} simulation workers)")
    if open_browser:
        threading.Timer(1.0, lambda: __import__("webbrowser").open(url)).start()
    app.run(host=host, port=port, threaded=True, use_reloader=False)
