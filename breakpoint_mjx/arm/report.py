"""Replays for the viewer, the canonical study report, and the self-contained HTML export.

``build_report`` writes ``breakpoint.ar4.report/v1``: plain JSON that
``breakpoint sign`` / ``breakpoint verify`` (ed25519, ``attest.py``) accept unchanged, so
a stress-test result can be handed to someone else and checked for edits.
``export_html`` writes one HTML file: the Stress Lab UI in read-only mode with the
study, the replays of every failure family and the robot meshes embedded.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from breakpoint_mjx.arm.robots import get_robot

WEB = Path(__file__).parent / "web"
REPLAY_HZ = 25.0

LIMITS = [
    "Simulation only. Results come from a model of the AR4 (MuJoCo rigid-body dynamics plus "
    "a stepper, gearbox and controller model); no hardware was involved unless a world model "
    "fitted to a real log is named in this report.",
    "Drive-train constants that the AR4 sources do not publish (stepper torque curves, gearbox "
    "stiffness and backlash) are typical datasheet values until a world model calibrates them.",
    "A search is not a proof: a stress test that finds no failure family is a result under its "
    "budget, not evidence that none exists.",
    "Plausibility is a transparent heuristic on the stress-space bands, not a measured "
    "probability of the condition occurring.",
]


def mesh_dir(robot_id: str) -> str:
    return "ar4" if robot_id.startswith("ar4") else robot_id


def _r(a, n=5):
    return np.round(np.asarray(a, float), n).tolist()


def compact_replay(r: dict, kin=None) -> dict:
    """A simulate() result trimmed for the viewer: 25 Hz frames of every body (actual and
    commanded), the TCP paths and per-joint telemetry, plus the verdict."""
    from breakpoint_mjx.arm.program import Kinematics

    tr = r.get("trace")
    out = {k: r.get(k) for k in ("valid", "failed", "failure_score", "dominant", "modes",
                                 "metrics", "events", "issues", "taught_points", "stress",
                                 "invalid_reason")}
    if not tr:
        return out
    t = np.asarray(tr["t"])
    step = max(1, int(round((len(t) - 1) / max(t[-1], 1e-9) / REPLAY_HZ))) if len(t) > 1 else 1
    sel = slice(None, None, step)
    q_ref = np.asarray(tr["q_ref"])[sel]
    kin = kin or Kinematics((r.get("robot") or "mk5"))
    xr, qr = [], []
    import mujoco
    for q in q_ref:
        kin.d.qpos[:] = q
        mujoco.mj_kinematics(kin.m, kin.d)
        xr.append(kin.d.xpos[1:].copy())
        qr.append(kin.d.xquat[1:].copy())
    slips = np.asarray(tr["slips"])
    out["replay"] = {
        "t": _r(t[sel], 3),
        "xpos": _r(np.asarray(tr["xpos"])[sel]), "xquat": _r(np.asarray(tr["xquat"])[sel]),
        "xpos_ref": _r(xr), "xquat_ref": _r(qr),
        "tcp": _r(np.asarray(tr["tcp"])[sel]), "tcp_ref": _r(np.asarray(tr["tcp_ref"])[sel]),
        "q": _r(np.rad2deg(np.asarray(tr["q"])[sel]), 4),
        "q_ref": _r(np.rad2deg(q_ref), 4),
        "q_enc": _r(np.rad2deg(np.asarray(tr["q_enc"])[sel]), 4),
        "util": _r(np.asarray(tr["util"])[sel], 3),
        "aux": _r(np.asarray(tr["aux"])[sel], 3),
        "aux_label": tr.get("aux_label", ""), "aux_lines": tr.get("aux_lines", []),
        "slips": (slips[sel] * 4).astype(int).tolist(),
        "measure": tr.get("measure", []),
    }
    # slip events at full trace rate so a 1-sample slip is not lost to downsampling
    ev = []
    d = np.diff(slips, axis=0)
    for i, j in zip(*np.nonzero(d)):
        ev.append({"t": round(float(t[i + 1]), 3), "joint": int(j) + 1, "steps": int(d[i, j]) * 4})
    out["replay"]["slip_events"] = ev[:200]
    return out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_report(lab, sid: str) -> dict:
    """The canonical, signable JSON for one study."""
    from breakpoint_mjx import __version__

    doc = lab.study(sid)
    result = dict(doc["result"])
    n_records = len(result.pop("records", []) or [])
    prog_info = None
    if doc.get("program_id"):
        prog = lab.program(doc["program_id"])
        src = Path(prog.source).read_text() if prog.source else ""
        prog_info = {"id": doc["program_id"], **prog.to_dict(), "sha256": _sha(src)}
    wm = None
    if doc.get("world_model"):
        try:
            w = lab.world_model(doc["world_model"])
            wm = {"id": doc["world_model"], "params": w.params,
                  "fit": {k: w.report.get(k) for k in ("before", "after", "identifiability")},
                  "log_source": w.report.get("log", {}).get("meta", {}).get("source")}
        except FileNotFoundError:
            wm = {"id": doc["world_model"], "missing": True}
    return {
        "schema": "breakpoint.ar4.report/v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": {"name": "breakpoint", "version": __version__},
        "robot": get_robot(prog_info["model"] if prog_info else "mk5").info(),
        "program": prog_info,
        "study": {"id": sid, "kind": doc["kind"], "created": doc["created"],
                  "summary": doc["summary"], "worlds_evaluated": n_records or None},
        "simulator": {"timestep_s": 1e-3,
                      "controller": get_robot(prog_info["model"] if prog_info else "mk5").controller,
                      "world_model": wm},
        "results": result,
        "limits": LIMITS,
    }


def export_html(lab, sids, *, title=None) -> str:
    """One self-contained HTML file: the Stress Lab UI, read-only, with these studies (one
    per tab: search, envelope, endurance, calibration), the replays of every failure family
    and the robot meshes inlined. Needs network only for three.js."""
    sids = [sids] if isinstance(sids, str) else list(sids)
    docs = [lab.study(s) for s in sids]
    pid = next((d["program_id"] for d in docs if d.get("program_id")), None)
    replays = {}
    if pid:
        use_wm = any(d.get("world_model") and d["kind"] != "calibration" for d in docs)
        for d in docs:
            if d["kind"] == "search":
                for fam in d["result"]["families"][:6]:
                    replays[fam["key"]] = lab.replay(pid, fam["theta"], use_wm)
        replays["__nominal__"] = lab.replay(pid, None, use_wm)
    robot = get_robot(lab.program(pid).model if pid else "mk5")
    mdir = WEB / "meshes" / mesh_dir(robot.id)
    manifest = json.loads((mdir / "manifest.json").read_text())
    files = {p["file"]: base64.b64encode((mdir / p["file"]).read_bytes()).decode()
             for p in manifest["parts"]}
    from breakpoint_mjx.arm.sim import MODE_LABEL
    wm_name = next((d.get("world_model") for d in docs if d.get("world_model")), None)
    static = {
        "title": title,
        "state": {"robot": robot.info(), "programs": [], "world_models": lab.world_models(),
                  "active_world_model": wm_name, "studies": [], "modes": MODE_LABEL},
        "preview": lab.preview(pid) if pid else None,
        "study": docs[0], "studies": docs, "replays": replays,
        "report": build_report(lab, sids[0]),
        "meshes": {"manifest": manifest, "files": files},
    }
    html = (WEB / "index.html").read_text()
    css = (WEB / "style.css").read_text()
    js = (WEB / "app.js").read_text()
    data = json.dumps(_finite(static)).replace("</", "<\\/")
    html = html.replace('<link rel="stylesheet" href="/static/style.css">', f"<style>{css}</style>")
    html = html.replace('<script type="module" src="/static/app.js"></script>',
                        f'<script>window.BP_STATIC = {data};</script>\n'
                        f'<script type="module">{js}</script>')
    if title:
        html = html.replace("<title>Stress Lab · BreakPoint</title>", f"<title>{title}</title>")
    return html


def _finite(o):
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o
