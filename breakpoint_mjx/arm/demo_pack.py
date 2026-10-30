"""``breakpoint arm demo-pack``: one shareable folder with a demo for every target robot.

For each arm program it runs the full set of studies on this machine (stress search with
the minimal cause of the top family, an operating envelope, a repeatability run and a
calibration against a SYNTHETIC log) and exports them as one self-contained, read-only
HTML page. Legged-policy reports from the engine (``breakpoint run``) are copied in. A hub
``index.html`` links everything, so the folder can be zipped, attached or hosted as is.
"""
from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path

from breakpoint_mjx.arm.robots import get_robot

ARMS = [  # (program, priority)
    ("ar4_pick_and_place", "P0"),
    ("ur5e_pick_and_place", "P0"),
    ("xarm6_pick_and_place", "P1"),
]
POLICIES = [  # (key, name, priority, how to make the report)
    ("go1", "Unitree Go1", "P0",
     "breakpoint run --robot go1 --policy <policy> --budget standard"),
    ("g1", "Unitree G1", "P0",
     "python scripts/train.py --target g1 --out out && "
     "breakpoint run --robot g1 --policy out/g1/g1_policy.pkl --budget standard"),
    ("spot", "Boston Dynamics Spot", "P1",
     "python scripts/train.py --target spot --out out && "
     "breakpoint run --robot spot --policy out/spot/spot_policy.pkl --budget standard"),
]
SAMPLE_GO1 = Path(__file__).resolve().parents[1] / "assets" / "demo" / "sample_report" / "report.html"


class _Job(dict):
    def __init__(self, note):
        super().__init__(done=0, total=1, note=note)


def build(out_dir, *, programs=None, budget="quick", policy_reports=None, workers=None,
          log=print) -> Path:
    from breakpoint_mjx.arm import report as rep
    from breakpoint_mjx.arm.server import Lab

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    lab = Lab(out / ".workspace", workers)
    lab.set_active_wm(None)
    cards = []
    for pid, prio in (programs or ARMS):
        prog = lab.program(pid)
        robot = get_robot(prog.model)
        t0 = time.time()
        log(f"[{robot.name}] stress search ({budget})")
        s_search = lab._job_search(_Job("search"), pid, budget=budget, use_wm=False)
        search = lab.study(s_search)["result"]
        fam = search["families"][0] if search["families"] else None
        cause = None
        if fam:
            log(f"[{robot.name}] minimal cause of '{fam['name']}'")
            lab._job_mincause(_Job("cause"), pid, fam["theta"], use_wm=False, study=s_search,
                              family=fam["key"])
            cause = next(f for f in lab.study(s_search)["result"]["families"]
                         if f["key"] == fam["key"]).get("minimal_cause")
        log(f"[{robot.name}] operating envelope")
        s_env = lab._job_envelope(_Job("envelope"), pid, "payload_kg", "accel_scale", n=7,
                                  use_wm=False)
        env = lab.study(s_env)["result"]
        log(f"[{robot.name}] repeatability, 10 cycles")
        s_rep = lab._job_endurance(_Job("repeat"), pid, cycles=10, use_wm=False)
        rpt = lab.study(s_rep)["result"]
        log(f"[{robot.name}] calibration against a synthetic log")
        lid = lab.synthetic_log(pid)
        s_cal = lab._job_calibrate(_Job("cal"), lid, name=f"wm-{pid}")
        cal = lab.study(s_cal)["result"]["report"]
        lab.set_active_wm(None)
        log(f"[{robot.name}] exporting")
        fn = f"{robot.id}.html"
        (out / fn).write_text(rep.export_html(lab, [s_search, s_env, s_rep, s_cal],
                                              title=f"{robot.name} · Stress Lab"))
        cards.append(_arm_card(robot, prio, fn, prog, search, fam, cause, env, rpt, cal))
        log(f"[{robot.name}] done in {time.time() - t0:.0f} s")

    for key, name, prio, how in POLICIES:
        src = (policy_reports or {}).get(key)
        if src is None and key == "go1" and SAMPLE_GO1.exists():
            src, note = SAMPLE_GO1, "Bundled CPU smoke-budget run of the demo Go1 policy."
        else:
            note = "Report from the BreakPoint engine."
        if src:
            shutil.copy(src, out / f"{key}.html")
            cards.append({"name": name, "priority": prio, "product": "Policy testing",
                          "href": f"{key}.html", "lines": [note], "pending": False})
        else:
            cards.append({"name": name, "priority": prio, "product": "Policy testing",
                          "href": None, "pending": True,
                          "lines": ["Needs a trained policy and a GPU run:", how]})
    (out / "index.html").write_text(_hub(cards))
    (out / "pack.json").write_text(json.dumps({"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "budget": budget, "cards": cards}, indent=1,
                                              default=str))
    shutil.rmtree(out / ".workspace" / "logs", ignore_errors=True)
    return out / "index.html"


LABELS = {"payload_kg": ("payload", "kg"), "cog_x_mm": ("CoG x offset", "mm"),
          "cog_y_mm": ("CoG y offset", "mm"), "cog_z_mm": ("CoG z offset", "mm"),
          "speed_scale": ("speed", "×"), "accel_scale": ("acceleration", "×"),
          "decel_scale": ("deceleration", "×"), "torque_scale": ("motor torque", "×"),
          "supply_scale": ("supply voltage", "×"), "friction_scale": ("friction", "×"),
          "backlash_scale": ("backlash", "×"), "stiffness_scale": ("gearbox stiffness", "×"),
          "shift_x_mm": ("fixture shift x", "mm"), "shift_y_mm": ("fixture shift y", "mm"),
          "shift_z_mm": ("fixture shift z", "mm"), "reach_scale": ("reach", "×"),
          "ff_scale": ("payload as configured", "×"), "gain_scale": ("servo gain", "×")}


def _fmt_cause(cause):
    if not cause or not cause.get("reproduced"):
        return None
    parts = []
    for t in cause["thresholds"]:
        op = "≥" if t["direction"] == "above" else "≤"
        name, unit = LABELS.get(t["name"], (t["name"], ""))
        v = t["threshold"]
        num = f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
        parts.append(f"{name} {op} {'×' + num if unit == '×' else num + ' ' + unit}")
    return ", ".join(parts)


def _arm_card(robot, prio, fn, prog, search, fam, cause, env, rpt, cal):
    lines = [f"{search['n_worlds']} conditions searched, {len(search['families'])} failure "
             f"families"]
    if fam:
        c = _fmt_cause(cause)
        lines.append(f"Most plausible failure: {fam['name']}" + (f" (needs {c})" if c else ""))
    fails = [c for c in env["cells"] if c["failed"]]
    lines.append(f"Envelope payload × acceleration: {len(fails)} of {len(env['cells'])} cells fail")
    lines.append(f"10 cycles at nominal: drift {rpt['metrics']['repeat_drift_mm']:.3f} mm")
    b, a = cal["before"]["joint_rms_deg"], cal["after"]["joint_rms_deg"]
    lines.append(f"Calibration (synthetic log): joint error {b:.3f}° → {a:.3f}°")
    return {"name": robot.name, "priority": prio, "product": "Stress Lab", "href": fn,
            "lines": lines, "pending": False, "program": prog.name}


def _hub(cards) -> str:
    def card(c):
        body = "".join(f"<li>{html.escape(x)}</li>" for x in c["lines"])
        link = (f'<a class="open" href="{html.escape(c["href"])}">Open demo →</a>' if c["href"]
                else '<span class="todo">Not built yet</span>')
        return (f'<article class="{"pending" if c["pending"] else ""}"><header><span class="p">'
                f'{c["priority"]}</span><span class="prod">{html.escape(c["product"])}</span></header>'
                f'<h2>{html.escape(c["name"])}</h2><ul>{body}</ul>{link}</article>')
    built = time.strftime("%d %B %Y")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BreakPoint demos</title><style>
:root {{ --bg:#0d0f12; --panel:#14171b; --line:#262b32; --text:#e7e9ec; --muted:#8d949e;
  --accent:#3d82ff; --warn:#f5a524; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:48px 16px 64px; }}
h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:-.01em; }}
.lede {{ color:var(--muted); max-width:720px; margin:0 0 32px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:14px; }}
article {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px;
  display:flex; flex-direction:column; }}
article.pending {{ opacity:.75; }}
header {{ display:flex; gap:8px; font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
.p {{ color:var(--warn); font-weight:700; }} .prod {{ color:var(--muted); }}
h2 {{ font-size:17px; margin:8px 0 10px; }}
ul {{ margin:0 0 16px; padding-left:18px; color:var(--muted); font-size:13px; flex:1; }}
li {{ margin-bottom:4px; word-break:break-word; }}
a.open {{ color:#fff; background:var(--accent); text-decoration:none; padding:8px 12px;
  border-radius:6px; font-weight:600; align-self:flex-start; }}
.todo {{ color:var(--muted); font-size:12px; }}
footer {{ margin-top:40px; color:var(--muted); font-size:12px; border-top:1px solid var(--line);
  padding-top:16px; }}
</style></head><body><main>
<h1>BreakPoint demos</h1>
<p class="lede">Stress tests of real robot platforms in simulation: the conditions where each
one stops doing its job, the smallest change that causes it, and how the simulator is
calibrated against a robot's own logs. Each demo is a self-contained interactive page.</p>
<div class="grid">{''.join(card(c) for c in cards)}</div>
<footer>Built {built}. Simulation only: results come from models of each robot built from the
makers' published descriptions; drive-train values they do not publish are typical values
until calibrated against a real log. Calibration examples use SYNTHETIC logs and say so.
</footer></main></body></html>"""
