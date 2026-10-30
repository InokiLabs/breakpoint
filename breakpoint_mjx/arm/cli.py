"""``breakpoint arm ...``: the AR4 Stress Lab from the command line."""
from __future__ import annotations

import json
from pathlib import Path


def _wm(path):
    if not path:
        return None
    from breakpoint_mjx.arm.world_model import WorldModel
    return WorldModel.from_json(Path(path).read_text())


def cmd_serve(a):
    from breakpoint_mjx.arm.server import serve
    serve(a.workspace, a.host, a.port, a.workers, open_browser=not a.no_browser)
    return 0


def cmd_check(a):
    from breakpoint_mjx.arm.program import load_program
    from breakpoint_mjx.arm.sim import simulate, taught_offsets
    prog = load_program(a.program)
    wm = _wm(a.world_model)
    st = {**(wm.nominal_stress() if wm else {}), **json.loads(a.stress or "{}")}
    r = simulate(prog, st, world_model=wm, keep_trace=False,
                 taught=taught_offsets(prog, world_model=wm))
    m = r["metrics"]
    print(f"{prog.name}: {'FAIL (' + r['dominant'] + ')' if r['failed'] else 'PASS'}  "
          f"score {r['failure_score']:.3f}")
    print(f"  off taught point {m['max_tcp_error_mm']:.3f} mm (encoder-blind "
          f"{m['max_hidden_error_mm']:.3f} mm), drift {m['repeat_drift_mm']:.3f} mm, "
          f"lost steps {m['lost_steps']}, peak torque {m['peak_torque_util'] * 100:.0f}%")
    for e in r["events"]:
        print("  -", e)
    for i in r["issues"]:
        print("  ! ", i["detail"])
    return 1 if r["failed"] else 0


def cmd_stress(a):
    from breakpoint_mjx.arm.program import load_program
    from breakpoint_mjx.arm.search import Runner, minimal_cause, run_search, stress_space
    prog = load_program(a.program)
    wm = _wm(a.world_model)
    with Runner(a.workers) as rn:
        out = run_search(prog, budget=a.budget, seed=a.seed, world_model=wm, runner=rn,
                         progress=lambda d, n, s: print(f"\r  {s}: {d}/{n}", end="", flush=True))
        print()
        if a.isolate:
            sp = stress_space(prog, world_model=wm)
            for f in out["families"][:a.isolate]:
                f["minimal_cause"] = minimal_cause(prog, f["theta"], sp, world_model=wm, runner=rn)
    print(f"{out['n_worlds']} conditions, {out['n_failed']} failed, "
          f"{len(out['families'])} families, {out['seconds']} s")
    for f in out["families"]:
        dev = ", ".join(f"{d['name']}={d['value']:g}" for d in f["deviations"][:4])
        print(f"  [{f['plausibility']:.2f}] {f['name']} ({f['count']}): {dev}")
        mc = f.get("minimal_cause")
        if mc and mc.get("reproduced"):
            for t in mc["thresholds"]:
                op = ">=" if t["direction"] == "above" else "<="
                print(f"        cause: {t['name']} {op} {t['threshold']:g}"
                      f"{'' if t['within_plausible'] else '  (outside plausible band)'}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {a.out}")
    return 1 if out["families"] else 0


def cmd_envelope(a):
    from breakpoint_mjx.arm.program import load_program
    from breakpoint_mjx.arm.search import envelope
    prog = load_program(a.program)
    out = envelope(prog, a.x, a.y, nx=a.n, ny=a.n, world_model=_wm(a.world_model))
    nx = len(out["xs"])
    print(f"{a.y} (rows, top = high) x {a.x} (columns): . pass, X fail")
    for iy in reversed(range(len(out["ys"]))):
        row = out["cells"][iy * nx:(iy + 1) * nx]
        print(f"  {out['ys'][iy]:8.3g} " + " ".join("X" if c["failed"] else "." for c in row))
    print("           " + " ".join(f"{v:.2g}"[:1] for v in out["xs"]))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


def cmd_calibrate(a):
    from breakpoint_mjx.arm import world_model as WM
    log = WM.log_from_csv(Path(a.log).read_text())
    wm = WM.fit(log, model=a.model, fit_payload=a.fit_payload,
                progress=lambda d, n, s: print(f"\r  {s}: {d}/{n}", end="", flush=True))
    print()
    r = wm.report
    print(f"joint RMS {r['before']['joint_rms_deg']:.4f} -> {r['after']['joint_rms_deg']:.4f} deg")
    if "tcp_rms_mm" in r["before"]:
        print(f"TCP RMS   {r['before']['tcp_rms_mm']:.3f} -> {r['after']['tcp_rms_mm']:.3f} mm")
    for k, v in wm.params.items():
        print(f"  {k:16s} {v:8.3f}   {r['identifiability'][k]['status']}")
    Path(a.out).write_text(wm.to_json())
    print(f"wrote {a.out}")
    return 0


def cmd_synth(a):
    from breakpoint_mjx.arm import world_model as WM
    from breakpoint_mjx.arm.program import load_program
    Path(a.out).write_text(WM.log_to_csv(WM.synthetic_log(load_program(a.program))))
    print(f"wrote {a.out} (SYNTHETIC: a simulated stand-in for a real AR4 log)")
    return 0


def cmd_demo_pack(a):
    from breakpoint_mjx.arm.demo_pack import build
    reports = dict(x.split("=", 1) for x in (a.policy_report or []))
    print(f"wrote {build(a.out, budget=a.budget, policy_reports=reports, workers=a.workers)}")
    return 0


def cmd_meshes(a):
    from breakpoint_mjx.arm.meshes import fetch
    print(f"wrote {fetch(Path(a.out))}")
    return 0


def add_parser(sub):
    arm = sub.add_parser("arm", help="AR4 Stress Lab: stress-test programmed arm motion")
    s = arm.add_subparsers(dest="arm_cmd", required=True)

    sv = s.add_parser("serve", help="open the Stress Lab web app")
    sv.add_argument("--workspace", default="~/.breakpoint/ar4")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8740)
    sv.add_argument("--workers", type=int, default=None, help="simulation processes")
    sv.add_argument("--no-browser", action="store_true")
    sv.set_defaults(func=cmd_serve)

    ck = s.add_parser("check", help="run a program once and print the verdict")
    ck.add_argument("program", help="built-in name (pick_and_place, ...) or a YAML path")
    ck.add_argument("--stress", default=None, help='JSON, e.g. \'{"payload_kg": 1.5}\'')
    ck.add_argument("--world-model", default=None)
    ck.set_defaults(func=cmd_check)

    st = s.add_parser("stress", help="search for the conditions where a program fails")
    st.add_argument("program")
    st.add_argument("--budget", default="quick", help="quick | standard | deep | a number")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--isolate", type=int, default=0, metavar="N",
                    help="find the minimal cause of the top N families")
    st.add_argument("--world-model", default=None)
    st.add_argument("--workers", type=int, default=None)
    st.add_argument("--out", default=None)
    st.set_defaults(func=cmd_stress)

    ev = s.add_parser("envelope", help="map pass/fail over two stress axes")
    ev.add_argument("program")
    ev.add_argument("--x", default="payload_kg")
    ev.add_argument("--y", default="accel_scale")
    ev.add_argument("--n", type=int, default=7)
    ev.add_argument("--world-model", default=None)
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=cmd_envelope)

    cb = s.add_parser("calibrate", help="fit a world model to a robot log (CSV)")
    cb.add_argument("log")
    cb.add_argument("--model", default="mk5")
    cb.add_argument("--fit-payload", action="store_true")
    cb.add_argument("--out", default="world_model.json")
    cb.set_defaults(func=cmd_calibrate)

    sy = s.add_parser("synth-log", help="write a SYNTHETIC robot log for trying calibrate")
    sy.add_argument("program")
    sy.add_argument("--out", default="synthetic_log.csv")
    sy.set_defaults(func=cmd_synth)

    dp = s.add_parser("demo-pack", help="build a shareable folder of demos, one per robot")
    dp.add_argument("--out", default="demo_pack")
    dp.add_argument("--budget", default="quick")
    dp.add_argument("--policy-report", action="append", metavar="ROBOT=REPORT.html",
                    help="engine report to include, e.g. g1=out/g1/report.html")
    dp.add_argument("--workers", type=int, default=None)
    dp.set_defaults(func=cmd_demo_pack)

    fm = s.add_parser("fetch-meshes", help="re-download and decimate the AR4 meshes")
    fm.add_argument("--out", default=str(Path(__file__).parent / "web" / "meshes" / "ar4"))
    fm.set_defaults(func=cmd_meshes)
