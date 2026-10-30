"""`breakpoint` - the installable CLI. Point it at a robot and a policy, get a
failure report.

    pip install "breakpoint-mjx[sim] @ git+https://github.com/InokiLabs/breakpoint"
    breakpoint doctor
    breakpoint demo                                   # bundled Go1 policy, smoke budget
    breakpoint check-policy --robot go1 --policy params:./weights.npz
    breakpoint run --robot go1 --policy params:./weights.npz --budget standard
    breakpoint run --robot ./my_robot.urdf --policy onnx:./policy.onnx
    breakpoint init --from rsl_rl ./logs/rsl_rl/go1_flat/2026-01-01 --robot go1
    breakpoint check-policy --robot go1 --policy ./policy.onnx --infer-contract
    breakpoint run --robot go1 --policy ./policy.onnx --obs-spec contract.yaml
    breakpoint baseline --target go1 --budget 384 --seeds 2 --preset smoke
    breakpoint rate --robot go1 --policy ./go1_policy.pkl --battery legged-quadruped-v1
    breakpoint keys new --out ~/.breakpoint/keys
    breakpoint sign report.json --key ~/.breakpoint/keys/breakpoint.key
    breakpoint verify report.json --pubkey breakpoint.pub [--policy ckpt.pkl]

Design intent: the first run should work without reading documentation, on a
laptop, without a Colab notebook, a GitHub token, or a paste-back loop. Every
command that can fail for an environmental reason (no GPU, missing dependency,
unknown robot) says exactly what to do next instead of printing a traceback.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from breakpoint_mjx import corpus, tier

__version__ = "0.1.0"

_CONSENT_LABEL = {None: "not asked yet", True: "yes", False: "no"}


def _banner(t=None):
    return f"breakpoint {__version__}  [{t or tier.resolve_tier()}]"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _find_script(name):
    """Locate a repo script for the entries that have not moved into the package yet.

    The PIPELINE no longer goes through here: it lives at breakpoint_mjx.pipeline and
    is invoked as a module, so a wheel-only install can run a search. Hardening is
    still a repo script, so this resolver stays for that until it moves too.
    """
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "scripts" / name, Path.cwd() / "scripts" / name):
        if cand.exists():
            return cand
    return None


def _pipeline_cmd(extra):
    """Invoke the pipeline as an installed MODULE, not a file path.

    `-m` works identically from a repo checkout and from a wheel, which is the whole
    point of having moved it into the package.
    """
    return [sys.executable, "-m", "breakpoint_mjx.pipeline", *extra]


def _check_policy_cmd(robot, policy, n=16, obs_spec=None):
    """The nominal-behavior preflight, run as its own process so the JAX/MJX memory it
    uses is released before the (much larger) pipeline process starts."""
    cmd = [sys.executable, "-m", "breakpoint_mjx.check_policy",
           "--robot", str(robot), "--policy", str(policy), "--n", str(n)]
    if obs_spec:
        cmd += ["--obs-spec", str(obs_spec)]
    return cmd


def _exit_hint(rc):
    """One actionable line for a non-zero pipeline exit code."""
    if rc < 0:
        return (f"the pipeline was killed by signal {-rc}. If nothing else stopped it, the "
                f"likely cause is running out of memory: try --budget smoke.")
    if rc in (3221225786, 130):  # Windows CTRL_C_EXIT / POSIX SIGINT convention
        return "the pipeline was interrupted."
    if rc in (3221225477, 3221226505):  # Windows access violation / stack buffer overrun
        return ("the pipeline process crashed (native error). On a CPU-only machine this is "
                "usually memory: close other programs or use --budget smoke.")
    if rc > 255:  # a Windows NTSTATUS / TerminateProcess code (4294967295 = killed, -1)
        return (f"the pipeline process was killed or crashed natively (status "
                f"0x{rc & 0xFFFFFFFF:08X}). If you did not stop it, the likely cause is "
                f"running out of memory: try --budget smoke.")
    return ("the traceback above says why. Common fixes: `breakpoint doctor`, "
            "`breakpoint check-policy`, or a smaller --budget.")


def _exit_code(rc):
    """Map a child's return code to a valid, NON-ZERO exit status for this process.

    POSIX signals come back negative (-9) and map to the shell's 128+N; Windows kill and
    crash statuses are 32-bit values (4294967295 after a forced kill) that some shells
    truncate to 8 bits, so they map to 1 rather than risk wrapping to 0.
    """
    if rc < 0:
        return 128 + (-rc)
    if rc > 255:
        return 1
    return rc


def _budget_env(env, budget):
    """Export a --budget preset (and its restart count) into the child environment."""
    if not budget:
        return env
    from breakpoint_mjx.config import BUDGET_RESTARTS

    env["BP_BUDGET"] = budget
    if budget in BUDGET_RESTARTS and "BP_RESTARTS" not in env:
        env["BP_RESTARTS"] = str(BUDGET_RESTARTS[budget])
    return env


def _is_robot_file(spec):
    return str(spec).lower().endswith((".urdf", ".xacro", ".xml", ".mjcf"))


def _print_kv(rows, indent="  "):
    if not rows:
        return
    w = max(len(str(k)) for k, _ in rows)
    for k, v in rows:
        print(f"{indent}{str(k).ljust(w)}  {v}")


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def _probe(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def cmd_doctor(args):
    print(_banner())
    print("\nenvironment")

    jax_v = _probe("jax")
    rows = [
        ("python", sys.version.split()[0]),
        ("jax", jax_v or "MISSING (pip install jax[cuda12])"),
        ("mujoco", _probe("mujoco") or "MISSING (pip install mujoco)"),
        ("mujoco_playground", "ok" if _probe("mujoco_playground") else "MISSING"),
        ("brax", _probe("brax") or "MISSING"),
    ]

    devices, gpu = "unknown", False
    if jax_v:
        try:
            import jax

            devices = ", ".join(sorted({d.platform for d in jax.devices()}))
            gpu = any(d.platform in ("gpu", "cuda", "rocm") for d in jax.devices())
        except Exception as e:  # noqa: BLE001 - report it, never crash the doctor
            devices = f"jax present but devices unavailable: {type(e).__name__}"
    rows.append(("compute", devices + ("" if gpu else
                 "   <- CPU only: fine for a smoke run, too slow for a real search")))

    import importlib.util

    from breakpoint_mjx.paths import configs_dir

    rows.append(("pipeline", "breakpoint_mjx.pipeline"
                 if importlib.util.find_spec("breakpoint_mjx.pipeline")
                 else "MISSING - reinstall breakpoint-mjx"))
    try:
        rows.append(("configs", str(configs_dir())))
    except FileNotFoundError as e:
        rows.append(("configs", f"MISSING - {e}"))
    _print_kv(rows)

    print("\nmemory")
    from breakpoint_mjx import stagecache

    mem = stagecache.free_memory()

    def _gb(v):
        return "unknown" if v is None else f"{v / 1024:.1f} GB"

    _print_kv([("free RAM", f"{_gb(mem['free_ram_mb'])} of {_gb(mem['total_ram_mb'])}"),
               ("free commit", f"{_gb(mem['free_commit_mb'])} of "
                               f"{_gb(mem['commit_limit_mb'])} (RAM + pagefile)"),
               ("demo peak", f"{_gb(stagecache.DEMO_PEAK_RSS_MB)} RAM, "
                             f"{_gb(stagecache.DEMO_PEAK_COMMIT_MB)} commit (measured, "
                             f"CPU smoke demo; see README)"),
               ("source", mem.get("source") or "unavailable")])
    warn = stagecache.memory_warning(mem)
    if warn:
        print(f"  {warn}")

    print("\nlicence")
    d = tier.describe()
    _print_kv([("tier", d["tier"]), ("usage", d["usage"]),
               ("unlocked", ", ".join(d["unlocked"])),
               ("locked", ", ".join(d["locked"]) or "-")])

    print("\ncorpus")
    _print_kv([("contribute", _CONSENT_LABEL[corpus.read_consent()])])

    problems = [k for k, v in rows if isinstance(v, str) and "MISSING" in v]
    if problems:
        print(f"\n{len(problems)} thing(s) to fix: {', '.join(problems)}")
        return 1
    print("\nready.")
    return 0


# --------------------------------------------------------------------------
# licence
# --------------------------------------------------------------------------

def cmd_activate(args):
    try:
        p = tier.save_key(args.key)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"activated ({tier.resolve_tier()}). key stored at {p}")
    return 0


def cmd_license(args):
    d = tier.describe()
    print(_banner(d["tier"]))
    _print_kv([("usage", d["usage"]), ("unlocked", ", ".join(d["unlocked"])),
               ("locked", ", ".join(d["locked"]) or "-")])
    if d["tier"] == tier.FREE:
        print("\nThe open-source edition finds and reports failures with no cap. Hardening")
        print("a policy and issuing a certificate are commercial-edition features.")
    return 0


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

_PREVIEW_RECORD = {
    "theta_named": {"friction_floor": 0.42, "latency_steps": 3.0},
    "failure_score": 0.91,
    "component_signals": {"fall": 1.0},
    "severity": "critical",
    "transfer_score": 0.88,
    "family_id": "fam-01",
}


def cmd_corpus(args):
    if args.corpus_cmd == "status":
        _print_kv([("contribute", _CONSENT_LABEL[corpus.read_consent()]),
                   ("manifest", corpus.MANIFEST_VERSION),
                   ("outbox", corpus.outbox_dir())])
        return 0

    if args.corpus_cmd == "consent":
        corpus.write_consent(args.answer == "yes")
        print(f"corpus contribution set to {args.answer}")
        return 0

    if args.corpus_cmd == "preview":
        payload = corpus.build_payload(
            [_PREVIEW_RECORD], robot_class="quadruped", dof=12, total_mass_kg=12.7,
            engine_version=__version__, search_mode="agentic", horizon=500)
        print(corpus.CONSENT_PROMPT)
        print("\nexactly what one contributed finding looks like:\n")
        print(json.dumps(payload, indent=2))
        return 0

    if args.corpus_cmd == "export":
        src = Path(args.failures)
        if not src.exists():
            print(f"error: no failures file at {src}", file=sys.stderr)
            return 2
        data = json.loads(src.read_text(encoding="utf-8"))
        records = data.get("records", data) if isinstance(data, dict) else data
        payload = corpus.build_payload(
            records, robot_class=args.robot_class, dof=args.dof,
            total_mass_kg=args.mass, engine_version=__version__,
            search_mode=args.search_mode, horizon=args.horizon)
        out = corpus.stage(payload)
        print(f"staged {len(payload['findings'])} findings -> {out}")
        print("nothing was uploaded: there is no corpus endpoint yet, so this is the")
        print("file you can inspect and send yourself.")
        return 0
    return 2


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _resolve_consent(explicit=None):
    """Ask once, on the first real run. Declining leaves every feature working."""
    if explicit is not None:
        corpus.write_consent(explicit)
        return explicit
    existing = corpus.read_consent()
    if existing is not None:
        return existing
    if not sys.stdin.isatty():
        return False  # never opt a non-interactive session in by default
    print(corpus.CONSENT_PROMPT)
    val = input("\ncontribute? [y/N] ").strip().lower() in ("y", "yes")
    corpus.write_consent(val)
    return val


def _missing_policy(args):
    """Return an actionable message if there is no policy to test, else None.

    BreakPoint tests a FROZEN policy, so 'which policy?' is the one question a first
    run cannot proceed without. Either --policy points at a BYO export, or a trained
    checkpoint is already sitting in the output directory.
    """
    if args.policy:
        return None
    if _is_robot_file(args.robot):
        # A robot you brought yourself has no trained checkpoint anywhere by
        # definition, so --policy is not optional on this path.
        return (f"error: testing {Path(args.robot).name} needs a policy to test.\n"
                f"A robot supplied as a URDF has no trained checkpoint, so point at "
                f"your own export:\n"
                f"  breakpoint run --robot {args.robot} --policy onnx:/path/policy.onnx\n"
                f"  breakpoint run --robot {args.robot} --policy params:/path/weights.npz\n"
                f"Note: BreakPoint builds its own observation layout from the model. Run\n"
                f"`python -c \"from breakpoint_mjx.urdf_target import load_urdf_model, "
                f"format_report; print(format_report(load_urdf_model(r'{args.robot}')[1]))\"`\n"
                f"to see it, and map your policy's ordering with an ObsSpec if it differs.")
    try:
        from breakpoint_mjx.targets import get_target

        ckpt = Path(args.out).resolve() / args.robot / get_target(args.robot).checkpoint_name
    except Exception:  # noqa: BLE001 - unknown target is reported by the caller
        return None
    if ckpt.exists():
        return None
    return (f"error: no policy to test.\n"
            f"  looked for a trained checkpoint at: {ckpt}\n\n"
            f"Either point at your own policy export:\n"
            f"  breakpoint run --robot {args.robot} --policy onnx:/path/to/policy.onnx\n"
            f"  breakpoint run --robot {args.robot} --policy params:/path/to/weights.pkl\n"
            f"or train a baseline first:\n"
            f"  python scripts/train.py --target {args.robot}")


def cmd_run(args):
    t = tier.resolve_tier()
    print(_banner(t), flush=True)   # flush: errors go to stderr and must not jump ahead
    from breakpoint_mjx import stagecache

    warn = stagecache.memory_warning()
    if warn and not getattr(args, "low_mem", False):
        print(warn, flush=True)

    import importlib.util

    if importlib.util.find_spec("breakpoint_mjx.pipeline") is None:
        print("error: breakpoint_mjx.pipeline is missing from this install.\n"
              "Reinstall with: pip install --force-reinstall breakpoint-mjx[sim]",
              file=sys.stderr)
        return 2

    if _is_robot_file(args.robot):
        if not Path(args.robot).exists():
            print(f"error: no such robot description: {args.robot}", file=sys.stderr)
            return 2
        try:
            import breakpoint_mjx.urdf_target  # noqa: F401
        except ImportError:
            print("error: URDF ingestion is unavailable in this build "
                  "(breakpoint_mjx.urdf_target is missing).", file=sys.stderr)
            return 2
    else:
        from breakpoint_mjx.targets import TARGETS

        if args.robot not in TARGETS:
            print(f"error: unknown robot {args.robot!r}.\n"
                  f"registered: {', '.join(sorted(TARGETS))}\n"
                  f"or pass a path to a .urdf file.", file=sys.stderr)
            return 2

    # Preflight the policy BEFORE spawning the pipeline. Without this the user waits
    # through a slow env load only to hit a bare FileNotFoundError, which is exactly
    # the first-run experience this CLI exists to remove.
    missing = _missing_policy(args)
    if missing:
        print(missing, file=sys.stderr)
        return 2

    explicit = True if args.contribute else (False if args.no_contribute else None)
    contribute = _resolve_consent(explicit)
    if contribute and t == tier.OEM:
        print("note: commercial runs are private by default; not contributing.")
        contribute = False

    obs_spec = getattr(args, "obs_spec", None)
    if obs_spec:
        if not args.policy:
            print("error: --obs-spec describes a policy's interface; it needs --policy",
                  file=sys.stderr)
            return 2
        if not Path(obs_spec).exists():
            print(f"error: no contract file at {obs_spec}", file=sys.stderr)
            return 2
        obs_spec = str(Path(obs_spec).resolve())

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # PYTHONUNBUFFERED: the children write to this console; unbuffered keeps their lines
    # in order with ours and visible live instead of in bursts.
    env = dict(os.environ, BP_OUT=str(out), BP_SEARCH=args.search, PYTHONUNBUFFERED="1")
    budget = getattr(args, "budget", None)
    _budget_env(env, budget)
    if obs_spec:
        env["BP_OBS_SPEC"] = obs_spec     # read by adapters.build_policy in the pipeline
    if getattr(args, "low_mem", False):
        env["BP_LOWMEM"] = "1"            # pipeline skips mp4 rendering (stagecache)
    if args.policy:
        env["BP_POLICY"] = args.policy
        # Nominal-behavior preflight. A BYO policy fed the wrong obs order or scale
        # still runs; it just falls everywhere and the report is garbage that looks
        # real. Refuse before spending the search budget on it.
        if getattr(args, "skip_check", False):
            print("preflight: skipped (--skip-check)", flush=True)
        elif _is_robot_file(args.robot):
            print("preflight: check-policy supports registered robots only; skipped for "
                  "a URDF robot", flush=True)
        else:
            ccmd = _check_policy_cmd(args.robot, args.policy, obs_spec=obs_spec)
            print(f"\npreflight: {' '.join(ccmd)}", flush=True)
            crc = subprocess.run(ccmd, env=env).returncode
            if crc != 0:
                print(f"\npreflight failed (exit {crc}): the policy does not walk at "
                      f"nominal, so a search would only report noise.\n"
                      f"Fix the export (see the diagnosis above), let BreakPoint infer "
                      f"the policy's obs/action contract with\n"
                      f"  breakpoint check-policy --robot {args.robot} --policy "
                      f"{args.policy} --infer-contract\n"
                      f"or pass --skip-check to search anyway.", file=sys.stderr)
                return _exit_code(crc)
    if args.autolytic > 0:
        env["BP_AUTOLYTIC"] = "1"
        env["BP_AUTOLYTIC_TOP"] = str(args.autolytic)
    if getattr(args, "risk", 0) > 0:
        env["BP_RISK"] = "1"
        env["BP_RISK_N"] = str(args.risk)
        if getattr(args, "risk_dist", None):
            env["BP_RISK_DIST"] = str(Path(args.risk_dist).resolve())
    if getattr(args, "train_dr", None):
        env["BP_TRAIN_DR"] = str(Path(args.train_dr).resolve())   # training_gap.py

    # A URDF run derives its target from the model itself; a registered name uses the
    # Playground env. Either way the run dir is namespaced by the same label.
    if _is_robot_file(args.robot):
        label = Path(args.robot).stem
        cmd = _pipeline_cmd(["--urdf", str(Path(args.robot).resolve())])
    else:
        label = args.robot
        cmd = _pipeline_cmd(["--target", args.robot])

    # flush: the child writes straight to the same console, so without it these lines
    # can appear AFTER the pipeline's own output.
    print(f"\nrunning: {' '.join(cmd)}", flush=True)
    print(f"  out={out}  search={args.search}  budget={env.get('BP_BUDGET') or 'full'}"
          f"  contribute={contribute}\n", flush=True)
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except KeyboardInterrupt:
        rc = 130
    if rc != 0:
        print(f"\npipeline exited {rc}: {_exit_hint(rc)}", file=sys.stderr, flush=True)
        print(_recovery_hint(out / label, args), file=sys.stderr, flush=True)
        # A negative rc (killed by a signal) is not a valid process exit status; map it
        # to the shell convention 128+N so callers always see non-zero.
        return _exit_code(rc)

    report = out / label / "report.html"
    print(f"\ndone. report: {report if report.exists() else out}")
    if contribute:
        failures = out / label / "failures.json"
        print(f"corpus: breakpoint corpus export --failures {failures} ...")
    if t == tier.FREE:
        print("\nfound the failures. fixing them (hardening + certificate) is the")
        print("commercial edition: breakpoint activate <key>")
    return 0


def _resume_command(args):
    """The command that resumes this run (same out dir: the search and every finished
    evidence stage are reused)."""
    if getattr(args, "_demo", False):
        cmd = ["breakpoint", "demo", "--out", str(args.out)]
        if getattr(args, "budget", None) and args.budget != "smoke":
            cmd += ["--budget", args.budget]
    else:
        cmd = ["breakpoint", "run", "--robot", str(args.robot), "--out", str(args.out),
               "--search", str(args.search)]
        for flag, val in (("--policy", args.policy),
                          ("--budget", getattr(args, "budget", None)),
                          ("--obs-spec", getattr(args, "obs_spec", None))):
            if val:
                cmd += [flag, str(val)]
    if getattr(args, "low_mem", False):
        cmd.append("--low-mem")
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def _recovery_hint(run_dir, args):
    """What survived a crash and how to pick up from it."""
    from breakpoint_mjx import stagecache

    lines = []
    report = run_dir / "report.html"
    if report.exists():
        lines.append(f"partial report (evidence complete, videos may be missing): {report}")
    stages = [n for n in stagecache.STAGES if Path(stagecache.stage_path(run_dir, n)).exists()]
    if stages:
        lines.append(f"finished stages saved: {', '.join(stages)} (in {run_dir})")
    if (run_dir / "failures.json").exists():
        cmd = _resume_command(args)
        lines.append(f"resume (skips the search and the saved stages):\n  {cmd}")
        if "--low-mem" not in cmd:
            lines.append("or without mp4 rendering, the step where a CPU box runs out of "
                         f"memory:\n  {cmd} --low-mem")
    else:
        lines.append("nothing to resume yet (the search did not finish); rerun the same "
                     "command, with --budget smoke on a small machine")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# demo + check-policy
# --------------------------------------------------------------------------

def demo_asset():
    """Path of the bundled Go1 demo checkpoint (package data)."""
    return Path(__file__).resolve().parent / "assets" / "demo" / "go1_policy.pkl"


def sample_report_path():
    """The bundled, measured CPU smoke-run report (package data)."""
    return (Path(__file__).resolve().parent / "assets" / "demo" / "sample_report"
            / "report.html")


def cmd_demo_view(args):
    """`breakpoint demo --view`: the bundled sample report in seconds, no JAX, no run."""
    p = sample_report_path()
    if not p.exists():
        print(f"error: the bundled sample report is missing from this install ({p})",
              file=sys.stderr)
        return 2
    print(f"sample report (a measured CPU smoke run of the bundled Go1 policy; not a GPU "
          f"or standard-budget result):\n  {p}")
    print("standard budget on a GPU: colab/evidence_cell.py (UNVERIFIED on this build)")
    if not getattr(args, "no_open", False) and os.environ.get("BP_NO_BROWSER") != "1":
        try:
            import webbrowser

            webbrowser.open(p.as_uri())
        except Exception:  # noqa: BLE001 -- printing the path is enough
            pass
    return 0


def cmd_demo(args):
    """First result with nothing brought: the bundled Go1 policy, smoke budget, cmaes."""
    if getattr(args, "view", False):
        return cmd_demo_view(args)
    src = demo_asset()
    if not src.exists():
        print(f"error: the bundled demo policy is not in this install ({src}).\n"
              f"Train one (python scripts/train.py --target go1) or bring your own, then:\n"
              f"  breakpoint run --robot go1 --policy <checkpoint>", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    dst = out / "go1" / "go1_policy.pkl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if (dst.parent / "failures.json").exists():
        print(f"note: reusing the previous search in {dst.parent} (delete that folder "
              f"to search again)", flush=True)
    print(f"demo: bundled Go1 policy -> {dst}\n"
          f"      budget={args.budget}, search=cmaes. On a CPU-only laptop the smoke "
          f"budget takes a few minutes;\n      it is a tour of the product, not a "
          f"converged search (use --budget standard or a GPU for that).", flush=True)
    run_args = argparse.Namespace(
        robot="go1", policy=None, out=str(out), search="cmaes", autolytic=0, risk=0,
        risk_dist=None, contribute=False, no_contribute=True, budget=args.budget,
        skip_check=True, obs_spec=None, low_mem=getattr(args, "low_mem", False),
        train_dr=str(demo_asset().with_name("go1_training_dr.yaml")), _demo=True)
    rc = cmd_run(run_args)
    if rc == 0:
        report = dst.parent / "report.html"
        print(f"\nopen the report: {report}" if report.exists() else
              f"\nno report.html was written under {dst.parent}")
    return rc


def cmd_check_policy(args):
    from breakpoint_mjx.check_policy import check

    return check(args)


# --------------------------------------------------------------------------
# init: a framework's training config -> BreakPoint contract YAML
# --------------------------------------------------------------------------

def cmd_init(args):
    """Write a contract YAML from the config a training framework saved next to the
    policy, and print every field BreakPoint cannot honor. Refuses (exit 2) when the
    policy needs inputs the target cannot produce."""
    from breakpoint_mjx.adapters.importers import IMPORTERS
    from breakpoint_mjx.adapters.obs_spec import build_contract

    src = Path(args.path)
    if not src.exists():
        print(f"error: no such file or directory: {src}", file=sys.stderr)
        return 2
    names = [n.strip() for n in args.joint_names.split(",")] if args.joint_names else None
    try:
        res = IMPORTERS[args.source](src, robot=args.robot, joint_names=names)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"error: cannot import {src}: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    print(f"init: {args.source} config {res.source} -> {args.robot} contract")
    print("fields BreakPoint cannot honor exactly:")
    print(res.table())
    if res.hints:
        print("network hints: " + ", ".join(f"{k}={v}" for k, v in res.hints.items()))
    if res.fatal:
        print(f"\nnot written: {len(res.fatal)} fatal field(s) above. This policy needs "
              f"inputs the {args.robot} target does not produce.", file=sys.stderr)
        return 2
    try:
        c = build_contract(res.contract, robot=args.robot)
    except (KeyError, ValueError) as e:
        print(f"error: the imported contract does not resolve on {args.robot}: {e}",
              file=sys.stderr)
        return 2
    out = Path(args.out)
    res.to_yaml(out)
    print(f"\nwrote {out} ({c.obs_dim} obs channels, {c.action_dim} actions).")
    print(f"next: breakpoint check-policy --robot {args.robot} --policy <your policy> "
          f"--obs-spec {out}")
    print("      (add --infer-contract to let BreakPoint verify or correct it)")
    return 0


# --------------------------------------------------------------------------
# harden (commercial)
# --------------------------------------------------------------------------

def cmd_harden(args):
    tier.require("harden")
    script = _find_script("harden.py")
    if script is None:
        print("error: hardening is part of the BreakPoint commercial edition and is not "
              "included in this install (scripts/harden.py not found)",
              file=sys.stderr)
        return 2
    env = dict(os.environ, BP_OUT=str(Path(args.out).resolve()))
    return subprocess.run(
        [sys.executable, str(script), "--target", args.robot], env=env).returncode


def cmd_gate(args):
    """`breakpoint gate`: candidate vs baseline checkpoint on the same fixed conditions and
    seeds; exits 1 when the candidate falls where the baseline survived
    (breakpoint_mjx/gate.py)."""
    from breakpoint_mjx.gate import gate

    return gate(args)


def cmd_gate_boundary(args):
    """`breakpoint gate-boundary`: write a battery of conditions the baseline fails on some
    seeds, for `breakpoint gate --battery` (breakpoint_mjx/boundary.py)."""
    from breakpoint_mjx.boundary import build

    return build(args)


def cmd_baseline(args):
    """`breakpoint baseline`: directed search vs uniform (vs declared training DR) at an
    equal rollout budget, scored on transfer-free metrics (breakpoint_mjx/equal_budget.py)."""
    from breakpoint_mjx.equal_budget import run

    return run(args)


# --------------------------------------------------------------------------
# rate: a fixed crash-test battery (breakpoint_mjx/ratings.py)
# --------------------------------------------------------------------------

def cmd_rate(args):
    """`breakpoint rate`: run a versioned battery of named conditions on a policy and write
    the rating report JSON (breakpoint.report/v1, kind "rating", attestation null)."""
    from breakpoint_mjx import ratings

    opts = {k: getattr(args, k, None)
            for k in ("obs_norm", "action_scale", "squash", "activation", "mean_dim",
                      "obs_spec")}
    if opts["obs_spec"] is None and os.environ.get("BP_OBS_SPEC"):
        opts["obs_spec"] = os.environ["BP_OBS_SPEC"]
    try:
        rep = ratings.rate(args.robot, args.policy, battery_name=args.battery,
                           budget=args.budget, opts=opts, policy_name=args.policy_name,
                           policy_source=args.policy_source)
    except (ratings.BatteryError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"rate: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    except SystemExit as e:  # the BYO adapter raises SystemExit(str) on a dim mismatch
        if isinstance(e.code, str):
            print(f"rate: cannot load the policy: {e.code}", file=sys.stderr)
            return 2
        raise
    problems = ratings.validate_report(rep)
    if problems:
        print("rate: internal error, report violates the contract:\n  "
              + "\n  ".join(problems), file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(ratings.format_table(rep))
    print(f"\nwrote {out}")
    if rep["run"]["budget"] == "smoke":
        print("NOTE: --budget smoke is a plumbing preview (fewer seeds, shorter horizon). "
              "It is labelled smoke in the report and is not a rating.")
    return 0


# --------------------------------------------------------------------------
# entry

# --------------------------------------------------------------------------
# signed reports: keys / sign / verify
# --------------------------------------------------------------------------

def cmd_keys(args):
    from breakpoint_mjx import attest
    try:
        priv_p, pub_p, kid = attest.write_keypair(args.out, args.name)
    except attest.AttestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"key_id      {kid}")
    print(f"private key {priv_p}  (keep secret; never commit, never share)")
    print(f"public key  {pub_p}  (publish this so others can verify)")
    return 0


def cmd_sign(args):
    from breakpoint_mjx import attest
    try:
        report = attest.load_report(args.report)
        signed = attest.sign_report(report, attest.load_private_key(args.key),
                                    signer=args.signer)
        out = args.out or args.report
        attest.save_report(signed, out)
    except attest.AttestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    a = signed["attestation"]
    print(f"signed {out}  signer={a['signer']} key_id={a['key_id']} "
          f"payload_sha256={a['payload_sha256']}")
    return 0


def cmd_verify(args):
    from breakpoint_mjx import attest
    try:
        report = attest.load_report(args.report)
        pub = attest.load_public_key(args.pubkey)
    except attest.AttestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = attest.verify_report(report, pub)
    if res.ok and args.policy:
        pol = attest.check_policy_file(report, args.policy)
        if not pol.ok:
            res = pol
        else:
            res.reason += "; " + pol.reason
    print(f"{'PASS' if res.ok else 'FAIL'}: {res.reason}")
    return 0 if res.ok else 1


# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="breakpoint",
        description="Crash-test a robot control policy: find the physical conditions "
                    "where it fails, before the field does.")
    p.add_argument("--version", action="version", version=f"breakpoint {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check this machine can run a search").set_defaults(
        func=cmd_doctor)

    r = sub.add_parser("run", help="search for failures of a policy")
    r.add_argument("--robot", required=True,
                   help="registered name (go1, spot, g1, ...) or a path to a .urdf")
    r.add_argument("--policy", default=None,
                   help="onnx:/path/to/policy.onnx or params:/path/to/weights.pkl")
    r.add_argument("--out", default="out", help="output directory (default: ./out)")
    r.add_argument("--search", default="active",
                   choices=("random", "cmaes", "multi", "active", "llm", "agentic", "atlas"))
    r.add_argument("--autolytic", type=int, default=0, metavar="N",
                   help="measure autolytic gain on the top N failures: did the "
                        "perturbation break the policy, or did the policy break itself? "
                        "Costs a second rollout per measured world (default: off)")
    r.add_argument("--risk", type=int, default=0, metavar="N",
                   help="estimate the OPERATIONAL failure probability under the declared "
                        "deployment distribution (default: each dim's plausible band) with "
                        "an anytime-valid 95%% CI, spending up to N fresh rollouts; also "
                        "certifies the safety envelope with simulator rollouts (default: off)")
    r.add_argument("--risk-dist", dest="risk_dist", default=None, metavar="YAML",
                   help="YAML with an `operational:` block overriding the per-dim "
                        "deployment ranges ({dim: {range: [lo, hi]}} or {mean, sd})")
    r.add_argument("--train-dr", dest="train_dr", default=None, metavar="YAML",
                   help="YAML with a `training_dr:` block, {dim: {range: [lo, hi]}}: the "
                        "randomization the policy was TRAINED with. Families are then "
                        "split into defects (inside it) and coverage gaps (outside it), "
                        "with a suggested randomization patch. A contract from "
                        "`breakpoint init --from ...` already carries it")
    r.add_argument("--budget", default=None, choices=("smoke", "standard", "full"),
                   help="search size: smoke (minutes on a CPU laptop, a preview), standard "
                        "(a real search on a small GPU), full (search.yaml, the default). "
                        "Sets BP_BUDGET; BP_HORIZON / BP_N_EVAL / BP_GENERATIONS override "
                        "single knobs")
    r.add_argument("--skip-check", dest="skip_check", action="store_true",
                   help="skip the nominal-behavior preflight that runs when --policy is given")
    r.add_argument("--obs-spec", dest="obs_spec", default=None, metavar="YAML",
                   help="the policy's obs/action contract (from `breakpoint init` or "
                        "`check-policy --infer-contract`); exported as BP_OBS_SPEC")
    r.add_argument("--low-mem", dest="low_mem", action="store_true",
                   help="skip mp4 rendering (the step that exhausts RAM on small CPU "
                        "boxes); the report keeps every verdict. Also BP_LOWMEM=1")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--contribute", action="store_true", help="contribute to the corpus")
    g.add_argument("--no-contribute", action="store_true", help="do not contribute")
    r.set_defaults(func=cmd_run)

    dm = sub.add_parser("demo", help="run the bundled Go1 policy end to end (no files needed)")
    dm.add_argument("--out", default="breakpoint-demo",
                    help="output directory (default: ./breakpoint-demo)")
    dm.add_argument("--budget", default="smoke", choices=("smoke", "standard", "full"))
    dm.add_argument("--low-mem", dest="low_mem", action="store_true",
                    help="skip mp4 rendering (the step that exhausts RAM on small CPU "
                         "boxes); the report keeps every verdict. Also BP_LOWMEM=1")
    dm.add_argument("--view", action="store_true",
                    help="open the bundled sample report (a measured CPU smoke run) "
                         "without running anything")
    dm.add_argument("--no-open", dest="no_open", action="store_true",
                    help="with --view: print the path only, do not open a browser")
    dm.set_defaults(func=cmd_demo)

    from breakpoint_mjx.check_policy import add_args as _check_args

    cp = sub.add_parser("check-policy",
                        help="refuse a policy that cannot walk at nominal (exit 3)")
    _check_args(cp)
    cp.set_defaults(func=cmd_check_policy)

    ini = sub.add_parser("init", help="write a policy contract YAML from an Isaac Lab / "
                                      "RSL-RL / unitree_rl_gym config")
    ini.add_argument("--from", dest="source", required=True,
                     choices=("isaaclab", "rsl_rl", "unitree_rl_gym"),
                     help="isaaclab: params/env.yaml; rsl_rl: the run directory; "
                          "unitree_rl_gym: deploy/deploy_mujoco/configs/<robot>.yaml")
    ini.add_argument("path", help="the config file or run directory")
    ini.add_argument("--robot", default="go1", help="BreakPoint target (default go1)")
    ini.add_argument("--out", default="contract.yaml", help="default contract.yaml")
    ini.add_argument("--joint-names", dest="joint_names", default=None,
                     help="comma-separated joint names in the POLICY's order, when the "
                          "config does not list them")
    ini.set_defaults(func=cmd_init)

    rt = sub.add_parser("rate", help="crash rating: run a fixed, versioned battery of "
                                     "named conditions on a policy (simulation only)")
    rt.add_argument("--robot", required=True, help="go1, barkour (quadruped) or g1 (humanoid)")
    rt.add_argument("--policy", required=True,
                    help="a brax params .pkl, params:/w.npz or onnx:/p.onnx")
    rt.add_argument("--battery", default=None,
                    help="battery id (legged-quadruped-v1, legged-humanoid-v1) or a YAML "
                         "path; default: the robot class's v1 battery")
    rt.add_argument("--out", default="report.json", help="report JSON (default report.json)")
    rt.add_argument("--budget", default="full", choices=("full", "smoke"),
                    help="full: the battery as written (the rating); smoke: 4 seeds x 150 "
                         "steps, a plumbing preview labelled smoke in the report")
    rt.add_argument("--policy-name", dest="policy_name", default=None,
                    help="name shown in the report (default: the checkpoint file name)")
    rt.add_argument("--policy-source", dest="policy_source", default="local",
                    help="URL the checkpoint came from (default: local)")
    rt.add_argument("--obs-norm", dest="obs_norm", default=None,
                    help="BYO: npz with mean/std (also BP_OBS_NORM)")
    rt.add_argument("--action-scale", dest="action_scale", type=float, default=None,
                    help="BYO: scalar action scale (also BP_ACTION_SCALE)")
    rt.add_argument("--squash", action="store_true", help="BYO: tanh-squash the output")
    rt.add_argument("--activation", default=None, help="BYO MLP activation")
    rt.add_argument("--mean-dim", dest="mean_dim", type=int, default=None)
    rt.add_argument("--obs-spec", dest="obs_spec", default=None, metavar="YAML",
                    help="the policy's obs/action contract (also BP_OBS_SPEC)")
    rt.set_defaults(func=cmd_rate)

    from breakpoint_mjx.equal_budget import build_parser as _eb_parser

    bl = sub.add_parser("baseline", parents=[_eb_parser(add_help=False)],
                        help="equal-budget benchmark: directed search vs uniform random "
                             "at the same number of rollouts (simulation only)")
    bl.set_defaults(func=cmd_baseline)

    from breakpoint_mjx.gate import add_args as _gate_args

    g = sub.add_parser("gate", help="CI gate: fail when a candidate checkpoint falls in "
                                    "conditions the baseline survived")
    _gate_args(g)
    g.set_defaults(func=cmd_gate)

    from breakpoint_mjx.boundary import add_args as _boundary_args

    gb = sub.add_parser("gate-boundary", help="build a boundary battery for `gate` from the "
                                              "baseline: conditions it fails sometimes")
    _boundary_args(gb)
    gb.set_defaults(func=cmd_gate_boundary)

    h = sub.add_parser("harden", help="retrain against found failures (commercial)")
    h.add_argument("--robot", required=True)
    h.add_argument("--out", default="out")
    h.set_defaults(func=cmd_harden)

    a = sub.add_parser("activate", help="store a commercial licence key")
    a.add_argument("key")
    a.set_defaults(func=cmd_activate)

    sub.add_parser("license", help="show tier and what it unlocks").set_defaults(
        func=cmd_license)

    c = sub.add_parser("corpus", help="inspect or change corpus contribution")
    csub = c.add_subparsers(dest="corpus_cmd", required=True)
    csub.add_parser("status")
    cons = csub.add_parser("consent")
    cons.add_argument("answer", choices=("yes", "no"))
    csub.add_parser("preview", help="print exactly what a contribution contains")
    ex = csub.add_parser("export", help="stage a contribution from a failures.json")
    ex.add_argument("--failures", required=True)
    ex.add_argument("--robot-class", dest="robot_class", default="quadruped",
                    choices=corpus.ROBOT_CLASSES)
    ex.add_argument("--dof", type=int, default=12)
    ex.add_argument("--mass", type=float, default=12.7)
    ex.add_argument("--search-mode", dest="search_mode", default="unknown")
    ex.add_argument("--horizon", type=int, default=None)
    c.set_defaults(func=cmd_corpus)
    k = sub.add_parser("keys", help="manage ed25519 report-signing keys")
    ksub = k.add_subparsers(dest="keys_cmd", required=True)
    kn = ksub.add_parser("new", help="generate a keypair (refuses a path inside a git repo)")
    kn.add_argument("--out", required=True, help="directory outside any repository")
    kn.add_argument("--name", default="breakpoint",
                    help="file stem: <name>.key and <name>.pub (default breakpoint)")
    k.set_defaults(func=cmd_keys)

    sg = sub.add_parser("sign", help="add an ed25519 attestation block to a report")
    sg.add_argument("report", help="report JSON (breakpoint.report/v1)")
    sg.add_argument("--key", required=True, help="PEM private key from `keys new`")
    sg.add_argument("--signer", default="self", choices=("self", "inoki"),
                    help="who is signing: self (your own runs) or inoki (default self)")
    sg.add_argument("--out", default=None, help="write here instead of in place")
    sg.set_defaults(func=cmd_sign)

    vf = sub.add_parser("verify", help="check a signed report: PASS/FAIL, exit 1 on FAIL")
    vf.add_argument("report", help="signed report JSON")
    vf.add_argument("--pubkey", required=True, help="the signer's PEM public key")
    vf.add_argument("--policy", default=None, metavar="CHECKPOINT",
                    help="also check policy.sha256 against this local checkpoint file")
    vf.set_defaults(func=cmd_verify)

    from breakpoint_mjx.arm.cli import add_parser as add_arm_parser
    add_arm_parser(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except tier.LicenseRequired as e:
        print(str(e), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
