"""Train and save a policy under test (run once). Usage:
  python scripts/train.py [--target go1] --timesteps 100000000 --num-envs 2048 --out out
  python scripts/train.py --target go1 --robust --out out_robust   # robust policy (commercial edition)
"""
import argparse, os, sys
from breakpoint_mjx import policy
from breakpoint_mjx.targets import get_target

def main():
    ap = argparse.ArgumentParser(description="Train a policy under test (frozen or robust).")
    ap.add_argument("--target", default="go1", help="registered target name (default: go1)")
    ap.add_argument("--timesteps", type=int, default=100_000_000)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robust", action="store_true",
                    help="train FROM SCRATCH under BreakPoint's full-space DR with a staged "
                         "width CURRICULUM (narrow around nominal -> the target band) and "
                         "BreakPoint-in-the-loop adversarial concentration between stages. "
                         "Point --out at a DISTINCT dir to keep it beside the frozen baseline.")
    ap.add_argument("--schedule", choices=["adaptive", "staged", "loop"], default="adaptive",
                    help="robust: 'adaptive' (default) = per-dim ADR -- each dim widens as "
                         "fast as the policy can handle its boundary, with best-checkpoint "
                         "selection; 'staged' = fixed linear width schedule; 'loop' = "
                         "BreakPoint Loop -- full target-band width every round + probe for "
                         "best-checkpoint selection and unclipped failure concentration")
    ap.add_argument("--chunks", type=int, default=6,
                    help="robust adaptive/loop: training rounds between probes (default 6)")
    ap.add_argument("--stages", type=int, default=None,
                    help="robust staged: curriculum stages (default 3); 1 = no curriculum")
    ap.add_argument("--dr", choices=["plausible", "bounds"], default="plausible",
                    help="robust: target DR band -- 'plausible' (sim-to-real band; matches the "
                         "realistic-world metric; default) or 'bounds' (full theta bounds)")
    ap.add_argument("--start-frac", type=float, default=None,
                    help="robust: stage-1 band width as a fraction of the target (default 0.4)")
    ap.add_argument("--no-adversarial", dest="adversarial", action="store_false",
                    help="robust: disable the between-stage failure probe + concentration")
    ap.set_defaults(adversarial=True)
    ap.add_argument("--init-from", default=None,
                    help="robust: warm-start round 1 from an existing checkpoints dir "
                         "(e.g. the frozen baseline's <out>/<target>/checkpoints) so the "
                         "budget buys robustness instead of gait formation")
    ap.add_argument("--over-frac", type=float, default=1.0,
                    help="robust loop: train-band width as a multiple of the target band "
                         "(>1 over-randomizes, clipped to hard bounds; e.g. 1.15). The "
                         "eval/probe band is ALWAYS the unscaled target band.")
    ap.add_argument("--adv-strength", type=float, default=0.10,
                    help="robust: fraction of training mass concentrated on probed failures "
                         "(default 0.10; the first go1 run showed 0.25 collapses target-band "
                         "robustness the round it switches on)")
    ap.add_argument("--out", default=os.environ.get("BP_OUT", "out"),
                    help="parent/root output dir; policy + checkpoints are written under <out>/<target>/")
    a = ap.parse_args()
    target = get_target(a.target)
    # --out / BP_OUT is the parent root; the policy + checkpoints for this target
    # live under <root>/<target>/ so run_pipeline.py reads back the same dir and a
    # second target's training can't clobber the first's checkpoints.
    out_dir = os.path.join(a.out, target.name)
    os.makedirs(out_dir, exist_ok=True)
    if a.robust:
        try:
            from breakpoint_mjx.training import train_robust_policy
        except ImportError:
            sys.exit("error: --robust (robust retraining) is part of the BreakPoint commercial "
                     "edition and is not included in the open-source engine")
        train_robust_policy(out_dir, target=target, num_timesteps=a.timesteps,
                            num_envs=a.num_envs, seed=a.seed, stages=a.stages,
                            dr=a.dr, start_frac=a.start_frac,
                            adversarial=a.adversarial, schedule=a.schedule,
                            chunks=a.chunks, adv_strength=a.adv_strength,
                            over_frac=a.over_frac, init_from=a.init_from)
    else:
        policy.train_policy(out_dir, target=target, num_timesteps=a.timesteps,
                            num_envs=a.num_envs, seed=a.seed)
    print(f"saved {'ROBUST ' if a.robust else ''}policy under", out_dir)

if __name__ == "__main__":
    main()
