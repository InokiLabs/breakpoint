# Behavioral Contract Identification (BCI)

**Problem.** A policy trained in Isaac Lab, legged_gym, RSL-RL or unitree_rl_gym consumes
a specific observation vector (term order, per-term scales, joint order, default pose)
and emits actions in a specific convention (joint order, action scale). BreakPoint's
target env produces its own layout. If the two disagree the policy still runs; it falls,
and a failure search then "finds" noise. `check-policy` detected this but repaired
nothing: an outside team had to hand-build a Python `ObsSpec`.

**What ships.**

| piece | where |
|---|---|
| declarative contract YAML (terms in policy order, scales, joint names, default pose, action scale) resolved to index arrays | `breakpoint_mjx/adapters/obs_spec.py` (`PolicyContract`, `build_contract`, `ObsSpec.from_yaml/to_yaml`) |
| action joint permutation + per-joint scale/offset | `adapters/mlp.py` (`OutputTransform.perm`) |
| `BP_OBS_SPEC` in the pipeline loader; brax `.pkl` weight import; `C:\...` paths no longer parsed as a policy kind | `adapters/importers.py` (`build_policy`, `parse_policy_spec`, `from_brax_pkl`) |
| config importers with a table of every field that cannot be honored | `adapters/importers.py` (`from_isaaclab_cfg`, `from_rsl_rl_run`, `from_unitree_rl_gym`) |
| the identification method | `adapters/contract_id.py` (`identify_contract`) |
| CLI | `breakpoint init --from ...`, `check-policy --obs-spec / --infer-contract`, `run --obs-spec` |
| benchmark | `scripts/bench_contract.py` -> `docs/contract_results.json` |

Walkthrough for users: [`docs/byo-policy.md`](byo-policy.md).

## Method

A contract is data: `policy_input = (src[obs_idx] - obs_offset) * obs_scale` and
`env_action[j] = y[act_idx[j]] * act_scale[j] + act_offset[j]`. So candidate contracts
`vmap` together, and one jitted rollout call evaluates many of them at once.

**Hypothesis space.** Term order (for Go1's 48-dim input: the 35 ways to interleave four
3-dim and three 12-dim terms, times assignments of terms to slots), a joint order from
the robot's **leg-symmetry group** (leg-major or joint-type-major layout x every leg
order = 48 orders for a quadruped, instead of 12! = 479,001,600), continuous per-term
scales with per-channel sign flips on 3-vectors, and a continuous action scale.

**Stage A, moment matching (zero policy rollouts).** The reference is the obs statistics
of 256 env resets. The checkpoint's own running mean/std (brax, RSL-RL and Isaac Lab all
ship one; without it the first-layer weight-column norms stand in as a weak prior) is
matched against it:

* *Exogenous* terms (gravity, the sampled velocity command) have an exact reference, so
  their cost is weighted 10x and their scales are closed-form (gravity from the mean,
  which pins scale and sign exactly; the command from the std ratio).
* Joint *layout* comes from a leg-order-free statistic: channels of the same joint type
  (all hips, all calves) should look alike in the policy's own statistics.
* Term-to-slot assignment within each dimension group is solved by exhaustive ranking
  (the groups are 4! and 3!), producing a ranked list, not one answer.
* A standing robot's dynamic channels are sensor noise, so their scales start at the
  unit prior and the action scale at a coarse log2 grid (x0.5, x1, x2 of the target's).

**Closed-loop moment re-estimation (the EM step).** Once any hypothesis walks, its
pre-fall closed-loop statistics are a valid reference for every hypothesis (the env
channels are physical), so Stage A re-ranks against them; each leading hypothesis also
refits its own dynamic-term scales and its action scale to its own rollouts. The action
scale is identifiable here because the policy sees its own raw output in its
`last_action` slot while the env reports the applied, scaled action.

**Stage B, racing.** All Stage A hypotheses run once, then successive halving with more
rollouts per survivor, on common reset keys (paired comparisons). Ranking uses a
continuous loss: command-tracking error over the steps **before** any fall plus 0.5 per
fall. Charging post-fall steps ranks a contract that makes the robot stand still (a tiny
action scale) above a correct-structure contract that walks and occasionally falls; that
attractor was the main failure mode during development. A local log-scale refinement
(one-coordinate moves of x0.5/x2, then x0.71/x1.41) finishes the scales.

**Acceptance (Clopper-Pearson).** The winner and the best hypothesis with a *different*
discrete structure (term order or joint order) are each topped up to at least 32 nominal
rollouts. Accept only if the one-sided 95% CP upper bound on the winner's fall rate is
<= 0.2 **and** the CP lower bound on its success rate (no fall, tracking error within the
target's threshold) exceeds the CP upper bound on the runner-up's. If the fall bound
holds but the separation does not, the result is **ambiguous**: both contracts are
written and none is chosen. Otherwise **failed**.

## Prior art

* **Isaac Lab sim-to-sim transfer** remaps observations and actions between physics
  backends with a hand-written YAML of joint names in each backend's order. BCI produces
  that mapping without a human, and checks it.
* **unitree_rl_gym** deploy configs hand-list default angles, obs scales, action scale
  and PD gains. `breakpoint init --from unitree_rl_gym` reads those files; BCI covers the
  case where they are missing or wrong.
* **Zhang et al. 2021, "Learning Cross-Domain Correspondence for Control with Dynamics
  Cycle-Consistency"** learns state/action correspondences between domains by training
  mapping networks. BCI is training-free: it identifies a discrete-plus-scale interface
  of a frozen checkpoint from its normalization statistics and closed-loop behaviour,
  with a statistical accept/refuse rule.

I did not find a published tool that infers the obs/action contract of a third-party
locomotion checkpoint automatically with a statistical acceptance rule. That is a search
result, not a proof of absence.

## Benchmark (SYNTHETIC scrambles of a real trained policy, CPU)

The bundled Go1 policy (`media/dl/frozen_go1_policy.pkl`, a brax PPO checkpoint trained
in MuJoCo Playground) is re-expressed behind 40 seeded scrambled contracts. Each scramble
reorders the 7 obs terms, picks a joint order from the symmetry group (every 4th uses
Isaac Lab's exact joint-type-major order), rescales 1 to 3 obs terms by x0.5 or x2 and
sets the action scale to x0.5, x1 or x2. `fold_contract` folds each scramble exactly into
the weights (first-layer rows, normalizer statistics, output-head columns), so each
scrambled policy is a real trained network that walks only through its true contract.
All four methods get the same rollout budget (1024 nominal rollouts, horizon 150);
evaluation is a separate 16-rollout nominal run on a fixed seed.

```bash
PYTHONPATH=. .venv-cpu/Scripts/python.exe scripts/bench_contract.py --n 40 --budget 1024 --seed-base 5000
```

**Development hygiene.** BCI was developed and tuned on scrambles with seed base 1000,
i = 0..7. The reported run uses seed base 5000, which was never run before this
benchmark. The bars below were committed in `scripts/bench_contract.py` (commit
`35a0d27`) before the reported run.

### Pre-registered bars and measured results

| bar | measured | pass |
|---|---|---|
| BCI recovers an ACCEPTED contract with fall rate <= true + 1/16 in >= 80% | **39/40 = 97.5%** | yes |
| no accepted contract with fall rate >= 0.5 (false accept) | **0/40** | yes |
| BCI beats uniform random search at equal budget by >= 20 pp | **97.5% vs 0.0% (+97.5 pp)** | yes |
| identity (the target's native layout) recovers <= 10% | **0/40** | yes |
| median identification wall-clock <= 10 min on this CPU | **64 s** (range 49 to 90 s), plus a one-time 33 s racer compile | yes |

| method | accepted | ambiguous | failed | recovered | exact term + joint structure | median rollouts |
|---|---|---|---|---|---|---|
| identity | n/a | n/a | n/a | 0/40 | n/a | 0 |
| uniform random + same racing/acceptance | 0 | 23 | 17 | 0/40 | 0/40 | 883 |
| Stage A top-1 only | n/a | n/a | n/a | 0/40 | 0/40 | 0 |
| **BCI** | **40** | 0 | 0 | **39/40** | **40/40** | 930 |

Machine: Windows 11, CPU-only JAX (MJX on CPU), `batch=32` rollouts per jitted call.
Per-scramble logs, chosen contracts, evidence and timings are in
`docs/contract_results.json`.

### What the numbers do and do not say

* **The one miss** (scramble 23) was accepted with the exact term and joint structure,
  but its evaluation run fell 2/16 against a bar of 1/16. It is counted as not recovered.
  It is not a false accept (fall rate 0.125, far below 0.5).
* **Scales are approximate.** Structure came back exactly in 40/40, but the scales are
  "good enough to walk and track", not the truth: median |log2 error| over non-action
  terms 0.18 (a factor of 1.13), 90th percentile 0.52, maximum 2.0 (a factor of 4). The
  large errors are all `joint_vel`, which this policy is nearly insensitive to at
  nominal. The recovered action scale is within x0.91 to x1.53 of the truth (median
  x1.13). A contract that is behaviourally adequate at nominal may still differ from the
  truth under perturbation, which is what the failure search then probes.
* **Fall rate alone is a lenient bar.** Ignoring its acceptance rule, random search
  returned a contract meeting the fall-rate bar in 24/40 scrambles, with the wrong
  structure every time: wrong contracts can stand without falling. That is why the
  acceptance rule separates on success (walking AND tracking) against a structurally
  different runner-up, and why random search never got an accept. On the lenient
  metric the gap is 39/40 vs 24/40 (+37.5 pp), still above the 20 pp bar.
* **Stage A alone never gets the structure right at top-1**: statistics cannot tell the
  24 leg orders apart, nor linear from angular velocity. It is a pruning step; the true
  structure has to survive into the racing set, which it did in every run.
* **Synthetic.** Every scramble is a permutation/rescaling of ONE real policy on ONE
  robot. It does not cover policies trained elsewhere (different gains, control rate,
  noise model, URDF axis signs), which is the real-world case.

## Third-party configs (stretch, reported as run)

I fetched two real unitree_rl_gym deploy configs
(`deploy/deploy_mujoco/configs/g1.yaml` and `h1.yaml` from
github.com/unitreerobotics/unitree_rl_gym, main branch, 2026-09-27) and ran
`breakpoint init --from unitree_rl_gym` on them:

* Both parsed: kps/kds, default angles, obs scales, action scale, command scale and
  control period were all read.
* Both were refused (exit 2), correctly: `num_obs` exceeds the standard layout by 2
  (a gait phase clock BreakPoint cannot produce); h1 also has 10 actuators vs Go1's 12;
  with `--robot g1` there is no registered declarative interface for G1 yet.
* The quadruped deploy config (`go2.yaml`) does not exist at that path upstream, and no
  public Isaac Lab `params/env.yaml` dump was located. So **there is no worked
  third-party example on Go1**: the RSL-RL and Isaac Lab importers are tested only on
  hand-written fixtures that mirror the real layouts (`tests/fixtures/contract/`).

The end-to-end declarative path is verified on the bundled policy:
`breakpoint init --from rsl_rl tests/fixtures/contract/bundled_go1_rsl_rl` followed by
`breakpoint check-policy --robot go1 --policy <bundled go1_policy.pkl> --obs-spec
contract.yaml` passes (0.00 fall rate, tracking error 0.281, 16 rollouts, CPU).

## Limitations

* One registered declarative interface (Go1). Other targets need a `TargetInterface`
  entry (layout, joint names, default pose, action scale, gains), or the raw contract.
* Joint axis sign conventions are not searched (only 3-vector sign flips from decisive
  means). A URDF whose hip axis is flipped needs the sign in the contract by hand.
* PD gains, control period, obs noise and clipping, and history stacking are reported
  by the importers, never reproduced.
* The acceptance rule is on nominal behaviour. It certifies that the contract walks
  and tracks at nominal, not that it equals the training contract.
* CPU cost: about 900 to 980 nominal rollouts per identification (about 1 minute on this
  laptop after a 33 s compile). A GPU would batch the racing rounds wider; that is not
  measured here.
