# Bring your own policy: the contract walkthrough

BreakPoint crash-tests a frozen policy. A policy trained somewhere else (Isaac Lab,
RSL-RL, legged_gym, unitree_rl_gym, your own PPO) was trained on a specific observation
vector and action convention. BreakPoint calls that the policy's **contract**:

* which observation terms it reads, and in what order,
* the per-term scale (legged_gym's `dof_vel: 0.05`, `ang_vel: 0.25`, ...),
* the joint order (Isaac Lab lists all hips, then thighs, then calves; MuJoCo Playground
  lists leg by leg),
* the default pose that joint positions are measured from,
* the action scale (`target = default + scale * action`).

Get any of these wrong and nothing crashes. The robot just falls, and a failure search on
it reports noise. This page shows the three ways to give BreakPoint the contract, from
most to least work on your side.

## 0. Check first

```bash
breakpoint check-policy --robot go1 --policy ./policy.onnx
```

`check-policy` runs 16 nominal rollouts and fails (exit 3) when the fall rate is above
0.25 or the policy does not track its command. If it passes, you are done: run the
search. If it fails, read on.

## 1. You have the training config: `breakpoint init`

```bash
# Isaac Lab / RSL-RL: point at the run directory (it holds params/env.yaml, agent.yaml)
breakpoint init --from rsl_rl ./logs/rsl_rl/unitree_go1_flat/2026-01-01_12-00-00 --robot go1

# a bare Isaac Lab env.yaml dump
breakpoint init --from isaaclab ./params/env.yaml --robot go1

# unitree_rl_gym deploy config
breakpoint init --from unitree_rl_gym ./deploy/deploy_mujoco/configs/go2.yaml --robot go1
```

`init` writes `contract.yaml` and prints every field BreakPoint cannot honor exactly,
for example:

```
fields BreakPoint cannot honor exactly:
  WARN   joint order          the config does not list resolved joint names, so the Isaac Lab
                              articulation order (joint-type-major ...) is ASSUMED
  INFO   actuators.stiffness  policy gains kp=25.0 kd=0.5 vs target kp=35 kd=0.5 (29% apart);
                              HONORED: the target's actuators are set to the policy's gains
  INFO   events.add_base_mass.mass_distribution_params  [-1, 3] -> operational.torso_mass_add_kg
  WARN   events.reset_base    event 'reset_root_state_uniform' has no ThetaSpace equivalent
  WARN   agent.empirical_normalization  the obs mean/std live in the checkpoint ...
```

Besides the obs/action map, `init` writes two blocks that describe how the policy was
trained (sections below): `actuation:` (its PD gains, applied to the target) and
`operational:` (its training domain randomization, as the operating domain for the risk
stage). An **INFO** row says what was carried over.

Nothing is dropped silently. A **WARN** row means the run is still meaningful with that
caveat. A **FATAL** row (a height scan, a gait phase clock, stacked history, a different
number of actuators) means the policy needs inputs the target cannot produce; `init`
refuses to write a contract and exits 2.

Then check through the contract:

```bash
breakpoint check-policy --robot go1 --policy ./policy.onnx --obs-spec contract.yaml
```

Only YAML is read. Python configs and pickles are never executed or unpickled.

## 2. You do not have the config (or it is wrong): `--infer-contract`

```bash
breakpoint check-policy --robot go1 --policy ./policy.onnx --infer-contract
```

When the nominal check fails, BreakPoint runs **Behavioral Contract Identification**
(`docs/contract.md`): it matches the checkpoint's own normalization statistics against
the simulator, races the plausible contracts in batched rollouts, and accepts one only
when a Clopper-Pearson test says it walks and beats the next-best different contract.
On success it prints the difference from what was declared and writes `contract.yaml`.
The numbers below are scramble 0 of the synthetic benchmark (`docs/contract_results.json`:
a scrambled version of the bundled Go1 policy, CPU), shown in the CLI's format:

```
infer-contract: ACCEPTED after 960 rollouts (90s)
  best:      0/32 falls (95% CP upper 0.089), 31/32 walk and track
  runner-up: 24/34 falls, 1/34 walk and track (a different term/joint structure)
  contract vs declared:
    obs term order: ['base_lin_vel', 'base_ang_vel', 'projected_gravity', ...] ->
                    ['base_ang_vel', 'commands', 'base_lin_vel', 'joint_vel', ...]
    scale[base_lin_vel]: 1.0 -> 1.72
    joint order: (target order) -> ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', ...]
    action scale: (target) -> 0.382
  wrote contract.yaml
```

The true contract of that scramble had `base_lin_vel` scale 2.0 and action scale 0.25:
the term order and joint order came back exactly, the scales came back close enough to
walk and track, not exactly. Treat inferred scales as calibrated, not as ground truth.

If the evidence cannot separate two contracts it says **AMBIGUOUS**, writes both
(`contract.candidate-1.yaml`, `contract.candidate-2.yaml`) and exits 3. It never picks
one for you. Combine both paths by passing your imported contract as the starting point:
`--obs-spec contract.yaml --infer-contract` races it alongside the inferred hypotheses.

## 3. Run the search with the contract

```bash
breakpoint run --robot go1 --policy ./policy.onnx --obs-spec contract.yaml --budget smoke
```

`--obs-spec` is exported to the pipeline as `BP_OBS_SPEC`, so
`BP_POLICY=./policy.onnx BP_OBS_SPEC=contract.yaml python -m breakpoint_mjx.pipeline`
works too. Policy specs: `onnx:<path>`, `params:<path>` (npz of weights),
`brax:<path>`, or a bare path ending in `.onnx`, `.npz` or `.pkl`.

## The contract file

```yaml
version: 1
robot: go1
joint_names: [FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
              FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
              FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint]  # POLICY order
default_joint_pos: [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
action: {scale: 0.25}          # rad per unit of policy output
obs:                           # POLICY order
- {term: base_lin_vel, scale: 2.0}
- {term: base_ang_vel, scale: 0.25}
- {term: projected_gravity}
- {term: commands, scale: [2.0, 2.0, 0.25]}
- {term: joint_pos_rel}
- {term: joint_vel, scale: 0.05}
- {term: last_action}
```

Term names accept the common spellings (`dof_pos`, `joint_pos`, `qj`, `actions`,
`velocity_commands`, `gravity`, ...). A scale may be a list (a negative entry flips a
sign). `last_action` is fed back as the policy's own raw output. A term BreakPoint does
not know can read raw channels with `{term: name, start: i, dim: n}`.

## Limits

* Registered declarative interface: **Go1** (`TargetInterface` in
  `breakpoint_mjx/adapters/obs_spec.py`). Other robots use the raw form of the contract
  (`raw: {keys, indices, scale, offset}`) until their interface is registered.
* Joint axis sign conventions are not modelled: if your URDF's hip axis points the other
  way, the default pose and the joint signs differ (Isaac Lab's Go1 has left hips at
  +0.1; Playground's has right hips at +0.1). Put the sign in the per-term scale list.
* PD gains ARE reproduced on Go1 (`actuation:` with `honor: true`). Control period,
  actuator torque/velocity limits, learned actuator networks, obs noise and obs/action
  clipping are reported, not reproduced.
* Stacked observation history is refused (see below).
* The `kd_scale` search dim is a no-op on the Playground Go1 whether or not gains are
  honored: it multiplies `actuator_biasprm[:, 2]`, which is 0 on that model (Kd lives in
  `dof_damping`). `kp_scale` works and scales the honored gains.
* MLP policies only (the adapters' scope): recurrent and vision policies are out.

## The policy's own PD gains: `actuation:`

```yaml
actuation: {kp: 25.0, kd: 0.5, action_scale: 0.25, source: actuators, honor: true}
```

A position-controlled policy is tuned to the PD loop it trained on. The Playground Go1
uses Kp 35 / Kd 0.5; an Isaac Lab or legged_gym Go1/Go2 policy trained at 20 to 25. Run
unchanged, it walks on a stiffer robot than it learned on, and a failure found there can
be an artifact of the gain change. With `honor: true`, `targets/go1.load_env` sets the
actuators to the contract's gains on both the MuJoCo model and the MJX model
(`breakpoint_mjx/adapters/actuation.py`): `gainprm[:,0] = kp`, `biasprm[:,1] = -kp`, and
`kd` where the model keeps its derivative term (`dof_damping` on the Playground Go1,
`biasprm[:,2]` on models that use it). It prints the change:

```
actuation: honoring the policy's PD gains (actuators): kp 35 -> 25, kd 0.5 -> 0.5 (kd written
to dof_damping); ThetaSpace kp_scale/kd_scale now scale these
```

`kp` and `kd` may be per-joint lists in the policy's joint order. `init` fills the block
from the training config (Isaac Lab `actuators.*.stiffness/damping`, unitree_rl_gym
`kps`/`kds`, or legged_gym `control.stiffness`). `BP_HONOR_ACTUATION=0` switches honoring
off everywhere. `honor: false` with gains more than 20% from the target's is **FATAL** in
`init` (written only with `BP_HONOR_GAINS=0`) and in `check-policy`; override with
`--allow-gain-mismatch` or `BP_ALLOW_GAIN_MISMATCH=1`.

### Sim-to-sim gap check

```bash
breakpoint check-policy --robot go1 --policy <policy> --obs-spec contract.yaml --gap-check
```

Runs the same 32 nominal seeds (`--gap-n`) twice, at the contract's gains and at the
target's native gains, and prints survival and tracking error for both plus a verdict:
`no nominal shift`, `NOMINAL SHIFT` (both walk, but survival differs by 0.10 or more or
tracking error by 1.25x or more), `WALKS AT ONE GAIN SET ONLY` (refused, exit 3: a
search would measure the gain choice, not the policy; `--allow-gain-shift` overrides) or
`WALKS AT NEITHER` (exit 3).

## Training domain randomization as the operating domain: `operational:`

```yaml
operational:
  friction_floor: {range: [0.5, 1.2]}
  torso_mass_add_kg: {range: [-1.0, 3.0]}
  push_mag: {range: [0.0, 1.0]}
```

The ranges a policy was randomized over in training are the domain it can be expected to
handle. `init` reads them (legged_gym / unitree_rl_gym `domain_rand`: `friction_range`,
`added_mass_range` with `randomize_base_mass`, `push_robots` + `max_push_vel_xy`; Isaac
Lab `events`: `physics_material`, `add_base_mass`, `base_com`, `push_robot`) and maps
them onto the robot's ThetaSpace dims. The block is exactly the
`riskcert.OperationalDist.from_space` schema, so the contract file itself is the risk
distribution:

```bash
breakpoint run --robot go1 --policy <policy> --obs-spec contract.yaml \
    --risk 8192 --risk-dist contract.yaml
```

Every key that does not map (reset randomization, external forces, motor strength) is a
WARN row, and so is a range the trainer held fixed (a point is not a range; the dim keeps
its plausible band). unitree_rl_gym deploy YAMLs carry no DR, so `init --from
unitree_rl_gym` reads it from, in order: a `domain_rand:` block pasted into the YAML, the
checkout the YAML lives in (`legged_gym/envs/base/legged_robot_config.py` plus
`legged_gym/envs/<robot>/<robot>_config.py`), or `BP_TRAIN_CFG`. The python files are
parsed with `ast` and only literal assignments are read; nothing is imported or run.

## Stacked observation history: refused

`history: {length: H, ...}` in a contract, or Isaac Lab's `history_length`, is refused.
The shared rollout calls a stateless `policy_fn(obs, key)` inside the evaluator's vmap,
and its scan carry is `(env state, action latency buffer, key)`, so there is no place for
a per-env observation ring buffer without changing `rollout.py` / `evaluator.py`. That
change was deliberately not made in this build (`tests/test_history_obs.py` pins the
reason and fails if the carry ever grows a policy state).

## Measurements (CPU, 2026-09-27)

All numbers below are CPU runs of `python -m breakpoint_mjx.check_policy` from the repo
root (JAX CPU, Playground `Go1JoystickFlatTerrain`, horizon 150 steps, seed 0). They are
nominal-behaviour checks, not failure searches.

**No regression from honoring gains.** The bundled policy, 32 nominal rollouts:

| path | command | falls | tracking error |
|---|---|---|---|
| unmodified (brax pickle, no contract) | `--robot go1 --policy breakpoint_mjx/assets/demo/go1_policy.pkl --n 32` | 0/32 | 0.275 |
| contract with `actuation: {kp: 35, kd: 0.5, honor: true}` | same `--obs-spec docs/byo/bundled_go1_kp35.yaml` | 0/32 | 0.275 |

**Gap check, bundled policy at Kp 25 vs the native Kp 35** (`--obs-spec
docs/byo/bundled_go1_kp25.yaml --gap-check --gap-n 32`):

```
sim-to-sim gap check (same policy, same seeds, only the PD gains differ):
  contract gains (kp 25 kd 0.5): survival 1.000 (32/32)  tracking error 0.320  walks
  native gains   (kp 35 kd 0.5): survival 1.000 (32/32)  tracking error 0.275  walks
  survival delta +0.000, tracking ratio 1.16x -> verdict: no nominal shift
```

The bundled policy stays up at the softer gains and tracks 16% worse; under the 1.25x
threshold, so no shift is declared.

## Worked third-party checkpoint (Go2 policy on the Go1 model; embodiment mismatch stated)

**What was asked for and what exists.** The plan named unitree_rl_gym's
`deploy/pre_train/go2/motion.pt` with its go2 deploy YAML. Neither exists: on 2026-09-27
every commit on every branch of github.com/unitreerobotics/unitree_rl_gym (59 commits)
was listed, and the shipped pre-trained policies are `g1`, `h1` and `h1_2` (humanoids).
There is no Go2 deploy config either. The nearest real, public quadruped checkpoint with
a readable training config is Isaac Lab's pretrained RSL-RL policy for
`Isaac-Velocity-Flat-Unitree-Go2-v0`, which is what was run.

* Source: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt`
  (sha256 `47ccb2046ec3064257997f879c62d64217b937713c30d827883b2c2f40c50439`,
  iteration 299, actor 48 -> 128 -> 128 -> 128 -> 12, ELU, no obs normalizer).
* Licence: Isaac Lab's source is BSD-3-Clause. No licence is published next to the
  checkpoint on NVIDIA's asset bucket, so its terms are unverified and it is **not
  vendored**: `scripts/fetch_thirdparty.py` downloads it, checks the hash and exports the
  actor to `thirdparty/isaaclab_go2_flat/actor.npz` (git-ignored).
* Training config: Isaac Lab publishes only `checkpoint.pt` there, no `params/env.yaml`.
  The fetch script writes one TRANSCRIBED by hand from IsaacLab@b0542fe (unitree.py
  `UNITREE_GO2_CFG`: DCMotor stiffness 25, damping 0.5; velocity_env_cfg.py obs, actions
  and events with the go2 rough/flat overrides: action scale 0.25, friction fixed at 0.8,
  base mass +[-1, 3] kg, no pushes). If the config changed between the version that
  trained the checkpoint and b0542fe, the transcription differs from what the policy saw.
* Embodiment mismatch: this is a **Go2** policy run on the **Go1** model (different mass,
  link lengths and motor limits). The Isaac Lab and legged_gym Go2 configs put the LEFT
  hips at +0.1 rad; the Playground Go1 puts the RIGHT hips at +0.1. No robot-specific
  tuning of the target was done.

Commands (repo root, `.venv-cpu`, CPU):

```bash
python scripts/fetch_thirdparty.py
breakpoint init --from rsl_rl thirdparty/isaaclab_go2_flat --robot go1 \
    --out thirdparty/isaaclab_go2_flat/contract.yaml
export BP_ACTIVATION=elu
breakpoint check-policy --robot go1 --policy params:thirdparty/isaaclab_go2_flat/actor.npz \
    --obs-spec thirdparty/isaaclab_go2_flat/contract.yaml --gap-check --gap-n 32
breakpoint check-policy --robot go1 --policy params:thirdparty/isaaclab_go2_flat/actor.npz \
    --obs-spec thirdparty/isaaclab_go2_flat/contract.yaml --infer-contract \
    --contract-out thirdparty/isaaclab_go2_flat/contract.inferred.yaml
breakpoint run --robot go1 --policy params:thirdparty/isaaclab_go2_flat/actor.npz \
    --obs-spec thirdparty/isaaclab_go2_flat/contract.yaml --budget smoke \
    --out thirdparty/isaaclab_go2_flat/run
```

The contract `init` wrote is committed as `docs/byo/isaaclab_go2_flat_contract.yaml`
(`actuation: {kp: 25, kd: 0.5, action_scale: 0.25, source: actuators, honor: true}`,
`operational: {torso_mass_add_kg: {range: [-1, 3]}}`; friction was a point, so it is a
WARN row, not a range).

**Result: a measured refusal (CPU smoke).** The policy does not walk on the Go1 model at
nominal, at either gain set, so no search ran and no report was written.

```
sim-to-sim gap check (same policy, same seeds, only the PD gains differ):
  contract gains (kp 25 kd 0.5): survival 0.250 (8/32)  tracking error 1.074  DOES NOT WALK
  native gains   (kp 35 kd 0.5): survival 0.688 (22/32)  tracking error 0.659  DOES NOT WALK
  survival delta -0.438, tracking ratio 1.63x -> verdict: WALKS AT NEITHER
```

Honoring the training gains made it worse here, not better: the Go2 policy on the Go1 body
stays up in 8/32 rollouts at Kp 25 and 22/32 at Kp 35.
That is the gain sensitivity the gap check exists to expose, on a policy that is also
off-embodiment. `--infer-contract` (957 rollouts, 192 s) returned **AMBIGUOUS**: its best
candidate falls 2/32 but walks and tracks 1/32 (it barely acts, action scale 0.05), the
runner-up 5/36 and 1/36; both candidates are written and none is chosen.
`breakpoint run --budget smoke` then stopped at its preflight (16 rollouts: 11/16 falls,
tracking error 1.015) with exit 3, so `thirdparty/isaaclab_go2_flat/run/` holds no
report. The target was not tuned to force a pass.

What this does and does not show: the full BYO path (fetch, `init` with gains and ODD,
honored gains on both models, gap check, BCI, `run` preflight) ran end to end on a real
third-party checkpoint. It does not show BreakPoint finding failures in a third-party
policy: that needs a policy trained on the same embodiment as a registered target.
