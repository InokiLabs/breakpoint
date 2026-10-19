# BreakPoint and the alternatives

An honest comparison as of October 2026: what already exists, what BreakPoint adds, where
it is behind, and what has been measured head to head. Sources were checked on
2026-10-16; anything not confirmed from a primary page is marked.

## The short version

Searching an environment's parameters for the conditions where a fixed learned policy
fails is **not a new idea**. ETH's adversarial attacks on quadruped controllers (RSS 2024),
RoboGate's boundary-focused sampler (2026), coverage-aware failure discovery (2026) and
NVIDIA Isaac Lab-Arena's sensitivity analysis all cover parts of it. Nobody should pitch
the search itself as the novelty.

What BreakPoint adds is the **engineering around the search**, aimed at teams that ship
legged-robot RL policies and need a regression check they can trust in CI:

1. **A statistically sound baseline-vs-candidate CI gate.** Exact paired sign tests over
   seed-paired rollouts, pooled plus Holm per cell, with escalation of suspect cells to
   fresh seeds. False alarms are held at or below 5% while single-cell regressions are
   caught 91-99% of the time (measured: `docs/ci-gate.md`). None of the tools below
   compares two checkpoints statistically; RoboGate has an absolute pass/fail gate, Arena
   has none.
2. **GPU-parallel adversarial search in MJX**, thousands of worlds per batch, against
   physics and sensing parameters (friction, mass, latency, pushes, noise, dropout,
   bias, slope, actuator strength, payload).
3. **Evidence on every failure**: bit-exact replay, a physical-validity screen that
   keeps simulator artifacts out of the counts, and a cross-check in a second engine
   (C MuJoCo, different contact model) with an exact group-sequential decision rule.
4. **Verifiable reports**: fixed, hashed rating batteries with exact intervals, and
   ed25519-signed reports.
5. **Turnkey import of the policies teams actually train**: RSL-RL / Isaac Lab /
   legged_gym / unitree_rl_gym configs to a contract, including observation history and
   LSTM/GRU recurrent actors.

## Who else is in the space

| | what it is | overlap | what it has that BreakPoint lacks |
|---|---|---|---|
| **Shi et al., "Rethinking Robustness Assessment" (ETH, RSS 2024)** [arXiv 2405.12424](https://arxiv.org/abs/2405.12424) | Sequential low-magnitude adversarial attacks on learned quadruped controllers; validated on a real robot; failures used to harden the policy | High: the same thesis | Time-varying perturbation sequences; real-robot validation |
| **RoboGate (2026)** [arXiv 2603.22126](https://arxiv.org/abs/2603.22126) | Two-stage sampling (Latin hypercube, then the 30-70% success transition zone) over 8 parameters in Isaac Sim; Franka and UR arms; 50,000+ experiments; logistic failure-boundary model | High on manipulation | A fitted failure-boundary model; VLA evaluation. No baseline-vs-candidate gate, no confidence intervals stated on the abstract page |
| **Coverage-aware active evaluation with paired systems (Parashar, Pavone, Fan et al., Aug 2026)** [arXiv 2608.13719](https://arxiv.org/abs/2608.13719) (from the market scan; not re-checked) | Cheap proxy evaluations corrected toward scarce target ones; diversity-aware coverage; tested on quadruped velocity tracking among others | High | Proxy-to-target calibration; diversity-driven coverage |
| **NVIDIA Isaac Lab-Arena** [docs](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/index.html), [blog](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena) | Policy-agnostic parallel evaluation on Isaac; development-version docs describe a sensitivity analysis ("perturb environment factors to reveal policy robustness", "failure cliffs"); the blog calls the release "pre-alpha" | Medium-high | NVIDIA distribution; vision and VLA policies (GR00T, pi0); appearance variations. No directed search, intervals or checkpoint gating described |
| **Runtime.si (YC Fall 2026)** [YC page](https://www.ycombinator.com/companies/runtime-si) | "Building the evaluation layer for physical intelligence"; tests robot control software across many simulated scenarios before deployment; 5 people | Likely high (direct startup competitor); product details not public | Unknown |
| **SimplerEnv / SureSim** (from the market scan) | Sim-to-real correlation for manipulation evaluation; prediction-powered real-world intervals | Medium | Real-hardware correlation, which BreakPoint does not attempt |
| **S-TaLiRo / PSY-TaLiRo, Breach, VerifAI + Scenic, Stanford AST Toolbox** | Falsification of cyber-physical systems against temporal-logic specs; scenario languages; adaptive stress testing | Medium (method), low (domain) | Temporal-logic specs, scenario languages, rare-event estimation. Mostly CPU-bound, Simulink/CARLA-oriented, not applied to learned legged policies |
| **Foretellix, Applied Intuition** | Coverage-driven scenario verification for autonomous vehicles | Low today | Mature verification workflows and sales channels; Applied Intuition is hiring for robot-RL simulation (from the market scan), a likely future entrant |
| **Standards**: ISO 25785-1 (dynamically stable industrial mobile robots, committee draft), ANSI/A3 R15.108 (bipedal/quadrupedal technical report) | Stability and safety requirements for legged robots | Opportunity, not competition | None yet prescribe simulation falsification evidence; signed, replayable reports are a head start if they come to |

## Measured head to head

Equal-budget CPU smoke on the bundled Go1 policy: 384 rollouts per arm per seed, two
seeds, horizon 150 (`docs/equal_budget.md`). `lhs_boundary` is our implementation of
RoboGate's published sampler, not their code.

| arm | closest failure to nominal (L2, lower = more deployable; brackets = range over the 2 seeds) | in-band failures (mean per seed, of 384) |
|---|---|---|
| uniform random (what "evaluate on N random DR configs" means) | 1.070 [1.033, 1.106] | 155.5 |
| RoboGate-style LHS + boundary | 1.121 [1.061, 1.180] | 153.0 |
| **BreakPoint directed (multi-restart CMA-ES)** | **0.821 [0.785, 0.857]** | **304.0** |

Directed search finds failures about a quarter closer to the nominal operating point and
about twice as many inside the plausible deployment bands, on both seeds. Raw failure
counts and failure-mode counts are saturated for all three on this fragile demo policy,
and the second-engine replay rates (2/8, 4/8, 2/8) are too small a sample to separate the
arms; an earlier run's replay advantage for directed search did not reproduce. A
standard-budget GPU run is the measurement that would settle both.

CI gate, simulated with the shipped battery's shape (8 seeds, 15 cells, `docs/ci-gate.md`):

| | false alarms when nothing changed | catches a one-cell 2% to 50% regression |
|---|---|---|
| the original pass-rule gate | 50% of builds | 98% |
| BreakPoint statistical gate + escalation | **1%** | **99%** |

On the real simulator the old rule failed a build on one fall in four seeds that 16 fresh
seeds showed was noise; the statistical gate cleared it, and both failed genuinely
degraded checkpoints.

## Where BreakPoint is behind (and what would close it)

Ranked by how often it would decide an evaluation against BreakPoint:

1. **No real-hardware correlation.** Every number is simulation. A calibration layer
   (a few paired real rollouts turned into real-world intervals, as SureSim and the
   Parashar work do) would replace the hand-set plausibility heuristic, the weakest claim
   in the product.
2. **Static perturbations only.** Shi et al. break controllers that survive static
   randomization with low-magnitude *sequences*. The rollout already carries per-step
   state, so push trains and latency or dropout bursts are a natural next search axis.
3. **No terrain.** Heightfield and stairs are the dominant legged failure axis; MJX's
   heightfield collision support is partial, so this is real work.
4. **Locomotion and one pick task only; no vision or VLA policies.** Arena and RoboGate
   own manipulation and VLAs. Competing there head-on is not the best use of effort;
   interoperating with Isaac Lab tasks is.
5. **"Fell" is the only spec.** Buyers who know falsification tools expect temporal
   properties (foot slip, torque limits, safe-stop time).

Closed in this round: observation-history and LSTM/GRU policies (most deployed RSL-RL and
Isaac Lab locomotion policies use one or the other), the gate's false-alarm problem, the
gate crashing into a "regression" verdict, the physical-validity screen not running in
reports, the cross-engine replay over-spending its error rate on repeated looks, and a
broken `pip install -e ".[sim]"`.

## Positioning that follows from this

Be the most trustworthy, CI-native regression gate for teams shipping RSL-RL / Isaac Lab
legged policies, with signed, replayable evidence, and build on top of the big
evaluation frameworks rather than against them. Do not claim the search idea as new. Move
next on time-varying perturbations and sim-to-real calibration, where published work
exists and no product does.
