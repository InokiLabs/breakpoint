# Perturbations and ranges (Go1)

Source of every number: `breakpoint_mjx/configs/go1/theta_space.yaml` (sha256 recorded in
each run's `provenance.json`). The benchmark does not change any range. The rationale
column explains why each range is physically reasonable; the `plausible` bands are the
repository's own plausibility model, set by hand from typical hardware and deployment
figures, **not measured on a physical Go1**.

Control step = 0.02 s (50 Hz). Nominal trunk height about 0.28 m (search.yaml).

| perturbation | dim | near-nominal | conservative (plausible) | broad (stress test) | rationale |
|---|---|---|---|---|---|
| Control latency | `latency_steps` (rounded to whole steps) | 0 to 0.5 (rounds to **0**: effectively off) | 0 to 2 steps (0 to 40 ms) | 0 to 4 steps (0 to 80 ms) | On-board estimation plus actuation pipelines commonly add 10 to 40 ms; 80 ms is a degraded or networked stack |
| Ground friction | `friction_floor` | 0.70 to 0.85 | 0.40 to 1.00 | 0.15 to 1.20 | Rubber feet on dry concrete ~0.8 to 1.0; smooth tile or wet floors ~0.4; 0.15 is ice-like |
| Trunk mass change | `torso_mass_add_kg` | -0.25 to +0.5 kg | -1 to +2 kg | -1 to +6 kg | +2 kg is a sensor or compute payload; +6 kg is a heavy payload for a robot of roughly 12 kg |
| Centre-of-mass offset | `com_offset_{x,y,z}_m` | +-1.25 cm (z +-0.75 cm) | +-5 cm (z +-3 cm) | +-8 cm (z +-5 cm) | Off-centre payload mounting or battery placement |
| Motor stiffness (strength proxy) | `kp_scale`, `kd_scale` | 0.96 to 1.04 | 0.85 to 1.15 | 0.5 to 1.5 | Gain mismatch between training and deployment, motor heating and wear. **Torque limits are not perturbed** (not implemented in the engine) |
| Joint friction / armature | `joint_frictionloss_scale`, `joint_armature_scale` | 0.98 to 1.08 / 1.0 to 1.03 | 0.9 to 1.3 / 1.0 to 1.1 | 0.5 to 3.0 / 0.8 to 2.0 | Gearbox wear, temperature, lubrication; model error in reflected inertia |
| Sensor noise | `obs_noise_scale` (std, observation units) | 0 to 0.01 | 0 to 0.04 | 0 to 0.10 | IMU and encoder noise beyond what the policy saw in training |
| Sensor dropout | `obs_dropout_prob` (per channel per step) | 0 to 5% | 0 to 20% | 0 to 50% | Dropped or stale packets on the sensor bus |
| Sensor bias | `obs_bias_scale` (constant per episode) | 0 to 0.01 | 0 to 0.04 | 0 to 0.10 | Calibration drift (IMU bias, encoder offset) |
| External push | `push_mag` (horizontal velocity impulse, m/s) at `push_step_frac` of the episode | 0 to 0.25 m/s, first 25% | 0 to 1.0 m/s, any time | 0 to 2.0 m/s, any time | 1 m/s added to the trunk is a firm shove or collision; 2 m/s is a severe impact. Applied as a velocity change, so no unrealistic force spike enters the solver beyond that impulse |
| Floor slope | `floor_slope_rad` | +-0.025 rad (+-1.4 deg) | +-0.10 rad (+-5.7 deg) | +-0.20 rad (+-11.5 deg) | Ramps and uneven floors; a 1:12 accessibility ramp is 4.8 deg |

Not supported (not faked): torque / current limits, terrain roughness or stairs,
time-varying perturbation sequences, vision inputs.

## Regimes

* **near_nominal**: each plausible band shrunk toward nominal to 25% of its distance on
  each side. A well-controlled deployment; failures should be rare.
* **conservative**: the plausible bands. Realistic deployment conditions.
* **broad**: the stress-test bounds. Deliberately beyond normal operation.

## Failure criterion (robot-specific)

A rollout is a failure if the engine's fall signal fires (trunk height below the target's
`trunk_height_min`: 0.18 m for Go1, 0.40 m for G1; or trunk up-axis below 0.5) and the
physical-validity gate does not flag it as a simulator artifact (non-finite state, floor
penetration > 4 cm before the fall, joint accelerations > 1e4; see `docs/validity.md`).
