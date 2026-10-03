# Contributing to BreakPoint

Thanks for helping. Bug reports, new robot targets, policy importers, batteries and
evidence methods are all welcome.

## Ground rules

- **Licence.** BreakPoint is Apache-2.0. By submitting a contribution you agree it is
  licensed under the Apache License, Version 2.0, as described in section 5 of the
  licence. Only submit code you have the right to contribute.
- **Tests first.** Every behaviour change comes with a test. The suite must pass on CPU:

  ```bash
  python -m pip install -e ".[dev,attest]"
  python -m pytest -q
  ```

  GPU paths are covered by `tests/test_gpu_smoke.py`, which skips without a GPU. If your
  change touches a GPU path, say in the pull request how you ran it.
- **Honest numbers.** Any number in a doc or report must come from a run you can
  reproduce from the repo, with the command written next to it. Synthetic or planted
  benchmarks must be labelled as synthetic. Never present a smoke-budget result as a
  converged one.
- **No weights or private data in the repo.** Do not commit checkpoints, trajectories,
  API keys or `.env` files. Reports must not contain weights, trajectories, file paths
  beyond the policy name, or the robot owner's identity.
- **Report schema.** `breakpoint.report/v1` field names are a shared contract with other
  tools. Do not rename or remove fields; propose additive changes in an issue first.

## Workflow

1. Open an issue describing the bug or feature (for anything larger than a small fix).
2. Fork, branch, and keep the change focused.
3. Run the test suite, then open a pull request that explains what changed and how you
   verified it.

## Adding a robot target

Targets live in `breakpoint_mjx/targets/` behind the `Target` abstraction, with their
parameter space in `breakpoint_mjx/configs/<robot>/theta_space.yaml`. Add a target, a
theta space, and tests in `tests/test_targets.py`; `breakpoint check-policy` must pass a
trained policy at nominal before any search result for that robot is meaningful.

## Security

Please do not report security issues in public issues. Use GitHub's private
vulnerability reporting on this repository (Security tab, "Report a vulnerability").
