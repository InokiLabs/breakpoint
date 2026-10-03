"""Actuation-faithful BYO: run a foreign policy at the PD gains it was trained with.

A position-controlled legged policy is trained against one PD loop: ``tau = kp * (q* - q)
- kd * qd``. The policy's action scale, its gait and its stability margins are all tuned to
that loop. BreakPoint's targets ship their own gains (the Playground Go1 uses Kp 35 / Kd
0.5), so a policy trained at Kp 20 and dropped in unchanged runs on a stiffer robot than it
learned on. Any failure the search then finds may be a sim-to-sim artifact of the gain
change rather than a property of the policy.

A contract YAML can therefore carry an ``actuation:`` block::

    actuation:
      kp: 20.0            # scalar, or a list in the POLICY's joint order
      kd: 0.5
      action_scale: 0.25  # informational; the contract's action.scale does the mapping
      source: deploy      # where the numbers came from (init writes this)
      honor: true         # apply them to the target's actuators

With ``honor: true`` the Go1 target (``targets/go1.load_env``) rewrites its actuators to
those gains on BOTH the C MuJoCo model and the MJX model, before any evaluator captures
the base model. The ThetaSpace ``kp_scale`` / ``kd_scale`` multipliers then scale the
policy's own gains instead of the target's. ``BP_HONOR_ACTUATION=0`` switches honoring off
globally (the gap check in ``check_policy`` uses it to load the native-gain twin).

Where each gain lives depends on the target's model: ``kp`` always goes to
``actuator_gainprm[:, 0]`` and ``-kp`` to ``actuator_biasprm[:, 1]`` (a MuJoCo position
actuator). ``kd`` goes where the target already keeps its derivative term: the Playground
Go1 has ``biasprm[:, 2] == 0`` and puts Kd in ``dof_damping`` of the actuated dofs, so
there it is written to ``dof_damping``; a model that uses ``biasprm[:, 2] = -kd`` gets it
there. Writing both would double the damping.

Import-safe: numpy only at import; ``jax.numpy`` is imported inside ``apply_actuation``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GAIN_MISMATCH_FATAL = 0.20   # > 20% relative kp/kd mismatch with honor:false is fatal
_ENV_OFF = "BP_HONOR_ACTUATION"


def actuation_block(kp, kd, action_scale=None, source="", honor=True):
    """The contract's ``actuation:`` mapping. A per-joint list collapses to a scalar when
    every entry is equal, so the common case stays readable."""
    def _clean(v):
        if v is None:
            return None
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 1 or np.allclose(a, a[0]):
            return float(a[0])
        return [float(x) for x in a]

    d = {"kp": _clean(kp), "kd": _clean(kd)}
    if action_scale is not None:
        d["action_scale"] = _clean(action_scale)
    d["source"] = str(source)
    d["honor"] = bool(honor)
    return {k: v for k, v in d.items() if v is not None}


def parse_actuation(contract):
    """The validated ``actuation:`` block of a contract dict / path / text, or ``None``."""
    from breakpoint_mjx.adapters.obs_spec import _read_yaml

    d = contract if isinstance(contract, dict) else _read_yaml(contract)
    a = d.get("actuation")
    if not a:
        return None
    if not isinstance(a, dict):
        raise ValueError("actuation: must be a mapping {kp, kd, action_scale, source, honor}")
    if a.get("kp") is None:
        raise ValueError("actuation: needs kp (the policy's training stiffness)")
    for nm in ("kp", "kd"):
        if a.get(nm) is not None and np.any(np.asarray(a[nm], dtype=float) < 0):
            raise ValueError(f"actuation.{nm} must be non-negative")
    out = dict(a)
    out["honor"] = bool(a.get("honor", True))
    return out


def resolve_gains(block, n_act, perm=None, native_kd=None):
    """Per-ACTUATOR ``(kp, kd)`` arrays in the target's actuator order.

    ``perm[j]`` is the env actuator index of the policy's j-th joint (from the contract's
    joint_names); a per-joint list in the block is in the POLICY's order and is permuted
    here. A missing ``kd`` keeps ``native_kd``.
    """
    def _per(v, name):
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 1:
            return np.full(n_act, float(a[0]))
        if a.size != n_act:
            raise ValueError(f"actuation.{name} has {a.size} entries; the target has "
                             f"{n_act} actuators")
        if perm is None:
            return a
        out = np.empty(n_act)
        out[np.asarray(perm)] = a
        return out

    kp = _per(block["kp"], "kp")
    if block.get("kd") is None:
        kd = (np.zeros(n_act) if native_kd is None
              else np.asarray(native_kd, dtype=float).reshape(-1))
    else:
        kd = _per(block["kd"], "kd")
    return kp, kd


def gain_mismatch(block, native_kp, native_kd):
    """Largest relative difference between the block's gains and the target's native ones
    (``|policy - native| / native``, over kp and kd, every joint)."""
    rel = []
    for v, nat in ((block.get("kp"), native_kp), (block.get("kd"), native_kd)):
        if v is None or nat is None:
            continue
        a = np.asarray(v, dtype=float).reshape(-1)
        n = np.asarray(nat, dtype=float).reshape(-1)
        denom = np.maximum(np.abs(n), 1e-9)
        rel.append(float(np.max(np.abs(np.broadcast_to(a, np.broadcast_shapes(a.shape, n.shape))
                                        - n) / denom)))
    return max(rel) if rel else 0.0


def gain_gate_row(block, native_kp, native_kd, allow=False):
    """The init / check-policy row for a gain mismatch, or ``None``.

    ``honor: true`` -> an ``info`` row (the gains are applied). ``honor: false`` with a
    mismatch above 20% -> ``fatal`` (a failure found at the target's gains may be a
    sim-to-sim artifact), downgraded to ``warn`` when ``allow`` (the explicit override).
    """
    if block is None:
        return None
    mm = gain_mismatch(block, native_kp, native_kd)
    if mm <= 0.05:
        return None
    tail = (f"policy gains kp={block.get('kp')} kd={block.get('kd')} vs target "
            f"kp={_fmt(native_kp)} kd={_fmt(native_kd)} ({mm:.0%} apart)")
    if block.get("honor", True):
        return ("info", f"{tail}; HONORED: the target's actuators are set to the "
                        f"policy's gains (actuation.honor: true)")
    if mm > GAIN_MISMATCH_FATAL and not allow:
        return ("fatal", f"{tail} and actuation.honor is false: every failure found would "
                         f"be measured on a robot the policy never trained on. Set honor: "
                         f"true, or override with --allow-gain-mismatch / "
                         f"BP_ALLOW_GAIN_MISMATCH=1")
    return ("warn", f"{tail}; NOT honored (actuation.honor: false), the target's gains "
                    f"are used")


def _fmt(v):
    a = np.asarray(v, dtype=float).reshape(-1)
    return f"{a[0]:g}" if a.size == 1 or np.allclose(a, a[0]) else str([round(float(x), 3)
                                                                          for x in a])


def honoring_enabled():
    return os.environ.get(_ENV_OFF, "1").strip().lower() not in ("0", "false", "no", "off")


def native_gains(mj_model):
    """(kp, kd, kd_site) of a MuJoCo model's position actuators.

    ``kd_site`` is ``"biasprm"`` when the model keeps a derivative term in
    ``actuator_biasprm[:, 2]`` and ``"dof_damping"`` otherwise (Playground Go1)."""
    kp = np.asarray(mj_model.actuator_gainprm[:, 0], dtype=float).copy()
    b2 = np.asarray(mj_model.actuator_biasprm[:, 2], dtype=float)
    if np.any(b2 != 0):
        return kp, -b2.copy(), "biasprm"
    dofs = _actuator_dofs(mj_model)
    return kp, np.asarray(mj_model.dof_damping, dtype=float)[dofs].copy(), "dof_damping"


def _actuator_dofs(mj_model):
    """dof index driven by each joint actuator (trntype joint)."""
    jids = np.asarray(mj_model.actuator_trnid[:, 0], dtype=int)
    return np.asarray(mj_model.jnt_dofadr, dtype=int)[jids]


def apply_actuation(env, block, perm=None, log=print):
    """Set ``env``'s actuators to the block's gains on the mj model AND the MJX model.

    The env is a Playground ``MjxEnv``: its evaluators read ``env.mjx_model`` (a property
    over ``env._mjx_model``), and a model passed to the constructor is ignored, so the MJX
    model is swapped in place. Returns a record of what changed (also stored on
    ``env._bp_actuation``). Idempotent for the same block.
    """
    import jax.numpy as jnp

    mj = env._mj_model
    n_act = int(mj.nu)
    kp0, kd0, site = native_gains(mj)
    kp, kd = resolve_gains(block, n_act, perm=perm, native_kd=kd0)
    mj.actuator_gainprm[:, 0] = kp
    mj.actuator_biasprm[:, 1] = -kp
    dofs = _actuator_dofs(mj)
    if site == "biasprm":
        mj.actuator_biasprm[:, 2] = -kd
    else:
        mj.dof_damping[dofs] = kd
    mx = env._mjx_model
    repl = {
        "actuator_gainprm": jnp.asarray(mj.actuator_gainprm, dtype=mx.actuator_gainprm.dtype),
        "actuator_biasprm": jnp.asarray(mj.actuator_biasprm, dtype=mx.actuator_biasprm.dtype),
        "dof_damping": jnp.asarray(mj.dof_damping, dtype=mx.dof_damping.dtype),
    }
    env._mjx_model = mx.replace(**repl)
    rec = {"native_kp": kp0.tolist(), "native_kd": kd0.tolist(), "kd_site": site,
           "kp": kp.tolist(), "kd": kd.tolist(), "source": block.get("source", "")}
    env._bp_actuation = rec
    if log is not None:
        log(f"actuation: honoring the policy's PD gains ({block.get('source') or 'contract'}"
            f"): kp {_fmt(kp0)} -> {_fmt(kp)}, kd {_fmt(kd0)} -> {_fmt(kd)} "
            f"(kd written to {site}); ThetaSpace kp_scale/kd_scale now scale these")
    return rec


def contract_perm(contract_dict, robot):
    """env actuator index of each policy joint, from the contract's joint_names."""
    names = contract_dict.get("joint_names")
    if not names:
        return None
    from breakpoint_mjx.adapters.obs_spec import joint_permutation, target_interface

    return joint_permutation(names, target_interface(robot).joint_names)


def maybe_apply_from_env(env, robot, spec_path=None, force=False, log=print):
    """Apply the actuation block of the contract named by ``spec_path`` / ``BP_OBS_SPEC``
    when it says ``honor: true`` (or ``force``). Returns the change record or ``None``.

    Skipped when ``BP_HONOR_ACTUATION=0`` (unless ``force``) or when there is no contract
    or no block."""
    if not force and not honoring_enabled():
        return None
    path = spec_path or os.environ.get("BP_OBS_SPEC")
    if not path or not Path(path).exists():
        return None
    from breakpoint_mjx.adapters.obs_spec import _read_yaml

    d = _read_yaml(path)
    block = parse_actuation(d)
    if block is None or not (block["honor"] or force):
        return None
    c_robot = str(d.get("robot") or robot).lower()
    if c_robot != str(robot).lower():
        raise ValueError(f"contract {path} is for robot {c_robot!r}; the target is {robot!r}")
    return apply_actuation(env, block, perm=contract_perm(d, robot), log=log)
