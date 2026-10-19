"""Observation shim: map the env's ``obs`` dict to the flat vector a foreign policy expects.

A foreign MLP policy (RSL-RL / Isaac Lab / a torch/ONNX export) was trained on some
concrete observation vector -- a specific ordering of a specific set of channels, often
with a per-dim running-mean/std normalization baked into training. BreakPoint's env, by
contrast, hands the policy a dict (``obs["state"]``, ``obs["privileged_state"]``, ...).

``ObsSpec`` is the adapter between the two: it says WHICH key(s) to read, in what order,
optionally which indices to select/reorder, and the per-dim ``mean``/``std`` to apply. It
is deliberately array-module-agnostic (``transform(obs, xp)`` takes ``numpy`` in tests and
``jax.numpy`` in production) and pure, so the mapping logic is fully CPU-testable and the
same code runs on-device inside the evaluator's ``jax.vmap``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ObsSpec:
    """Map ``obs`` (a dict) -> the flat input vector a foreign MLP policy expects.

    Parameters
    ----------
    keys : str | sequence[str]
        The obs dict key(s) to read. Multiple keys are concatenated in order along the
        last axis. Default ``("state",)`` -- the non-privileged observation the evaluator
        feeds the policy.
    indices : sequence[int] | None
        Optional index selection/reordering applied AFTER concatenation. Use when the
        foreign policy expects a different channel order or a subset of ``obs["state"]``.
    mean, std : array | None
        Optional per-dim normalization applied AFTER index selection as
        ``(x - mean) / std`` (the standard running-statistics normalization most RL
        frameworks bake into their observation pipeline). Length must equal the mapped
        vector length.
    expected_dim : int | None
        The expected mapped vector length. ``transform`` raises a clear error if the
        actual obs length differs -- this is the guard against a silent obs/policy
        mismatch. ``from_params`` / ``from_onnx`` set this to the MLP's first-layer input
        dim automatically, so a wrong obs is caught before it produces garbage actions.

    Notes
    -----
    ``transform`` is pure and uses only ``xp`` ops (``asarray``/``concatenate``/``take``
    plus arithmetic), so it is vmappable on-device. The length check reads a STATIC shape
    (``x.shape[-1]``), which is a concrete int under both numpy and a jax trace, so it does
    not introduce a data-dependent branch.
    """

    keys: object = ("state",)
    indices: object = None
    mean: object = None
    std: object = None
    expected_dim: object = None
    # Contract fields (Behavioral Contract Identification, adapters/contract_id.py): a
    # per-dim affine map applied AFTER index selection and BEFORE normalization,
    # ``x = (x - offset) * scale``. This is where a policy's own obs scales (legged_gym's
    # ``dof_vel: 0.05``), a different default pose, or a sign flip live. ``contract`` is
    # the declarative term list the spec was built from (kept only so ``to_yaml`` can
    # write it back); it never changes the math.
    offset: object = None
    scale: object = None
    contract: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if isinstance(self.keys, str):
            self.keys = (self.keys,)
        else:
            self.keys = tuple(self.keys)
        if not self.keys:
            raise ValueError("ObsSpec.keys must name at least one obs key")
        if self.indices is not None:
            self.indices = tuple(int(i) for i in self.indices)
        if self.mean is not None:
            self.mean = np.asarray(self.mean, dtype=float)
        if self.std is not None:
            self.std = np.asarray(self.std, dtype=float)
            if np.any(self.std == 0):
                raise ValueError("ObsSpec.std has a zero entry (division by zero)")
        if self.expected_dim is not None:
            self.expected_dim = int(self.expected_dim)
        if self.offset is not None:
            self.offset = np.asarray(self.offset, dtype=float)
        if self.scale is not None:
            self.scale = np.asarray(self.scale, dtype=float)

        od = self.output_dim()
        for nm, v in (("mean", self.mean), ("std", self.std), ("offset", self.offset),
                      ("scale", self.scale)):
            if v is not None and od is not None and v.shape[-1] != od:
                raise ValueError(
                    f"ObsSpec.{nm} has length {v.shape[-1]} but the mapped vector length "
                    f"is {od}; normalization must match the mapped obs length"
                )

    def output_dim(self):
        """The mapped vector length if it can be determined statically, else ``None``."""
        if self.expected_dim is not None:
            return int(self.expected_dim)
        if self.indices is not None:
            return len(self.indices)
        if self.mean is not None:
            return int(self.mean.shape[-1])
        if self.std is not None:
            return int(self.std.shape[-1])
        return None

    def with_expected_dim(self, dim):
        """Return a copy with ``expected_dim`` set (re-validates mean/std lengths)."""
        return dataclasses.replace(self, expected_dim=int(dim))

    def transform(self, obs, xp):
        """Map ``obs`` (dict) -> flat vector via ``xp``. Raises on a length mismatch."""
        parts = []
        for k in self.keys:
            if k not in obs:
                raise KeyError(
                    f"ObsSpec key {k!r} not in obs (available keys: {sorted(obs)})"
                )
            parts.append(xp.asarray(obs[k]))
        x = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=-1)

        if self.indices is not None:
            # static numpy index array: works under numpy and inside a jax trace (jnp.take
            # rejects a python list, which is how this path used to fail on-device)
            x = x[..., np.asarray(self.indices, dtype=int)]

        n = int(x.shape[-1])
        if self.expected_dim is not None and n != self.expected_dim:
            raise ValueError(
                f"observation length {n} does not match the policy's expected input dim "
                f"{self.expected_dim}. Check ObsSpec.keys/indices and that this policy was "
                f"trained on this env's observation."
            )

        if self.offset is not None:
            x = x - xp.asarray(self.offset)
        if self.scale is not None:
            x = x * xp.asarray(self.scale)
        if self.mean is not None:
            x = x - xp.asarray(self.mean)
        if self.std is not None:
            x = x / xp.asarray(self.std)
        return x

    # ------------------------------------------------------------------ #
    # Declarative contract (YAML)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path_or_dict, robot=None):
        """Build an ``ObsSpec`` from a contract YAML (path, YAML text or dict).

        The YAML lists named obs terms in the POLICY's order; see ``PolicyContract`` for
        the schema. Only the observation side is returned here; use
        ``load_contract(...).to_output_transform()`` for the action side.
        """
        return load_contract(path_or_dict, robot=robot).to_obs_spec()

    def to_yaml(self, path=None):
        """Serialize to contract YAML. A spec built from a term list writes that list back
        (term form); any other spec writes its raw ``keys``/``indices``/affine/norm arrays
        (raw form), which ``from_yaml`` also reads."""
        import yaml

        if self.contract is not None:
            d = dict(self.contract)
        else:
            d = {"version": 1, "raw": {"keys": list(self.keys)}}
            for nm in ("indices", "offset", "scale"):
                v = getattr(self, nm)
                if v is not None:
                    d["raw"][nm] = [float(x) if nm != "indices" else int(x)
                                    for x in np.asarray(v).reshape(-1)]
            if self.expected_dim is not None:
                d["raw"]["expected_dim"] = int(self.expected_dim)
        if self.mean is not None or self.std is not None:
            d["normalization"] = {
                k: [float(x) for x in np.asarray(v).reshape(-1)]
                for k, v in (("mean", self.mean), ("std", self.std)) if v is not None}
        text = yaml.safe_dump(d, sort_keys=False, default_flow_style=None)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


# --------------------------------------------------------------------------- #
# Target interfaces: where each named obs term lives in a target env's obs vector
# --------------------------------------------------------------------------- #
# Canonical term names and the spellings the big frameworks use for them. Isaac Lab
# (``mdp.base_lin_vel``, ``generated_commands``, ``joint_pos_rel``, ``last_action``),
# legged_gym / RSL-RL task configs (``lin_vel``, ``dof_pos``, ``actions``) and
# unitree_rl_gym deploy scripts (``ang_vel``, ``qj``, ``dqj``, ``cmd``) all appear here.
TERM_ALIASES = {
    "base_lin_vel": ("base_lin_vel", "lin_vel", "linvel", "base_linear_velocity"),
    "base_ang_vel": ("base_ang_vel", "ang_vel", "gyro", "angvel",
                     "base_angular_velocity"),
    "projected_gravity": ("projected_gravity", "gravity", "gravity_vec",
                          "projected_gravity_b"),
    "commands": ("commands", "command", "velocity_commands", "generated_commands",
                 "cmd", "base_velocity"),
    "joint_pos_rel": ("joint_pos_rel", "joint_pos", "dof_pos", "qj", "joint_angles"),
    "joint_vel": ("joint_vel", "joint_vel_rel", "dof_vel", "dqj"),
    "last_action": ("last_action", "last_act", "actions", "action", "prev_action",
                    "previous_action"),
}
JOINT_TERMS = ("joint_pos_rel", "joint_vel", "last_action")
_ALIAS_TO_TERM = {a: t for t, al in TERM_ALIASES.items() for a in al}


def canonical_term(name):
    """Canonical term name for any known alias, else ``None``."""
    return _ALIAS_TO_TERM.get(str(name).strip().lower())


@dataclass(frozen=True)
class TargetInterface:
    """The obs/action interface of one BreakPoint target env, as data.

    ``layout`` maps each canonical term to ``(start, dim)`` inside ``obs[keys[0]]``;
    ``joint_names`` / ``default_pose`` are in the env's ACTUATOR order; the env applies
    ``motor_target = default_pose + action_scale * action`` with PD gains ``kp``/``kd``.
    """

    robot: str
    obs_dim: int
    layout: dict
    joint_names: tuple
    default_pose: tuple
    action_scale: float
    kp: float
    kd: float
    keys: tuple = ("state",)
    control_dt: float = 0.02


# Go1JoystickFlatTerrain (mujoco_playground go1/joystick.py ``_get_obs``): state =
# [linvel 3, gyro 3, gravity 3, joint_angles - default 12, joint_vel 12, last_act 12,
# command 3]; actuators are leg-major FR, FL, RR, RL; home keyframe; Kp 35 Kd 0.5;
# motor_targets = default_pose + action * 0.5; ctrl_dt 0.02. Verified against the loaded
# env by tests/test_contract_id.py (slow) and scripts/bench_contract.py.
GO1_INTERFACE = TargetInterface(
    robot="go1",
    obs_dim=48,
    layout={"base_lin_vel": (0, 3), "base_ang_vel": (3, 3), "projected_gravity": (6, 3),
            "joint_pos_rel": (9, 12), "joint_vel": (21, 12), "last_action": (33, 12),
            "commands": (45, 3)},
    joint_names=tuple(f"{leg}_{j}_joint" for leg in ("FR", "FL", "RR", "RL")
                      for j in ("hip", "thigh", "calf")),
    default_pose=(0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8),
    action_scale=0.5,
    kp=35.0,
    kd=0.5,
)

TARGET_INTERFACES = {"go1": GO1_INTERFACE}


def target_interface(robot):
    """The ``TargetInterface`` for a registered robot name (or pass one through)."""
    if isinstance(robot, TargetInterface):
        return robot
    key = str(robot).lower()
    if key not in TARGET_INTERFACES:
        raise KeyError(
            f"no declarative obs interface is registered for robot {robot!r} (known: "
            f"{sorted(TARGET_INTERFACES)}). Use the raw form of the contract YAML "
            f"(raw: {{keys, indices, scale, offset}}) or register a TargetInterface in "
            f"breakpoint_mjx/adapters/obs_spec.py."
        )
    return TARGET_INTERFACES[key]


def _norm_joint(name):
    s = str(name).strip().lower()
    for suf in ("_joint", "joint"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.replace(".", "_").strip("_")


def joint_permutation(policy_joint_names, env_joint_names):
    """``perm[j]`` = env actuator index of the policy's j-th joint (name match).

    Matching ignores case and a trailing ``_joint``. Raises naming every joint that has
    no match, since a guessed joint mapping is exactly the silent error this exists to
    prevent.
    """
    env_n = [_norm_joint(n) for n in env_joint_names]
    pol_n = [_norm_joint(n) for n in policy_joint_names]
    if len(pol_n) != len(env_n):
        raise ValueError(
            f"policy lists {len(pol_n)} joints but the target has {len(env_n)} actuators"
        )
    missing = [p for p in pol_n if p not in env_n]
    if missing:
        raise ValueError(
            f"policy joints {missing} have no match among the target's actuators "
            f"{list(env_joint_names)}; rename them in the contract's joint_names"
        )
    perm = [env_n.index(p) for p in pol_n]
    if len(set(perm)) != len(perm):
        raise ValueError(f"joint_names map two policy joints onto one actuator: {perm}")
    return perm


# --------------------------------------------------------------------------- #
# The contract: a policy's full obs + action interface as data
# --------------------------------------------------------------------------- #
def _vec(v, n, name):
    a = np.asarray(v if v is not None else 1.0, dtype=float).reshape(-1)
    if a.size == 1:
        return np.full(n, float(a[0]))
    if a.size != n:
        raise ValueError(f"{name} has {a.size} entries, expected 1 or {n}")
    return a


@dataclass
class PolicyContract:
    """A frozen policy's observation/action interface, resolved against one target.

    Everything is index arrays and per-dim affine coefficients, which is the point: a
    contract is DATA, so many candidate contracts ``vmap`` together in one batched
    rollout (``adapters/contract_id.py``).

    Semantics (``src`` = ``concat(obs[k] for k in keys)``, ``y`` = raw policy output):

        policy_input = (src[obs_idx] - obs_offset) * obs_scale
        env_action[j] = y[act_idx[j]] * act_scale[j] + act_offset[j]

    ``terms`` / ``policy_joint_names`` / ``action`` keep the declarative description the
    arrays were built from, so ``to_dict`` round-trips.
    """

    robot: str
    keys: tuple
    obs_idx: np.ndarray
    obs_offset: np.ndarray
    obs_scale: np.ndarray
    act_idx: np.ndarray
    act_scale: np.ndarray
    act_offset: np.ndarray
    terms: list = field(default_factory=list)
    policy_joint_names: tuple = ()
    action: dict = field(default_factory=dict)
    normalization: object = None      # (mean, std) in policy-input space, or None
    warnings: list = field(default_factory=list)
    default_joint_pos: object = None  # policy-order default pose, when it was declared
    # Blocks that describe the policy's training conditions rather than its obs/action
    # map: ``actuation`` (PD gains the policy was trained at; adapters/actuation.py) and
    # ``operational`` (the training domain-randomization ranges as a riskcert ODD,
    # ``{theta_dim: {range: [lo, hi]}}``). Carried so to_dict / to_yaml round-trip them.
    extras: dict = field(default_factory=dict)
    # Observation history (adapters/stateful.py): None, or {length, layout, fill, order}.
    # ``term_dims`` are the per-term widths of ONE frame, in term order.
    history: object = None
    term_dims: tuple = ()

    @property
    def obs_dim(self):
        return int(self.obs_idx.shape[0])

    @property
    def action_dim(self):
        return int(self.act_idx.shape[0])

    def arrays(self):
        """The vmappable part: ``(obs_idx, obs_offset, obs_scale, act_idx, act_scale,
        act_offset)`` as numpy arrays."""
        return (self.obs_idx, self.obs_offset, self.obs_scale, self.act_idx,
                self.act_scale, self.act_offset)

    def to_obs_spec(self, mean=None, std=None):
        if mean is None and std is None and self.normalization is not None:
            mean, std = self.normalization
        return ObsSpec(keys=self.keys, indices=self.obs_idx, offset=self.obs_offset,
                       scale=self.obs_scale, mean=mean, std=std,
                       expected_dim=self.obs_dim, contract=self.to_dict())

    def to_output_transform(self, squash=False, mean_dim=None):
        from breakpoint_mjx.adapters.mlp import OutputTransform

        return OutputTransform(squash=squash, scale=self.act_scale, bias=self.act_offset,
                               mean_dim=mean_dim, perm=self.act_idx)

    def to_dict(self):
        d = {"version": 1, "robot": self.robot}
        if self.policy_joint_names:
            d["joint_names"] = list(self.policy_joint_names)
        if self.default_joint_pos is not None:
            d["default_joint_pos"] = [float(x) for x in self.default_joint_pos]
        d["obs"] = [dict(t) for t in self.terms]
        if self.history:
            d["history"] = dict(self.history)
        if self.action:
            d["action"] = dict(self.action)
        for k in EXTRA_BLOCKS:
            if self.extras.get(k):
                d[k] = self.extras[k]
        return d

    def to_yaml(self, path=None, include_normalization=True):
        import yaml

        d = self.to_dict()
        if include_normalization and self.normalization is not None:
            m, s = self.normalization
            d["normalization"] = {"mean": [float(x) for x in np.asarray(m)],
                                  "std": [float(x) for x in np.asarray(s)]}
        text = yaml.safe_dump(d, sort_keys=False, default_flow_style=None)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def describe(self):
        """One line per term: ``term[dim] <- src[a:b] scale s``; used by the CLI diff."""
        rows = []
        for t in self.terms:
            sc = t.get("scale", 1.0)
            rows.append(f"{t['term']}  scale={sc}")
        a = self.action or {}
        rows.append(f"action  scale={a.get('scale', '(target)')}  "
                    f"joint order={list(self.policy_joint_names) or '(target order)'}")
        return rows


def _clean_scale(v):
    a = np.asarray(v, dtype=float).reshape(-1)
    return float(a[0]) if a.size == 1 else [float(x) for x in a]


# Contract blocks kept verbatim alongside the obs/action map (see PolicyContract.extras).
EXTRA_BLOCKS = ("actuation", "operational")

def build_contract(spec, robot=None):
    """Resolve a declarative contract (dict) against a target into a ``PolicyContract``.

    Schema (all optional except ``obs``)::

        version: 1
        robot: go1
        joint_names: [FL_hip_joint, FR_hip_joint, ...]   # POLICY order
        default_joint_pos: [0.1, 0.8, -1.5, ...]         # policy order; default = target's
        action: {scale: 0.25}                            # rad per unit; default = target's
        obs:                                             # POLICY order
          - {term: base_ang_vel, scale: 0.25}
          - {term: projected_gravity}
          - {term: commands, scale: [2.0, 2.0, 0.25]}
          - {term: joint_pos_rel}
          - {term: joint_vel, scale: 0.05}
          - {term: last_action}
        history: {length: 5, layout: term}               # optional (adapters/stateful.py)
        normalization: {mean: [...], std: [...]}          # optional

    Joint terms are permuted from the target's actuator order into the policy's order
    via ``joint_names``. ``joint_pos_rel`` is taken relative to the POLICY's default pose
    (an offset when it differs from the target's). The action maps
    ``env_action = (policy_scale * y + default_p - default_env) / env_scale`` per joint,
    and ``last_action`` feeds the policy its own raw previous output by inverting that
    map. A per-term ``scale`` may be a scalar or a per-channel list (negative = sign flip).
    A term with ``start``/``dim`` instead of a known name reads raw channels.
    """
    if "raw" in spec:
        raise ValueError("raw-form contracts have no terms; use ObsSpec.from_yaml")
    from breakpoint_mjx.adapters.stateful import parse_history

    history = parse_history(spec.get("history"))
    extras = {}
    if spec.get("actuation"):
        from breakpoint_mjx.adapters.actuation import parse_actuation

        extras["actuation"] = parse_actuation(spec)
    if spec.get("operational"):
        op = spec["operational"]
        if not isinstance(op, dict) or not all(
                isinstance(v, dict) and ("range" in v or "mean" in v) for v in op.values()):
            raise ValueError("operational: must map theta dim names to {range: [lo, hi]} "
                             "(riskcert.OperationalDist.from_space schema)")
        extras["operational"] = {k: dict(v) for k, v in op.items()}
    iface = target_interface(robot or spec.get("robot") or "go1")
    n_act = len(iface.joint_names)
    env_default = np.asarray(iface.default_pose, dtype=float)
    warns = []

    names = spec.get("joint_names")
    perm = (np.arange(n_act) if not names
            else np.asarray(joint_permutation(names, iface.joint_names)))
    pol_default = spec.get("default_joint_pos")
    if pol_default is None:
        pol_default = env_default[perm]
    pol_default = _vec(pol_default, n_act, "default_joint_pos")

    act = dict(spec.get("action") or {})
    pol_scale = _vec(act.get("scale", iface.action_scale), n_act, "action.scale")
    if np.any(pol_scale == 0):
        raise ValueError("action.scale has a zero entry")
    # env actuator e = perm[j] receives y[j]:  env_a = (s_p*y + q0_p - q0_e) / s_e
    act_idx = np.empty(n_act, dtype=int)
    act_idx[perm] = np.arange(n_act)
    act_scale = np.empty(n_act)
    act_scale[perm] = pol_scale / iface.action_scale
    act_offset = np.empty(n_act)
    act_offset[perm] = (pol_default - env_default[perm]) / iface.action_scale
    for k in ("clip", "clip_actions"):
        if act.get(k) is not None:
            warns.append(f"action.{k}={act[k]} is not applied: BreakPoint does not clip "
                         f"policy actions (the target's actuator limits still apply)")

    idx, off, sc, terms, term_dims = [], [], [], [], []
    for raw_t in spec.get("obs") or []:
        t = dict(raw_t) if isinstance(raw_t, dict) else {"term": str(raw_t)}
        name = t.get("term")
        canon = canonical_term(name) if name is not None else None
        if canon is None:
            if "start" in t and "dim" in t:
                s0, dim = int(t["start"]), int(t["dim"])
                ii = np.arange(s0, s0 + dim)
                oo = _vec(t.get("offset", 0.0), dim, f"{name}.offset")
                ss = _vec(t.get("scale", 1.0), dim, f"{name}.scale")
            else:
                raise ValueError(
                    f"obs term {name!r} is not one BreakPoint can produce on "
                    f"{iface.robot} (known: {sorted(TERM_ALIASES)}). Give it start/dim "
                    f"to read raw channels, or drop it from the policy.")
        else:
            if canon not in iface.layout:
                raise ValueError(f"target {iface.robot} has no {canon!r} channels")
            s0, dim = iface.layout[canon]
            base = np.arange(s0, s0 + dim)
            term_scale = _vec(t.get("scale", 1.0), dim, f"{name}.scale")
            if canon in JOINT_TERMS:
                ii = base[perm]
                if canon == "joint_pos_rel":
                    # policy wants q - q0_p = (q - q0_e) - (q0_p - q0_e)
                    oo = pol_default - env_default[perm]
                elif canon == "last_action":
                    # policy wants its own raw y: y = (env_a - act_offset) / act_scale
                    oo = act_offset[perm]
                    term_scale = term_scale / act_scale[perm]
                else:
                    oo = np.zeros(dim)
            else:
                ii = base
                oo = np.zeros(dim)
            if "offset" in t:
                oo = oo + _vec(t["offset"], dim, f"{name}.offset")
            ss = term_scale
            t["term"] = canon
        if "clip" in t:
            warns.append(f"obs term {t['term']}: clip={t['clip']} is not applied")
        idx.append(ii)
        off.append(oo)
        sc.append(ss)
        term_dims.append(int(len(ii)))
        clean = {"term": t["term"]}
        if canon is None:
            clean.update(start=int(t["start"]), dim=int(t["dim"]))
        if "scale" in t:
            clean["scale"] = _clean_scale(t["scale"])
        if "offset" in t:
            clean["offset"] = _clean_scale(t["offset"])
        terms.append(clean)
    if not idx:
        raise ValueError("contract has no obs terms")

    norm = spec.get("normalization")
    normalization = None
    if norm:
        normalization = (np.asarray(norm["mean"], dtype=float),
                         np.asarray(norm["std"], dtype=float))
    action_out = {}
    if "scale" in act:
        action_out["scale"] = _clean_scale(act["scale"])
    return PolicyContract(
        robot=iface.robot, keys=tuple(iface.keys),
        obs_idx=np.concatenate(idx).astype(int), obs_offset=np.concatenate(off),
        obs_scale=np.concatenate(sc), act_idx=act_idx, act_scale=act_scale,
        act_offset=act_offset, terms=terms,
        policy_joint_names=tuple(names) if names else (),
        action=action_out, normalization=normalization, warnings=warns,
        default_joint_pos=(None if spec.get("default_joint_pos") is None
                           else [float(x) for x in pol_default]),
        extras=extras, history=history, term_dims=tuple(term_dims))


def _read_yaml(path_or_dict):
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    import yaml

    s = str(path_or_dict)
    p = Path(s)
    try:
        is_file = p.exists()
    except OSError:  # a long YAML string is not a path
        is_file = False
    text = p.read_text(encoding="utf-8") if is_file else s
    d = yaml.safe_load(text)
    if not isinstance(d, dict):
        raise ValueError(f"contract YAML must be a mapping, got {type(d).__name__}")
    return d


def load_contract(path_or_dict, robot=None):
    """Read a contract YAML (path / text / dict). Term form -> ``PolicyContract``; raw form
    (``raw: {keys, indices, ...}``) -> an ``ObsSpec``-only contract wrapper."""
    d = _read_yaml(path_or_dict)
    if "raw" in d:
        return _RawContract(d)
    return build_contract(d, robot=robot)


class _RawContract:
    """A raw-form contract: obs side only (the action side is the target's identity)."""

    def __init__(self, d):
        self._d = d

    def to_obs_spec(self, mean=None, std=None):
        r = self._d["raw"]
        norm = self._d.get("normalization") or {}
        return ObsSpec(keys=r.get("keys", ("state",)), indices=r.get("indices"),
                       offset=r.get("offset"), scale=r.get("scale"),
                       mean=mean if mean is not None else norm.get("mean"),
                       std=std if std is not None else norm.get("std"),
                       expected_dim=r.get("expected_dim"))

    def to_output_transform(self, squash=False, mean_dim=None):
        return None

    warnings = ()
