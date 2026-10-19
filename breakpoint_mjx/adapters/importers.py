"""Weight importers: foreign policy weights -> a JAX-vmappable ``MLPPolicy`` (T4).

Three entry points, in increasing dependency weight:

* ``from_params(layers, ...)`` -- the dependency-free core. Build an ``MLPPolicy`` from a
  plain list/dict of numpy arrays (a torch ``state_dict`` dumped to numpy, flax params, or
  hand-built weights). Everything else funnels through this.
* ``from_npz(path, ...)`` -- ``np.load`` an ``.npz`` of named arrays, then ``from_params``.
* ``from_onnx(path, ...)`` -- READ an ONNX graph's initializers + activation nodes (does
  NOT execute the graph), extract ordered ``(W, b)`` layers in canonical orientation, then
  ``from_params``. ``onnx`` is imported lazily so this package imports without it.

Weight orientation: ``mlp_forward`` uses the canonical ``(in, out)`` layout (``h @ W + b``).
A torch ``nn.Linear`` weight is ``(out, in)`` -> pass ``weight_layout="out_in"`` (importers
transpose). Flax kernels are already ``(in, out)`` -> ``weight_layout="in_out"`` (default).
ONNX carries the orientation in the ``Gemm.transB`` attribute, so ``from_onnx`` normalizes
to canonical form itself.
"""
from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from breakpoint_mjx.adapters.mlp import MLPPolicy, OutputTransform
from breakpoint_mjx.adapters.obs_spec import ObsSpec, load_contract

# ONNX op_type -> our activation name. Supported HIDDEN activations only.
_ONNX_HIDDEN_ACT = {"Relu": "relu", "Tanh": "tanh", "Elu": "elu"}
# ONNX op_types we CLASSIFY as activations (superset of the hidden set): used to route an
# unrecognized activation into a clear "unknown hidden activation" / "unsupported trailing
# squash" error instead of silently treating it as an unknown op.
_ONNX_ACT_OPS = {"Relu": "relu", "Tanh": "tanh", "Elu": "elu", "Sigmoid": "sigmoid"}
# op_types that are safe to ignore (structural no-ops that don't change the math).
_ONNX_NOOP = {"Identity"}
# Weight/bias key suffixes recognized in a state-dict-like mapping. Torch stores a Linear
# weight as ``*.weight`` (out, in); flax stores a Dense kernel as ``*kernel`` (in, out).
_WEIGHT_SUFFIXES = (("weight", "bias"), ("kernel", "bias"))


def _to_pairs(layers):
    """Coerce various weight containers to an ordered list of ``(W, b)`` pairs.

    Accepts:
      * a mapping (torch ``state_dict`` / flax params / npz dict): keys ending in
        ``weight``/``bias`` (torch) or ``kernel``/``bias`` (flax) are paired by shared
        prefix in insertion order; otherwise the values are treated as a flat
        ``[W0, b0, W1, b1, ...]`` list.
      * a list of ``(W, b)`` pairs.
      * a flat list ``[W0, b0, W1, b1, ...]``.
    ``b`` may be ``None`` (bias-less layer). Note orientation is applied later by
    ``from_params`` (``weight_layout``): torch ``weight`` is ``(out, in)`` -> ``out_in``;
    flax ``kernel`` is already ``(in, out)`` -> ``in_out``.
    """
    if hasattr(layers, "keys"):  # mapping-like (dict / OrderedDict / NpzFile-as-dict)
        items = list(layers.keys())
        for wsuf, bsuf in _WEIGHT_SUFFIXES:
            wkeys = [k for k in items if str(k).endswith(wsuf)]
            if wkeys:
                pairs = []
                for wk in wkeys:
                    prefix = str(wk)[: -len(wsuf)]
                    bk = prefix + bsuf
                    W = layers[wk]
                    b = layers[bk] if bk in layers else None
                    pairs.append((W, b))
                return pairs
        # No weight/bias/kernel naming -> treat the values as a flat array list.
        return _pairs_from_flat([layers[k] for k in items])

    seq = list(layers)
    if not seq:
        raise ValueError("no layers provided")
    first = seq[0]
    # A list of (W, b) pairs: each element is a length-2 sequence whose first entry is 2-D.
    if (
        isinstance(first, (tuple, list))
        and len(first) == 2
        and np.asarray(first[0]).ndim == 2
    ):
        return [(W, b) for (W, b) in seq]
    return _pairs_from_flat(seq)


def _pairs_from_flat(arrays):
    """Pair a flat ``[W0, b0, W1, b1, ...]`` sequence into ``[(W0, b0), ...]``."""
    arrays = list(arrays)
    if len(arrays) % 2 != 0:
        raise ValueError(
            f"flat weight list has odd length {len(arrays)}; expected "
            f"[W0, b0, W1, b1, ...]"
        )
    return [(arrays[i], arrays[i + 1]) for i in range(0, len(arrays), 2)]


def from_params(
    layers,
    *,
    activation="tanh",
    obs_spec=None,
    out_transform=None,
    weight_layout="in_out",
):
    """Build an ``MLPPolicy`` from plain numpy weights (the dependency-free core path).

    Parameters
    ----------
    layers : mapping | list
        Weights (see ``_to_pairs`` for accepted shapes).
    activation : str | callable
        Hidden-layer activation (``"tanh"``/``"relu"``/``"elu"`` or a callable).
    obs_spec : ObsSpec | None
        Observation shim; defaults to reading ``obs["state"]`` unchanged. Its
        ``expected_dim`` is auto-set to the first layer's input dim if unset, so an
        obs/policy mismatch is caught at eval time.
    out_transform : OutputTransform | None
        Optional deterministic action post-processing (tanh squash / scale / bias / mean
        slice).
    weight_layout : {"in_out", "out_in"}
        Orientation of each ``W``. ``"in_out"`` (flax kernel, default) is used as-is;
        ``"out_in"`` (torch ``nn.Linear``) is transposed to canonical form.
    """
    pairs = _to_pairs(layers)
    if weight_layout not in ("in_out", "out_in"):
        raise ValueError(
            f"weight_layout must be 'in_out' or 'out_in', got {weight_layout!r}"
        )
    norm = []
    for W, b in pairs:
        W = np.asarray(W)
        if weight_layout == "out_in":
            W = W.T
        norm.append((W, None if b is None else np.asarray(b)))

    # obs_spec may be None (MLPPolicy defaults it) or user-supplied; MLPPolicy pins its
    # expected_dim to the first-layer input dim so an obs mismatch is caught.
    return MLPPolicy(
        params=norm,
        activation=activation,
        obs_spec=obs_spec,
        out_transform=out_transform,
    )


def from_npz(path, **kwargs):
    """Load an ``.npz`` of named arrays and build an ``MLPPolicy`` via ``from_params``.

    Array-name pairing follows ``_to_pairs`` (``*weight``/``*bias`` by prefix, else a flat
    ``[W0, b0, ...]`` list in file order). Pass ``weight_layout="out_in"`` for a torch dump.
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return from_params(arrays, **kwargs)


def _init_array(inits, name, node):
    """Fetch an initializer by name, or raise naming the node (weight kept as a graph input)."""
    if name not in inits:
        raise ValueError(
            f"ONNX node {node.name or node.op_type!r} ({node.op_type}) expects a weight "
            f"initializer at input {name!r}, but it is not a graph initializer. Weights kept "
            f"as graph inputs are unsupported; export with weights folded as initializers, or "
            f"use the params: path with hand-extracted weights."
        )
    return np.asarray(inits[name])


def from_onnx(
    path,
    *,
    activation=None,
    obs_spec=None,
    out_transform=None,
):
    """Parse an ONNX MLP graph (Gemm / MatMul+Add + activation nodes) into an ``MLPPolicy``.

    Reads (does NOT execute) the graph: initializers give the weights/biases, and the node
    order gives the layer order. ``Gemm`` carries its weight orientation in ``transB`` (torch
    exports ``nn.Linear`` as ``Gemm(transB=1)`` with a ``(out, in)`` weight), normalized to
    canonical ``(in, out)`` here. Recognized hidden activations (``Relu``/``Tanh``/``Elu``)
    between layers set the network activation.

    FAIL-LOUD contract (a silently-dropped node = a wrong policy from information present in
    the graph, so this never guesses):

    * ANY unconsumed op that is not a known no-op raises ``unsupported ONNX op <T>``.
    * An unknown HIDDEN activation raises (it is NOT defaulted to tanh).
    * A trailing activation feeding the graph output is CAPTURED as an ``OutputTransform``
      squash (``Tanh``/``Sigmoid``); an unrecognized trailing activation raises telling you
      to pass ``--squash``.
    * Leading ``Sub``/``Div`` (baked-in obs normalization) is CAPTURED into
      ``obs_spec.mean``/``std`` when unambiguous (``(x - mean) / std``); anything else raises
      telling you to pass ``--obs-norm``.

    An explicit ``activation`` argument overrides detection. ``onnx`` is imported lazily;
    without it this raises a clear ``ImportError`` while the rest of the package imports.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as e:  # pragma: no cover - exercised only where onnx is absent
        raise ImportError(
            "from_onnx requires the 'onnx' package to READ the graph "
            "(`pip install onnx`); it is not executed."
        ) from e

    model = onnx.load(path)
    graph = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    nodes = list(graph.node)
    consumed = [False] * len(nodes)
    output_names = {o.name for o in graph.output}

    layers = []
    hidden_acts = []
    detected_mean = None
    detected_std = None
    trailing_squash = None  # "tanh" | "sigmoid"
    seen_layer = False

    for i, node in enumerate(nodes):
        op = node.op_type
        if op == "Gemm":
            attrs = {a.name: a for a in node.attribute}
            if "transA" in attrs and attrs["transA"].i:
                raise ValueError(
                    f"ONNX Gemm node {node.name or i!r} has transA set, which is "
                    f"unsupported; the input activation must not be transposed"
                )
            trans_b = attrs["transB"].i if "transB" in attrs else 0
            alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
            beta = attrs["beta"].f if "beta" in attrs else 1.0
            W = _init_array(inits, node.input[1], node)  # Gemm(A=x, B=W, C=b)
            b = None
            if len(node.input) > 2 and node.input[2]:
                b = _init_array(inits, node.input[2], node)
            W = W.T if trans_b else W  # -> canonical (in, out)
            if alpha != 1.0:
                W = W * alpha
            if b is not None and beta != 1.0:
                b = b * beta
            layers.append((W, b))
            consumed[i] = True
            seen_layer = True
        elif op == "MatMul":
            wname = next((inp for inp in node.input if inp in inits), None)
            if wname is None:
                raise ValueError(
                    f"ONNX MatMul node {node.name or i!r} has no initializer weight input "
                    f"(a dynamic matmul is unsupported); use the params: path"
                )
            W = _init_array(inits, wname, node)  # MatMul weight is already (in, out)
            consumed[i] = True
            seen_layer = True
            b = None
            out_name = node.output[0]
            for j, n2 in enumerate(nodes):  # a following Add supplies the bias
                if n2.op_type == "Add" and out_name in n2.input:
                    bnames = [x for x in n2.input if x in inits]
                    if bnames:
                        b = _init_array(inits, bnames[0], n2)
                        consumed[j] = True
                    break
            layers.append((W, b))
        elif op in _ONNX_ACT_OPS:
            feeds_output = any(o in output_names for o in node.output)
            if feeds_output:
                # Trailing activation on the action head -> capture as an output squash.
                if op == "Tanh":
                    trailing_squash = "tanh"
                elif op == "Sigmoid":
                    trailing_squash = "sigmoid"
                else:
                    raise ValueError(
                        f"ONNX trailing activation {op!r} feeds the graph output but is not "
                        f"a recognized squash (Tanh/Sigmoid). Pass --squash/--action-scale to "
                        f"model your policy's output post-processing, or preprocess the graph."
                    )
                consumed[i] = True
            else:
                if op not in _ONNX_HIDDEN_ACT:
                    raise ValueError(
                        f"unknown hidden activation {op!r}; supported: Relu, Tanh, Elu. "
                        f"Import via the params: path with hand-extracted weights, or "
                        f"preprocess the graph."
                    )
                hidden_acts.append(_ONNX_HIDDEN_ACT[op])
                consumed[i] = True
        elif op in ("Sub", "Div") and not seen_layer:
            # Leading (x - mean) / std baked normalization. Only the canonical operand order
            # (constant as the SECOND input) is unambiguous; anything else must be explicit.
            const_name = node.input[1] if node.input[1] in inits else None
            if const_name is None or node.input[0] in inits:
                raise ValueError(
                    f"leading {op} node {node.name or i!r} is not an unambiguous "
                    f"(x - mean) / std normalization (constant must be the 2nd operand); "
                    f"pass --obs-norm with the mean/std instead."
                )
            val = np.asarray(inits[const_name]).reshape(-1).astype(float)
            if op == "Sub":
                if detected_mean is not None:
                    raise ValueError(
                        "multiple leading Sub nodes; ambiguous obs normalization -- "
                        "pass --obs-norm."
                    )
                detected_mean = val
            else:  # Div
                if detected_std is not None:
                    raise ValueError(
                        "multiple leading Div nodes; ambiguous obs normalization -- "
                        "pass --obs-norm."
                    )
                detected_std = val
            consumed[i] = True

    # FAIL LOUD on anything we did not explicitly account for -- a dropped node is a wrong
    # policy built from information that was right there in the graph.
    for i, node in enumerate(nodes):
        if consumed[i] or node.op_type in _ONNX_NOOP:
            continue
        raise ValueError(
            f"unsupported ONNX op {node.op_type!r}; import via the params: path with "
            f"hand-extracted weights, or preprocess the graph."
        )

    if not layers:
        raise ValueError(
            f"no Gemm/MatMul layers found in ONNX graph at {path!r}; only MLP "
            f"(fully-connected) policies are supported"
        )

    # Resolve the hidden activation without guessing.
    if activation is not None:
        act = activation
    elif hidden_acts:
        uniq = sorted(set(hidden_acts))
        if len(uniq) > 1:
            raise ValueError(
                f"ONNX graph uses multiple distinct hidden activations {uniq}; per-layer "
                f"activations are unsupported -- use the params: path or pass activation="
            )
        act = hidden_acts[0]
    else:
        # No activation nodes at all -> the graph is a linear stack (correct, not a guess).
        act = "identity"

    # Merge captured obs normalization into obs_spec (conflict if the caller also passed it).
    if detected_mean is not None or detected_std is not None:
        caller_norm = obs_spec is not None and (
            obs_spec.mean is not None or obs_spec.std is not None
        )
        if caller_norm:
            raise ValueError(
                "ONNX graph has baked-in obs normalization (leading Sub/Div) AND obs "
                "normalization was passed explicitly (--obs-norm); provide only one."
            )
        base = obs_spec if obs_spec is not None else ObsSpec()
        obs_spec = dataclasses.replace(base, mean=detected_mean, std=detected_std)

    # Merge a captured trailing squash into out_transform (conflict if the caller set one).
    if trailing_squash is not None:
        if out_transform is None:
            out_transform = OutputTransform(squash=trailing_squash)
        elif out_transform.squash:
            raise ValueError(
                "ONNX graph has a trailing squash activation AND a squash was passed "
                "(--squash / out_transform); provide only one."
            )
        else:
            out_transform = dataclasses.replace(out_transform, squash=trailing_squash)

    return from_params(
        layers,
        activation=act,
        obs_spec=obs_spec,
        out_transform=out_transform,
        weight_layout="in_out",  # already canonicalized above
    )


def build_policy(
    spec,
    *,
    obs_keys=("state",),
    obs_norm_path=None,
    activation=None,
    action_scale=None,
    action_bias=None,
    squash=False,
    mean_dim=None,
    weight_layout="out_in",
    expected_dim=None,
    obs_spec_path=None,
):
    """Build an ``MLPPolicy`` from a ``kind:path`` spec (the pipeline entry point).

    ``spec`` is ``"onnx:/path/model.onnx"`` or ``"params:/path/weights.npz"``. An optional
    ``.npz`` at ``obs_norm_path`` supplies ``mean``/``std`` for observation normalization.
    ``action_scale``/``action_bias``/``squash``/``mean_dim`` configure the output transform.

    Defaults: ``weight_layout="out_in"`` (torch, the common BYO case) for the params path;
    ONNX ignores it (orientation comes from the graph). ``activation=None`` lets ONNX
    auto-detect and falls back to ``"tanh"`` for the params path.

    A bare path is accepted too; its extension picks the kind (``.onnx``, ``.npz``, and
    ``.pkl`` for a brax/Playground checkpoint). A prefix that is not a known kind is part
    of the path, so ``C:/runs/policy.pkl`` is a Windows path, not a policy of kind ``c``.

    ``obs_spec_path`` (or ``BP_OBS_SPEC``) names a contract YAML (``breakpoint init`` /
    ``check-policy --infer-contract`` write one): it sets the obs indices / per-term scales
    / offsets and the action joint permutation and scale. See ``obs_spec.build_contract``.
    """
    kind, path = parse_policy_spec(spec)
    if obs_spec_path is None:
        obs_spec_path = os.environ.get("BP_OBS_SPEC") or None
    contract = load_contract(obs_spec_path) if obs_spec_path else None

    if kind == "brax":
        if obs_norm_path or action_scale is not None or action_bias is not None or squash \
                or mean_dim is not None:
            raise ValueError(
                "a brax checkpoint carries its own obs normalization and tanh head; "
                "--obs-norm/--action-scale/--squash/--mean-dim do not apply (put obs "
                "remaps and action scale in the --obs-spec contract instead)")
        policy = from_brax_pkl(path)
        if contract is not None:
            policy = apply_contract(policy, contract)
        return policy

    mean = std = None
    if obs_norm_path:
        with np.load(obs_norm_path, allow_pickle=False) as nz:
            if "mean" in nz.files:
                mean = nz["mean"]
            if "std" in nz.files:
                std = nz["std"]
    obs_spec = ObsSpec(keys=obs_keys, mean=mean, std=std, expected_dim=expected_dim)

    out_transform = None
    if squash or action_scale is not None or action_bias is not None or mean_dim is not None:
        out_transform = OutputTransform(
            squash=bool(squash),
            scale=1.0 if action_scale is None else action_scale,
            bias=0.0 if action_bias is None else action_bias,
            mean_dim=mean_dim,
        )

    if kind == "onnx":
        policy = from_onnx(
            path, activation=activation, obs_spec=obs_spec, out_transform=out_transform
        )
    elif kind in ("params", "npz", "weights"):
        with np.load(path, allow_pickle=False) as nz:
            recurrent = any(k.startswith("memory_a.rnn.") for k in nz.files)
            sd = {k: nz[k] for k in nz.files} if recurrent else None
        if recurrent:
            # an RSL-RL ActorCriticRecurrent state_dict dumped to npz
            from breakpoint_mjx.adapters.stateful import recurrent_from_state_dict

            policy = recurrent_from_state_dict(
                sd, obs_spec=obs_spec, activation=activation or "elu",
                out_transform=out_transform)
        else:
            policy = from_npz(
                path,
                activation=activation or "tanh",
                obs_spec=obs_spec,
                out_transform=out_transform,
                weight_layout=weight_layout,
            )
    else:  # pragma: no cover - parse_policy_spec only returns known kinds
        raise ValueError(f"unknown policy kind {kind!r}; use 'onnx:' or 'params:'")

    if contract is not None:
        policy = apply_contract(policy, contract)
    _warn_if_untransformed(policy)
    return policy


POLICY_KINDS = ("onnx", "params", "npz", "weights", "brax")
_EXT_KIND = {".onnx": "onnx", ".npz": "params", ".pkl": "brax", ".pickle": "brax"}


def parse_policy_spec(spec):
    """Split ``kind:path`` -> ``(kind, path)``; infer the kind of a bare path.

    Only a prefix in ``POLICY_KINDS`` is a kind. Anything else (``C:`` on Windows, a URL
    scheme, no colon at all) is part of the path, and the kind comes from the extension.
    """
    s = str(spec)
    head, sep, tail = s.partition(":")
    if sep and head.lower() in POLICY_KINDS:
        return head.lower(), tail
    ext = Path(s).suffix.lower()
    if ext in _EXT_KIND:
        if not Path(s).exists():
            raise ValueError(
                f"no policy file at {s!r} (a policy spec is onnx:<path>, params:<path>, "
                f"brax:<path> or a path ending in .onnx/.npz/.pkl)")
        return _EXT_KIND[ext], s
    raise ValueError(
        f"unknown policy kind in {s!r}: use onnx:<path>, params:<path> or brax:<path>, or "
        f"a path ending in .onnx, .npz or .pkl")


def from_brax_params(params, *, obs_key="state", activation="silu"):
    """Weight-import a brax PPO checkpoint ``(normalizer, policy_params, value_params)``.

    brax's policy is an MLP (flax ``hidden_i`` kernels in ``(in, out)`` layout, SiLU by
    default) on running-statistics-normalized ``obs[obs_key]``, emitting ``[loc, scale]``;
    the deterministic action is ``tanh(loc)``. That maps exactly onto ``MLPPolicy`` with
    ``ObsSpec(mean, std)`` and ``OutputTransform(squash=tanh, mean_dim=A)``, so a
    Playground/brax checkpoint gets the same contract machinery as an ONNX export.
    """
    norm, pol = params[0], params[1]
    mean, std = norm.mean, norm.std
    if isinstance(mean, dict):
        mean, std = mean[obs_key], std[obs_key]
    p = pol["params"] if "params" in pol else pol
    names = sorted((k for k in p if str(k).startswith("hidden_")),
                   key=lambda k: int(str(k).split("_")[1]))
    if not names:
        raise ValueError("brax params have no hidden_<i> layers; not a brax MLP policy")
    layers = [(np.asarray(p[k]["kernel"]), np.asarray(p[k]["bias"])) for k in names]
    out = layers[-1][0].shape[1]
    return from_params(
        layers, activation=activation,
        obs_spec=ObsSpec(keys=(obs_key,), mean=np.asarray(mean, dtype=float),
                         std=np.asarray(std, dtype=float)),
        out_transform=OutputTransform(squash="tanh", mean_dim=out // 2),
        weight_layout="in_out")


def from_brax_pkl(path, **kwargs):
    """``from_brax_params`` on a pickled brax checkpoint (BreakPoint's ``*_policy.pkl``)."""
    import pickle

    try:
        params = pickle.loads(Path(path).read_bytes())
    except ModuleNotFoundError as e:
        raise ImportError(
            f"unpickling a brax checkpoint needs brax installed ({e}); "
            f"pip install breakpoint-mjx[sim]") from e
    return from_brax_params(params, **kwargs)


def apply_contract(policy, contract):
    """Return a copy of ``policy`` driven through ``contract`` (a ``PolicyContract``).

    The policy's own normalization (brax running stats, ONNX-baked Sub/Div, --obs-norm)
    is kept and applied after the contract's per-term affine; the contract's action map
    (joint permutation, per-joint scale/offset) composes after the policy's squash.
    """
    from breakpoint_mjx.adapters.stateful import (HistoryPolicy, RecurrentPolicy,
                                                  stack_history)

    spec = contract.to_obs_spec()
    if isinstance(policy, RecurrentPolicy):
        old = policy.obs_spec
        frame = dataclasses.replace(
            spec, mean=old.mean if old.mean is not None else spec.mean,
            std=old.std if old.std is not None else spec.std)
        head = policy.mlp
        head = MLPPolicy(params=head.params, activation=head.activation,
                         obs_spec=head.obs_spec,
                         out_transform=_compose_output(head.out_transform, contract))
        out = RecurrentPolicy(kind=policy.kind, layers=policy.layers, mlp=head,
                              obs_spec=frame, contract=contract)
        return out
    old = policy.obs_spec
    mean = old.mean if old is not None and old.mean is not None else spec.mean
    std = old.std if old is not None and old.std is not None else spec.std
    ot = _compose_output(policy.out_transform, contract)
    if contract.history:
        h = contract.history
        d = spec.expected_dim
        if mean is not None and mean.shape[-1] == d != policy.in_dim:
            # a per-frame normalization: lay it out exactly like the stacked input
            mean = stack_history(np.tile(mean, (h["length"], 1)), h["layout"], h["order"],
                                 contract.term_dims, np)
            std = stack_history(np.tile(std, (h["length"], 1)), h["layout"], h["order"],
                                contract.term_dims, np)
        head = MLPPolicy(params=policy.params, activation=policy.activation,
                         obs_spec=ObsSpec(keys=("_history",), mean=mean, std=std,
                                          expected_dim=policy.in_dim),
                         out_transform=ot)
        return HistoryPolicy(frame_spec=dataclasses.replace(spec, mean=None, std=None),
                             mlp=head, length=h["length"], layout=h["layout"],
                             fill=h["fill"], order=h["order"],
                             term_dims=contract.term_dims, contract=contract)
    if spec.expected_dim is not None and spec.expected_dim != policy.in_dim:
        raise ValueError(
            f"the contract lists {spec.expected_dim} obs channels but the policy's first "
            f"layer takes {policy.in_dim}; the contract's term list does not match this "
            f"checkpoint (a policy trained on stacked observations needs a history: "
            f"block in the contract)")
    new_spec = dataclasses.replace(spec, mean=mean, std=std)
    out = MLPPolicy(params=policy.params, activation=policy.activation,
                    obs_spec=new_spec, out_transform=ot)
    out.contract = contract
    return out


def _compose_output(base, contract):
    """The contract's action map (joint permutation, per-joint scale/offset) composed
    after the policy's own output transform (squash, scale, bias)."""
    ot = contract.to_output_transform()
    if ot is None:
        return base
    base = base or OutputTransform()
    a = contract.action_dim
    s0 = np.broadcast_to(np.asarray(base.scale, dtype=float), (a,))
    b0 = np.broadcast_to(np.asarray(base.bias, dtype=float), (a,))
    perm = np.asarray(ot.perm)
    return OutputTransform(squash=base.squash, mean_dim=base.mean_dim, perm=ot.perm,
                           scale=s0[perm] * ot.scale, bias=b0[perm] * ot.scale + ot.bias)


def _warn_if_untransformed(policy):
    """Warn loudly if a BYO policy has NEITHER obs normalization NOR any action transform.

    Most exported policies were trained on normalized observations and/or squash+scale their
    actions; with neither configured, BreakPoint drives the policy on RAW obs and emits its
    RAW head. Only obs/action DIMENSION mismatches are auto-caught, so a wrong normalization
    or missing action post-processing is silent -- this is the one place we flag it.
    """
    # a stateful wrapper (adapters/stateful.py) keeps its head network in ``.mlp``; the
    # normalization can sit on either the frame spec or the head's stacked spec
    head = getattr(policy, "mlp", policy)
    specs = [sp for sp in (policy.obs_spec, head.obs_spec) if sp is not None]
    has_norm = any(sp.mean is not None or sp.std is not None for sp in specs)
    ot = head.out_transform
    has_action_tf = ot is not None and (
        bool(ot.squash)
        or bool(np.any(np.asarray(ot.scale) != 1.0))
        or bool(np.any(np.asarray(ot.bias) != 0.0))
        or ot.mean_dim is not None
    )
    if not has_norm and not has_action_tf:
        warnings.warn(
            "BYO policy: no obs normalization or action transform set; verify your export "
            "was trained on RAW observations and needs no action post-processing. Only "
            "obs/action DIMENSION mismatches are auto-caught -- a wrong normalization or "
            "missing squash/scale will silently produce wrong actions.",
            stacklevel=2,
        )


# --------------------------------------------------------------------------- #
# Framework config importers: Isaac Lab / RSL-RL / unitree_rl_gym -> contract YAML
# --------------------------------------------------------------------------- #
# Each importer reads the config a framework already writes next to a trained policy and
# emits (a) a contract dict for ``obs_spec.build_contract`` and (b) a table of every field
# BreakPoint cannot honor. Nothing is dropped silently: a field that changes the policy's
# behaviour and has no BreakPoint equivalent is a row in that table, at level "warn"
# (the run is still meaningful, with a stated caveat) or "fatal" (the policy needs inputs
# the target cannot produce; no contract is written).

# Isaac Lab articulation order for Unitree quadrupeds: PhysX parses the USD breadth-first,
# which yields JOINT-TYPE-major order (all hips, then all thighs, then all calves) with
# legs FL, FR, RL, RR. An importer that has to assume it says so in the warning table,
# and ``check-policy --infer-contract`` can verify it.
ISAACLAB_QUADRUPED_ORDER = tuple(f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf")
                                 for leg in ("FL", "FR", "RL", "RR"))
# legged_gym / unitree_rl_gym (Isaac Gym) DOF order for Unitree quadrupeds: leg-major,
# legs FL, FR, RL, RR.
LEGGED_GYM_QUADRUPED_ORDER = tuple(f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR")
                                   for j in ("hip", "thigh", "calf"))


@dataclass
class ImportResult:
    """What an importer produced: the contract dict, the caveat table and provenance."""

    contract: dict
    rows: list = field(default_factory=list)   # [{"level", "field", "detail"}]
    source: str = ""
    hints: dict = field(default_factory=dict)  # e.g. activation / hidden dims

    @property
    def fatal(self):
        return [r for r in self.rows if r["level"] == "fatal"]

    def add(self, level, fld, detail):
        self.rows.append({"level": level, "field": fld, "detail": detail})

    def table(self):
        """The warning table as text (one row per unhonored field)."""
        if not self.rows:
            return "  (every field maps onto the target; nothing to report)"
        w = max(len(r["field"]) for r in self.rows)
        return "\n".join(f"  {r['level'].upper():<5}  {r['field'].ljust(w)}  {r['detail']}"
                         for r in self.rows)

    def to_yaml(self, path=None):
        import yaml

        text = yaml.safe_dump(self.contract, sort_keys=False, default_flow_style=None)
        head = f"# generated by breakpoint init from {self.source}\n"
        head += "".join(f"# {r['level'].upper()}: {r['field']}: {r['detail']}\n"
                        for r in self.rows)
        text = head + text
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _permissive_yaml(text):
    """``yaml.safe_load`` that turns python-object tags (Isaac Lab's dumps carry
    ``!!python/tuple`` and ``!!python/object/apply:...``) into plain data instead of
    refusing. It never constructs a python object or executes anything."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _any(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _any)
    _Loader.add_multi_constructor("!", _any)
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - SafeLoader subclass


def _load_cfg(src):
    if isinstance(src, dict):
        return src, "<dict>"
    p = Path(src)
    if p.suffix.lower() in (".py", ".pkl", ".pickle"):
        raise ValueError(
            f"{p.name}: only YAML dumps are read (Isaac Lab writes params/env.yaml next to "
            f"every run); python/pickle configs are never executed or unpickled here")
    d = _permissive_yaml(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{p}: expected a YAML mapping")
    return d, str(p)


def _func_term(func):
    """Isaac Lab obs term func ``isaaclab.envs.mdp.observations:base_lin_vel`` -> name."""
    s = str(func or "")
    return s.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def _resolve_joint_regex(mapping, names, what):
    """Isaac Lab ``{regex: value}`` dicts (init_state.joint_pos, stiffness) -> per joint."""
    import re

    if isinstance(mapping, (int, float)):
        return [float(mapping)] * len(names)
    out = []
    for n in names:
        v = None
        for pat, val in mapping.items():
            if re.fullmatch(str(pat), n):
                v = float(val)
                break
        if v is None:
            raise ValueError(f"{what}: no pattern in {list(mapping)} matches joint {n!r}")
        out.append(v)
    return out


def _gain_rows(res, iface, kps, kds, src, action_scale=None):
    """Write the contract's ``actuation:`` block from the training gains, and a row saying
    what that does. ``BP_HONOR_GAINS=0`` writes ``honor: false`` (then a >20% mismatch
    is FATAL unless ``BP_ALLOW_GAIN_MISMATCH=1``)."""
    from breakpoint_mjx.adapters.actuation import actuation_block, gain_gate_row

    if kps is None and kds is None:
        return
    honor = os.environ.get("BP_HONOR_GAINS", "1").strip().lower() not in (
        "0", "false", "no", "off")
    allow = os.environ.get("BP_ALLOW_GAIN_MISMATCH", "").strip().lower() in (
        "1", "true", "yes", "on")
    block = actuation_block(iface.kp if kps is None else kps,
                            None if kds is None else kds,
                            action_scale=action_scale, source=src, honor=honor)
    res.contract["actuation"] = block
    row = gain_gate_row(block, iface.kp, iface.kd, allow=allow)
    if row is None:
        return
    level, detail = row
    fld = f"{src}.stiffness" if kps is not None else f"{src}.damping"
    res.add(level, fld, detail)


# --------------------------------------------------------------------------- #
# Training domain randomization -> the operational ODD (riskcert schema)
# --------------------------------------------------------------------------- #
# Physical quantity -> the ThetaSpace dim that models it on the legged targets. The dim
# must also exist in the robot's theta_space.yaml (checked at import time), otherwise the
# range becomes a WARN row instead of a silent drop.
ODD_DIMS = {"friction": "friction_floor", "added_mass": "torso_mass_add_kg",
            "com_x": "com_offset_x_m", "com_y": "com_offset_y_m", "com_z": "com_offset_z_m",
            "push_vel": "push_mag"}
_ODD_NOTES = {
    "friction": "training randomized the robot's contact friction; mapped 1:1 onto the "
                "floor friction (MuJoCo combines a contact pair by max, the trainer by its "
                "own rule)",
    "push_vel": "training pushed at an interval; BreakPoint applies ONE gait-timed push of "
                "this magnitude",
}


def _theta_dims(robot):
    """{dim name: (lo, hi) search bounds} of the robot's ThetaSpace, or None."""
    try:
        from breakpoint_mjx.config import load_theta_space
        from breakpoint_mjx.paths import config_path
        from breakpoint_mjx.targets import get_target

        space = load_theta_space(config_path(get_target(robot).theta_space_path))
        return {n: tuple(space.dims[n].bounds) for n in space.names()}
    except Exception:  # noqa: BLE001 - an ODD is optional; say so instead of failing init
        return None


def _odd_add(res, robot, qty, lo, hi, fld):
    """Map one training range onto the operational block (or a WARN row)."""
    if "_theta_dims" not in res.__dict__:
        res.__dict__["_theta_dims"] = _theta_dims(robot)
    dims = res.__dict__["_theta_dims"]
    dim = ODD_DIMS.get(qty)
    lo, hi = float(lo), float(hi)
    if dims is None or dim not in dims:
        res.add("warn", fld, f"training range [{lo:g}, {hi:g}] has no ThetaSpace dim on "
                             f"{robot} ({dim or qty}); not part of the operational ODD")
        return
    if hi <= lo:
        res.add("warn", fld, f"held fixed at {lo:g} during training (not randomized); a "
                             f"point is not a range, so the operational ODD keeps the "
                             f"{dim} plausible band")
        return
    blo, bhi = dims[dim]
    op = res.contract.setdefault("operational", {})
    op[dim] = {"range": [lo, hi]}
    if lo < blo or hi > bhi:
        res.add("warn", fld, f"[{lo:g}, {hi:g}] -> {dim}; riskcert clips it to the search "
                             f"bounds [{blo:g}, {bhi:g}]")
    note = _ODD_NOTES.get(qty, "")
    res.add("info", fld, f"[{lo:g}, {hi:g}] -> operational.{dim}" + (f": {note}" if note
                                                                        else ""))


def _pair(v):
    a = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v, v])]
    return (min(a), max(a)) if len(a) >= 2 else (a[0], a[0])


# legged_gym / unitree_rl_gym ``LeggedRobotCfg.domain_rand`` keys that are mapped or are
# bookkeeping for a mapped key (push_interval_s times the pushes).
_LG_MAPPED = {"randomize_friction", "friction_range", "randomize_base_mass",
              "added_mass_range", "push_robots", "max_push_vel_xy", "push_interval_s"}


def _legged_gym_odd(res, robot, dr, src="domain_rand"):
    """legged_gym ``domain_rand`` (dict) -> operational block + rows."""
    if dr.get("friction_range") is not None:
        if dr.get("randomize_friction", True):
            lo, hi = _pair(dr["friction_range"])
            _odd_add(res, robot, "friction", lo, hi, f"{src}.friction_range")
        else:
            res.add("warn", f"{src}.randomize_friction",
                    "false: friction was not randomized in training")
    if dr.get("added_mass_range") is not None:
        if dr.get("randomize_base_mass", False):
            lo, hi = _pair(dr["added_mass_range"])
            _odd_add(res, robot, "added_mass", lo, hi, f"{src}.added_mass_range")
        else:
            res.add("warn", f"{src}.randomize_base_mass",
                    f"false: the policy trained at nominal base mass (added_mass_range "
                    f"{dr['added_mass_range']} was unused); the operational ODD keeps the "
                    f"plausible band for torso_mass_add_kg")
    if dr.get("push_robots") and dr.get("max_push_vel_xy") is not None:
        _odd_add(res, robot, "push_vel", 0.0, float(dr["max_push_vel_xy"]),
                 f"{src}.max_push_vel_xy")
    for k, v in dr.items():
        if k not in _LG_MAPPED and v not in (None, False, 0, 0.0, [], {}):
            res.add("warn", f"{src}.{k}", f"{v}: no BreakPoint equivalent; not part of the "
                                          f"operational ODD")


def _isaaclab_events_odd(res, robot, events):
    """Isaac Lab ``events`` (EventCfg dump) -> operational block + rows."""
    for key, term in (events or {}).items():
        if not isinstance(term, dict):
            continue
        fn = _func_term(term.get("func"))
        p = term.get("params") or {}
        fld = f"events.{key}"
        if fn == "randomize_rigid_body_material":
            lo, hi = _pair(p.get("static_friction_range", [1.0, 1.0]))
            _odd_add(res, robot, "friction", lo, hi, f"{fld}.static_friction_range")
        elif fn == "randomize_rigid_body_mass":
            body = str(((p.get("asset_cfg") or {}).get("body_names")) or "")
            if str(p.get("operation", "add")) != "add":
                res.add("warn", fld, f"operation {p.get('operation')!r}: only an additive "
                                     f"base-mass range maps onto torso_mass_add_kg")
            elif body and not any(b in body for b in ("base", "trunk", "torso")):
                res.add("warn", fld, f"mass randomized on {body!r}, not the base; unmapped")
            else:
                lo, hi = _pair(p.get("mass_distribution_params", [0.0, 0.0]))
                _odd_add(res, robot, "added_mass", lo, hi,
                         f"{fld}.mass_distribution_params")
        elif fn == "randomize_rigid_body_com":
            for ax, r in (p.get("com_range") or {}).items():
                lo, hi = _pair(r)
                _odd_add(res, robot, f"com_{ax}", lo, hi, f"{fld}.com_range.{ax}")
        elif fn == "push_by_setting_velocity":
            vr = p.get("velocity_range") or {}
            vx = max(abs(x) for x in _pair(vr.get("x", [0.0, 0.0])))
            vy = max(abs(x) for x in _pair(vr.get("y", [0.0, 0.0])))
            _odd_add(res, robot, "push_vel", 0.0, round(float(np.hypot(vx, vy)), 4),
                     f"{fld}.velocity_range")
        else:
            res.add("warn", fld, f"event {fn or key!r} has no ThetaSpace equivalent; not "
                                 f"part of the operational ODD")


def _literal_blocks(py_paths, blocks=("domain_rand", "control")):
    """Literal class attributes of legged_gym config classes, WITHOUT executing them.

    legged_gym configs are python classes (``class domain_rand: friction_range = [...]``)
    that inherit field by field. The files are parsed with ``ast`` and only literal
    assignments are read (``ast.literal_eval``), base file first so a robot's overrides
    win. Nothing is imported or run.
    """
    import ast

    out = {b: {} for b in blocks}
    for p in py_paths:
        tree = ast.parse(Path(p).read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in out:
                for st in node.body:
                    if (isinstance(st, ast.Assign) and len(st.targets) == 1
                            and isinstance(st.targets[0], ast.Name)):
                        try:
                            out[node.name][st.targets[0].id] = ast.literal_eval(st.value)
                        except (ValueError, TypeError, SyntaxError, RecursionError):
                            continue
    return out


def _find_legged_gym_cfgs(deploy_yaml):
    """[base cfg, robot cfg] python files of the unitree_rl_gym checkout a deploy YAML lives
    in (``deploy/deploy_mujoco/configs/<robot>.yaml`` -> ``legged_gym/envs/base/
    legged_robot_config.py`` + ``legged_gym/envs/<robot>/<robot>_config.py``), or
    ``BP_TRAIN_CFG`` (``os.pathsep``-separated) when set."""
    env = os.environ.get("BP_TRAIN_CFG")
    if env:
        return [Path(x) for x in env.split(os.pathsep) if x]
    if not isinstance(deploy_yaml, (str, Path)):
        return []
    p = Path(deploy_yaml).resolve()
    for root in p.parents:
        base = root / "legged_gym" / "envs" / "base" / "legged_robot_config.py"
        if base.exists():
            robot_cfg = root / "legged_gym" / "envs" / p.stem / f"{p.stem}_config.py"
            return [base] + ([robot_cfg] if robot_cfg.exists() else [])
    return []


def _joint_dict_values(mapping, names):
    """legged_gym ``{'joint': 20.}`` substring-keyed gains -> per joint, or None."""
    out = []
    for n in names:
        v = next((float(val) for k, val in mapping.items() if str(k) in n), None)
        if v is None:
            return None
        out.append(v)
    return out


def from_isaaclab_cfg(env_cfg, agent_cfg=None, *, robot="go1", joint_names=None):
    """Isaac Lab env config dump (``params/env.yaml``, or its dict) -> ``ImportResult``.

    Reads ``observations.policy`` terms in order (func -> term name, ``scale``, ``clip``),
    ``actions.joint_pos`` (``scale``, ``offset``, ``use_default_offset``, ``joint_names``,
    ``preserve_order``), ``scene.robot.init_state.joint_pos`` (the default pose,
    regex-keyed), ``scene.robot.actuators`` (stiffness/damping), ``decimation`` and
    ``sim.dt``. ``agent_cfg`` (``params/agent.yaml``) adds network hints and the
    empirical-normalization caveat.
    """
    from breakpoint_mjx.adapters.obs_spec import canonical_term, target_interface

    cfg, src = _load_cfg(env_cfg)
    iface = target_interface(robot)
    res = ImportResult(contract={"version": 1, "robot": iface.robot}, source=src)

    # ---- joint order ----
    actions = cfg.get("actions") or {}
    act_cfg = actions.get("joint_pos") or next(
        (v for v in actions.values() if isinstance(v, dict)), None) or {}
    names = list(joint_names) if joint_names else None
    cfg_names = act_cfg.get("joint_names")
    if names is None and cfg_names and act_cfg.get("preserve_order") and \
            all("*" not in str(n) and "[" not in str(n) for n in cfg_names):
        names = [str(n) for n in cfg_names]
    if names is None:
        names = list(ISAACLAB_QUADRUPED_ORDER)
        res.add("warn", "joint order",
                "the config does not list resolved joint names, so the Isaac Lab "
                "articulation order (joint-type-major: FL,FR,RL,RR hips, then thighs, then "
                "calves) is ASSUMED; confirm with `breakpoint check-policy --infer-contract`")
    res.contract["joint_names"] = names

    # ---- default pose ----
    robot_cfg = (cfg.get("scene") or {}).get("robot") or {}
    jp = (robot_cfg.get("init_state") or {}).get("joint_pos")
    if jp:
        try:
            res.contract["default_joint_pos"] = _resolve_joint_regex(
                jp, names, "init_state.joint_pos")
        except ValueError as e:
            res.add("warn", "init_state.joint_pos", f"{e}; using the target's default pose")
    if act_cfg and act_cfg.get("use_default_offset") is False:
        res.add("warn", "actions.use_default_offset",
                "false: the policy's action is an ABSOLUTE joint target; BreakPoint "
                "offsets by the default pose, so set default_joint_pos to zeros by hand")

    # ---- action ----
    if act_cfg:
        if act_cfg.get("scale") is not None:
            sc = act_cfg["scale"]
            if isinstance(sc, dict):
                sc = _resolve_joint_regex(sc, names, "actions.scale")
            res.contract["action"] = {"scale": sc}
        off = act_cfg.get("offset")
        if off not in (None, 0, 0.0):
            res.add("warn", "actions.offset",
                    f"{off}: not applied on top of the default pose; fold it into "
                    f"default_joint_pos")
        if act_cfg.get("clip") is not None:
            res.add("warn", "actions.clip", f"{act_cfg['clip']}: BreakPoint does not clip")

    # ---- observations ----
    pol = (cfg.get("observations") or {}).get("policy") or {}
    reserved = {"concatenate_terms", "enable_corruption", "history_length",
                "flatten_history_dim", "concatenate_dim"}
    # Observation history (adapters/stateful.py). Isaac Lab keeps H frames per term and,
    # with flatten_history_dim (the default), flattens each term's history before
    # concatenating terms: the contract's "term" layout. A per-term history_length
    # overrides the group's; BreakPoint needs one length for every term.
    lengths = {int(t.get("history_length") or pol.get("history_length") or 0)
               for k, t in pol.items() if k not in reserved and isinstance(t, dict)}
    lengths.discard(0)
    if len(lengths) > 1:
        res.add("fatal", "observations.policy.history_length",
                f"terms use different history lengths {sorted(lengths)}; BreakPoint "
                f"stacks one length for every term")
    elif lengths and max(lengths) > 1:
        if pol.get("flatten_history_dim") is False:
            res.add("fatal", "observations.policy.flatten_history_dim",
                    "false: the policy takes a (history, dim) tensor per term, not a flat "
                    "vector; only flattened history is supported")
        else:
            res.contract["history"] = {"length": max(lengths), "layout": "term",
                                       "fill": "first"}
            res.add("info", "observations.policy.history_length",
                    f"{max(lengths)}: stacked per term (Isaac Lab flatten_history_dim), "
                    f"history filled with the first observation at reset")
    if pol.get("enable_corruption"):
        res.add("warn", "observations.policy.enable_corruption",
                "training-time obs noise is not reproduced; BreakPoint applies the "
                "target's own sensor-noise model")
    obs = []
    for key, term in pol.items():
        if key in reserved or not isinstance(term, dict):
            continue
        fname = _func_term(term.get("func")) or key
        canon = canonical_term(fname) or canonical_term(key)
        if canon is None:
            res.add("fatal", f"observations.policy.{key}",
                    f"term {fname!r} has no BreakPoint equivalent on {iface.robot} (a "
                    f"height scan, for one, needs terrain sensing); the policy cannot be "
                    f"driven")
            continue
        entry = {"term": canon}
        if term.get("scale") is not None:
            entry["scale"] = term["scale"]
        if term.get("clip") is not None:
            res.add("warn", f"observations.policy.{key}.clip",
                    f"{term['clip']}: obs clipping is not applied")
        if canon == "commands":
            p = term.get("params") or {}
            if p.get("command_name") not in (None, "base_velocity"):
                res.add("warn", f"observations.policy.{key}",
                        f"command {p.get('command_name')!r}: BreakPoint feeds the target's "
                        f"base-velocity command (vx, vy, yaw rate)")
        obs.append(entry)
    res.contract["obs"] = obs

    # ---- timing + gains ----
    dec, dt = cfg.get("decimation"), (cfg.get("sim") or {}).get("dt")
    if dec and dt:
        cdt = float(dec) * float(dt)
        if abs(cdt - iface.control_dt) > 1e-6:
            res.add("warn", "decimation*sim.dt",
                    f"policy control period {cdt:.4f}s vs the target's "
                    f"{iface.control_dt:.4f}s; the policy runs at the target's rate")
    kps = kds = None
    for aname, a in (robot_cfg.get("actuators") or {}).items():
        if not isinstance(a, dict):
            continue
        cls = str(a.get("class_type") or "")
        if "ActuatorNet" in cls:
            res.add("warn", f"actuators.{aname}",
                    f"{cls.rsplit(':', 1)[-1]}: the policy trained against a LEARNED "
                    f"actuator network, which has no PD gains to honor; the target's PD "
                    f"actuators are used")
        for lim in ("effort_limit", "saturation_effort", "velocity_limit"):
            if a.get(lim) is not None:
                res.add("warn", f"actuators.{aname}.{lim}",
                        f"{a[lim]}: not reproduced; the target's actuator limits apply")
        for nm in ("stiffness", "damping"):
            v = a.get(nm)
            if v is None:
                continue
            vals = (_resolve_joint_regex(v, names, f"actuators.{nm}")
                    if isinstance(v, dict) else [float(v)] * len(names))
            if nm == "stiffness":
                kps = vals
            else:
                kds = vals
    sc = (res.contract.get("action") or {}).get("scale")
    _gain_rows(res, iface, kps, kds, "actuators", action_scale=sc)

    # ---- training domain randomization -> operational ODD ----
    _isaaclab_events_odd(res, iface.robot, cfg.get("events"))

    if agent_cfg is not None:
        _agent_rows(res, agent_cfg)
    return res


def _agent_rows(res, agent_cfg):
    agent, _ = _load_cfg(agent_cfg)
    pol = agent.get("policy") or {}
    if pol.get("actor_hidden_dims"):
        res.hints["actor_hidden_dims"] = list(pol["actor_hidden_dims"])
    if pol.get("activation"):
        res.hints["activation"] = str(pol["activation"])
    if agent.get("empirical_normalization") or pol.get("actor_obs_normalization"):
        res.add("warn", "agent.empirical_normalization",
                "true: the obs mean/std live in the checkpoint, not the YAML. Isaac Lab's "
                "ONNX exporter bakes them into the graph (BreakPoint reads them from it); "
                "for a params: export pass --obs-norm")
    if agent.get("clip_actions") is not None:
        res.add("warn", "agent.clip_actions",
                f"{agent['clip_actions']}: BreakPoint does not clip actions")
    cls = str(pol.get("class_name") or "")
    if "Recurrent" in cls:
        rnn = str(pol.get("rnn_type") or "lstm").lower()
        res.hints["recurrent"] = {"type": rnn, "hidden": pol.get("rnn_hidden_dim"),
                                  "layers": pol.get("rnn_num_layers")}
        res.add("info", "agent.policy.class_name",
                f"{cls} ({rnn}): supported as a params: export of the actor state_dict "
                f"(memory_a.rnn.* and actor.* tensors in one .npz); the hidden state "
                f"resets to zeros each episode, as in RSL-RL")


def from_rsl_rl_run(run_dir, *, robot="go1", joint_names=None):
    """An RSL-RL run directory (Isaac Lab ``logs/rsl_rl/<task>/<run>``) -> ``ImportResult``.

    Reads ``params/env.yaml`` (required) and ``params/agent.yaml`` (optional), the two
    files Isaac Lab's ``train.py`` dumps next to the checkpoints.
    """
    d = Path(run_dir)
    env_y = next((p for p in (d / "params" / "env.yaml", d / "env.yaml") if p.exists()),
                 None)
    if env_y is None:
        raise FileNotFoundError(f"no params/env.yaml under {d}; is this an RSL-RL run dir?")
    agent_y = next((p for p in (d / "params" / "agent.yaml", d / "agent.yaml")
                    if p.exists()), None)
    res = from_isaaclab_cfg(env_y, agent_y, robot=robot, joint_names=joint_names)
    res.source = str(d)
    return res


def from_unitree_rl_gym(deploy_yaml, *, robot="go1", joint_names=None):
    """A unitree_rl_gym deploy config (``deploy/deploy_mujoco/configs/<robot>.yaml``).

    Those scripts build the obs as ``[ang_vel*ang_vel_scale, gravity, cmd*cmd_scale,
    (qj-default)*dof_pos_scale, dqj*dof_vel_scale, action]`` and command
    ``target = action*action_scale + default_angles`` with PD gains ``kps``/``kds``. A
    ``num_obs`` larger than that layout means extra terms (the humanoid configs append a
    gait phase clock) that BreakPoint cannot produce: reported as fatal.
    """
    from breakpoint_mjx.adapters.obs_spec import target_interface

    cfg, src = _load_cfg(deploy_yaml)
    iface = target_interface(robot)
    res = ImportResult(contract={"version": 1, "robot": iface.robot}, source=src)
    n = int(cfg.get("num_actions") or len(iface.joint_names))
    if n != len(iface.joint_names):
        res.add("fatal", "num_actions",
                f"{n} actions but {iface.robot} has {len(iface.joint_names)} actuators")
    names = list(joint_names or cfg.get("joint_names") or [])
    if not names:
        names = list(LEGGED_GYM_QUADRUPED_ORDER)
        res.add("warn", "joint order",
                "the deploy config does not name its joints, so the legged_gym DOF order "
                "(leg-major FL, FR, RL, RR) is ASSUMED; confirm with "
                "`breakpoint check-policy --infer-contract`")
    if n == len(iface.joint_names):
        res.contract["joint_names"] = names
        if cfg.get("default_angles") is not None:
            res.contract["default_joint_pos"] = [float(x) for x in cfg["default_angles"]]
    if cfg.get("action_scale") is not None:
        res.contract["action"] = {"scale": float(cfg["action_scale"])}
    obs = []
    if cfg.get("lin_vel_scale") is not None:   # legged_gym-style configs carry it
        obs.append({"term": "base_lin_vel", "scale": float(cfg["lin_vel_scale"])})
    obs += [{"term": "base_ang_vel", "scale": float(cfg.get("ang_vel_scale", 1.0))},
            {"term": "projected_gravity"},
            {"term": "commands",
             "scale": [float(x) for x in cfg.get("cmd_scale", [1.0, 1.0, 1.0])]},
            {"term": "joint_pos_rel", "scale": float(cfg.get("dof_pos_scale", 1.0))},
            {"term": "joint_vel", "scale": float(cfg.get("dof_vel_scale", 1.0))},
            {"term": "last_action"}]
    res.contract["obs"] = obs
    dim = 3 * sum(1 for t in obs if t["term"] not in
                  ("joint_pos_rel", "joint_vel", "last_action")) + 3 * n
    if cfg.get("num_obs") is not None and int(cfg["num_obs"]) != dim:
        res.add("fatal", "num_obs",
                f"{cfg['num_obs']} obs but the standard deploy layout is {dim}; the extra "
                f"{int(cfg['num_obs']) - dim} channels (a gait phase clock, for one) are "
                f"not something BreakPoint can produce")
    sim_dt, dec = cfg.get("simulation_dt"), cfg.get("control_decimation")
    if sim_dt and dec:
        cdt = float(sim_dt) * float(dec)
        if abs(cdt - iface.control_dt) > 1e-6:
            res.add("warn", "simulation_dt*control_decimation",
                    f"policy control period {cdt:.4f}s vs the target's "
                    f"{iface.control_dt:.4f}s; the policy runs at the target's rate")
    # ---- gains: the deploy YAML's kps/kds, else the legged_gym training config ----
    py_cfgs = _find_legged_gym_cfgs(deploy_yaml)
    lg = _literal_blocks(py_cfgs) if py_cfgs else {"domain_rand": {}, "control": {}}
    kps, kds, gsrc = cfg.get("kps"), cfg.get("kds"), "deploy"
    ctl = lg.get("control") or {}
    if kps is None and isinstance(ctl.get("stiffness"), dict):
        kps = _joint_dict_values(ctl["stiffness"], names)
        kds = (_joint_dict_values(ctl["damping"], names)
               if isinstance(ctl.get("damping"), dict) else None)
        gsrc = "control"
    _gain_rows(res, iface, kps, kds, gsrc, action_scale=cfg.get("action_scale"))

    # ---- training domain randomization -> operational ODD ----
    # A deploy YAML carries no DR, so it is read from (in order) a ``domain_rand:`` block
    # pasted into the YAML, then the legged_gym config classes of the checkout the YAML
    # lives in (or BP_TRAIN_CFG), parsed with ast and never executed.
    dr = dict(lg.get("domain_rand") or {})
    dr_src = "domain_rand"
    if py_cfgs:
        res.source = f"{src} + " + " + ".join(str(p) for p in py_cfgs)
    if isinstance(cfg.get("domain_rand"), dict):
        dr.update(cfg["domain_rand"])
    if dr:
        _legged_gym_odd(res, iface.robot, dr, dr_src)
    else:
        res.add("warn", "domain_rand",
                "no training domain randomization found (paste the legged_gym domain_rand "
                "block into the YAML, run init inside the unitree_rl_gym checkout, or set "
                "BP_TRAIN_CFG); the operational ODD defaults to the plausible bands")
    return res


IMPORTERS = {"isaaclab": from_isaaclab_cfg, "rsl_rl": from_rsl_rl_run,
             "unitree_rl_gym": from_unitree_rl_gym}
