"""CPU unit tests for the BYO-policy adapters (breakpoint_mjx/adapters, T4).

jax is absent on the dev box, so these exercise ONLY the ``xp=numpy`` forward + the pure
importer/shim logic (the jax forward is Colab-validated inside the evaluator's vmap). Covers:

  * ``mlp_forward`` vs a hand-computed 2-layer reference, incl. relu/elu activation math;
  * ``from_params`` (in_out + torch out_in transpose + state_dict dict) -> numpy forward;
  * ``ObsSpec`` key concat / index reorder / normalization + length-mismatch error;
  * ``OutputTransform`` squash/scale/bias + mean slice;
  * import-safety (package imports with no jax/onnx; policy_fn defers jax);
  * ``from_onnx`` via ``importorskip("onnx")`` on a tiny 2-layer Gemm MLP.
"""
from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

from breakpoint_mjx.adapters import (
    MLPPolicy,
    ObsSpec,
    OutputTransform,
    build_policy,
    from_npz,
    from_params,
    mlp_forward,
)


# --------------------------------------------------------------------------- #
# mlp_forward: hand-computed reference + activation correctness
# --------------------------------------------------------------------------- #
def _two_layer():
    W0 = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])  # (3, 2)
    b0 = np.array([0.05, -0.1])                            # (2,)
    W1 = np.array([[0.7, -0.8, 0.2], [0.1, 0.9, -0.3]])    # (2, 3)
    b1 = np.array([0.0, 0.25, -0.15])                      # (3,)
    return [(W0, b0), (W1, b1)]


def test_mlp_forward_two_layer_tanh_reference():
    params = _two_layer()
    x = np.array([1.0, 2.0, 3.0])
    (W0, b0), (W1, b1) = params
    expected = np.tanh(x @ W0 + b0) @ W1 + b1  # tanh hidden, LINEAR output
    out = mlp_forward(params, x, "tanh", np)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    assert out.shape == (3,)


def test_mlp_forward_relu_activation():
    params = _two_layer()
    x = np.array([1.0, 2.0, 3.0])
    (W0, b0), (W1, b1) = params
    hidden = x @ W0 + b0
    assert np.any(hidden < 0) and np.any(hidden > 0)  # both branches exercised
    expected = np.maximum(hidden, 0.0) @ W1 + b1
    np.testing.assert_allclose(mlp_forward(params, x, "relu", np), expected, atol=1e-12)


def test_mlp_forward_elu_activation():
    params = _two_layer()
    x = np.array([1.0, 2.0, 3.0])
    (W0, b0), (W1, b1) = params
    hidden = x @ W0 + b0
    elu = np.where(hidden > 0, hidden, np.expm1(hidden))
    expected = elu @ W1 + b1
    np.testing.assert_allclose(mlp_forward(params, x, "elu", np), expected, atol=1e-12)


def test_mlp_forward_bias_none_is_zero():
    W = np.array([[1.0, 0.0], [0.0, 2.0]])
    x = np.array([3.0, 4.0])
    out = mlp_forward([(W, None)], x, "tanh", np)  # single linear layer, no bias
    np.testing.assert_allclose(out, x @ W, atol=1e-12)


def test_mlp_forward_empty_raises():
    with pytest.raises(ValueError):
        mlp_forward([], np.zeros(3), "tanh", np)


def test_mlp_forward_unknown_activation_raises():
    with pytest.raises(ValueError):
        mlp_forward(_two_layer(), np.zeros(3), "sigmoid", np)


# --------------------------------------------------------------------------- #
# from_params: numpy forward matches, layouts, state_dict dict
# --------------------------------------------------------------------------- #
def test_from_params_numpy_forward_matches_mlp_forward():
    params = _two_layer()
    pol = from_params(params, activation="tanh")
    x = np.array([0.5, -1.0, 2.0])
    obs = {"state": x}
    np.testing.assert_allclose(pol.act_np(obs), mlp_forward(params, x, "tanh", np), atol=1e-12)
    assert pol.in_dim == 3 and pol.action_size == 3


def test_from_params_out_in_transpose_matches_in_out():
    # torch nn.Linear weight is (out, in); its transpose is the canonical (in, out).
    (W0, b0), (W1, b1) = _two_layer()
    torch_layers = [(W0.T, b0), (W1.T, b1)]  # (out, in)
    pol_torch = from_params(torch_layers, activation="tanh", weight_layout="out_in")
    pol_flax = from_params([(W0, b0), (W1, b1)], activation="tanh", weight_layout="in_out")
    x = np.array([1.0, -2.0, 0.5])
    obs = {"state": x}
    np.testing.assert_allclose(pol_torch.act_np(obs), pol_flax.act_np(obs), atol=1e-12)
    np.testing.assert_allclose(pol_torch.params[0][0], W0, atol=1e-12)


def test_from_params_accepts_state_dict_dict():
    (W0, b0), (W1, b1) = _two_layer()
    # torch-style state_dict (weights (out, in)); insertion order = network order.
    sd = {
        "net.0.weight": W0.T,
        "net.0.bias": b0,
        "net.2.weight": W1.T,
        "net.2.bias": b1,
    }
    pol = from_params(sd, activation="tanh", weight_layout="out_in")
    x = np.array([0.2, 0.4, -0.6])
    ref = from_params([(W0, b0), (W1, b1)])
    np.testing.assert_allclose(pol.act_np({"state": x}), ref.act_np({"state": x}), atol=1e-12)


def test_from_params_flat_list():
    (W0, b0), (W1, b1) = _two_layer()
    pol = from_params([W0, b0, W1, b1], activation="tanh")
    ref = from_params([(W0, b0), (W1, b1)])
    x = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(pol.act_np({"state": x}), ref.act_np({"state": x}), atol=1e-12)


def test_from_params_accepts_flax_kernel_dict():
    # flax params: '*kernel' is already (in, out) -> weight_layout stays in_out (no transpose).
    (W0, b0), (W1, b1) = _two_layer()  # W0/W1 are already canonical (in, out)
    flax = {
        "Dense_0/kernel": W0,
        "Dense_0/bias": b0,
        "Dense_1/kernel": W1,
        "Dense_1/bias": b1,
    }
    pol = from_params(flax, activation="tanh", weight_layout="in_out")
    ref = from_params([(W0, b0), (W1, b1)])
    x = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(pol.act_np({"state": x}), ref.act_np({"state": x}), atol=1e-12)
    np.testing.assert_allclose(pol.params[0][0], W0, atol=1e-12)  # kernel not transposed


# --------------------------------------------------------------------------- #
# ObsSpec: mapping, ordering, normalization, validation
# --------------------------------------------------------------------------- #
def test_obs_spec_default_reads_state():
    spec = ObsSpec()
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.transform({"state": x}, np), x)


def test_obs_spec_indices_reorder_and_subselect():
    spec = ObsSpec(indices=(2, 0))  # select+reorder to length 2
    x = np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(spec.transform({"state": x}, np), np.array([30.0, 10.0]))
    assert spec.output_dim() == 2


def test_obs_spec_normalization_mean_std():
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([2.0, 4.0, 0.5])
    spec = ObsSpec(mean=mean, std=std)
    x = np.array([3.0, 6.0, 4.0])
    np.testing.assert_allclose(spec.transform({"state": x}, np), (x - mean) / std, atol=1e-12)


def test_obs_spec_multi_key_concat_in_order():
    spec = ObsSpec(keys=("state", "extra"))
    obs = {"state": np.array([1.0, 2.0]), "extra": np.array([3.0])}
    np.testing.assert_allclose(spec.transform(obs, np), np.array([1.0, 2.0, 3.0]))


def test_obs_spec_length_mismatch_raises():
    spec = ObsSpec(expected_dim=4)
    with pytest.raises(ValueError, match="does not match"):
        spec.transform({"state": np.zeros(3)}, np)


def test_obs_spec_missing_key_raises():
    spec = ObsSpec(keys=("state",))
    with pytest.raises(KeyError):
        spec.transform({"privileged_state": np.zeros(3)}, np)


def test_obs_spec_zero_std_raises():
    with pytest.raises(ValueError, match="zero"):
        ObsSpec(std=np.array([1.0, 0.0, 2.0]))


def test_obs_spec_mean_length_mismatch_with_expected_dim_raises():
    with pytest.raises(ValueError):
        ObsSpec(mean=np.zeros(3), expected_dim=4)


# --------------------------------------------------------------------------- #
# OutputTransform
# --------------------------------------------------------------------------- #
def test_output_transform_squash_scale_bias():
    ot = OutputTransform(squash=True, scale=np.array([2.0, 3.0]), bias=np.array([1.0, -1.0]))
    a = np.array([0.5, -0.5])
    expected = np.tanh(a) * np.array([2.0, 3.0]) + np.array([1.0, -1.0])
    np.testing.assert_allclose(ot.apply(a, np), expected, atol=1e-12)


def test_output_transform_identity_default():
    ot = OutputTransform()
    a = np.array([0.5, -0.5, 2.0])
    np.testing.assert_allclose(ot.apply(a, np), a, atol=1e-12)


def test_output_transform_mean_dim_slices_gaussian_head():
    # A [mean(2), log_std(2)] head -> keep the mean for deterministic eval.
    ot = OutputTransform(mean_dim=2)
    a = np.array([0.1, 0.2, -5.0, -5.0])
    np.testing.assert_allclose(ot.apply(a, np), np.array([0.1, 0.2]), atol=1e-12)


def test_output_transform_sigmoid_squash():
    ot = OutputTransform(squash="sigmoid")
    a = np.array([0.0, 2.0, -1.0])
    np.testing.assert_allclose(ot.apply(a, np), 1.0 / (1.0 + np.exp(-a)), atol=1e-12)


def test_output_transform_squash_true_is_tanh():
    ot = OutputTransform(squash=True)
    a = np.array([0.3, -0.7])
    np.testing.assert_allclose(ot.apply(a, np), np.tanh(a), atol=1e-12)


def test_mlp_policy_action_size_reflects_mean_dim():
    # last layer emits 4 (= 2 mean + 2 log_std); action_size must be the mean_dim.
    W = np.zeros((3, 4))
    pol = from_params([(W, np.zeros(4))], out_transform=OutputTransform(mean_dim=2))
    assert pol.action_size == 2


# --------------------------------------------------------------------------- #
# MLPPolicy build-time validation
# --------------------------------------------------------------------------- #
def test_mlp_policy_obs_spec_dim_mismatch_raises():
    W = np.zeros((5, 2))  # expects input dim 5
    with pytest.raises(ValueError, match="first layer expects"):
        MLPPolicy(params=[(W, np.zeros(2))], obs_spec=ObsSpec(expected_dim=4))


def test_mlp_policy_broken_chain_raises():
    W0 = np.zeros((3, 2))
    W1 = np.zeros((7, 4))  # in-dim 7 != previous out-dim 2
    with pytest.raises(ValueError, match="valid MLP chain"):
        MLPPolicy(params=[(W0, np.zeros(2)), (W1, np.zeros(4))])


def test_from_params_auto_sets_expected_dim_and_catches_bad_obs():
    pol = from_params(_two_layer())  # first layer in-dim 3
    assert pol.obs_spec.expected_dim == 3
    with pytest.raises(ValueError, match="does not match"):
        pol.act_np({"state": np.zeros(4)})  # wrong obs length -> caught


# --------------------------------------------------------------------------- #
# Import-safety: no jax / onnx required to import or run the numpy path
# --------------------------------------------------------------------------- #
def test_package_imports_and_forward_without_jax():
    import breakpoint_mjx.adapters as A  # noqa: F401  (already imported at top; explicit)

    pol = from_params([(np.eye(3), np.zeros(3))])
    obs = {"state": np.arange(3.0)}
    np.testing.assert_allclose(pol.act_np(obs), np.arange(3.0), atol=1e-12)


def test_policy_fn_defers_jax_import():
    pol = from_params([(np.eye(3), np.zeros(3))])
    obs = {"state": np.arange(3.0)}
    if importlib.util.find_spec("jax") is None:
        # jnp is imported lazily inside policy_fn, so calling it (not importing the
        # package) is what surfaces the missing dependency.
        with pytest.raises(ModuleNotFoundError):
            pol.policy_fn(obs, None)


def test_from_onnx_missing_dep_raises_clearly():
    if importlib.util.find_spec("onnx") is None:
        from breakpoint_mjx.adapters import from_onnx

        with pytest.raises(ImportError, match="onnx"):
            from_onnx("nonexistent.onnx")


# --------------------------------------------------------------------------- #
# from_npz + build_policy (params path, torch layout, obs normalization)
# --------------------------------------------------------------------------- #
def test_from_npz_and_build_policy_params_path(tmp_path):
    (W0, b0), (W1, b1) = _two_layer()
    npz = tmp_path / "weights.npz"
    # torch-style dump: weights (out, in), named *.weight / *.bias.
    np.savez(
        npz,
        **{
            "0.weight": W0.T,
            "0.bias": b0,
            "2.weight": W1.T,
            "2.bias": b1,
        },
    )
    pol = from_npz(str(npz), activation="tanh", weight_layout="out_in")
    ref = from_params([(W0, b0), (W1, b1)])
    x = np.array([0.3, -0.2, 0.9])
    np.testing.assert_allclose(pol.act_np({"state": x}), ref.act_np({"state": x}), atol=1e-12)

    # build_policy: same npz via the "params:" spec + an obs-norm npz + action scaling.
    norm = tmp_path / "norm.npz"
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([1.0, 1.0, 1.0])
    np.savez(norm, mean=mean, std=std)
    pol2 = build_policy(
        f"params:{npz}",
        obs_norm_path=str(norm),
        activation="tanh",
        action_scale=2.0,
        weight_layout="out_in",
    )
    expected = ref.act_np({"state": x}) * 2.0
    np.testing.assert_allclose(pol2.act_np({"state": x}), expected, atol=1e-12)


def test_build_policy_bad_spec_raises():
    # a bare path is accepted now (kind from the extension); a missing file says so and
    # still spells out the spec syntax
    with pytest.raises(ValueError, match="no policy file.*onnx:<path>, params:<path>"):
        build_policy("model.onnx")
    with pytest.raises(ValueError, match="unknown policy kind"):
        build_policy("torch:model.pt")


def _params_npz(tmp_path):
    (W0, b0), (W1, b1) = _two_layer()
    npz = tmp_path / "w.npz"
    np.savez(
        npz,
        **{"0.weight": W0.T, "0.bias": b0, "2.weight": W1.T, "2.bias": b1},
    )
    return npz, (W0, b0, W1, b1)


def test_build_policy_warns_when_no_norm_or_action_transform(tmp_path):
    npz, _ = _params_npz(tmp_path)
    with pytest.warns(UserWarning, match="no obs normalization"):
        build_policy(f"params:{npz}", weight_layout="out_in")


def test_build_policy_action_bias_applied_and_no_warning(tmp_path):
    npz, (W0, b0, W1, b1) = _params_npz(tmp_path)
    ref = from_params([(W0, b0), (W1, b1)])
    x = np.array([0.3, -0.2, 0.9])
    # action_bias reaches OutputTransform (the CLI/README `s*tanh(mean)+bias` is reachable)
    # AND counts as an action transform, so no "untransformed" warning fires.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pol = build_policy(f"params:{npz}", weight_layout="out_in", action_bias=0.5)
    np.testing.assert_allclose(
        pol.act_np({"state": x}), ref.act_np({"state": x}) + 0.5, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# from_onnx: build a tiny 2-layer MLP graph and check extracted layers
# --------------------------------------------------------------------------- #
def test_from_onnx_matches_from_params(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(0)
    IN, H, OUT = 4, 5, 3
    # torch orientation (out, in) exported as Gemm(transB=1).
    W0 = rng.standard_normal((H, IN)).astype(np.float32)
    b0 = rng.standard_normal(H).astype(np.float32)
    W1 = rng.standard_normal((OUT, H)).astype(np.float32)
    b1 = rng.standard_normal(OUT).astype(np.float32)

    initializers = [
        numpy_helper.from_array(W0, "W0"),
        numpy_helper.from_array(b0, "b0"),
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(b1, "b1"),
    ]
    nodes = [
        helper.make_node("Gemm", ["obs", "W0", "b0"], ["h0"], transB=1),
        helper.make_node("Tanh", ["h0"], ["a0"]),
        helper.make_node("Gemm", ["a0", "W1", "b1"], ["act"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [IN])],
        [helper.make_tensor_value_info("act", TensorProto.FLOAT, [OUT])],
        initializer=initializers,
    )
    model = helper.make_model(graph)
    path = tmp_path / "policy.onnx"
    onnx.save(model, str(path))

    pol = from_onnx(str(path))
    # activation auto-detected from the Tanh node; layers canonicalized to (in, out).
    assert pol.activation == "tanh"
    assert pol.in_dim == IN and pol.action_size == OUT
    np.testing.assert_allclose(pol.params[0][0], W0.T, atol=1e-6)
    np.testing.assert_allclose(pol.params[1][0], W1.T, atol=1e-6)

    ref = from_params([(W0, b0), (W1, b1)], activation="tanh", weight_layout="out_in")
    x = rng.standard_normal(IN).astype(np.float32)
    np.testing.assert_allclose(pol.act_np({"state": x}), ref.act_np({"state": x}), atol=1e-5)


# --------------------------------------------------------------------------- #
# from_onnx: FAIL-LOUD contract (never silently drop a node / guess an activation)
# --------------------------------------------------------------------------- #
def _save_onnx(tmp_path, initializers, nodes, in_dim, out_dim, inputs=None, output_name="act"):
    import onnx
    from onnx import helper, TensorProto

    if inputs is None:
        inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [in_dim])]
    graph = helper.make_graph(
        nodes,
        "mlp",
        inputs,
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [out_dim])],
        initializer=initializers,
    )
    model = helper.make_model(graph)
    path = tmp_path / "policy.onnx"
    onnx.save(model, str(path))
    return str(path)


def _gemm_layers(rng, dims):
    """Return (initializers, W_list) for torch-style Gemm layers over the given dim chain."""
    from onnx import numpy_helper

    inits, weights = [], []
    for li, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        W = rng.standard_normal((dout, din)).astype(np.float32)  # torch (out, in)
        b = rng.standard_normal(dout).astype(np.float32)
        inits += [numpy_helper.from_array(W, f"W{li}"), numpy_helper.from_array(b, f"b{li}")]
        weights.append((W, b))
    return inits, weights


def test_from_onnx_unknown_hidden_activation_raises(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(1)
    inits, _ = _gemm_layers(rng, [4, 5, 3])
    nodes = [
        helper.make_node("Gemm", ["obs", "W0", "b0"], ["h0"], transB=1),
        helper.make_node("Sigmoid", ["h0"], ["a0"]),  # unsupported HIDDEN activation
        helper.make_node("Gemm", ["a0", "W1", "b1"], ["act"], transB=1),
    ]
    path = _save_onnx(tmp_path, inits, nodes, 4, 3)
    with pytest.raises(ValueError, match="unknown hidden activation"):
        from_onnx(path)


def test_from_onnx_trailing_tanh_captured_not_dropped(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(2)
    inits, weights = _gemm_layers(rng, [3, 4, 2])
    (W0, b0), (W1, b1) = weights
    nodes = [
        helper.make_node("Gemm", ["obs", "W0", "b0"], ["h0"], transB=1),
        helper.make_node("Relu", ["h0"], ["a0"]),
        helper.make_node("Gemm", ["a0", "W1", "b1"], ["z"], transB=1),
        helper.make_node("Tanh", ["z"], ["act"]),  # trailing squash -> graph OUTPUT
    ]
    path = _save_onnx(tmp_path, inits, nodes, 3, 2)
    pol = from_onnx(path)
    assert pol.activation == "relu"
    assert pol.out_transform is not None and pol.out_transform.squash == "tanh"
    x = rng.standard_normal(3).astype(np.float32)
    expected = np.tanh(np.maximum(x @ W0.T + b0, 0.0) @ W1.T + b1)  # squash NOT dropped
    np.testing.assert_allclose(pol.act_np({"state": x}), expected, atol=1e-5)


def test_from_onnx_unconsumed_op_raises(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(3)
    inits, _ = _gemm_layers(rng, [4, 5, 3])
    nodes = [
        helper.make_node("Gemm", ["obs", "W0", "b0"], ["h0"], transB=1),
        helper.make_node("Relu", ["h0"], ["a0"]),
        helper.make_node("Mul", ["a0", "a0"], ["stray"]),  # unconsumed stray op
        helper.make_node("Gemm", ["a0", "W1", "b1"], ["act"], transB=1),
    ]
    path = _save_onnx(tmp_path, inits, nodes, 4, 3)
    with pytest.raises(ValueError, match="unsupported ONNX op 'Mul'"):
        from_onnx(path)


def test_from_onnx_transA_raises(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(4)
    inits, _ = _gemm_layers(rng, [4, 3])
    nodes = [helper.make_node("Gemm", ["obs", "W0", "b0"], ["act"], transA=1, transB=1)]
    path = _save_onnx(tmp_path, inits, nodes, 4, 3)
    with pytest.raises(ValueError, match="transA"):
        from_onnx(path)


def test_from_onnx_weight_as_graph_input_raises(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(5)
    b0 = rng.standard_normal(3).astype(np.float32)
    inits = [numpy_helper.from_array(b0, "b0")]  # only the bias is an initializer
    nodes = [helper.make_node("Gemm", ["obs", "W0", "b0"], ["act"], transB=1)]
    inputs = [
        helper.make_tensor_value_info("obs", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("W0", TensorProto.FLOAT, [3, 4]),  # weight as INPUT
    ]
    path = _save_onnx(tmp_path, inits, nodes, 4, 3, inputs=inputs)
    with pytest.raises(ValueError, match="not a graph initializer|graph inputs"):
        from_onnx(path)


def test_from_onnx_leading_sub_div_captured_as_normalization(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(6)
    IN, OUT = 4, 3
    mean = rng.standard_normal(IN).astype(np.float32)
    std = (np.abs(rng.standard_normal(IN)) + 0.5).astype(np.float32)
    W0 = rng.standard_normal((OUT, IN)).astype(np.float32)
    b0 = rng.standard_normal(OUT).astype(np.float32)
    inits = [
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(std, "std"),
        numpy_helper.from_array(W0, "W0"),
        numpy_helper.from_array(b0, "b0"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["s"]),  # (x - mean)
        helper.make_node("Div", ["s", "std"], ["d"]),      # / std
        helper.make_node("Gemm", ["d", "W0", "b0"], ["act"], transB=1),
    ]
    path = _save_onnx(tmp_path, inits, nodes, IN, OUT)
    pol = from_onnx(path)
    np.testing.assert_allclose(pol.obs_spec.mean, mean, atol=1e-6)
    np.testing.assert_allclose(pol.obs_spec.std, std, atol=1e-6)
    x = rng.standard_normal(IN).astype(np.float32)
    expected = ((x - mean) / std) @ W0.T + b0  # normalization NOT dropped
    np.testing.assert_allclose(pol.act_np({"state": x}), expected, atol=1e-5)


def test_from_onnx_norm_conflict_with_explicit_obs_spec_raises(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper
    from breakpoint_mjx.adapters import from_onnx

    rng = np.random.default_rng(7)
    IN, OUT = 4, 3
    mean = rng.standard_normal(IN).astype(np.float32)
    W0 = rng.standard_normal((OUT, IN)).astype(np.float32)
    b0 = rng.standard_normal(OUT).astype(np.float32)
    inits = [
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(W0, "W0"),
        numpy_helper.from_array(b0, "b0"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["s"]),
        helper.make_node("Gemm", ["s", "W0", "b0"], ["act"], transB=1),
    ]
    path = _save_onnx(tmp_path, inits, nodes, IN, OUT)
    with pytest.raises(ValueError, match="baked-in obs normalization"):
        from_onnx(path, obs_spec=ObsSpec(mean=np.zeros(IN)))


# --------------------------------------------------------------------------- #
# Contracts (obs_spec.PolicyContract), brax import, spec parsing, config importers
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from breakpoint_mjx.adapters import (  # noqa: E402
    apply_contract,
    build_contract,
    from_brax_params,
    from_isaaclab_cfg,
    from_rsl_rl_run,
    from_unitree_rl_gym,
    load_contract,
    parse_policy_spec,
)
from breakpoint_mjx.adapters.importers import ISAACLAB_QUADRUPED_ORDER  # noqa: E402
from breakpoint_mjx.adapters.obs_spec import GO1_INTERFACE  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "contract"
GO1_TERMS = ["base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos_rel",
             "joint_vel", "last_action", "commands"]


def test_contract_native_layout_is_the_identity():
    c = build_contract({"robot": "go1", "obs": GO1_TERMS})
    np.testing.assert_array_equal(c.obs_idx, np.arange(48))
    assert not c.obs_offset.any() and np.all(c.obs_scale == 1.0)
    np.testing.assert_array_equal(c.act_idx, np.arange(12))
    assert np.all(c.act_scale == 1.0) and not c.act_offset.any()


def test_term_builder_indices_scales_and_joint_order_on_go1():
    c = build_contract({
        "robot": "go1", "joint_names": list(ISAACLAB_QUADRUPED_ORDER),
        "action": {"scale": 0.25},
        "obs": [{"term": "ang_vel", "scale": 0.25}, "gravity",
                {"term": "dof_vel", "scale": 0.05}, "actions"]})
    # policy joint 0 is FL_hip = Go1 actuator 3 -> joint_vel channel 21 + 3
    assert c.obs_idx[:6].tolist() == [3, 4, 5, 6, 7, 8]
    assert c.obs_idx[6:10].tolist() == [24, 21, 30, 27]
    assert c.obs_scale[6] == pytest.approx(0.05)
    # last_action feeds the policy its OWN raw output: env action / (0.25 / 0.5)
    assert c.obs_scale[18] == pytest.approx(2.0)
    # env actuator 0 (FR_hip) takes policy output 1 (FR_hip in type-major order)
    assert c.act_idx[0] == 1 and c.act_idx[3] == 0
    assert np.allclose(c.act_scale, 0.5)


def test_default_pose_offsets_obs_and_action():
    c = build_contract({"robot": "go1", "default_joint_pos": [0.0] * 12,
                        "obs": ["joint_pos_rel"]})
    env_def = np.asarray(GO1_INTERFACE.default_pose)
    # policy wants q - 0 = (q - q0_env) + q0_env  -> offset = -q0_env
    np.testing.assert_allclose(c.obs_offset, -env_def)
    np.testing.assert_allclose(c.act_offset, -env_def / GO1_INTERFACE.action_scale)


def test_contract_yaml_round_trip(tmp_path):
    spec = {"robot": "go1", "joint_names": list(ISAACLAB_QUADRUPED_ORDER),
            "action": {"scale": 0.25},
            "obs": [{"term": "base_ang_vel", "scale": 0.25},
                    {"term": "projected_gravity", "scale": [1.0, -1.0, 1.0]},
                    {"term": "commands", "scale": [2.0, 2.0, 0.25]},
                    "joint_pos_rel", {"term": "joint_vel", "scale": 0.05}, "last_action"],
            "normalization": {"mean": [0.0] * 45, "std": [1.0] * 45}}
    c = build_contract(spec)
    p = tmp_path / "c.yaml"
    c.to_yaml(p)
    c2 = load_contract(p)
    for a, b in zip(c.arrays(), c2.arrays()):
        np.testing.assert_allclose(a, b)
    assert c2.normalization is not None
    # and through ObsSpec (term form) + the raw form
    sp = c.to_obs_spec()
    sp2 = ObsSpec.from_yaml(sp.to_yaml())
    np.testing.assert_array_equal(sp.indices, sp2.indices)
    raw = ObsSpec(indices=[2, 0, 1], scale=[1, 2, 3], offset=[0, 0, 1])
    raw2 = ObsSpec.from_yaml(raw.to_yaml())
    x = {"state": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(raw.transform(x, np), raw2.transform(x, np))


def test_unknown_term_and_unmatched_joint_are_loud():
    with pytest.raises(ValueError, match="height_scan"):
        build_contract({"robot": "go1", "obs": ["height_scan"]})
    with pytest.raises(ValueError, match="no match"):
        build_contract({"robot": "go1", "joint_names": ["a"] * 12, "obs": ["joint_vel"]})
    with pytest.raises(KeyError, match="no declarative obs interface"):
        build_contract({"robot": "spot", "obs": ["joint_vel"]})
    raw = build_contract({"robot": "go1", "obs": [{"term": "clock", "start": 0, "dim": 2}]})
    assert raw.obs_idx.tolist() == [0, 1]


def test_output_transform_permutes_policy_order_to_env_order():
    ot = OutputTransform(perm=[2, 0, 1], scale=np.array([1.0, 10.0, 100.0]))
    np.testing.assert_allclose(ot.apply(np.array([1.0, 2.0, 3.0]), np), [3.0, 10.0, 200.0])
    with pytest.raises(ValueError, match="permutation"):
        OutputTransform(perm=[0, 0, 1])


def test_apply_contract_matches_manual_map():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(48, 12))
    pol = from_params([(W, np.zeros(12))], activation="identity",
                      obs_spec=ObsSpec(mean=np.zeros(48), std=np.ones(48)),
                      out_transform=OutputTransform(squash="tanh"))
    c = build_contract({"robot": "go1", "joint_names": list(ISAACLAB_QUADRUPED_ORDER),
                        "action": {"scale": 0.25},
                        "obs": ["commands", "base_ang_vel", "base_lin_vel",
                                "projected_gravity", "joint_vel", "joint_pos_rel",
                                {"term": "last_action"}]})
    driven = apply_contract(pol, c)
    o = rng.normal(size=48)
    x = (o[c.obs_idx] - c.obs_offset) * c.obs_scale
    y = np.tanh(x @ W)
    expected = y[c.act_idx] * c.act_scale + c.act_offset
    np.testing.assert_allclose(driven.act_np({"state": o}), expected, atol=1e-12)


def test_parse_policy_spec_windows_paths(tmp_path):
    pkl = tmp_path / "go1_policy.pkl"
    pkl.write_bytes(b"x")
    win = str(pkl)                       # C:\...\go1_policy.pkl on Windows
    kind, path = parse_policy_spec(win)
    assert kind == "brax" and path == win
    assert parse_policy_spec(win.replace("\\", "/"))[0] == "brax"
    assert parse_policy_spec(f"brax:{win}") == ("brax", win)
    assert parse_policy_spec("params:C:/w.npz") == ("params", "C:/w.npz")
    assert parse_policy_spec("ONNX:C:/p.onnx") == ("onnx", "C:/p.onnx")
    with pytest.raises(ValueError, match="unknown policy kind"):
        parse_policy_spec("C:/runs/policy.pt")


def _fake_brax(rng):
    norm = SimpleNamespace(mean={"state": rng.normal(size=6), "privileged_state": 0},
                           std={"state": rng.uniform(0.5, 2, 6), "privileged_state": 1})
    pol = {"params": {f"hidden_{i}": {"kernel": rng.normal(size=s),
                                      "bias": rng.normal(size=s[1])}
                      for i, s in enumerate([(6, 8), (8, 4)])}}
    return (norm, pol, None)


def test_from_brax_params_matches_the_brax_math():
    rng = np.random.default_rng(4)
    params = _fake_brax(rng)
    pol = from_brax_params(params)
    x = rng.normal(size=6)
    n = params[0]
    h = (x - n.mean["state"]) / n.std["state"]
    L = params[1]["params"]
    h = h @ L["hidden_0"]["kernel"] + L["hidden_0"]["bias"]
    h = h / (1 + np.exp(-h))                                  # silu
    h = h @ L["hidden_1"]["kernel"] + L["hidden_1"]["bias"]
    np.testing.assert_allclose(pol.act_np({"state": x}), np.tanh(h[:2]), atol=1e-12)
    assert pol.action_size == 2


def test_build_policy_applies_bp_obs_spec(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    W = rng.normal(size=(12, 48))                       # torch (out, in)
    npz = tmp_path / "w.npz"
    np.savez(npz, **{"0.weight": W, "0.bias": np.zeros(12)})
    spec = {"robot": "go1", "joint_names": list(ISAACLAB_QUADRUPED_ORDER),
            "obs": ["commands", "base_lin_vel", "base_ang_vel", "projected_gravity",
                    "joint_pos_rel", "joint_vel", "last_action"]}
    y = tmp_path / "c.yaml"
    build_contract(spec).to_yaml(y)
    monkeypatch.setenv("BP_OBS_SPEC", str(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pol = build_policy(f"params:{npz}", activation="identity")
    c = load_contract(y)
    o = rng.normal(size=48)
    x = o[c.obs_idx]
    np.testing.assert_allclose(pol.act_np({"state": o}), (x @ W.T)[c.act_idx], atol=1e-10)


# ---- importers on fixtures that mirror real config layouts -------------------
def test_rsl_rl_run_dir_import():
    res = from_rsl_rl_run(FIX / "rsl_rl_go1_flat")
    c = res.contract
    assert [t["term"] for t in c["obs"]] == [
        "base_lin_vel", "base_ang_vel", "projected_gravity", "commands", "joint_pos_rel",
        "joint_vel", "last_action"]
    assert c["joint_names"] == list(ISAACLAB_QUADRUPED_ORDER)     # assumed, and said so
    assert c["action"] == {"scale": 0.25}
    assert c["default_joint_pos"][:4] == [0.1, -0.1, 0.1, -0.1]   # L hips +, R hips -
    assert c["default_joint_pos"][4:8] == [0.8, 0.8, 1.0, 1.0]
    fields = {r["field"] for r in res.rows}
    assert {"joint order", "actuators.stiffness", "agent.empirical_normalization",
            "observations.policy.enable_corruption"} <= fields
    assert not res.fatal
    assert res.hints["activation"] == "elu"
    built = build_contract(c)
    assert built.obs_dim == 48 and built.action_dim == 12


def test_bundled_policy_fixture_imports_to_the_native_contract():
    res = from_rsl_rl_run(FIX / "bundled_go1_rsl_rl")
    assert res.rows == []                                  # nothing to caveat
    c = build_contract(res.contract)
    np.testing.assert_array_equal(c.obs_idx, np.arange(48))
    np.testing.assert_allclose(c.obs_offset, 0.0, atol=1e-12)
    np.testing.assert_array_equal(c.act_idx, np.arange(12))
    np.testing.assert_allclose(c.act_scale, 1.0)
    np.testing.assert_allclose(c.act_offset, 0.0, atol=1e-12)


def test_unitree_rl_gym_deploy_import():
    res = from_unitree_rl_gym(FIX / "unitree_go2_deploy.yaml")
    c = build_contract(res.contract)
    assert c.obs_dim == 45                      # no linear velocity in deploy obs
    assert [t["term"] for t in res.contract["obs"]][0] == "base_ang_vel"
    assert c.obs_scale[0] == pytest.approx(0.25)
    assert res.contract["action"] == {"scale": 0.25}
    fields = {r["field"]: r for r in res.rows}
    assert "deploy.stiffness" in fields and "joint order" in fields
    assert "simulation_dt*control_decimation" not in fields   # 0.005*4 = 0.02 matches
    assert not res.fatal


def test_unitree_phase_clock_is_fatal():
    res = from_unitree_rl_gym(FIX / "unitree_phase_deploy.yaml")
    assert [r["field"] for r in res.fatal] == ["num_obs"]


def test_isaaclab_dict_input_and_python_configs_refused(tmp_path):
    cfg = {"observations": {"policy": {
        "height_scan": {"func": "isaaclab.envs.mdp.observations:height_scan"},
        "base_ang_vel": {"func": "isaaclab.envs.mdp.observations:base_ang_vel",
                         "scale": 0.25}}}}
    res = from_isaaclab_cfg(cfg)
    assert [r["field"] for r in res.fatal] == ["observations.policy.height_scan"]
    assert res.contract["obs"] == [{"term": "base_ang_vel", "scale": 0.25}]
    py = tmp_path / "env_cfg.py"
    py.write_text("x = 1")
    with pytest.raises(ValueError, match="never executed"):
        from_isaaclab_cfg(py)
