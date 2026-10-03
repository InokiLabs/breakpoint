"""Every BP_SEARCH branch must be reachable without an unbound name.

The `multi` branch shipped referencing `seed_theta`, which the cmaes branch defines
LATER, so `BP_SEARCH=multi` died with UnboundLocalError after ~150s of real GPU setup.
No unit test caught it because the bug lived in the dispatch, not in any search.

These tests compile the dispatch and check each branch's free variables against what
is actually bound before it, which catches this whole class statically and in
milliseconds.
"""
import ast
import inspect
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "breakpoint_mjx" / "pipeline.py"
MODES = ("multi", "agentic", "llm", "atlas", "active", "cmaes")


def _main_ast():
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("pipeline.main not found")


def _names_bound_before(main_node, target_lineno):
    """Every name assigned anywhere in main() strictly before target_lineno."""
    bound = set()
    for node in ast.walk(main_node):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            if getattr(node, "lineno", 1 << 30) < target_lineno:
                for t in ast.walk(node):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        bound.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if getattr(node, "lineno", 1 << 30) < target_lineno:
                for a in node.names:
                    bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.For) and getattr(node, "lineno", 1 << 30) < target_lineno:
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    for a in main_node.args.args:
        bound.add(a.arg)
    return bound


def _search_calls():
    """(mode, call_node) for each `cr = <search>(...)` in the dispatch."""
    main_node = _main_ast()
    out = []
    for node in ast.walk(main_node):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "cr" for t in node.targets):
            continue
        # Only direct calls; `cr` is also rebuilt by a comprehension elsewhere in main.
        if not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func.id if isinstance(node.value.func, ast.Name) else "?"
        out.append((fn, node, main_node))
    return out


def test_every_search_mode_is_dispatched():
    src = PIPELINE.read_text(encoding="utf-8")
    for mode in MODES:
        assert f'"{mode}"' in src, f"BP_SEARCH={mode} has no dispatch branch"


def test_dispatch_has_a_branch_per_search_backend():
    calls = _search_calls()
    assert len(calls) >= len(MODES) - 1, (
        f"expected a `cr = ...` per search mode, found {len(calls)}")


@pytest.mark.parametrize("idx", range(6))
def test_no_search_branch_uses_an_unbound_local(idx):
    """THE REGRESSION: a branch must not read a name a later branch defines."""
    calls = _search_calls()
    if idx >= len(calls):
        pytest.skip("fewer dispatch branches than parametrized")
    fn, node, main_node = calls[idx]
    bound = _names_bound_before(main_node, node.lineno)
    import builtins

    free = set()
    for kw in node.value.keywords:
        for n in ast.walk(kw.value):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                free.add(n.id)
    unbound = {n for n in free
               if n not in bound and not hasattr(builtins, n)}
    assert not unbound, (
        f"`cr = {fn}(...)` reads {sorted(unbound)}, which nothing binds before "
        f"line {node.lineno}. This is the BP_SEARCH=multi crash."
    )


def test_multi_branch_seeds_restart_zero():
    """multi's 'never worse than single-start' guarantee depends on the LLM seed."""
    src = PIPELINE.read_text(encoding="utf-8")
    multi = src[src.index('if search_mode == "multi"'):]
    multi = multi[:multi.index("elif search_mode")]
    assert "propose_seed" in multi, "multi must seed restart 0 like cmaes does"
    assert "seed_theta=seed_theta" in multi


# --------------------------------------------------------------------------- #
# --policy routing: a Windows path is a brax pickle, not an adapter kind "c"
# --------------------------------------------------------------------------- #
def _ns(policy):
    import types
    return types.SimpleNamespace(policy=policy, obs_norm=None, activation=None,
                                 action_scale=None, action_bias=None, squash=False,
                                 weight_layout=None, mean_dim=None)


def test_windows_pkl_path_routes_to_brax_loader(tmp_path, monkeypatch):
    import pickle
    pytest.importorskip("jax")
    from breakpoint_mjx import pipeline

    pkl = tmp_path / "frozen_go1_policy.pkl"
    pkl.write_bytes(pickle.dumps({"w": 1}))
    seen = {}

    def fake_make_policy_fn(params=None, target=None, **kw):
        seen["params"], seen["target"] = params, target
        return "brax-fn"

    monkeypatch.setattr(pipeline, "make_policy_fn", fake_make_policy_fn)
    win = str(pkl).replace("\\", "/")                   # e.g. C:/Users/.../x.pkl
    assert pipeline._maybe_adapter_policy(None, _ns(win), target="T") == "brax-fn"
    assert seen == {"params": {"w": 1}, "target": "T"}
    assert pipeline._maybe_adapter_policy(None, _ns("brax:" + win), target="T") == "brax-fn"


def test_missing_brax_path_is_a_clear_error():
    pytest.importorskip("jax")
    from breakpoint_mjx import pipeline

    with pytest.raises(SystemExit, match="no brax params pickle"):
        pipeline._maybe_adapter_policy(None, _ns("C:/nope/does_not_exist.pkl"))


def test_main_line_buffers_stdout():
    src = PIPELINE.read_text(encoding="utf-8")
    main = src[src.index("def main("):]
    assert "sys.stdout.reconfigure(line_buffering=True)" in main[:600]


def test_replay_stage_runs_after_cmc_and_before_render():
    src = PIPELINE.read_text(encoding="utf-8")
    main = src[src.index("def main("):]
    assert main.index("_cmc_stage(") < main.index("replay_stage(") < main.index(
        "render_failure(")
