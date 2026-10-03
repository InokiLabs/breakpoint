"""Stage checkpoints (breakpoint_mjx/stagecache.py): hash invalidation, atomic writes,
family round-trip, low-memory switch, memory preflight. Pure stdlib, no JAX."""
import json
import os

import pytest

from breakpoint_mjx import stagecache
from breakpoint_mjx.schemas import FailureFamily


def _fam(i=1, **kw):
    base = dict(id=f"fam-{i}", name=f"n{i}", member_ids=["a", "b"], representative_id="a",
                triggering_parameters={"x": 1.0}, mean_failure_score=0.9,
                mean_transfer_score=0.5, priority_rank=i)
    base.update(kw)
    return FailureFamily(**base)


def _failures(tmp_path, content="[1, 2, 3]"):
    p = tmp_path / "failures.json"
    p.write_text(content)
    return p


def test_key_changes_with_failures_and_budget(tmp_path):
    fp = _failures(tmp_path)
    k = stagecache.run_key(fp, {"budget": "smoke", "n_families": 8})
    assert k == stagecache.run_key(fp, {"n_families": 8, "budget": "smoke"})  # order-free
    assert k != stagecache.run_key(fp, {"budget": "standard", "n_families": 8})
    assert k != stagecache.run_key(fp, {"budget": "smoke", "n_families": 9})
    fp.write_text("[1, 2, 4]")
    assert k != stagecache.run_key(fp, {"budget": "smoke", "n_families": 8})


def test_stage_invalidated_when_failures_or_budget_change(tmp_path):
    fp = _failures(tmp_path)
    budget = {"budget": "smoke"}
    k = stagecache.run_key(fp, budget)
    stagecache.save_stage(tmp_path, "cmc", k, [_fam(1)], extra={"cmc_ran": True})
    name, d = stagecache.load_latest(tmp_path, k)
    assert name == "cmc" and d["extra"]["cmc_ran"] is True
    # a new search result -> the stage is ignored
    fp.write_text("[9]")
    assert stagecache.load_latest(tmp_path, stagecache.run_key(fp, budget)) == (None, None)
    # a different budget -> ignored too
    fp.write_text("[1, 2, 3]")
    assert stagecache.load_latest(
        tmp_path, stagecache.run_key(fp, {"budget": "standard"})) == (None, None)
    # and the original combination still resumes
    assert stagecache.load_latest(tmp_path, k)[0] == "cmc"


def test_latest_stage_wins_and_torn_stage_is_skipped(tmp_path):
    k = "abc"
    stagecache.save_stage(tmp_path, "cluster", k, [_fam(1)])
    stagecache.save_stage(tmp_path, "cmc", k, [_fam(1)])
    assert stagecache.load_latest(tmp_path, k)[0] == "cmc"
    stagecache.save_stage(tmp_path, "replay", k, [_fam(1)],
                          extra={"replay_summary": {"status": "ran"}})
    assert stagecache.load_latest(tmp_path, k)[0] == "replay"
    # a torn/garbage replay file falls back to the previous valid stage
    with open(stagecache.stage_path(tmp_path, "replay"), "w") as fh:
        fh.write('{"stage": "replay", "key": "ab')
    assert stagecache.load_latest(tmp_path, k)[0] == "cmc"
    stagecache.clear_stages(tmp_path)
    assert stagecache.load_latest(tmp_path, k) == (None, None)


def test_families_round_trip_with_numpy_values(tmp_path):
    np = pytest.importorskip("numpy")
    f = _fam(2, minimal_cause=[{"dim": "x", "value": np.float32(0.25), "direction": "high"}],
             cause_certified=True, replay={"cp_lo": np.float64(0.6), "k": np.int64(7)},
             mean_theta={"x": np.float32(1.5)})
    stagecache.save_stage(tmp_path, "replay", "k", [f])
    _, d = stagecache.load_latest(tmp_path, "k")
    (g,) = stagecache.families_from_dicts(d["families"])
    assert g.id == "fam-2" and g.cause_certified is True
    assert g.minimal_cause[0]["value"] == pytest.approx(0.25)
    assert g.replay["k"] == 7
    # unknown keys from a newer writer are dropped, not fatal
    assert stagecache.families_from_dicts([{**d["families"][0], "future": 1}])[0].id == "fam-2"


def test_atomic_write_leaves_old_file_on_failure(tmp_path):
    p = tmp_path / "families.json"
    stagecache.write_json_atomic(p, {"ok": 1})

    class Boom:
        pass

    def bad(o):
        raise RuntimeError("serializer died mid-write")

    # a failure while writing must leave the previous complete file and no tmp litter
    orig = stagecache._jsonable
    stagecache._jsonable = bad
    try:
        with pytest.raises(RuntimeError):
            stagecache.write_json_atomic(p, {"ok": 2, "x": Boom()})
    finally:
        stagecache._jsonable = orig
    assert json.loads(p.read_text()) == {"ok": 1}
    assert [n for n in os.listdir(tmp_path) if n.startswith(".tmp-")] == []


def test_unknown_stage_rejected(tmp_path):
    with pytest.raises(ValueError):
        stagecache.save_stage(tmp_path, "render", "k", [])


def test_lowmem_switch():
    plenty = {"free_ram_mb": 64000}
    tight = {"free_ram_mb": stagecache.LOWMEM_AUTO_MB - 1}
    assert stagecache.lowmem_enabled({"BP_LOWMEM": "1"}, plenty)[0] is True
    assert stagecache.lowmem_enabled({"BP_LOWMEM": "0"}, tight)[0] is False
    assert stagecache.lowmem_enabled({}, tight)[0] is True
    assert stagecache.lowmem_enabled({}, plenty)[0] is False
    assert stagecache.lowmem_enabled({}, {"free_ram_mb": None})[0] is False


def test_memory_warning_thresholds():
    ok = {"free_ram_mb": stagecache.DEMO_PEAK_RSS_MB + 1,
          "free_commit_mb": stagecache.DEMO_PEAK_COMMIT_MB + 1}
    assert stagecache.memory_warning(ok) is None
    w = stagecache.memory_warning({**ok, "free_ram_mb": 100})
    assert w and "free RAM 100 MB" in w and "--low-mem" in w
    assert "free commit" in stagecache.memory_warning({**ok, "free_commit_mb": 5})


def test_free_memory_reads_this_machine():
    m = stagecache.free_memory()
    assert m["free_ram_mb"] is None or m["free_ram_mb"] > 0
    assert set(m) >= {"free_ram_mb", "total_ram_mb", "free_commit_mb", "commit_limit_mb"}


def test_crash_point_is_inert_unless_named():
    stagecache.crash_point("render", environ={})
    stagecache.crash_point("render", environ={"BP_TEST_CRASH_AT": "cmc"})
