"""CPU tests for the autolytic-gain measurement (pure numpy, no sim)."""
import numpy as np
import pytest

from breakpoint_mjx.autolytic import (
    classify, collapse_step, gain_window, rank_worlds, step_input_power,
    step_work, summarize_batch,
)


def test_step_work_shapes_and_value():
    f = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert np.allclose(step_work(f, v), [3.0, 8.0])
    assert step_work(f[0], v[0]) == pytest.approx(3.0)


def test_step_work_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        step_work(np.zeros((3, 2)), np.zeros((3, 4)))


def test_step_input_power_sums_wrench_dot_velocity():
    w = np.zeros((2, 3, 6))
    v = np.zeros((2, 3, 6))
    w[0, 1, :3] = [1.0, 0.0, 0.0]
    v[0, 1, :3] = [2.0, 0.0, 0.0]
    assert np.allclose(step_input_power(w, v), [2.0, 0.0])


def test_collapse_step_finds_first_fall():
    assert collapse_step([False, False, True, True]) == 2
    assert collapse_step([False, False]) is None


# ---------------------------------------------------------------------------
# gain_window -- the sign discipline the refuter flagged
# ---------------------------------------------------------------------------

def test_gain_is_ratio_of_excess_work_to_input():
    wi = np.full(10, 3.0)
    wt = np.full(10, 1.0)          # excess 2.0/step over 10 steps = 20
    pin = np.full(10, 0.5)         # input 5.0
    gain, G, E_in, (lo, hi) = gain_window(wi, wt, pin)
    assert (lo, hi) == (0, 10)
    assert G == pytest.approx(20.0) and E_in == pytest.approx(5.0)
    assert gain == pytest.approx(4.0)


def test_window_ends_at_collapse_not_episode_end():
    """Post-collapse steps must not enter G: a fallen robot does little work, which
    would drag the gain of the WORST failures negative."""
    wi = np.concatenate([np.full(5, 3.0), np.zeros(5)])   # collapses at step 5
    wt = np.full(10, 1.0)
    pin = np.full(10, 0.5)
    fallen = [False] * 5 + [True] * 5

    gain_full, _, _, _ = gain_window(wi, wt, pin)                       # naive, whole ep
    gain_win, G, _, (lo, hi) = gain_window(wi, wt, pin, fallen_flags=fallen)

    assert (lo, hi) == (0, 5)
    assert G == pytest.approx(10.0) and gain_win == pytest.approx(4.0)
    assert gain_full < gain_win  # the naive window is the trap this guards against


def test_window_steps_caps_to_pre_collapse_span():
    wi = np.concatenate([np.ones(10), np.full(10, 5.0), np.zeros(5)])
    wt = np.ones(25)
    pin = np.full(25, 1.0)
    fallen = [False] * 20 + [True] * 5
    _, G, E_in, (lo, hi) = gain_window(wi, wt, pin, fallen_flags=fallen, window_steps=10)
    assert (lo, hi) == (10, 20)             # G integrates the 10 steps before collapse
    assert G == pytest.approx(40.0)
    # E_in counts everything injected up to the window's end, not just inside it.
    assert E_in == pytest.approx(20.0)


def test_impulse_before_the_window_still_counts_as_input():
    """A gait-timed push is one impulse and usually lands long before a late window.

    Counting E_in only inside the window reported 0 and silently discarded the row
    (this is what the first real probe run did to a third of its worlds).
    """
    wi = np.concatenate([np.ones(10), np.full(10, 4.0)])
    wt = np.ones(20)
    pin = np.zeros(20)
    pin[2] = 5.0                              # the push, at step 2
    gain, G, E_in, (lo, hi) = gain_window(wi, wt, pin, window_steps=10)
    assert (lo, hi) == (10, 20)
    assert E_in == pytest.approx(5.0), "the impulse must not vanish"
    assert G == pytest.approx(30.0) and gain == pytest.approx(6.0)


def test_bind_step_excludes_pre_attack_steps():
    wi = np.concatenate([np.full(5, 99.0), np.full(5, 3.0)])  # noise before the attack
    wt = np.ones(10)
    pin = np.concatenate([np.zeros(5), np.full(5, 0.5)])
    _, G, E_in, (lo, hi) = gain_window(wi, wt, pin, bind_step=5)
    assert (lo, hi) == (5, 10)
    assert G == pytest.approx(10.0) and E_in == pytest.approx(2.5)


def test_dust_input_yields_none_not_huge_gain():
    gain, _, E_in, _ = gain_window(np.ones(5), np.zeros(5), np.zeros(5))
    assert E_in == 0.0 and gain is None


def test_gain_window_rejects_ragged_input():
    with pytest.raises(ValueError):
        gain_window(np.ones(5), np.ones(4), np.ones(5))


# ---------------------------------------------------------------------------
# classify / rank / summarize
# ---------------------------------------------------------------------------

def test_classify_bands():
    assert classify(-5.0) == "damped"        # less work than the twin: opposite sign
    assert classify(0.0) == "sledgehammer"
    assert classify(0.4) == "sledgehammer"
    assert classify(1.0) == "sledgehammer"      # boundary belongs to sledgehammer
    assert classify(2.0) == "amplified"
    assert classify(9.0) == "autolytic"
    assert classify(None) == "unmeasured"


def test_rank_prefers_gain_then_failure_and_keeps_unmeasured():
    order = rank_worlds([1.0, None, 8.0, 8.0], [0.9, 0.99, 0.2, 0.7])
    assert order[0] == 3 and order[1] == 2   # equal gain, higher failure first
    assert order[-1] == 1                    # unmeasured sinks, is not dropped
    assert len(order) == 4


def test_summarize_batch_reports_headline_fraction():
    s = summarize_batch([5.0, 6.0, 0.5, None])
    assert s["n"] == 4
    assert s["counts"] == {"autolytic": 2, "amplified": 0, "sledgehammer": 1,
                           "damped": 0, "unmeasured": 1}
    assert s["autolytic_fraction"] == pytest.approx(0.5)
    assert s["measured_fraction"] == pytest.approx(0.75)
    assert s["peak_gain"] == pytest.approx(6.0)


def test_summarize_empty_batch():
    s = summarize_batch([])
    assert s["n"] == 0 and s["peak_gain"] is None and s["autolytic_fraction"] == 0.0
