"""Observation adversaries (breakpoint_mjx/obs_adversary.py) -- pure numpy, no JAX."""
import numpy as np

from breakpoint_mjx.obs_adversary import obs_sensor_mask, perturb_obs, set_obs_non_sensor


class _Env:
    pass


def test_undeclared_env_perturbs_every_channel():
    assert obs_sensor_mask(_Env(), 8) is None


def test_go1_layout_masks_last_act_and_command():
    env = set_obs_non_sensor(_Env(), [(33, 12), (45, 3)])
    m = obs_sensor_mask(env, 48)
    assert m[:33].all() and not m[33:].any()


def test_negative_start_counts_from_the_end():
    env = set_obs_non_sensor(_Env(), [(9, 3), (-(12 + 4), 12 + 4)])   # phase humanoid
    m = obs_sensor_mask(env, 52)
    assert not m[9:12].any() and not m[36:].any()
    assert m[:9].all() and m[12:36].all()


def test_frame_repeats_the_slices_in_every_history_frame():
    env = set_obs_non_sensor(_Env(), [(4, 3), (19, 12)], frame=31)   # barkour, 15 frames
    m = obs_sensor_mask(env, 15 * 31)
    for f in range(15):
        fr = m[f * 31:(f + 1) * 31]
        assert not fr[4:7].any() and not fr[19:31].any()
        assert fr[:4].all() and fr[7:19].all()
    assert int((~m).sum()) == 15 * 15


def test_nominal_returns_the_clean_obs_exactly():
    clean = np.array([0.1, -2.0, 3.5, 7.0])
    out, held = perturb_obs(np, clean, 0.0, 0.0, np.zeros(4, bool), clean * 9.0, None)
    assert np.array_equal(out, clean) and np.array_equal(held, clean)


def test_dropout_holds_the_last_delivered_reading_not_zero():
    clean = np.array([1.0, 2.0, 3.0])
    gauss = np.array([0.5, 0.5, 0.5])
    held = np.array([9.0, 9.0, 9.0])
    out, new_held = perturb_obs(np, clean, gauss, 0.0, np.array([True, False, True]), held,
                                None)
    assert out.tolist() == [9.0, 2.5, 9.0]
    assert np.array_equal(new_held, out)


def test_non_sensor_channels_pass_through_clean():
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, True, False, False])            # last two: last_act / command
    out, _ = perturb_obs(np, clean, np.full(4, 0.3), np.full(4, 0.1),
                         np.array([False, True, True, False]), np.zeros(4), mask)
    assert np.allclose(out, [1.4, 0.0, 3.0, 4.0])        # dropped sensor holds 0.0 = held
    assert out[2] == 3.0 and out[3] == 4.0               # non-sensor: exact, untouched


def test_a_held_channel_repeats_across_consecutive_drops():
    clean = np.array([1.0])
    held = np.array([0.0])
    seq = []
    for t, drop in enumerate([False, True, True, False]):
        out, held = perturb_obs(np, clean + t, 0.0, 0.0, np.array([drop]), held, None)
        seq.append(float(out[0]))
    assert seq == [1.0, 1.0, 1.0, 4.0]
