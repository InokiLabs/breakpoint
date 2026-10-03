from breakpoint_mjx.schemas import BreakPointReport, FailureFamily
from breakpoint_mjx.report import render_html


def _make_report():
    fam0 = FailureFamily(
        id="f0",
        name="high-latency collapse",
        member_ids=["r0", "r1", "r2"],
        representative_id="r0",
        triggering_parameters={"latency_steps": 3.5, "friction_floor": 0.4},
        mean_failure_score=0.72,
        mean_transfer_score=0.65,
        priority_rank=2,
    )
    fam1 = FailureFamily(
        id="f1",
        name="slippery floor slip",
        member_ids=["r3", "r4"],
        representative_id="r3",
        triggering_parameters={"friction_floor": 0.12, "torso_mass_add_kg": 1.0},
        mean_failure_score=0.88,
        mean_transfer_score=0.80,
        priority_rank=1,
    )
    return BreakPointReport(
        run_id="test-run",
        policy_descriptor="Go1JoystickFlatTerrain",
        comparison={"random_top20": 0.3, "directed_top20": 0.7},
        families=[fam0, fam1],
        mp4_paths={"f0": "f0.mp4", "f1": "f1.mp4"},
    )


def test_priority_order():
    """Priority-1 family must appear before priority-2 family in the HTML output."""
    report = _make_report()
    html = render_html(report)
    idx_f1 = html.index("f1.mp4")
    idx_f0 = html.index("f0.mp4")
    assert idx_f1 < idx_f0, (
        f"Priority-1 family (f1.mp4 at {idx_f1}) should appear before "
        f"priority-2 family (f0.mp4 at {idx_f0})"
    )


def test_mp4_filenames_present():
    html = render_html(_make_report())
    assert "f0.mp4" in html
    assert "f1.mp4" in html


def test_directed_keyword_present():
    html = render_html(_make_report())
    assert "directed" in html


def test_explanation_included():
    report = _make_report()
    html = render_html(report, explanations={"f0": "because latency"})
    assert "because latency" in html


def test_no_explanation_when_absent():
    report = _make_report()
    html = render_html(report, explanations={})
    assert "because latency" not in html


def test_stats_always_on_shows_both_ranking_views():
    """T2: the failure-rate/ranking display is always-on. Both the expected-value and the
    severity (worst-case-first) views appear."""
    html = render_html(_make_report())
    assert "By expected risk (severity &times; probability)" in html
    assert "By severity (worst-case first)" in html


def test_bare_render_labels_probability_as_relative_share():
    """T2 fix: on the stats=None fallback, per-family probability is a relative member-fraction
    (sums to 1), so it must be labelled as a share, not an absolute failure rate."""
    html = render_html(_make_report())   # no stats -> fallback
    assert "Relative share among families" in html
    assert "Failure rate (probability)" not in html


def test_render_with_real_stats_uses_absolute_failure_rate_label():
    from breakpoint_mjx.stats import compute_report_stats
    report = _make_report()
    stats = compute_report_stats(report.families, n_total=100, n_failures=5, threshold=0.5)
    html = render_html(report, stats=stats)
    assert "Share of search-selected failures" in html
    assert "Relative share among families" not in html
    assert "Overall failure rate" in html


# --------------------------------------------------------------------------- #
# Skeptic-grade evidence: sections, contiguous ranks, escaping, cause tables
# --------------------------------------------------------------------------- #
def _evidence_report():
    rep = _make_report()
    f0, f1 = rep.families
    # f1: certified, replay reproduced, evidence videos.
    f1.cause_certified = True
    f1.minimal_cause = [{"dim": "friction_floor", "value": 0.2, "direction": "low"}]
    f1.margin = "fails when friction_floor < 0.25 <script>"
    f1.replay = {"k": 8, "n": 8, "cp_lo": 0.75, "bar": 0.5, "reproduces": True,
                 "nominal_twin": {"fell": False}}
    f1.evidence_videos = {"counterexample": "f1-mincause.mp4", "nominal": "f1-nominal.mp4",
                          "source": "c_mujoco_full_collision_replay"}
    f1.mjx_min_floor_height = -0.31
    # f0: CMC could not certify it.
    f0.cause_certified = False
    f0.dominant_dims = [[1.0, "latency_steps", "high"], [0.5, "friction_floor", "low"]]
    f0.priority_rank = 111
    # f2: certified in MJX, demoted by the replay.
    from breakpoint_mjx.schemas import FailureFamily
    f2 = FailureFamily(id="f2", name="heavy torso", member_ids=["r5"], representative_id="r5",
                       triggering_parameters={"torso_mass_add_kg": 2.0},
                       mean_failure_score=0.9, mean_transfer_score=0.7, priority_rank=3,
                       cause_certified=True,
                       minimal_cause=[{"dim": "torso_mass_add_kg", "value": 1.5,
                                       "direction": "high"}],
                       replay={"k": 3, "n": 4, "cp_lo": 0.25, "bar": 0.5,
                               "reproduces": False})
    rep.families.append(f2)
    rep.mp4_paths["f2"] = "f2.mp4"
    return rep


def test_uncertified_family_only_in_unconfirmed_section_without_rank_or_severity():
    import re

    html = render_html(_evidence_report(),
                       nominal={"friction_floor": 0.6, "torso_mass_add_kg": 0.0})
    unconf = html.index("Unconfirmed (not reproducible under seed noise)")
    assert "uncertified: latency_steps, friction_floor" in html[unconf:]
    assert "high-latency collapse" not in html          # causal name stripped
    assert "#111" not in html and "Priority rank" not in html
    # Only the confirmed family is ranked, starting at #1, and nothing else is.
    ranks = [int(m) for m in re.findall(r"<h2>#(\d+) &mdash;", html)]
    assert ranks == list(range(1, len(ranks) + 1)) == [1]
    # The unconfirmed and demoted families appear in no ranking list.
    head = html[:html.index("<section>")]
    assert "latency_steps" not in head and "heavy torso" not in head


def test_demoted_family_is_in_its_own_section_with_badge():
    html = render_html(_evidence_report())
    sec = html.index("Did not reproduce outside MJX at the certification bar</h2>")
    assert html.index("heavy torso") > sec
    assert "reproduces in C-MuJoCo full-collision: <b>3/4</b>" in html
    assert "reproduces in C-MuJoCo full-collision: <b>8/8</b>" in html
    assert "did not reproduce outside MJX</b>" in html   # the f2 badge (no verdict field)


def test_cause_table_lists_only_certified_dims_with_nominal_to_counterexample():
    html = render_html(_evidence_report(),
                       nominal={"friction_floor": 0.6, "torso_mass_add_kg": 0.0})
    s = html[html.index("slippery floor slip"):html.index("Did not reproduce outside MJX at")]
    assert "<code>friction_floor</code></td><td>0.6</td>" in s
    assert "<b>0.2</b>" in s
    assert "torso_mass_add_kg" not in s                 # a representative-only dim


def test_counterexample_and_nominal_twin_videos_side_by_side():
    html = render_html(_evidence_report())
    s = html[html.index("slippery floor slip"):]
    assert 'src="f1-mincause.mp4"' in s and 'src="f1-nominal.mp4"' in s
    assert "<details><summary>Original search representative" in s
    assert "independent C-MuJoCo full-collision replay" in s


def test_margin_and_label_strings_are_html_escaped():
    import re

    html = render_html(_evidence_report())
    assert "<script>" not in html
    assert "friction_floor &lt; 0.25 &lt;script&gt;" in html
    for m in re.findall(r"<b>Safety margin:</b>([^\n]*?)</li>", html):
        assert "<" not in m


def test_floor_flag_when_mjx_torso_below_floor():
    html = render_html(_evidence_report())
    assert "Physics fidelity flag" in html and "31 cm below the floor plane" in html


def test_fidelity_section_states_search_scene_and_replay():
    ev = {"search_collision_model": "feet-only",
          "replay": {"status": "ran", "k": 8, "n_replayed": 2, "n_reproduced": 1,
                     "n_demoted": 1, "seconds": 42.0,
                     "collision_model": {"source": "playground_scene",
                                         "scene": "scene_mjx_fullcollisions_flat_terrain.xml"}},
          "surrogate_fit": {"r2": None, "hidden_reason": "not shown: 192 labeled worlds"}}
    html = render_html(_evidence_report(), evidence=ev)
    assert "<b>Search scene:</b> feet-only." in html
    assert "1 of 2 families reproduced; 1 did not clear the bar" in html
    assert "not shown: 192 labeled worlds" in html
    off = render_html(_evidence_report(), evidence={"replay": {"status": "unavailable",
                                                               "reason": "no full model"}})
    assert "Independent replay not run:</b> no full model" in off


def test_imbalanced_envelope_message_does_not_contradict_nominal_passes():
    from breakpoint_mjx.envelope import NO_BOX_IMBALANCED, annotate_label_balance

    env = {"envelope": {"friction_floor": [0.6, 0.6]},
           "summary": "no failure-free box found around nominal", "safe_volume_fraction": 0.0}
    annotate_label_balance(env, 192, 179, nominal_check={"n": 16, "fails": 0})
    assert env["summary"] == NO_BOX_IMBALANCED
    html = render_html(_make_report(), envelope=env)
    assert "no failure-free box found at this budget and label balance" in html
    assert "nominal parameters: 16/16 seeds pass" in html
    assert "No failure-free box found around nominal" not in html
    # Balanced labels keep the plain message, with the nominal count alongside.
    env2 = {"envelope": {"friction_floor": [0.6, 0.6]}, "summary": "x"}
    annotate_label_balance(env2, 400, 100, nominal_check={"n": 16, "fails": 1})
    assert env2["summary"] == "x" and "15/16 seeds pass" in env2["summary_long"]


def test_report_says_when_nothing_is_ranked_and_flag_matches_video_source():
    rep = _evidence_report()
    rep.families[1].replay = {"k": 6, "n": 8, "cp_lo": 0.46, "cp_hi": 0.93, "bar": 0.5,
                              "reproduces": False, "verdict": "inconclusive"}
    html = render_html(rep)
    assert "Ranked failure families</h2><p>None." in html
    assert "inconclusive at this seed budget" in html and "CP interval 0.46-0.93" in html
    import re
    assert re.findall(r"<h2>#(\d+) &mdash;", html) == []
    # the unconfirmed family has no replay videos, so its flag must not claim any
    rep.families[0].mjx_min_floor_height = -0.3
    html = render_html(rep)
    unconf = html[html.index("Unconfirmed (not reproducible"):]
    assert "Physics fidelity flag" in unconf
    assert "counterexample videos above" not in unconf
