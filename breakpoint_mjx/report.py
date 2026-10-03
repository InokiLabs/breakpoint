"""Pure static-report assembly: one HTML page, families ordered by priority_rank.

When `media_dir` is given (write_report does this by default), each family's mp4 is
base64-embedded as a data URI so the report is a single self-contained file that
plays everywhere: inline in Colab, downloaded, or shared. Without it, the report
references the mp4s by relative filename (smaller, but must sit beside them).
"""
from __future__ import annotations

import base64
import os

from breakpoint_mjx.schemas import BreakPointReport


def _video_src(mp4_name: str, media_dir) -> str:
    if mp4_name and media_dir:
        full = os.path.join(media_dir, mp4_name)
        if os.path.exists(full):
            with open(full, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            return f"data:video/mp4;base64,{b64}"
    return mp4_name


def _validity_html(f, validity_configs) -> str:
    """Render a family's MEASURED cross-config validity (T1), if it was validated.

    Leads with the VALIDITY fraction -- the exact quantity families are ranked by (see
    render_html) -- then shows the per-config reproduction line and, as a secondary
    figure, the validity-weighted combined evidence. Empty string when the family was
    not validated (BP_VALIDATE off), so the report degrades cleanly to the
    plausibility-only view.
    """
    if f.validity is None:
        return ""
    # Local import keeps report.py's only hard deps stdlib (crossval is import-safe).
    from breakpoint_mjx.crossval import combined_transfer

    repro = f.reproduction or []
    n_ok = sum(1 for r in repro if r)
    names = validity_configs or [f"cfg{i}" for i in range(len(repro))]
    cells = "".join(
        f'<span title="{_e(names[i] if i < len(names) else i)}" '
        f'style="color:{"#128a12" if ok else "#b00"}">'
        f'{"&#10003;" if ok else "&#10007;"}</span> '
        f'<small>{_e(names[i] if i < len(names) else i)}</small>&nbsp;&nbsp;'
        for i, ok in enumerate(repro)
    )
    combined = combined_transfer(f.validity, f.mean_transfer_score)
    return (
        f'<li><b>Validity (measured &mdash; ranked by this):</b> '
        f'{f.validity:.2f} &mdash; {n_ok}/{len(repro)} configs reproduce</li>'
        f'<li><b>Per-config reproduction:</b><br>{cells}</li>'
        f'<li>Validity-weighted evidence: {combined:.3f} '
        f'<small>(0.75&middot;validity + 0.25&middot;plausibility {f.mean_transfer_score:.2f})</small></li>'
    )


def _expected_risk_ol(ranking, plabel) -> str:
    lis = "".join(
        f'<li>{_e(d["name"])} &mdash; sev&times;prob <b>{d["sev_prob"]:.3f}</b> '
        f'<small>(severity {d["severity_score"]:.2f} &middot; {plabel} {d["probability"]:.3f}, '
        f'95% CI {d["prob_lo"]:.3f}&ndash;{d["prob_hi"]:.3f})</small></li>'
        for d in ranking
    )
    return f'<ol>{lis}</ol>'


def _severity_ol(ranking, plabel) -> str:
    lis = "".join(
        f'<li>{_e(d["name"])} &mdash; severity <b>{d["severity_score"]:.3f}</b> '
        f'<small>({_e(d["severity"])}, {plabel} {d["probability"]:.3f})</small></li>'
        for d in ranking
    )
    return f'<ol>{lis}</ol>'


_REASON_LABELS = {
    "nonfinite": "non-finite state",
    "below_floor": "torso below the floor plane",
    "penetration": "contact penetration",
    "blowup": "acceleration blow-up",
}


def _physical_validity_html(stats) -> str:
    """Top-of-report physical-validity line (the sim-invalid gate, failure.py).

    Rendered only when the stats bundle carries ``validity`` (compute_report_stats was
    given the evaluated records). Flagged rollouts are simulator artifacts (torso below
    the floor, deep contact penetration, acceleration blow-up, NaN), scored 0 and excluded
    from every rate and family below."""
    v = (stats or {}).get("validity")
    if not v:
        return ""
    if not v.get("gate_active"):
        return ('<p><b>Physical validity:</b> gate not active for this run (the '
                'trajectories carry no floor_height / penetration / qacc_max channels), '
                'so simulator artifacts were NOT screened out.</p>')
    n_bad = int(v.get("n_sim_invalid", 0))
    n_all = int(v.get("n_evaluated", 0))
    reasons = ", ".join(
        f"{_REASON_LABELS.get(k, k)} {c}" for k, c in (v.get("reasons") or {}).items() if c
    )
    reason_txt = f" Reasons: {reasons}." if reasons else ""
    return (
        f'<p><b>Physical validity:</b> {n_bad} of {n_all} evaluated worlds were excluded '
        f'as simulator artifacts (penetration / below-floor / blow-up).{reason_txt} '
        f'<small>Excluded worlds score 0 and are left out of every failure rate and '
        f'family below; see docs/validity.md for the thresholds.</small></p>'
    )


def envelope_is_zero_volume(envelope) -> bool:
    """True iff every axis of a ``safety_envelope`` box collapsed to a single point."""
    box = (envelope or {}).get("envelope") or {}
    if not box:
        return False
    return all(abs(float(hi) - float(lo)) <= 1e-12 for lo, hi in box.values())


def _envelope_html(envelope) -> str:
    """The M3 safety envelope (surrogate-predicted), when the caller passes it.

    A zero-volume box (every axis ``[nominal, nominal]``) is not an operating envelope,
    so it is NOT printed as one; the report says no failure-free box was found."""
    if not envelope:
        return ""
    from breakpoint_mjx.envelope import NO_BOX_IMBALANCED

    if envelope.get("summary") == NO_BOX_IMBALANCED:
        return (f'<h2>Safety envelope</h2><p>{_e(envelope.get("summary_long") or NO_BOX_IMBALANCED)}.'
                f' <small>With most searched worlds labeled as failures the surrogate '
                f'predicts "unsafe" almost everywhere, so an empty box here reflects the '
                f'search budget and label balance, not a measured absence of safe '
                f'operating points.</small></p>')
    if envelope_is_zero_volume(envelope):
        return ('<h2>Safety envelope</h2><p>No failure-free box found around nominal: '
                'the surrogate could not certify any neighbourhood of the nominal '
                'parameters as safe at the requested tolerance.'
                + (f' <small>{_e(envelope["summary_long"])}.</small>'
                   if envelope.get("summary_long") else "") + '</p>')
    from html import escape

    line = envelope.get("summary_long") or envelope.get("summary", "")
    return (f'<h2>Safety envelope</h2><p>{escape(str(line))} '
            f'<small>(surrogate-predicted, safe-volume fraction '
            f'{float(envelope.get("safe_volume_fraction", 0.0)):.3g}; not a simulator '
            f'certificate unless one is shown).</small></p>')


def _stats_summary_html(stats) -> str:
    """Top-of-report always-on stats (T2, Workstream 1b). Two complementary family views:
    EXPECTED-VALUE risk (severity x probability, the headline) AND SEVERITY (worst-case
    first) -- because expected-value risk does not guarantee a rare catastrophic family
    outranks a common mild one. Additive; empty string if no ranking exists.

    On the bare-render fallback (``relative`` set), ``probability`` is a relative share among
    families (sums to 1), so it is labelled "share" rather than an absolute failure rate."""
    if not stats:
        return ""
    relative = bool(stats.get("relative"))
    plabel = "share" if relative else "search share"
    parts = []
    ov = stats.get("overall")
    if ov and ov.get("n"):
        parts.append(
            f'<p><b>Overall failure rate among search-evaluated worlds:</b> {ov["rate"]:.3f} '
            f'(95% CI {ov["lo"]:.3f}&ndash;{ov["hi"]:.3f}) '
            f'over {ov["n"]} {"physically valid " if stats.get("validity") else ""}'
            f'evaluated envs. <small>These worlds were chosen by the search, an '
            f'adversarial process that concentrates on failures, so this is NOT a deployment '
            f'failure probability; see the operational risk section when present.</small></p>'
        )
    cov = stats.get("coverage")
    if cov:
        from html import escape

        from breakpoint_mjx.coverage import coverage_sentence
        parts.append(f'<p><b>Campaign coverage:</b> {escape(coverage_sentence(cov))}</p>')
    ranking = stats.get("ranking") or []
    if ranking:
        parts.append(
            '<h2>By expected risk (severity &times; probability)</h2>'
            '<p><small>Expected-value risk = probability &times; severity (the standard risk '
            'metric). This does NOT guarantee a rare catastrophic family outranks a common '
            'mild one &mdash; see the severity view below.</small></p>'
        )
        parts.append(_expected_risk_ol(ranking, plabel))
    sev_ranking = stats.get("severity_ranking") or []
    if sev_ranking:
        parts.append(
            '<h2>By severity (worst-case first)</h2>'
            '<p><small>Worst outcomes first, regardless of frequency.</small></p>'
        )
        parts.append(_severity_ol(sev_ranking, plabel))
    return "".join(parts)


def _margins_html(f) -> str:
    """Per-family safety margin + axis attribution (T2, Workstream 1c), when annotated by
    the BP_MARGINS stage. Empty string otherwise (clean degrade)."""
    out = []
    if getattr(f, "margin", None):
        out.append(f'<li><b>Safety margin:</b> {_e(f.margin)}</li>')
    attr = getattr(f, "attribution", None)
    if attr:
        top = sorted(attr.items(), key=lambda kv: kv[1], reverse=True)
        attr_str = ", ".join(f"{_e(k)} {v:.2f}" for k, v in top if v > 0)
        if attr_str:
            out.append(f'<li><b>Axis attribution:</b> {attr_str}</li>')
    return "".join(out)


def _operational_risk_html(o) -> str:
    """The BP_RISK block: a deployment failure probability with an anytime-valid CI."""
    if not o:
        return ""
    b = o.get("plain_mc_baseline") or {}
    dims = (o.get("dist") or {}).get("dims") or {}
    dist_line = ", ".join(f"{k} [{v[0]:.3g}, {v[1]:.3g}]" for k, v in dims.items())
    return (
        '<h2>Operational failure probability</h2>'
        f'<p><b>{o["estimate"]:.4f}</b> (95% anytime-valid CI '
        f'{o["lo"]:.4f}&ndash;{o["hi"]:.4f}) from {o["n_rollouts"]} fresh simulator '
        f'rollouts ({o["n_failures"]} failures). Method: {o["method"]}; defensive '
        f'mixture alpha={o["alpha"]:.2f} (importance weights &le; {o["w_max_bound"]:.2f}, '
        f'max observed {o["w_max_observed"]:.2f}), ESS {o["ess"]:.0f}.</p>'
        f'<p>Plain Monte Carlo baseline on the {b.get("n", 0)} defensive p-draws: '
        f'{b.get("lo", 0):.4f}&ndash;{b.get("hi", 1):.4f} (Clopper-Pearson).</p>'
        f'<p><small>Declared operational distribution ({(o.get("dist") or {}).get("kind")}, '
        f'source: {(o.get("dist") or {}).get("source")}): {dist_line}. '
        f'{o.get("caveat", "")}</small></p>'
    )


_ODD_CLASS_TEXT = {"in_domain_risk": "in-domain risk", "in_domain_low": "in-domain, low",
                   "outside_odd": "outside your operating domain"}


def _g(x) -> str:
    return "0" if x == 0 else f"{float(x):.2g}"


def _odd_risk_row(f) -> str:
    """Per-family 'Operational risk (declared ODD)' row (famrisk.py). Empty when absent."""
    o = getattr(f, "odd_risk", None)
    if not o:
        return ""
    cls = _ODD_CLASS_TEXT.get(o.get("class"), o.get("class"))
    if o.get("n"):
        cond = (f'{int(o["k"])}/{int(o["n"])} fresh conditional rollouts fail (CP '
                f'{o["cond_lo"]:.3f}&ndash;{o["cond_hi"]:.3f})')
    else:
        cond = "no rollouts spent (the region has zero mass under the declared domain)"
    if o.get("region_basis") == "relaxed":
        cond += (" &middot; the certified threshold lies just outside the domain, so this is "
                 "measured on the region widened by the minimal-cause bisection resolution")
    return (f'<li><b>Operational risk (declared ODD):</b> {_g(o["risk_lo"])}&ndash;'
            f'{_g(o["risk_hi"])} <small>[{_e(cls)}] &middot; region mass {_g(o["mass"])} '
            f'&times; {cond} &middot; SIMULATION</small></li>')


def _famrisk_table(fams) -> str:
    rows = "".join(
        f'<tr><td>{_e(f.name)}</td><td><code>{_e(f.odd_risk["region_str"])}</code></td>'
        f'<td>{_g(f.odd_risk["mass"])}</td>'
        f'<td>{int(f.odd_risk["k"])}/{int(f.odd_risk["n"])}</td>'
        f'<td>{_g(f.odd_risk["risk_lo"])}&ndash;{_g(f.odd_risk["risk_hi"])}</td>'
        f'<td>{_e(_ODD_CLASS_TEXT.get(f.odd_risk["class"], ""))}</td></tr>'
        for f in fams)
    return ('<table class="cause"><tr><th>Family</th><th>Minimal-cause region</th>'
            '<th>Mass under ODD</th><th>Fail k/n</th><th>Risk (95%)</th><th>Class</th>'
            f'</tr>{rows}</table>')


def _famrisk_html(families, summary) -> str:
    """'Operational risk by family (declared ODD)': in-domain families ordered by their
    upper risk bound, out-of-ODD families in a collapsed section. SIMULATION."""
    if not summary:
        return ""
    st = summary.get("status")
    head = '<h2>Operational risk by family (declared operating domain)</h2>'
    if st == "disabled":
        return ""
    if st in ("cmc_not_run", "error"):
        return (head + '<p>Not computed: ' + _e(summary.get("reason") or st)
                + '. Families below carry no per-family operational risk.</p>')
    src = _e(summary.get("odd_source", ""))
    if st == "no_certified":
        return (head + '<p>No family was certified by the minimal-cause stage at this '
                'budget, so no per-family operational risk is reported (it needs a '
                'certified cause region). Raise the budget (<code>--budget standard</code>) '
                f'or <code>BP_MINCAUSE_TOP</code>. Declared domain: {src}. '
                'SIMULATION.</p>')
    have = [f for f in families if getattr(f, "odd_risk", None)]
    inside = sorted((f for f in have if f.odd_risk["class"] != "outside_odd"),
                    key=lambda f: -f.odd_risk["risk_hi"])
    out = [f for f in have if f.odd_risk["class"] == "outside_odd"]
    parts = [head,
             '<p><small>For each certified family: the exact probability, under the '
             'declared operating domain p, of landing in its minimal-cause region (a '
             'product of per-dim tail probabilities, no sampling error), times a '
             'Clopper-Pearson interval on the failure rate of fresh rollouts drawn from p '
             'inside that region (Bonferroni across families, 95% overall). '
             f'In-domain risk means the lower bound exceeds {_g(summary.get("min_risk", 1e-3))}; '
             'outside the operating domain means the region has zero mass under p or the '
             f'upper bound is below {_g(summary.get("out_eps", 1e-6))}. Declared domain: '
             f'{src}. SIMULATION probabilities, not real-world failure rates.</small></p>']
    if inside:
        parts.append(_famrisk_table(inside))
    else:
        parts.append('<p>No certified family has non-negligible mass inside the declared '
                     'operating domain.</p>')
    if out:
        noun = "family" if len(out) == 1 else "families"
        parts.append(f'<details><summary>Outside your operating domain ({len(out)} '
                     f'{noun})</summary>'
                     '<p><small>These failures need conditions the declared domain does '
                     'not include. They are real in simulation but are not a deployment '
                     'risk unless the domain is wrong.</small></p>'
                     f'{_famrisk_table(out)}</details>')
    parts.append(f'<p><small>Union upper bound over named families: '
                 f'{_g(summary.get("union_upper", 0.0))} ({int(summary.get("rollouts", 0))} '
                 f'fresh rollouts).')
    r = summary.get("residual")
    if r:
        if r.get("comparable"):
            parts.append(f' Whole-policy operational failure probability (BP_RISK): '
                         f'{_g(r["total_lo"])}&ndash;{_g(r["total_hi"])}; at least '
                         f'{_g(r["unexplained_mass_lower"])} of it lies outside every '
                         f'named family region.')
        else:
            parts.append(' BP_RISK ran under a different operating domain, so its total '
                         'is not compared.')
    parts.append('</small></p>')
    return "".join(parts)


FLOOR_FLAG_M = -0.02   # torso more than 2 cm below the floor plane = a rendering artifact


def _e(x) -> str:
    """html-escape any value for text or attribute context."""
    from html import escape

    return escape(str(x), quote=True)


def _video_tag(src, width=360) -> str:
    return f'<video src="{_e(src)}" controls width="{width}" loop muted></video>'


def _cause_table_html(f, nominal) -> str:
    """'Triggering parameters' for a certified family: ONLY the certified cause dims, as
    nominal -> minimal-counterexample value. Falls back to the representative's values
    (flagged as such) when no certified cause exists."""
    mc = getattr(f, "minimal_cause", None)
    if getattr(f, "cause_certified", None) is True and mc is not None:
        if not mc:
            return ('<li><b>Triggering parameters:</b> none (fails at nominal: no '
                    'perturbation needed)</li>')
        rows = []
        for c in mc:
            dim = c["dim"]
            nom = f"{float(nominal[dim]):.3g}" if dim in nominal else "?"
            rows.append(f'<tr><td><code>{_e(dim)}</code></td><td>{_e(nom)}</td>'
                        f'<td>&rarr;</td><td><b>{float(c["value"]):.3g}</b></td>'
                        f'<td><small>{_e(c.get("direction", ""))}</small></td></tr>')
        rows = "".join(rows)
        return ('<li><b>Triggering parameters</b> <small>(certified cause only; every '
                'other dim at nominal)</small>:<table class="cause"><tr><th>dim</th>'
                f'<th>nominal</th><th></th><th>counterexample</th><th></th></tr>{rows}'
                '</table></li>')
    params = ", ".join(f"{_e(k)}={float(v):.3g}"
                       for k, v in (f.triggering_parameters or {}).items())
    return f'<li><b>Representative parameters</b> (not a certified cause): {params}</li>'


def _replay_badge_html(f) -> str:
    rp = getattr(f, "replay", None)
    if not rp:
        return ""
    ok = bool(rp.get("reproduces"))
    verdict_code = rp.get("verdict") or ("reproduced" if ok else "not_reproduced")
    color = {"reproduced": "#128a12", "inconclusive": "#8a5a00"}.get(verdict_code, "#b00")
    twin = rp.get("nominal_twin") or {}
    twin_txt = ""
    if twin:
        twin_txt = (f' Nominal twin (same seed, all dims nominal): '
                    f'{"FELL" if twin.get("fell") else "did not fall"}.')
    verdict = {"reproduced": "reproduces", "inconclusive": "inconclusive at this seed "
               "budget", "not_reproduced": "did not reproduce outside MJX"}[verdict_code]
    hi = rp.get("cp_hi")
    ci = (f"CP interval {float(rp.get('cp_lo', 0)):.2f}-{float(hi):.2f}" if hi is not None
          else f"CP lower {float(rp.get('cp_lo', 0)):.2f}")
    return (f'<p class="badge" style="border-color:{color}"><b style="color:{color}">'
            f'{_e(verdict)}</b>: reproduces in C-MuJoCo full-collision: '
            f'<b>{int(rp.get("k", 0))}/{int(rp.get("n", 0))}</b> '
            f'({ci}, bar {float(rp.get("bar", 0.5)):.2f})'
            f'.{_e(twin_txt)}</p>')


def _floor_flag_html(f) -> str:
    z = getattr(f, "mjx_min_floor_height", None)
    if z is None or z >= FLOOR_FLAG_M:
        return ""
    ev = getattr(f, "evidence_videos", None) or {}
    tail = (" The counterexample videos above are the full-collision replay."
            if ev.get("source") == "c_mujoco_full_collision_replay" else "")
    return (f'<li class="flag"><b>Physics fidelity flag:</b> in the MJX search scene this '
            f'rollout ends with the torso {abs(z) * 100:.0f} cm below the floor plane (the '
            f'feet-only scene gives the trunk no floor contact once it has fallen). The '
            f'fall itself happened; the post-fall sinking in the representative video is '
            f'a contact-model artifact.{tail}</li>')


def _evidence_videos_html(f, report, media_dir) -> str:
    ev = getattr(f, "evidence_videos", None) or {}
    ce, nom = ev.get("counterexample"), ev.get("nominal")
    if not ce:
        return ""
    src = ("independent C-MuJoCo full-collision replay"
           if ev.get("source") == "c_mujoco_full_collision_replay" else "MJX search scene")
    cells = [f'<figure>{_video_tag(_video_src(ce, media_dir))}<figcaption>Minimal '
             f'counterexample (only the certified dims moved)</figcaption></figure>']
    if nom:
        cells.append(f'<figure>{_video_tag(_video_src(nom, media_dir))}<figcaption>Nominal '
                     f'twin (same seed, every dim nominal)</figcaption></figure>')
    return (f'<div class="pair">{"".join(cells)}</div><p><small>Rendered from the '
            f'{_e(src)}, clipped 1 s after the fall.</small></p>')


def _family_section(f, heading, report, media_dir, explanations, validity_configs,
                    stats, rank_by_id, nominal, ranked=True) -> str:
    relative = bool(stats.get("relative"))
    expl = explanations.get(f.id, "")
    expl_html = f"<p><em>{_e(expl)}</em></p>" if expl else ""
    sd = rank_by_id.get(f.id) if ranked else None
    if sd and relative:
        stats_html = (
            f'<li><b>Relative share among families:</b> {sd["probability"]:.3f} '
            f'(95% CI {sd["prob_lo"]:.3f}&ndash;{sd["prob_hi"]:.3f})</li>'
            f'<li><b>Severity &times; share:</b> {sd["sev_prob"]:.3f}</li>'
        )
    elif sd:
        nf = int(((stats.get("overall") or {}).get("successes")) or 0)
        m = len(f.member_ids)
        share = f'{m / nf:.3f} ({m} of {nf})' if nf else f'n/a ({m} members)'
        stats_html = (
            f'<li><b>Share of search-selected failures:</b> {share} '
            f'<small>&middot; {sd["probability"]:.3f} of the {sd["n"]} search-evaluated '
            f'worlds (95% CI {sd["prob_lo"]:.3f}&ndash;{sd["prob_hi"]:.3f}); not a '
            f'deployment failure rate</small></li>'
            f'<li><b>Severity &times; search share:</b> {sd["sev_prob"]:.3f}</li>'
        )
    else:
        stats_html = ""
    rep_src = _video_src(report.mp4_paths.get(f.id, ""), media_dir)
    evidence = _evidence_videos_html(f, report, media_dir)
    if evidence:
        rep_video = (f'<details><summary>Original search representative (all 16 dims as '
                     f'found by the search, MJX scene)</summary>{_video_tag(rep_src, 480)}'
                     f'</details>')
    else:
        rep_video = _video_tag(rep_src, 480)
    margin_html = _margins_html(f)
    return f'''
        <section>
          <h2>{heading}</h2>
          {_replay_badge_html(f)}
          {evidence}
          {rep_video}
          {expl_html}
          <ul>
            {_cause_table_html(f, nominal)}
            {_floor_flag_html(f)}
            <li><b>Mean failure score:</b> {f.mean_failure_score:.3f}</li>
            <li><b>Mean transfer score:</b> {f.mean_transfer_score:.3f}</li>
            {stats_html}
            {_validity_html(f, validity_configs)}
            {margin_html}
            {_odd_risk_row(f)}
            <li><b>Members:</b> {len(f.member_ids)}</li>
          </ul>
        </section>'''


def _fidelity_html(evidence) -> str:
    """Physics-fidelity disclosure: which contact model the search ran on and what the
    independent replay ran on (or why it did not)."""
    if not evidence:
        return ""
    parts = []
    scm = evidence.get("search_collision_model")
    if scm:
        parts.append(f'<b>Search scene:</b> {_e(scm)}.')
    rp = evidence.get("replay") or {}
    st = rp.get("status")
    if st == "ran":
        cm = rp.get("collision_model") or {}
        where = (f"Playground's {cm.get('scene')}" if cm.get("source") == "playground_scene"
                 else f"the search model with body-floor contacts enabled on "
                      f"{cm.get('n_enabled', '?')} robot geoms ({cm.get('reason', '')})")
        parts.append(
            f'<b>Independent replay:</b> every certified minimal counterexample was re-run '
            f'on up to {int(rp.get("k", 0))} seeds in C MuJoCo on {_e(where)}, a different '
            f'contact geometry from the search scene. {int(rp.get("n_reproduced", 0))} of '
            f'{int(rp.get("n_replayed", 0))} families reproduced; '
            f'{int(rp.get("n_demoted", 0))} did not clear the bar'
            + (f' ({int(rp["n_inconclusive"])} of those inconclusive at this seed budget)'
               if rp.get("n_inconclusive") else "")
            + f' ({float(rp.get("seconds", 0)):.0f} s CPU).')
    elif st:
        parts.append(f'<b>Independent replay not run:</b> {_e(rp.get("reason") or st)}. '
                     f'Families below carry MJX-only evidence.')
    sf = evidence.get("surrogate_fit")
    if sf:
        if sf.get("r2") is not None:
            parts.append(f'<b>Failure surrogate:</b> {_e(sf.get("method"))} R&sup2; '
                         f'{float(sf["r2"]):.3f} over {int(sf.get("n", 0))} worlds.')
        elif sf.get("hidden_reason"):
            parts.append(f'<b>Failure surrogate fit:</b> {_e(sf["hidden_reason"])}.')
    return '<h2>Evidence and physics fidelity</h2><p>' + " ".join(parts) + '</p>' if parts \
        else ""


def render_html(report: BreakPointReport, explanations=None, media_dir=None,
                validity_configs=None, stats=None, operational_risk=None,
                envelope=None, evidence=None, nominal=None, famrisk=None) -> str:
    """One self-contained HTML page.

    Families are split by ``schemas.family_status``: CONFIRMED families are ranked
    1..n (contiguous) with severity labels; families whose certified cause did not
    reproduce in the independent C-MuJoCo replay and families CMC could not certify are
    listed in their own unranked sections, with no severity label. ``evidence`` carries
    the fidelity disclosure (search collision model, replay summary, surrogate fit);
    ``nominal`` ({dim: nominal value}) feeds the cause tables. ``famrisk`` is the
    per-family operational-risk summary (famrisk.run_stage), rendered as its own section.
    """
    from breakpoint_mjx.schemas import family_status

    explanations = explanations or {}
    nominal = nominal or {}
    from breakpoint_mjx.stats import severity_primary_rank, severity_probability_rank

    if stats is None:
        stats = {
            "overall": None,
            "ranking": severity_probability_rank(report.families),
            "severity_ranking": severity_primary_rank(report.families),
            "relative": True,
        }
    confirmed_ids = {f.id for f in report.families if family_status(f) == "confirmed"}
    confirmed_names = {f.name for f in report.families if f.id in confirmed_ids}
    # Only confirmed families get a rank or a severity label anywhere on the page.
    stats = dict(stats)
    for key in ("ranking", "severity_ranking"):
        stats[key] = [d for d in (stats.get(key) or [])
                      if d.get("family_id", None) in confirmed_ids
                      or ("family_id" not in d and d.get("name") in confirmed_names)]
    rank_by_id = {d["family_id"]: d for d in (stats.get("ranking") or []) if "family_id" in d}
    from breakpoint_mjx.crossval import rank_families_by_validity

    fams = rank_families_by_validity(report.families)
    groups = {"confirmed": [], "not_confirmed_outside_mjx": [], "unconfirmed": []}
    for f in fams:
        groups[family_status(f)].append(f)
    rows = []
    if not groups["confirmed"] and fams:
        rows.append('<h2 class="group">Ranked failure families</h2><p>None. No family '
                    'cleared every certification bar (seed noise in the search scene and, '
                    'where it ran, the independent full-collision replay), so nothing is '
                    'ranked or given a severity. The families found are listed below with '
                    'the evidence for and against each.</p>')
    for display_rank, f in enumerate(groups["confirmed"], 1):
        rows.append(_family_section(f, f'#{display_rank} &mdash; {_e(f.name)}', report,
                                    media_dir, explanations, validity_configs, stats,
                                    rank_by_id, nominal))
    if groups["not_confirmed_outside_mjx"]:
        rows.append('<h2 class="group">Did not reproduce outside MJX at the certification '
                    'bar</h2><p><small>Certified under seed noise in the MJX search scene, '
                    'but the independent C-MuJoCo full-collision replay of the minimal '
                    'counterexample did not clear the same Clopper-Pearson bar: either it '
                    'reproduced too rarely (did not reproduce) or too few seeds were run to '
                    'decide (inconclusive at this seed budget; raise BP_REPLAY_K or '
                    'BP_REPLAY_MAXMULT). Not ranked.</small></p>')
        for f in groups["not_confirmed_outside_mjx"]:
            rows.append(_family_section(f, _e(f.name), report, media_dir, explanations,
                                        validity_configs, stats, {}, nominal, ranked=False))
    if groups["unconfirmed"]:
        from breakpoint_mjx.schemas import uncertified_name

        rows.append('<h2 class="group">Unconfirmed (not reproducible under seed noise)</h2>'
                    '<p><small>The certified-minimal-cause stage could not certify the '
                    'representative as a failure across seeds, so no cause, severity or '
                    'rank is claimed.</small></p>')
        for f in groups["unconfirmed"]:
            nm = f.name if f.name.startswith("uncertified:") else uncertified_name(f)
            rows.append(_family_section(f, _e(nm), report, media_dir, explanations,
                                        validity_configs, stats, {}, nominal, ranked=False))
    comp = report.comparison
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>BreakPoint report &mdash; {_e(report.policy_descriptor)}</title>
<style>body{{font-family:system-ui;max-width:820px;margin:2rem auto;line-height:1.5}}
section{{border-top:1px solid #ccc;padding:1rem 0}}video{{background:#000}}
.pair{{display:flex;gap:.5rem;flex-wrap:wrap}}.pair figure{{margin:0}}
.badge{{border:2px solid;border-radius:6px;padding:.3rem .6rem}}
table.cause{{border-collapse:collapse}}table.cause td,table.cause th{{padding:0 .5rem}}
.flag{{color:#8a5a00}}h2.group{{margin-top:2rem;color:#555}}</style>
</head><body>
<h1>BreakPoint failure report</h1>
<p>Policy <code>{_e(report.policy_descriptor)}</code> &middot; run <code>{_e(report.run_id)}</code></p>
<p>Search comparison (top-20 transfer-weighted): directed
<b>{comp.get("directed_top20", 0):.3f}</b> vs random <b>{comp.get("random_top20", 0):.3f}</b>.</p>
{_physical_validity_html(stats)}
{_fidelity_html(evidence)}
{_stats_summary_html(stats)}
{_operational_risk_html(operational_risk)}
{_famrisk_html(fams, famrisk)}
{_envelope_html(envelope)}
{"".join(rows)}
</body></html>'''


def _with_sibling_validity(report, path, stats):
    """Attach the physical-validity counts when the caller's stats bundle lacks them.

    run_pipeline writes EVERY evaluated record to ``failures.json`` in the report's
    directory before it writes the report, but computes its stats bundle without the
    records. When that file is present AND holds exactly ``stats["n_total"]`` records (the
    same run), the bundle is recomputed with ``records=`` so the report states how many
    worlds were excluded as simulator artifacts and quotes rates over the valid ones.
    Anything else (no file, a count mismatch, a bare/relative bundle, a read error)
    returns ``stats`` unchanged."""
    if not stats or stats.get("relative") or "validity" in stats:
        return stats
    fpath = os.path.join(os.path.dirname(os.path.abspath(path)), "failures.json")
    if not os.path.exists(fpath):
        return stats
    try:
        import json

        with open(fpath, encoding="utf-8") as fh:
            recs = json.load(fh)
        if not isinstance(recs, list) or len(recs) != int(stats.get("n_total", -1)):
            return stats
        from breakpoint_mjx.stats import compute_report_stats

        ov = stats.get("overall") or {}
        new = compute_report_stats(
            report.families, n_total=int(stats["n_total"]),
            n_failures=int(ov.get("successes", 0)), threshold=stats.get("threshold"),
            z=float(ov.get("z", 1.96)), coverage=stats.get("coverage"), records=recs)
        return {**stats, **new}
    except Exception:  # never fail a report over the validity annotation
        return stats


def write_report(report: BreakPointReport, path, explanations=None, embed=True,
                 validity_configs=None, stats=None, operational_risk=None, envelope=None,
                 evidence=None, nominal=None, famrisk=None):
    # embed=True -> base64-embed mp4s found beside `path` so the report is self-contained.
    # validity_configs: the per-config label vector (from crossval.validate_failures)
    # aligned with each family's reproduction vector, for the T1 per-config display.
    # stats: the T2 always-on stats bundle (stats.compute_report_stats). When None,
    # render_html builds a self-contained fallback from the families alone.
    # envelope: the optional M3 safety_envelope dict; a zero-volume box is replaced by a
    # "no failure-free box found around nominal" line instead of being printed.
    media_dir = os.path.dirname(os.path.abspath(path)) if embed else None
    stats = _with_sibling_validity(report, path, stats)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(report, explanations, media_dir=media_dir,
                             validity_configs=validity_configs, stats=stats,
                             operational_risk=operational_risk, envelope=envelope,
                             evidence=evidence, nominal=nominal, famrisk=famrisk))
    return path


# --------------------------------------------------------------------------- #
# Workstream 3 (T5): hardening before/after + robustness-vs-performance Pareto.
# Dependency-light: an INLINE SVG scatter/line + an HTML table (NO matplotlib). All pure
# stdlib string assembly over a hardening assembly dict (produced by the commercial
# hardening loop; this renderer only formats it).
# --------------------------------------------------------------------------- #
def _pareto_svg(points, width=560, height=360, pad=52) -> str:
    """Inline-SVG scatter of the sweep: x = nominal reward (HIGHER better, right), y =
    robustness cost (LOWER better, so it is drawn with the best/lowest cost at the TOP).
    Frontier points are filled + connected by a polyline; dominated points are hollow.

    Pure string assembly (no plotting dep). Degrades cleanly for a single point / a zero
    range (draws the point(s) centered)."""
    pts = list(points or [])
    if not pts:
        return "<p><em>No hardening points to plot.</em></p>"

    xs = [float(p["nominal_reward"]) for p in pts]
    ys = [float(p["robustness"]) for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def sx(x):
        if xmax == xmin:
            return pad + plot_w / 2.0
        return pad + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        # Lower cost (more robust) -> smaller pixel y (higher on screen), so "better" is up.
        if ymax == ymin:
            return pad + plot_h / 2.0
        return pad + (y - ymin) / (ymax - ymin) * plot_h

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="border:1px solid #ddd;background:#fff;font-family:system-ui;font-size:11px">',
        # axes
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+plot_h}" stroke="#888"/>',
        f'<line x1="{pad}" y1="{pad+plot_h}" x2="{pad+plot_w}" y2="{pad+plot_h}" stroke="#888"/>',
        f'<text x="{pad+plot_w/2}" y="{height-14}" text-anchor="middle">'
        f'nominal reward proxy &#8594; (higher better)</text>',
        f'<text x="14" y="{pad+plot_h/2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {pad+plot_h/2})">robustness cost &#8595; (lower better)</text>',
    ]

    # Frontier polyline (sorted by x so the line reads left-to-right).
    front = sorted((p for p in pts if p.get("on_frontier")),
                   key=lambda p: float(p["nominal_reward"]))
    if len(front) >= 2:
        poly = " ".join(f"{sx(float(p['nominal_reward'])):.1f},{sy(float(p['robustness'])):.1f}"
                        for p in front)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="#128a12" '
                     f'stroke-width="2" stroke-dasharray="4 3"/>')

    for p in pts:
        cx, cy = sx(float(p["nominal_reward"])), sy(float(p["robustness"]))
        on = bool(p.get("on_frontier"))
        fill = "#128a12" if on else "#fff"
        stroke = "#128a12" if on else "#b00"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="2"/>')
        label = str(p.get("label", ""))
        parts.append(f'<text x="{cx+9:.1f}" y="{cy+3:.1f}">{label}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _hardening_table_html(points) -> str:
    """Before/after table: one row per point (baseline + each strength), showing the
    nominal reward, robustness cost, family count, and how many baseline families were
    resolved / new / worsened by that hardening strength, and whether it is Pareto-optimal."""
    head = (
        "<tr><th>Policy</th><th>Nominal reward</th><th>Robustness cost</th>"
        "<th>Families</th><th>Resolved</th><th>New</th><th>Worsened</th>"
        "<th>Pareto</th></tr>"
    )
    rows = []
    for p in points or []:
        mark = "&#10003;" if p.get("on_frontier") else ""
        rows.append(
            "<tr>"
            f"<td>{p.get('label','')}</td>"
            f"<td>{float(p['nominal_reward']):.3f}</td>"
            f"<td>{float(p['robustness']):.3f}</td>"
            f"<td>{p.get('n_families',0)}</td>"
            f"<td>{p.get('families_resolved',0)}</td>"
            f"<td>{p.get('families_new',0)}</td>"
            f"<td>{p.get('families_worsened',0)}</td>"
            f'<td style="text-align:center;color:#128a12">{mark}</td>'
            "</tr>"
        )
    return (
        '<table style="border-collapse:collapse" border="1" cellpadding="6">'
        f"{head}{''.join(rows)}</table>"
    )


def render_hardening_html(assembly: dict, policy_descriptor: str = "",
                          title: str = "BreakPoint hardening report") -> str:
    """Standalone before/after + Pareto HTML from a hardening assembly dict.

    ``assembly`` is ``{"points": [...], "frontier": [...], "baseline_point": {...}}``. The
    page leads with the buy-now story (families shrink at a small nominal-reward cost), shows
    the robustness-vs-performance Pareto curve (inline SVG) and the before/after table."""
    points = assembly.get("points", []) if assembly else []
    base = assembly.get("baseline_point") if assembly else None
    frontier = assembly.get("frontier", []) if assembly else []

    story = ""
    if base is not None and len(points) > 1:
        # Best hardened point = most families resolved, breaking ties toward higher reward.
        hardened = [p for p in points if p.get("strength") is not None]
        if hardened:
            best = max(hardened, key=lambda p: (p["families_resolved"], p["nominal_reward"]))
            dr = base["nominal_reward"] - best["nominal_reward"]
            # Do NOT let the headline hide regressions: surface any families the hardening
            # INTRODUCED or WORSENED right next to the resolved count.
            n_new = int(best.get("families_new", 0))
            n_worse = int(best.get("families_worsened", 0))
            regressions = ""
            if n_new or n_worse:
                regressions = (f' (but introduced <b>{n_new}</b> new and '
                               f'<b>{n_worse}</b> worsened)')
            story = (
                f'<p>Baseline had <b>{base["n_families"]}</b> failure families. '
                f'Hardening at <b>{best["label"]}</b> RESOLVED '
                f'<b>{best["families_resolved"]}</b> of them{regressions} '
                f'(down to <b>{best["n_families"]}</b> families) '
                f'at a nominal-reward cost of <b>{dr:+.3f}</b> '
                f'({base["nominal_reward"]:.3f} &#8594; {best["nominal_reward"]:.3f}).</p>'
            )

    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>{title} &mdash; {policy_descriptor}</title>
<style>body{{font-family:system-ui;max-width:860px;margin:2rem auto;line-height:1.5}}
table{{margin:1rem 0}}th{{background:#f4f4f4}}</style>
</head><body>
<h1>{title}</h1>
<p>Policy <code>{policy_descriptor}</code> &middot; feed discovered failures back as
domain randomization CONCENTRATED on the failure regions, fine-tune, and re-search.</p>
{story}
<h2>Robustness vs performance (Pareto)</h2>
<p><small>Each point is a policy: the frozen baseline plus one per hardening strength.
Filled green = Pareto-optimal (non-dominated); the dashed line is the frontier.
Model-dim failure regions only (obs-noise / latency hardening is out of scope).</small></p>
{_pareto_svg(points)}
<h2>Before / after</h2>
{_hardening_table_html(points)}
<p><small>Robustness cost is lower-is-better (failure rate or family count). "Resolved" =
baseline failure families with no fingerprint match after hardening (via
regression.diff_atlases). {len(frontier)} of {len(points)} policies are Pareto-optimal.
Nominal reward is a PROXY (clean-config tracking &times; upright, in [0,1]), not the true PPO
return.</small></p>
</body></html>'''


def write_hardening_report(assembly: dict, path, policy_descriptor: str = "",
                           title: str = "BreakPoint hardening report"):
    """Write the standalone hardening report (before/after + Pareto) to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_hardening_html(assembly, policy_descriptor=policy_descriptor,
                                       title=title))
    return path
