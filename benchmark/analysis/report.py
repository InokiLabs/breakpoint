"""Analysis stage: raw run files -> CSV summaries, figures (PNG + PDF), results tables.

Reads only what the search, replay and regression stages wrote. A missing result is shown
as missing (an empty panel with the reason), never filled in.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.analysis import metrics as M
from benchmark.evaluation.setup import METHOD_LABELS


def _load(dirpath: Path):
    return {p.stem: json.loads(p.read_text()) for p in sorted(dirpath.glob("*.json"))
            if not p.name.endswith(".gate.json")} if dirpath.exists() else {}


def collect_runs(outdir: Path, bc):
    runs = _load(outdir / "runs")
    replay = _load(outdir / "replay")
    eps = f"{bc.cluster_eps:g}"
    rows = []
    for rid, r in runs.items():
        an, v, tm = r["analysis"], r["validation"], r["timing"]
        rp = replay.get(rid, {})
        res = rp.get("result") or {}
        est = v["estimated_validated_distinct"]
        rows.append({
            "run_id": rid, "policy": r["policy"], "regime": r["regime"],
            "method": r["method"], "rep": r["rep"], "budget": r["plan"]["budget"],
            "horizon": r["horizon"], "sim_steps": r["sim_steps"],
            "failures": an["n_failures"],
            "failure_rate": an["n_failures"] / r["plan"]["budget"],
            "invalid_rollouts": an["n_invalid"],
            "clusters": an["clusters"][eps]["n"],
            **{f"clusters_eps{e}": c["n"] for e, c in an["clusters"].items() if e != eps},
            "modes": an["n_modes"],
            "closest_l2": an["closest_l2"], "closest_linf": an["closest_linf"],
            "median_ttf_steps": an["median_ttf_steps"],
            "leaders_checked": v["leaders_checked"],
            "leaders_validated": v["leaders_validated"],
            "validation_rate": (v["leaders_validated"] / v["leaders_checked"]
                                if v["leaders_checked"] else None),
            "est_validated_distinct": est["estimate"],
            "est_validated_distinct_lo": est["lo"], "est_validated_distinct_hi": est["hi"],
            "est_exact": est["exact"],
            # coarse grouping (eps 0.3, a predefined sensitivity value) x validation rate
            "coarse_est_validated_distinct": (
                an["clusters"]["0.3"]["n"] * v["leaders_validated"] / v["leaders_checked"]
                if "0.3" in an["clusters"] and v["leaders_checked"] else None),
            "time_to_first_validated": v["time_to_first_validated_rollouts"],
            "search_wall_s": tm["method_wall_s"], "search_eval_s": tm["eval_s"],
            "search_overhead_s": tm["overhead_s"],
            "validation_rollouts": v["rollouts"], "validation_wall_s": v["wall_s"],
            "replay_n": res.get("n") if res else None,
            "replay_confirmed": res.get("confirmed") if res else None,
            "replay_note": rp.get("skipped"),
        })
    return runs, replay, rows


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


SUMMARY_METRICS = ("failure_rate", "clusters", "modes", "closest_l2", "validation_rate",
                   "est_validated_distinct", "coarse_est_validated_distinct",
                   "time_to_first_validated", "search_wall_s",
                   "search_overhead_s", "invalid_rollouts")


def summarize(rows, bc):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["regime"], r["method"])].append(r)
    out = []
    for (pol, reg, meth), rs in sorted(groups.items()):
        rec = {"policy": pol, "regime": reg, "method": meth, "reps": len(rs)}
        for m in SUMMARY_METRICS:
            s = M.summarize_reps([x[m] for x in rs])
            rec[f"{m}_mean"] = s["mean"]
            rec[f"{m}_min"] = s["min"]
            rec[f"{m}_max"] = s["max"]
        rec["ttfv_not_reached"] = sum(1 for x in rs if x["time_to_first_validated"] is None)
        n_rep = sum((x["replay_n"] or 0) for x in rs)
        k_rep = sum((x["replay_confirmed"] or 0) for x in rs)
        rec["replay_confirmed"] = k_rep
        rec["replay_n"] = n_rep
        vk = sum(x["leaders_validated"] for x in rs)
        vn = sum(x["leaders_checked"] for x in rs)
        lo, hi = M.clopper_pearson(vk, vn) if vn else (None, None)
        rec["pooled_validation"] = f"{vk}/{vn}"
        rec["pooled_validation_ci95"] = (f"[{lo:.2f}, {hi:.2f}]" if vn else None)
        out.append(rec)
    return out


def ratio_targets(bc):
    """directed, plus every declared variant that was run."""
    return [m for m in bc.methods if m == "directed" or m in bc.method_variants]


def ratios_vs(rows, bc, metric, baseline="uniform", target="directed"):
    out = []
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r[metric] is not None:
            by[(r["policy"], r["regime"])][r["method"]].append(r[metric])
    for (pol, reg), d in sorted(by.items()):
        for base in [m for m in bc.methods if m != target]:
            if d.get(target) and d.get(base):
                br = M.bootstrap_ratio(d[target], d[base], seed=hash((pol, reg, base)) % 2**31)
                out.append({"policy": pol, "regime": reg, "metric": metric,
                            "target": target, "directed_vs": base, **br})
    return out


# ------------------------------------------------------------------------------ figures
COLORS = {"uniform": "#7f7f7f", "sobol": "#1f77b4", "lhs_boundary": "#2ca02c",
          "directed": "#d62728"}
_EXTRA = ["#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def _color(meth):
    if meth not in COLORS:
        COLORS[meth] = _EXTRA[(len(COLORS) - 4) % len(_EXTRA)]
    return COLORS[meth]


def _save(fig, figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, color="#a00",
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def figures(bc, runs, replay, rows, regression, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    eps = f"{bc.cluster_eps:g}"
    pols = [p.name for p in bc.policies]
    regs = bc.regimes
    foot = "Simulation only (MuJoCo MJX). Bands = min-max across search repetitions."

    # 1. discovery vs budget (rollouts) and vs wall clock
    for xaxis in ("rollouts", "wall"):
        fig, axes = plt.subplots(len(pols), len(regs), figsize=(5.2 * len(regs), 3.4 * len(pols)),
                                 squeeze=False)
        for i, pol in enumerate(pols):
            for j, reg in enumerate(regs):
                ax = axes[i][j]
                any_ = False
                for meth in bc.methods:
                    rs = [r for r in runs.values() if r["policy"] == pol and r["regime"] == reg
                          and r["method"] == meth]
                    if not rs:
                        continue
                    any_ = True
                    # scale by the validation rate POOLED over this cell's repetitions:
                    # one run's 8 sampled leaders give too noisy a rate on their own
                    vk = sum(r["validation"]["leaders_validated"] for r in rs)
                    vn = sum(r["validation"]["leaders_checked"] for r in rs)
                    rate = vk / vn if vn else 0.0
                    curves = [np.asarray(r["analysis"]["clusters"][eps]["curve"], dtype=float)
                              * rate for r in rs]
                    C = np.vstack(curves)
                    if xaxis == "rollouts":
                        x = np.arange(1, C.shape[1] + 1)
                    else:
                        # interpolate onto a common elapsed-time grid per method
                        ts = []
                        for r in rs:
                            tm = r["timing"]
                            ends = np.asarray(tm["batch_end_elapsed_s"])
                            sizes = np.asarray(tm["batch_sizes"])
                            ts.append(np.repeat(ends, sizes)[:C.shape[1]])
                        T = np.vstack(ts)
                        x = np.linspace(0, T.max(), 200)
                        C = np.vstack([np.interp(x, T[k], C[k], left=0) for k in range(len(T))])
                    ax.plot(x, C.mean(0), color=_color(meth), label=bc.label(meth), lw=1.6)
                    ax.fill_between(x, C.min(0), C.max(0), color=_color(meth), alpha=0.15, lw=0)
                if not any_:
                    _empty(ax, "no runs (configuration excluded or not run)")
                ax.set_title(f"{pol} | {reg} bounds", fontsize=9)
                ax.set_xlabel("rollouts spent" if xaxis == "rollouts" else "search wall-clock (s, CPU)")
                ax.set_ylabel("est. distinct validated failures")
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle("Distinct validated failures vs " +
                     ("simulation budget" if xaxis == "rollouts" else "wall-clock time") +
                     f"\n(failure clusters at L-inf eps={eps}, x validation rate pooled over "
                     "repetitions)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.text(0.01, -0.02, foot, fontsize=7)
        _save(fig, figdir, "fig1_discovery_vs_budget" if xaxis == "rollouts"
              else "fig1b_discovery_vs_wallclock")
        plt.close(fig)

    # 2. time to first validated failure
    fig, axes = plt.subplots(1, len(regs), figsize=(5.2 * len(regs), 3.4), squeeze=False)
    for j, reg in enumerate(regs):
        ax = axes[0][j]
        budget = bc.plan.budget
        for k, meth in enumerate(bc.methods):
            vals = [r["time_to_first_validated"] for r in rows
                    if r["regime"] == reg and r["method"] == meth]
            reached = [v for v in vals if v is not None]
            miss = len(vals) - len(reached)
            xs = k + (np.random.default_rng(k).random(len(reached)) - 0.5) * 0.25
            ax.scatter(xs, reached, color=_color(meth), s=18, zorder=3)
            if miss:
                ax.scatter([k], [budget * 1.08], marker="x", color=_color(meth), s=30)
                ax.text(k, budget * 1.13, f"{miss} not reached", ha="center", fontsize=7)
        ax.set_xticks(range(len(bc.methods)))
        ax.set_xticklabels([bc.label(m).replace(" (", "\n(") for m in bc.methods], fontsize=7)
        ax.set_ylabel("rollouts to first validated failure")
        ax.set_title(f"{reg} bounds (all configurations, all repetitions)", fontsize=9)
        ax.set_ylim(0, budget * 1.2)
    fig.text(0.01, -0.04, foot + " x = none of the first in-order failures validated.", fontsize=7)
    _save(fig, figdir, "fig2_time_to_first_validated")
    plt.close(fig)

    # 3. reproducibility
    fig, axes = plt.subplots(1, len(regs), figsize=(5.2 * len(regs), 3.4), squeeze=False)
    for j, reg in enumerate(regs):
        ax = axes[0][j]
        for k, meth in enumerate(bc.methods):
            ps, vk, vn = [], 0, 0
            for r in runs.values():
                if r["regime"] != reg or r["method"] != meth:
                    continue
                v = r["validation"]
                for row in v["leader_rows_sampled"]:
                    res = v["results"][str(row)]
                    ps.append(res["p"])
                    vk += int(res["validated"])
                    vn += 1
            if vn:
                lo, hi = M.clopper_pearson(vk, vn)
                ax.errorbar([k], [vk / vn], yerr=[[vk / vn - lo], [hi - vk / vn]], fmt="o",
                            color=_color(meth), capsize=4, zorder=3)
                ax.text(k + 0.12, vk / vn, f"{vk}/{vn}", fontsize=7, va="center")
                xs = k + (np.random.default_rng(k).random(len(ps)) - 0.5) * 0.3
                ax.scatter(xs, ps, color=_color(meth), s=6, alpha=0.35)
            else:
                ax.text(k, 0.5, "none", ha="center", fontsize=7)
        ax.axhline(0.5, color="#999", lw=0.7, ls="--")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(range(len(bc.methods)))
        ax.set_xticklabels([bc.label(m).replace(" (", "\n(") for m in bc.methods], fontsize=7)
        ax.set_ylabel("fraction validated (dots: per-candidate fall prob.)")
        ax.set_title(f"{reg} bounds: re-run on {bc.validation_n} fresh seeds", fontsize=9)
    fig.text(0.01, -0.04, foot + " Bars: 95% Clopper-Pearson. Validated = one-sided 95% lower bound >= 0.5.",
             fontsize=7)
    _save(fig, figdir, "fig3_reproducibility")
    plt.close(fig)

    # 4. failure distribution over parameters (validated failures vs all evaluated)
    for reg in regs:
        names = None
        allX, valX = [], []
        for r in runs.values():
            if r["regime"] != reg:
                continue
            names = r["box"]["names"]
            lo, hi = np.asarray(r["box"]["lo"]), np.asarray(r["box"]["hi"])
            th = np.asarray(r["rows"]["theta"], dtype=float)
            allX.append(M.box_normalize(th, lo, hi))
            for row, res in r["validation"]["results"].items():
                if res["validated"]:
                    valX.append(M.box_normalize(th[int(row)], lo, hi)[0])
        if names is None:
            continue
        A = np.vstack(allX)
        V = np.asarray(valX) if valX else np.zeros((0, len(names)))
        nd = len(names)
        cols = 4
        fig, axes = plt.subplots(int(np.ceil(nd / cols)), cols, figsize=(11, 2.0 * np.ceil(nd / cols)))
        for d, ax in enumerate(np.ravel(axes)):
            if d >= nd:
                ax.axis("off")
                continue
            bins = np.linspace(0, 1, 11)
            ax.hist(A[:, d], bins=bins, density=True, color="#ccc", label="all evaluated")
            if len(V):
                ax.hist(V[:, d], bins=bins, density=True, histtype="step", color="#d62728",
                        lw=1.5, label="validated failures")
            ax.set_title(names[d], fontsize=8)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["lo", "", "hi"], fontsize=7)
        np.ravel(axes)[0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"Where validated failures sit, {reg} bounds (n={len(V)} validated, "
                     f"all methods and configurations; x = position within each range)", fontsize=10)
        fig.tight_layout()
        _save(fig, figdir, f"fig4_failure_distribution_{reg}")
        plt.close(fig)

    # 5. regression
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels, b_rates, c_rates, verdicts = [], [], [], []
    for name, rg in sorted(regression.items()):
        rep = (rg.get("gate") or {}).get("report")
        if not rep:
            labels.append(f"{name}\n({rg.get('status', 'no report')})")
            b_rates.append(np.nan)
            c_rates.append(np.nan)
            verdicts.append(rg.get("reason", "incomplete"))
            continue
        bf, bn, cf, cn = _gate_totals(rep)
        labels.append(f"{name}\n[{rg['kind']}]")
        b_rates.append(bf / bn if bn else np.nan)
        c_rates.append(cf / cn if cn else np.nan)
        verdicts.append("REGRESSION" if rg["gate"]["exit_code"] == 1 else
                        ("pass" if rg["gate"]["exit_code"] == 0 else "error"))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.18, b_rates, 0.36, color="#7f7f7f", label="baseline")
        ax.bar(x + 0.18, c_rates, 0.36, color="#d62728", label="candidate")
        for k, v in enumerate(verdicts):
            ax.text(k, (np.nanmax([b_rates[k], c_rates[k], 0]) or 0) + 0.03, v,
                    ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("fall rate on fixed battery")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7, frameon=False)
    else:
        _empty(ax, "no regression results")
    ax.set_title("Policy regression gate on legged-quadruped-v1 (fixed battery)", fontsize=9)
    fig.text(0.01, -0.06, "No real checkpoint pair exists in this repository: 'identical' = the same checkpoint twice "
             "checkpoints, 'synthetic' = weight-noise candidates (not real policies).", fontsize=7)
    _save(fig, figdir, "fig5_regression")
    plt.close(fig)


def _gate_totals(rep):
    """(baseline falls, n, candidate falls, n) on the battery's own seeds.

    The gate report's ``cells`` are the candidate's per-cell results; the baseline's
    per-cell falls are in ``gate.baseline_falls`` on the same seeds (same n)."""
    cells = rep.get("cells", []) or []
    base = (rep.get("gate") or {}).get("baseline_falls") or {}
    cf = sum(int(c.get("falls", 0)) for c in cells)
    cn = sum(int(c.get("n", 0)) for c in cells)
    bf = sum(int(base.get(c["id"], 0)) for c in cells)
    return bf, cn, cf, cn


# ------------------------------------------------------------------------------- tables
def summary_markdown(summary, ratio_rows, bc):
    def f(x, fmt="{:.2f}"):
        return "-" if x is None else fmt.format(x)

    lines = ["| config | bounds | method | reps | failure rate | distinct clusters | modes | "
             "closest L2 | leaders validated | est. distinct validated | rollouts to 1st validated | "
             "replay (C MuJoCo) | search wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            f"| {s['policy']} | {s['regime']} | {bc.label(s['method'])} | {s['reps']} | "
            f"{f(s['failure_rate_mean'])} | {f(s['clusters_mean'], '{:.1f}')} "
            f"[{f(s['clusters_min'], '{:.0f}')}-{f(s['clusters_max'], '{:.0f}')}] | "
            f"{f(s['modes_mean'], '{:.1f}')} | {f(s['closest_l2_mean'], '{:.3f}')} | "
            f"{s['pooled_validation']} {s['pooled_validation_ci95'] or ''} | "
            f"{f(s['est_validated_distinct_mean'], '{:.1f}')} | "
            f"{f(s['time_to_first_validated_mean'], '{:.0f}')}"
            f"{' (' + str(s['ttfv_not_reached']) + ' not reached)' if s['ttfv_not_reached'] else ''} | "
            f"{s['replay_confirmed']}/{s['replay_n']} | {f(s['search_wall_s_mean'], '{:.0f}')} |")
    lines += ["", "Directed / baseline ratio of means, 95% bootstrap over repetitions:", "",
              "| config | bounds | method | metric | vs | ratio | 95% CI |",
              "|---|---|---|---|---|---|---|"]
    for r in ratio_rows:
        lines.append(f"| {r['policy']} | {r['regime']} | {r.get('target', 'directed')} | "
                     f"{r['metric']} | {r['directed_vs']} | "
                     f"{f(r['ratio'])} | [{f(r['lo'])}, {f(r['hi'])}] |")
    return "\n".join(lines)


def analyze(bc, outdir: Path):
    runs, replay, rows = collect_runs(outdir, bc)
    regression = _load(outdir / "regression")
    status = _load(outdir / "configs_status")
    csvdir = outdir / "csv"
    csvdir.mkdir(exist_ok=True)
    write_csv(csvdir / "runs.csv", rows)
    summary = summarize(rows, bc)
    write_csv(csvdir / "summary.csv", summary)
    ratio_rows = []
    for target in ratio_targets(bc):
        for metric in ("est_validated_distinct", "coarse_est_validated_distinct", "clusters",
                       "modes", "closest_l2", "failure_rate"):
            ratio_rows += ratios_vs(rows, bc, metric, target=target)
    write_csv(csvdir / "ratios.csv", ratio_rows)
    vrows = []
    for rid, r in runs.items():
        for row, v in r["validation"]["results"].items():
            vrows.append({"run_id": rid, "policy": r["policy"], "regime": r["regime"],
                          "method": r["method"], "rep": r["rep"], "row": int(row),
                          "selected_as": ("first" if int(row) in r["validation"]["first_rows"]
                                          else "") + ("+leader" if int(row) in
                                          r["validation"]["leader_rows_sampled"] else ""),
                          "k": v["k"], "n": v["n"], "p": v["p"], "ci95_lo": v["ci95"][0],
                          "ci95_hi": v["ci95"][1], "validated": v["validated"],
                          "n_invalid": v["n_invalid"],
                          "invalid_reasons": json.dumps(v["invalid_reasons"]),
                          "median_ttf_steps": v["median_ttf"]})
    write_csv(csvdir / "validation.csv", vrows)
    reg_rows = []
    for name, rg in sorted(regression.items()):
        rep = (rg.get("gate") or {}).get("report") or {}
        bf, bn, cf, cn = _gate_totals(rep) if rep else (0, 0, 0, 0)
        gblock = rep.get("gate", {}) if rep else {}
        adv = rg.get("adversarial", {})
        reg_rows.append({"name": name, "kind": rg.get("kind"), "status": rg.get("status", "ran"),
                         "gate_exit": (rg.get("gate") or {}).get("exit_code"),
                         "base_falls": bf, "base_n": bn, "cand_falls": cf, "cand_n": cn,
                         "regressed": gblock.get("regressed"),
                         "pooled_p": (gblock.get("pooled") or {}).get("p"),
                         "adversarial_conditions": adv.get("conditions"),
                         "adversarial_regressed": (adv.get("decision") or {}).get("regressed")})
    write_csv(csvdir / "regression.csv", reg_rows)
    figures(bc, runs, replay, rows, regression, outdir / "figures")
    md = ["# Results tables (generated)", "",
          f"Config `{bc.name}`; runs: {len(rows)}; generated from raw files in this directory.",
          "", "## Configuration checks", ""]
    for k, s in status.items():
        md.append(f"* {k}: nominal valid falls {s['valid_falls']}/{s['n']} -> "
                  f"{'walks' if s['walks'] else 'EXCLUDED'}")
    md += ["", "## Experiment A / B summary", "", summary_markdown(summary, ratio_rows, bc),
           "", "## Experiment C", ""]
    for r in reg_rows:
        md.append(f"* {r['name']} [{r['kind']}]: status {r['status']}, gate exit "
                  f"{r['gate_exit']}, baseline {r['base_falls']}/{r['base_n']}, candidate "
                  f"{r['cand_falls']}/{r['cand_n']}; adversarial battery: "
                  f"{r['adversarial_conditions']} conditions, regressed={r['adversarial_regressed']}")
    (outdir / "RESULTS_TABLES.md").write_text("\n".join(md) + "\n")
    print(f"[analyze] wrote {csvdir} and {outdir / 'figures'} and RESULTS_TABLES.md")
