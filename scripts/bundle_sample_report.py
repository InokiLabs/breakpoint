"""Copy a finished demo run's report.html + families.json into the package as the
bundled sample report (breakpoint_mjx/assets/demo/sample_report/), with an in-page
label saying exactly what it is. `breakpoint demo --view` opens it.

    python scripts/bundle_sample_report.py <run_dir> --machine "Windows 11, 16 GB, CPU-only"

<run_dir> is the go1 folder of a `breakpoint demo` output (it holds report.html).
The label is inserted, never the numbers: everything else is the run's own output.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "breakpoint_mjx" / "assets" / "demo" / "sample_report"


def label_html(page, label):
    banner = (
        '<div id="bp-sample-label" style="background:#fff4e5;border:1px solid #f0b86e;'
        'padding:10px 14px;margin:0 0 16px;font:14px/1.4 system-ui,sans-serif;'
        f'color:#5a3a00"><b>Bundled sample.</b> {html.escape(label)}</div>')
    m = re.search(r"<body[^>]*>", page, flags=re.I)
    if not m:
        return banner + page
    return page[:m.end()] + banner + page[m.end():]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--machine", required=True, help="where it was measured")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args(argv)
    src = Path(a.run_dir)
    label = (f"CPU smoke run, bundled Go1, {a.date}, measured on {a.machine}. This is "
             f"the smoke budget (a tour of the product, not a converged search) and "
             f"not a GPU or standard-budget result. For the standard budget on a GPU "
             f"see colab/evidence_cell.py (UNVERIFIED on this build).")
    DEST.mkdir(parents=True, exist_ok=True)
    page = (src / "report.html").read_text(encoding="utf-8")
    (DEST / "report.html").write_text(label_html(page, label), encoding="utf-8")
    fam = json.loads((src / "families.json").read_text(encoding="utf-8"))
    fam["sample_label"] = label
    (DEST / "families.json").write_text(json.dumps(fam), encoding="utf-8")
    for name in ("report.html", "families.json"):
        print(f"wrote {DEST / name} ({(DEST / name).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
