"""Back-compat shim: the pipeline now lives inside the installed package.

It moved to ``breakpoint_mjx/pipeline.py`` so that `pip install breakpoint-mjx` can
run a search without a repo checkout. Every existing invocation keeps working through
this file (the Colab notebooks, scripts/run_shootout.py, the README, older docs).

Prefer one of these going forward:
    breakpoint run --robot go1 --policy onnx:./policy.onnx
    python -m breakpoint_mjx.pipeline --target go1
"""
from breakpoint_mjx.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
