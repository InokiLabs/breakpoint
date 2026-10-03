"""Where the shipped YAML configs live, for both a repo checkout and a wheel.

The configs (search.yaml, per-robot theta spaces) are DATA the engine cannot run
without, so they ship inside the package rather than sitting at the repo root
where an installed wheel would never find them. This module is the single place
that knows how to locate them, so nothing else has to guess.

Both layouts resolve:
  * installed / normal  ->  <site-packages>/breakpoint_mjx/configs/...
  * a stale checkout    ->  <repo>/configs/...   (kept as a fallback so an older
                            working tree, or a user who kept configs outside the
                            package, does not break)
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def configs_dir() -> Path:
    """The directory holding search.yaml and the per-robot config folders."""
    for cand in (PACKAGE_DIR / "configs", REPO_ROOT / "configs"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "BreakPoint's configs directory is missing. Expected it at "
        f"{PACKAGE_DIR / 'configs'} (installed) or {REPO_ROOT / 'configs'} "
        "(repo checkout). A broken install can cause this: reinstall with "
        "`pip install --force-reinstall breakpoint-mjx`."
    )


def config_path(rel) -> Path:
    """Resolve a config reference to a real path.

    Accepts any of the forms that exist in the codebase and in generated targets:
      * "search.yaml"
      * "configs/go1/theta_space.yaml"      (the legacy repo-relative form stored
                                             on Target.theta_space_path)
      * an ABSOLUTE path                    (URDF-derived targets write their
                                             derived theta space next to the run,
                                             which has no home inside the package)
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "configs":
        p = Path(*parts[1:])
    return configs_dir() / p


def search_config_path() -> Path:
    return config_path("search.yaml")
