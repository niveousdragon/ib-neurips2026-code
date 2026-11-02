"""Where data, cached results and figures live.

Every script resolves its inputs and outputs through these helpers, so the
same code runs on any machine. Defaults are folders inside the repository;
each can be overridden with an environment variable:

    IB_DATA_SNN, IB_DATA_CORTEX, IB_DATA_HIPPOCAMPUS   raw data per system
    IB_RESULTS                                          root of cached results
    IB_FIGURES                                          rendered figures
"""
import os
from pathlib import Path

SYSTEMS = ("snn", "cortex", "hippocampus")


def repo_root() -> Path:
    """Repository root: IB_ROOT if set, else the checkout this package lives in."""
    env = os.environ.get("IB_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _check(system: str) -> None:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; expected one of {SYSTEMS}")


def data_dir(system: str) -> Path:
    """Raw data of one system (IB_DATA_<SYSTEM>, default <repo>/data/<system>)."""
    _check(system)
    env = os.environ.get(f"IB_DATA_{system.upper()}")
    return Path(env) if env else repo_root() / "data" / system


def results_dir(system: str, *sub: str, create: bool = True) -> Path:
    """Cached results of one system (IB_RESULTS, default <repo>/results)."""
    _check(system)
    root = os.environ.get("IB_RESULTS")
    path = (Path(root) if root else repo_root() / "results").joinpath(system, *sub)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(create: bool = True) -> Path:
    """Rendered figures (IB_FIGURES, default <repo>/figures/out)."""
    env = os.environ.get("IB_FIGURES")
    path = Path(env) if env else repo_root() / "figures" / "out"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
