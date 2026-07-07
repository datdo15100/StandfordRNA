"""Repo-root and path resolution utilities.

Everything in the pipeline resolves paths through here so that scripts and
notebooks behave identically regardless of the working directory.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

# Repo root = two levels up from this file (src/rna3d/paths.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "paths.yaml"

# Environment overrides (set these on a new machine so nothing needs editing):
#   RNA3D_DATA  -> absolute path to the competition data dir (the 61 GB folder)
#   RNA3D_CACHE -> absolute path for derived caches (default: <repo>/data/cache)
ENV_DATA = "RNA3D_DATA"
ENV_CACHE = "RNA3D_CACHE"


@functools.lru_cache(maxsize=1)
def cfg() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def comp_dir() -> Path:
    """Competition data dir. Overridable via the RNA3D_DATA env var so the 61 GB
    dataset can live anywhere (e.g. a fast local SSD) without editing configs."""
    env = os.environ.get(ENV_DATA)
    return Path(env) if env else _resolve(cfg()["comp_data"])


def comp_file(key: str) -> Path:
    """Resolve a competition file by its key in configs/paths.yaml -> files."""
    rel = cfg()["files"][key]
    return comp_dir() / rel


def interim() -> Path:
    d = _resolve(cfg()["interim"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed() -> Path:
    d = _resolve(cfg()["processed"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache() -> Path:
    env = os.environ.get(ENV_CACHE)
    d = Path(env) if env else _resolve(cfg()["cache"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures() -> Path:
    d = _resolve(cfg()["figures"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables() -> Path:
    d = _resolve(cfg()["tables"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def usalign_bin() -> Path:
    return _resolve(cfg()["usalign"])


def casp15_safe_cutoff() -> str:
    return cfg()["casp15_safe_cutoff"]


def coord_sentinel() -> float:
    return float(cfg()["coord_sentinel"])
