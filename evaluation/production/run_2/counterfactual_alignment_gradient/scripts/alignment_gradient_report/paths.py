from __future__ import annotations

"""Shared filesystem and runtime setup for the branch-local report builder.

This module discovers the repository root robustly, configures Matplotlib for headless exports, and exposes the small IO helpers used by the report modules.
"""

import os
import sys
from pathlib import Path

_mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "run2_alignment_gradient_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

GRAPH_ID = "politisky24_bluesky_v1"


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from this branch-local script package."""
    cursor = Path(start or __file__).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "src" / "backend").is_dir() and (candidate / "evaluation" / "production" / "run_2").is_dir():
            return candidate
    raise RuntimeError(f"Could not find project root above {cursor}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.io import abs_path, ensure_dir, read_json, write_json


def graph_root() -> Path:
    """Return the fixed empirical exposure-network directory used by this report."""
    return PROJECT_ROOT / "data" / "exposure_networks" / GRAPH_ID
