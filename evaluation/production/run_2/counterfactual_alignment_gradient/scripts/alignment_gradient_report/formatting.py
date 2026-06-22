from __future__ import annotations

"""Formatting helpers shared by tables, HTML, and static figure exports."""

import html
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _relative_src(path: Path, from_dir: Path) -> str:
    return html.escape(os.path.relpath(path, from_dir))


def _clean_leaf(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).split(" > ")[-1]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{digits}f}"
    return str(value)


def _fmt_p(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _signed_tick_label(value: float, _position: int | None = None) -> str:
    if abs(float(value)) < 1e-9:
        return "0"
    if math.isclose(float(value), round(float(value)), abs_tol=1e-6):
        return f"{float(value):+.0f}"
    return f"{float(value):+.1f}"


def _sender_alignment_label(value: Any) -> str:
    """Format achieved sender-reach susceptibility alignment for compact panel labels."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "sender alignment z = -"
    return f"sender alignment z = {float(value):+.2f}"


def _label(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).replace("_", " ")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
