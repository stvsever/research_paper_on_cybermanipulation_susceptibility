from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DERIVED_DIR = ROOT / "data" / "derived"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"
GRAPH_ID = "politisky24_bluesky_v1"
GRAPH_RELATIVE_ROOT = Path("data") / "exposure_networks" / GRAPH_ID


def _find_repo_root() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / GRAPH_RELATIVE_ROOT / "manifest.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find canonical graph package at {GRAPH_RELATIVE_ROOT}")


REPO_ROOT = _find_repo_root()
SUBSTRATE_DIR = REPO_ROOT / GRAPH_RELATIVE_ROOT

INTERACTION_WEIGHTS = {
    "Like": 0.35,
    "Repost": 0.80,
    "Quote": 0.90,
}

RANDOM_SEED = 42
PROMPT_TOP_K = 8

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#A3BEFA",
    "gold": "#FFE15B",
    "orange": "#F0986E",
    "olive": "#A3D576",
    "pink": "#F390CA",
    "neutral": "#C5CAD3",
    "neutral_dark": "#464C55",
}

ROLE_PALETTE = {
    "high_visibility_sender": TOKENS["orange"],
    "high_exposure_receiver": TOKENS["blue"],
    "bridge": TOKENS["pink"],
    "peripheral": TOKENS["neutral"],
    "context_position": "#E2E5EA",
}

ROLE_LABELS = {
    "high_visibility_sender": "high visibility sender",
    "high_exposure_receiver": "high exposure receiver",
    "bridge": "bridge",
    "peripheral": "peripheral",
    "context_position": "context position",
}

ROLE_DEFINITIONS = {
    "high_visibility_sender": (
        "Positions selected because they rank high on full-network weighted out-degree, "
        "with eigenvector centrality as a tie-breaker. These positions represent actors whose content "
        "is plausibly visible to many others."
    ),
    "high_exposure_receiver": (
        "Positions selected because they rank high on full-network weighted in-degree, "
        "with eigenvector centrality as a tie-breaker. These positions represent actors with many incoming "
        "exposure sources."
    ),
    "bridge": (
        "Positions selected because they rank high on approximate betweenness and bridge score. "
        "These positions connect otherwise more separated parts of the directed exposure graph."
    ),
    "peripheral": (
        "Positions selected because both weighted out-degree and weighted in-degree are low while "
        "still satisfying the prompt-peer-capacity requirement."
    ),
    "context_position": (
        "Remaining selected positions retained for community coverage and induced connectivity. They are shown as "
        "network context rather than interpreted as a mechanism class in this figure."
    ),
}

INPUT_FILES = {
    "edges_prompt_top30.csv": SUBSTRATE_DIR / "edges_prompt_top30.csv",
    "edges_full.csv": SUBSTRATE_DIR / "edges_full.csv",
    "node_metrics.csv": SUBSTRATE_DIR / "node_metrics.csv",
    "neighborhood_metrics.csv": SUBSTRATE_DIR / "neighborhood_metrics.csv",
    "propagation_metrics.csv": SUBSTRATE_DIR / "propagation_metrics.csv",
    "assignment_positions.csv": SUBSTRATE_DIR / "assignment_positions.csv",
    "manifest.json": SUBSTRATE_DIR / "manifest.json",
    "pilot_60_position_slice.csv": DERIVED_DIR / "pilot_60_position_slice.csv",
    "pilot_60_position_slice_edges.csv": DERIVED_DIR / "pilot_60_position_slice_edges.csv",
}


def input_path(filename: str) -> Path:
    try:
        return INPUT_FILES[filename]
    except KeyError as exc:
        raise KeyError(f"Unknown exposure-network report input: {filename}") from exc


def ensure_dirs() -> None:
    for path in (DERIVED_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def relative_to_root(path: Path) -> str:
    return str(path.relative_to(ROOT))


def artifact_path(path: Path) -> str:
    resolved = Path(path).resolve()
    for base in (REPO_ROOT, ROOT):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(path)
