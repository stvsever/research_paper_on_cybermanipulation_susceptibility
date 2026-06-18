from __future__ import annotations

import csv
from pathlib import Path

from .defaults import (
    DEFAULT_GRAPH_ID,
    EDGE_DIRECTION,
    REQUIRED_ASSIGNMENT_COLUMNS,
    REQUIRED_EDGE_COLUMNS,
    REQUIRED_PACKAGE_FILES,
    REQUIRED_POSITION_COLUMNS,
)
from .schemas import ExposureNetworkPackage


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _require_columns(path: Path, required: tuple[str, ...]) -> None:
    header = set(_header(path))
    missing = [column for column in required if column not in header]
    if missing:
        raise RuntimeError(f"{path.name} missing required columns: {', '.join(missing)}")


def validate_exposure_network_package(
    package: ExposureNetworkPackage,
    *,
    expected_graph_id: str = DEFAULT_GRAPH_ID,
    check_edges: bool = False,
) -> dict[str, object]:
    missing = [filename for filename in REQUIRED_PACKAGE_FILES if not package.path(filename).exists()]
    if missing:
        raise RuntimeError(f"Exposure-network package missing files: {', '.join(missing)}")
    if package.graph_id != expected_graph_id:
        raise RuntimeError(f"Expected graph_id={expected_graph_id}, found {package.graph_id}")
    if package.manifest.get("edge_direction") != EDGE_DIRECTION:
        raise RuntimeError(f"Unexpected edge direction: {package.manifest.get('edge_direction')}")

    _require_columns(package.path("nodes.csv"), REQUIRED_POSITION_COLUMNS)
    _require_columns(package.path("node_metrics.csv"), REQUIRED_POSITION_COLUMNS)
    _require_columns(package.path("neighborhood_metrics.csv"), REQUIRED_POSITION_COLUMNS)
    _require_columns(package.path("propagation_metrics.csv"), REQUIRED_POSITION_COLUMNS)
    _require_columns(package.path("assignment_positions.csv"), REQUIRED_ASSIGNMENT_COLUMNS)
    _require_columns(package.path("edges_full.csv"), REQUIRED_EDGE_COLUMNS)
    _require_columns(package.path("edges_prompt_top30.csv"), REQUIRED_EDGE_COLUMNS)

    manifest_files = dict(package.manifest.get("files") or {})
    row_counts: dict[str, int] = {}
    for filename in REQUIRED_PACKAGE_FILES:
        if not filename.endswith(".csv"):
            continue
        actual = _row_count(package.path(filename))
        row_counts[filename] = actual
        expected = manifest_files.get(filename, {}).get("rows")
        if expected is not None and int(expected) != actual:
            raise RuntimeError(f"{filename} row count mismatch: manifest={expected}, actual={actual}")

    position_ids = package.position_ids
    missing_assignment_positions = [
        row["position_id"]
        for row in package.assignment_positions
        if str(row.get("position_id") or "") not in package.node_metrics
    ]
    if missing_assignment_positions:
        raise RuntimeError(f"Assignment positions missing node metrics: {missing_assignment_positions[:5]}")

    scanned_edges = 0
    if check_edges:
        for edge_file in ("edges_full.csv", "edges_prompt_top30.csv"):
            with package.path(edge_file).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    source = str(row["source_position_id"])
                    target = str(row["target_position_id"])
                    weight = float(row["exposure_weight"])
                    if source not in position_ids or target not in position_ids:
                        raise RuntimeError(f"{edge_file} contains unknown endpoint: {source}->{target}")
                    if weight < 0.0 or weight > 1.0:
                        raise RuntimeError(f"{edge_file} contains out-of-range exposure_weight: {weight}")
                    scanned_edges += 1
    return {
        "graph_id": package.graph_id,
        "root": str(package.root),
        "row_counts": row_counts,
        "check_edges": bool(check_edges),
        "scanned_edges": scanned_edges,
    }

