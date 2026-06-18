from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .defaults import (
    DEFAULT_GRAPH_ID,
    EDGE_FILE_FULL,
    REQUIRED_PACKAGE_FILES,
)
from .schemas import ExposureEdge, ExposureEdgeIndex, ExposureNetworkPackage


def default_exposure_network_root() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "exposure_networks" / DEFAULT_GRAPH_ID


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_csv_by_position(path: Path) -> dict[str, dict[str, str]]:
    return {str(row["position_id"]): row for row in read_csv_rows(path)}


def _load_json(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _require_files(root: Path, filenames: Iterable[str]) -> None:
    missing = [filename for filename in filenames if not (root / filename).exists()]
    if missing:
        raise RuntimeError(f"Exposure-network package missing required files: {', '.join(missing)}")


def load_exposure_network_package(
    graph_root: str | Path | None = None,
    *,
    graph_id: str = DEFAULT_GRAPH_ID,
    validate: bool = True,
) -> ExposureNetworkPackage:
    root = Path(graph_root) if graph_root is not None else default_exposure_network_root()
    _require_files(root, REQUIRED_PACKAGE_FILES)
    manifest = _load_json(root / "manifest.json")
    if str(manifest.get("graph_id") or "") != graph_id:
        raise RuntimeError(f"Expected graph_id={graph_id}, found {manifest.get('graph_id')}")
    package = ExposureNetworkPackage(
        root=root,
        graph_id=graph_id,
        manifest=manifest,
        data_dictionary=_load_json(root / "data_dictionary.json"),
        nodes=read_csv_by_position(root / "nodes.csv"),
        node_metrics=read_csv_by_position(root / "node_metrics.csv"),
        neighborhood_metrics=read_csv_by_position(root / "neighborhood_metrics.csv"),
        propagation_metrics=read_csv_by_position(root / "propagation_metrics.csv"),
        assignment_positions=tuple(read_csv_rows(root / "assignment_positions.csv")),
    )
    if validate:
        from .validation import validate_exposure_network_package

        validate_exposure_network_package(package, check_edges=False)
    return package


def load_edge_index(
    package: ExposureNetworkPackage,
    *,
    edge_file: str = EDGE_FILE_FULL,
    target_positions: set[str] | None = None,
    source_positions: set[str] | None = None,
) -> ExposureEdgeIndex:
    path = package.path(edge_file)
    if not path.exists():
        raise RuntimeError(f"Edge file not found: {path}")
    target_filter = {str(value) for value in target_positions} if target_positions is not None else None
    source_filter = {str(value) for value in source_positions} if source_positions is not None else None
    incoming: dict[str, list[ExposureEdge]] = defaultdict(list)
    outgoing: dict[str, list[ExposureEdge]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = str(row["source_position_id"])
            target = str(row["target_position_id"])
            if target_filter is not None and target not in target_filter:
                continue
            if source_filter is not None and source not in source_filter:
                continue
            if source == target:
                continue
            edge = ExposureEdge.from_row(row)
            incoming[target].append(edge)
            outgoing[source].append(edge)
    incoming_sorted = {
        target: tuple(sorted(edges, key=lambda edge: (edge.rank_for_receiver or 10**9, -edge.exposure_weight, edge.source_position_id)))
        for target, edges in incoming.items()
    }
    outgoing_sorted = {
        source: tuple(sorted(edges, key=lambda edge: (-edge.exposure_weight, edge.target_position_id)))
        for source, edges in outgoing.items()
    }
    return ExposureEdgeIndex(edge_file=edge_file, incoming_by_target=incoming_sorted, outgoing_by_source=outgoing_sorted)

