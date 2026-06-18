from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from common import GRAPH_ID, REPORTS_DIR, SUBSTRATE_DIR, artifact_path, ensure_dirs, input_path, write_json


REQUIRED_COLUMNS = {
    "edges_prompt_top30.csv": {
        "source_position_id",
        "target_position_id",
        "raw_weight",
        "total_events",
        "interaction_types",
        "exposure_weight",
        "rank_for_receiver",
    },
    "edges_full.csv": {
        "source_position_id",
        "target_position_id",
        "raw_weight",
        "total_events",
        "interaction_types",
        "exposure_weight",
        "rank_for_receiver",
    },
    "node_metrics.csv": {
        "position_id",
        "in_degree",
        "out_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "eigenvector_centrality",
        "approx_betweenness",
        "local_clustering",
        "community_id",
        "bridge_score",
        "in_largest_component",
        "has_prompt_peer_capacity",
    },
    "neighborhood_metrics.csv": {
        "position_id",
        "incoming_peer_count",
        "outgoing_receiver_count",
        "incoming_exposure_weight",
        "outgoing_visibility_weight",
        "incoming_top1_share",
        "incoming_top5_share",
        "incoming_effective_peer_count",
        "incoming_peer_community_count",
        "cross_community_incoming_share",
        "h2_neighborhood_activation_readiness",
        "dominant_structural_role",
    },
    "propagation_metrics.csv": {
        "position_id",
        "prompt_topk_out_reach_count",
        "two_hop_reach_count",
        "combined_two_step_reach_count",
        "cascade_reach_potential",
        "h3_central_susceptible_sender_readiness",
        "h4_central_resilient_sender_dampening_capacity",
    },
    "assignment_positions.csv": {
        "assignment_rank",
        "position_id",
        "display_role",
        "community_id",
        "weighted_in_degree",
        "weighted_out_degree",
        "bridge_score",
        "prompt_ready",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_columns(path: Path, required: set[str]) -> list[str]:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return sorted(columns)


def _row_count(path: Path) -> int:
    if path.suffix == ".json":
        return 1
    return sum(1 for _ in path.open("r", encoding="utf-8")) - 1


def _build_pilot_slice() -> dict[str, object]:
    assignment = pd.read_csv(input_path("assignment_positions.csv"))
    node_metrics = pd.read_csv(input_path("node_metrics.csv"))

    pilot = assignment.sort_values("assignment_rank").head(60).copy()
    pilot = pilot.merge(
        node_metrics[
            [
                "position_id",
                "in_degree",
                "out_degree",
                "in_largest_component",
                "has_prompt_peer_capacity",
            ]
        ],
        on="position_id",
        how="left",
    )
    pilot["sample_role"] = pilot["display_role"]
    pilot["has_prompt_peer_capacity"] = pilot["has_prompt_peer_capacity"].fillna(pilot["prompt_ready"])

    selected_ids = set(pilot["position_id"].astype(int))
    edges = pd.read_csv(input_path("edges_prompt_top30.csv"))
    pilot_edges = edges[
        edges["source_position_id"].astype(int).isin(selected_ids)
        & edges["target_position_id"].astype(int).isin(selected_ids)
    ].copy()

    graph = nx.from_pandas_edgelist(
        pilot_edges,
        source="source_position_id",
        target="target_position_id",
        edge_attr=["exposure_weight", "raw_weight", "total_events", "interaction_types", "rank_for_receiver"],
        create_using=nx.DiGraph,
    )
    graph.add_nodes_from(selected_ids)
    component_by_node: dict[int, int] = {}
    for component_id, component in enumerate(nx.weakly_connected_components(graph), start=1):
        for node_id in component:
            component_by_node[int(node_id)] = int(component_id)

    incoming_counts = pilot_edges.groupby("target_position_id").size()
    outgoing_counts = pilot_edges.groupby("source_position_id").size()
    pilot["slice_component_id"] = pilot["position_id"].map(component_by_node).fillna(0).astype(int)
    pilot["incoming_selected_edges"] = pilot["position_id"].map(incoming_counts).fillna(0).astype(int)
    pilot["outgoing_selected_edges"] = pilot["position_id"].map(outgoing_counts).fillna(0).astype(int)

    output_columns = [
        "position_id",
        "sample_role",
        "in_degree",
        "out_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "eigenvector_centrality",
        "approx_betweenness",
        "local_clustering",
        "community_id",
        "bridge_score",
        "has_prompt_peer_capacity",
        "incoming_top1_share",
        "incoming_top5_share",
        "incoming_effective_peer_count",
        "incoming_peer_community_count",
        "cross_community_incoming_share",
        "h2_neighborhood_activation_readiness",
        "dominant_structural_role",
        "prompt_topk_out_reach_count",
        "two_hop_reach_count",
        "combined_two_step_reach_count",
        "cascade_reach_potential",
        "h3_central_susceptible_sender_readiness",
        "h4_central_resilient_sender_dampening_capacity",
        "slice_component_id",
        "incoming_selected_edges",
        "outgoing_selected_edges",
    ]
    pilot.loc[:, output_columns].to_csv(input_path("pilot_60_position_slice.csv"), index=False)
    pilot_edges.to_csv(input_path("pilot_60_position_slice_edges.csv"), index=False)
    return {
        "pilot_nodes": int(len(pilot)),
        "pilot_edges": int(len(pilot_edges)),
        "source": "assignment_positions.csv rows assignment_rank <= 60",
        "edge_source": "edges_prompt_top30.csv induced over pilot positions",
    }


def prepare_inputs() -> dict[str, object]:
    ensure_dirs()
    manifest = json.loads(input_path("manifest.json").read_text(encoding="utf-8"))
    if manifest.get("graph_id") != GRAPH_ID:
        raise ValueError(f"Expected graph_id {GRAPH_ID}, found {manifest.get('graph_id')}")

    validated: list[dict[str, object]] = []
    for filename, required_columns in REQUIRED_COLUMNS.items():
        source = input_path(filename)
        if not source.exists():
            raise FileNotFoundError(f"Required canonical input is missing: {source}")
        columns = _validate_columns(source, required_columns)
        validated.append(
            {
                "logical_file": filename,
                "source": artifact_path(source),
                "rows": int(_row_count(source)),
                "columns": columns,
                "sha256": _sha256(source),
            }
        )

    pilot = _build_pilot_slice()
    for filename in ("pilot_60_position_slice.csv", "pilot_60_position_slice_edges.csv"):
        path = input_path(filename)
        validated.append(
            {
                "logical_file": filename,
                "source": artifact_path(path),
                "rows": int(_row_count(path)),
                "columns": sorted(pd.read_csv(path, nrows=0).columns),
                "sha256": _sha256(path),
                "generated": True,
            }
        )

    direction = {
        "edge_orientation": manifest.get("edge_direction"),
        "interpretation": manifest.get("edge_meaning"),
        "interaction_weight_formula": manifest.get("interaction_weight_formula"),
        "weight_processing": manifest.get("weight_processing"),
    }
    payload = {
        "graph_id": manifest.get("graph_id"),
        "canonical_graph_root": artifact_path(SUBSTRATE_DIR),
        "input_files": validated,
        "pilot_slice": pilot,
        "direction": direction,
        "status": "ready",
    }
    write_json(REPORTS_DIR / "input_manifest.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(prepare_inputs(), indent=2))
