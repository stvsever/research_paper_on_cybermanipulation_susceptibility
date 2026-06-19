from __future__ import annotations

import json

import pandas as pd
import networkx as nx

from common import (
    DERIVED_DIR,
    PROMPT_TOP_K,
    REPORTS_DIR,
    ROLE_DEFINITIONS,
    ROLE_LABELS,
    TABLES_DIR,
    ensure_dirs,
    input_path,
    write_json,
)


MECHANISM_ROLES = {
    "high_visibility_sender",
    "high_exposure_receiver",
    "bridge",
    "peripheral",
}


def _directed_graph(edges: pd.DataFrame, nodes: pd.Series) -> nx.DiGraph:
    graph = nx.from_pandas_edgelist(
        edges,
        source="source_position_id",
        target="target_position_id",
        edge_attr=["exposure_weight", "raw_weight", "total_events", "interaction_types"],
        create_using=nx.DiGraph,
    )
    graph.add_nodes_from(nodes.astype(int).tolist())
    return graph


def _reciprocal_dyad_count(graph: nx.DiGraph) -> int:
    reciprocal = 0
    seen: set[tuple[int, int]] = set()
    for source, target in graph.edges():
        pair = tuple(sorted((int(source), int(target))))
        if pair in seen:
            continue
        if graph.has_edge(target, source):
            reciprocal += 1
            seen.add(pair)
    return reciprocal


def analyze_network() -> dict[str, object]:
    ensure_dirs()
    nodes = pd.read_csv(input_path("pilot_60_position_slice.csv"))
    edges = pd.read_csv(input_path("pilot_60_position_slice_edges.csv"))
    full_edges = pd.read_csv(input_path("edges_full.csv"), usecols=["source_position_id", "target_position_id", "exposure_weight"])
    top_edges = pd.read_csv(input_path("edges_prompt_top30.csv"), usecols=["source_position_id", "target_position_id", "exposure_weight"])

    if len(nodes) != 60:
        raise ValueError(f"Expected exactly 60 pilot nodes, found {len(nodes)}")
    if not edges["exposure_weight"].between(0, 1).all():
        raise ValueError("Pilot edge exposure_weight values must be bounded in [0, 1]")

    selected_ids = set(nodes["position_id"].astype(int))
    invalid_edges = edges[
        ~edges["source_position_id"].astype(int).isin(selected_ids)
        | ~edges["target_position_id"].astype(int).isin(selected_ids)
    ]
    if not invalid_edges.empty:
        raise ValueError(f"Pilot edge table contains {len(invalid_edges)} edges outside the selected pilot nodes")

    graph = _directed_graph(edges, nodes["position_id"])
    edges = edges.copy()
    nodes = nodes.copy()
    nodes["display_role"] = nodes["sample_role"].where(nodes["sample_role"].isin(MECHANISM_ROLES), "context_position")
    nodes["display_role_label"] = nodes["display_role"].map(ROLE_LABELS)
    edges["visible_position_id"] = edges["source_position_id"].astype(int)
    edges["exposed_position_id"] = edges["target_position_id"].astype(int)
    edges["direction_label"] = edges["visible_position_id"].astype(str) + " -> " + edges["exposed_position_id"].astype(str)

    selected_out_weight = edges.groupby("source_position_id")["exposure_weight"].sum()
    selected_in_weight = edges.groupby("target_position_id")["exposure_weight"].sum()
    selected_out_edges = edges.groupby("source_position_id").size()
    selected_in_edges = edges.groupby("target_position_id").size()

    nodes["selected_visibility_sent_weight"] = nodes["position_id"].map(selected_out_weight).fillna(0.0)
    nodes["selected_exposure_received_weight"] = nodes["position_id"].map(selected_in_weight).fillna(0.0)
    nodes["selected_outgoing_exposure_edges"] = nodes["position_id"].map(selected_out_edges).fillna(0).astype(int)
    nodes["selected_incoming_exposure_edges"] = nodes["position_id"].map(selected_in_edges).fillna(0).astype(int)
    nodes["selected_sender_receiver_asymmetry"] = (
        nodes["selected_visibility_sent_weight"] - nodes["selected_exposure_received_weight"]
    )

    weak_components = list(nx.weakly_connected_components(graph))
    strongly_connected = list(nx.strongly_connected_components(graph))
    possible_edges = len(nodes) * (len(nodes) - 1)
    reciprocal_dyads = _reciprocal_dyad_count(graph)

    role_summary = (
        nodes.groupby(["display_role", "display_role_label"])
        .agg(
            nodes=("position_id", "size"),
            selected_visibility_sent_weight=("selected_visibility_sent_weight", "sum"),
            selected_exposure_received_weight=("selected_exposure_received_weight", "sum"),
            mean_full_weighted_out_degree=("weighted_out_degree", "mean"),
            mean_full_weighted_in_degree=("weighted_in_degree", "mean"),
            mean_bridge_score=("bridge_score", "mean"),
            prompt_ready_nodes=("has_prompt_peer_capacity", "sum"),
        )
        .reset_index()
        .sort_values("display_role")
    )
    role_definitions = pd.DataFrame(
        [
            {
                "display_role": role,
                "role_label": ROLE_LABELS[role],
                "definition": definition,
            }
            for role, definition in ROLE_DEFINITIONS.items()
        ]
    )

    top_senders = (
        nodes.sort_values(["selected_visibility_sent_weight", "weighted_out_degree"], ascending=False)
        .head(15)
        .loc[
            :,
            [
                "position_id",
                "display_role_label",
                "selected_visibility_sent_weight",
                "selected_outgoing_exposure_edges",
                "weighted_out_degree",
                "bridge_score",
                "community_id",
            ],
        ]
    )
    top_receivers = (
        nodes.sort_values(["selected_exposure_received_weight", "weighted_in_degree"], ascending=False)
        .head(15)
        .loc[
            :,
            [
                "position_id",
                "display_role_label",
                "selected_exposure_received_weight",
                "selected_incoming_exposure_edges",
                "weighted_in_degree",
                "bridge_score",
                "community_id",
            ],
        ]
    )

    summary = {
        "pilot_nodes": int(len(nodes)),
        "pilot_directed_edges": int(len(edges)),
        "pilot_directed_density": float(len(edges) / possible_edges),
        "pilot_total_exposure_weight": float(edges["exposure_weight"].sum()),
        "weak_components": int(len(weak_components)),
        "largest_weak_component": int(max((len(component) for component in weak_components), default=0)),
        "strong_components": int(len(strongly_connected)),
        "largest_strong_component": int(max((len(component) for component in strongly_connected), default=0)),
        "reciprocal_dyads": int(reciprocal_dyads),
        "all_nodes_prompt_ready": bool((nodes["in_degree"] >= PROMPT_TOP_K).all()),
        "full_exposure_edges_available": int(len(full_edges)),
        "top30_exposure_edges_available": int(len(top_edges)),
        "top_sender_position_id": int(top_senders.iloc[0]["position_id"]),
        "top_receiver_position_id": int(top_receivers.iloc[0]["position_id"]),
    }

    pilot_node_columns = [
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
        "has_prompt_peer_capacity",
        "incoming_top1_share",
        "incoming_top5_share",
        "incoming_effective_peer_count",
        "incoming_peer_community_count",
        "prompt_topk_out_reach_count",
        "two_hop_reach_count",
        "combined_two_step_reach_count",
        "cascade_reach_potential",
        "slice_component_id",
        "incoming_selected_edges",
        "outgoing_selected_edges",
        "display_role",
        "display_role_label",
        "selected_visibility_sent_weight",
        "selected_exposure_received_weight",
        "selected_outgoing_exposure_edges",
        "selected_incoming_exposure_edges",
        "selected_sender_receiver_asymmetry",
    ]
    nodes.loc[:, pilot_node_columns].to_csv(DERIVED_DIR / "pilot_nodes.csv", index=False)
    edges.to_csv(DERIVED_DIR / "pilot_edges.csv", index=False)
    pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()]).to_csv(
        TABLES_DIR / "pilot_network_summary.csv",
        index=False,
    )
    role_summary.to_csv(TABLES_DIR / "pilot_role_summary.csv", index=False)
    role_definitions.to_csv(TABLES_DIR / "role_definitions.csv", index=False)
    top_senders.to_csv(TABLES_DIR / "top_pilot_senders.csv", index=False)
    top_receivers.to_csv(TABLES_DIR / "top_pilot_receivers.csv", index=False)
    write_json(REPORTS_DIR / "pilot_network_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze_network(), indent=2))
