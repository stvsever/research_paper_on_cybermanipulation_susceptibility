from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from common import (
    DERIVED_DIR,
    INTERACTION_WEIGHTS,
    RANDOM_SEED,
    REPORTS_DIR,
    ROLE_DEFINITIONS,
    ROLE_LABELS,
    ROLE_PALETTE,
    TOKENS,
    artifact_path,
    ensure_dirs,
    input_path,
    write_json,
)


MAX_SAMPLE_N = 500
DEFAULT_SAMPLE_N = 60
MIN_SAMPLE_N = 30
SAMPLE_STEP = 10
SAMPLE_STEPS = list(range(MIN_SAMPLE_N, MAX_SAMPLE_N + 1, SAMPLE_STEP))

ROLE_TARGET_SHARES = {
    "high_visibility_sender": 0.22,
    "high_exposure_receiver": 0.22,
    "bridge": 0.20,
    "peripheral": 0.16,
    "context_position": 0.20,
}
ROLE_ORDER = list(ROLE_TARGET_SHARES.keys())

EDGE_MODE_EDGE_MULTIPLIERS = {
    "backbone": 1.6,
    "medium": 3.0,
    "dense": 6.0,
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return pd.Series(0.0, index=values.index)
    return (values - low) / (high - low)


def _sample_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _load_node_frame() -> pd.DataFrame:
    node_metrics = pd.read_csv(input_path("node_metrics.csv"))
    neighborhoods = pd.read_csv(
        input_path("neighborhood_metrics.csv"),
        usecols=[
            "position_id",
            "incoming_peer_count",
            "outgoing_receiver_count",
            "incoming_top1_share",
            "incoming_top5_share",
            "incoming_effective_peer_count",
            "incoming_peer_community_count",
            "incoming_community_entropy",
            "h2_neighborhood_activation_readiness",
        ],
    )
    propagation = pd.read_csv(
        input_path("propagation_metrics.csv"),
        usecols=[
            "position_id",
            "prompt_topk_out_reach_count",
            "two_hop_reach_count",
            "combined_two_step_reach_count",
            "cascade_reach_potential",
            "h3_central_susceptible_sender_readiness",
            "h4_central_resilient_sender_dampening_capacity",
        ],
    )
    nodes = node_metrics.merge(neighborhoods, on="position_id", how="left").merge(
        propagation,
        on="position_id",
        how="left",
    )
    nodes["in_largest_component"] = _as_bool(nodes["in_largest_component"])
    nodes["has_prompt_peer_capacity"] = _as_bool(nodes["has_prompt_peer_capacity"])
    numeric_columns = [
        column
        for column in nodes.columns
        if column not in {"position_id", "in_largest_component", "has_prompt_peer_capacity"}
    ]
    for column in numeric_columns:
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0.0)
    return nodes


def _add_scores(nodes: pd.DataFrame) -> pd.DataFrame:
    scored = nodes.copy()
    scored["sender_score"] = _minmax(scored["weighted_out_degree"])
    scored["receiver_score"] = _minmax(scored["weighted_in_degree"])
    scored["bridge_score_norm"] = _minmax(scored["bridge_score"])
    scored["betweenness_score"] = _minmax(scored["approx_betweenness"])
    scored["eigenvector_score"] = _minmax(scored["eigenvector_centrality"])
    scored["local_clustering_score"] = _minmax(scored["local_clustering"])
    scored["cascade_score"] = _minmax(scored["cascade_reach_potential"])
    scored["community_diversity_score"] = _minmax(scored["incoming_peer_community_count"])
    exposure_combined = 0.5 * scored["sender_score"] + 0.5 * scored["receiver_score"]
    scored["peripheral_score"] = 1.0 - exposure_combined
    scored["context_score"] = (
        0.30 * scored["community_diversity_score"]
        + 0.25 * scored["cascade_score"]
        + 0.20 * scored["eigenvector_score"]
        + 0.15 * scored["local_clustering_score"]
        + 0.10 * scored["receiver_score"]
    )
    scored["role_score_high_visibility_sender"] = scored["sender_score"]
    scored["role_score_high_exposure_receiver"] = scored["receiver_score"]
    scored["role_score_bridge"] = 0.60 * scored["bridge_score_norm"] + 0.40 * scored["betweenness_score"]
    scored["role_score_peripheral"] = scored["peripheral_score"]
    scored["role_score_context_position"] = scored["context_score"]
    exposure_rank = (scored["sender_score"] + scored["receiver_score"]).rank(method="first")
    scored["exposure_band"] = pd.qcut(
        exposure_rank,
        q=5,
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int)
    return scored


def _eligible_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    required = [
        "weighted_out_degree",
        "weighted_in_degree",
        "bridge_score",
        "community_id",
        "prompt_topk_out_reach_count",
    ]
    eligible = nodes[
        nodes["in_largest_component"]
        & nodes["has_prompt_peer_capacity"]
        & nodes[required].notna().all(axis=1)
    ].copy()
    if len(eligible) < MAX_SAMPLE_N:
        raise ValueError(f"Need at least {MAX_SAMPLE_N} eligible nodes, found {len(eligible)}")
    return eligible


def _role_deficit(role_counts: Counter[str], rank: int, role: str) -> float:
    target = ROLE_TARGET_SHARES[role] * rank
    return target - role_counts[role]


def _role_candidate_score(
    row: pd.Series,
    role: str,
    selected_ids: set[int],
    community_counts: Counter[int],
    band_counts: Counter[int],
    adjacent_selected_count: int,
    adjacent_selected_weight: float,
    rank: int,
    community_targets: dict[int, float],
    band_targets: dict[int, float],
) -> float:
    role_score = float(row[f"role_score_{role}"])
    community_id = int(row["community_id"])
    exposure_band = int(row["exposure_band"])
    community_expected = community_targets.get(community_id, 0.0) * rank
    band_expected = band_targets.get(exposure_band, 0.0) * rank
    community_deficit = max(0.0, community_expected - community_counts[community_id]) / max(1.0, community_expected)
    band_deficit = max(0.0, band_expected - band_counts[exposure_band]) / max(1.0, band_expected)
    community_saturation = max(0.0, community_counts[community_id] - community_expected) / max(1.0, rank)
    band_saturation = max(0.0, band_counts[exposure_band] - band_expected) / max(1.0, rank)
    connectivity_bonus = 0.55 * min(adjacent_selected_count, 6) / 6.0 + 0.45 * min(adjacent_selected_weight, 3.0) / 3.0
    community_fit = 0.65 * community_deficit + 0.35 * (1.0 - min(1.0, community_saturation * 6.0))
    band_fit = 0.65 * band_deficit + 0.35 * (1.0 - min(1.0, band_saturation * 6.0))
    cold_start_bonus = 0.15 if not selected_ids else 0.0
    return (
        0.52 * role_score
        + 0.22 * connectivity_bonus
        + 0.14 * community_fit
        + 0.07 * band_fit
        + 0.05 * float(row["candidate_degree_score"])
        + cold_start_bonus
        - 0.08 * community_saturation
        - 0.05 * band_saturation
    )


def select_representative_nested_sample(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    eligible = _eligible_nodes(nodes)
    eligible_ids = set(eligible["position_id"].astype(int))
    candidate_edges = edges[
        edges["source_position_id"].astype(int).isin(eligible_ids)
        & edges["target_position_id"].astype(int).isin(eligible_ids)
    ].copy()
    adjacency: dict[int, set[int]] = defaultdict(set)
    adjacency_weight: dict[int, dict[int, float]] = defaultdict(dict)
    for source, target, weight in candidate_edges[
        ["source_position_id", "target_position_id", "exposure_weight"]
    ].itertuples(index=False):
        source_id = int(source)
        target_id = int(target)
        edge_weight = float(weight)
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)
        adjacency_weight[source_id][target_id] = max(edge_weight, adjacency_weight[source_id].get(target_id, 0.0))
        adjacency_weight[target_id][source_id] = max(edge_weight, adjacency_weight[target_id].get(source_id, 0.0))

    eligible = eligible.copy()
    eligible["candidate_degree_score"] = _minmax(
        eligible["position_id"].map(lambda node_id: sum(adjacency_weight[int(node_id)].values()))
    )
    community_targets = eligible["community_id"].astype(int).value_counts(normalize=True).to_dict()
    band_targets = eligible["exposure_band"].astype(int).value_counts(normalize=True).to_dict()

    selected_ids: set[int] = set()
    role_counts: Counter[str] = Counter()
    community_counts: Counter[int] = Counter()
    band_counts: Counter[int] = Counter()
    selected_rows: list[dict[str, object]] = []
    remaining = eligible.set_index("position_id", drop=False)

    for rank in range(1, MAX_SAMPLE_N + 1):
        deficits = sorted(
            ((role, _role_deficit(role_counts, rank, role)) for role in ROLE_ORDER),
            key=lambda item: (item[1], ROLE_TARGET_SHARES[item[0]]),
            reverse=True,
        )
        role = deficits[0][0]
        best_id: int | None = None
        best_score = -10.0
        for row in remaining.itertuples(index=False):
            position_id = int(row.position_id)
            if position_id in selected_ids:
                continue
            row_series = remaining.loc[position_id]
            adjacent_ids = adjacency[position_id] & selected_ids
            adjacent_count = len(adjacent_ids)
            adjacent_weight = sum(adjacency_weight[position_id].get(node_id, 0.0) for node_id in adjacent_ids)
            score = _role_candidate_score(
                row_series,
                role,
                selected_ids,
                community_counts,
                band_counts,
                adjacent_count,
                adjacent_weight,
                rank,
                community_targets,
                band_targets,
            )
            if score > best_score or (math.isclose(score, best_score) and (best_id is None or position_id < best_id)):
                best_score = score
                best_id = position_id
        if best_id is None:
            raise ValueError(f"Could not select node for sample rank {rank}")
        selected_ids.add(best_id)
        selected = remaining.loc[best_id].copy()
        selected["sample_rank"] = rank
        selected["display_role"] = role
        selected["display_role_label"] = ROLE_LABELS[role]
        selected["selection_score"] = best_score
        selected["adjacent_selected_at_selection"] = len(adjacency[best_id] & (selected_ids - {best_id}))
        role_counts[role] += 1
        community_counts[int(selected["community_id"])] += 1
        band_counts[int(selected["exposure_band"])] += 1
        selected_rows.append(selected.to_dict())

    return pd.DataFrame(selected_rows)


def _sample_edges(edges: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(selected["position_id"].astype(int))
    sample_edges = edges[
        edges["source_position_id"].astype(int).isin(selected_ids)
        & edges["target_position_id"].astype(int).isin(selected_ids)
    ].copy()
    ranks = selected.set_index("position_id")["sample_rank"].to_dict()
    sample_edges["source_rank"] = sample_edges["source_position_id"].map(ranks)
    sample_edges["target_rank"] = sample_edges["target_position_id"].map(ranks)
    sample_edges["edge_min_sample_n"] = sample_edges[["source_rank", "target_rank"]].max(axis=1).astype(int)
    sample_edges["visible_position_id"] = sample_edges["source_position_id"].astype(int)
    sample_edges["exposed_position_id"] = sample_edges["target_position_id"].astype(int)
    return sample_edges.sort_values(["edge_min_sample_n", "source_position_id", "target_position_id"])


def _initial_position_for_new_node(
    node_id: int,
    selected_ids: set[int],
    previous_positions: dict[int, tuple[float, float]],
    adjacency_weight: dict[int, dict[int, float]],
) -> tuple[float, float]:
    positioned_neighbors = [
        (neighbor_id, adjacency_weight[node_id].get(neighbor_id, 0.0))
        for neighbor_id in adjacency_weight[node_id]
        if neighbor_id in previous_positions and neighbor_id in selected_ids
    ]
    if positioned_neighbors:
        total_weight = sum(weight for _, weight in positioned_neighbors) or 1.0
        x = sum(previous_positions[neighbor_id][0] * weight for neighbor_id, weight in positioned_neighbors) / total_weight
        y = sum(previous_positions[neighbor_id][1] * weight for neighbor_id, weight in positioned_neighbors) / total_weight
    else:
        angle = (node_id * 0.61803398875) % 1.0 * math.tau
        x = 0.28 * math.cos(angle)
        y = 0.28 * math.sin(angle)
    jitter_angle = (node_id * 0.754877666) % 1.0 * math.tau
    return (x + 0.035 * math.cos(jitter_angle), y + 0.035 * math.sin(jitter_angle))


def _normalize_positions(positions: dict[int, tuple[float, float]], node_ids: list[int]) -> dict[int, tuple[float, float]]:
    x_values = [float(positions[node_id][0]) for node_id in node_ids]
    y_values = [float(positions[node_id][1]) for node_id in node_ids]
    x_low = _sample_quantile(x_values, 0.03)
    x_high = _sample_quantile(x_values, 0.97)
    y_low = _sample_quantile(y_values, 0.03)
    y_high = _sample_quantile(y_values, 0.97)
    x_center = (x_low + x_high) / 2.0
    y_center = (y_low + y_high) / 2.0
    span = max(x_high - x_low, y_high - y_low, 1e-9)
    margin = 0.055
    scale = 1.0 - 2.0 * margin
    normalized: dict[int, tuple[float, float]] = {}
    for node_id in node_ids:
        x_raw, y_raw = positions[node_id]
        x = margin + scale * ((float(x_raw) - x_center) / span + 0.5)
        y = margin + scale * ((float(y_raw) - y_center) / span + 0.5)
        normalized[node_id] = (min(0.97, max(0.03, x)), min(0.97, max(0.03, y)))
    return normalized


def compute_step_layouts(selected: pd.DataFrame, sample_edges: pd.DataFrame) -> pd.DataFrame:
    edge_lookup = sample_edges.copy()
    adjacency_weight: dict[int, dict[int, float]] = defaultdict(dict)
    for source, target, weight in edge_lookup[["source_position_id", "target_position_id", "exposure_weight"]].itertuples(
        index=False
    ):
        source_id = int(source)
        target_id = int(target)
        edge_weight = float(weight)
        adjacency_weight[source_id][target_id] = max(edge_weight, adjacency_weight[source_id].get(target_id, 0.0))
        adjacency_weight[target_id][source_id] = max(edge_weight, adjacency_weight[target_id].get(source_id, 0.0))

    previous_positions: dict[int, tuple[float, float]] = {}
    rows: list[dict[str, object]] = []
    for sample_n in SAMPLE_STEPS:
        nodes_n = selected[selected["sample_rank"] <= sample_n]
        node_ids = nodes_n["position_id"].astype(int).tolist()
        selected_ids = set(node_ids)
        edges_n = edge_lookup[edge_lookup["edge_min_sample_n"] <= sample_n]
        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        for row in edges_n.itertuples(index=False):
            source = int(row.source_position_id)
            target = int(row.target_position_id)
            if source in selected_ids and target in selected_ids:
                graph.add_edge(source, target, exposure_weight=float(row.exposure_weight))

        initial_positions = {
            node_id: previous_positions.get(
                node_id,
                _initial_position_for_new_node(node_id, selected_ids, previous_positions, adjacency_weight),
            )
            for node_id in node_ids
        }
        layout_k = max(0.18, 0.86 / math.sqrt(sample_n / MIN_SAMPLE_N))
        iterations = 180 if sample_n <= 80 else 125
        raw_positions = nx.spring_layout(
            graph,
            pos=initial_positions,
            seed=RANDOM_SEED,
            weight="exposure_weight",
            k=layout_k,
            iterations=iterations,
        )
        previous_positions = {int(node_id): (float(x), float(y)) for node_id, (x, y) in raw_positions.items()}
        normalized = _normalize_positions(previous_positions, node_ids)
        for node_id in node_ids:
            rows.append(
                {
                    "sample_n": sample_n,
                    "position_id": node_id,
                    "layout_x": normalized[node_id][0],
                    "layout_y": normalized[node_id][1],
                }
            )
    return pd.DataFrame(rows)


def _summary_table(selected: pd.DataFrame, sample_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in SAMPLE_STEPS:
        nodes_n = selected[selected["sample_rank"] <= n]
        ids = set(nodes_n["position_id"].astype(int))
        edges_n = sample_edges[
            sample_edges["source_position_id"].astype(int).isin(ids)
            & sample_edges["target_position_id"].astype(int).isin(ids)
        ]
        role_counts = nodes_n["display_role"].value_counts().to_dict()
        rows.append(
            {
                "sample_n": n,
                "nodes": int(len(nodes_n)),
                "edges": int(len(edges_n)),
                "communities": int(nodes_n["community_id"].nunique()),
                "total_exposure_weight": float(edges_n["exposure_weight"].sum()),
                "mean_sender_reach": float(nodes_n["weighted_out_degree"].mean()),
                "mean_receiver_exposure": float(nodes_n["weighted_in_degree"].mean()),
                **{f"role_{role}": int(role_counts.get(role, 0)) for role in ROLE_ORDER},
            }
        )
    return pd.DataFrame(rows)


def _maximum_spanning_forest_indices(active_edges: pd.DataFrame) -> set[int]:
    graph = nx.Graph()
    for row in active_edges.itertuples(index=False):
        source = int(row.source_position_id)
        target = int(row.target_position_id)
        weight = float(row.exposure_weight)
        edge_index = int(row.edge_index)
        if graph.has_edge(source, target):
            if weight > float(graph[source][target]["weight"]):
                graph[source][target]["weight"] = weight
                graph[source][target]["edge_index"] = edge_index
        else:
            graph.add_edge(source, target, weight=weight, edge_index=edge_index)
    if graph.number_of_edges() == 0:
        return set()
    forest = nx.maximum_spanning_tree(graph, weight="weight")
    return {int(data["edge_index"]) for _, _, data in forest.edges(data=True)}


def _mode_cap(sample_n: int, active_edge_count: int, mode: str) -> int:
    if active_edge_count == 0:
        return 0
    minimum = max(1, sample_n - 1)
    cap = math.ceil(sample_n * EDGE_MODE_EDGE_MULTIPLIERS[mode])
    return min(active_edge_count, max(minimum, cap))


def compute_readable_backbone_edges(sample_edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, list[int]]]]:
    indexed_edges = sample_edges.reset_index(drop=True).copy()
    indexed_edges["edge_index"] = indexed_edges.index.astype(int)
    selections: dict[str, dict[str, list[int]]] = {}
    for sample_n in SAMPLE_STEPS:
        active = indexed_edges[indexed_edges["edge_min_sample_n"] <= sample_n].copy()
        active = active.sort_values(["exposure_weight", "total_events"], ascending=[False, False])
        skeleton = _maximum_spanning_forest_indices(active)
        sorted_indices = [int(index) for index in active["edge_index"].tolist()]
        selections[str(sample_n)] = {}
        for mode in ("backbone", "medium", "dense"):
            cap = _mode_cap(sample_n, len(active), mode)
            chosen = list(skeleton)
            chosen_set = set(chosen)
            for edge_index in sorted_indices:
                if len(chosen) >= cap:
                    break
                if edge_index not in chosen_set:
                    chosen.append(edge_index)
                    chosen_set.add(edge_index)
            selections[str(sample_n)][mode] = sorted(chosen)
    return indexed_edges, selections


def compute_sample_quality(
    selected: pd.DataFrame,
    sample_edges: pd.DataFrame,
    edge_selections: dict[str, dict[str, list[int]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_n in SAMPLE_STEPS:
        nodes_n = selected[selected["sample_rank"] <= sample_n]
        ids = set(nodes_n["position_id"].astype(int))
        active_edges = sample_edges[sample_edges["edge_min_sample_n"] <= sample_n]
        graph = nx.Graph()
        graph.add_nodes_from(ids)
        for row in active_edges.itertuples(index=False):
            source = int(row.source_position_id)
            target = int(row.target_position_id)
            if source in ids and target in ids:
                graph.add_edge(source, target)
        component_sizes = [len(component) for component in nx.connected_components(graph)] if graph.number_of_nodes() else [0]
        largest_component_share = max(component_sizes) / max(1, len(ids))
        role_counts = nodes_n["display_role"].value_counts().to_dict()
        role_share_deviations = [
            abs((role_counts.get(role, 0) / max(1, sample_n)) - ROLE_TARGET_SHARES[role])
            for role in ROLE_ORDER
        ]
        rows.append(
            {
                "sample_n": sample_n,
                "nodes": int(len(nodes_n)),
                "all_edges": int(len(active_edges)),
                "backbone_edges": int(len(edge_selections[str(sample_n)]["backbone"])),
                "medium_edges": int(len(edge_selections[str(sample_n)]["medium"])),
                "dense_edges": int(len(edge_selections[str(sample_n)]["dense"])),
                "communities": int(nodes_n["community_id"].nunique()),
                "largest_component_share": float(largest_component_share),
                "role_max_share_deviation": float(max(role_share_deviations) if role_share_deviations else 0.0),
                "total_exposure_weight": float(active_edges["exposure_weight"].sum()),
                "mean_sender_reach": float(nodes_n["weighted_out_degree"].mean()),
                "mean_receiver_exposure": float(nodes_n["weighted_in_degree"].mean()),
                **{f"role_{role}": int(role_counts.get(role, 0)) for role in ROLE_ORDER},
            }
        )
    return pd.DataFrame(rows)


def _top_neighbors(sample_edges: pd.DataFrame) -> tuple[dict[int, list[dict[str, object]]], dict[int, list[dict[str, object]]]]:
    incoming: dict[int, list[dict[str, object]]] = defaultdict(list)
    outgoing: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in sample_edges.itertuples(index=False):
        item = {
            "source": int(row.source_position_id),
            "target": int(row.target_position_id),
            "weight": round(float(row.exposure_weight), 4),
            "types": str(row.interaction_types),
        }
        incoming[int(row.target_position_id)].append(item)
        outgoing[int(row.source_position_id)].append(item)
    for mapping in (incoming, outgoing):
        for node, items in mapping.items():
            items.sort(key=lambda item: item["weight"], reverse=True)
            mapping[node] = items[:8]
    return incoming, outgoing


def _html_escape_json(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def _build_payload(
    selected: pd.DataFrame,
    sample_edges: pd.DataFrame,
    layouts: pd.DataFrame,
    quality: pd.DataFrame,
    edge_selections: dict[str, dict[str, list[int]]],
) -> dict[str, object]:
    incoming, outgoing = _top_neighbors(sample_edges)
    max_out = float(selected["weighted_out_degree"].max()) or 1.0
    max_in = float(selected["weighted_in_degree"].max()) or 1.0
    nodes = []
    for row in selected.itertuples(index=False):
        position_id = int(row.position_id)
        nodes.append(
            {
                "id": position_id,
                "rank": int(row.sample_rank),
                "role": str(row.display_role),
                "roleLabel": str(row.display_role_label),
                "community": int(row.community_id),
                "x": round(float(row.layout_x), 6),
                "y": round(float(row.layout_y), 6),
                "weightedOut": round(float(row.weighted_out_degree), 4),
                "weightedIn": round(float(row.weighted_in_degree), 4),
                "outNorm": round(float(row.weighted_out_degree) / max_out, 6),
                "inNorm": round(float(row.weighted_in_degree) / max_in, 6),
                "bridge": round(float(row.bridge_score), 6),
                "incomingPeers": int(row.incoming_peer_count),
                "outgoingReceivers": int(row.outgoing_receiver_count),
                "twoHopReach": int(row.two_hop_reach_count),
                "cascade": round(float(row.cascade_reach_potential), 6),
                "selectionScore": round(float(row.selection_score), 6),
                "incoming": incoming.get(position_id, []),
                "outgoing": outgoing.get(position_id, []),
            }
        )
    edges = [
        {
            "index": int(row.edge_index),
            "source": int(row.source_position_id),
            "target": int(row.target_position_id),
            "minN": int(row.edge_min_sample_n),
            "weight": round(float(row.exposure_weight), 5),
            "events": int(row.total_events),
            "types": str(row.interaction_types),
            "rankForReceiver": None if pd.isna(row.rank_for_receiver) else float(row.rank_for_receiver),
        }
        for row in sample_edges.itertuples(index=False)
    ]
    layout_payload: dict[str, dict[str, list[float]]] = {}
    for sample_n, group in layouts.groupby("sample_n", sort=True):
        layout_payload[str(int(sample_n))] = {
            str(int(row.position_id)): [round(float(row.layout_x), 6), round(float(row.layout_y), 6)]
            for row in group.itertuples(index=False)
        }
    return {
        "defaults": {
            "minN": MIN_SAMPLE_N,
            "maxN": MAX_SAMPLE_N,
            "defaultN": DEFAULT_SAMPLE_N,
            "step": SAMPLE_STEP,
            "edgeMode": "backbone",
        },
        "direction": "TargetUserId -> SourceUserId",
        "interactionWeights": INTERACTION_WEIGHTS,
        "roles": {
            role: {
                "label": ROLE_LABELS[role],
                "color": ROLE_PALETTE[role],
                "definition": ROLE_DEFINITIONS[role],
                "targetShare": ROLE_TARGET_SHARES[role],
            }
            for role in ROLE_ORDER
        },
        "palette": {
            "ink": TOKENS["ink"],
            "muted": TOKENS["muted"],
            "grid": TOKENS["grid"],
            "neutralDark": TOKENS["neutral_dark"],
        },
        "nodes": nodes,
        "edges": edges,
        "layouts": layout_payload,
        "edgeSelections": edge_selections,
        "summary": quality.to_dict(orient="records"),
    }


def _render_html(payload: dict[str, object]) -> str:
    data_json = _html_escape_json(payload)
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Directed Exposure Map</title>
  <style>
    :root {
      --surface: #fcfcfd;
      --panel: #ffffff;
      --ink: #1f2430;
      --muted: #6f768a;
      --grid: #e6e8f0;
      --axis: #d7dbe7;
      --accent: #f0986e;
      --blue: #a3befa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }
    main {
      width: min(1380px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 30px 0 46px;
    }
    h1 { margin: 0 0 8px; font-size: 34px; line-height: 1.08; }
    h2 { margin: 0 0 10px; font-size: 16px; }
    p { color: #303746; }
    .summary {
      max-width: 980px;
      margin: 0 0 20px;
      color: #303746;
    }
    .layout {
      display: grid;
      grid-template-columns: 300px minmax(640px, 1fr) 300px;
      gap: 14px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 10px;
      padding: 16px;
    }
    .controls label {
      display: block;
      margin: 14px 0 6px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .11em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .value-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font-weight: 800;
      margin-bottom: 4px;
    }
    input[type="range"] { width: 100%; }
    select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--axis);
      border-radius: 7px;
      background: white;
      color: var(--ink);
      font-weight: 700;
      padding: 0 8px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 10px;
      background: #fbfbfc;
    }
    .metric span {
      display: block;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .09em;
      color: var(--muted);
      font-weight: 800;
    }
    .metric strong { display: block; margin-top: 3px; font-size: 19px; }
    .canvas-wrap {
      position: relative;
      min-height: 760px;
      overflow: hidden;
    }
    canvas {
      display: block;
      width: 100%;
      height: 760px;
      background: #fff;
      border-radius: 8px;
    }
    .tooltip {
      position: absolute;
      pointer-events: none;
      display: none;
      max-width: 260px;
      padding: 10px 12px;
      border: 1px solid var(--axis);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 10px 30px rgba(31, 36, 48, 0.12);
      font-size: 12px;
      color: var(--ink);
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin-top: 10px;
      font-size: 12px;
      color: #303746;
    }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .inspector h2 { margin-bottom: 2px; }
    .muted { color: var(--muted); font-size: 13px; }
    .rows { margin-top: 14px; }
    .row {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      border-bottom: 1px solid var(--grid);
      padding: 7px 0;
      font-size: 13px;
    }
    .row span:first-child { color: var(--muted); }
    .row strong { text-align: right; }
    .edge-list {
      margin: 14px 0 0;
      padding: 0;
      list-style: none;
      max-height: 185px;
      overflow: auto;
      border-top: 1px solid var(--grid);
    }
    .edge-list li {
      padding: 7px 0;
      border-bottom: 1px solid var(--grid);
      font-size: 12px;
      color: #303746;
    }
    .method {
      margin-top: 18px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .method code {
      font-family: "SF Mono", Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f4f5f7;
      border: 1px solid var(--grid);
      border-radius: 5px;
      padding: 2px 4px;
    }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .canvas-wrap { min-height: 620px; }
      canvas { height: 620px; }
      .method { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <h1>Interactive Directed Exposure Map</h1>
  <p class="summary">This explorer expands one deterministic nested sample from the empirical PolitiSky24 exposure graph. A directed edge means <strong>visible target -> exposed source</strong>: if a source user liked, reposted, or quoted a target user, the target was plausibly visible to the source.</p>

  <section class="layout">
    <aside class="panel controls">
      <h2>Controls</h2>
      <label for="sampleSize">Sample size</label>
      <div class="value-row"><span>Nodes</span><strong id="sampleSizeValue">60</strong></div>
      <input id="sampleSize" type="range" min="30" max="500" value="60" step="10">

      <label for="edgeMode">Edge visibility</label>
      <select id="edgeMode">
        <option value="backbone">backbone</option>
        <option value="medium">medium</option>
        <option value="dense">dense</option>
      </select>

      <label for="colorMode">Color mode</label>
      <select id="colorMode">
        <option value="role">structural role</option>
        <option value="community">community</option>
      </select>

      <div class="metric-grid">
        <div class="metric"><span>Visible edges</span><strong id="edgeCount">--</strong></div>
        <div class="metric"><span>Communities</span><strong id="communityCount">--</strong></div>
        <div class="metric"><span>Component</span><strong id="componentShare">--</strong></div>
        <div class="metric"><span>Role dev.</span><strong id="roleDeviation">--</strong></div>
        <div class="metric"><span>Total weight</span><strong id="totalWeight">--</strong></div>
        <div class="metric"><span>Selected N</span><strong id="selectedN">--</strong></div>
      </div>
      <p class="muted">The sample is nested: increasing N only adds nodes. It never resamples the already selected positions.</p>
    </aside>

    <section class="panel canvas-wrap">
      <canvas id="networkCanvas"></canvas>
      <div id="tooltip" class="tooltip"></div>
      <div id="legend" class="legend"></div>
    </section>

    <aside class="panel inspector">
      <h2 id="inspectorTitle">Select a node</h2>
      <p id="inspectorSubtitle" class="muted">Hover or click a node to inspect exposure position.</p>
      <div class="rows" id="inspectorRows"></div>
      <h2 style="margin-top:18px;">Top incoming exposures</h2>
      <ul class="edge-list" id="incomingList"></ul>
      <h2 style="margin-top:18px;">Top outgoing visibility</h2>
      <ul class="edge-list" id="outgoingList"></ul>
    </aside>
  </section>

  <section class="method">
    <div class="panel">
      <h2>Edge construction</h2>
      <p><code>exposure_raw_weight = 0.35 * Like + 0.80 * Repost + 0.90 * Quote</code></p>
      <p>Repeated interactions are summed, log-compressed, and normalized to <code>[0, 1]</code>. The browser filters precomputed edges; it does not recompute the graph.</p>
    </div>
    <div class="panel">
      <h2>Sampling rule</h2>
      <p>Eligible nodes are in the largest component, have prompt-peer capacity, and have sender, receiver, bridge, community, and propagation metrics. A greedy sampler keeps every prefix role-balanced while also rewarding proportional community coverage, exposure-band coverage, and induced connectivity.</p>
    </div>
  </section>
</main>

<script id="network-data" type="application/json">__DATA__</script>
<script>
const payload = JSON.parse(document.getElementById("network-data").textContent);
const nodes = payload.nodes;
const edges = payload.edges;
const layouts = payload.layouts;
const nodeById = new Map(nodes.map(node => [node.id, node]));
const edgeByIndex = new Map(edges.map(edge => [edge.index, edge]));
const summaryByN = new Map(payload.summary.map(row => [row.sample_n, row]));
const canvas = document.getElementById("networkCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const sampleSize = document.getElementById("sampleSize");
const sampleSizeValue = document.getElementById("sampleSizeValue");
const edgeMode = document.getElementById("edgeMode");
const colorMode = document.getElementById("colorMode");
const inspectorRows = document.getElementById("inspectorRows");
const inspectorTitle = document.getElementById("inspectorTitle");
const inspectorSubtitle = document.getElementById("inspectorSubtitle");
const incomingList = document.getElementById("incomingList");
const outgoingList = document.getElementById("outgoingList");
const legend = document.getElementById("legend");
const communityPalette = ["#F0986E", "#A3BEFA", "#F390CA", "#A3D576", "#FFE15B", "#B9B0F8", "#8FD6D2", "#D8C09D", "#B8C2CC", "#F6B1A3"];
let hoveredNode = null;
let selectedNode = null;
let drawnNodes = [];

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * scale);
  canvas.height = Math.floor(rect.height * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function colorFor(node) {
  if (colorMode.value === "community") {
    return communityPalette[Math.abs(node.community) % communityPalette.length];
  }
  return payload.roles[node.role]?.color || "#C5CAD3";
}

function activeNodes() {
  const n = Number(sampleSize.value);
  return nodes.filter(node => node.rank <= n);
}

function edgeLimitMultiplier() {
  if (edgeMode.value === "dense") return "dense";
  if (edgeMode.value === "medium") return "medium";
  return "backbone";
}

function activeEdges(activeNodeIds) {
  const n = Number(sampleSize.value);
  const mode = edgeLimitMultiplier();
  const selectedIndexes = payload.edgeSelections[String(n)]?.[mode] || [];
  return selectedIndexes
    .map(index => edgeByIndex.get(index))
    .filter(edge => edge && edge.minN <= n && activeNodeIds.has(edge.source) && activeNodeIds.has(edge.target));
}

function toScreen(node, width, height) {
  const n = Number(sampleSize.value);
  const layout = layouts[String(n)]?.[String(node.id)] || [node.x, node.y];
  const pad = 30;
  return {
    x: pad + layout[0] * (width - pad * 2),
    y: pad + layout[1] * (height - pad * 2),
  };
}

function quantile(values, q) {
  if (!values.length) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)));
  return sorted[index] || 1;
}

function radiusFor(node, currentNodes) {
  const n = currentNodes.length;
  const cap = Math.max(1, quantile(currentNodes.map(item => item.weightedOut), 0.94));
  const norm = Math.min(1, Math.max(0, node.weightedOut / cap));
  const minRadius = n <= 80 ? 4.4 : n <= 250 ? 3.4 : 2.5;
  const maxRadius = n <= 80 ? 15.5 : n <= 250 ? 11.5 : 8.0;
  return minRadius + (maxRadius - minRadius) * Math.sqrt(norm);
}

function drawArrow(source, target, width, alpha, color) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  if (length < 2) return;
  const unitX = dx / length;
  const unitY = dy / length;
  const endX = target.x - unitX * 7;
  const endY = target.y - unitY * 7;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(source.x, source.y);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  const arrowSize = 5 + width;
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(endX - unitX * arrowSize - unitY * arrowSize * 0.55, endY - unitY * arrowSize + unitX * arrowSize * 0.55);
  ctx.lineTo(endX - unitX * arrowSize + unitY * arrowSize * 0.55, endY - unitY * arrowSize - unitX * arrowSize * 0.55);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

function draw() {
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const currentNodes = activeNodes();
  const ids = new Set(currentNodes.map(node => node.id));
  const currentEdges = activeEdges(ids);
  const screen = new Map(currentNodes.map(node => [node.id, toScreen(node, width, height)]));
  const summary = summaryByN.get(Number(sampleSize.value));
  sampleSizeValue.textContent = sampleSize.value;
  document.getElementById("selectedN").textContent = sampleSize.value;
  document.getElementById("edgeCount").textContent = fmt(currentEdges.length, 0);
  document.getElementById("communityCount").textContent = summary ? fmt(summary.communities, 0) : "--";
  document.getElementById("componentShare").textContent = summary ? `${fmt(summary.largest_component_share * 100, 0)}%` : "--";
  document.getElementById("roleDeviation").textContent = summary ? `${fmt(summary.role_max_share_deviation * 100, 1)} pp` : "--";
  document.getElementById("totalWeight").textContent = summary ? fmt(summary.total_exposure_weight, 1) : "--";

  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(230, 232, 240, 0.42)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 72) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 72) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  currentEdges.forEach(edge => {
    const source = screen.get(edge.source);
    const target = screen.get(edge.target);
    if (!source || !target) return;
    drawArrow(source, target, 0.2 + edge.weight * 1.8, 0.07 + edge.weight * 0.22, "#6F768A");
  });

  drawnNodes = currentNodes.map(node => {
    const point = screen.get(node.id);
    const radius = radiusFor(node, currentNodes);
    const isSelected = selectedNode && selectedNode.id === node.id;
    const isHovered = hoveredNode && hoveredNode.id === node.id;
    ctx.globalAlpha = 0.98;
    ctx.fillStyle = colorFor(node);
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = isSelected || isHovered ? 3 : 1 + node.inNorm * 4;
    ctx.strokeStyle = isSelected || isHovered ? "#1F2430" : "#FFFFFF";
    ctx.stroke();
    ctx.globalAlpha = 1;
    return { node, x: point.x, y: point.y, r: radius };
  });
  renderLegend();
}

function renderLegend() {
  if (colorMode.value === "community") {
    const communities = [...new Set(activeNodes().map(node => node.community))].sort((a, b) => a - b).slice(0, 10);
    legend.innerHTML = communities.map(id => `<span class="legend-item"><span class="swatch" style="background:${communityPalette[Math.abs(id) % communityPalette.length]}"></span>community ${id}</span>`).join("");
    return;
  }
  legend.innerHTML = Object.values(payload.roles).map(role => `<span class="legend-item"><span class="swatch" style="background:${role.color}"></span>${role.label}</span>`).join("");
}

function nodeAt(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  for (let i = drawnNodes.length - 1; i >= 0; i--) {
    const item = drawnNodes[i];
    const dx = x - item.x;
    const dy = y - item.y;
    if (Math.sqrt(dx * dx + dy * dy) <= item.r + 4) return item.node;
  }
  return null;
}

function showTooltip(node, event) {
  if (!node) {
    tooltip.style.display = "none";
    return;
  }
  tooltip.innerHTML = `<strong>position ${node.id}</strong><br>${node.roleLabel}<br>community ${node.community}<br>sender reach ${fmt(node.weightedOut, 1)} · receiver exposure ${fmt(node.weightedIn, 1)}<br>bridge ${fmt(node.bridge, 3)}`;
  const rect = canvas.getBoundingClientRect();
  tooltip.style.left = `${event.clientX - rect.left + 14}px`;
  tooltip.style.top = `${event.clientY - rect.top + 14}px`;
  tooltip.style.display = "block";
}

function renderInspector(node) {
  if (!node) {
    inspectorTitle.textContent = "Select a node";
    inspectorSubtitle.textContent = "Hover or click a node to inspect exposure position.";
    inspectorRows.innerHTML = "";
    incomingList.innerHTML = "";
    outgoingList.innerHTML = "";
    return;
  }
  inspectorTitle.textContent = `position ${node.id}`;
  inspectorSubtitle.textContent = `${node.roleLabel} · community ${node.community}`;
  const rows = [
    ["Sample rank", node.rank],
    ["Sender reach", fmt(node.weightedOut, 2)],
    ["Receiver exposure", fmt(node.weightedIn, 2)],
    ["Bridge score", fmt(node.bridge, 3)],
    ["Incoming peers", node.incomingPeers],
    ["Outgoing receivers", node.outgoingReceivers],
    ["Two-hop reach", node.twoHopReach],
    ["Cascade potential", fmt(node.cascade, 3)],
  ];
  inspectorRows.innerHTML = rows.map(([label, value]) => `<div class="row"><span>${label}</span><strong>${value}</strong></div>`).join("");
  incomingList.innerHTML = node.incoming.length ? node.incoming.map(edge => `<li>${edge.source} -> <strong>${edge.target}</strong> · weight ${fmt(edge.weight, 3)} · ${edge.types}</li>`).join("") : `<li>No incoming edge in selected sample.</li>`;
  outgoingList.innerHTML = node.outgoing.length ? node.outgoing.map(edge => `<li><strong>${edge.source}</strong> -> ${edge.target} · weight ${fmt(edge.weight, 3)} · ${edge.types}</li>`).join("") : `<li>No outgoing edge in selected sample.</li>`;
}

canvas.addEventListener("mousemove", event => {
  hoveredNode = nodeAt(event);
  showTooltip(hoveredNode, event);
  draw();
});
canvas.addEventListener("mouseleave", () => {
  hoveredNode = null;
  tooltip.style.display = "none";
  draw();
});
canvas.addEventListener("click", event => {
  selectedNode = nodeAt(event);
  renderInspector(selectedNode);
  draw();
});
[sampleSize, edgeMode, colorMode].forEach(control => control.addEventListener("input", () => {
  if (selectedNode && selectedNode.rank > Number(sampleSize.value)) selectedNode = null;
  renderInspector(selectedNode);
  draw();
}));
window.addEventListener("resize", draw);
renderInspector(null);
draw();
</script>
</body>
</html>
""".replace("__DATA__", data_json)


def build_interactive_exposure_map() -> dict[str, object]:
    ensure_dirs()
    nodes = _add_scores(_load_node_frame())
    edges = pd.read_csv(input_path("edges_prompt_top30.csv"))
    for column in ["source_position_id", "target_position_id", "total_events"]:
        edges[column] = pd.to_numeric(edges[column], errors="coerce").fillna(0).astype(int)
    for column in ["exposure_weight", "raw_weight", "rank_for_receiver"]:
        edges[column] = pd.to_numeric(edges[column], errors="coerce")

    selected = select_representative_nested_sample(nodes, edges)
    sample_edges = _sample_edges(edges, selected)
    sample_edges, edge_selections = compute_readable_backbone_edges(sample_edges)
    layouts = compute_step_layouts(selected, sample_edges)
    default_layout = layouts[layouts["sample_n"] == MAX_SAMPLE_N][["position_id", "layout_x", "layout_y"]]
    selected = selected.drop(columns=["layout_x", "layout_y"], errors="ignore").merge(
        default_layout,
        on="position_id",
        how="left",
    )
    summary = _summary_table(selected, sample_edges)
    quality = compute_sample_quality(selected, sample_edges, edge_selections)

    selected_columns = [
        "position_id",
        "sample_rank",
        "display_role",
        "display_role_label",
        "selection_score",
        "community_id",
        "layout_x",
        "layout_y",
        "weighted_in_degree",
        "weighted_out_degree",
        "eigenvector_centrality",
        "approx_betweenness",
        "local_clustering",
        "bridge_score",
        "has_prompt_peer_capacity",
        "incoming_peer_count",
        "outgoing_receiver_count",
        "incoming_effective_peer_count",
        "incoming_peer_community_count",
        "prompt_topk_out_reach_count",
        "two_hop_reach_count",
        "combined_two_step_reach_count",
        "cascade_reach_potential",
        "adjacent_selected_at_selection",
    ]
    edge_columns = [
        "edge_index",
        "source_position_id",
        "target_position_id",
        "visible_position_id",
        "exposed_position_id",
        "edge_min_sample_n",
        "exposure_weight",
        "raw_weight",
        "total_events",
        "interaction_types",
        "rank_for_receiver",
    ]
    selected.loc[:, selected_columns].sort_values("sample_rank").to_csv(
        DERIVED_DIR / "interactive_sample_nodes.csv",
        index=False,
    )
    sample_edges.loc[:, edge_columns].to_csv(DERIVED_DIR / "interactive_sample_edges.csv", index=False)
    summary.to_csv(DERIVED_DIR / "interactive_sample_summary.csv", index=False)
    layouts.to_csv(DERIVED_DIR / "interactive_sample_layouts.csv", index=False)
    quality.to_csv(DERIVED_DIR / "interactive_sample_quality.csv", index=False)

    payload = _build_payload(selected, sample_edges, layouts, quality, edge_selections)
    html_path = REPORTS_DIR / "exposure_network_interactive.html"
    html_path.write_text(_render_html(payload), encoding="utf-8")
    manifest = {
        "html": artifact_path(html_path),
        "nodes": artifact_path(DERIVED_DIR / "interactive_sample_nodes.csv"),
        "edges": artifact_path(DERIVED_DIR / "interactive_sample_edges.csv"),
        "summary": artifact_path(DERIVED_DIR / "interactive_sample_summary.csv"),
        "layouts": artifact_path(DERIVED_DIR / "interactive_sample_layouts.csv"),
        "quality": artifact_path(DERIVED_DIR / "interactive_sample_quality.csv"),
        "sample": {
            "min_n": MIN_SAMPLE_N,
            "default_n": DEFAULT_SAMPLE_N,
            "max_n": MAX_SAMPLE_N,
            "step": SAMPLE_STEP,
            "role_target_shares": ROLE_TARGET_SHARES,
        },
    }
    write_json(REPORTS_DIR / "interactive_map_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_interactive_exposure_map(), indent=2))
