from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

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


LEVELS_BASE = [500, 1000, 2000, 4000, 6000]
NODE_LEVEL_MAX = 2000
NODE_LEVELS = [500, 1000, 2000]
MACRO_TOP_COMMUNITIES = 10
MACRO_MIN_COMMUNITY_SIZE = 10
COMMUNITY_LENS_MAX_NODES = 180
EGO_TOP_K = 8
ROLE_TARGET_SHARES = {
    "high_visibility_sender": 0.22,
    "high_exposure_receiver": 0.22,
    "bridge": 0.20,
    "peripheral": 0.16,
    "context_position": 0.20,
}
ROLE_ORDER = list(ROLE_TARGET_SHARES.keys())


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return pd.Series(0.0, index=values.index)
    return (values - low) / (high - low)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _normalize_positions(
    positions: dict[int | str, tuple[float, float]],
    keys: list[int | str],
    *,
    margin: float = 0.055,
) -> dict[int | str, tuple[float, float]]:
    xs = [float(positions[key][0]) for key in keys]
    ys = [float(positions[key][1]) for key in keys]
    x_low = _quantile(xs, 0.025)
    x_high = _quantile(xs, 0.975)
    y_low = _quantile(ys, 0.025)
    y_high = _quantile(ys, 0.975)
    x_center = (x_low + x_high) / 2.0
    y_center = (y_low + y_high) / 2.0
    span = max(x_high - x_low, y_high - y_low, 1e-9)
    scale = 1.0 - 2.0 * margin
    normalized = {}
    for key in keys:
        x, y = positions[key]
        nx_ = margin + scale * ((float(x) - x_center) / span + 0.5)
        ny_ = margin + scale * ((float(y) - y_center) / span + 0.5)
        normalized[key] = (min(0.97, max(0.03, nx_)), min(0.97, max(0.03, ny_)))
    return normalized


def _load_nodes() -> pd.DataFrame:
    node_metrics = pd.read_csv(input_path("node_metrics.csv"))
    neighborhoods = pd.read_csv(input_path("neighborhood_metrics.csv"))
    propagation = pd.read_csv(input_path("propagation_metrics.csv"))
    nodes = node_metrics.merge(
        neighborhoods.drop(
            columns=[
                "community_id",
                "weighted_in_degree",
                "weighted_out_degree",
                "eigenvector_centrality",
                "approx_betweenness",
                "local_clustering",
                "bridge_score",
                "dominant_structural_role",
            ],
            errors="ignore",
        ),
        on="position_id",
        how="left",
    ).merge(
        propagation.drop(
            columns=[
                "community_id",
                "weighted_in_degree",
                "weighted_out_degree",
                "bridge_score",
                "dominant_structural_role",
            ],
            errors="ignore",
        ),
        on="position_id",
        how="left",
    )
    nodes["in_largest_component"] = _as_bool(nodes["in_largest_component"])
    nodes["has_prompt_peer_capacity"] = _as_bool(nodes["has_prompt_peer_capacity"])
    for column in nodes.columns:
        if column not in {"position_id", "in_largest_component", "has_prompt_peer_capacity"}:
            nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0.0)
    return nodes


def _score_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    scored = nodes.copy()
    for optional in ("incoming_peer_community_count", "cascade_reach_potential"):
        if optional not in scored.columns:
            scored[optional] = 0.0
    scored["sender_score"] = _minmax(scored["weighted_out_degree"])
    scored["receiver_score"] = _minmax(scored["weighted_in_degree"])
    scored["bridge_score_norm"] = _minmax(scored["bridge_score"])
    scored["betweenness_score"] = _minmax(scored["approx_betweenness"])
    scored["eigenvector_score"] = _minmax(scored["eigenvector_centrality"])
    scored["community_diversity_score"] = _minmax(scored["incoming_peer_community_count"])
    scored["cascade_score"] = _minmax(scored["cascade_reach_potential"])
    scored["role_score_high_visibility_sender"] = scored["sender_score"]
    scored["role_score_high_exposure_receiver"] = scored["receiver_score"]
    scored["role_score_bridge"] = 0.62 * scored["bridge_score_norm"] + 0.38 * scored["betweenness_score"]
    scored["role_score_peripheral"] = 1.0 - 0.5 * scored["sender_score"] - 0.5 * scored["receiver_score"]
    scored["role_score_context_position"] = (
        0.34 * scored["community_diversity_score"]
        + 0.26 * scored["cascade_score"]
        + 0.22 * scored["eigenvector_score"]
        + 0.18 * scored["receiver_score"]
    )
    return scored


def _default_role(row: pd.Series) -> str:
    sender = float(row.get("weighted_out_degree_pct", row.get("sender_score", 0.0)))
    receiver = float(row.get("weighted_in_degree_pct", row.get("receiver_score", 0.0)))
    bridge = float(row.get("bridge_score_pct", row.get("bridge_score_norm", 0.0)))
    betweenness = float(row.get("approx_betweenness_pct", row.get("betweenness_score", 0.0)))
    if sender >= 0.84 and sender >= receiver:
        return "high_visibility_sender"
    if receiver >= 0.84 and receiver > sender:
        return "high_exposure_receiver"
    if bridge >= 0.78 or betweenness >= 0.82:
        return "bridge"
    if sender <= 0.28 and receiver <= 0.28:
        return "peripheral"
    return "context_position"


def _rank_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    scored = nodes.copy()
    scored["default_role"] = scored.apply(_default_role, axis=1)
    seed_path = DERIVED_DIR / "interactive_sample_nodes.csv"
    seed_ids: list[int] = []
    seed_roles: dict[int, str] = {}
    if seed_path.exists():
        seed = pd.read_csv(seed_path).sort_values("sample_rank").drop_duplicates("position_id").head(500)
        seed_ids = seed["position_id"].astype(int).tolist()
        seed_roles = dict(zip(seed["position_id"].astype(int), seed["display_role"].astype(str), strict=False))

    scored["extension_score"] = (
        0.30 * scored["sender_score"]
        + 0.28 * scored["receiver_score"]
        + 0.20 * scored["bridge_score_norm"]
        + 0.12 * scored["eigenvector_score"]
        + 0.10 * scored["role_score_context_position"]
    )
    indexed = scored.set_index("position_id", drop=False)
    used: set[int] = set()
    rows = []
    role_counts: Counter[str] = Counter()

    def append_node(position_id: int, role: str | None = None) -> None:
        if position_id in used or position_id not in indexed.index:
            return
        row = indexed.loc[position_id].copy()
        row["sample_rank"] = len(rows) + 1
        row["display_role"] = role if role in ROLE_ORDER else str(row["default_role"])
        rows.append(row.to_dict())
        used.add(position_id)
        role_counts[str(row["display_role"])] += 1

    for position_id in seed_ids:
        append_node(int(position_id), seed_roles.get(int(position_id)))

    eligible = scored[scored["in_largest_component"] & scored["has_prompt_peer_capacity"]].copy()
    role_lists = {
        role: eligible.sort_values(
            [f"role_score_{role}", "extension_score", "position_id"],
            ascending=[False, False, True],
        )["position_id"].astype(int).tolist()
        for role in ROLE_ORDER
    }
    pointers = {role: 0 for role in ROLE_ORDER}
    eligible_ids = set(eligible["position_id"].astype(int))
    while len(used & eligible_ids) < len(eligible_ids):
        rank = len(rows) + 1
        role = sorted(
            ROLE_ORDER,
            key=lambda item: (ROLE_TARGET_SHARES[item] * rank - role_counts[item], ROLE_TARGET_SHARES[item]),
            reverse=True,
        )[0]
        selected_id: int | None = None
        while pointers[role] < len(role_lists[role]):
            candidate = role_lists[role][pointers[role]]
            pointers[role] += 1
            if candidate not in used:
                selected_id = candidate
                break
        if selected_id is None:
            remainder = eligible[~eligible["position_id"].astype(int).isin(used)].sort_values(
                ["extension_score", "position_id"], ascending=[False, True]
            )
            if remainder.empty:
                break
            selected_id = int(remainder.iloc[0]["position_id"])
            role = str(remainder.iloc[0]["default_role"])
        append_node(selected_id, role)

    non_prompt = scored[~scored["position_id"].astype(int).isin(used)].sort_values(
        ["in_largest_component", "extension_score", "position_id"],
        ascending=[False, False, True],
    )
    for position_id in non_prompt["position_id"].astype(int):
        append_node(int(position_id), None)

    ranked = pd.DataFrame(rows).sort_values("sample_rank")
    ranked["display_role_label"] = ranked["display_role"].map(ROLE_LABELS)
    ranked["prompt_ready"] = ranked["has_prompt_peer_capacity"].astype(bool)
    ranked["observed_status"] = ranked["prompt_ready"].map(lambda value: "prompt_ready" if value else "observed_not_prompt_ready")
    if ranked["position_id"].nunique() != len(scored):
        raise ValueError("The multiresolution rank does not cover all observed nodes")
    return ranked


def _add_macro_communities(ranked: pd.DataFrame) -> pd.DataFrame:
    out = ranked.copy()
    counts = out["community_id"].astype(int).value_counts()
    major = [
        int(community_id)
        for community_id, count in counts.head(MACRO_TOP_COMMUNITIES).items()
        if int(count) >= MACRO_MIN_COMMUNITY_SIZE
    ]
    major_set = set(major)
    out["macro_community"] = out["community_id"].astype(int).map(lambda value: f"c{value}" if value in major_set else "long_tail")
    out["macro_label"] = out["macro_community"].map(lambda value: "long-tail communities" if value == "long_tail" else value)
    return out


def _induced_edges(edges: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    return edges[
        edges["source_position_id"].astype(int).isin(ids)
        & edges["target_position_id"].astype(int).isin(ids)
    ].copy()


def _level_layouts(ranked: pd.DataFrame, top_edges: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    previous_positions: dict[int, tuple[float, float]] = {}
    rows = []
    for level in [level for level in levels if level <= NODE_LEVEL_MAX]:
        nodes = ranked[ranked["sample_rank"] <= level]
        node_ids = nodes["position_id"].astype(int).tolist()
        ids = set(node_ids)
        edges = _induced_edges(top_edges, ids)
        if len(edges) > level * 10:
            edges = edges.sort_values(["exposure_weight", "total_events"], ascending=[False, False]).head(level * 10)
        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        for row in edges.itertuples(index=False):
            graph.add_edge(int(row.source_position_id), int(row.target_position_id), weight=float(row.exposure_weight))
        pos = {
            node_id: previous_positions.get(
                node_id,
                (
                    0.35 * math.cos((node_id * 0.61803398875) % 1.0 * math.tau),
                    0.35 * math.sin((node_id * 0.61803398875) % 1.0 * math.tau),
                ),
            )
            for node_id in node_ids
        }
        raw = nx.spring_layout(
            graph,
            pos=pos,
            seed=RANDOM_SEED,
            weight="weight",
            k=max(0.075, 0.74 / math.sqrt(level / 500)),
            iterations=130 if level <= 1000 else 95,
        )
        previous_positions = {int(node_id): (float(x), float(y)) for node_id, (x, y) in raw.items()}
        normalized = _normalize_positions(previous_positions, node_ids, margin=0.045)
        for node_id in node_ids:
            rows.append({"resolution_level": level, "position_id": node_id, "layout_x": normalized[node_id][0], "layout_y": normalized[node_id][1]})
    return pd.DataFrame(rows)


def _maximum_spanning_forest(active: pd.DataFrame) -> set[int]:
    graph = nx.Graph()
    for row in active.itertuples(index=False):
        source = int(row.source_position_id)
        target = int(row.target_position_id)
        weight = float(row.exposure_weight)
        edge_index = int(row.edge_index)
        if graph.has_edge(source, target):
            if weight > graph[source][target]["weight"]:
                graph[source][target]["weight"] = weight
                graph[source][target]["edge_index"] = edge_index
        else:
            graph.add_edge(source, target, weight=weight, edge_index=edge_index)
    if graph.number_of_edges() == 0:
        return set()
    tree = nx.maximum_spanning_tree(graph, weight="weight")
    return {int(data["edge_index"]) for _, _, data in tree.edges(data=True)}


def _node_level_edges(ranked: pd.DataFrame, top_edges: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    ranks = ranked.set_index("position_id")["sample_rank"].astype(int).to_dict()
    scoped = top_edges.copy()
    scoped["source_rank"] = scoped["source_position_id"].map(ranks)
    scoped["target_rank"] = scoped["target_position_id"].map(ranks)
    scoped = scoped.dropna(subset=["source_rank", "target_rank"]).copy()
    scoped["edge_min_sample_n"] = scoped[["source_rank", "target_rank"]].max(axis=1).astype(int)
    scoped = scoped.sort_values(["exposure_weight", "total_events"], ascending=[False, False]).reset_index(drop=True)
    scoped["edge_index"] = scoped.index.astype(int)
    rows = []
    for level in [level for level in levels if level <= NODE_LEVEL_MAX]:
        active = scoped[scoped["edge_min_sample_n"] <= level].copy()
        active = active.sort_values(["exposure_weight", "total_events"], ascending=[False, False])
        skeleton = _maximum_spanning_forest(active)
        cap = min(len(active), max(level - 1, math.ceil(level * (1.8 if level <= 1000 else 2.0))))
        chosen = list(skeleton)
        seen = set(chosen)
        for edge_index in active["edge_index"].astype(int):
            if len(chosen) >= cap:
                break
            if edge_index not in seen:
                chosen.append(edge_index)
                seen.add(edge_index)
        selected = scoped[scoped["edge_index"].isin(chosen)].copy()
        selected["resolution_level"] = level
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _macro_flow_rows(ranked: pd.DataFrame, full_edges: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    ranks = ranked.set_index("position_id")["sample_rank"].astype(int).to_dict()
    macros = ranked.set_index("position_id")["macro_community"].to_dict()
    edges = full_edges.copy()
    edges["source_rank"] = edges["source_position_id"].map(ranks)
    edges["target_rank"] = edges["target_position_id"].map(ranks)
    edges["source_macro"] = edges["source_position_id"].map(macros)
    edges["target_macro"] = edges["target_position_id"].map(macros)
    edges = edges.dropna(subset=["source_rank", "target_rank", "source_macro", "target_macro"]).copy()
    edges["edge_min_sample_n"] = edges[["source_rank", "target_rank"]].max(axis=1).astype(int)
    rows = []
    for level in levels:
        active = edges[edges["edge_min_sample_n"] <= level]
        grouped = (
            active.groupby(["source_macro", "target_macro"], as_index=False)
            .agg(flow_weight=("exposure_weight", "sum"), edge_count=("exposure_weight", "size"))
            .sort_values("flow_weight", ascending=False)
        )
        grouped["resolution_level"] = level
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _macro_positions(macros: list[str]) -> dict[str, tuple[float, float]]:
    sorted_macros = sorted(macros, key=lambda value: (value == "long_tail", value))
    count = len(sorted_macros)
    if count == 1:
        return {sorted_macros[0]: (0.5, 0.5)}
    out = {}
    for index, macro in enumerate(sorted_macros):
        angle = -math.pi / 2 + index * math.tau / count
        radius = 0.34 if macro != "long_tail" else 0.43
        out[macro] = (0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle))
    return out


def _macro_communities(ranked: pd.DataFrame, macro_flows: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    rows = []
    for level in levels:
        active = ranked[ranked["sample_rank"] <= level]
        macros = active["macro_community"].drop_duplicates().astype(str).tolist()
        positions = _macro_positions(macros)
        internal = macro_flows[(macro_flows["resolution_level"] == level) & (macro_flows["source_macro"] == macro_flows["target_macro"])]
        internal_lookup = dict(zip(internal["source_macro"], internal["flow_weight"], strict=False))
        grouped = active.groupby(["macro_community", "macro_label"], as_index=False).agg(
            node_count=("position_id", "size"),
            original_community_count=("community_id", "nunique"),
            prompt_ready_share=("prompt_ready", "mean"),
            sender_reach=("weighted_out_degree", "sum"),
            receiver_exposure=("weighted_in_degree", "sum"),
            bridge_mean=("bridge_score", "mean"),
        )
        for row in grouped.itertuples(index=False):
            macro = str(row.macro_community)
            x, y = positions[macro]
            rows.append(
                {
                    "resolution_level": level,
                    "macro_community": macro,
                    "macro_label": str(row.macro_label),
                    "node_count": int(row.node_count),
                    "original_community_count": int(row.original_community_count),
                    "prompt_ready_share": float(row.prompt_ready_share),
                    "sender_reach": float(row.sender_reach),
                    "receiver_exposure": float(row.receiver_exposure),
                    "bridge_mean": float(row.bridge_mean),
                    "internal_flow_weight": float(internal_lookup.get(macro, 0.0)),
                    "layout_x": x,
                    "layout_y": y,
                }
            )
    return pd.DataFrame(rows)


def _representative_nodes(group: pd.DataFrame, limit: int = COMMUNITY_LENS_MAX_NODES) -> pd.DataFrame:
    if len(group) <= limit:
        return group.copy()
    selected_ids: set[int] = set()
    rows = []
    per_role = max(12, limit // max(1, len(ROLE_ORDER)))
    for role in ROLE_ORDER:
        subset = group[group["display_role"] == role].sort_values(
            [f"role_score_{role}", "extension_score", "sample_rank"], ascending=[False, False, True]
        )
        for row in subset.head(per_role).itertuples(index=False):
            if int(row.position_id) not in selected_ids:
                rows.append(row._asdict())
                selected_ids.add(int(row.position_id))
    if len(rows) < limit:
        filler = group[~group["position_id"].astype(int).isin(selected_ids)].sort_values(
            ["extension_score", "sample_rank"], ascending=[False, True]
        )
        for row in filler.head(limit - len(rows)).itertuples(index=False):
            rows.append(row._asdict())
    return pd.DataFrame(rows).head(limit)


def _layout_subset(nodes: pd.DataFrame, edges: pd.DataFrame, x_name: str, y_name: str) -> pd.DataFrame:
    node_ids = nodes["position_id"].astype(int).tolist()
    ids = set(node_ids)
    induced = _induced_edges(edges, ids)
    if len(induced) > max(1, len(nodes) * 4):
        induced = induced.sort_values(["exposure_weight", "total_events"], ascending=[False, False]).head(len(nodes) * 4)
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for row in induced.itertuples(index=False):
        graph.add_edge(int(row.source_position_id), int(row.target_position_id), weight=float(row.exposure_weight))
    if graph.number_of_edges():
        raw = nx.spring_layout(
            graph,
            seed=RANDOM_SEED,
            weight="weight",
            k=max(0.10, 0.82 / math.sqrt(max(1, len(nodes)) / 60)),
            iterations=120,
        )
        raw = {int(key): (float(value[0]), float(value[1])) for key, value in raw.items()}
    else:
        raw = {}
        for index, node_id in enumerate(node_ids):
            angle = index * math.tau / max(1, len(node_ids))
            raw[node_id] = (math.cos(angle), math.sin(angle))
    normalized = _normalize_positions(raw, node_ids, margin=0.06)
    out = nodes.copy()
    out[x_name] = out["position_id"].astype(int).map(lambda node_id: normalized[int(node_id)][0])
    out[y_name] = out["position_id"].astype(int).map(lambda node_id: normalized[int(node_id)][1])
    return out


def _community_lenses(ranked: pd.DataFrame, top_edges: pd.DataFrame, levels: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lens_rows = []
    edge_rows = []
    for level in [level for level in levels if level > NODE_LEVEL_MAX]:
        active = ranked[ranked["sample_rank"] <= level]
        for macro, group in active.groupby("macro_community", sort=False):
            reps = _representative_nodes(group)
            reps = _layout_subset(reps, top_edges, "lens_x", "lens_y")
            reps["resolution_level"] = level
            reps["macro_community"] = str(macro)
            lens_rows.append(reps)
            ids = set(reps["position_id"].astype(int))
            induced = _induced_edges(top_edges, ids).sort_values(["exposure_weight", "total_events"], ascending=[False, False])
            cap = min(len(induced), max(len(reps) - 1, math.ceil(len(reps) * 2.2)))
            induced = induced.head(cap).copy()
            induced["resolution_level"] = level
            induced["macro_community"] = str(macro)
            edge_rows.append(induced)
    lens = pd.concat(lens_rows, ignore_index=True) if lens_rows else pd.DataFrame()
    lens_edges = pd.concat(edge_rows, ignore_index=True) if edge_rows else pd.DataFrame()
    return lens, lens_edges


def _ego_edges(top_edges: pd.DataFrame) -> pd.DataFrame:
    incoming = top_edges.sort_values(["target_position_id", "exposure_weight", "total_events"], ascending=[True, False, False])
    incoming = incoming.groupby("target_position_id", as_index=False).head(EGO_TOP_K).copy()
    incoming["ego_position_id"] = incoming["target_position_id"].astype(int)
    incoming["ego_edge_type"] = "incoming_exposure"
    outgoing = top_edges.sort_values(["source_position_id", "exposure_weight", "total_events"], ascending=[True, False, False])
    outgoing = outgoing.groupby("source_position_id", as_index=False).head(EGO_TOP_K).copy()
    outgoing["ego_position_id"] = outgoing["source_position_id"].astype(int)
    outgoing["ego_edge_type"] = "outgoing_visibility"
    return pd.concat([incoming, outgoing], ignore_index=True)


def _quality(ranked: pd.DataFrame, node_edges: pd.DataFrame, macro_nodes: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    rows = []
    for level in levels:
        active = ranked[ranked["sample_rank"] <= level]
        layout = "node" if level <= NODE_LEVEL_MAX else "macro"
        edge_count = int(len(node_edges[node_edges["resolution_level"] == level])) if level <= NODE_LEVEL_MAX else 0
        macro_count = int(len(macro_nodes[macro_nodes["resolution_level"] == level]))
        rows.append(
            {
                "resolution_level": level,
                "selected_nodes": int(len(active)),
                "observed_all_share": float(len(active) / len(ranked)),
                "prompt_ready_share": float(active["prompt_ready"].mean()),
                "original_communities": int(active["community_id"].nunique()),
                "macro_communities": macro_count,
                "rendered_node_edges": edge_count,
                "view_mode": layout,
            }
        )
    return pd.DataFrame(rows)


def _payload(
    ranked: pd.DataFrame,
    levels: list[int],
    layouts: pd.DataFrame,
    node_edges: pd.DataFrame,
    macro_nodes: pd.DataFrame,
    macro_flows: pd.DataFrame,
    lens_nodes: pd.DataFrame,
    lens_edges: pd.DataFrame,
    ego_edges: pd.DataFrame,
    quality: pd.DataFrame,
) -> dict[str, object]:
    max_out = float(ranked["weighted_out_degree"].max()) or 1.0
    max_in = float(ranked["weighted_in_degree"].max()) or 1.0
    node_rows = []
    for row in ranked.itertuples(index=False):
        node_rows.append(
            {
                "id": int(row.position_id),
                "rank": int(row.sample_rank),
                "role": str(row.display_role),
                "roleLabel": str(row.display_role_label),
                "community": int(row.community_id),
                "macro": str(row.macro_community),
                "macroLabel": str(row.macro_label),
                "promptReady": bool(row.prompt_ready),
                "weightedOut": round(float(row.weighted_out_degree), 4),
                "weightedIn": round(float(row.weighted_in_degree), 4),
                "outNorm": round(float(row.weighted_out_degree) / max_out, 6),
                "inNorm": round(float(row.weighted_in_degree) / max_in, 6),
                "bridge": round(float(row.bridge_score), 6),
                "incomingPeers": int(row.incoming_peer_count) if "incoming_peer_count" in ranked.columns else 0,
                "outgoingReceivers": int(row.out_degree),
            }
        )
    layout_payload = {
        str(int(level)): {
            str(int(row.position_id)): [round(float(row.layout_x), 6), round(float(row.layout_y), 6)]
            for row in group.itertuples(index=False)
        }
        for level, group in layouts.groupby("resolution_level", sort=True)
    }
    return {
        "defaults": {"levels": levels, "defaultLevel": 1000, "nodeLevelMax": NODE_LEVEL_MAX, "finalLevel": levels[-1]},
        "direction": "TargetUserId -> SourceUserId",
        "interactionWeights": INTERACTION_WEIGHTS,
        "roles": {
            role: {"label": ROLE_LABELS[role], "color": ROLE_PALETTE[role], "definition": ROLE_DEFINITIONS[role]}
            for role in ROLE_ORDER
        },
        "palette": {"ink": TOKENS["ink"], "muted": TOKENS["muted"], "grid": TOKENS["grid"]},
        "nodes": node_rows,
        "layouts": layout_payload,
        "nodeEdges": [
            {
                "level": int(row.resolution_level),
                "source": int(row.source_position_id),
                "target": int(row.target_position_id),
                "weight": round(float(row.exposure_weight), 5),
                "types": str(row.interaction_types),
            }
            for row in node_edges.itertuples(index=False)
        ],
        "macroNodes": [
            {
                "level": int(row.resolution_level),
                "id": str(row.macro_community),
                "label": str(row.macro_label),
                "count": int(row.node_count),
                "originalCommunities": int(row.original_community_count),
                "promptReadyShare": round(float(row.prompt_ready_share), 5),
                "senderReach": round(float(row.sender_reach), 4),
                "receiverExposure": round(float(row.receiver_exposure), 4),
                "internalFlow": round(float(row.internal_flow_weight), 4),
                "x": round(float(row.layout_x), 6),
                "y": round(float(row.layout_y), 6),
            }
            for row in macro_nodes.itertuples(index=False)
        ],
        "macroFlows": [
            {
                "level": int(row.resolution_level),
                "source": str(row.source_macro),
                "target": str(row.target_macro),
                "weight": round(float(row.flow_weight), 4),
                "edges": int(row.edge_count),
            }
            for row in macro_flows.itertuples(index=False)
        ],
        "lensNodes": [
            {
                "level": int(row.resolution_level),
                "macro": str(row.macro_community),
                "id": int(row.position_id),
                "x": round(float(row.lens_x), 6),
                "y": round(float(row.lens_y), 6),
            }
            for row in lens_nodes.itertuples(index=False)
        ],
        "lensEdges": [
            {
                "level": int(row.resolution_level),
                "macro": str(row.macro_community),
                "source": int(row.source_position_id),
                "target": int(row.target_position_id),
                "weight": round(float(row.exposure_weight), 5),
                "types": str(row.interaction_types),
            }
            for row in lens_edges.itertuples(index=False)
        ],
        "egoEdges": [
            {
                "ego": int(row.ego_position_id),
                "type": str(row.ego_edge_type),
                "source": int(row.source_position_id),
                "target": int(row.target_position_id),
                "weight": round(float(row.exposure_weight), 5),
                "types": str(row.interaction_types),
            }
            for row in ego_edges.itertuples(index=False)
        ],
        "quality": quality.to_dict(orient="records"),
    }


def _json(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def _render_html(payload: dict[str, object]) -> str:
    data = _json(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hierarchical Directed Exposure Network</title>
  <style>
    :root {{
      --surface: #fcfcfd; --panel: #ffffff; --ink: #1f2430; --muted: #6f768a;
      --grid: #e6e8f0; --axis: #d7dbe7; --accent: #f0986e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif; line-height: 1.45; }}
    main {{ width: min(1500px, calc(100vw - 34px)); margin: 0 auto; padding: 22px 0 42px; }}
    h1 {{ margin: 0 0 6px; font-size: 32px; line-height: 1.08; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .summary {{ max-width: 1120px; margin: 0 0 16px; color: #303746; }}
    .layout {{ display: grid; grid-template-columns: 292px minmax(760px, 1fr) 330px; gap: 14px; align-items: stretch; }}
    .panel {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 10px; padding: 16px; }}
    label {{ display: block; margin: 13px 0 6px; font-size: 11px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; color: var(--muted); }}
    select, button {{ width: 100%; height: 34px; border: 1px solid var(--axis); border-radius: 7px; background: white; color: var(--ink); font-weight: 750; padding: 0 8px; }}
    button {{ cursor: pointer; background: #fff7f0; border-color: #f3c1a6; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 14px; }}
    .metric {{ border: 1px solid var(--grid); border-radius: 8px; padding: 10px; background: #fbfbfc; }}
    .metric span {{ display: block; font-size: 10px; text-transform: uppercase; letter-spacing: .09em; color: var(--muted); font-weight: 800; }}
    .metric strong {{ display: block; margin-top: 3px; font-size: 18px; }}
    .canvas-wrap {{ position: relative; min-height: 780px; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 780px; background: #fff; border-radius: 8px; }}
    .tooltip {{ position: absolute; pointer-events: none; display: none; max-width: 300px; padding: 10px 12px; border: 1px solid var(--axis); border-radius: 8px; background: rgba(255,255,255,.96); box-shadow: 0 10px 30px rgba(31,36,48,.12); font-size: 12px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 10px; font-size: 12px; color: #303746; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .breadcrumb {{ margin: 10px 0 0; color: #303746; font-size: 13px; min-height: 34px; }}
    .row {{ display: flex; justify-content: space-between; gap: 14px; border-bottom: 1px solid var(--grid); padding: 7px 0; font-size: 13px; }}
    .row span:first-child {{ color: var(--muted); }}
    .row strong {{ text-align: right; }}
    .edge-list {{ margin: 14px 0 0; padding: 0; list-style: none; max-height: 205px; overflow: auto; border-top: 1px solid var(--grid); }}
    .edge-list li {{ padding: 7px 0; border-bottom: 1px solid var(--grid); font-size: 12px; color: #303746; }}
    .method {{ margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .method code {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; background: #f4f5f7; border: 1px solid var(--grid); border-radius: 5px; padding: 2px 4px; }}
    @media (max-width: 1180px) {{ .layout {{ grid-template-columns: 1fr; }} .canvas-wrap {{ min-height: 640px; }} canvas {{ height: 640px; }} .method {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Hierarchical Directed Exposure Network</h1>
  <p class="summary">This explorer keeps the network visually explorable from 500 positions to all observed positions. Low and mid resolutions use node-link maps; large resolutions use macro-community traffic with drill-down into community and ego exposure lenses. Direction is always <strong>visible target -> exposed source</strong>.</p>
  <section class="layout">
    <aside class="panel">
      <h2>Controls</h2>
      <label for="resolution">Resolution</label><select id="resolution"></select>
      <label for="viewMode">View</label>
      <select id="viewMode"><option value="global">Global</option><option value="community">Community lens</option><option value="ego">Ego lens</option></select>
      <label for="colorMode">Color mode</label>
      <select id="colorMode"><option value="role">structural role</option><option value="community">community</option><option value="prompt">prompt readiness</option></select>
      <div class="breadcrumb" id="breadcrumb">Global view</div>
      <button id="backButton" type="button">Back to global</button>
      <div class="metric-grid">
        <div class="metric"><span>Nodes</span><strong id="nodeCount">--</strong></div>
        <div class="metric"><span>Edges</span><strong id="edgeCount">--</strong></div>
        <div class="metric"><span>Macro groups</span><strong id="macroCount">--</strong></div>
        <div class="metric"><span>Prompt-ready</span><strong id="promptReady">--</strong></div>
        <div class="metric"><span>Original comm.</span><strong id="communityCount">--</strong></div>
        <div class="metric"><span>Layer</span><strong id="layerLabel">--</strong></div>
      </div>
      <p class="muted">Click a macro community to open its readable lens. Click a node to inspect its ego exposure neighborhood.</p>
    </aside>
    <section class="panel canvas-wrap">
      <canvas id="networkCanvas"></canvas>
      <div id="tooltip" class="tooltip"></div>
      <div id="legend" class="legend"></div>
    </section>
    <aside class="panel">
      <h2 id="inspectorTitle">Select a community or node</h2>
      <p id="inspectorSubtitle" class="muted">Hover or click to inspect exposure position.</p>
      <div id="inspectorRows"></div>
      <h2 style="margin-top:18px;">Top incoming exposures</h2>
      <ul class="edge-list" id="incomingList"></ul>
      <h2 style="margin-top:18px;">Top outgoing visibility</h2>
      <ul class="edge-list" id="outgoingList"></ul>
    </aside>
  </section>
  <section class="method">
    <div class="panel"><h2>Edge construction</h2><p><code>exposure_raw_weight = 0.35 * Like + 0.80 * Repost + 0.90 * Quote</code></p><p>Repeated interactions are summed, log-compressed, and normalized to <code>[0, 1]</code>.</p></div>
    <div class="panel"><h2>Hierarchy rule</h2><p>At 500-2,000 positions, the map uses a scale-specific node layout. At 4,000 and above, the global layer is macro-community traffic; individual nodes appear through community and ego lenses.</p></div>
  </section>
</main>
<script id="network-data" type="application/json">{data}</script>
<script>
const payload = JSON.parse(document.getElementById("network-data").textContent);
const nodes = payload.nodes, nodeById = new Map(nodes.map(n => [n.id, n]));
const layouts = payload.layouts;
const nodeEdges = payload.nodeEdges;
const macroNodes = payload.macroNodes;
const macroFlows = payload.macroFlows;
const lensNodes = payload.lensNodes;
const lensEdges = payload.lensEdges;
const egoEdges = payload.egoEdges;
const quality = new Map(payload.quality.map(row => [row.resolution_level, row]));
const canvas = document.getElementById("networkCanvas"), ctx = canvas.getContext("2d");
const resolution = document.getElementById("resolution"), viewMode = document.getElementById("viewMode"), colorMode = document.getElementById("colorMode");
const tooltip = document.getElementById("tooltip"), legend = document.getElementById("legend");
const inspectorTitle = document.getElementById("inspectorTitle"), inspectorSubtitle = document.getElementById("inspectorSubtitle"), inspectorRows = document.getElementById("inspectorRows");
const incomingList = document.getElementById("incomingList"), outgoingList = document.getElementById("outgoingList"), breadcrumb = document.getElementById("breadcrumb");
const communityPalette = ["#F0986E","#A3BEFA","#F390CA","#A3D576","#FFE15B","#B9B0F8","#8FD6D2","#D8C09D","#B8C2CC","#F6B1A3"];
let selectedMacro = null, selectedNode = null, drawnNodes = [], drawnMacros = [];
for (const level of payload.defaults.levels) {{
  const option = document.createElement("option");
  option.value = String(level);
  option.textContent = level === payload.defaults.finalLevel ? `all nodes (${{level.toLocaleString()}})` : `${{level.toLocaleString()}} nodes`;
  if (level === 1000) option.selected = true;
  resolution.appendChild(option);
}}
function fmt(value, digits=1) {{ if (value === undefined || value === null || Number.isNaN(Number(value))) return "--"; return Number(value).toLocaleString(undefined, {{maximumFractionDigits: digits}}); }}
function level() {{ return Number(resolution.value); }}
function resizeCanvas() {{ const r = canvas.getBoundingClientRect(), s = window.devicePixelRatio || 1; canvas.width = Math.floor(r.width*s); canvas.height = Math.floor(r.height*s); ctx.setTransform(s,0,0,s,0,0); }}
function screen(x,y,w,h) {{ const p=28; return {{x:p+x*(w-p*2), y:p+y*(h-p*2)}}; }}
function roleColor(node) {{ if (colorMode.value === "community") return communityPalette[Math.abs(node.community)%communityPalette.length]; if (colorMode.value === "prompt") return node.promptReady ? "#A3BEFA" : "#D7DBE7"; return payload.roles[node.role]?.color || "#C5CAD3"; }}
function radius(node,count) {{ const min=count<=1000?3.1:2.2, max=count<=1000?12:7.8; return min+(max-min)*Math.sqrt(Math.min(1,node.outNorm)); }}
function drawArrow(a,b,width,alpha,color) {{ const dx=b.x-a.x, dy=b.y-a.y, len=Math.hypot(dx,dy); if(len<2)return; const ux=dx/len, uy=dy/len, ex=b.x-ux*8, ey=b.y-uy*8; ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(ex,ey); ctx.stroke(); const ar=5+width; ctx.beginPath(); ctx.moveTo(ex,ey); ctx.lineTo(ex-ux*ar-uy*ar*.55,ey-uy*ar+ux*ar*.55); ctx.lineTo(ex-ux*ar+uy*ar*.55,ey-uy*ar-ux*ar*.55); ctx.closePath(); ctx.fill(); ctx.globalAlpha=1; }}
function background(w,h) {{ ctx.fillStyle="#fff"; ctx.fillRect(0,0,w,h); ctx.strokeStyle="rgba(230,232,240,.34)"; ctx.lineWidth=1; for(let x=0;x<w;x+=92){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}} for(let y=0;y<h;y+=92){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}} }}
function activeNodes() {{ const n=level(); return nodes.filter(node => node.rank <= n); }}
function activeMacroNodes() {{ const n=level(); return macroNodes.filter(m => m.level === n); }}
function activeMacroFlows() {{ const n=level(); return macroFlows.filter(f => f.level === n && f.source !== f.target); }}
function nodeLayer(w,h) {{
  const n=level(), current=activeNodes(), ids=new Set(current.map(d=>d.id)), layout=layouts[String(n)]||{{}};
  const pos=new Map(current.map(node => [node.id, screen(layout[String(node.id)]?.[0] ?? .5, layout[String(node.id)]?.[1] ?? .5, w, h)]));
  const edges=nodeEdges.filter(e => e.level===n && ids.has(e.source) && ids.has(e.target));
  for(const e of edges.slice().reverse()) drawArrow(pos.get(e.source), pos.get(e.target), .45+2.6*Math.sqrt(e.weight), .07+.30*Math.sqrt(e.weight), "#8E97A6");
  drawnNodes=[]; drawnMacros=[];
  for(const node of current) {{ const p=pos.get(node.id), r=radius(node,current.length); ctx.beginPath(); ctx.fillStyle=roleColor(node); ctx.strokeStyle=node.promptReady?"#fff":"#8E97A6"; ctx.lineWidth=node.promptReady?1.5:1; ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill(); ctx.stroke(); drawnNodes.push({{...node,x:p.x,y:p.y,r}}); }}
  return edges.length;
}}
function macroLayer(w,h) {{
  const macros=activeMacroNodes(), byId=new Map(macros.map(m=>[m.id,m])), flows=activeMacroFlows();
  const maxFlow=Math.max(1,...flows.map(f=>f.weight)), maxCount=Math.max(1,...macros.map(m=>m.count)), maxInternal=Math.max(1,...macros.map(m=>m.internalFlow));
  drawnNodes=[]; drawnMacros=[];
  for(const f of flows.slice().sort((a,b)=>a.weight-b.weight)) {{ const a=byId.get(f.source), b=byId.get(f.target); if(!a||!b)continue; drawArrow(screen(a.x,a.y,w,h), screen(b.x,b.y,w,h), .8+5.6*Math.sqrt(f.weight/maxFlow), .10+.36*Math.sqrt(f.weight/maxFlow), "#8E97A6"); }}
  for(const m of macros) {{
    const p=screen(m.x,m.y,w,h), r=9+42*Math.sqrt(m.count/maxCount), halo=r+18*Math.sqrt(m.internalFlow/maxInternal);
    ctx.beginPath(); ctx.fillStyle="rgba(240,152,110,.10)"; ctx.arc(p.x,p.y,halo,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.fillStyle=m.id==="long_tail"?"#D7DBE7":communityPalette[Math.abs(parseInt(m.id.replace("c",""))||0)%communityPalette.length]; ctx.strokeStyle="#fff"; ctx.lineWidth=2.5; ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle="#1f2430"; ctx.font="750 12px Inter, Arial"; ctx.fillText(m.label, p.x+r+5, p.y+4);
    drawnMacros.push({{...m,x:p.x,y:p.y,r}});
  }}
  return flows.length;
}}
function communityLens(w,h) {{
  if(!selectedMacro) return macroLayer(w,h);
  const n=level(), current=lensNodes.filter(d=>d.level===n && d.macro===selectedMacro), ids=new Set(current.map(d=>d.id)), pos=new Map(current.map(d=>[d.id,screen(d.x,d.y,w,h)]));
  const edges=lensEdges.filter(e=>e.level===n && e.macro===selectedMacro && ids.has(e.source) && ids.has(e.target));
  for(const e of edges.slice().reverse()) drawArrow(pos.get(e.source), pos.get(e.target), .45+2.8*Math.sqrt(e.weight), .08+.32*Math.sqrt(e.weight), "#8E97A6");
  drawnNodes=[]; drawnMacros=[];
  for(const item of current) {{ const node=nodeById.get(item.id), p=pos.get(item.id), r=radius(node,current.length); ctx.beginPath(); ctx.fillStyle=roleColor(node); ctx.strokeStyle="#fff"; ctx.lineWidth=1.5; ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill(); ctx.stroke(); drawnNodes.push({{...node,x:p.x,y:p.y,r}}); }}
  return edges.length;
}}
function egoLens(w,h) {{
  if(!selectedNode) return selectedMacro ? communityLens(w,h) : macroLayer(w,h);
  const center=nodeById.get(selectedNode), edges=egoEdges.filter(e=>e.ego===selectedNode), incoming=edges.filter(e=>e.type==="incoming_exposure"), outgoing=edges.filter(e=>e.type==="outgoing_visibility");
  const c={{x:w*.50,y:h*.50}}, positions=new Map([[center.id,c]]), allIds=new Set([center.id]);
  incoming.forEach((e,i)=>{{ const angle=-Math.PI*.72 + i*Math.PI*.44/Math.max(1,incoming.length-1); positions.set(e.source,{{x:w*.31+90*Math.cos(angle),y:h*.50+230*Math.sin(angle)}}); allIds.add(e.source); }});
  outgoing.forEach((e,i)=>{{ const angle=Math.PI*.28 + i*Math.PI*.44/Math.max(1,outgoing.length-1); positions.set(e.target,{{x:w*.69+90*Math.cos(angle),y:h*.50+230*Math.sin(angle)}}); allIds.add(e.target); }});
  for(const e of edges) {{ const a=positions.get(e.source), b=positions.get(e.target); if(a&&b) drawArrow(a,b,.8+3.4*Math.sqrt(e.weight),.14+.40*Math.sqrt(e.weight),"#8E97A6"); }}
  drawnNodes=[]; drawnMacros=[];
  for(const id of allIds) {{ const node=nodeById.get(id); if(!node)continue; const p=positions.get(id), r=id===center.id?17:radius(node,edges.length+1); ctx.beginPath(); ctx.fillStyle=roleColor(node); ctx.strokeStyle=id===center.id?"#F0986E":"#fff"; ctx.lineWidth=id===center.id?3:1.5; ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.fillStyle="#1f2430"; ctx.font="700 11px Inter, Arial"; ctx.fillText(String(id),p.x+r+4,p.y+4); drawnNodes.push({{...node,x:p.x,y:p.y,r}}); }}
  return edges.length;
}}
function metrics(edgeCount, layer) {{ const q=quality.get(level()); document.getElementById("nodeCount").textContent=fmt(q?.selected_nodes,0); document.getElementById("edgeCount").textContent=fmt(edgeCount,0); document.getElementById("macroCount").textContent=fmt(q?.macro_communities,0); document.getElementById("promptReady").textContent=q?`${{fmt(q.prompt_ready_share*100,1)}}%`:"--"; document.getElementById("communityCount").textContent=fmt(q?.original_communities,0); document.getElementById("layerLabel").textContent=layer; }}
function drawLegend() {{ if(colorMode.value==="prompt") {{ legend.innerHTML='<span class="legend-item"><span class="swatch" style="background:#A3BEFA"></span>prompt-ready</span><span class="legend-item"><span class="swatch" style="background:#D7DBE7"></span>observed only</span>'; return; }} legend.innerHTML=Object.values(payload.roles).map(spec=>`<span class="legend-item"><span class="swatch" style="background:${{spec.color}}"></span>${{spec.label}}</span>`).join(""); }}
function draw() {{
  resizeCanvas(); const r=canvas.getBoundingClientRect(), w=r.width, h=r.height; background(w,h);
  let edges=0, layer="global"; const n=level();
  if(viewMode.value==="ego" && selectedNode) {{ edges=egoLens(w,h); layer="ego"; }}
  else if(viewMode.value==="community" && selectedMacro) {{ edges=communityLens(w,h); layer="community"; }}
  else if(n<=payload.defaults.nodeLevelMax) {{ edges=nodeLayer(w,h); layer="node"; }}
  else {{ edges=macroLayer(w,h); layer="macro"; }}
  breadcrumb.textContent = selectedNode ? `Ego lens · position ${{selectedNode}}` : selectedMacro ? `Community lens · ${{selectedMacro}}` : "Global view";
  metrics(edges,layer); drawLegend();
}}
function row(label,value) {{ return `<div class="row"><span>${{label}}</span><strong>${{value}}</strong></div>`; }}
function inspectNode(node) {{ selectedNode=node.id; viewMode.value="ego"; inspectorTitle.textContent=`position ${{node.id}}`; inspectorSubtitle.textContent=node.roleLabel; inspectorRows.innerHTML=[row("community",node.community),row("macro",node.macroLabel),row("prompt-ready",node.promptReady?"yes":"no"),row("sender reach",fmt(node.weightedOut,2)),row("receiver exposure",fmt(node.weightedIn,2)),row("bridge",fmt(node.bridge,3))].join(""); const incoming=egoEdges.filter(e=>e.ego===node.id&&e.type==="incoming_exposure"), outgoing=egoEdges.filter(e=>e.ego===node.id&&e.type==="outgoing_visibility"); incomingList.innerHTML=incoming.map(e=>`<li>${{e.source}} -> ${{e.target}} · ${{fmt(e.weight,3)}} · ${{e.types}}</li>`).join("")||"<li>No incoming exposure edges retained.</li>"; outgoingList.innerHTML=outgoing.map(e=>`<li>${{e.source}} -> ${{e.target}} · ${{fmt(e.weight,3)}} · ${{e.types}}</li>`).join("")||"<li>No outgoing visibility edges retained.</li>"; draw(); }}
function inspectMacro(macro) {{ selectedMacro=macro.id; selectedNode=null; viewMode.value="community"; inspectorTitle.textContent=macro.label; inspectorSubtitle.textContent="macro-community exposure group"; inspectorRows.innerHTML=[row("positions",fmt(macro.count,0)),row("original communities",fmt(macro.originalCommunities,0)),row("prompt-ready",`${{fmt(macro.promptReadyShare*100,1)}}%`),row("sender reach",fmt(macro.senderReach,1)),row("receiver exposure",fmt(macro.receiverExposure,1)),row("internal flow",fmt(macro.internalFlow,1))].join(""); incomingList.innerHTML="<li>Community lens shows representative internal positions.</li>"; outgoingList.innerHTML="<li>Macro arrows show directed cross-community exposure.</li>"; draw(); }}
function pick(event) {{ const rect=canvas.getBoundingClientRect(), x=event.clientX-rect.left, y=event.clientY-rect.top; let best=null, dist=Infinity; for(const n of drawnNodes){{const d=Math.hypot(n.x-x,n.y-y); if(d<n.r+6&&d<dist){{best={{type:"node",item:n}};dist=d;}}}} for(const m of drawnMacros){{const d=Math.hypot(m.x-x,m.y-y); if(d<m.r+8&&d<dist){{best={{type:"macro",item:m}};dist=d;}}}} return best; }}
canvas.addEventListener("mousemove",e=>{{ const p=pick(e); if(!p){{tooltip.style.display="none";return;}} tooltip.style.display="block"; tooltip.style.left=`${{e.offsetX+14}}px`; tooltip.style.top=`${{e.offsetY+14}}px`; tooltip.innerHTML=p.type==="node"?`<strong>position ${{p.item.id}}</strong><br>${{p.item.roleLabel}}<br>${{p.item.macroLabel}}`:`<strong>${{p.item.label}}</strong><br>${{fmt(p.item.count,0)}} positions<br>${{fmt(p.item.promptReadyShare*100,1)}}% prompt-ready`; }});
canvas.addEventListener("mouseleave",()=>tooltip.style.display="none");
canvas.addEventListener("click",e=>{{ const p=pick(e); if(!p)return; if(p.type==="node") inspectNode(p.item); else inspectMacro(p.item); }});
document.getElementById("backButton").addEventListener("click",()=>{{selectedMacro=null;selectedNode=null;viewMode.value="global";draw();}});
resolution.addEventListener("change",()=>{{selectedMacro=null;selectedNode=null;viewMode.value="global";draw();}});
viewMode.addEventListener("change",draw); colorMode.addEventListener("change",draw); window.addEventListener("resize",draw);
draw();
</script>
</body>
</html>"""


def build_multiresolution_exposure_map() -> dict[str, object]:
    ensure_dirs()
    top_edges = pd.read_csv(input_path("edges_prompt_top30.csv"))
    full_edges = pd.read_csv(input_path("edges_full.csv"))
    ranked = _add_macro_communities(_rank_nodes(_score_nodes(_load_nodes())))
    levels = [level for level in LEVELS_BASE if level < len(ranked)] + [len(ranked)]

    layouts = _level_layouts(ranked, top_edges, levels)
    node_edges = _node_level_edges(ranked, top_edges, levels)
    macro_flows = _macro_flow_rows(ranked, full_edges, levels)
    macro_nodes = _macro_communities(ranked, macro_flows, levels)
    lens_nodes, lens_edges = _community_lenses(ranked, top_edges, levels)
    ego_edges = _ego_edges(top_edges)
    quality = _quality(ranked, node_edges, macro_nodes, levels)

    node_columns = [
        "position_id",
        "sample_rank",
        "display_role",
        "display_role_label",
        "community_id",
        "macro_community",
        "macro_label",
        "prompt_ready",
        "observed_status",
        "weighted_out_degree",
        "weighted_in_degree",
        "bridge_score",
    ]
    ranked[node_columns].to_csv(DERIVED_DIR / "multiresolution_nodes.csv", index=False)
    node_edges.to_csv(DERIVED_DIR / "multiresolution_edges_backbone.csv", index=False)
    macro_nodes.to_csv(DERIVED_DIR / "multiresolution_macro_communities.csv", index=False)
    macro_flows.to_csv(DERIVED_DIR / "multiresolution_macro_flows.csv", index=False)
    layouts.to_csv(DERIVED_DIR / "multiresolution_level_layouts.csv", index=False)
    lens_nodes[["resolution_level", "macro_community", "position_id", "lens_x", "lens_y", "display_role", "community_id"]].to_csv(
        DERIVED_DIR / "multiresolution_community_lenses.csv", index=False
    )
    lens_edges.to_csv(DERIVED_DIR / "multiresolution_community_lens_edges.csv", index=False)
    ego_edges.to_csv(DERIVED_DIR / "multiresolution_ego_edges.csv", index=False)
    quality.to_csv(DERIVED_DIR / "multiresolution_quality.csv", index=False)

    payload = _payload(ranked, levels, layouts, node_edges, macro_nodes, macro_flows, lens_nodes, lens_edges, ego_edges, quality)
    html_path = REPORTS_DIR / "exposure_network_multiresolution.html"
    html_path.write_text(_render_html(payload), encoding="utf-8")
    manifest = {
        "report": artifact_path(html_path),
        "levels": levels,
        "observed_nodes": int(len(ranked)),
        "prompt_ready_nodes": int(ranked["prompt_ready"].sum()),
        "node_edges": int(len(node_edges)),
        "macro_communities": int(len(macro_nodes)),
        "macro_flows": int(len(macro_flows)),
        "community_lens_nodes": int(len(lens_nodes)),
        "ego_edges": int(len(ego_edges)),
    }
    write_json(REPORTS_DIR / "multiresolution_map_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_multiresolution_exposure_map(), indent=2))
