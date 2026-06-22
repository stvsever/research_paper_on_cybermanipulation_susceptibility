from __future__ import annotations

"""Network-specific table builders and figures for the branch exposure substrate.

This module owns the empirical exposure edge loading, directed backbone selection, and full 35-condition network overlay.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .formatting import _save_fig, _sender_alignment_label
from .paths import PROJECT_ROOT, GRAPH_ID, ensure_dir, graph_root


def _graph_root() -> Path:
    return PROJECT_ROOT / "data" / "exposure_networks" / GRAPH_ID


def _load_edges(positions: set[str]) -> pd.DataFrame:
    path = _graph_root() / "edges_prompt_top30.csv"
    edges = pd.read_csv(path, dtype={"source_position_id": str, "target_position_id": str})
    edges = edges[
        edges["source_position_id"].isin(positions)
        & edges["target_position_id"].isin(positions)
        & (edges["source_position_id"] != edges["target_position_id"])
    ].copy()
    edges["exposure_weight"] = pd.to_numeric(edges["exposure_weight"], errors="coerce").fillna(0.0)
    return edges[edges["exposure_weight"] > 0].reset_index(drop=True)


def _unit_interval(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if values.empty:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def _branch_overlay_backbone_edges(edges: pd.DataFrame, *, top_incoming: int = 1, global_top_n: int = 50) -> pd.DataFrame:
    if edges.empty:
        return edges.assign(backbone_reason=pd.Series(dtype=str))
    by_receiver = (
        edges.sort_values(["target_position_id", "exposure_weight"], ascending=[True, False])
        .groupby("target_position_id", as_index=False)
        .head(top_incoming)
    )
    global_strong = edges.nlargest(min(global_top_n, len(edges)), "exposure_weight")
    reason_by_edge: dict[tuple[str, str], set[str]] = {}
    for row in by_receiver.itertuples(index=False):
        reason_by_edge.setdefault((str(row.source_position_id), str(row.target_position_id)), set()).add(
            f"top_{top_incoming}_incoming_per_receiver"
        )
    for row in global_strong.itertuples(index=False):
        reason_by_edge.setdefault((str(row.source_position_id), str(row.target_position_id)), set()).add(
            f"global_top_{global_top_n}_exposure_weight"
        )
    backbone = pd.concat([by_receiver, global_strong], ignore_index=True).drop_duplicates(
        ["source_position_id", "target_position_id"]
    )
    backbone = backbone.copy()
    backbone["backbone_reason"] = [
        "+".join(sorted(reason_by_edge[(str(row.source_position_id), str(row.target_position_id))]))
        for row in backbone.itertuples(index=False)
    ]
    return backbone.reset_index(drop=True)


def _network_layout(positions: set[str], edges: pd.DataFrame) -> dict[str, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(positions)
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.source_position_id), str(row.target_position_id), weight=float(row.exposure_weight))
    return nx.spring_layout(graph, seed=42, weight="weight", iterations=350, k=0.55)


def _edge_segments(edges: pd.DataFrame, layout: dict[str, np.ndarray]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in edges.itertuples(index=False):
        source = str(row.source_position_id)
        target = str(row.target_position_id)
        if source in layout and target in layout:
            segments.append(
                (
                    (float(layout[source][0]), float(layout[source][1])),
                    (float(layout[target][0]), float(layout[target][1])),
                )
            )
    return segments


def _branch_overlay_nodes(sem: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    required = [
        "alignment_condition_id",
        "profile_id",
        "ae_private",
        "exposure_position_id",
        "exposure_outgoing_visibility_weight",
    ]
    missing = [col for col in required if col not in sem.columns]
    if missing:
        raise RuntimeError("Branch SEM data is missing network overlay columns: " + ", ".join(missing))
    condition_cols = [
        "alignment_condition_id",
        "opinion_label",
        "attack_label",
        "target_alignment_z",
        "achieved_alignment_z",
        "condition_index",
        "opinion_index",
        "attack_index",
    ]
    plot_df = sem.merge(
        condition[[col for col in condition_cols if col in condition.columns]],
        on="alignment_condition_id",
        how="left",
        suffixes=("", "_condition"),
    ).copy()
    plot_df["exposure_position_id"] = plot_df["exposure_position_id"].astype(str)
    plot_df["ae_private"] = pd.to_numeric(plot_df["ae_private"], errors="coerce")
    plot_df["exposure_outgoing_visibility_weight"] = pd.to_numeric(
        plot_df["exposure_outgoing_visibility_weight"], errors="coerce"
    ).fillna(0.0)
    plot_df["ae_private_percentile_within_condition"] = plot_df.groupby("alignment_condition_id")[
        "ae_private"
    ].rank(pct=True)
    plot_df["sender_reach_percentile_within_condition"] = plot_df.groupby("alignment_condition_id")[
        "exposure_outgoing_visibility_weight"
    ].rank(pct=True)

    positions = set(plot_df["exposure_position_id"])
    edges = _load_edges(positions)
    layout = _network_layout(positions, edges)
    plot_df["layout_x"] = plot_df["exposure_position_id"].map(lambda pos: float(layout[str(pos)][0]))
    plot_df["layout_y"] = plot_df["exposure_position_id"].map(lambda pos: float(layout[str(pos)][1]))
    plot_df["node_area"] = 18 + 118 * plot_df["sender_reach_percentile_within_condition"].fillna(0.5)
    return plot_df


def _write_branch_overlay_nodes(plot_df: pd.DataFrame, branch_root: Path) -> Path:
    tables_dir = ensure_dir(branch_root / "network_exposure_analysis" / "tables")
    path = tables_dir / "branch_full_network_overlay_nodes.csv"
    keep = [
        "alignment_condition_id",
        "condition_index",
        "opinion_index",
        "attack_index",
        "opinion_label",
        "attack_label",
        "target_alignment_z",
        "achieved_alignment_z",
        "profile_id",
        "exposure_position_id",
        "ae_private",
        "ae_private_percentile_within_condition",
        "exposure_outgoing_visibility_weight",
        "sender_reach_percentile_within_condition",
        "node_area",
        "layout_x",
        "layout_y",
    ]
    plot_df[[col for col in keep if col in plot_df.columns]].sort_values(
        ["condition_index", "sender_reach_percentile_within_condition", "ae_private_percentile_within_condition"],
        ascending=[True, False, False],
    ).to_csv(path, index=False)
    return path


def _write_branch_overlay_edges(edges: pd.DataFrame, backbone_edges: pd.DataFrame, branch_root: Path) -> Path:
    tables_dir = ensure_dir(branch_root / "network_exposure_analysis" / "tables")
    path = tables_dir / "branch_full_network_overlay_edges.csv"
    context = edges.copy()
    context["edge_layer"] = "context_induced_prompt_top30"
    context["backbone_reason"] = ""
    backbone = backbone_edges.copy()
    backbone["edge_layer"] = "directed_backbone"
    if "backbone_reason" not in backbone.columns:
        backbone["backbone_reason"] = ""
    combined = pd.concat([context, backbone], ignore_index=True)
    for column in ["rank_for_receiver", "interaction_types"]:
        if column not in combined.columns:
            combined[column] = ""
    keep = [
        "source_position_id",
        "target_position_id",
        "exposure_weight",
        "rank_for_receiver",
        "interaction_types",
        "edge_layer",
        "backbone_reason",
    ]
    combined[keep].sort_values(
        ["edge_layer", "target_position_id", "exposure_weight"],
        ascending=[True, True, False],
    ).to_csv(path, index=False)
    return path


def _plot_full_condition_network_overlays(
    sem: pd.DataFrame, condition: pd.DataFrame, branch_root: Path, figures_dir: Path
) -> tuple[Path, Path, Path]:
    plot_df = _branch_overlay_nodes(sem, condition)
    nodes_path = _write_branch_overlay_nodes(plot_df, branch_root)
    positions = set(plot_df["exposure_position_id"])
    edges = _load_edges(positions)
    backbone_edges = _branch_overlay_backbone_edges(edges)
    edges_path = _write_branch_overlay_edges(edges, backbone_edges, branch_root)
    layout = {
        str(pos): np.array([float(row["layout_x"]), float(row["layout_y"])])
        for pos, row in plot_df.drop_duplicates("exposure_position_id")
        .set_index("exposure_position_id")[["layout_x", "layout_y"]]
        .iterrows()
    }
    segments = _edge_segments(edges, layout)
    backbone_edgelist = list(
        zip(backbone_edges["source_position_id"].astype(str), backbone_edges["target_position_id"].astype(str))
    )
    backbone_graph = nx.from_pandas_edgelist(
        backbone_edges,
        source="source_position_id",
        target="target_position_id",
        create_using=nx.DiGraph,
    )
    backbone_weight_unit = _unit_interval(backbone_edges["exposure_weight"])
    backbone_colors = [
        mcolors.to_rgba("#464C55", alpha=0.30 + 0.42 * float(weight)) for weight in backbone_weight_unit
    ]
    backbone_widths = (0.38 + 1.05 * backbone_weight_unit).to_numpy()

    condition_order = condition.copy()
    if "opinion_index" in condition_order.columns and "attack_index" in condition_order.columns:
        opinion_order = (
            condition_order.sort_values("opinion_index")
            .drop_duplicates("opinion_label")["opinion_label"]
            .tolist()
        )
        attack_order = (
            condition_order.sort_values("attack_index")
            .drop_duplicates("attack_label")["attack_label"]
            .tolist()
        )
    else:
        opinion_order = sorted(condition_order["opinion_label"].dropna().unique())
        attack_order = sorted(condition_order["attack_label"].dropna().unique())

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(
        len(opinion_order),
        len(attack_order),
        figsize=(19.2, 22.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("#FCFCFD")
    cmap = LinearSegmentedColormap.from_list(
        "susceptibility_blue_orange_percentile",
        ["#2E4780", "#FFFFFF", "#804126"],
        N=256,
    )
    x_vals = np.array([xy[0] for xy in layout.values()], dtype=float)
    y_vals = np.array([xy[1] for xy in layout.values()], dtype=float)
    x_pad = max((x_vals.max() - x_vals.min()) * 0.08, 0.02)
    y_pad = max((y_vals.max() - y_vals.min()) * 0.08, 0.02)
    xlim = (float(x_vals.min() - x_pad), float(x_vals.max() + x_pad))
    ylim = (float(y_vals.min() - y_pad), float(y_vals.max() + y_pad))

    for row_idx, opinion in enumerate(opinion_order):
        for col_idx, attack in enumerate(attack_order):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if segments:
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors="#C5CAD3",
                        linewidths=0.20,
                        alpha=0.12,
                        zorder=1,
                    )
                )
            if backbone_edgelist:
                arrow_artists = nx.draw_networkx_edges(
                    backbone_graph,
                    pos=layout,
                    ax=ax,
                    edgelist=backbone_edgelist,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=5.8,
                    connectionstyle="arc3,rad=0.035",
                    edge_color=backbone_colors,
                    width=backbone_widths,
                    node_size=0,
                    min_source_margin=2.0,
                    min_target_margin=4.5,
                )
                if arrow_artists is not None:
                    for artist in arrow_artists:
                        artist.set_zorder(2)
            group = plot_df[plot_df["opinion_label"].eq(opinion) & plot_df["attack_label"].eq(attack)]
            if group.empty:
                continue
            ax.scatter(
                group["layout_x"],
                group["layout_y"],
                s=group["node_area"],
                c=group["ae_private_percentile_within_condition"],
                cmap=cmap,
                vmin=0,
                vmax=1,
                edgecolors="white",
                linewidths=0.42,
                alpha=0.96,
                zorder=3,
            )
            achieved = float(group["achieved_alignment_z"].iloc[0])
            if row_idx == 0:
                ax.set_title(attack.replace("_", " "), fontsize=8.2, color="#1F2937", pad=7)
            if col_idx == 0:
                ax.text(
                    -0.09,
                    0.5,
                    opinion.replace("_", " "),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=8.0,
                    color="#374151",
                )
            ax.text(
                0.03,
                0.04,
                _sender_alignment_label(achieved),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.3,
                color="#4B5563",
            )

    cbar_ax = fig.add_axes([0.91, 0.20, 0.012, 0.58])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), cax=cbar_ax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["lower\nresilient", "condition\nmedian", "higher\nsusceptible"])
    cbar.set_label("Relative AE_private percentile within condition", fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(
        "Full 35-Condition Network Overlay",
        fontsize=13.5,
        color="#111827",
        y=0.982,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#7A828F",
            markeredgewidth=0.9,
            markersize=4.5,
            label="low sender reach",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#7A828F",
            markeredgewidth=0.9,
            markersize=7.2,
            label="median sender reach",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#7A828F",
            markeredgewidth=0.9,
            markersize=9.4,
            label="high sender reach",
        ),
        Line2D([0], [0], color="#C5CAD3", lw=0.9, alpha=0.65, label="induced exposure edge"),
        Line2D([0], [0], color="#464C55", lw=1.05, marker=">", markersize=5, label="directed exposure backbone"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=False,
        fontsize=8.0,
        handlelength=2.1,
        columnspacing=1.8,
    )
    fig.text(
        0.5,
        0.054,
        "Faint lines show all induced prompt-top30 exposure edges; darker arrows show a sparse high-signal directed backbone.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.065, right=0.895, top=0.945, bottom=0.085, hspace=0.04, wspace=0.035)
    return _save_fig(fig, figures_dir / "branch_full_35_condition_network_overlay.png"), nodes_path, edges_path
