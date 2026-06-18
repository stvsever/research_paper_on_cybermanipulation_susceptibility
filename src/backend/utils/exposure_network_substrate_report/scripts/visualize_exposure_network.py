from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd

from common import (
    DERIVED_DIR,
    FIGURES_DIR,
    INTERACTION_WEIGHTS,
    RANDOM_SEED,
    ROLE_PALETTE,
    ROLE_LABELS,
    TABLES_DIR,
    TOKENS,
    artifact_path,
    ensure_dirs,
    write_json,
)


FONT = ["DejaVu Sans", "Arial", "sans-serif"]
MONO = ["DejaVu Sans Mono", "monospace"]
ROLE_ORDER = [
    "high_visibility_sender",
    "high_exposure_receiver",
    "bridge",
    "peripheral",
    "context_position",
]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "font.family": FONT,
            "font.size": 10,
        }
    )


def _save(fig: plt.Figure, name: str, *, svg: bool = False) -> dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    paths = {"png": artifact_path(png_path)}
    if svg:
        svg_path = FIGURES_DIR / f"{name}.svg"
        fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths["svg"] = artifact_path(svg_path)
    plt.close(fig)
    return paths


def _role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " "))


def _build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    graph = nx.from_pandas_edgelist(
        edges,
        source="source_position_id",
        target="target_position_id",
        edge_attr=["exposure_weight", "raw_weight", "total_events", "interaction_types"],
        create_using=nx.DiGraph,
    )
    graph.add_nodes_from(nodes["position_id"].astype(int).tolist())
    return graph


def _draw_direction_legend() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.set_axis_off()
    left = (0.24, 0.58)
    right = (0.68, 0.58)
    ax.scatter([left[0]], [left[1]], s=1800, color=TOKENS["orange"], edgecolors=TOKENS["panel"], linewidths=2)
    ax.scatter([right[0]], [right[1]], s=1800, color=TOKENS["blue"], edgecolors=TOKENS["panel"], linewidths=2)
    ax.annotate(
        "",
        xy=(right[0] - 0.08, right[1]),
        xytext=(left[0] + 0.08, left[1]),
        arrowprops=dict(arrowstyle="-|>", color=TOKENS["neutral_dark"], lw=2.0, mutation_scale=18),
    )
    ax.text(left[0], left[1], "Target\nvisible", ha="center", va="center", fontsize=10, fontweight="bold", color=TOKENS["ink"])
    ax.text(right[0], right[1], "Source\nexposed", ha="center", va="center", fontsize=10, fontweight="bold", color=TOKENS["ink"])
    ax.text(0.46, 0.55, "TargetUserId -> SourceUserId", ha="center", va="center", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    formula = (
        "exposure_raw_weight = "
        f"{INTERACTION_WEIGHTS['Like']:.2f} * Like + "
        f"{INTERACTION_WEIGHTS['Repost']:.2f} * Repost + "
        f"{INTERACTION_WEIGHTS['Quote']:.2f} * Quote\n"
        "Repeated interactions are summed, log-compressed, and normalized to [0, 1]."
    )
    ax.text(
        0.5,
        0.12,
        formula,
        ha="center",
        va="center",
        fontsize=10,
        family=MONO,
        color=TOKENS["ink"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F5F7", edgecolor=TOKENS["axis"], linewidth=1),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, "exposure_direction_and_formula", svg=True)


def _draw_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    graph = _build_graph(edges, nodes)
    pos = nx.spring_layout(graph.to_undirected(), seed=RANDOM_SEED, weight="exposure_weight", k=0.78)
    node_index = nodes.set_index("position_id")
    max_out = float(np.log1p(nodes["weighted_out_degree"]).max()) or 1.0
    node_sizes = [
        70 + 520 * float(np.log1p(node_index.loc[node, "weighted_out_degree"])) / max_out
        if node in node_index.index
        else 90
        for node in graph.nodes()
    ]
    node_colors = [ROLE_PALETTE.get(str(node_index.loc[node, "display_role"]), TOKENS["neutral"]) for node in graph.nodes()]
    widths = [0.25 + 2.2 * float(data.get("exposure_weight", 0.2)) for _, _, data in graph.edges(data=True)]
    alphas = [0.12 + 0.28 * float(data.get("exposure_weight", 0.2)) for _, _, data in graph.edges(data=True)]

    for (source, target, data), width, alpha in zip(graph.edges(data=True), widths, alphas):
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(source, target)],
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=8,
            alpha=alpha,
            width=width,
            edge_color=TOKENS["muted"],
            connectionstyle="arc3,rad=0.045",
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=TOKENS["panel"],
        linewidths=1.0,
        alpha=0.96,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ROLE_PALETTE[role],
            markeredgecolor=TOKENS["panel"],
            label=_role_label(role),
            markersize=8,
        )
        for role in ROLE_ORDER
        if role in set(nodes["display_role"])
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False, fontsize=8.5)
    ax.set_axis_off()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, "pilot_directed_exposure_network", svg=True)


def _draw_rankings(nodes: pd.DataFrame) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    chart_specs = [
        (
            "pilot_sender_reach_ranking",
            "Highest sender-reach positions in the pilot slice",
            "Bars show selected-slice outgoing exposure weight; color indicates sampling role.",
            "selected_visibility_sent_weight",
            "Selected outgoing exposure weight",
            False,
        ),
        (
            "pilot_receiver_exposure_ranking",
            "Highest receiver-exposure positions in the pilot slice",
            "Bars show selected-slice incoming exposure weight; color indicates sampling role.",
            "selected_exposure_received_weight",
            "Selected incoming exposure weight",
            False,
        ),
    ]
    for name, title, subtitle, metric, xlabel, _ in chart_specs:
        top = nodes.sort_values(metric, ascending=False).head(14).copy()
        top["label"] = top["position_id"].astype(str)
        colors = [ROLE_PALETTE.get(role, TOKENS["neutral"]) for role in top["display_role"]]
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.barh(top["label"], top[metric], color=colors, edgecolor=TOKENS["panel"], linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Position id")
        ax.grid(axis="x", alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))
        for y, value in enumerate(top[metric]):
            ax.text(value, y, f" {value:.2f}", va="center", ha="left", fontsize=8, color=TOKENS["muted"])
        fig.tight_layout()
        outputs[name] = _save(fig, name, svg=False)
    return outputs


def _draw_role_summary(role_summary: pd.DataFrame) -> dict[str, str]:
    ordered = [role for role in ROLE_ORDER if role in set(role_summary["display_role"])]
    data = role_summary.set_index("display_role").loc[ordered].reset_index()
    y = np.arange(len(data))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.barh(
        y - height / 2,
        data["selected_visibility_sent_weight"],
        height=height,
        color=TOKENS["orange"],
        label="Visibility sent",
    )
    ax.barh(
        y + height / 2,
        data["selected_exposure_received_weight"],
        height=height,
        color=TOKENS["blue"],
        label="Exposure received",
    )
    ax.set_yticks(y, [_role_label(role) for role in data["display_role"]])
    ax.invert_yaxis()
    ax.set_xlabel("Summed normalized exposure weight")
    ax.grid(axis="x", alpha=0.75)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "pilot_role_exposure_summary", svg=False)


def make_figures() -> dict[str, object]:
    ensure_dirs()
    _setup_style()
    nodes = pd.read_csv(DERIVED_DIR / "pilot_nodes.csv")
    edges = pd.read_csv(DERIVED_DIR / "pilot_edges.csv")
    role_summary = pd.read_csv(TABLES_DIR / "pilot_role_summary.csv")

    figures: dict[str, object] = {
        "exposure_direction_and_formula": _draw_direction_legend(),
        "pilot_directed_exposure_network": _draw_network(nodes, edges),
        "pilot_role_exposure_summary": _draw_role_summary(role_summary),
    }
    figures.update(_draw_rankings(nodes))
    write_json(FIGURES_DIR / "figure_manifest.json", figures)
    return figures


if __name__ == "__main__":
    print(json.dumps(make_figures(), indent=2))
