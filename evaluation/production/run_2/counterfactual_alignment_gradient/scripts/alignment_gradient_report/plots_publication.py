from __future__ import annotations

"""Publication-grade composite figure builders for the alignment-gradient mechanism.

The panel helpers reuse the report plotting data but draw into tightly controlled A/B and A/B/C layouts for manuscript assets.
"""

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .data import _condition_vulnerability_plane_frame
from .formatting import _fmt, _fmt_p, _label, _sender_alignment_label, _signed_tick_label
from .models import smf
from .paths import ensure_dir
from .plots_network import (
    _branch_overlay_backbone_edges,
    _branch_overlay_nodes,
    _edge_segments,
    _load_edges,
    _unit_interval,
)


def _condition_panel_order(condition: pd.DataFrame) -> tuple[list[str], list[str]]:
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
    return opinion_order, attack_order


def _publication_panel_header(fig: plt.Figure, spec: Any, panel_letter: str, title: str) -> Any:
    header_ax = fig.add_subplot(spec)
    header_ax.axis("off")
    header_ax.text(
        0.0,
        0.57,
        panel_letter,
        ha="left",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#1F2430",
    )
    header_ax.text(
        0.050,
        0.59,
        title,
        ha="left",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#1F2430",
    )
    return header_ax


def _draw_publication_network_overlay_panel(fig: plt.Figure, spec: Any, sem: pd.DataFrame, condition: pd.DataFrame) -> None:
    plot_df = _branch_overlay_nodes(sem, condition)
    positions = set(plot_df["exposure_position_id"])
    edges = _load_edges(positions)
    backbone_edges = _branch_overlay_backbone_edges(edges)
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
    opinion_order, attack_order = _condition_panel_order(condition)
    cmap = LinearSegmentedColormap.from_list(
        "publication_network_blue_orange",
        ["#2E4780", "#FFFFFF", "#804126"],
        N=256,
    )
    x_vals = np.array([xy[0] for xy in layout.values()], dtype=float)
    y_vals = np.array([xy[1] for xy in layout.values()], dtype=float)
    x_pad = max((x_vals.max() - x_vals.min()) * 0.08, 0.02)
    y_pad = max((y_vals.max() - y_vals.min()) * 0.08, 0.02)
    xlim = (float(x_vals.min() - x_pad), float(x_vals.max() + x_pad))
    ylim = (float(y_vals.min() - y_pad), float(y_vals.max() + y_pad))

    grid = spec.subgridspec(
        9,
        6,
        height_ratios=[0.18, 1, 1, 1, 1, 1, 1, 1, 0.20],
        width_ratios=[1, 1, 1, 1, 1, 0.075],
        hspace=0.052,
        wspace=0.050,
    )
    _publication_panel_header(
        fig,
        grid[0, :],
        "A",
        "Full 35-Condition Network Overlay",
    )
    for row_idx, opinion in enumerate(opinion_order):
        for col_idx, attack in enumerate(attack_order):
            ax = fig.add_subplot(grid[row_idx + 1, col_idx])
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if segments:
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors="#C5CAD3",
                        linewidths=0.19,
                        alpha=0.115,
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
                    arrowsize=5.2,
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
                s=group["node_area"] * 0.82,
                c=group["ae_private_percentile_within_condition"],
                cmap=cmap,
                vmin=0,
                vmax=1,
                edgecolors="white",
                linewidths=0.32,
                alpha=0.96,
                zorder=3,
            )
            achieved = float(group["achieved_alignment_z"].iloc[0])
            if row_idx == 0:
                ax.set_title(_label(attack), fontsize=6.8, color="#1F2937", pad=4)
            if col_idx == 0:
                ax.text(
                    -0.090,
                    0.5,
                    _label(opinion),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=6.5,
                    color="#374151",
                )
            ax.text(
                0.03,
                0.04,
                _sender_alignment_label(achieved),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=4.9,
                color="#4B5563",
            )

    cbar_ax = fig.add_subplot(grid[1:8, 5])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), cax=cbar_ax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["lower\nresilient", "condition\nmedian", "higher\nsusceptible"])
    cbar.set_label("Relative AE_private percentile within condition", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.outline.set_visible(False)

    legend_ax = fig.add_subplot(grid[8, :])
    legend_ax.axis("off")
    legend_ax.text(
        0.5,
        0.82,
        "Faint lines = induced prompt-top30 exposure edges; darker arrows = sparse directed backbone.",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#6F768A",
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#7A828F", markersize=3.6, label="low sender reach"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#7A828F", markersize=5.6, label="median sender reach"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#7A828F", markersize=7.0, label="high sender reach"),
        Line2D([0], [0], color="#C5CAD3", lw=0.8, alpha=0.65, label="induced exposure edge"),
        Line2D([0], [0], color="#464C55", lw=0.95, marker=">", markersize=4.4, label="directed backbone"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=6.2,
        handlelength=1.75,
        columnspacing=1.0,
    )


def _draw_publication_vulnerability_plane_panel(fig: plt.Figure, spec: Any, sem: pd.DataFrame, condition: pd.DataFrame) -> None:
    plot_df = _condition_vulnerability_plane_frame(sem, condition)
    opinion_order, attack_order = _condition_panel_order(condition)
    cmap = LinearSegmentedColormap.from_list(
        "publication_plane_blue_orange",
        ["#2E4780", "#FFFFFF", "#804126"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
    grid = spec.subgridspec(
        9,
        6,
        height_ratios=[0.18, 1, 1, 1, 1, 1, 1, 1, 0.20],
        width_ratios=[1, 1, 1, 1, 1, 0.075],
        hspace=0.17,
        wspace=0.105,
    )
    _publication_panel_header(
        fig,
        grid[0, :],
        "B",
        "Condition-Specific Vulnerability Planes",
    )
    for row_idx, opinion in enumerate(opinion_order):
        for col_idx, attack in enumerate(attack_order):
            ax = fig.add_subplot(grid[row_idx + 1, col_idx])
            group = plot_df[plot_df["opinion_label"].eq(opinion) & plot_df["attack_label"].eq(attack)].copy()
            ax.axhspan(0.80, 1.03, color="#F4F5F7", alpha=0.90, zorder=0)
            ax.axvline(0, color="#7A828F", linestyle=":", linewidth=0.58, zorder=1)
            ax.axhline(0.80, color="#7A828F", linestyle=":", linewidth=0.58, zorder=1)
            ax.set_xlim(-2.8, 2.8)
            ax.set_ylim(0, 1.03)
            ax.grid(True, color="#E6E8F0", linewidth=0.42, alpha=0.85)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#D7DBE7")
            ax.spines["bottom"].set_color("#D7DBE7")
            if group.empty:
                ax.axis("off")
                continue
            low = group[~group["high_reach_sender"]]
            high = group[group["high_reach_sender"]]
            top10 = group[group["top10_sender"]]
            ax.scatter(
                low["ae_private_z"],
                low["sender_reach_percentile"],
                s=8,
                c=low["ae_private_z"],
                cmap=cmap,
                norm=norm,
                alpha=0.28,
                edgecolor="none",
                zorder=2,
            )
            ax.scatter(
                high["ae_private_z"],
                high["sender_reach_percentile"],
                s=22,
                c=high["ae_private_z"],
                cmap=cmap,
                norm=norm,
                alpha=0.95,
                edgecolor="white",
                linewidth=0.30,
                zorder=3,
            )
            ax.scatter(
                top10["ae_private_z"],
                top10["sender_reach_percentile"],
                s=31,
                facecolors="none",
                edgecolors="#1F2430",
                linewidth=0.46,
                alpha=0.90,
                zorder=4,
            )
            achieved = float(group["achieved_alignment_z"].iloc[0])
            if row_idx == 0:
                ax.set_title(_label(attack), fontsize=6.7, color="#1F2937", pad=4)
            if col_idx == 0:
                ax.set_ylabel(_label(opinion), fontsize=6.5, color="#374151")
            else:
                ax.set_ylabel("")
            ax.text(
                0.04,
                0.05,
                _sender_alignment_label(achieved),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=4.8,
                color="#4B5563",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#E6E8F0", "alpha": 0.88},
            )
            ax.tick_params(labelsize=5.2, length=1.5)
            if row_idx < len(opinion_order) - 1:
                ax.tick_params(labelbottom=False)
            if col_idx > 0:
                ax.tick_params(labelleft=False)

    cbar_ax = fig.add_subplot(grid[1:8, 5])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("AE_private z within condition", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.outline.set_visible(False)
    footer_ax = fig.add_subplot(grid[8, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5,
        0.58,
        "x: within-condition AE_private z-score; y: direct sender-reach percentile. Shaded band = top 20%; outlined points = top 10%.",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#6F768A",
    )


def _draw_publication_alignment_outcome_panel(
    fig: plt.Figure,
    spec: Any,
    condition: pd.DataFrame,
    figure4_summary: pd.DataFrame,
) -> None:
    grid = spec.subgridspec(
        3,
        3,
        height_ratios=[0.22, 1.0, 0.12],
        width_ratios=[1.0, 1.0, 0.038],
        hspace=0.14,
        wspace=0.115,
    )
    _publication_panel_header(
        fig,
        grid[0, :],
        "C",
        "Alignment-Gradient Outcome Test",
    )
    cmap = LinearSegmentedColormap.from_list(
        "publication_outcome_blue_orange",
        ["#2E4780", "#FFFFFF", "#804126"],
        N=256,
    )
    norm = plt.Normalize(-0.95, 0.95)
    specs = [
        (
            "mean_pn_increment_effectivity",
            "Post-network amplification\nattack-aligned score points (FE residual)",
            "Primary H3/H4 mechanism",
        ),
        (
            "mean_ae_total_network",
            "Final network attack effect\nattack-aligned score points (FE residual)",
            "Secondary final-effect endpoint",
        ),
    ]
    for col_idx, (outcome, ylabel, title) in enumerate(specs):
        ax = fig.add_subplot(grid[1, col_idx])
        plot_df = condition.dropna(subset=["achieved_alignment_z", outcome]).copy()
        if smf is not None:
            x_model = smf.ols("achieved_alignment_z ~ C(opinion_label) + C(attack_label)", data=plot_df).fit()
            y_model = smf.ols(f"{outcome} ~ C(opinion_label) + C(attack_label)", data=plot_df).fit()
            plot_df["alignment_fe_residual"] = x_model.resid
            plot_df["outcome_fe_residual"] = y_model.resid
        else:
            plot_df["alignment_fe_residual"] = plot_df["achieved_alignment_z"]
            plot_df["outcome_fe_residual"] = plot_df[outcome]

        summary_row = figure4_summary[figure4_summary["outcome"].eq(outcome)]
        summary = summary_row.iloc[0].to_dict() if not summary_row.empty else {}
        max_abs_x = max(float(plot_df["alignment_fe_residual"].abs().max()), 0.01)
        max_abs_y = max(float(plot_df["outcome_fe_residual"].abs().max()), 0.01)
        xlim = (-max_abs_x * 1.12, max_abs_x * 1.12)
        ylim = (-max_abs_y * 1.16, max_abs_y * 1.16)

        ax.set_facecolor("#FFFFFF")
        ax.axvline(0, color="#7A828F", linestyle=":", linewidth=0.95, zorder=1)
        ax.axhline(0, color="#7A828F", linestyle=":", linewidth=0.95, zorder=1)
        beta = float(summary.get("beta", np.nan))
        ci_low = float(summary.get("hc3_ci_low", np.nan))
        ci_high = float(summary.get("hc3_ci_high", np.nan))
        if not any(math.isnan(value) for value in [beta, ci_low, ci_high]):
            xs = np.linspace(xlim[0], xlim[1], 220)
            lower = np.minimum(ci_low * xs, ci_high * xs)
            upper = np.maximum(ci_low * xs, ci_high * xs)
            ax.fill_between(xs, lower, upper, color="#C5CAD3", alpha=0.42, linewidth=0, zorder=2)
            ax.plot(xs, beta * xs, color="#1F2430", linewidth=1.75, zorder=3)

        ax.scatter(
            plot_df["alignment_fe_residual"],
            plot_df["outcome_fe_residual"],
            s=78,
            c=plot_df["achieved_alignment_z"],
            cmap=cmap,
            norm=norm,
            edgecolor="#FFFFFF",
            linewidth=0.75,
            alpha=0.95,
            zorder=4,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10.3, color="#1F2430", pad=8, fontweight="bold")
        ax.set_xlabel("Adjusted sender-reach susceptibility alignment (z)", fontsize=8.7)
        ax.set_ylabel(ylabel, fontsize=8.7)
        ax.xaxis.set_major_formatter(FuncFormatter(_signed_tick_label))
        ax.yaxis.set_major_formatter(FuncFormatter(_signed_tick_label))
        ax.grid(True, color="#E6E8F0", linewidth=0.72)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#D7DBE7")
        ax.spines["bottom"].set_color("#D7DBE7")
        ax.tick_params(labelsize=7.1, length=2.2)
        ax.text(
            0.035,
            0.960,
            "\n".join(
                [
                    f"beta = {_fmt(summary.get('beta'))} score points / 1 SD",
                    f"95% HC3 CI [{_fmt(summary.get('hc3_ci_low'))}, {_fmt(summary.get('hc3_ci_high'))}]",
                    f"HC3 p = {_fmt_p(summary.get('hc3_p'))}",
                    f"permutation p = {_fmt_p(summary.get('permutation_p'))}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#1F2430",
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#D7DBE7", "alpha": 0.93},
        )

    cbar_ax = fig.add_subplot(grid[1, 2])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_ticks([-0.9, 0.0, 0.9])
    cbar.set_ticklabels(["resilient\nhub", "neutral", "susceptible\nhub"])
    cbar.set_label("Raw sender alignment z", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.4, length=2)
    cbar.outline.set_visible(False)

    footer_ax = fig.add_subplot(grid[2, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5,
        0.52,
        "Each point is one opinion x attack condition. Adjusted axes remove opinion and attack fixed effects; line is the HC3 fixed-effect coefficient and shaded wedge is coefficient uncertainty.",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#6F768A",
    )


def _pad_figure_bounds(bounds: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
    left, bottom, width, height = bounds
    right = left + width
    top = bottom + height
    left = max(0.0, left - pad)
    bottom = max(0.0, bottom - pad)
    right = min(1.0, right + pad)
    top = min(1.0, top + pad)
    return (left, bottom, right - left, top - bottom)


def _union_figure_bounds(bounds_list: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    left = min(bounds[0] for bounds in bounds_list)
    bottom = min(bounds[1] for bounds in bounds_list)
    right = max(bounds[0] + bounds[2] for bounds in bounds_list)
    top = max(bounds[1] + bounds[3] for bounds in bounds_list)
    return (left, bottom, right - left, top - bottom)


def _add_figure_rect(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    zorder: float,
) -> None:
    fig.add_artist(
        Rectangle(
            (bounds[0], bounds[1]),
            bounds[2],
            bounds[3],
            transform=fig.transFigure,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            clip_on=False,
            zorder=zorder,
        )
    )


def _add_publication_composite_frames(
    fig: plt.Figure,
    panel_bounds: list[tuple[float, float, float, float]],
    *,
    background: bool,
) -> None:
    if background:
        for bounds in panel_bounds:
            _add_figure_rect(
                fig,
                bounds,
                facecolor="#FFFFFF",
                edgecolor="none",
                linewidth=0.0,
                zorder=-20,
            )
        return

    for bounds in panel_bounds:
        _add_figure_rect(
            fig,
            bounds,
            facecolor="none",
            edgecolor="#C5CAD3",
            linewidth=0.8,
            zorder=80,
        )
    _add_figure_rect(
        fig,
        _pad_figure_bounds(_union_figure_bounds(panel_bounds), 0.010),
        facecolor="none",
        edgecolor="#7A828F",
        linewidth=1.0,
        zorder=90,
    )


def _publication_panel_inner_spec(
    spec: Any,
    *,
    left: float = 0.025,
    right: float = 0.025,
    top: float = 0.022,
    bottom: float = 0.022,
) -> Any:
    inner_grid = spec.subgridspec(
        3,
        3,
        height_ratios=[top, 1.0, bottom],
        width_ratios=[left, 1.0, right],
        hspace=0.0,
        wspace=0.0,
    )
    return inner_grid[1, 1]


def _plot_publication_network_mechanism_composite(
    sem: pd.DataFrame, condition: pd.DataFrame, branch_root: Path
) -> dict[str, Path]:
    publication_dir = ensure_dir(branch_root / "network_exposure_analysis" / "publication_figures")
    sns.set_theme(style="white", context="paper")
    fig = plt.figure(figsize=(32.6, 18.9), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.050,
        left=0.026,
        right=0.985,
        top=0.982,
        bottom=0.035,
    )
    _draw_publication_network_overlay_panel(fig, outer[0], sem, condition)
    _draw_publication_vulnerability_plane_panel(fig, outer[1], sem, condition)

    base = publication_dir / "main_figure_network_mechanism_ab"
    outputs = {
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
        "png": base.with_suffix(".png"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def _plot_publication_network_mechanism_composite_abc(
    sem: pd.DataFrame,
    condition: pd.DataFrame,
    figure4_summary: pd.DataFrame,
    branch_root: Path,
) -> dict[str, Path]:
    publication_dir = ensure_dir(
        branch_root
        / "network_exposure_analysis"
        / "publication_figures"
        / "network_mechanism_abc_with_outcome_test"
    )
    sns.set_theme(style="white", context="paper")
    fig = plt.figure(figsize=(32.6, 29.0), facecolor="#FCFCFD")
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[18.9, 9.0],
        width_ratios=[1.0, 1.0],
        hspace=0.075,
        wspace=0.055,
        left=0.034,
        right=0.976,
        top=0.970,
        bottom=0.042,
    )
    panel_bounds = [
        outer[0, 0].get_position(fig).bounds,
        outer[0, 1].get_position(fig).bounds,
        outer[1, :].get_position(fig).bounds,
    ]
    _add_publication_composite_frames(fig, panel_bounds, background=True)
    _draw_publication_network_overlay_panel(
        fig,
        _publication_panel_inner_spec(outer[0, 0], left=0.030, right=0.026, top=0.024, bottom=0.024),
        sem,
        condition,
    )
    _draw_publication_vulnerability_plane_panel(
        fig,
        _publication_panel_inner_spec(outer[0, 1], left=0.026, right=0.034, top=0.024, bottom=0.024),
        sem,
        condition,
    )
    _draw_publication_alignment_outcome_panel(
        fig,
        _publication_panel_inner_spec(outer[1, :], left=0.018, right=0.026, top=0.025, bottom=0.028),
        condition,
        figure4_summary,
    )
    _add_publication_composite_frames(fig, panel_bounds, background=False)

    base = publication_dir / "main_figure_network_mechanism_abc_with_outcome_test"
    outputs = {
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
        "png": base.with_suffix(".png"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs
