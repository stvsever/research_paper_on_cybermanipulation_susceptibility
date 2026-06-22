from __future__ import annotations

"""Standard diagnostic and hypothesis figures for the HTML report.

These figures check manipulation quality, fixed-effect outcome relationships, robustness, and branch quality gates.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .data import _condition_vulnerability_plane_frame
from .formatting import _fmt, _fmt_p, _label, _save_fig, _sender_alignment_label, _signed_tick_label
from .models import smf
from .plots_network import _load_edges


def _plot_target_vs_achieved(condition: pd.DataFrame, figures_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    sns.scatterplot(
        data=condition,
        x="target_alignment_z",
        y="achieved_alignment_z",
        hue="attack_label",
        s=65,
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
    )
    lo = min(condition["target_alignment_z"].min(), condition["achieved_alignment_z"].min()) - 0.05
    hi = max(condition["target_alignment_z"].max(), condition["achieved_alignment_z"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="#374151", linestyle="--", linewidth=1.0, label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Target sender-reach susceptibility alignment z")
    ax.set_ylabel("Achieved alignment z")
    ax.set_title("Alignment manipulation quality")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, figures_dir / "target_vs_achieved_alignment.png")


def _plot_alignment_outcomes(condition: pd.DataFrame, figure4_summary: pd.DataFrame, figures_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2))
    fig.patch.set_facecolor("#FCFCFD")
    cmap = LinearSegmentedColormap.from_list(
        "alignment_blue_orange",
        ["#2E4780", "#FFFFFF", "#804126"],
        N=256,
    )
    norm = plt.Normalize(-0.95, 0.95)
    specs = [
        (
            "mean_pn_increment_effectivity",
            "Post-network amplification,\nattack-aligned score points (FE residual)",
            "Primary H3/H4 mechanism",
        ),
        (
            "mean_ae_total_network",
            "Final network attack effect,\nattack-aligned score points (FE residual)",
            "Secondary final-effect endpoint",
        ),
    ]
    for ax, (outcome, ylabel, title) in zip(axes, specs):
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
        if summary_row.empty:
            summary = {
                "beta": np.nan,
                "hc3_ci_low": np.nan,
                "hc3_ci_high": np.nan,
                "hc3_p": np.nan,
                "permutation_p": np.nan,
            }
        else:
            summary = summary_row.iloc[0].to_dict()
        max_abs_x = max(float(plot_df["alignment_fe_residual"].abs().max()), 0.01)
        max_abs_y = max(float(plot_df["outcome_fe_residual"].abs().max()), 0.01)
        xlim = (-max_abs_x * 1.15, max_abs_x * 1.15)
        ylim = (-max_abs_y * 1.20, max_abs_y * 1.20)
        ax.axvline(0, color="#6F768A", linestyle=":", linewidth=0.95, zorder=1)
        ax.axhline(0, color="#6F768A", linestyle=":", linewidth=0.95, zorder=1)
        beta = float(summary.get("beta", np.nan))
        ci_low = float(summary.get("hc3_ci_low", np.nan))
        ci_high = float(summary.get("hc3_ci_high", np.nan))
        if not any(math.isnan(value) for value in [beta, ci_low, ci_high]):
            xs = np.linspace(xlim[0], xlim[1], 200)
            lower = np.minimum(ci_low * xs, ci_high * xs)
            upper = np.maximum(ci_low * xs, ci_high * xs)
            ax.fill_between(xs, lower, upper, color="#C5CAD3", alpha=0.45, linewidth=0, zorder=2)
            ax.plot(xs, beta * xs, color="#1F2430", linewidth=1.55, zorder=3)
        ax.scatter(
            plot_df["alignment_fe_residual"],
            plot_df["outcome_fe_residual"],
            s=76,
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
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FuncFormatter(_signed_tick_label))
        ax.yaxis.set_major_formatter(FuncFormatter(_signed_tick_label))
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.03,
            0.96,
            "\n".join(
                [
                    f"β = {_fmt(summary.get('beta'))} score points / 1 SD",
                    f"95% HC3 CI [{_fmt(summary.get('hc3_ci_low'))}, {_fmt(summary.get('hc3_ci_high'))}]",
                    f"HC3 p = {_fmt_p(summary.get('hc3_p'))}",
                    f"permutation p = {_fmt_p(summary.get('permutation_p'))}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#1F2430",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#D7DBE7", "alpha": 0.92},
        )
    cbar_ax = fig.add_axes([0.31, 0.09, 0.38, 0.022])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.set_ticks([-0.9, 0.0, 0.9])
    cbar.set_ticklabels(["resilient-hub", "neutral", "susceptible-hub"])
    cbar.set_label("Raw sender alignment z (context only)", fontsize=8.5)
    cbar.ax.xaxis.set_label_position("top")
    cbar.outline.set_visible(False)
    fig.suptitle("Alignment-Gradient Outcome Test", y=0.982, fontsize=13.6, color="#1F2430")
    fig.supxlabel(
        "Adjusted sender-reach susceptibility alignment (z): resilient high-reach <- 0 -> susceptible high-reach",
        y=0.19,
        fontsize=9.8,
    )
    fig.text(
        0.5,
        0.025,
        "Line is the HC3 fixed-effect coefficient; shaded wedge is coefficient uncertainty, not a prediction interval.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#6F768A",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.30, wspace=0.24)
    return _save_fig(fig, figures_dir / "alignment_gradient_fixed_effect_outcomes.png")


def _plot_branch_alignment_vs_network_effect(condition: pd.DataFrame, figures_dir: Path) -> Path:
    plot_df = condition.dropna(
        subset=["achieved_alignment_z", "mean_pn_increment_effectivity", "mean_ae_total_network"]
    ).copy()
    max_abs_x = max(float(plot_df["achieved_alignment_z"].abs().max()), 0.1)
    x_pad = max_abs_x * 0.15
    xlim = (-max_abs_x - x_pad, max_abs_x + x_pad)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-max_abs_x, vmax=max_abs_x)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True)
    panels = [
        (
            "mean_pn_increment_effectivity",
            "Mean post-network amplification\nmean((PN - P) x d)",
            "Primary network mechanism",
        ),
        (
            "mean_ae_total_network",
            "Mean final network attack effect\nmean((PN - B) x d)",
            "Secondary final endpoint",
        ),
    ]
    for ax, (outcome, ylabel, title) in zip(axes, panels):
        ax.axvspan(xlim[0], 0, color="#EAF1FE", alpha=0.55, zorder=0)
        ax.axvspan(0, xlim[1], color="#FFEDDE", alpha=0.42, zorder=0)
        ax.axvline(0, color="#1F2937", linewidth=0.9)
        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle=":")

        trend = plot_df[["achieved_alignment_z", outcome]].dropna()
        if trend.shape[0] >= 3 and trend["achieved_alignment_z"].nunique() > 1:
            slope, intercept = np.polyfit(trend["achieved_alignment_z"], trend[outcome], 1)
            xs = np.linspace(xlim[0], xlim[1], 80)
            ax.plot(xs, intercept + slope * xs, color="#374151", linewidth=1.15, zorder=2)

        ax.scatter(
            plot_df["achieved_alignment_z"],
            plot_df[outcome],
            s=82,
            c=plot_df["achieved_alignment_z"],
            cmap=cmap,
            norm=norm,
            edgecolor="#374151",
            linewidth=0.65,
            alpha=0.94,
            zorder=3,
        )
        ax.set_xlim(xlim)
        ax.set_xlabel("Sender-reach susceptibility alignment (z)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            "resilient\nhigh-reach",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#2E4780",
        )
        ax.text(
            0.98,
            0.98,
            "susceptible\nhigh-reach",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#804126",
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    cbar_ax = fig.add_axes([0.22, 0.12, 0.56, 0.026])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Sender alignment: resilient high-reach <- 0 -> susceptible high-reach",
        fontsize=8.5,
    )
    cbar.outline.set_visible(False)
    fig.suptitle("Branch Alignment-Gradient Mechanism Test", y=0.97, fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.26, wspace=0.22)
    return _save_fig(fig, figures_dir / "branch_alignment_vs_network_effect.png")


def _plot_level_means(level: pd.DataFrame, figures_dir: Path) -> Path:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharex=True)
    specs = [
        ("mean_pn_increment_effectivity", "se_pn_increment_effectivity", "PN increment effectivity"),
        ("mean_ae_total_network", "se_ae_total_network", "Total network effectivity"),
    ]
    for ax, (mean_col, se_col, title) in zip(axes, specs):
        ax.errorbar(
            level["target_alignment_z"],
            level[mean_col],
            yerr=level[se_col].fillna(0.0),
            marker="o",
            color="#2E4780",
            ecolor="#9CA3AF",
            linewidth=1.8,
            capsize=3,
        )
        ax.axhline(0, color="#6B7280", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Target alignment z")
        ax.set_ylabel("Mean score points")
        ax.set_title(title)
    fig.tight_layout()
    return _save_fig(fig, figures_dir / "alignment_level_marginal_means.png")


def _plot_condition_vulnerability_planes(sem: pd.DataFrame, condition: pd.DataFrame, figures_dir: Path) -> Path:
    plot_df = _condition_vulnerability_plane_frame(sem, condition)

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

    cmap = LinearSegmentedColormap.from_list(
        "susceptibility_blue_orange",
        ["#2E4780", "#FFFFFF", "#804126"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(
        len(opinion_order),
        len(attack_order),
        figsize=(18.8, 21.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_idx, opinion in enumerate(opinion_order):
        for col_idx, attack in enumerate(attack_order):
            ax = axes[row_idx][col_idx]
            group = plot_df[plot_df["opinion_label"].eq(opinion) & plot_df["attack_label"].eq(attack)].copy()
            ax.axhspan(0.80, 1.03, color="#F4F5F7", alpha=0.90, zorder=0)
            ax.axvline(0, color="#7A828F", linestyle=":", linewidth=0.75, zorder=1)
            ax.axhline(0.80, color="#7A828F", linestyle=":", linewidth=0.75, zorder=1)
            ax.set_xlim(-2.8, 2.8)
            ax.set_ylim(0, 1.03)
            ax.grid(True, color="#E6E8F0", linewidth=0.55, alpha=0.85)
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
                s=14,
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
                s=38,
                c=high["ae_private_z"],
                cmap=cmap,
                norm=norm,
                alpha=0.95,
                edgecolor="white",
                linewidth=0.42,
                zorder=3,
            )
            ax.scatter(
                top10["ae_private_z"],
                top10["sender_reach_percentile"],
                s=52,
                facecolors="none",
                edgecolors="#1F2430",
                linewidth=0.62,
                alpha=0.90,
                zorder=4,
            )
            achieved = float(group["achieved_alignment_z"].iloc[0])
            if row_idx == 0:
                ax.set_title(_label(attack), fontsize=8.1, color="#1F2937", pad=7)
            if col_idx == 0:
                ax.set_ylabel(_label(opinion), fontsize=8.1, color="#374151")
            else:
                ax.set_ylabel("")
            ax.text(
                0.04,
                0.05,
                _sender_alignment_label(achieved),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.3,
                color="#4B5563",
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#E6E8F0", "alpha": 0.88},
            )
            ax.tick_params(labelsize=6.5, length=2)
            if row_idx < len(opinion_order) - 1:
                ax.tick_params(labelbottom=False)
            if col_idx > 0:
                ax.tick_params(labelleft=False)
    fig.supxlabel("Private susceptibility within condition, z-scored AE_private", y=0.035, fontsize=9)
    fig.supylabel("Direct sender-reach percentile", x=0.018, fontsize=9)
    cbar_ax = fig.add_axes([0.915, 0.21, 0.012, 0.56])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("AE_private z within condition", fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(
        "Condition-Specific Vulnerability Planes",
        y=0.982,
        fontsize=13.5,
    )
    fig.subplots_adjust(left=0.07, right=0.895, top=0.945, bottom=0.065, hspace=0.18, wspace=0.10)
    return _save_fig(fig, figures_dir / "branch_condition_vulnerability_planes.png")


def _plot_representative_overlays(sem: pd.DataFrame, condition: pd.DataFrame, figures_dir: Path) -> Path:
    selected_conditions = (
        condition.sort_values(["target_alignment_z", "alignment_condition_id"])
        .groupby("target_alignment_z", as_index=False)
        .head(1)
    )
    selected_ids = set(selected_conditions["alignment_condition_id"])
    plot_sem = sem[sem["alignment_condition_id"].isin(selected_ids)].copy()
    plot_sem["exposure_position_id"] = plot_sem["exposure_position_id"].astype(str)
    positions = set(plot_sem["exposure_position_id"])
    edges = _load_edges(positions)

    graph = nx.Graph()
    graph.add_nodes_from(positions)
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.source_position_id), str(row.target_position_id), weight=float(row.exposure_weight))
    layout = nx.spring_layout(graph, seed=42, weight="weight", iterations=350, k=0.55)

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), squeeze=False)
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("coolwarm")

    for ax, row in zip(axes_flat, selected_conditions.itertuples(index=False)):
        group = plot_sem[plot_sem["alignment_condition_id"].eq(row.alignment_condition_id)].copy()
        group["ae_pct"] = pd.to_numeric(group["ae_private"], errors="coerce").rank(pct=True)
        group["sender_pct"] = pd.to_numeric(group["exposure_outgoing_visibility_weight"], errors="coerce").rank(pct=True)
        ax.axis("off")
        ax.set_aspect("equal")
        for edge in edges.itertuples(index=False):
            source = str(edge.source_position_id)
            target = str(edge.target_position_id)
            if source in layout and target in layout:
                xs = [layout[source][0], layout[target][0]]
                ys = [layout[source][1], layout[target][1]]
                ax.plot(xs, ys, color="#9CA3AF", alpha=0.13, linewidth=0.45, zorder=1)
        xs = group["exposure_position_id"].map(lambda pos: layout[str(pos)][0])
        ys = group["exposure_position_id"].map(lambda pos: layout[str(pos)][1])
        ax.scatter(
            xs,
            ys,
            s=24 + 125 * group["sender_pct"].fillna(0.5),
            c=group["ae_pct"].fillna(0.5),
            cmap=cmap,
            vmin=0,
            vmax=1,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        ax.set_title(_sender_alignment_label(row.achieved_alignment_z), fontsize=9)
    for ax in axes_flat[len(selected_conditions) :]:
        ax.axis("off")
    fig.suptitle("Representative counterfactual profile-position overlays by alignment level", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return _save_fig(fig, figures_dir / "representative_alignment_network_overlays.png")


def _plot_quality_gates(quality: pd.DataFrame, figures_dir: Path) -> Path:
    plot_df = quality.copy()
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    colors = ["#1B9E77" if status == "pass" else "#D95F02" for status in plot_df["status"]]
    ax.barh(plot_df["metric"], plot_df["value"], color=colors, alpha=0.86)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Branch quality gates")
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(float(row.value) + 0.5, idx, f"{row.value:g} ({row.status})", va="center", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, figures_dir / "alignment_gradient_quality_gates.png")


def _plot_robustness_results(robustness: pd.DataFrame, figures_dir: Path) -> Path:
    plot_df = robustness[
        robustness["model_id"].isin(
            [
                "main_fe_ols",
                "main_fe_hc3",
                "exclude_upstream_fallback_rows",
                "exclude_post_heuristic_warning_conditions",
            ]
        )
    ].copy()
    labels = {
        "main_fe_ols": "Main FE OLS",
        "main_fe_hc3": "Main FE HC3",
        "exclude_upstream_fallback_rows": "Exclude fallback rows",
        "exclude_post_heuristic_warning_conditions": "Exclude warning conditions",
    }
    outcomes = [
        ("mean_pn_increment_effectivity", "PN increment effectivity"),
        ("mean_ae_total_network", "Total network effect"),
    ]
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=False)
    for ax, (outcome, title) in zip(axes, outcomes):
        group = plot_df[plot_df["outcome"].eq(outcome)].copy()
        group["display_label"] = group["model_id"].map(labels)
        group = group.set_index("model_id").loc[list(labels)].reset_index()
        y = np.arange(group.shape[0])
        estimate = pd.to_numeric(group["estimate"], errors="coerce")
        se = pd.to_numeric(group["std_error"], errors="coerce")
        ax.errorbar(
            estimate,
            y,
            xerr=1.96 * se,
            fmt="o",
            color="#2E4780",
            ecolor="#9CA3AF",
            elinewidth=1.4,
            capsize=3,
        )
        ax.axvline(0, color="#6B7280", linestyle=":", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(group["display_label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Alignment coefficient with 95% interval")
        ax.set_title(title)
        for idx, row in enumerate(group.itertuples(index=False)):
            ax.text(
                float(row.estimate) + float(row.std_error) * 2.1,
                idx,
                f"p={_fmt_p(row.p_value)}",
                va="center",
                fontsize=7.5,
                color="#4B5563",
            )
    fig.suptitle("H3/H4 robustness checks for the achieved-alignment coefficient", y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_fig(fig, figures_dir / "alignment_gradient_robustness_coefficients.png")
