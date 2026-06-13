from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.backend.utils.generate_main_readme_figures import generate_main_readme_figures
from src.backend.utils.io import abs_path, ensure_dir, write_json, write_text
from src.backend.utils.scenario_realism import pretty_label


PALETTE = {
    "navy": "#14213d",
    "blue": "#1d4e89",
    "teal": "#1f7a8c",
    "coral": "#d95d39",
    "sand": "#f4e3b2",
    "ink": "#222222",
    "grid": "#d9dde6",
    "gold": "#c89b3c",
    "mint": "#2a9d8f",
}


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _pretty(value: str) -> str:
    text = value
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_", "abs_delta_indicator__"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("_z", "")
    text = text.replace("__", " ")
    return pretty_label(text).title()


def _wrap(value: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.edgecolor": PALETTE["navy"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, base_path: Path) -> List[str]:
    ensure_dir(base_path.parent)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [abs_path(png_path), abs_path(pdf_path)]


def _latex_escape_text(value: str) -> str:
    escaped = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        escaped = escaped.replace(old, new)
    return escaped


def _write_table_bundle(df: pd.DataFrame, base_path: Path, caption: str, note: str, label: str) -> List[str]:
    ensure_dir(base_path.parent)
    csv_path = base_path.with_suffix(".csv")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=False)

    resize_wide = len(df.columns) > 6
    column_format = None
    if len(df.columns) == 2:
        column_format = r"p{0.24\linewidth}p{0.70\linewidth}"
    elif "moderator" in base_path.stem and len(df.columns) >= 6:
        column_format = r"p{0.34\linewidth}" + "r" * (len(df.columns) - 1)
    elif "ontology_leaves_used" in base_path.stem:
        column_format = r"p{0.14\linewidth}p{0.22\linewidth}p{0.54\linewidth}"
        resize_wide = False

    table_latex = df.to_latex(index=False, escape=True, na_rep="", float_format=lambda x: f"{x:.3f}", column_format=column_format)
    body_lines = [r"\small", r"\setlength{\tabcolsep}{4.5pt}", r"\renewcommand{\arraystretch}{1.10}"]
    if resize_wide:
        body_lines.extend([r"\begin{adjustbox}{max width=\linewidth}", table_latex.rstrip(), r"\end{adjustbox}"])
    else:
        body_lines.append(table_latex.rstrip())

    tex_content = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        f"\\caption{{{_latex_escape_text(caption)}}}",
        f"\\label{{{label}}}",
        *body_lines,
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        f"\\item Note. {_latex_escape_text(note)}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    write_text(tex_path, tex_content)
    return [abs_path(csv_path), abs_path(tex_path)]


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> List[str]:
    ensure_dir(target_dir)
    copied: List[str] = []
    for item in source_dir.iterdir():
        target = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied.append(abs_path(target))
    return copied


def _coefficient_lookup(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    match = df.loc[df["term"] == column]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _draw_study_design(base_path: Path, config: Dict[str, Any]) -> List[str]:
    fig, ax = plt.subplots(figsize=(14.4, 6.5))
    ax.set_axis_off()
    n_profiles = int(config.get("n_profiles") or config.get("n_scenarios") or 0)
    attack_leaves_str = config.get("attack_leaves", config.get("attack_leaf", "")) or ""
    n_attacks = len([leaf for leaf in str(attack_leaves_str).split(",") if leaf.strip()]) or 1
    n_opinions = int(config.get("max_opinion_leaves") or 0)

    boxes = [
        (0.03, 0.66, 0.18, 0.18, f"PROFILE ontology\\nleaf sampling -> {n_profiles}\\ndiverse pseudoprofiles"),
        (0.03, 0.38, 0.18, 0.18, f"ATTACK ontology\n{n_attacks} selected leaves\nacross mechanism families"),
        (0.03, 0.10, 0.18, 0.18, f"OPINION ontology\n{n_opinions} sampled leaves\nacross policy domains"),
        (0.29, 0.46, 0.18, 0.24, f"Profile-panel manifest\n{n_profiles} profiles x {n_attacks} attacks\nx {n_opinions} opinion leaves"),
        (0.54, 0.63, 0.18, 0.18, "Baseline opinion\nassessment"),
        (0.54, 0.35, 0.18, 0.18, "Attack exposure\ngeneration + realism\naudit"),
        (0.54, 0.07, 0.18, 0.18, "Post-exposure opinion\nassessment + coherence\naudit"),
        (0.79, 0.46, 0.18, 0.24, "Profile-level effectivity\nabsolute attacked deltas\n+ latent SEM / robust OLS"),
    ]

    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.025",
                linewidth=1.5,
                edgecolor=PALETTE["navy"],
                facecolor="#f7fbff",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=PALETTE["navy"], fontweight="bold")

    arrows = [
        ((0.21, 0.75), (0.29, 0.58)),
        ((0.21, 0.47), (0.29, 0.58)),
        ((0.21, 0.19), (0.29, 0.58)),
        ((0.47, 0.58), (0.54, 0.72)),
        ((0.47, 0.58), (0.54, 0.44)),
        ((0.47, 0.58), (0.54, 0.16)),
        ((0.72, 0.72), (0.79, 0.58)),
        ((0.72, 0.44), (0.79, 0.58)),
        ((0.72, 0.16), (0.79, 0.58)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color=PALETTE["blue"]))

    design_label = f"Full factorial attacked-only profile-panel design ({n_attacks} attacks)" if n_attacks > 1 else "Attacked-only profile-panel design"
    ax.text(0.5, 0.95, design_label, ha="center", va="center", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    design_note = (
        f"Hierarchical ontologies are preserved upstream. Each profile is crossed with {n_attacks} attack leaves and {n_opinions} opinion leaves in a full factorial design."
        if n_attacks > 1
        else "Hierarchical ontologies are preserved upstream. Estimation uses repeated leaf nodes so the fixed attack leaf connects to multiple attacked opinion-shift indicators per profile."
    )
    ax.text(
        0.29,
        0.88,
        design_note,
        ha="left",
        va="center",
        fontsize=9.4,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_abs_delta_distribution(long_df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    sns.boxplot(data=long_df, x="opinion_leaf_label", y="abs_delta_score", color="#dce8f7", fliersize=0, ax=ax)
    sns.stripplot(data=long_df, x="opinion_leaf_label", y="abs_delta_score", color=PALETTE["navy"], alpha=0.65, jitter=0.18, size=4.5, ax=ax)
    ax.set_title("Absolute attacked opinion shift by repeated opinion leaf")
    ax.set_xlabel("Opinion leaf")
    ax.set_ylabel("Absolute post-baseline shift")
    ax.tick_params(axis="x", rotation=24)
    return _save_figure(fig, base_path)


def _draw_moderator_forest(weight_df: pd.DataFrame, base_path: Path) -> List[str]:
    work = weight_df.copy()
    for column in ["normalized_weight_pct", "estimate"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[
        work["normalized_weight_pct"].notna()
        & np.isfinite(work["normalized_weight_pct"])
    ].sort_values("normalized_weight_pct", ascending=True)
    if work.empty:
        return []
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    palette = sns.color_palette("blend:#1f4b99,#d96c06", n_colors=max(3, work["ontology_group"].nunique()))
    group_palette = {group: palette[idx] for idx, group in enumerate(work["ontology_group"].dropna().unique())}
    bar_colors = [group_palette.get(group, PALETTE["blue"]) for group in work["ontology_group"]]
    ax.barh(work["moderator_label"], work["normalized_weight_pct"], color=bar_colors, alpha=0.9)
    for y, (_, row) in enumerate(work.iterrows()):
        ax.text(
            row["normalized_weight_pct"] + 0.35,
            y,
            f"{row['normalized_weight_pct']:.1f}% | b={row['estimate']:.2f}",
            va="center",
            fontsize=8.8,
        )
    ax.set_title("Descriptive susceptibility weights across profile moderators")
    ax.set_xlabel("Normalized weight share (%)")
    ax.set_ylabel("Moderator")
    return _save_figure(fig, base_path)


def _draw_sem_diagram(
    sem_result: Dict[str, Any],
    exploratory_df: pd.DataFrame,
    config: Dict[str, Any],
    indicator_columns: List[str],
    base_path: Path,
) -> List[str]:
    sem_lookup = pd.DataFrame(sem_result.get("coefficients", []))
    if sem_lookup.empty:
        return []
    sem_lookup = sem_lookup.loc[sem_lookup["op"].astype(str) == "~"].copy()
    sem_lookup["estimate"] = pd.to_numeric(sem_lookup["estimate"], errors="coerce")
    sem_lookup["p_value"] = pd.to_numeric(sem_lookup["p_value"], errors="coerce")
    sem_lookup = sem_lookup.loc[
        sem_lookup["lhs"].isin(indicator_columns)
        & sem_lookup["estimate"].notna()
        & np.isfinite(sem_lookup["estimate"])
    ]
    if sem_lookup.empty:
        return []

    heatmap_df = sem_lookup.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")
    ordered_columns = (
        exploratory_df.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
        if "normalized_weight_pct" in exploratory_df.columns
        else list(heatmap_df.index)
    )
    ordered_columns = [column for column in ordered_columns if column in heatmap_df.index]
    heatmap_df = heatmap_df.loc[ordered_columns]
    heatmap_df = heatmap_df[[column for column in indicator_columns if column in heatmap_df.columns]]

    annot = heatmap_df.copy().astype(str)
    for rhs in heatmap_df.index:
        for lhs in heatmap_df.columns:
            row = sem_lookup.loc[(sem_lookup["lhs"] == lhs) & (sem_lookup["rhs"] == rhs)]
            if row.empty:
                annot.loc[rhs, lhs] = ""
                continue
            first = row.iloc[0]
            p_value = first["p_value"]
            stars = "***" if pd.notna(p_value) and p_value < 0.001 else "**" if pd.notna(p_value) and p_value < 0.01 else "*" if pd.notna(p_value) and p_value < 0.05 else "†" if pd.notna(p_value) and p_value < 0.10 else ""
            annot.loc[rhs, lhs] = f"{first['estimate']:.2f}{stars}"

    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.6,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar_kws={"label": "SEM path coefficient"},
        ax=ax,
    )
    ax.set_title("Path-SEM coefficients from profile moderators to attacked opinion shifts")
    ax.set_xlabel("Repeated attacked opinion outcome")
    ax.set_ylabel("Profile moderator")
    ax.set_yticklabels([_wrap(_pretty(index), 24) for index in heatmap_df.index], rotation=0)
    ax.set_xticklabels([_wrap(_pretty(column), 18) for column in heatmap_df.columns], rotation=15, ha="right")
    fig.text(
        0.01,
        0.01,
        "Note. Cells show path coefficients from the repeated-outcome SEM. Stars mark p < .05; dagger marks p < .10. Opinion indicators are averaged across attack vectors.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_baseline_post_scatter(long_df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    n_points = len(long_df)
    marker_size = 46 if n_points < 500 else 28
    alpha = 0.85 if n_points < 500 else 0.55
    sns.scatterplot(data=long_df, x="baseline_score", y="post_score", hue="opinion_leaf_label", palette="Set2", s=marker_size, alpha=alpha, ax=ax)
    min_axis = float(min(long_df["baseline_score"].min(), long_df["post_score"].min()))
    max_axis = float(max(long_df["baseline_score"].max(), long_df["post_score"].max()))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="#666")
    ax.set_title("Baseline versus post-attack opinion scores")
    ax.set_xlabel("Baseline score")
    ax.set_ylabel("Post-attack score")
    return _save_figure(fig, base_path)


def _draw_profile_heatmap(long_df: pd.DataFrame, profile_index_df: pd.DataFrame, base_path: Path) -> List[str]:
    matrix = long_df.pivot_table(index="profile_id", columns="opinion_leaf_label", values="abs_delta_score", aggfunc="mean")
    if not profile_index_df.empty:
        ordered_ids = [profile_id for profile_id in profile_index_df["profile_id"].tolist() if profile_id in matrix.index]
        matrix = matrix.loc[ordered_ids]
    fig, ax = plt.subplots(figsize=(11.2, max(6.2, 0.22 * len(matrix.index) + 2.2)))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=0.3, linecolor="white", cbar_kws={"label": "|Delta|"}, ax=ax)
    ax.set_title("Per-profile attack effectivity heatmap")
    ax.set_xlabel("Opinion leaf")
    ax.set_ylabel("Profile")
    return _save_figure(fig, base_path)


def _draw_susceptibility_distribution(profile_index_df: pd.DataFrame, base_path: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    sns.histplot(profile_index_df["susceptibility_index_pct"], bins=12, color=PALETTE["coral"], edgecolor="white", ax=ax)
    ax.set_title("Distribution of post hoc empirical susceptibility index")
    ax.set_xlabel("Susceptibility index percentile")
    ax.set_ylabel("Profiles")
    return _save_figure(fig, base_path)


def _draw_attack_comparison(long_df: pd.DataFrame, base_path: Path) -> List[str]:
    work = long_df.copy()
    if "attack_leaf_label" not in work.columns and "attack_leaf" in work.columns:
        work["attack_leaf_label"] = work["attack_leaf"].apply(lambda x: x.rsplit(" > ", 1)[-1] if isinstance(x, str) and " > " in x else x)
    if "attack_leaf_label" not in work.columns:
        return []
    n_attacks = work["attack_leaf_label"].nunique()
    if n_attacks < 2:
        return []
    outcome = "adversarial_effectivity" if "adversarial_effectivity" in work.columns else "abs_delta_score"
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    order = work.groupby("attack_leaf_label")[outcome].mean().sort_values(ascending=False).index.tolist()
    sns.violinplot(data=work, x="attack_leaf_label", y=outcome, order=order, inner="box", palette="Set2", ax=ax)
    ax.set_title(f"Attack-vector comparison: {_pretty(outcome)} by attack mechanism")
    ax.set_xlabel("Attack vector")
    ax.set_ylabel(_pretty(outcome))
    ax.tick_params(axis="x", rotation=18)
    fig.text(
        0.01, 0.01,
        f"Note. Violins show the distribution of {_pretty(outcome)} across all profiles and opinion leaves for each attack mechanism. Inner box shows IQR and median.",
        ha="left", va="bottom", fontsize=9.5, color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _leaf_display(value: str) -> str:
    raw = str(value).rsplit(">", 1)[-1].strip() if ">" in str(value) else str(value)
    return pretty_label(raw)


def _draw_task_reliability_surface(task_summary_df: pd.DataFrame, base_path: Path) -> List[str]:
    if task_summary_df.empty:
        return []
    work = task_summary_df.copy()
    work["attack_label"] = work["attack_leaf"].map(_leaf_display)
    work["opinion_label"] = work["opinion_leaf"].map(_leaf_display)
    reliability = work.pivot_table(index="attack_label", columns="opinion_label", values="reliability_weight", aggfunc="mean")
    cv_mse = work.pivot_table(index="attack_label", columns="opinion_label", values="cv_mse", aggfunc="mean")
    if reliability.empty or cv_mse.empty:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.4), gridspec_kw={"width_ratios": [1.0, 1.0]})
    sns.heatmap(
        reliability,
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Reliability weight"},
        ax=axes[0],
    )
    axes[0].set_title("Task reliability weights")
    axes[0].set_xlabel("Opinion leaf")
    axes[0].set_ylabel("Attack vector")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cv_mse,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Cross-validated MSE"},
        ax=axes[1],
    )
    axes[1].set_title("Task-specific predictive error")
    axes[1].set_xlabel("Opinion leaf")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].tick_params(axis="y", rotation=0)

    fig.suptitle("Conditional susceptibility task reliability across the 4 x 10 attack-opinion surface", fontsize=14, fontweight="bold", color=PALETTE["navy"], y=0.98)
    fig.text(
        0.01,
        0.01,
        "Note. Reliability weights are proportional to n / CV-MSE for each task-specific ridge model. Narrow weight dispersion indicates that no single attack-opinion cell dominates conditional susceptibility ranking.",
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=PALETTE["ink"],
    )
    return _save_figure(fig, base_path)


def _draw_network_summary(
    centrality_df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    base_path: Path,
) -> List[str]:
    if centrality_df.empty:
        return []

    work = centrality_df.copy()
    if "ontology_family" not in work.columns and "ontology_group" in work.columns:
        work["ontology_family"] = work["ontology_group"].astype(str).str.split(":").str[0]
    if "feature_type" not in work.columns:
        if "is_categorical" in work.columns:
            work["feature_type"] = work["is_categorical"].apply(lambda x: "categorical" if x else "continuous")
        else:
            work["feature_type"] = "continuous"
    hub_metric = "strength" if "strength" in work.columns else "eigenvector_centrality"
    bridge_metric = "participation_coefficient" if "participation_coefficient" in work.columns else "betweenness_centrality"
    hubs = work.sort_values(hub_metric, ascending=False).head(10).iloc[::-1]
    bridges = work.sort_values(bridge_metric, ascending=False).head(10).iloc[::-1]
    if hubs.empty or bridges.empty:
        return []

    families = pd.concat([hubs["ontology_family"], bridges["ontology_family"]], axis=0).dropna().unique().tolist()
    palette = sns.color_palette("blend:#1d4e89,#2a9d8f,#e76f51", n_colors=max(3, len(families)))
    family_colors = {family: palette[idx] for idx, family in enumerate(families)}

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.6), gridspec_kw={"width_ratios": [1.0, 1.0]})
    axes[0].barh(
        [_wrap(label, 28) for label in hubs["label"]],
        hubs[hub_metric],
        color=[family_colors.get(family, PALETTE["blue"]) for family in hubs["ontology_family"]],
        alpha=0.92,
    )
    axes[0].set_title(f"Top hub variables by {_pretty(hub_metric)}")
    axes[0].set_xlabel(_pretty(hub_metric))
    axes[0].set_ylabel("")

    axes[1].barh(
        [_wrap(label, 28) for label in bridges["label"]],
        bridges[bridge_metric],
        color=[family_colors.get(family, PALETTE["teal"]) for family in bridges["ontology_family"]],
        alpha=0.92,
    )
    axes[1].set_title(f"Top bridge variables by {_pretty(bridge_metric)}")
    axes[1].set_xlabel(_pretty(bridge_metric))
    axes[1].set_ylabel("")

    legend_handles = [
        plt.Line2D([0], [0], color=family_colors[family], lw=8, label=family)
        for family in families
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    gm_lines = [
        f"Nodes = {int(global_metrics.get('n_nodes', len(work)))}",
        f"Edges = {int(global_metrics.get('n_edges', 0))}",
        f"Density = {float(global_metrics.get('density', 0.0)):.3f}",
        f"Communities = {int(global_metrics.get('n_communities', 0))}",
        f"Modularity = {float(global_metrics.get('modularity_score', 0.0)):.3f}",
        f"Cross-family share = {float(global_metrics.get('between_family_edge_share', 0.0)) * 100:.1f}%",
    ]
    fig.text(
        0.50,
        0.04,
        " | ".join(gm_lines),
        ha="center",
        va="center",
        fontsize=9.4,
        color=PALETTE["ink"],
    )
    fig.suptitle("Profile feature network: local hubs versus cross-community bridges", fontsize=14, fontweight="bold", color=PALETTE["navy"], y=0.98)
    return _save_figure(fig, base_path)


def _ontology_table(ontology_catalog: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    selected_attacks = ontology_catalog.get("selected_attack_leaves", [])
    if selected_attacks:
        for leaf in selected_attacks:
            rows.append({"Ontology": "ATTACK", "Role": "Factorial attack leaf", "Leaf": leaf})
    else:
        selected_attack = ontology_catalog.get("selected_attack_leaf")
        if selected_attack:
            rows.append({"Ontology": "ATTACK", "Role": "Single attack leaf", "Leaf": selected_attack})
    for leaf in ontology_catalog.get("selected_opinion_leaves", []):
        rows.append({"Ontology": "OPINION", "Role": "Repeated indicator leaf", "Leaf": leaf})
    return pd.DataFrame(rows)


def generate_publication_assets(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    bootstrap_params_csv_path: str | Path,
    exploratory_comparison_csv_path: str | Path,
    config_json_path: str | Path,
    ontology_catalog_path: str | Path,
    assumptions_json_path: str | Path,
    critiques_json_path: str | Path,
    output_dir: str | Path,
    report_assets_root: str | Path,
    run_id: str,
    paper_title: str,
) -> Dict[str, Any]:
    _setup_theme()
    output_root = ensure_dir(output_dir)
    figures_dir = ensure_dir(Path(output_root) / "figures")
    tables_dir = ensure_dir(Path(output_root) / "tables")
    snapshots_dir = ensure_dir(Path(output_root) / "snapshots")
    report_assets_root = ensure_dir(report_assets_root)
    report_figures_dir = ensure_dir(Path(report_assets_root) / "figures")
    report_tables_dir = ensure_dir(Path(report_assets_root) / "tables")

    long_df = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)
    bootstrap_params = pd.read_csv(bootstrap_params_csv_path)
    exploratory_df = pd.read_csv(exploratory_comparison_csv_path)
    config = json.loads(Path(config_json_path).read_text(encoding="utf-8"))
    ontology_catalog = json.loads(Path(ontology_catalog_path).read_text(encoding="utf-8"))
    assumptions = json.loads(Path(assumptions_json_path).read_text(encoding="utf-8"))
    critiques = json.loads(Path(critiques_json_path).read_text(encoding="utf-8"))

    stage05_dir = Path(sem_long_csv_path).resolve().parent
    stage06_dir = Path(sem_result_json_path).resolve().parent
    profile_summary_df = pd.read_csv(stage05_dir / "profile_level_effectivity.csv")
    profile_wide_df = pd.read_csv(stage05_dir / "profile_sem_wide.csv")
    profile_index_df = pd.read_csv(stage06_dir / "profile_susceptibility_index.csv")
    weight_table_path = stage06_dir / "moderator_weight_table.csv"
    weight_df = pd.read_csv(weight_table_path) if weight_table_path.exists() else pd.DataFrame()
    task_summary_path = stage06_dir / "conditional_susceptibility_task_summary.csv"
    task_summary_df = pd.read_csv(task_summary_path) if task_summary_path.exists() else pd.DataFrame()
    network_centrality_path = stage06_dir / "profile_network_centrality.csv"
    network_centrality_df = pd.read_csv(network_centrality_path) if network_centrality_path.exists() else pd.DataFrame()
    network_global_path = stage06_dir / "profile_network_global_metrics.json"
    network_global_metrics = json.loads(network_global_path.read_text(encoding="utf-8")) if network_global_path.exists() else {}
    quality_diagnostics_path = stage06_dir / "analysis_quality_diagnostics.json"
    quality_diagnostics = json.loads(quality_diagnostics_path.read_text(encoding="utf-8")) if quality_diagnostics_path.exists() else {}
    icc_path = stage06_dir / "intraclass_correlation.json"
    icc_payload = json.loads(icc_path.read_text(encoding="utf-8")) if icc_path.exists() else {}
    ridge_summary_path = stage06_dir / "ridge_full_summary.json"
    ridge_summary = json.loads(ridge_summary_path.read_text(encoding="utf-8")) if ridge_summary_path.exists() else {}
    rf_summary_path = stage06_dir / "rf_summary.json"
    rf_summary = json.loads(rf_summary_path.read_text(encoding="utf-8")) if rf_summary_path.exists() else {}
    enet_summary_path = stage06_dir / "elastic_net_summary.json"
    enet_summary = json.loads(enet_summary_path.read_text(encoding="utf-8")) if enet_summary_path.exists() else {}

    figure_files: List[str] = []
    table_files: List[str] = []
    snapshot_files: List[str] = []

    indicator_columns = [column for column in profile_wide_df.columns if column.startswith("abs_delta_indicator__") and not column.endswith("_z")]

    figure_files.extend(_draw_study_design(figures_dir / "figure_1_study_design", config))
    figure_files.extend(_draw_task_reliability_surface(task_summary_df, figures_dir / "figure_2_task_reliability_surface"))
    figure_files.extend(_draw_network_summary(network_centrality_df, network_global_metrics, figures_dir / "figure_3_profile_network_bridge_summary"))
    figure_files.extend(_draw_baseline_post_scatter(long_df, figures_dir / "supplementary_figure_s1_baseline_post_scatter"))
    figure_files.extend(_draw_profile_heatmap(long_df, profile_index_df, figures_dir / "supplementary_figure_s2_profile_effectivity_heatmap"))
    figure_files.extend(_draw_susceptibility_distribution(profile_index_df, figures_dir / "supplementary_figure_s3_susceptibility_distribution"))
    figure_files.extend(_draw_attack_comparison(long_df, figures_dir / "supplementary_figure_s4_attack_comparison_panel"))
    if not weight_df.empty:
        figure_files.extend(_draw_moderator_forest(weight_df, figures_dir / "supplementary_figure_s5_moderator_weight_forest"))
    figure_files.extend(_draw_sem_diagram(sem_result, exploratory_df, config, indicator_columns, figures_dir / "supplementary_figure_s6_sem_path_diagram"))
    figure_files.extend(
        generate_main_readme_figures(
            stage05_dir=stage05_dir,
            stage06_dir=stage06_dir,
            output_dirs=[figures_dir],
            config=config,
            ontology_catalog=ontology_catalog,
        )
    )

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    if not sem_coeff_df.empty:
        sem_coeff_df.to_csv(snapshots_dir / "sem_coefficients_snapshot.csv", index=False)
        snapshot_files.append(abs_path(snapshots_dir / "sem_coefficients_snapshot.csv"))

    attack_leaves_str = config.get("attack_leaves", config.get("attack_leaf", ""))
    n_attack_leaves = len(attack_leaves_str.split(",")) if attack_leaves_str and "," in attack_leaves_str else 1
    config_table = pd.DataFrame(
        [
            {"Parameter": "Run ID", "Value": run_id},
            {"Parameter": "Paper title", "Value": paper_title},
            {"Parameter": "Profiles", "Value": config.get("n_profiles") or profile_summary_df["profile_id"].nunique()},
            {"Parameter": "Attacked rows", "Value": len(long_df)},
            {"Parameter": "Attack leaves", "Value": f"{n_attack_leaves} ({attack_leaves_str})"},
            {"Parameter": "Opinion domain", "Value": config.get("focus_opinion_domain") or "Multiple"},
            {"Parameter": "Repeated opinion leaves", "Value": long_df["opinion_leaf_label"].nunique()},
            {"Parameter": "Design", "Value": f"Full factorial ({config.get('n_profiles', '?')} x {n_attack_leaves} x {long_df['opinion_leaf_label'].nunique()})" if n_attack_leaves > 1 else "Profile-panel"},
            {"Parameter": "Model", "Value": config.get("openrouter_model")},
        ]
    )
    design_note = (
        f"Full factorial design: each profile is crossed with {n_attack_leaves} attack leaves and {long_df['opinion_leaf_label'].nunique()} opinion leaves. All rows are attacked; there is no no-attack control condition."
        if n_attack_leaves > 1
        else "One fixed ATTACK leaf is applied across repeated OPINION leaves for each profile. All rows are attacked."
    )
    table_files.extend(
        _write_table_bundle(
            config_table,
            tables_dir / "supplementary_table_s1_run_configuration",
            "Study design and configuration for the attacked-only profile-panel study.",
            design_note,
            "tab:design",
        )
    )

    agg_dict: Dict[str, Any] = {
        "n_rows": ("scenario_id", "count"),
        "mean_baseline": ("baseline_score", "mean"),
        "mean_post": ("post_score", "mean"),
        "mean_signed_delta": ("delta_score", "mean"),
        "mean_abs_delta": ("abs_delta_score", "mean"),
        "sd_abs_delta": ("abs_delta_score", lambda s: float(s.std(ddof=0))),
    }
    if "adversarial_effectivity" in long_df.columns:
        agg_dict["mean_adv_eff"] = ("adversarial_effectivity", "mean")
        agg_dict["sd_adv_eff"] = ("adversarial_effectivity", lambda s: float(s.std(ddof=0)))
    descriptive_table = (
        long_df.groupby("opinion_leaf_label", as_index=False)
        .agg(**agg_dict)
        .sort_values("mean_abs_delta", ascending=False)
    )
    desc_note = (
        f"Descriptive statistics aggregated across {n_attack_leaves} attack vectors per opinion leaf. Adversarial effectivity weights shift by the per-leaf adversarial goal direction."
        if n_attack_leaves > 1
        else "Absolute shift is the primary effectivity metric because the same fixed attack leaf is linked to multiple opinion deltas that can move in different signed directions."
    )
    table_files.extend(
        _write_table_bundle(
            descriptive_table,
            tables_dir / "supplementary_table_s2_condition_descriptive_statistics",
            "Attacked effectivity descriptive statistics by repeated opinion leaf.",
            desc_note,
            "tab:descriptives",
        )
    )

    sem_path_df = pd.DataFrame(sem_result.get("coefficients", []))
    sem_path_df = sem_path_df.loc[sem_path_df["op"].astype(str) == "~"].copy() if not sem_path_df.empty else pd.DataFrame()
    if not sem_path_df.empty:
        sem_path_df["estimate"] = pd.to_numeric(sem_path_df["estimate"], errors="coerce")
        sem_path_df["p_value"] = pd.to_numeric(sem_path_df["p_value"], errors="coerce")
        sem_summary = (
            sem_path_df.groupby("rhs", as_index=False)
            .agg(
                mean_sem_b=("estimate", "mean"),
                mean_abs_sem_b=("estimate", lambda s: float(np.mean(np.abs(s)))),
                min_sem_p=("p_value", "min"),
                n_sem_paths=("lhs", "count"),
            )
            .rename(columns={"rhs": "Moderator"})
        )
    else:
        sem_summary = pd.DataFrame(columns=["Moderator", "mean_sem_b", "mean_abs_sem_b", "min_sem_p", "n_sem_paths"])

    model_table = exploratory_df.rename(
        columns={
            "moderator_label": "Moderator",
            "univariate_estimate": "Controlled mean b",
            "univariate_p_value": "Controlled p",
            "ridge_mean_estimate": "Ridge mean b",
            "normalized_weight_pct": "Weight %",
            "role": "Role",
        }
    )[["Moderator", "Role", "Controlled mean b", "Controlled p", "Ridge mean b", "Weight %"]]
    model_table = model_table.merge(sem_summary, on="Moderator", how="left")
    model_table = model_table.rename(
        columns={
            "mean_sem_b": "SEM mean b",
            "mean_abs_sem_b": "SEM mean |b|",
            "min_sem_p": "SEM min p",
            "n_sem_paths": "SEM paths",
        }
    )
    table_files.extend(
        _write_table_bundle(
            model_table,
            tables_dir / "supplementary_table_s3_moderator_model_summary",
            "Profile moderators of attacked effectivity: controlled contrasts, SEM paths, and descriptive susceptibility weights.",
            "Controlled coefficients summarize moderator associations with mean attacked effectivity. SEM columns summarize the repeated-outcome path model across the attacked opinion leaves. Weight percentages come from the target-conditional regularized aggregation used to compute the post hoc susceptibility index.",
            "tab:multivariate",
        )
    )

    ontology_note = (
        f"Only ontology leaves are sampled for estimation. The ATTACK ontology contributes {n_attack_leaves} factorial leaves; the OPINION ontology contributes repeated indicator leaves."
        if n_attack_leaves > 1
        else "Only ontology leaves are sampled for estimation. The ATTACK ontology contributes one fixed leaf; the OPINION ontology contributes repeated indicator leaves."
    )
    table_files.extend(
        _write_table_bundle(
            _ontology_table(ontology_catalog),
            tables_dir / "supplementary_table_s4_ontology_leaves_used",
            f"Ontology leaves used in {run_id}.",
            ontology_note,
            "tab:ontologies",
        )
    )

    moderator_table = exploratory_df.rename(
        columns={
            "moderator_label": "Moderator",
            "multivariate_estimate": "Multivariate b",
            "multivariate_p_value": "Multivariate p",
            "univariate_estimate": "Univariate b",
            "univariate_p_value": "Univariate p",
            "role": "Role",
        }
    )[["Moderator", "Role", "Multivariate b", "Multivariate p", "Univariate b", "Univariate p"]]
    if not weight_df.empty:
        moderator_table = moderator_table.merge(
            weight_df.rename(
                columns={
                    "moderator_label": "Moderator",
                    "ontology_group": "Ontology group",
                    "normalized_weight_pct": "Normalized weight %",
                }
            )[["Moderator", "Ontology group", "Normalized weight %"]],
            on="Moderator",
            how="left",
        )
    table_files.extend(
        _write_table_bundle(
            moderator_table,
            tables_dir / "supplementary_table_s5_moderator_comparison",
            "Core and exploratory profile moderator comparison.",
            "Core terms are entered into the latent SEM or primary multivariate profile model. Normalized weight percentages summarize each moderator's share of the total fitted importance after accounting for the moderator's observed variability.",
            "tab:moderators",
        )
    )

    if not network_centrality_df.empty:
        network_table = network_centrality_df.copy()
        if "ontology_family" not in network_table.columns and "ontology_group" in network_table.columns:
            network_table["ontology_family"] = network_table["ontology_group"].astype(str).str.split(":").str[0]
        _sort_cols = [c for c in ["participation_coefficient", "strength", "betweenness_centrality"] if c in network_table.columns]
        if _sort_cols:
            network_table = network_table.sort_values(_sort_cols, ascending=[False] * len(_sort_cols))
        _rename_map = {
            "label": "Feature",
            "ontology_group": "Ontology group",
            "ontology_family": "Ontology family",
            "feature_type": "Feature type",
            "community": "Community",
            "strength": "Strength",
            "participation_coefficient": "Participation coefficient",
            "bridge_ratio": "Bridge ratio",
            "within_module_zscore": "Within-module Z",
            "k_core": "K-core",
        }
        _keep_cols_ordered = [
            "Feature", "Ontology family", "Ontology group", "Feature type",
            "Community", "Strength", "Participation coefficient",
            "Bridge ratio", "Within-module Z", "K-core",
        ]
        network_table = network_table.rename(columns=_rename_map)
        _available_cols = [c for c in _keep_cols_ordered if c in network_table.columns]
        network_summary_table = network_table[_available_cols].head(20)
        table_files.extend(
            _write_table_bundle(
                network_summary_table,
                tables_dir / "supplementary_table_s6_network_topology_summary",
                "Top network hubs and bridge variables in the profile feature correlation graph.",
                "Rows are ordered primarily by participation coefficient, then by weighted degree. Bridge-oriented metrics distinguish cross-community connectors from within-community hubs in the mixed continuous-plus-dummy feature panel.",
                "tab:networksummary",
            )
        )

    quality_rows = []
    if quality_diagnostics:
        quality_rows.extend(
            [
                {"Diagnostic": "Baseline fallback rate", "Value": float(quality_diagnostics.get("baseline_fallback_used_rate", 0.0)) * 100, "Scale": "%"},
                {"Diagnostic": "Post fallback rate", "Value": float(quality_diagnostics.get("post_fallback_used_rate", 0.0)) * 100, "Scale": "%"},
                {"Diagnostic": "Attack heuristic pass rate", "Value": float(quality_diagnostics.get("attack_heuristic_pass_rate", 0.0)) * 100, "Scale": "%"},
                {"Diagnostic": "Post heuristic pass rate", "Value": float(quality_diagnostics.get("post_heuristic_pass_rate", 0.0)) * 100, "Scale": "%"},
            ]
        )
    if icc_payload:
        quality_rows.append({"Diagnostic": "ICC(1) |Delta|", "Value": float(icc_payload.get("abs_delta_score", {}).get("icc1", 0.0)), "Scale": "ratio"})
    if ridge_summary:
        quality_rows.append({"Diagnostic": "Ridge CV-R^2", "Value": float(ridge_summary.get("cv_r2", 0.0)), "Scale": "R^2"})
    if rf_summary:
        quality_rows.append({"Diagnostic": "Random forest OOB R^2", "Value": float(rf_summary.get("oob_r2", 0.0)), "Scale": "R^2"})
    if enet_summary:
        quality_rows.append({"Diagnostic": "Elastic-net selected features", "Value": int(enet_summary.get("n_features_selected", 0)), "Scale": "count"})
    if quality_rows:
        table_files.extend(
            _write_table_bundle(
                pd.DataFrame(quality_rows),
                tables_dir / "supplementary_table_s7_run_quality_diagnostics",
                "Run_10 execution-quality diagnostics and model robustness indicators.",
                "These diagnostics are intended to prevent overinterpretation. High fallback rates or near-zero predictive fit imply that downstream susceptibility coefficients are methodological diagnostics rather than substantive evidence about human-like persuasion dynamics.",
                "tab:qualitydiag",
            )
        )

    if not sem_path_df.empty:
        sem_table = sem_path_df.rename(
            columns={
                "lhs": "Outcome leaf",
                "rhs": "Moderator",
                "estimate": "SEM b",
                "std_error": "SE",
                "p_value": "p",
            }
        )[["Outcome leaf", "Moderator", "SEM b", "SE", "p"]]
        sem_table["Outcome leaf"] = sem_table["Outcome leaf"].map(_pretty)
        table_files.extend(
            _write_table_bundle(
                sem_table,
                tables_dir / "supplementary_table_s10_sem_path_coefficients",
                "Leaf-specific path-SEM coefficients.",
                "Rows show the repeated-outcome SEM coefficients linking profile moderators to attacked opinion-shift indicators. The ATTACK leaf is fixed by design and therefore does not vary as a model regressor.",
                "tab:sempaths",
            )
        )

    risk_rows = []
    for item in assumptions:
        risk_rows.append({"Type": "Assumption", "Item": item["assumption"], "Status": item["status"], "Mitigation": item["mitigation"]})
    for item in critiques:
        risk_rows.append({"Type": "Critique", "Item": item["critique"], "Status": "addressed", "Mitigation": item["implemented_change"]})
    table_files.extend(
        _write_table_bundle(
            pd.DataFrame(risk_rows),
            tables_dir / "supplementary_table_s8_assumption_and_risk_register",
            "Assumption register and peer-review risk register.",
            "This study is methodological and exploratory. Risks are surfaced explicitly to support transparency and future replication.",
            "tab:risks",
        )
    )

    reproducibility_rows = pd.DataFrame(
        [{"Field": key, "Value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in config.items()]
    )
    table_files.extend(
        _write_table_bundle(
            reproducibility_rows,
            tables_dir / "supplementary_table_s9_reproducibility_manifest",
            f"Reproducibility manifest for {run_id}.",
            "The manifest captures the full pipeline configuration used to generate the study outputs and manuscript assets.",
            "tab:repro",
        )
    )

    long_df.to_csv(snapshots_dir / "sem_long_encoded_snapshot.csv", index=False)
    profile_summary_df.to_csv(snapshots_dir / "profile_level_effectivity_snapshot.csv", index=False)
    profile_index_df.to_csv(snapshots_dir / "profile_susceptibility_index_snapshot.csv", index=False)
    if not weight_df.empty:
        weight_df.to_csv(snapshots_dir / "moderator_weight_table_snapshot.csv", index=False)
    snapshot_files.extend(
        [
            abs_path(snapshots_dir / "sem_long_encoded_snapshot.csv"),
            abs_path(snapshots_dir / "profile_level_effectivity_snapshot.csv"),
            abs_path(snapshots_dir / "profile_susceptibility_index_snapshot.csv"),
        ]
    )
    if not weight_df.empty:
        snapshot_files.append(abs_path(snapshots_dir / "moderator_weight_table_snapshot.csv"))

    copied_figures = _copy_tree_contents(figures_dir, report_figures_dir)
    copied_tables = _copy_tree_contents(tables_dir, report_tables_dir)

    manifest_payload = {
        "run_id": run_id,
        "paper_title": paper_title,
        "figure_count": len(figure_files),
        "table_count": len(table_files),
        "copied_figures": copied_figures,
        "copied_tables": copied_tables,
    }
    manifest_path = Path(output_root) / "publication_assets_manifest.json"
    write_json(manifest_path, manifest_payload)

    return {
        "manifest_path": abs_path(manifest_path),
        "visual_files": figure_files,
        "table_files": table_files,
        "snapshot_files": snapshot_files,
        "copied_figures": copied_figures,
        "copied_tables": copied_tables,
    }
