"""
Generate the two root README / paper matrix figures with ontology-aware dendrograms.

Key properties:
- no in-figure titles
- top / right dendrograms follow ontology or feature hierarchy, not incidental value clustering
- attack / opinion ordering respects the active ontology selection for test, deployment, or custom runs
- profile dummy variables remain grouped under their parent categorical dimension
- command-line overrides are available for custom control outside the stage-08 pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.utils.analysis.susceptibility_scoring.feature_registry import (
    FeatureRegistry,
)
from src.backend.utils.ontology_utils import (
    flatten_leaf_paths,
    leaf_to_key,
    load_ontology_triplet,
)
DEFAULT_STAGE05 = ROOT / "evaluation" / "run_1" / "stage_outputs" / "05_compute_effectivity_deltas"
DEFAULT_STAGE06 = ROOT / "evaluation" / "run_1" / "stage_outputs" / "06_construct_structural_equation_model"
DEFAULT_CONFIG = ROOT / "evaluation" / "run_1" / "config" / "pipeline_config.json"
DEFAULT_ONTOLOGY_CATALOG = (
    ROOT / "evaluation" / "run_1" / "stage_outputs" / "01_create_scenarios" / "ontology_leaf_catalog.json"
)

DEFAULT_OUTPUT_DIRS = [
    ROOT / "research_report" / "assets" / "figures",
    ROOT / "evaluation" / "run_1" / "publication_assets" / "figures",
    ROOT / "evaluation" / "run_1" / "paper" / "publication_assets" / "figures",
    ROOT / "evaluation" / "run_1" / "stage_outputs" / "08_generate_publication_assets" / "figures",
]

WHITE = "#ffffff"
NAVY = "#14213d"
INK = "#222222"
DENDROGRAM_COLOR = "#8d8d8d"

CANONICAL_CORE_PREDICTORS: list[tuple[tuple[str, ...], str]] = [
    (("profile_cont_chronological_age", "profile_cont_age_years"), "Age"),
    (("profile_cont_big_five_conscientiousness_mean_pct",), "Conscientiousness"),
    (("profile_cont_big_five_neuroticism_mean_pct",), "Neuroticism"),
    (("profile_cont_big_five_openness_to_experience_mean_pct",), "Openness"),
    (("profile_cont_big_five_agreeableness_mean_pct",), "Agreeableness"),
    (("profile_cont_big_five_extraversion_mean_pct",), "Extraversion"),
    (("profile_cat__profile_cat_sex_Female",), "Sex: Female"),
    (("profile_cat__profile_cat_sex_Other",), "Sex: Other"),
]


@dataclass(frozen=True)
class PredictorSpec:
    column: str
    label: str
    hierarchy_path: tuple[str, ...]
    score: float = 0.0


def _setup() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.edgecolor": NAVY,
            "axes.labelcolor": INK,
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _normalize_output_dirs(output_dirs: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(output_dirs, (str, Path)):
        return [Path(output_dirs)]
    return [Path(path) for path in output_dirs]


def _save(fig: plt.Figure, stem: str, output_dirs: Path | str | Sequence[Path | str]) -> list[str]:
    saved: list[str] = []
    for out_dir in _normalize_output_dirs(output_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in ("png", "pdf"):
            path = out_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
            saved.append(str(path.resolve()))
            print(f"  saved {path}")
    plt.close(fig)
    return saved


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered


def _split_path(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(">") if part.strip())


def _last_leaf(value: str) -> str:
    parts = _split_path(value)
    return parts[-1] if parts else str(value).strip()


def _display_text(value: str) -> str:
    words = _last_leaf(value).replace("_", " ").split()
    if not words:
        return str(value)
    text = " ".join(word if word.isupper() else word.capitalize() for word in words)
    replacements = {
        "Llm": "LLM",
        "Nato": "NATO",
        "Eu": "EU",
        "Us": "US",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _wrap_label(value: str, width: int = 14) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _cluster_label_from_value(value: str, width: int) -> str:
    return _wrap_label(_display_text(value), width=width)


def _load_json_if_exists(path: str | Path | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _load_ontologies(config: dict | None, ontology_catalog: dict | None) -> dict | None:
    ontology_root = None
    if config and config.get("ontology_root"):
        ontology_root = config["ontology_root"]
    elif ontology_catalog and ontology_catalog.get("ontology_root"):
        ontology_root = ontology_catalog["ontology_root"]
    if not ontology_root:
        return None
    candidate = Path(str(ontology_root))
    if not candidate.exists():
        return None
    try:
        return load_ontology_triplet(candidate)
    except Exception as exc:
        print(f"  warning: could not load ontology tree from {candidate}: {exc}")
        return None


def _preferred_paths_from_config(kind: str, config: dict | None, ontology_catalog: dict | None) -> list[str]:
    if kind == "attack":
        if ontology_catalog and ontology_catalog.get("selected_attack_leaves"):
            return _ordered_unique(ontology_catalog["selected_attack_leaves"])
        if config and config.get("attack_leaves"):
            return _ordered_unique(part.strip() for part in str(config["attack_leaves"]).split(","))
        if config and config.get("attack_leaf"):
            return _ordered_unique([config["attack_leaf"]])
        return []
    if ontology_catalog and ontology_catalog.get("selected_opinion_leaves"):
        return _ordered_unique(ontology_catalog["selected_opinion_leaves"])
    return []


def _match_preferred_paths(preferred_paths: Sequence[str], discovered_paths: Sequence[str]) -> list[str]:
    discovered = _ordered_unique(discovered_paths)
    discovered_set = set(discovered)
    by_leaf: dict[str, list[str]] = {}
    for path in discovered:
        by_leaf.setdefault(_last_leaf(path), []).append(path)

    resolved: list[str] = []
    for preferred in preferred_paths:
        preferred_text = str(preferred).strip()
        if preferred_text in discovered_set and preferred_text not in resolved:
            resolved.append(preferred_text)
            continue
        leaf = _last_leaf(preferred_text)
        matches = by_leaf.get(leaf, [])
        if len(matches) == 1 and matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def _resolve_ontology_order(
    kind: str,
    discovered_paths: Sequence[str],
    config: dict | None,
    ontology_catalog: dict | None,
    ontologies: dict | None,
) -> list[str]:
    discovered = _ordered_unique(discovered_paths)
    discovered_set = set(discovered)
    ordered: list[str] = []

    ontology_tree = None
    if ontologies:
        ontology_tree = ontologies["ATTACK" if kind == "attack" else "OPINION"]
    if ontology_tree is not None:
        ordered.extend(path for path in flatten_leaf_paths(ontology_tree) if path in discovered_set)

    preferred = _preferred_paths_from_config(kind, config, ontology_catalog)
    for path in _match_preferred_paths(preferred, discovered):
        if path not in ordered:
            ordered.append(path)

    for path in discovered:
        if path not in ordered:
            ordered.append(path)
    return ordered


def _build_path_tree(paths: Sequence[str]) -> dict[str, dict]:
    tree: dict[str, dict] = {}
    for path in _ordered_unique(paths):
        node = tree
        for part in _split_path(path):
            node = node.setdefault(part, {})
    return tree


def _draw_hierarchy_dendrogram(ax: plt.Axes, paths: Sequence[str], orientation: str) -> list[int]:
    ordered_paths = _ordered_unique(paths)
    if not ordered_paths:
        ax.set_axis_off()
        return []

    if len(ordered_paths) == 1:
        ax.set_axis_off()
        return [0]

    tree = _build_path_tree(ordered_paths)
    leaf_positions = {path: idx * 10 + 5 for idx, path in enumerate(ordered_paths)}

    def visit(node: dict[str, dict], prefix: tuple[str, ...]) -> tuple[float, float]:
        child_points: list[tuple[float, float]] = []
        for key, child in node.items():
            current_prefix = prefix + (key,)
            if child:
                child_points.append(visit(child, current_prefix))
            else:
                child_points.append((leaf_positions[" > ".join(current_prefix)], 0.0))

        child_positions = [point[0] for point in child_points]
        child_heights = [point[1] for point in child_points]
        node_position = float(sum(child_positions) / len(child_positions))
        node_height = float(max(child_heights) + 1.0) if child_points else 0.0

        if orientation == "top":
            for child_position, child_height in child_points:
                ax.plot(
                    [child_position, child_position],
                    [child_height, node_height],
                    color=DENDROGRAM_COLOR,
                    linewidth=1.4,
                    solid_capstyle="butt",
                )
            if len(child_positions) > 1:
                ax.plot(
                    [min(child_positions), max(child_positions)],
                    [node_height, node_height],
                    color=DENDROGRAM_COLOR,
                    linewidth=1.4,
                    solid_capstyle="butt",
                )
        else:
            for child_position, child_height in child_points:
                ax.plot(
                    [child_height, node_height],
                    [child_position, child_position],
                    color=DENDROGRAM_COLOR,
                    linewidth=1.4,
                    solid_capstyle="butt",
                )
            if len(child_positions) > 1:
                ax.plot(
                    [node_height, node_height],
                    [min(child_positions), max(child_positions)],
                    color=DENDROGRAM_COLOR,
                    linewidth=1.4,
                    solid_capstyle="butt",
                )
        return node_position, node_height

    if len(tree) == 1:
        root_label, root_node = next(iter(tree.items()))
        _, root_height = visit(root_node, (root_label,))
    else:
        _, root_height = visit(tree, ())

    if orientation == "top":
        ax.set_xlim(0, len(ordered_paths) * 10)
        ax.set_ylim(0, root_height + 0.35)
    else:
        ax.set_xlim(0, root_height + 0.35)
        ax.set_ylim(len(ordered_paths) * 10, 0)
    ax.margins(x=0, y=0)
    ax.set_axis_off()
    return list(range(len(ordered_paths)))


def _style_dendrogram_axis(
    ax: plt.Axes,
    orientation: str,
    x_max: float | None = None,
    y_max: float | None = None,
) -> None:
    if x_max is not None:
        ax.set_xlim(0, x_max)
    if y_max is not None:
        if orientation == "right":
            ax.set_ylim(y_max, 0)
        else:
            ax.set_ylim(0, y_max)
    if orientation == "right":
        ax.set_xlim(left=0)
    ax.margins(x=0, y=0)
    ax.set_axis_off()


def _imshow_heat(
    ax: plt.Axes,
    data: np.ndarray,
    cmap: str,
    norm_or_vmax,
    extent: tuple[float, float, float, float],
    annot: np.ndarray | None = None,
):
    n_rows, n_cols = data.shape
    kwargs = dict(aspect="auto", extent=extent, interpolation="nearest")
    if isinstance(norm_or_vmax, (int, float)):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=norm_or_vmax, **kwargs)
        vmax_color = float(norm_or_vmax) if norm_or_vmax else 1.0
    else:
        im = ax.imshow(data, cmap=cmap, norm=norm_or_vmax, **kwargs)
        vmax_color = float(getattr(norm_or_vmax, "vmax", 1.0) or 1.0)

    for i in range(1, n_cols):
        ax.axvline(x=i * 10, color="white", linewidth=0.55)
    for i in range(1, n_rows):
        ax.axhline(y=i * 10, color="white", linewidth=0.55)

    if annot is not None:
        for row in range(n_rows):
            for col in range(n_cols):
                text = annot[row, col]
                if not text:
                    continue
                cell_value = float(np.nan_to_num(data[row, col], nan=0.0))
                brightness = abs(cell_value) / max(vmax_color, 1e-9)
                color = "white" if brightness > 0.55 else INK
                ax.text(
                    col * 10 + 5,
                    row * 10 + 5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    fontweight="bold",
                    color=color,
                )
    return im


def _outcome_column_for_path(profile_wide_df: pd.DataFrame, opinion_path: str) -> str | None:
    candidates = [
        f"adversarial_delta_indicator__{leaf_to_key(_last_leaf(opinion_path))}",
        f"adversarial_delta_indicator__{leaf_to_key(opinion_path)}",
    ]
    for candidate in candidates:
        if candidate in profile_wide_df.columns:
            return candidate
    return None


def _load_predictor_score_map(stage06_dir: Path) -> dict[str, float]:
    score_map: dict[str, float] = {}

    weight_path = stage06_dir / "moderator_weight_table.csv"
    if weight_path.exists():
        weight_df = pd.read_csv(weight_path)
        if "term" in weight_df.columns:
            value_col = "normalized_weight_pct" if "normalized_weight_pct" in weight_df.columns else None
            if value_col is None and "mean_abs_estimate" in weight_df.columns:
                value_col = "mean_abs_estimate"
            if value_col is not None:
                for row in weight_df.itertuples():
                    try:
                        score_map[str(row.term)] = float(abs(getattr(row, value_col)))
                    except Exception:
                        continue

    ridge_path = stage06_dir / "ridge_full_coefficients.csv"
    if ridge_path.exists():
        ridge_df = pd.read_csv(ridge_path)
        if "term" in ridge_df.columns and "ridge_estimate" in ridge_df.columns:
            for row in ridge_df.itertuples():
                score_map.setdefault(str(row.term), float(abs(row.ridge_estimate)))

    return score_map


def _prefer_raw_column(column: str, profile_wide_df: pd.DataFrame) -> str:
    if column.endswith("_z"):
        raw = column[:-2]
        if raw in profile_wide_df.columns:
            return raw
    return column


def _predictor_spec_from_column(
    column: str,
    registry: FeatureRegistry,
    score_map: dict[str, float],
    label_override: str | None = None,
) -> PredictorSpec:
    dimension = registry.col_to_dimension(column)
    inventory = registry.col_to_inventory(column)

    score = score_map.get(column, 0.0)
    if score == 0.0 and not column.endswith("_z"):
        score = score_map.get(f"{column}_z", 0.0)

    if dimension and dimension.is_categorical:
        level_label = next((leaf.label for leaf in dimension.leaves if leaf.col == column), registry.label(column))
        label = label_override or f"{dimension.label}: {level_label}"
        if inventory:
            hierarchy = (inventory.label, dimension.label, level_label)
        else:
            hierarchy = (dimension.label, level_label)
        return PredictorSpec(column=column, label=label, hierarchy_path=hierarchy, score=score)

    if dimension:
        label = label_override or dimension.label
        if inventory:
            hierarchy = (inventory.label, dimension.label)
        else:
            hierarchy = (dimension.label,)
        return PredictorSpec(column=column, label=label, hierarchy_path=hierarchy, score=score)

    label = label_override or registry.label(column)
    if inventory:
        hierarchy = (inventory.label, label)
    else:
        hierarchy = (label,)
    return PredictorSpec(column=column, label=label, hierarchy_path=hierarchy, score=score)


def _canonical_predictors(
    profile_wide_df: pd.DataFrame,
    registry: FeatureRegistry,
    score_map: dict[str, float],
) -> list[PredictorSpec]:
    specs: list[PredictorSpec] = []
    used_columns: set[str] = set()
    for aliases, label in CANONICAL_CORE_PREDICTORS:
        column = next((candidate for candidate in aliases if candidate in profile_wide_df.columns), None)
        if column is None:
            continue
        column = _prefer_raw_column(column, profile_wide_df)
        if column in used_columns:
            continue
        specs.append(_predictor_spec_from_column(column, registry, score_map, label_override=label))
        used_columns.add(column)
    return specs


def _reference_level(levels: Sequence[tuple[str, str]], profile_wide_df: pd.DataFrame) -> str | None:
    if not levels:
        return None
    ranked = sorted(
        ((float(profile_wide_df[col].fillna(0).mean()), col) for _label, col in levels if col in profile_wide_df.columns),
        reverse=True,
    )
    if ranked:
        return ranked[0][1]
    return levels[0][1]


def _generic_predictors(
    profile_wide_df: pd.DataFrame,
    registry: FeatureRegistry,
    score_map: dict[str, float],
    max_rows: int = 8,
) -> list[PredictorSpec]:
    candidates: list[PredictorSpec] = []
    seen: set[str] = set()

    for column, _dim_label, _inventory_label in registry.dimension_mean_cols():
        raw_column = _prefer_raw_column(column, profile_wide_df)
        if raw_column not in profile_wide_df.columns or raw_column in seen:
            continue
        if raw_column.endswith("_z"):
            continue
        candidates.append(_predictor_spec_from_column(raw_column, registry, score_map))
        seen.add(raw_column)

    for _dim_label, _inventory_label, levels in registry.categorical_group_info():
        reference_col = _reference_level(levels, profile_wide_df)
        for _level_label, column in levels:
            if column not in profile_wide_df.columns or column == reference_col or column in seen:
                continue
            candidates.append(_predictor_spec_from_column(column, registry, score_map))
            seen.add(column)

    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda spec: spec.score, reverse=True)
    selected_columns = [spec.column for spec in ranked if spec.score > 0][:max_rows]
    if len(selected_columns) < min(max_rows, 4):
        selected_columns = [spec.column for spec in ranked[:max_rows]]
    selected_set = set(selected_columns)
    return [spec for spec in candidates if spec.column in selected_set][:max_rows]


def _available_predictors(profile_wide_df: pd.DataFrame, stage06_dir: Path) -> list[PredictorSpec]:
    registry = FeatureRegistry(profile_wide_df)
    score_map = _load_predictor_score_map(stage06_dir)

    canonical = _canonical_predictors(profile_wide_df, registry, score_map)
    if len(canonical) >= 4:
        return canonical
    return _generic_predictors(profile_wide_df, registry, score_map)


def _valid_predictors(profile_wide_df: pd.DataFrame, specs: Sequence[PredictorSpec]) -> list[PredictorSpec]:
    valid: list[PredictorSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.column not in profile_wide_df.columns or spec.column in seen:
            continue
        values = profile_wide_df[spec.column]
        if values.dropna().nunique() <= 1:
            continue
        valid.append(spec)
        seen.add(spec.column)
    return valid


def _ordered_predictor_specs(specs: Sequence[PredictorSpec]) -> list[PredictorSpec]:
    return sorted(specs, key=lambda spec: (spec.hierarchy_path, spec.label))


def _make_figure_1(
    long_df: pd.DataFrame,
    output_dirs: Path | str | Sequence[Path | str],
    config: dict | None,
    ontology_catalog: dict | None,
    ontologies: dict | None,
) -> list[str]:
    attack_paths = _resolve_ontology_order(
        "attack",
        long_df["attack_leaf"].dropna().tolist(),
        config,
        ontology_catalog,
        ontologies,
    )
    opinion_paths = _resolve_ontology_order(
        "opinion",
        long_df["opinion_leaf"].dropna().tolist(),
        config,
        ontology_catalog,
        ontologies,
    )

    mean_piv = (
        long_df.groupby(["attack_leaf", "opinion_leaf"])["adversarial_effectivity"]
        .mean()
        .reset_index()
        .pivot(index="attack_leaf", columns="opinion_leaf", values="adversarial_effectivity")
        .reindex(index=attack_paths, columns=opinion_paths)
    )
    std_piv = (
        long_df.groupby(["attack_leaf", "opinion_leaf"])["adversarial_effectivity"]
        .std()
        .reset_index()
        .pivot(index="attack_leaf", columns="opinion_leaf", values="adversarial_effectivity")
        .reindex(index=attack_paths, columns=opinion_paths)
    )

    mean_vals = mean_piv.values.astype(float)
    if mean_vals.size == 0:
        return []

    n_rows, n_cols = mean_piv.shape
    fig = plt.figure(figsize=(18.5, 6.8))
    fig.patch.set_facecolor(WHITE)
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.18, 1.0, 0.12],
        width_ratios=[1.0, 0.14, 1.0, 0.14],
        hspace=0.0,
        wspace=0.0,
    )

    ax_dend_top_mean = fig.add_subplot(gs[0, 0])
    ax_dend_top_std = fig.add_subplot(gs[0, 2])
    ax_heat_mean = fig.add_subplot(gs[1, 0])
    ax_dend_right_mean = fig.add_subplot(gs[1, 1])
    ax_heat_std = fig.add_subplot(gs[1, 2])
    ax_dend_right_std = fig.add_subplot(gs[1, 3])

    col_order = _draw_hierarchy_dendrogram(ax_dend_top_mean, opinion_paths, orientation="top")
    _draw_hierarchy_dendrogram(ax_dend_top_std, opinion_paths, orientation="top")
    row_order = _draw_hierarchy_dendrogram(ax_dend_right_mean, attack_paths, orientation="right")
    _draw_hierarchy_dendrogram(ax_dend_right_std, attack_paths, orientation="right")

    mean_ordered = mean_piv.iloc[row_order, col_order].values.astype(float)
    std_ordered = std_piv.iloc[row_order, col_order].values.astype(float)
    col_labels = [_cluster_label_from_value(opinion_paths[idx], width=18) for idx in col_order]
    row_labels = [_cluster_label_from_value(attack_paths[idx], width=12) for idx in row_order]

    extent = (0, n_cols * 10, n_rows * 10, 0)
    x_ticks = np.arange(n_cols) * 10 + 5
    y_ticks = np.arange(n_rows) * 10 + 5

    vmax_mean = max(float(np.nanmax(np.abs(mean_ordered))) * 1.05, 1.0)
    norm_mean = TwoSlopeNorm(vmin=-vmax_mean, vcenter=0.0, vmax=vmax_mean)
    annot_mean = np.array(
        [
            [f"{mean_ordered[row, col]:.1f}" if not np.isnan(mean_ordered[row, col]) else "" for col in range(n_cols)]
            for row in range(n_rows)
        ]
    )
    im_mean = _imshow_heat(ax_heat_mean, mean_ordered, "RdBu_r", norm_mean, extent, annot_mean)

    ax_heat_mean.set_xticks(x_ticks)
    ax_heat_mean.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8.0)
    ax_heat_mean.set_yticks(y_ticks)
    ax_heat_mean.set_yticklabels(row_labels, fontsize=9.5)
    ax_heat_mean.set_xlim(0, n_cols * 10)
    ax_heat_mean.set_ylim(n_rows * 10, 0)
    ax_heat_mean.tick_params(axis="both", which="both", length=0)
    ax_heat_mean.set_xlabel("")
    ax_heat_mean.set_ylabel("")

    vmax_std = max(float(np.nanmax(std_ordered)) * 1.05, 1.0)
    annot_std = np.array(
        [
            [f"{std_ordered[row, col]:.1f}" if not np.isnan(std_ordered[row, col]) else "" for col in range(n_cols)]
            for row in range(n_rows)
        ]
    )
    im_std = _imshow_heat(ax_heat_std, std_ordered, "YlOrRd", vmax_std, extent, annot_std)

    ax_heat_std.set_xticks(x_ticks)
    ax_heat_std.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8.0)
    ax_heat_std.set_yticks(y_ticks)
    ax_heat_std.set_yticklabels([])
    ax_heat_std.set_xlim(0, n_cols * 10)
    ax_heat_std.set_ylim(n_rows * 10, 0)
    ax_heat_std.tick_params(axis="both", which="both", length=0)
    ax_heat_std.set_xlabel("")
    ax_heat_std.set_ylabel("")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.19, wspace=0.02, hspace=0.00)
    _style_dendrogram_axis(ax_dend_top_mean, orientation="top", x_max=n_cols * 10)
    _style_dendrogram_axis(ax_dend_top_std, orientation="top", x_max=n_cols * 10)
    _style_dendrogram_axis(ax_dend_right_mean, orientation="right", y_max=n_rows * 10)
    _style_dendrogram_axis(ax_dend_right_std, orientation="right", y_max=n_rows * 10)

    pos_mean = ax_heat_mean.get_position()
    pos_std = ax_heat_std.get_position()
    pos_dend_mean = ax_dend_right_mean.get_position()
    pos_dend_std = ax_dend_right_std.get_position()
    cbar_height = 0.022
    cbar_y = 0.09
    panel_width_mean = pos_dend_mean.x1 - pos_mean.x0
    panel_width_std = pos_dend_std.x1 - pos_std.x0
    cbar_width_mean = panel_width_mean * 0.72
    cbar_width_std = panel_width_std * 0.72
    cbar_x_mean = pos_mean.x0 + (panel_width_mean - cbar_width_mean) / 2
    cbar_x_std = pos_std.x0 + (panel_width_std - cbar_width_std) / 2
    cbar_ax1 = fig.add_axes([cbar_x_mean, cbar_y, cbar_width_mean, cbar_height])
    cbar_ax2 = fig.add_axes([cbar_x_std, cbar_y, cbar_width_std, cbar_height])

    cb1 = fig.colorbar(im_mean, cax=cbar_ax1, orientation="horizontal")
    cb1.set_label("Mean AE  (positive = attack succeeded)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=8)

    cb2 = fig.colorbar(im_std, cax=cbar_ax2, orientation="horizontal")
    cb2.set_label("SD(AE) across profiles  (inter-individual spread)", fontsize=8.5)
    cb2.ax.tick_params(labelsize=8)

    return _save(fig, "figure_readme_1_ae_factorial", output_dirs)


def _make_figure_2(
    profile_wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    stage06_dir: Path,
    output_dirs: Path | str | Sequence[Path | str],
    config: dict | None,
    ontology_catalog: dict | None,
    ontologies: dict | None,
) -> list[str]:
    predictor_specs = _ordered_predictor_specs(
        _valid_predictors(profile_wide_df, _available_predictors(profile_wide_df, stage06_dir))
    )
    if not predictor_specs:
        print("  skipped figure_readme_2_moderation_heatmap: no eligible moderator predictors available")
        return []

    opinion_paths = _resolve_ontology_order(
        "opinion",
        long_df["opinion_leaf"].dropna().tolist(),
        config,
        ontology_catalog,
        ontologies,
    )

    outcome_cols: list[str] = []
    outcome_paths: list[str] = []
    for path in opinion_paths:
        column = _outcome_column_for_path(profile_wide_df, path)
        if column is None or column.endswith("_z"):
            continue
        outcome_cols.append(column)
        outcome_paths.append(path)

    if not outcome_cols:
        print("  skipped figure_readme_2_moderation_heatmap: adversarial delta indicators not available")
        return []

    X = profile_wide_df[[spec.column for spec in predictor_specs]].copy()
    for spec in predictor_specs:
        if spec.column.startswith("profile_cont_"):
            mu = X[spec.column].mean()
            sd = X[spec.column].std(ddof=1)
            if pd.notna(sd) and sd > 0:
                X[spec.column] = (X[spec.column] - mu) / sd

    coef_df = pd.DataFrame(
        index=[spec.label for spec in predictor_specs],
        columns=outcome_paths,
        dtype=float,
    )
    pval_df = pd.DataFrame(
        index=[spec.label for spec in predictor_specs],
        columns=outcome_paths,
        dtype=float,
    )

    Xmat = X.values.astype(float)
    n_obs = len(Xmat)
    X_design = np.column_stack([np.ones(n_obs), Xmat])
    n_params = X_design.shape[1]

    for outcome_col, outcome_path in zip(outcome_cols, outcome_paths):
        y = profile_wide_df[outcome_col].fillna(0).values.astype(float)
        coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ coefs
        sse = float((resid**2).sum())
        df_error = n_obs - n_params
        if df_error <= 0 or sse <= 0:
            continue
        mse = sse / df_error
        xtx_inv = np.linalg.pinv(X_design.T @ X_design)
        se = np.sqrt(mse * np.diag(xtx_inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.divide(coefs, se, out=np.zeros_like(coefs), where=se > 0)
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df=df_error)
        for idx, spec in enumerate(predictor_specs):
            coef_df.loc[spec.label, outcome_path] = coefs[idx + 1]
            pval_df.loc[spec.label, outcome_path] = p_vals[idx + 1]

    heat_vals = coef_df.values.astype(float)
    n_rows, n_cols = heat_vals.shape
    if n_rows == 0 or n_cols == 0:
        return []

    fig = plt.figure(figsize=(16, 6.8))
    fig.patch.set_facecolor(WHITE)
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.20, 1.0, 0.10],
        width_ratios=[1.0, 0.16],
        hspace=0.0,
        wspace=0.0,
    )

    ax_dend_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_dend_right = fig.add_subplot(gs[1, 1])

    col_order = _draw_hierarchy_dendrogram(ax_dend_top, outcome_paths, orientation="top")
    row_paths = [" > ".join(spec.hierarchy_path) for spec in predictor_specs]
    row_order = _draw_hierarchy_dendrogram(ax_dend_right, row_paths, orientation="right")

    ordered_coef = coef_df.iloc[row_order, col_order]
    ordered_pval = pval_df.iloc[row_order, col_order]
    col_labels = [_cluster_label_from_value(outcome_paths[idx], width=16) for idx in col_order]
    row_labels = [predictor_specs[idx].label for idx in row_order]

    annot = np.array(
        [
            [
                (
                    f"{ordered_coef.iloc[row, col]:.1f}"
                    + (
                        "***"
                        if ordered_pval.iloc[row, col] < 0.001
                        else "**"
                        if ordered_pval.iloc[row, col] < 0.01
                        else "*"
                        if ordered_pval.iloc[row, col] < 0.05
                        else "†"
                        if ordered_pval.iloc[row, col] < 0.10
                        else ""
                    )
                )
                if not pd.isna(ordered_coef.iloc[row, col])
                else ""
                for col in range(n_cols)
            ]
            for row in range(n_rows)
        ]
    )

    vmax = max(float(np.nanmax(np.abs(ordered_coef.values))) * 1.05, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    extent = (0, n_cols * 10, n_rows * 10, 0)
    im = _imshow_heat(ax_heat, ordered_coef.values.astype(float), "RdBu_r", norm, extent, annot)

    x_ticks = np.arange(n_cols) * 10 + 5
    y_ticks = np.arange(n_rows) * 10 + 5
    ax_heat.set_xticks(x_ticks)
    ax_heat.set_xticklabels(col_labels, fontsize=8.6)
    ax_heat.set_yticks(y_ticks)
    ax_heat.set_yticklabels(row_labels, fontsize=10.5)
    ax_heat.set_xlim(0, n_cols * 10)
    ax_heat.set_ylim(n_rows * 10, 0)
    ax_heat.tick_params(axis="both", which="both", length=0)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    fig.subplots_adjust(left=0.17, right=0.95, top=0.88, bottom=0.20, wspace=0.00, hspace=0.00)
    _style_dendrogram_axis(ax_dend_top, orientation="top", x_max=n_cols * 10)
    _style_dendrogram_axis(ax_dend_right, orientation="right", y_max=n_rows * 10)

    pos = ax_heat.get_position()
    pos_dend = ax_dend_right.get_position()
    cbar_height = 0.022
    cbar_y = 0.09
    panel_width = pos_dend.x1 - pos.x0
    cbar_width = panel_width * 0.72
    cbar_x = pos.x0 + (panel_width - cbar_width) / 2
    cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_label(
        "OLS coefficient  (continuous moderators z-scored; non-reference categorical levels unstandardized)",
        fontsize=8.8,
    )
    cb.ax.tick_params(labelsize=8)

    return _save(fig, "figure_readme_2_moderation_heatmap", output_dirs)


def generate_main_readme_figures(
    *,
    stage05_dir: str | Path,
    stage06_dir: str | Path,
    output_dirs: Path | str | Sequence[Path | str],
    config: dict | None = None,
    ontology_catalog: dict | None = None,
) -> list[str]:
    _setup()
    stage05_dir = Path(stage05_dir)
    stage06_dir = Path(stage06_dir)
    long_df = pd.read_csv(stage05_dir / "sem_long_raw.csv")
    profile_wide_df = pd.read_csv(stage06_dir / "profile_sem_wide.csv")
    ontologies = _load_ontologies(config, ontology_catalog)

    saved: list[str] = []
    saved.extend(_make_figure_1(long_df, output_dirs, config, ontology_catalog, ontologies))
    saved.extend(
        _make_figure_2(
            profile_wide_df,
            long_df,
            stage06_dir,
            output_dirs,
            config,
            ontology_catalog,
            ontologies,
        )
    )
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate root README / paper matrix figures.")
    parser.add_argument("--stage05-dir", default=str(DEFAULT_STAGE05))
    parser.add_argument("--stage06-dir", default=str(DEFAULT_STAGE06))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ontology-catalog", default=str(DEFAULT_ONTOLOGY_CATALOG))
    parser.add_argument(
        "--output-dir",
        action="append",
        dest="output_dirs",
        help="Repeat to write to multiple output directories. Defaults to all standard publication mirrors.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_json_if_exists(args.config)
    ontology_catalog = _load_json_if_exists(args.ontology_catalog)
    output_dirs = args.output_dirs if args.output_dirs else DEFAULT_OUTPUT_DIRS
    files = generate_main_readme_figures(
        stage05_dir=args.stage05_dir,
        stage06_dir=args.stage06_dir,
        output_dirs=output_dirs,
        config=config,
        ontology_catalog=ontology_catalog,
    )
    print(f"Done. Wrote {len(files)} files.")
