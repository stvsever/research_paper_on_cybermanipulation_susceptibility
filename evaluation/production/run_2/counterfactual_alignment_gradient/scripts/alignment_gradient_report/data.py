from __future__ import annotations

"""Data loading and table construction for the alignment-gradient branch report.

The functions here transform Stage 05 SEM output and the branch design manifest into condition-level, quality, and plotting-ready tables.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formatting import _clean_leaf
from .paths import ensure_dir, read_json


def _read_design(branch_root: Path) -> dict[str, Any]:
    path = branch_root / "design" / "alignment_design_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing alignment design manifest: {path}")
    return read_json(path)


def _load_sem(branch_root: Path) -> pd.DataFrame:
    path = (
        branch_root
        / "merged_outputs"
        / "stage_outputs"
        / "05_compute_effectivity_deltas"
        / "sem_long_encoded.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing branch Stage 05 SEM data: {path}")
    sem = pd.read_csv(path).copy()
    required = [
        "alignment_condition_id",
        "target_alignment_z",
        "achieved_alignment_z",
        "pn_increment_effectivity",
        "ae_total_network",
        "ae_private",
    ]
    missing = [col for col in required if col not in sem.columns]
    if missing:
        raise RuntimeError("Branch SEM data is missing required columns: " + ", ".join(missing))
    sem["opinion_label"] = sem["opinion_leaf"].map(_clean_leaf)
    sem["attack_label"] = sem["attack_leaf"].map(_clean_leaf)
    return sem


def _schedule_table(design: dict[str, Any]) -> pd.DataFrame:
    schedule = pd.DataFrame(design.get("conditions") or [])
    if schedule.empty:
        raise RuntimeError("Alignment design manifest does not contain condition schedule rows.")
    schedule = schedule.rename(columns={"condition_id": "alignment_condition_id"})
    keep = [
        "alignment_condition_id",
        "condition_index",
        "opinion_index",
        "attack_index",
        "condition_input_path",
        "assignment_map_path",
    ]
    return schedule[[col for col in keep if col in schedule.columns]].copy()


def _condition_table(sem: pd.DataFrame, design: dict[str, Any]) -> pd.DataFrame:
    condition = (
        sem.groupby("alignment_condition_id", as_index=False)
        .agg(
            opinion_leaf=("opinion_leaf", "first"),
            opinion_label=("opinion_label", "first"),
            attack_leaf=("attack_leaf", "first"),
            attack_label=("attack_label", "first"),
            alignment_level=("alignment_level", "first"),
            alignment_level_index=("alignment_level_index", "first"),
            target_alignment_z=("target_alignment_z", "first"),
            achieved_alignment_z=("achieved_alignment_z", "first"),
            alignment_abs_error=("alignment_abs_error", "first"),
            n_profiles=("profile_id", "nunique"),
            mean_ae_private=("ae_private", "mean"),
            mean_pn_increment_effectivity=("pn_increment_effectivity", "mean"),
            mean_ae_total_network=("ae_total_network", "mean"),
            sd_pn_increment_effectivity=("pn_increment_effectivity", "std"),
            sd_ae_total_network=("ae_total_network", "std"),
        )
        .sort_values(["target_alignment_z", "opinion_label", "attack_label"])
        .reset_index(drop=True)
    )
    schedule = _schedule_table(design)
    if "alignment_condition_id" in schedule.columns:
        condition = condition.merge(schedule, on="alignment_condition_id", how="left")
    return condition


def _level_summary(condition: pd.DataFrame) -> pd.DataFrame:
    return (
        condition.groupby(["alignment_level", "target_alignment_z"], as_index=False)
        .agg(
            n_conditions=("alignment_condition_id", "count"),
            mean_achieved_alignment_z=("achieved_alignment_z", "mean"),
            mean_pn_increment_effectivity=("mean_pn_increment_effectivity", "mean"),
            se_pn_increment_effectivity=("mean_pn_increment_effectivity", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            mean_ae_total_network=("mean_ae_total_network", "mean"),
            se_ae_total_network=("mean_ae_total_network", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
        )
        .sort_values("target_alignment_z")
        .reset_index(drop=True)
    )


def _design_balance(condition: pd.DataFrame) -> pd.DataFrame:
    return (
        condition.groupby(["alignment_level", "target_alignment_z"], as_index=False)
        .agg(
            n_conditions=("alignment_condition_id", "count"),
            n_attacks=("attack_label", "nunique"),
            n_opinions=("opinion_label", "nunique"),
            mean_achieved_alignment_z=("achieved_alignment_z", "mean"),
            min_achieved_alignment_z=("achieved_alignment_z", "min"),
            max_achieved_alignment_z=("achieved_alignment_z", "max"),
            max_alignment_abs_error=("alignment_abs_error", "max"),
            attack_labels=("attack_label", lambda values: ", ".join(sorted({str(v) for v in values}))),
            opinion_labels=("opinion_label", lambda values: ", ".join(sorted({str(v) for v in values}))),
        )
        .sort_values("target_alignment_z")
        .reset_index(drop=True)
    )


def _condition_vulnerability_plane_frame(sem: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    required = [
        "alignment_condition_id",
        "profile_id",
        "ae_private",
        "exposure_outgoing_visibility_weight",
    ]
    missing = [col for col in required if col not in sem.columns]
    if missing:
        raise RuntimeError("Branch SEM data is missing vulnerability-plane columns: " + ", ".join(missing))

    condition_cols = [
        "alignment_condition_id",
        "opinion_label",
        "attack_label",
        "target_alignment_z",
        "achieved_alignment_z",
        "opinion_index",
        "attack_index",
    ]
    plot_df = sem.merge(
        condition[[col for col in condition_cols if col in condition.columns]],
        on="alignment_condition_id",
        how="left",
        suffixes=("", "_condition"),
    ).copy()
    plot_df["ae_private"] = pd.to_numeric(plot_df["ae_private"], errors="coerce")
    plot_df["sender_reach"] = pd.to_numeric(plot_df["exposure_outgoing_visibility_weight"], errors="coerce").fillna(0.0)
    ae_mean = plot_df.groupby("alignment_condition_id")["ae_private"].transform("mean")
    ae_sd = plot_df.groupby("alignment_condition_id")["ae_private"].transform(lambda x: x.std(ddof=0))
    plot_df["ae_private_z"] = (plot_df["ae_private"] - ae_mean) / ae_sd.replace(0.0, np.nan)
    plot_df["sender_reach_percentile"] = plot_df.groupby("alignment_condition_id")["sender_reach"].rank(pct=True)
    plot_df["high_reach_sender"] = plot_df["sender_reach_percentile"] >= 0.80
    plot_df["top10_sender"] = plot_df["sender_reach_percentile"] >= 0.90
    return plot_df


def _condition_vulnerability_plane_summary(sem: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    plot_df = _condition_vulnerability_plane_frame(sem, condition)
    rows: list[dict[str, Any]] = []
    for condition_id, group in plot_df.groupby("alignment_condition_id", sort=False):
        high = group[group["high_reach_sender"]]
        bottom = group[~group["high_reach_sender"]]
        top10 = group[group["top10_sender"]]
        rows.append(
            {
                "alignment_condition_id": condition_id,
                "opinion_label": group["opinion_label"].iloc[0],
                "attack_label": group["attack_label"].iloc[0],
                "target_alignment_z": float(group["target_alignment_z"].iloc[0]),
                "achieved_alignment_z": float(group["achieved_alignment_z"].iloc[0]),
                "top20_mean_ae_private_z": float(high["ae_private_z"].mean()),
                "bottom80_mean_ae_private_z": float(bottom["ae_private_z"].mean()),
                "top10_mean_ae_private_z": float(top10["ae_private_z"].mean()),
                "top20_positive_share": float((high["ae_private_z"] > 0).mean()),
                "high_reach_profile_count": int(high["profile_id"].nunique()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["target_alignment_z", "opinion_label", "attack_label"])
        .reset_index(drop=True)
    )


def _write_tables(
    branch_root: Path,
    condition: pd.DataFrame,
    level: pd.DataFrame,
    models: pd.DataFrame,
    robustness: pd.DataFrame,
    design_balance: pd.DataFrame,
    original_comparison: pd.DataFrame,
    vulnerability_plane_summary: pd.DataFrame,
    figure4_summary: pd.DataFrame,
    quality: pd.DataFrame,
) -> dict[str, Path]:
    tables_dir = ensure_dir(branch_root / "network_exposure_analysis" / "tables")
    paths = {
        "condition_level_h3h4": tables_dir / "condition_level_h3h4.csv",
        "alignment_level_summary": tables_dir / "alignment_level_summary.csv",
        "alignment_design_balance": tables_dir / "alignment_design_balance.csv",
        "h3h4_model_results": tables_dir / "h3h4_model_results.csv",
        "h3h4_robustness_results": tables_dir / "h3h4_robustness_results.csv",
        "original_vs_branch_alignment_comparison": tables_dir / "original_vs_branch_alignment_comparison.csv",
        "condition_vulnerability_plane_summary": tables_dir / "condition_vulnerability_plane_summary.csv",
        "figure4_outcome_test_summary": tables_dir / "figure4_outcome_test_summary.csv",
        "quality_gates": tables_dir / "quality_gates.csv",
    }
    condition.to_csv(paths["condition_level_h3h4"], index=False)
    level.to_csv(paths["alignment_level_summary"], index=False)
    design_balance.to_csv(paths["alignment_design_balance"], index=False)
    models.to_csv(paths["h3h4_model_results"], index=False)
    robustness.to_csv(paths["h3h4_robustness_results"], index=False)
    original_comparison.to_csv(paths["original_vs_branch_alignment_comparison"], index=False)
    vulnerability_plane_summary.to_csv(paths["condition_vulnerability_plane_summary"], index=False)
    figure4_summary.to_csv(paths["figure4_outcome_test_summary"], index=False)
    quality.to_csv(paths["quality_gates"], index=False)
    return paths


def _quality_table(branch_root: Path, condition: pd.DataFrame) -> pd.DataFrame:
    stage04b_summary_path = (
        branch_root
        / "merged_outputs"
        / "stage_outputs"
        / "04b_assess_post_attack_network_exposure_opinions"
        / "post_attack_network_exposure_summary.json"
    )
    stage05_summary_path = (
        branch_root
        / "merged_outputs"
        / "stage_outputs"
        / "05_compute_effectivity_deltas"
        / "delta_summary.json"
    )
    stage04b = read_json(stage04b_summary_path) if stage04b_summary_path.exists() else {}
    stage05 = read_json(stage05_summary_path) if stage05_summary_path.exists() else {}
    rows = [
        {"metric": "merged_stage04b_rows", "value": stage04b.get("n_records"), "expected": 3500},
        {"metric": "stage05_rows", "value": stage05.get("n_records"), "expected": 3500},
        {"metric": "condition_count", "value": condition["alignment_condition_id"].nunique(), "expected": 35},
        {"metric": "stage04b_fallback_count", "value": stage04b.get("fallback_count"), "expected": 0},
        {"metric": "stage04b_skipped_count", "value": stage04b.get("skipped_task_count"), "expected": 0},
        {
            "metric": "max_alignment_abs_error_x1000",
            "value": float(condition["alignment_abs_error"].max()) * 1000.0,
            "expected": 30.0,
        },
    ]
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["metric"].eq("max_alignment_abs_error_x1000"), out["value"] <= out["expected"], out["value"] == out["expected"])
    out["status"] = out["status"].map({True: "pass", False: "fail"})
    return out
