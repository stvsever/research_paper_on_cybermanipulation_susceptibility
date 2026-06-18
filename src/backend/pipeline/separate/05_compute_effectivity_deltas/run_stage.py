from __future__ import annotations

"""
Technical overview
------------------
Stage 05 is the bridge between raw scenario-level simulation outputs and the
 statistical datasets used downstream for moderation analysis.

It performs three jobs:
1. construct attacked effectivity outcomes for each scenario row
2. encode profile variables and opinion fixed effects into analysis-ready form
3. roll the long attacked table up into profile-level repeated-outcome tables

The stage keeps both signed and absolute opinion movement:

    delta_score     = post_score - baseline_score
    abs_delta_score = |post_score - baseline_score|

The absolute shift is important because one fixed attack can move different
opinion leaves in different signed directions. If only signed deltas were kept,
cross-leaf movement could cancel out.

This stage also creates the profile-level wide panel used by Stage 06. In that
wide table, each profile receives separate attacked outcome indicators for each
opinion leaf, which enables repeated-outcome SEM/path modeling rather than a
premature collapse to a single summary score.

Exposure-network update
-----------------------
When Stage 01b/02b/04b fields are present, Stage 05 also flattens empirical
PolitiSky24 exposure-network assignments and network-context summaries into the
analysis tables. These columns expose receiver exposure, sender reach,
bridge/centrality position, network peer counts, exposure-weighted peer means,
and formula-aligned variables for private attack effectivity, baseline-network
increment, post-attack-network increment, and total network-exposed effect.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.data_utils import choose_primary_moderator_column, one_hot_profile_categoricals, zscore_series
from src.backend.utils.io import (
    abs_path,
    ensure_dir,
    read_json,
    read_jsonl,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.ontology_utils import load_adversarial_directions_from_opinion
from src.backend.utils.scenario_realism import extract_leaf_label, extract_opinion_domain
from src.backend.utils.schemas import (
    AttackExposure,
    DeltaRecord,
    OpinionAssessment,
    ScenarioRecord,
    SemRow,
    StageArtifactManifest,
    StageConfig,
)


LOGGER = logging.getLogger(__name__)


class Stage05Config(StageConfig):
    primary_moderator: str = "profile_cont_age_years"
    ontology_root: Optional[str] = None


def _load_adversarial_directions(ontology_root: Optional[str]) -> Dict[str, int]:
    """Load per-leaf adversarial goal directions from the embedded opinion.json ontology.

    Returns a mapping from leaf name (last path component) to direction in {-1, +1}.
    Only non-zero directions are returned; 0-encoded leaves are excluded from scoring.

    If no ontology_root is given or opinion.json is not found, returns an empty dict
    (caller will default all directions to +1, equivalent to treating signed delta as
    the effectivity metric).
    """
    if not ontology_root:
        return {}
    opinion_path = Path(ontology_root) / "OPINION" / "opinion.json"
    if not opinion_path.exists():
        LOGGER.warning("opinion.json not found at %s; adversarial directions unavailable.", opinion_path)
        return {}
    opinion_tree = read_json(opinion_path)
    directions, goal = load_adversarial_directions_from_opinion(opinion_tree)
    LOGGER.info(
        "Loaded adversarial directions from opinion.json: %d non-zero leaf directions. Operator goal: %s",
        len(directions),
        goal[:80] if goal else "unspecified",
    )
    return directions


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace(">", "_")


def _add_fixed_effects(
    df: pd.DataFrame,
    source_column: str,
    prefix: str,
) -> tuple[pd.DataFrame, str | None]:
    unique_values = sorted(df[source_column].dropna().unique().tolist())
    if len(unique_values) <= 1:
        return df, unique_values[0] if unique_values else None

    reference_value = unique_values[0]
    for value in unique_values[1:]:
        column_name = f"{prefix}_{_slugify(value)}"
        df[column_name] = (df[source_column] == value).astype(float)
    return df, reference_value


EXPOSURE_ASSIGNMENT_STRING_FIELDS: Dict[str, str] = {
    "graph_id": "exposure_graph_id",
    "position_id": "exposure_position_id",
    "network_basis": "exposure_network_basis",
    "community_id": "exposure_community_id",
    "display_role": "exposure_display_role",
}

EXPOSURE_ASSIGNMENT_NUMERIC_FIELDS: Dict[str, str] = {
    "assignment_rank": "exposure_assignment_rank",
    "weighted_in_degree": "exposure_weighted_in_degree",
    "incoming_peer_count": "exposure_incoming_peer_count",
    "incoming_exposure_weight": "exposure_incoming_exposure_weight",
    "incoming_top1_share": "exposure_incoming_top1_share",
    "incoming_top5_share": "exposure_incoming_top5_share",
    "incoming_effective_peer_count": "exposure_incoming_effective_peer_count",
    "cross_community_incoming_share": "exposure_cross_community_incoming_share",
    "weighted_out_degree": "exposure_weighted_out_degree",
    "outgoing_receiver_count": "exposure_outgoing_receiver_count",
    "outgoing_visibility_weight": "exposure_outgoing_visibility_weight",
    "prompt_topk_out_reach_count": "exposure_prompt_topk_out_reach_count",
    "cascade_reach_potential": "exposure_cascade_reach_potential",
    "bridge_score": "exposure_bridge_score",
    "eigenvector_centrality": "exposure_eigenvector_centrality",
    "approx_betweenness": "exposure_approx_betweenness",
    "local_clustering": "exposure_local_clustering",
    "h2_neighborhood_activation_readiness": "exposure_h2_neighborhood_activation_readiness",
    "h3_central_susceptible_sender_readiness": "exposure_h3_central_susceptible_sender_readiness",
    "h4_central_resilient_sender_dampening_capacity": "exposure_h4_central_resilient_sender_dampening_capacity",
}

EXPOSURE_ASSIGNMENT_BOOL_FIELDS: Dict[str, str] = {
    "prompt_ready": "exposure_prompt_ready",
}

EXPOSURE_STABLE_COLUMNS = sorted(
    set(EXPOSURE_ASSIGNMENT_STRING_FIELDS.values())
    | set(EXPOSURE_ASSIGNMENT_NUMERIC_FIELDS.values())
    | set(EXPOSURE_ASSIGNMENT_BOOL_FIELDS.values())
    | {
        "exposure_sender_reach_share",
        "exposure_weighted_in_degree_z",
        "exposure_outgoing_visibility_weight_z",
        "exposure_bridge_score_z",
        "exposure_eigenvector_centrality_z",
        "exposure_cascade_reach_potential_z",
        "exposure_h2_neighborhood_activation_readiness_z",
        "exposure_h3_central_susceptible_sender_readiness_z",
        "exposure_h4_central_resilient_sender_dampening_capacity_z",
    }
)

NETWORK_CONTEXT_COLUMNS = {
    "network_exposure_full_incoming_peer_count": "mean",
    "network_exposure_scored_peer_count": "mean",
    "network_exposure_exemplar_count": "mean",
    "network_exposure_full_incoming_exposure_weight": "mean",
    "network_exposure_scored_exposure_weight": "mean",
    "network_exposure_peer_sd": "mean",
    "post_attack_network_full_incoming_peer_count": "mean",
    "post_attack_network_scored_same_condition_peer_count": "mean",
    "post_attack_network_exemplar_count": "mean",
    "post_attack_network_full_incoming_exposure_weight": "mean",
    "post_attack_network_scored_exposure_weight": "mean",
}

HYPOTHESIS_COLUMNS = {
    "ae_private": "mean",
    "bn_increment": "mean",
    "pn_increment": "mean",
    "ae_total_network": "mean",
    "pn_increment_effectivity": "mean",
    "peer_private_attack_activation": "mean",
}


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _exposure_assignment_from_scenario(scenario: ScenarioRecord) -> dict[str, Any]:
    assignment = scenario.metadata.get("exposure_network_assignment")
    if not isinstance(assignment, dict):
        assignment = scenario.profile.metadata.get("exposure_network_assignment")
    return assignment if isinstance(assignment, dict) else {}


def _flatten_exposure_assignment(scenario: ScenarioRecord) -> dict[str, object]:
    assignment = _exposure_assignment_from_scenario(scenario)
    flattened: dict[str, object] = {}
    for source_key, target_key in EXPOSURE_ASSIGNMENT_STRING_FIELDS.items():
        flattened[target_key] = assignment.get(source_key)
    for source_key, target_key in EXPOSURE_ASSIGNMENT_NUMERIC_FIELDS.items():
        flattened[target_key] = _safe_float(assignment.get(source_key))
    for source_key, target_key in EXPOSURE_ASSIGNMENT_BOOL_FIELDS.items():
        flattened[target_key] = _safe_bool(assignment.get(source_key))
    return flattened


def _flatten_baseline_network_context(context: dict[str, Any]) -> dict[str, object]:
    return {
        "network_exposure_full_incoming_peer_count": _safe_float(context.get("full_incoming_peer_count")),
        "network_exposure_scored_peer_count": _safe_float(context.get("peer_count")),
        "network_exposure_exemplar_count": _safe_float(context.get("exemplar_count")),
        "network_exposure_full_incoming_exposure_weight": _safe_float(context.get("full_incoming_exposure_weight")),
        "network_exposure_scored_exposure_weight": _safe_float(context.get("total_exposure_weight")),
        "network_exposure_peer_mean": _safe_float(context.get("peer_score_mean")),
        "network_exposure_peer_sd": _safe_float(context.get("peer_score_sd")),
        "network_exposure_peer_exposure_weighted_mean": _safe_float(context.get("exposure_weighted_peer_mean")),
    }


def _flatten_post_attack_network_context(context: dict[str, Any]) -> dict[str, object]:
    return {
        "post_attack_network_full_incoming_peer_count": _safe_float(context.get("full_incoming_peer_count")),
        "post_attack_network_scored_same_condition_peer_count": _safe_float(context.get("peer_count")),
        "post_attack_network_exemplar_count": _safe_float(context.get("exemplar_count")),
        "post_attack_network_full_incoming_exposure_weight": _safe_float(context.get("full_incoming_exposure_weight")),
        "post_attack_network_scored_exposure_weight": _safe_float(context.get("total_exposure_weight")),
        "post_attack_network_peer_post_mean": _safe_float(context.get("peer_post_mean")),
        "post_attack_network_peer_delta_mean": _safe_float(context.get("peer_delta_mean")),
        "post_attack_network_peer_exposure_weighted_post_mean": _safe_float(
            context.get("exposure_weighted_peer_post_mean")
        ),
        "post_attack_network_peer_exposure_weighted_delta_mean": _safe_float(
            context.get("exposure_weighted_peer_delta_mean")
        ),
    }


def _directed_effect(value: float | None, adversarial_direction: int | None) -> float | None:
    if value is None or adversarial_direction in (None, 0):
        return None
    return float(value * adversarial_direction)


def _add_exposure_role_dummies(df: pd.DataFrame) -> pd.DataFrame:
    if "exposure_display_role" not in df.columns:
        return df
    encoded = pd.get_dummies(df["exposure_display_role"].fillna("unknown"), prefix="exposure_role", dtype=float)
    return pd.concat([df, encoded], axis=1)


def _add_exposure_sender_reach_share(df: pd.DataFrame) -> pd.DataFrame:
    reach_col = None
    if "exposure_outgoing_visibility_weight" in df.columns:
        reach_col = "exposure_outgoing_visibility_weight"
    elif "exposure_weighted_out_degree" in df.columns:
        reach_col = "exposure_weighted_out_degree"
    if reach_col is None:
        df["exposure_sender_reach_share"] = None
        return df
    profile_reach = df.groupby("profile_id")[reach_col].first().fillna(0.0).astype(float)
    total_reach = float(profile_reach.sum())
    if total_reach <= 0:
        df["exposure_sender_reach_share"] = 0.0
        return df
    shares = (profile_reach / total_reach).to_dict()
    df["exposure_sender_reach_share"] = df["profile_id"].map(shares).astype(float)
    return df


def _add_zscores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[f"{column}_z"] = zscore_series(df[column].fillna(0.0).astype(float))
    return df


def _mean_if_present(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = df[column].dropna()
    return float(values.mean()) if len(values) else None


def _centrality_weighted_profile_mean(df: pd.DataFrame, value_col: str) -> float | None:
    if "exposure_sender_reach_share" not in df.columns or value_col not in df.columns:
        return None
    profile_values = (
        df[["profile_id", "exposure_sender_reach_share", value_col]]
        .dropna(subset=["exposure_sender_reach_share", value_col])
        .groupby("profile_id", as_index=False)
        .agg({"exposure_sender_reach_share": "first", value_col: "mean"})
    )
    if profile_values.empty:
        return None
    return float((profile_values["exposure_sender_reach_share"] * profile_values[value_col]).sum())


def _centrality_weighted_private_resistance(df: pd.DataFrame) -> float | None:
    if "exposure_sender_reach_share" not in df.columns or "ae_private" not in df.columns:
        return None
    profile_values = (
        df[["profile_id", "exposure_sender_reach_share", "ae_private"]]
        .dropna(subset=["exposure_sender_reach_share", "ae_private"])
        .groupby("profile_id", as_index=False)
        .agg({"exposure_sender_reach_share": "first", "ae_private": "mean"})
    )
    if profile_values.empty:
        return None
    resistant = profile_values["ae_private"].clip(upper=0.0)
    return float((profile_values["exposure_sender_reach_share"] * resistant).sum())


def _profile_level_rollup(df_encoded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_columns = [
        column
        for column in df_encoded.columns
        if column.startswith("profile_cont_") or column.startswith("profile_cat__")
    ]
    exposure_stable_columns = [
        column
        for column in df_encoded.columns
        if column in EXPOSURE_STABLE_COLUMNS or column.startswith("exposure_role_")
    ]

    aggregate_spec: Dict[str, str] = {
        "baseline_score": "mean",
        "post_score": "mean",
        "delta_score": "mean",
        "abs_delta_score": "mean",
        "baseline_abs_score": "mean",
        "exposure_quality_score": "mean",
        "attack_realism_score": "mean",
        "attack_coherence_score": "mean",
        "post_plausibility_score": "mean",
        "post_consistency_score": "mean",
        "network_exposure_score": "mean",
        "network_exposure_delta_score": "mean",
        "network_exposure_abs_delta_score": "mean",
        "network_exposure_confidence": "mean",
        "network_exposure_peer_mean": "mean",
        "network_exposure_peer_exposure_weighted_mean": "mean",
        "post_attack_network_score": "mean",
        "post_attack_network_delta_from_baseline": "mean",
        "post_attack_network_abs_delta_from_baseline": "mean",
        "post_attack_network_increment_from_private_post": "mean",
        "post_attack_network_abs_increment_from_private_post": "mean",
        "post_attack_network_adversarial_effectivity": "mean",
        "post_attack_network_increment_adversarial_effectivity": "mean",
        "post_attack_network_confidence": "mean",
        "post_attack_network_peer_post_mean": "mean",
        "post_attack_network_peer_delta_mean": "mean",
        "post_attack_network_peer_exposure_weighted_post_mean": "mean",
        "post_attack_network_peer_exposure_weighted_delta_mean": "mean",
        **NETWORK_CONTEXT_COLUMNS,
        **HYPOTHESIS_COLUMNS,
        "scenario_id": "count",
    }
    if "adversarial_effectivity" in df_encoded.columns:
        aggregate_spec["adversarial_effectivity"] = "mean"

    available_aggregates = {
        key: value for key, value in aggregate_spec.items() if key in df_encoded.columns
    }

    grouped = df_encoded.groupby("profile_id", as_index=False).agg(available_aggregates)
    rename_map = {
        "baseline_score": "mean_baseline_score",
        "post_score": "mean_post_score",
        "delta_score": "mean_signed_delta_score",
        "abs_delta_score": "mean_abs_delta_score",
        "baseline_abs_score": "mean_baseline_abs_score",
        "exposure_quality_score": "mean_exposure_quality_score",
        "attack_realism_score": "mean_attack_realism_score",
        "attack_coherence_score": "mean_attack_coherence_score",
        "post_plausibility_score": "mean_post_plausibility_score",
        "post_consistency_score": "mean_post_consistency_score",
        "network_exposure_score": "mean_network_exposure_score",
        "network_exposure_delta_score": "mean_network_exposure_delta_score",
        "network_exposure_abs_delta_score": "mean_network_exposure_abs_delta_score",
        "network_exposure_confidence": "mean_network_exposure_confidence",
        "network_exposure_peer_mean": "mean_network_exposure_peer_mean",
        "network_exposure_peer_exposure_weighted_mean": "mean_network_exposure_peer_exposure_weighted_mean",
        "post_attack_network_score": "mean_post_attack_network_score",
        "post_attack_network_delta_from_baseline": "mean_post_attack_network_delta_from_baseline",
        "post_attack_network_abs_delta_from_baseline": "mean_post_attack_network_abs_delta_from_baseline",
        "post_attack_network_increment_from_private_post": "mean_post_attack_network_increment_from_private_post",
        "post_attack_network_abs_increment_from_private_post": "mean_post_attack_network_abs_increment_from_private_post",
        "post_attack_network_adversarial_effectivity": "mean_post_attack_network_adversarial_effectivity",
        "post_attack_network_increment_adversarial_effectivity": "mean_post_attack_network_increment_adversarial_effectivity",
        "post_attack_network_confidence": "mean_post_attack_network_confidence",
        "post_attack_network_peer_post_mean": "mean_post_attack_network_peer_post_mean",
        "post_attack_network_peer_delta_mean": "mean_post_attack_network_peer_delta_mean",
        "post_attack_network_peer_exposure_weighted_post_mean": "mean_post_attack_network_peer_exposure_weighted_post_mean",
        "post_attack_network_peer_exposure_weighted_delta_mean": "mean_post_attack_network_peer_exposure_weighted_delta_mean",
        **{column: f"mean_{column}" for column in NETWORK_CONTEXT_COLUMNS},
        **{column: f"mean_{column}" for column in HYPOTHESIS_COLUMNS},
        "scenario_id": "n_attacked_opinion_leaves",
        "adversarial_effectivity": "mean_adversarial_effectivity",
    }
    grouped = grouped.rename(columns={k: v for k, v in rename_map.items() if k in grouped.columns})

    stable_columns = sorted(set(profile_columns + exposure_stable_columns))
    if stable_columns:
        profile_values = df_encoded.groupby("profile_id", as_index=False)[stable_columns].first()
        grouped = grouped.merge(profile_values, on="profile_id", how="left")

    leaf_key_col = "opinion_leaf_label"
    abs_pivot = df_encoded.pivot_table(
        index="profile_id",
        columns=leaf_key_col,
        values="abs_delta_score",
        aggfunc="mean",
    )
    signed_pivot = df_encoded.pivot_table(
        index="profile_id",
        columns=leaf_key_col,
        values="delta_score",
        aggfunc="mean",
    )

    abs_pivot = abs_pivot.rename(columns=lambda value: f"abs_delta_indicator__{_slugify(str(value))}")
    signed_pivot = signed_pivot.rename(columns=lambda value: f"signed_delta_indicator__{_slugify(str(value))}")

    wide = grouped.merge(abs_pivot.reset_index(), on="profile_id", how="left")
    wide = wide.merge(signed_pivot.reset_index(), on="profile_id", how="left")

    if "adversarial_effectivity" in df_encoded.columns:
        adv_pivot = df_encoded.pivot_table(
            index="profile_id",
            columns=leaf_key_col,
            values="adversarial_effectivity",
            aggfunc="mean",
        )
        adv_pivot = adv_pivot.rename(columns=lambda value: f"adversarial_delta_indicator__{_slugify(str(value))}")
        wide = wide.merge(adv_pivot.reset_index(), on="profile_id", how="left")

    if "mean_baseline_abs_score" in wide.columns:
        wide["mean_baseline_abs_score_z"] = zscore_series(wide["mean_baseline_abs_score"].astype(float))
    if "mean_exposure_quality_score" in wide.columns:
        wide["mean_exposure_quality_score_z"] = zscore_series(wide["mean_exposure_quality_score"].astype(float))
    if "mean_abs_delta_score" in wide.columns:
        wide["mean_abs_delta_score_z"] = zscore_series(wide["mean_abs_delta_score"].astype(float))
    if "mean_signed_delta_score" in wide.columns:
        wide["mean_signed_delta_score_z"] = zscore_series(wide["mean_signed_delta_score"].astype(float))
    if "mean_network_exposure_delta_score" in wide.columns:
        wide["mean_network_exposure_delta_score_z"] = zscore_series(
            wide["mean_network_exposure_delta_score"].astype(float)
        )
    if "mean_network_exposure_abs_delta_score" in wide.columns:
        wide["mean_network_exposure_abs_delta_score_z"] = zscore_series(
            wide["mean_network_exposure_abs_delta_score"].astype(float)
        )
    if "mean_post_attack_network_increment_from_private_post" in wide.columns:
        wide["mean_post_attack_network_increment_from_private_post_z"] = zscore_series(
            wide["mean_post_attack_network_increment_from_private_post"].astype(float)
        )
    if "mean_post_attack_network_abs_increment_from_private_post" in wide.columns:
        wide["mean_post_attack_network_abs_increment_from_private_post_z"] = zscore_series(
            wide["mean_post_attack_network_abs_increment_from_private_post"].astype(float)
        )
    if "mean_adversarial_effectivity" in wide.columns:
        wide["mean_adversarial_effectivity_z"] = zscore_series(wide["mean_adversarial_effectivity"].astype(float))

    indicator_columns = [column for column in wide.columns if column.startswith("abs_delta_indicator__")]
    for column in indicator_columns:
        wide[f"{column}_z"] = zscore_series(wide[column].astype(float))

    adv_indicator_columns = [column for column in wide.columns if column.startswith("adversarial_delta_indicator__")]
    for column in adv_indicator_columns:
        wide[f"{column}_z"] = zscore_series(wide[column].astype(float))

    return grouped, wide


def run_stage(input_path: str, output_dir: str, config: Stage05Config) -> StageArtifactManifest:
    ensure_dir(output_dir)
    rows = read_jsonl(input_path)

    adversarial_directions = _load_adversarial_directions(config.ontology_root)
    has_adversarial = bool(adversarial_directions)

    deltas: List[DeltaRecord] = []
    sem_rows: List[SemRow] = []
    flat_rows: List[Dict[str, object]] = []

    for row in rows:
        scenario = ScenarioRecord.model_validate(
            {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "baseline_assessment",
                    "attack_exposure",
                    "attack_vector_spec",
                    "post_attack_assessment",
                    "network_exposure_assessment",
                    "network_exposure_context",
                    "network_exposure_coherence_review",
                    "network_exposure_heuristic_checks",
                    "post_attack_network_exposure_assessment",
                    "post_attack_network_exposure_context",
                    "post_attack_network_exposure_coherence_review",
                    "post_attack_network_exposure_heuristic_checks",
                    "post_attack_network_exposure_increment_score",
                    "post_attack_network_exposure_delta_from_baseline",
                    "post_attack_network_exposure_skipped",
                    "post_attack_network_exposure_skip_reason",
                }
            }
        )
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])
        post = OpinionAssessment.model_validate(row["post_attack_assessment"])
        network_exposure = (
            OpinionAssessment.model_validate(row["network_exposure_assessment"])
            if isinstance(row, dict) and isinstance(row.get("network_exposure_assessment"), dict)
            else None
        )
        network_context = (
            row.get("network_exposure_context", {})
            if isinstance(row, dict) and isinstance(row.get("network_exposure_context"), dict)
            else {}
        )
        post_attack_network = (
            OpinionAssessment.model_validate(row["post_attack_network_exposure_assessment"])
            if isinstance(row, dict) and isinstance(row.get("post_attack_network_exposure_assessment"), dict)
            else None
        )
        post_attack_network_context = (
            row.get("post_attack_network_exposure_context", {})
            if isinstance(row, dict) and isinstance(row.get("post_attack_network_exposure_context"), dict)
            else {}
        )
        # run_11 final design: stage 03 emits a deterministic attack-vector
        # specification instead of a generated exposure artifact. The tier-based
        # intensity proxy replaces the old per-message intensity hint; legacy
        # rows with a generated exposure are still readable.
        spec = row.get("attack_vector_spec") if isinstance(row, dict) else None
        spec = spec if isinstance(spec, dict) else {}
        if spec:
            exposure_intensity = float(spec.get("intensity_proxy", 0.5) or 0.0)
        elif isinstance(row, dict) and isinstance(row.get("attack_exposure"), dict):
            exposure_intensity = float(row["attack_exposure"].get("intensity_hint", 0.5) or 0.0)
        else:
            exposure_intensity = 0.5
        review = row.get("attack_realism_review", {}) if isinstance(row, dict) else {}
        heuristics = row.get("attack_heuristic_checks", {}) if isinstance(row, dict) else {}
        baseline_review = row.get("baseline_coherence_review", {}) if isinstance(row, dict) else {}
        baseline_heuristics = row.get("baseline_heuristic_checks", {}) if isinstance(row, dict) else {}
        post_review = row.get("post_coherence_review", {}) if isinstance(row, dict) else {}
        post_heuristics = row.get("post_heuristic_checks", {}) if isinstance(row, dict) else {}

        signed_delta = int(post.score - baseline.score)
        abs_delta = int(abs(signed_delta))

        # Adversarially aligned effectivity: positive = adversary achieved goal for this leaf.
        # Direction is +1 if adversary wants score to increase, -1 if adversary wants decrease.
        # Leaves absent from the direction map are direction-neutral (0): no adversarial
        # goal exists for them, so AE is undefined rather than defaulted to +1. The old
        # +1 default silently converted neutral-leaf drift into spurious negative AE
        # that read as backfire downstream.
        leaf_label = extract_leaf_label(scenario.opinion_leaf)
        adv_direction: int = adversarial_directions.get(leaf_label, 0)
        adversarial_eff: Optional[float] = (
            float(signed_delta * adv_direction) if (has_adversarial and adv_direction != 0) else None
        )

        delta_record = DeltaRecord(
            scenario_id=scenario.scenario_id,
            opinion_leaf=scenario.opinion_leaf,
            baseline_score=baseline.score,
            post_score=post.score,
            delta_score=signed_delta,
            abs_delta_score=abs_delta,
            adversarial_effectivity=adversarial_eff,
            attack_present=scenario.attack_present,
            attack_leaf=scenario.attack_leaf,
            profile_id=scenario.profile.profile_id,
            profile_categorical=scenario.profile.categorical_attributes,
            profile_continuous=scenario.profile.continuous_attributes,
        )
        deltas.append(delta_record)

        features = {
            **{f"profile_cont_{k}": float(v) for k, v in scenario.profile.continuous_attributes.items()},
            **{f"profile_cat_{k}": v for k, v in scenario.profile.categorical_attributes.items()},
        }

        flat_row: Dict[str, object] = {
            "scenario_id": scenario.scenario_id,
            "opinion_leaf": scenario.opinion_leaf,
            "opinion_domain": extract_opinion_domain(scenario.opinion_leaf),
            "opinion_leaf_label": extract_leaf_label(scenario.opinion_leaf),
            "attack_present": int(scenario.attack_present),
            "attack_leaf": scenario.attack_leaf or "CONTROL_NONE",
            "attack_leaf_label": extract_leaf_label(scenario.attack_leaf) if scenario.attack_leaf else "CONTROL_NONE",
            "baseline_score": float(baseline.score),
            "post_score": float(post.score),
            "delta_score": float(signed_delta),
            "abs_delta_score": float(abs_delta),
            "adversarial_effectivity": adversarial_eff,
            "adversarial_direction": adv_direction if has_adversarial else None,
            "profile_id": scenario.profile.profile_id,
            "exposure_intensity_hint": exposure_intensity,
            "attack_complexity_tier": (
                str((spec.get("attack_context") or {}).get("attack_complexity_tier", ""))
                if spec else ""
            ),
            "post_confidence": float(getattr(post, "confidence", 0.0) or 0.0),
            "baseline_confidence": float(getattr(baseline, "confidence", 0.0) or 0.0),
            "attack_realism_score": review.get("realism_score"),
            "attack_coherence_score": review.get("coherence_score"),
            "attack_rewrite_required": review.get("rewrite_required"),
            "attack_heuristic_pass": (
                heuristics.get("checks", {}).get("overall_pass")
                if isinstance(heuristics, dict)
                else None
            ),
            "baseline_plausibility_score": baseline_review.get("plausibility_score"),
            "baseline_consistency_score": baseline_review.get("consistency_score"),
            "baseline_rewrite_required": baseline_review.get("rewrite_required"),
            "baseline_heuristic_pass": (
                baseline_heuristics.get("checks", {}).get("overall_pass")
                if isinstance(baseline_heuristics, dict)
                else None
            ),
            "post_plausibility_score": post_review.get("plausibility_score"),
            "post_consistency_score": post_review.get("consistency_score"),
            "post_rewrite_required": post_review.get("rewrite_required"),
            "post_heuristic_pass": (
                post_heuristics.get("checks", {}).get("overall_pass")
                if isinstance(post_heuristics, dict)
                else None
            ),
            "baseline_fallback_used": baseline.model_name == "fallback_deterministic",
            "post_fallback_used": post.model_name == "fallback_deterministic",
            "post_direction_clamped": bool(row.get("post_direction_clamped", False)) if isinstance(row, dict) else False,
            "scenario_design": scenario.metadata.get("scenario_design"),
            "profile_panel_index": scenario.metadata.get("profile_panel_index"),
            "leaf_repeat_index_within_profile": scenario.metadata.get("leaf_repeat_index_within_profile"),
            "ae_private": adversarial_eff,
        }
        flat_row.update(_flatten_exposure_assignment(scenario))
        flat_row.update(_flatten_baseline_network_context(network_context))
        flat_row.update(_flatten_post_attack_network_context(post_attack_network_context))
        flat_row["peer_private_attack_activation"] = _directed_effect(
            _safe_float(post_attack_network_context.get("exposure_weighted_peer_delta_mean")),
            adv_direction if has_adversarial else None,
        )
        if isinstance(row, dict) and "post_attack_network_exposure_skipped" in row:
            flat_row["post_attack_network_skipped"] = bool(row.get("post_attack_network_exposure_skipped", False))
        if network_exposure is not None:
            network_delta = int(network_exposure.score - baseline.score)
            flat_row.update(
                {
                    "network_exposure_score": float(network_exposure.score),
                    "network_exposure_delta_score": float(network_delta),
                    "network_exposure_abs_delta_score": float(abs(network_delta)),
                    "bn_increment": float(network_delta),
                    "network_exposure_confidence": float(getattr(network_exposure, "confidence", 0.0) or 0.0),
                    "network_exposure_fallback_used": network_exposure.model_name == "fallback_deterministic",
                    "network_exposure_peer_mean": network_context.get("peer_score_mean"),
                    "network_exposure_peer_exposure_weighted_mean": network_context.get(
                        "exposure_weighted_peer_mean"
                    ),
                }
            )
        if post_attack_network is not None:
            post_attack_network_delta = int(post_attack_network.score - baseline.score)
            post_attack_network_increment = int(post_attack_network.score - post.score)
            post_attack_network_adv_eff = (
                float(post_attack_network_delta * adv_direction)
                if (has_adversarial and adv_direction != 0)
                else None
            )
            post_attack_network_increment_adv_eff = (
                float(post_attack_network_increment * adv_direction)
                if (has_adversarial and adv_direction != 0)
                else None
            )
            flat_row.update(
                {
                    "post_attack_network_score": float(post_attack_network.score),
                    "post_attack_network_delta_from_baseline": float(post_attack_network_delta),
                    "post_attack_network_abs_delta_from_baseline": float(abs(post_attack_network_delta)),
                    "post_attack_network_increment_from_private_post": float(post_attack_network_increment),
                    "post_attack_network_abs_increment_from_private_post": float(abs(post_attack_network_increment)),
                    "pn_increment": float(post_attack_network_increment),
                    "ae_total_network": post_attack_network_adv_eff,
                    "pn_increment_effectivity": post_attack_network_increment_adv_eff,
                    "post_attack_network_adversarial_effectivity": post_attack_network_adv_eff,
                    "post_attack_network_increment_adversarial_effectivity": post_attack_network_increment_adv_eff,
                    "post_attack_network_confidence": float(getattr(post_attack_network, "confidence", 0.0) or 0.0),
                    "post_attack_network_fallback_used": post_attack_network.model_name == "fallback_deterministic",
                    "post_attack_network_peer_post_mean": post_attack_network_context.get("peer_post_mean"),
                    "post_attack_network_peer_delta_mean": post_attack_network_context.get("peer_delta_mean"),
                    "post_attack_network_peer_exposure_weighted_post_mean": post_attack_network_context.get(
                        "exposure_weighted_peer_post_mean"
                    ),
                    "post_attack_network_peer_exposure_weighted_delta_mean": post_attack_network_context.get(
                        "exposure_weighted_peer_delta_mean"
                    ),
                }
            )
        flat_row.update(features)
        flat_rows.append(flat_row)

        numeric_profile = {
            k: float(v)
            for k, v in features.items()
            if k.startswith("profile_cont_")
        }
        sem_rows.append(
            SemRow(
                scenario_id=scenario.scenario_id,
                opinion_leaf=scenario.opinion_leaf,
                baseline_score=float(baseline.score),
                post_score=float(post.score),
                delta_score=float(signed_delta),
                abs_delta_score=float(abs_delta),
                adversarial_effectivity=adversarial_eff,
                attack_present=int(scenario.attack_present),
                attack_leaf=scenario.attack_leaf or "CONTROL_NONE",
                profile_id=scenario.profile.profile_id,
                profile_features=numeric_profile,
            )
        )

    df_raw = pd.DataFrame(flat_rows)
    df_encoded = one_hot_profile_categoricals(df_raw.copy())
    df_encoded = _add_exposure_role_dummies(df_encoded)
    df_encoded = _add_exposure_sender_reach_share(df_encoded)
    df_encoded = _add_zscores(
        df_encoded,
        [
            "exposure_weighted_in_degree",
            "exposure_outgoing_visibility_weight",
            "exposure_bridge_score",
            "exposure_eigenvector_centrality",
            "exposure_cascade_reach_potential",
            "exposure_h2_neighborhood_activation_readiness",
            "exposure_h3_central_susceptible_sender_readiness",
            "exposure_h4_central_resilient_sender_dampening_capacity",
        ],
    )
    df_encoded["baseline_abs_score"] = df_encoded["baseline_score"].abs().astype(float)
    df_encoded["baseline_extremity_norm"] = df_encoded["baseline_abs_score"] / 1000.0
    quality_columns = [
        column
        for column in ["exposure_intensity_hint", "attack_realism_score", "attack_coherence_score"]
        if column in df_encoded.columns
    ]
    if quality_columns:
        df_encoded["exposure_quality_score"] = (
            df_encoded[quality_columns]
            .astype(float)
            .mean(axis=1, skipna=True)
            .fillna(df_encoded["exposure_intensity_hint"].astype(float) if "exposure_intensity_hint" in df_encoded.columns else 0.5)
        )
    else:
        df_encoded["exposure_quality_score"] = 0.5
    df_encoded["exposure_quality_z"] = zscore_series(df_encoded["exposure_quality_score"].astype(float))
    df_encoded, reference_leaf = _add_fixed_effects(df_encoded, "opinion_leaf", "opinion_leaf_fe")
    df_encoded, reference_domain = _add_fixed_effects(df_encoded, "opinion_domain", "opinion_domain_fe")

    moderator_col = choose_primary_moderator_column(df_encoded, preferred=config.primary_moderator)
    df_encoded["primary_moderator_value"] = df_encoded[moderator_col].astype(float)
    df_encoded["primary_moderator_z"] = zscore_series(df_encoded[moderator_col].astype(float))

    profile_summary_df, profile_wide_df = _profile_level_rollup(df_encoded)

    delta_jsonl = Path(output_dir) / "effectivity_deltas.jsonl"
    sem_rows_jsonl = Path(output_dir) / "sem_long_rows.jsonl"
    sem_raw_csv = Path(output_dir) / "sem_long_raw.csv"
    sem_encoded_csv = Path(output_dir) / "sem_long_encoded.csv"
    sem_encoded_jsonl = Path(output_dir) / "sem_long_encoded.jsonl"
    profile_summary_csv = Path(output_dir) / "profile_level_effectivity.csv"
    profile_wide_csv = Path(output_dir) / "profile_sem_wide.csv"
    summary_json = Path(output_dir) / "delta_summary.json"

    write_jsonl(delta_jsonl, (x.model_dump() for x in deltas))
    write_jsonl(sem_rows_jsonl, (x.model_dump() for x in sem_rows))
    df_raw.to_csv(sem_raw_csv, index=False)
    df_encoded.to_csv(sem_encoded_csv, index=False)
    write_jsonl(sem_encoded_jsonl, df_encoded.to_dict(orient="records"))
    profile_summary_df.to_csv(profile_summary_csv, index=False)
    profile_wide_df.to_csv(profile_wide_csv, index=False)

    summary_payload: Dict[str, object] = {
        "n_records": len(deltas),
        "n_profiles": int(df_encoded["profile_id"].nunique()),
        "analysis_mode": (
            "treated_only"
            if len(df_encoded) and int(df_encoded["attack_present"].min()) == 1 and int(df_encoded["attack_present"].max()) == 1
            else "mixed_condition"
        ),
        "mean_signed_delta": float(df_encoded["delta_score"].mean()),
        "std_signed_delta": float(df_encoded["delta_score"].std(ddof=0)),
        "mean_abs_delta": float(df_encoded["abs_delta_score"].mean()),
        "std_abs_delta": float(df_encoded["abs_delta_score"].std(ddof=0)),
        "primary_moderator_column": moderator_col,
        "reference_opinion_leaf": reference_leaf,
        "reference_opinion_domain": reference_domain,
        "n_unique_opinion_leaves": int(df_encoded["opinion_leaf"].nunique()),
        "scenarios_per_profile": float(df_encoded.groupby("profile_id")["scenario_id"].count().mean()),
        "attack_present_count": int(df_encoded["attack_present"].sum()),
        "control_count": int((1 - df_encoded["attack_present"]).sum()),
        "exposure_quality_mean": float(df_encoded["exposure_quality_score"].mean()),
        "adversarial_manifest_loaded": has_adversarial,
    }
    if has_adversarial and "adversarial_effectivity" in df_encoded.columns:
        adv_vals = df_encoded["adversarial_effectivity"].dropna()
        summary_payload["mean_adversarial_effectivity"] = float(adv_vals.mean()) if len(adv_vals) else None
        summary_payload["std_adversarial_effectivity"] = float(adv_vals.std(ddof=0)) if len(adv_vals) else None
        summary_payload["adversarial_effectivity_positive_pct"] = float((adv_vals > 0).mean() * 100.0) if len(adv_vals) else None
    if "network_exposure_delta_score" in df_encoded.columns:
        summary_payload["mean_network_exposure_delta_score"] = float(df_encoded["network_exposure_delta_score"].mean())
        summary_payload["mean_network_exposure_abs_delta_score"] = float(
            df_encoded["network_exposure_abs_delta_score"].mean()
        )
        summary_payload["network_exposure_fallback_count"] = int(
            df_encoded["network_exposure_fallback_used"].fillna(False).astype(bool).sum()
        )
        summary_payload["network_exposure_mean_full_incoming_peer_count"] = _mean_if_present(
            df_encoded, "network_exposure_full_incoming_peer_count"
        )
        summary_payload["network_exposure_mean_scored_peer_count"] = _mean_if_present(
            df_encoded, "network_exposure_scored_peer_count"
        )
        summary_payload["network_exposure_mean_exemplar_count"] = _mean_if_present(
            df_encoded, "network_exposure_exemplar_count"
        )
        summary_payload["network_exposure_mean_full_incoming_exposure_weight"] = _mean_if_present(
            df_encoded, "network_exposure_full_incoming_exposure_weight"
        )
        summary_payload["network_exposure_mean_scored_exposure_weight"] = _mean_if_present(
            df_encoded, "network_exposure_scored_exposure_weight"
        )
    if "post_attack_network_increment_from_private_post" in df_encoded.columns:
        summary_payload["mean_post_attack_network_increment_from_private_post"] = float(
            df_encoded["post_attack_network_increment_from_private_post"].mean()
        )
        summary_payload["mean_post_attack_network_abs_increment_from_private_post"] = float(
            df_encoded["post_attack_network_abs_increment_from_private_post"].mean()
        )
        summary_payload["post_attack_network_fallback_count"] = int(
            df_encoded["post_attack_network_fallback_used"].fillna(False).astype(bool).sum()
        )
        summary_payload["post_attack_network_skipped_count"] = int(
            df_encoded.get("post_attack_network_skipped", False).fillna(False).astype(bool).sum()
            if "post_attack_network_skipped" in df_encoded.columns
            else 0
        )
        summary_payload["post_attack_network_mean_full_incoming_peer_count"] = _mean_if_present(
            df_encoded, "post_attack_network_full_incoming_peer_count"
        )
        summary_payload["post_attack_network_mean_scored_same_condition_peer_count"] = _mean_if_present(
            df_encoded, "post_attack_network_scored_same_condition_peer_count"
        )
        summary_payload["post_attack_network_mean_exemplar_count"] = _mean_if_present(
            df_encoded, "post_attack_network_exemplar_count"
        )
        summary_payload["post_attack_network_mean_full_incoming_exposure_weight"] = _mean_if_present(
            df_encoded, "post_attack_network_full_incoming_exposure_weight"
        )
        summary_payload["post_attack_network_mean_scored_exposure_weight"] = _mean_if_present(
            df_encoded, "post_attack_network_scored_exposure_weight"
        )
    summary_payload["centrality_weighted_private_activation"] = _centrality_weighted_profile_mean(
        df_encoded, "ae_private"
    )
    summary_payload["centrality_weighted_total_network_effect"] = _centrality_weighted_profile_mean(
        df_encoded, "ae_total_network"
    )
    summary_payload["centrality_weighted_post_network_increment"] = _centrality_weighted_profile_mean(
        df_encoded, "pn_increment_effectivity"
    )
    summary_payload["centrality_weighted_private_resistance"] = _centrality_weighted_private_resistance(df_encoded)
    write_json(summary_json, summary_payload)

    manifest = StageArtifactManifest(
        stage_id="05",
        stage_name="compute_effectivity_deltas",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(sem_encoded_csv),
        output_files=[
            abs_path(delta_jsonl),
            abs_path(sem_rows_jsonl),
            abs_path(sem_raw_csv),
            abs_path(sem_encoded_csv),
            abs_path(sem_encoded_jsonl),
            abs_path(profile_summary_csv),
            abs_path(profile_wide_csv),
            abs_path(summary_json),
        ],
        record_count=len(deltas),
        metadata={
            "primary_moderator_column": moderator_col,
            "reference_opinion_leaf": reference_leaf,
            "reference_opinion_domain": reference_domain,
            "n_unique_opinion_leaves": int(df_encoded["opinion_leaf"].nunique()),
            "n_profiles": int(df_encoded["profile_id"].nunique()),
            "analysis_mode": (
                "treated_only"
                if len(df_encoded) and int(df_encoded["attack_present"].min()) == 1 and int(df_encoded["attack_present"].max()) == 1
                else "mixed_condition"
            ),
            "effectivity_outcome": "adversarial_effectivity_primary_abs_shift_secondary" if has_adversarial else "absolute_shift_primary_signed_shift_secondary",
            "adversarial_manifest_loaded": has_adversarial,
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 05 - Compute effectivity deltas")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-moderator", default="profile_cont_age_years")
    parser.add_argument("--ontology-root", default=None, help="Path to ontology root; used to load adversarial_manifest.json")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage05Config(
        stage_name="compute_effectivity_deltas",
        run_id=args.run_id,
        seed=args.seed,
        primary_moderator=args.primary_moderator,
        ontology_root=args.ontology_root,
    )
    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 05 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
