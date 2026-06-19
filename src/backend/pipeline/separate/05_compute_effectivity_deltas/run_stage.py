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
wide table, each profile receives 01_separated attacked outcome indicators for each
opinion leaf, which enables repeated-outcome SEM/path modeling rather than a
premature collapse to a single summary score.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from src.backend.utils.scenario.scenario_realism import extract_leaf_label, extract_opinion_domain
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


def _profile_level_rollup(df_encoded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_columns = [
        column
        for column in df_encoded.columns
        if column.startswith("profile_cont_") or column.startswith("profile_cat__")
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
        "scenario_id": "n_attacked_opinion_leaves",
        "adversarial_effectivity": "mean_adversarial_effectivity",
    }
    grouped = grouped.rename(columns={k: v for k, v in rename_map.items() if k in grouped.columns})

    if profile_columns:
        profile_values = df_encoded.groupby("profile_id", as_index=False)[profile_columns].first()
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
    if "mean_adversarial_effectivity" in wide.columns:
        wide["mean_adversarial_effectivity_z"] = zscore_series(wide["mean_adversarial_effectivity"].astype(float))

    indicator_columns = [column for column in wide.columns if column.startswith("abs_delta_indicator__")]
    for column in indicator_columns:
        wide[f"{column}_z"] = zscore_series(wide[column].astype(float))

    adv_indicator_columns = [column for column in wide.columns if column.startswith("adversarial_delta_indicator__")]
    for column in adv_indicator_columns:
        wide[f"{column}_z"] = zscore_series(wide[column].astype(float))

    return grouped, wide


def _expand_cluster_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Expand cluster-batched scenarios into the standard per-leaf legacy rows.

    Integrated-scenario runs assess a whole issue-domain cluster in one call
    (one baseline + one post per scenario). Here we fan each scenario back out
    into one legacy-shaped row per leaf, carrying that leaf's baseline score,
    post score, and baked adversarial direction. After this transform the rest
    of Stage 05 (and Stages 06/07/08) operate on the identical per-leaf long
    table they always have, so the analysis and visualisation structure is
    unchanged.
    """
    expanded: List[Dict[str, object]] = []
    for row in rows:
        if not (isinstance(row, dict) and "post_cluster_assessment" in row):
            expanded.append(row)
            continue
        scenario = ScenarioRecord.model_validate(
            {
                k: v
                for k, v in row.items()
                if k not in {
                    "baseline_cluster_assessment", "baseline_cluster_heuristics",
                    "post_cluster_assessment", "post_cluster_heuristics", "post_cluster_clamped_leaves",
                    "attack_vector_spec",
                }
            }
        )
        if scenario.opinion_cluster is None:
            expanded.append(row)
            continue
        base_meta = dict(scenario.metadata or {})
        spec = row.get("attack_vector_spec", {})
        baseline_cluster = row.get("baseline_cluster_assessment", {}) or {}
        post_cluster = row.get("post_cluster_assessment", {}) or {}
        b_by_leaf = {str(ls.get("leaf")): ls for ls in baseline_cluster.get("leaf_scores", [])}
        p_by_leaf = {str(ls.get("leaf")): ls for ls in post_cluster.get("leaf_scores", [])}
        baseline_model = baseline_cluster.get("model_name", "")
        post_model = post_cluster.get("model_name", "")
        profile_dump = scenario.profile.model_dump()

        for leaf_idx, leaf in enumerate(scenario.opinion_cluster.leaves, start=1):
            b = b_by_leaf.get(leaf.leaf)
            p = p_by_leaf.get(leaf.leaf)
            if b is None or p is None:
                continue
            leaf_meta = dict(base_meta)
            # Authoritative per-leaf adversarial direction for this row.
            leaf_meta["opinion_adversarial_direction"] = int(leaf.adversarial_direction)
            leaf_meta["leaf_repeat_index_within_profile"] = leaf_idx
            leaf_meta["opinion_cluster_key"] = scenario.opinion_cluster.key
            expanded.append(
                {
                    "scenario_id": f"{scenario.scenario_id}__{leaf.leaf}",
                    "scenario_index": scenario.scenario_index,
                    "random_seed": scenario.random_seed,
                    "profile": profile_dump,
                    "opinion_leaf": leaf.path,
                    "opinion_cluster": None,
                    "attack_present": scenario.attack_present,
                    "attack_leaf": scenario.attack_leaf,
                    "attack_primary_node": scenario.attack_primary_node,
                    "metadata": leaf_meta,
                    "attack_vector_spec": spec,
                    "baseline_assessment": {
                        "scenario_id": scenario.scenario_id,
                        "phase": "baseline",
                        "opinion_leaf": leaf.path,
                        "score": int(b.get("score", 0)),
                        "confidence": float(b.get("confidence", 0.5) or 0.5),
                        "reasoning": str(b.get("reasoning", "")),
                        "model_name": baseline_model,
                    },
                    "post_attack_assessment": {
                        "scenario_id": scenario.scenario_id,
                        "phase": "post_attack",
                        "opinion_leaf": leaf.path,
                        "score": int(p.get("score", 0)),
                        "confidence": float(p.get("confidence", 0.5) or 0.5),
                        "reasoning": str(p.get("reasoning", "")),
                        "model_name": post_model,
                    },
                    "post_direction_clamped": False,
                }
            )
    return expanded


def _augment_with_network_exposure(
    input_path: str,
    output_dir: str,
    rows: List[Dict[str, object]],
    adversarial_directions: Dict[str, int],
) -> List[str]:
    """Additive empirical exposure-network analysis variables (network layer).

    Fires only when the optional stages 02b / 04b have produced outputs in the
    sibling stage-output dirs next to the stage-04 input. For the core
    individual-layer run (network exposure OFF) those files are absent and this
    returns [] without touching any standard stage-05 table. The caller wraps
    this in a guard so a failure here can never corrupt the core artifacts.

    Emits, per (profile_id, opinion_leaf):
        B  = private baseline           (stage 02)
        BN = network-exposure baseline  (stage 02b)
        P  = private post-attack        (stage 04)
        PN = network-exposure post      (stage 04b)
    plus the hypothesis quantities ae_private, bn_increment, pn_increment,
    pn_increment_effectivity and ae_total_network.
    """
    stage_outputs_root = Path(input_path).resolve().parent.parent
    bn_path = stage_outputs_root / "02b_assess_network_exposure_opinions" / "network_exposure_assessments.jsonl"
    pn_path = (
        stage_outputs_root
        / "04b_assess_post_attack_network_exposure_opinions"
        / "post_attack_network_exposure_assessments.jsonl"
    )
    if not bn_path.exists() and not pn_path.exists():
        return []

    private: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        try:
            profile = row.get("profile") or {}
            profile_id = str(profile.get("profile_id") or profile.get("id") or "")
            leaf = str(row.get("opinion_leaf") or "")
            b = int((row.get("baseline_assessment") or {}).get("score"))
            p = int((row.get("post_attack_assessment") or {}).get("score"))
            # Authoritative per-leaf adversarial direction comes from the expanded
            # row metadata (set from the opinion cluster); fall back to the manifest.
            meta = row.get("metadata") or {}
            d = meta.get("opinion_adversarial_direction")
            if d is None:
                d = adversarial_directions.get(leaf, 0)
            private[(profile_id, leaf)] = {
                "scenario_id": row.get("scenario_id"),
                "B": b,
                "P": p,
                "d": int(d or 0),
                "attack_present": bool(row.get("attack_present")),
            }
        except Exception:
            continue

    def _index(path: Path, assessment_key: str) -> Dict[tuple, int]:
        out: Dict[tuple, int] = {}
        if not path.exists():
            return out
        for rec in read_jsonl(str(path)):
            key = (str(rec.get("profile_id")), str(rec.get("opinion_leaf")))
            score = (rec.get(assessment_key) or {}).get("score")
            if score is not None:
                out[key] = int(score)
        return out

    bn_by_key = _index(bn_path, "network_exposure_assessment")
    pn_by_key = _index(pn_path, "post_attack_network_exposure_assessment")

    records: List[Dict[str, object]] = []
    for key, vals in private.items():
        d = int(vals["d"])
        b = int(vals["B"])
        p = int(vals["P"])
        bn = bn_by_key.get(key)
        pn = pn_by_key.get(key)
        records.append(
            {
                "scenario_id": vals["scenario_id"],
                "profile_id": key[0],
                "opinion_leaf": key[1],
                "attack_present": vals["attack_present"],
                "adversarial_direction": d,
                "B_private_baseline": b,
                "P_private_post": p,
                "BN_network_baseline": bn,
                "PN_network_post": pn,
                "ae_private": (p - b) * d if d else None,
                "bn_increment": (bn - b) if bn is not None else None,
                "pn_increment": (pn - p) if pn is not None else None,
                "pn_increment_effectivity": ((pn - p) * d) if (pn is not None and d) else None,
                "ae_total_network": ((pn - b) * d) if (pn is not None and d) else None,
            }
        )
    if not records:
        return []

    df = pd.DataFrame(records)
    long_csv = Path(output_dir) / "network_exposure_long.csv"
    summary_json = Path(output_dir) / "network_exposure_summary.json"
    df.to_csv(long_csv, index=False)

    def _mean(col: str) -> Optional[float]:
        series = df[col].dropna()
        return float(series.mean()) if len(series) else None

    write_json(
        summary_json,
        {
            "n_network_records": int(len(df)),
            "n_with_BN": int(df["BN_network_baseline"].notna().sum()),
            "n_with_PN": int(df["PN_network_post"].notna().sum()),
            "mean_ae_private": _mean("ae_private"),
            "mean_bn_increment": _mean("bn_increment"),
            "mean_pn_increment": _mean("pn_increment"),
            "mean_pn_increment_effectivity": _mean("pn_increment_effectivity"),
            "mean_ae_total_network": _mean("ae_total_network"),
        },
    )
    return [abs_path(long_csv), abs_path(summary_json)]


def run_stage(input_path: str, output_dir: str, config: Stage05Config) -> StageArtifactManifest:
    ensure_dir(output_dir)
    rows = read_jsonl(input_path)
    rows = _expand_cluster_rows(rows)

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
                }
            }
        )
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])
        post = OpinionAssessment.model_validate(row["post_attack_assessment"])
        # run_1 final design: stage 03 emits a deterministic attack-vector
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
        # Direction precedence: the per-leaf adversarial direction baked into the
        # scenario record (authoritative for the integrated design, and the same
        # rule stages 03/04 score against) takes priority over the ontology map.
        leaf_label = extract_leaf_label(scenario.opinion_leaf)
        meta_direction: Optional[int] = None
        if isinstance(scenario.metadata, dict):
            raw_dir = scenario.metadata.get("opinion_adversarial_direction")
            try:
                meta_direction = int(raw_dir) if raw_dir is not None else None
            except (TypeError, ValueError):
                meta_direction = None
        adv_direction: int = (
            meta_direction if meta_direction is not None else adversarial_directions.get(leaf_label, 0)
        )
        adversarial_eff: Optional[float] = (
            float(signed_delta * adv_direction) if adv_direction != 0 else None
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
            "adversarial_direction": adv_direction if adv_direction != 0 else None,
            "profile_id": scenario.profile.profile_id,
            "exposure_intensity_hint": exposure_intensity,
            "attack_complexity_tier": (
                str((spec.get("attack_context") or {}).get("attack_complexity_tier", ""))
                if spec and spec.get("attack_context")
                else str((scenario.metadata or {}).get("attack_complexity_tier", ""))
            ),
            # Raw DISARM attack covariates (from the integrated metadata, when present).
            "attack_signal_total": (scenario.metadata or {}).get("attack_signal_total"),
            "attack_inclusion_route": (scenario.metadata or {}).get("attack_inclusion_route"),
            "attack_execute_tactic": (scenario.metadata or {}).get("attack_execute_tactic"),
            "attack_plan_tactic": (scenario.metadata or {}).get("attack_plan_tactic"),
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
        }
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
    # Treat adversarial effectivity as available whenever any row carried a
    # non-neutral direction (baked per-leaf directions cover the integrated
    # design even when no ontology directions map was passed).
    if "adversarial_effectivity" in df_raw.columns and df_raw["adversarial_effectivity"].notna().any():
        has_adversarial = True
    df_encoded = one_hot_profile_categoricals(df_raw.copy())
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
    write_json(summary_json, summary_payload)

    # Additive empirical exposure-network analysis variables. No-op unless the
    # optional stages 02b / 04b ran (network exposure ON). Guarded so it can
    # never corrupt the core stage-05 tables consumed by stages 06-08.
    network_extra_outputs: List[str] = []
    try:
        network_extra_outputs = _augment_with_network_exposure(
            input_path, output_dir, rows, adversarial_directions
        )
        if network_extra_outputs:
            LOGGER.info(
                "Stage 05 emitted %s network-exposure analysis file(s).",
                len(network_extra_outputs),
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Stage 05 network-exposure augmentation skipped: %s", exc)
        network_extra_outputs = []

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
            *network_extra_outputs,
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
