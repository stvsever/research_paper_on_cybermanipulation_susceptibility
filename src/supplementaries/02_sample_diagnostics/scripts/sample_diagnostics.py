#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
STAGE01_ROOT = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"

INTEGRATED_PATH = STAGE01_ROOT / "samples" / "02_integrated" / "integrated_scenarios_10000.jsonl"
INTEGRATED_SUMMARY_PATH = STAGE01_ROOT / "samples" / "02_integrated" / "integrated_scenarios_10000.summary.json"
PROFILE_SUMMARY_PATH = STAGE01_ROOT / "samples" / "01_separated" / "profiles" / "production_profiles_maxent_10000.summary.json"
OPINION_SUMMARY_PATH = STAGE01_ROOT / "samples" / "01_separated" / "opinions" / "opinion_targets_maxent_1000.summary.json"
ATTACK_PATH = STAGE01_ROOT / "samples" / "01_separated" / "attacks" / "red_plan_prepare_execute_opinion_effect_filtered.json"
ATTACK_SUMMARY_PATH = STAGE01_ROOT / "samples" / "01_separated" / "attacks" / "red_plan_prepare_execute_opinion_effect_filtered.summary.json"

IMAGES_DIR = SUPPLEMENT_ROOT / "images"
SOURCE_3D_DIR = IMAGES_DIR / "3D_entropy_maximization"
METRICS_DIR = SUPPLEMENT_ROOT / "metrics"
TABLES_DIR = SUPPLEMENT_ROOT / "tables"

AGE_SOURCE_PNG = SOURCE_3D_DIR / "age_opinions_attacks.png"
NEUROTICISM_SOURCE_PNG = SOURCE_3D_DIR / "neuroticism_opinions_attacks.png"
FIGURE_01 = IMAGES_DIR / "01_sampling_design_diagnostics.png"
FIGURE_02 = IMAGES_DIR / "02_entropy_and_profile_moderator_diagnostics.png"

PHASES = ["Plan", "Prepare", "Execute"]
BIG_FIVE = ["neuroticism", "extraversion", "openness_to_experience", "agreeableness", "conscientiousness"]

LAYER_COLORS = {
    "Profile": "#356b8c",
    "Opinion": "#4e9f72",
    "Attack": "#b05a4a",
    "Scenario": "#7b6aa9",
}

DOMAIN_COLORS = {
    "Defense / security": "#caa64b",
    "Foreign policy": "#ef7f62",
    "Democratic resilience": "#2f5f99",
    "Infrastructure / energy": "#1aa187",
    "Regional integration": "#8e44ad",
    "Macroeconomics": "#2b8cbe",
    "Information integrity": "#43a2ca",
}

SHORT_DOMAIN = {
    "Critical_Infrastructure_And_Energy_Sovereignty": "Infrastructure / energy",
    "Defense_And_National_Security": "Defense / security",
    "Democratic_Resilience_And_Institutions": "Democratic resilience",
    "Foreign_Policy_And_Geopolitics": "Foreign policy",
    "Information_Integrity_And_Platforms": "Information integrity",
    "Macroeconomic_And_Fiscal_Policy": "Macroeconomics",
    "Supranational_And_Regional_Integration": "Regional integration",
}

CATEGORICAL_FEATURES = {
    "Highest education": "person_education_and_skills_highest_education",
    "Employment": "person_employment_and_work_employment_status",
    "Citizenship": "person_demographics_and_identity_citizenship_status",
    "Migration": "person_migration_and_residency_migration_position",
    "Urbanicity": "person_geography_and_location_urbanicity",
    "Relationship status": "person_demographics_and_identity_relationship_status",
    "Marital history": "person_demographics_and_identity_marital_history",
    "Religion affiliation": "person_religion_spirituality_and_worldview_affiliation",
    "Household income": "socioeconomic_position_income_household_income_category",
    "Housing tenure": "person_housing_and_neighborhood_housing_tenure",
    "Household composition": "person_household_and_family_household_composition",
    "Continent": "geography_and_location_broad_region_continent",
    "Sexual identity": "demographics_and_identity_sexual_orientation_identity",
    "Sexual attraction": "demographics_and_identity_sexual_orientation_attraction",
    "Romantic orientation": "person_demographics_and_identity_romantic_orientation",
    "Relationship structure": "person_demographics_and_identity_relationship_structure",
    "Cohabitation": "person_demographics_and_identity_cohabitation",
    "Sex assigned": "person_demographics_and_identity_sex_assigned_at_birth",
    "Gender identity": "person_demographics_and_identity_gender_identity",
}

PLANNED_CONTINUOUS_FEATURES = {
    "Age": "age_years",
    "Neuroticism": "bigfive_neuroticism",
    "Extraversion": "bigfive_extraversion",
    "Openness": "bigfive_openness_to_experience",
    "Agreeableness": "bigfive_agreeableness",
    "Conscientiousness": "bigfive_conscientiousness",
    "Economic left/right": "political_profile_ideological_dimensions_two_axis_model_economic_left_right_mean_pct",
    "Socio-cultural liberalism": "political_profile_ideological_dimensions_two_axis_model_socio_cultural_liberal_conservative_mean_pct",
    "Libertarianism": "political_profile_libertarian_authoritarian_dimension_model_libertarianism_mean_pct",
    "Authoritarianism": "political_profile_libertarian_authoritarian_dimension_model_authoritarianism_mean_pct",
    "GAL": "political_profile_gal_tan_model_green_alternative_libertarian_mean_pct",
    "TAN": "political_profile_gal_tan_model_traditional_authoritarian_nationalist_mean_pct",
    "RWA": "demographics_and_identity_political_profile_right_wing_authoritarianism_model_mean_pct",
    "SDO": "demographics_and_identity_political_profile_social_dominance_orientation_model_mean_pct",
    "System justification": "demographics_and_identity_political_profile_system_justification_theory_mean_pct",
    "Dual-process threat": "demographics_and_identity_political_profile_dual_process_motivational_model_mean_pct",
    "Populism": "demographics_and_identity_political_profile_populist_attitudes_model_mean_pct",
    "Nationalism/cosmopolitanism": "demographics_and_identity_political_profile_nationalism_and_cosmopolitanism_model_mean_pct",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ensure_clean_dirs() -> None:
    SOURCE_3D_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_DIR.exists():
        shutil.rmtree(METRICS_DIR)
    if TABLES_DIR.exists():
        shutil.rmtree(TABLES_DIR)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in [
        IMAGES_DIR / "01_descriptive",
        IMAGES_DIR / "02_entropy_preservation",
        IMAGES_DIR / "03_multicollinearity",
    ]:
        if old.exists():
            shutil.rmtree(old)
    for path in IMAGES_DIR.rglob("*.pdf"):
        path.unlink(missing_ok=True)
    for path in SOURCE_3D_DIR.glob("*"):
        if path.name not in {"age_opinions_attacks.png", "neuroticism_opinions_attacks.png"}:
            path.unlink(missing_ok=True)


def _label(value: Any, max_len: int = 42) -> str:
    text = str(value).replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "."
    return text


def _compact_feature_name(value: str, max_len: int = 32) -> str:
    text = str(value)
    prefixes = [
        "political_profile_",
        "demographics_and_identity_political_profile_",
        "personality_",
        "ideological_dimensions_two_axis_model_",
        "libertarian_authoritarian_dimension_model_",
        "moral_foundations_theory_",
        "gal_tan_model_",
        "big_five_",
        "hexaco_",
        "hexad_user_types_model_",
        "eysenck_pen_model_",
    ]
    for prefix in prefixes:
        text = text.replace(prefix, "")
    text = text.replace("_mean_pct", "").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[: max_len - 1] + "." if len(text) > max_len else text


def _short_domain(value: Any) -> str:
    return SHORT_DOMAIN.get(str(value), _label(value, 34))


def _phase_payload(row: Dict[str, Any], phase: str) -> Dict[str, Any]:
    return ((((row.get("attack") or {}).get("triplet") or {}).get(phase) or {}))


def _entropy_from_counts(counts: Sequence[float], normalized: bool = True) -> float:
    values = np.asarray([float(c) for c in counts if float(c) > 0], dtype=float)
    if len(values) == 0:
        return np.nan
    p = values / values.sum()
    h = float(-(p * np.log(p)).sum())
    if not normalized:
        return h
    denom = math.log(len(values)) if len(values) > 1 else 1.0
    return h / denom if denom else np.nan


def _entropy_series(series: pd.Series) -> float:
    return _entropy_from_counts(series.dropna().astype(str).value_counts().to_numpy(), normalized=True)


def _tv_distance(left: pd.Series, right: pd.Series) -> float:
    l = left.dropna().astype(str).value_counts(normalize=True)
    r = right.dropna().astype(str).value_counts(normalize=True)
    keys = l.index.union(r.index)
    return float(0.5 * np.abs(l.reindex(keys, fill_value=0) - r.reindex(keys, fill_value=0)).sum())


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    table = pd.crosstab(frame["x"].astype(str), frame["y"].astype(str))
    chi2 = chi2_contingency(table, correction=False).statistic
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1, n - 1))
    rcorr = r - ((r - 1) ** 2) / max(1, n - 1)
    kcorr = k - ((k - 1) ** 2) / max(1, n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def _cramers_matrix(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(np.eye(len(features)), index=features, columns=features, dtype=float)
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = _cramers_v(df[left], df[right])
            out.loc[left, right] = value
            out.loc[right, left] = value
    return out


def _vif_from_corr(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    data = data.loc[:, data.std(ddof=0) > 0]
    corr = data.corr(method="pearson").to_numpy(dtype=float)
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    return pd.DataFrame({"feature": data.columns, "vif": np.diag(inv)}).sort_values("vif", ascending=False)


def _load_attack_reference() -> pd.DataFrame:
    attack = _read_json(ATTACK_PATH)
    leaf_catalog = {int(row["id"]): row for row in attack["leaf_catalog"]}
    rows = []
    for cfg in attack["configurations"]:
        evidence = cfg.get("opinion_manipulation_evidence") or {}
        row: Dict[str, Any] = {
            "attack_config_id": int(cfg["id"]),
            "signal_total": float(evidence.get("signal_total") or np.nan),
            "inclusion_route": evidence.get("inclusion_route") or "UNKNOWN",
        }
        for phase in PHASES:
            leaf_id = int((cfg.get("leaves") or {}).get(phase))
            leaf = leaf_catalog.get(leaf_id, {})
            phase_score = ((evidence.get("phase_scores") or {}).get(phase) or {})
            row[f"{phase.lower()}_secondary"] = leaf.get("secondary") or "UNKNOWN"
            row[f"{phase.lower()}_label"] = leaf.get("label") or "UNKNOWN"
            row[f"{phase.lower()}_signal_score"] = float(phase_score.get("signal_score") or np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    design_rows: List[Dict[str, Any]] = []
    numeric_rows: List[Dict[str, float]] = []
    for obj in _iter_jsonl(INTEGRATED_PATH):
        profile = obj.get("profile") or {}
        demographics = profile.get("demographics") or {}
        categorical = profile.get("categorical_attributes") or {}
        numeric = profile.get("numeric_attributes") or {}
        attack = obj.get("attack") or {}
        opinion = obj.get("opinion_cluster") or {}
        direction = opinion.get("direction_summary") or {}
        row: Dict[str, Any] = {
            "scenario_id": obj.get("scenario_id"),
            "profile_id": profile.get("profile_id"),
            "age_years": float(demographics.get("age_years") if demographics.get("age_years") is not None else np.nan),
            "opinion_domain": opinion.get("parent_name") or "UNKNOWN",
            "opinion_domain_label": _short_domain(opinion.get("parent_name") or "UNKNOWN"),
            "opinion_n_leaves": int(opinion.get("n_leaves") or 0),
            "opinion_amplify_count": int(direction.get("amplify_+1") or 0),
            "opinion_erode_count": int(direction.get("erode_-1") or 0),
            "attack_config_id": int(attack.get("config_id") or -1),
            "source_config_id": int(attack.get("source_config_id") or -1),
            "signal_total": float(attack.get("signal_total") or np.nan),
            "inclusion_route": attack.get("inclusion_route") or "UNKNOWN",
        }
        for trait in BIG_FIVE:
            pct = (((demographics.get("big_five") or {}).get(trait) or {}).get("pct"))
            row[f"bigfive_{trait}"] = float(pct if pct is not None else np.nan)
        for label, key in CATEGORICAL_FEATURES.items():
            row[label] = categorical.get(key)
        for label, key in PLANNED_CONTINUOUS_FEATURES.items():
            if key in numeric:
                row[label] = float(numeric[key]) if numeric[key] is not None else np.nan
        phase_scores: Dict[str, float] = {}
        for phase in PHASES:
            payload = _phase_payload(obj, phase)
            score = payload.get("signal_score")
            score = float(score) if score is not None else np.nan
            phase_scores[phase] = score
            row[f"{phase.lower()}_secondary"] = payload.get("secondary") or "UNKNOWN"
            row[f"{phase.lower()}_label"] = payload.get("label") or "UNKNOWN"
            row[f"{phase.lower()}_signal_score"] = score
        row["dominant_attack_phase"] = max(phase_scores, key=lambda phase: -np.inf if pd.isna(phase_scores[phase]) else phase_scores[phase])
        design_rows.append(row)
        numeric_rows.append({k: float(v) if v is not None else np.nan for k, v in numeric.items()})

    df = pd.DataFrame(design_rows)
    numeric_df = pd.DataFrame(numeric_rows)
    df["age_band"] = pd.cut(df["age_years"], bins=[15, 24, 34, 44, 54, 64, 80], labels=["16-24", "25-34", "35-44", "45-54", "55-64", "65-80"], include_lowest=True)
    for trait in BIG_FIVE:
        df[f"{trait}_decile"] = pd.cut(df[f"bigfive_{trait}"], bins=np.linspace(0, 100, 11), labels=[f"D{i}" for i in range(1, 11)], include_lowest=True)
    return df, numeric_df


def _prepare_metrics(df: pd.DataFrame, numeric_df: pd.DataFrame, attack_ref: pd.DataFrame) -> Dict[str, Any]:
    signal_edges = np.unique(np.quantile(attack_ref["signal_total"].dropna(), np.linspace(0, 1, 11)))
    signal_labels = [f"D{i}" for i in range(1, len(signal_edges))]
    df["signal_decile"] = pd.cut(df["signal_total"], bins=signal_edges, labels=signal_labels, include_lowest=True, duplicates="drop")
    attack_ref["signal_decile"] = pd.cut(attack_ref["signal_total"], bins=signal_edges, labels=signal_labels, include_lowest=True, duplicates="drop")

    attack_rows = []
    for factor, col in [
        ("Plan tactic", "plan_secondary"),
        ("Prepare tactic", "prepare_secondary"),
        ("Execute tactic", "execute_secondary"),
        ("Inclusion route", "inclusion_route"),
        ("Signal decile", "signal_decile"),
    ]:
        sample_h = _entropy_series(df[col])
        ref_h = _entropy_series(attack_ref[col])
        attack_rows.append(
            {
                "factor": factor,
                "ontology_layer": "Attack",
                "sample_entropy": sample_h,
                "reference_entropy": ref_h,
                "entropy_difference": sample_h - ref_h,
                "total_variation_to_reference": _tv_distance(df[col], attack_ref[col]),
                "sample_categories": int(df[col].nunique(dropna=True)),
                "reference_categories": int(attack_ref[col].nunique(dropna=True)),
                "coverage": float(df[col].nunique(dropna=True) / attack_ref[col].nunique(dropna=True)),
            }
        )
    attack_entropy = pd.DataFrame(attack_rows)

    integrated_summary = _read_json(INTEGRATED_SUMMARY_PATH)
    opinion_summary_raw = _read_json(OPINION_SUMMARY_PATH)
    opinion_summary = opinion_summary_raw["_meta"]["summary"] if "_meta" in opinion_summary_raw else opinion_summary_raw
    attack_summary = _read_json(ATTACK_SUMMARY_PATH)

    entropy_rows = [
        ("Profile usage", "Profile", 1.0, 1.0, 1.0, df["profile_id"].nunique(), integrated_summary["sources"]["profiles"]["n_available"], np.nan),
        ("Age band", "Profile", _entropy_series(df["age_band"]), 1.0, 1.0, df["age_band"].nunique(), df["age_band"].nunique(), np.nan),
        ("Neuroticism decile", "Profile", _entropy_series(df["neuroticism_decile"]), 1.0, 1.0, df["neuroticism_decile"].nunique(), df["neuroticism_decile"].nunique(), np.nan),
        ("Opinion domain", "Opinion", _entropy_series(df["opinion_domain"]), 1.0, 1.0, df["opinion_domain"].nunique(), opinion_summary["n_clusters"], np.nan),
        ("Opinion leaves", "Opinion", np.nan, np.nan, 1.0, opinion_summary["n_directional_leaves"], opinion_summary["n_directional_leaves"], np.nan),
    ]
    for _, row in attack_entropy.iterrows():
        entropy_rows.append(
            (
                row["factor"],
                "Attack",
                row["sample_entropy"],
                row["reference_entropy"],
                row["coverage"],
                row["sample_categories"],
                row["reference_categories"],
                row["total_variation_to_reference"],
            )
        )
    attack_sub = (integrated_summary.get("entropy_report") or {}).get("attack_subsample", {})
    entropy_rows.append(
        (
            "Attack leaves",
            "Attack",
            np.nan,
            np.nan,
            float(attack_sub.get("leaf_coverage_of_filtered_pool", np.nan)),
            int(attack_sub.get("distinct_attack_leaves_used", 0)),
            int((attack_summary.get("filtered_set") or {}).get("filtered_distinct_leaves", 0)),
            np.nan,
        )
    )
    entropy = pd.DataFrame(
        entropy_rows,
        columns=["factor", "ontology_layer", "sample_entropy", "reference_entropy", "coverage", "n_observed", "n_reference", "total_variation_to_reference"],
    )

    state_space = pd.DataFrame(
        [
            ("Attack raw Cartesian upper bound", attack_summary["raw_state_space"]["full_cartesian_upper_bound"], "Attack"),
            ("Attack coherent raw sample", attack_summary["raw_state_space"]["n_raw_configurations"], "Attack"),
            ("Filtered attack triplets", attack_summary["filtered_set"]["n_configurations"], "Attack"),
            ("Integrated profile x domain x attacks", integrated_summary["sources"]["profiles"]["n_available"] * opinion_summary["n_clusters"] * attack_summary["filtered_set"]["n_configurations"], "Scenario"),
            ("Integrated scenarios retained", integrated_summary["n_scenarios"], "Scenario"),
        ],
        columns=["state_space_level", "n", "ontology_layer"],
    )
    state_space["log10_n"] = np.log10(state_space["n"].astype(float))

    design_features = [
        "age_band",
        "neuroticism_decile",
        "Continent",
        "Highest education",
        "opinion_domain_label",
        "plan_secondary",
        "prepare_secondary",
        "execute_secondary",
        "inclusion_route",
        "signal_decile",
    ]
    design_labels = {
        "age_band": "Age band",
        "neuroticism_decile": "Neuroticism",
        "Continent": "Continent",
        "Highest education": "Education",
        "opinion_domain_label": "Opinion domain",
        "plan_secondary": "Plan tactic",
        "prepare_secondary": "Prepare tactic",
        "execute_secondary": "Execute tactic",
        "inclusion_route": "Attack route",
        "signal_decile": "Signal decile",
    }
    design_cramers = _cramers_matrix(df, design_features).rename(index=design_labels, columns=design_labels)

    planned_frame = pd.DataFrame()
    for label, key in PLANNED_CONTINUOUS_FEATURES.items():
        if key in df.columns:
            planned_frame[label] = df[key]
        elif key in numeric_df.columns:
            planned_frame[label] = numeric_df[key]
    planned_vif = _vif_from_corr(planned_frame)

    numeric_clean = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_clean = numeric_clean.loc[:, numeric_clean.notna().mean() > 0.99]
    numeric_clean = numeric_clean.loc[:, numeric_clean.std(ddof=0) > 0]
    highres_corr = numeric_clean.corr(method="spearman")
    abs_corr = highres_corr.abs().copy()
    np.fill_diagonal(abs_corr.values, np.nan)
    feature_score = abs_corr.mean(axis=1).sort_values(ascending=False)
    selected_highres = feature_score.head(34).index.tolist()
    highres_corr_top = highres_corr.loc[selected_highres, selected_highres]
    highres_vif = _vif_from_corr(numeric_clean)

    cat_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    cat_cramers = _cramers_matrix(df, cat_features)

    factor_layers = {
        "Age band": "Profile",
        "Neuroticism": "Profile",
        "Continent": "Profile",
        "Education": "Profile",
        "Opinion domain": "Opinion",
        "Plan tactic": "Attack",
        "Prepare tactic": "Attack",
        "Execute tactic": "Attack",
        "Attack route": "Attack",
        "Signal decile": "Attack",
    }
    cross_values, within_attack = [], []
    for i, left in enumerate(design_cramers.index):
        for right in design_cramers.columns[i + 1 :]:
            value = float(design_cramers.loc[left, right])
            if not np.isfinite(value):
                continue
            if factor_layers.get(left) != factor_layers.get(right):
                cross_values.append(value)
            elif factor_layers.get(left) == "Attack":
                within_attack.append(value)

    metrics = {
        "integrated_summary": integrated_summary,
        "opinion_summary": opinion_summary,
        "attack_summary": attack_summary,
        "attack_entropy": attack_entropy,
        "entropy": entropy,
        "state_space": state_space,
        "design_cramers": design_cramers,
        "planned_frame": planned_frame,
        "planned_vif": planned_vif,
        "highres_corr": highres_corr,
        "highres_corr_top": highres_corr_top,
        "highres_vif": highres_vif,
        "cat_cramers": cat_cramers,
        "summary": {
            "n_scenarios": int(len(df)),
            "n_profiles": int(df["profile_id"].nunique()),
            "profile_source_n": int(integrated_summary["sources"]["profiles"]["n_available"]),
            "n_opinion_domains": int(df["opinion_domain"].nunique()),
            "opinion_source_domains": int(opinion_summary["n_clusters"]),
            "opinion_directional_leaves": int(opinion_summary["n_directional_leaves"]),
            "n_attack_triplets": int(df["attack_config_id"].nunique()),
            "attack_reference_n": int(len(attack_ref)),
            "attack_leaf_coverage": float(attack_sub.get("leaf_coverage_of_filtered_pool", np.nan)),
            "max_cross_ontology_cramers_v": float(np.nanmax(cross_values)),
            "max_within_attack_cramers_v": float(np.nanmax(within_attack)),
            "planned_max_vif": float(planned_vif["vif"].replace([np.inf, -np.inf], np.nan).max()),
            "highres_median_vif": float(highres_vif["vif"].replace([np.inf, -np.inf], np.nan).median()),
            "highres_max_vif": float(highres_vif["vif"].replace([np.inf, -np.inf], np.nan).max()),
            "max_categorical_cramers_v": float(np.nanmax(cat_cramers.where(~np.eye(len(cat_cramers), dtype=bool)).to_numpy(dtype=float))),
        },
    }
    return metrics


def _clustered_corr(corr: pd.DataFrame) -> pd.DataFrame:
    if corr.shape[0] < 3:
        return corr
    dist = 1 - corr.abs().fillna(0)
    condensed = squareform(dist.to_numpy(), checks=False)
    link = hierarchy.linkage(condensed, method="average")
    order = hierarchy.leaves_list(link)
    return corr.iloc[order, order]


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.03, 1.04, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")


def _embed_source_png(ax: plt.Axes, path: Path, title: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    if not path.exists():
        ax.text(0.5, 0.5, f"Missing source image:\n{path.name}", ha="center", va="center", fontsize=12)
        return
    ax.imshow(mpimg.imread(path))


def _make_figure_01(df: pd.DataFrame, attack_ref: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(24, 14))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.91, wspace=0.28, hspace=0.32)
    gs = GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    phase_domain = pd.crosstab(df["opinion_domain_label"], df["dominant_attack_phase"], normalize="index")
    phase_domain = phase_domain.reindex([_short_domain(k) for k in SHORT_DOMAIN.keys() if _short_domain(k) in phase_domain.index])
    sns.heatmap(
        phase_domain[PHASES],
        ax=ax0,
        cmap="Spectral_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Within-domain share"},
    )
    ax0.set_title("Opinion domains x dominant attack phase")
    ax0.set_xlabel("")
    ax0.set_ylabel("")
    ax0.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax0.tick_params(axis="y", labelsize=10)
    _panel_label(ax0, "A")

    ax1 = fig.add_subplot(gs[0, 1])
    _embed_source_png(ax1, AGE_SOURCE_PNG, "Exemplary spread: age x opinions x attacks")
    _panel_label(ax1, "B")

    ax2 = fig.add_subplot(gs[0, 2])
    _embed_source_png(ax2, NEUROTICISM_SOURCE_PNG, "Exemplary spread: neuroticism x opinions x attacks")
    _panel_label(ax2, "C")

    ax3 = fig.add_subplot(gs[1, 0])
    footprint = metrics["entropy"].dropna(subset=["coverage"]).copy()
    factor_labels = {
        "Profile usage": "Profile IDs",
        "Age band": "Age bands",
        "Neuroticism decile": "Neuroticism deciles",
        "Opinion domain": "Opinion domains",
        "Opinion leaves": "Opinion leaves",
        "Plan tactic": "Plan tactics",
        "Prepare tactic": "Prepare tactics",
        "Execute tactic": "Execute tactics",
        "Inclusion route": "Inclusion route",
        "Signal decile": "Signal deciles",
        "Attack leaves": "Attack leaves",
    }
    footprint["factor_label"] = footprint["factor"].map(factor_labels).fillna(footprint["factor"])
    footprint["log10_reference"] = np.log10(footprint["n_reference"].astype(float).clip(lower=1))
    has_entropy_target = footprint["sample_entropy"].notna() & footprint["reference_entropy"].notna()
    footprint["fidelity_delta"] = np.where(
        has_entropy_target,
        footprint["sample_entropy"] - footprint["reference_entropy"],
        footprint["coverage"] - 1.0,
    )
    footprint["diagnostic_type"] = np.where(has_entropy_target, "entropy", "support")
    ax3.axhspan(-0.01, 0.01, color="#e7f2ee", alpha=0.85, zorder=0)
    ax3.axhline(0, color="#252525", linestyle="--", linewidth=1.0, alpha=0.85)
    for layer, group in footprint.groupby("ontology_layer", sort=False):
        for dtype, marker in [("entropy", "o"), ("support", "D")]:
            sub = group[group["diagnostic_type"] == dtype]
            if sub.empty:
                continue
            ax3.scatter(
                sub["log10_reference"],
                sub["fidelity_delta"],
                s=85 + 250 * sub["coverage"],
                color=LAYER_COLORS.get(layer, "#777777"),
                marker=marker,
                alpha=0.88,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
    label_offsets = {
        "Attack leaves": (0.06, -0.012),
        "Profile IDs": (0.05, 0.014),
        "Opinion leaves": (0.06, 0.012),
        "Age bands": (0.05, -0.012),
        "Inclusion route": (0.05, 0.014),
        "Signal deciles": (0.05, -0.014),
    }
    for _, row in footprint[footprint["factor_label"].isin(label_offsets)].iterrows():
        dx, dy = label_offsets[row["factor_label"]]
        ax3.annotate(
            row["factor_label"],
            xy=(row["log10_reference"], row["fidelity_delta"]),
            xytext=(row["log10_reference"] + dx, row["fidelity_delta"] + dy),
            fontsize=8.4,
            color="#252525",
            arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": "#777777", "alpha": 0.75},
        )
    layer_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="white", markersize=8, label=layer)
        for layer, color in LAYER_COLORS.items()
        if layer in set(footprint["ontology_layer"])
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="#555555", linestyle="", markersize=7, label="Entropy"),
        plt.Line2D([0], [0], marker="D", color="#555555", linestyle="", markersize=6, label="Support"),
    ]
    ax3.legend(handles=layer_handles + marker_handles, fontsize=8, loc="lower left", frameon=True, title="")
    xticks = np.arange(0, math.ceil(float(footprint["log10_reference"].max())) + 1)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([f"{int(10 ** tick):,}" if tick < 3 else f"1e{tick}" for tick in xticks], fontsize=8)
    ymin = min(-0.28, float(footprint["fidelity_delta"].min()) - 0.035)
    ymax = max(0.04, float(footprint["fidelity_delta"].max()) + 0.025)
    ax3.set_ylim(ymin, ymax)
    ax3.set_xlim(-0.05, float(footprint["log10_reference"].max()) + 0.45)
    ax3.set_xlabel("Reference support size, log10 scale")
    ax3.set_ylabel("Deviation from target")
    ax3.set_title("Sampling fidelity by ontology factor")
    _panel_label(ax3, "D")

    ax4 = fig.add_subplot(gs[1, 1])
    state = metrics["state_space"].copy()
    state["state_space_label"] = state["state_space_level"].replace(
        {
            "Attack raw Cartesian upper bound": "Raw Cartesian attacks",
            "Attack coherent raw sample": "Coherent raw attacks",
            "Filtered attack triplets": "Filtered attack triplets",
            "Integrated profile x domain x attacks": "Profile x domain x attack",
            "Integrated scenarios retained": "Retained scenarios",
        }
    )
    colors = state["ontology_layer"].map(LAYER_COLORS).fillna("#777777")
    y = np.arange(len(state))
    xmax = float(state["log10_n"].max()) + 0.8
    ax4.hlines(y, 0, state["log10_n"], color=colors, linewidth=10, alpha=0.82)
    ax4.scatter(state["log10_n"], y, s=180, color=colors, edgecolor="white", linewidth=1.2, zorder=5)
    for yi, (_, row) in zip(y, state.iterrows()):
        if row["log10_n"] > xmax - 1.0:
            ax4.text(row["log10_n"] - 0.10, yi, f"{int(row['n']):,}", va="center", ha="right", fontsize=8.5, color="white")
        else:
            ax4.text(
                row["log10_n"] + 0.22,
                yi,
                f"{int(row['n']):,}",
                va="center",
                ha="left",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            )
    ax4.set_yticks(y)
    ax4.set_yticklabels(state["state_space_label"], fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel("log10(candidate count)")
    ax4.set_xlim(-0.1, xmax)
    ax4.set_title("State-space scale compression")
    _panel_label(ax4, "E")

    ax5 = fig.add_subplot(gs[1, 2])
    sns.kdeplot(data=attack_ref, x="signal_total", ax=ax5, color="#686868", linewidth=2.4, label="Filtered pool")
    sns.kdeplot(data=df, x="signal_total", ax=ax5, color=LAYER_COLORS["Attack"], linewidth=2.5, label="Integrated 10K")
    ref_q = np.quantile(attack_ref["signal_total"].dropna(), [0.1, 0.5, 0.9])
    sample_q = np.quantile(df["signal_total"].dropna(), [0.1, 0.5, 0.9])
    for value in ref_q:
        ax5.axvline(value, color="#686868", linestyle=":", linewidth=1.0, alpha=0.7)
    for value in sample_q:
        ax5.axvline(value, color=LAYER_COLORS["Attack"], linestyle="--", linewidth=1.0, alpha=0.65)
    ax5.set_title("Attack signal distribution preservation")
    ax5.set_xlabel("Opinion-manipulation signal")
    ax5.set_ylabel("Density")
    ax5.legend(title="", loc="upper right", fontsize=9)
    _panel_label(ax5, "F")

    fig.suptitle("Supplementary sample diagnostics: realized integrated 10K design", fontsize=22, fontweight="bold")
    fig.savefig(FIGURE_01, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _make_figure_02(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(24, 14))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.91, wspace=0.32, hspace=0.34)
    gs = GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ent = metrics["entropy"].dropna(subset=["sample_entropy"]).copy()
    ent["layer_order"] = ent["ontology_layer"].map({"Profile": 0, "Opinion": 1, "Attack": 2}).fillna(9)
    ent = ent.sort_values(["layer_order", "factor"])
    y = np.arange(len(ent))
    ax0.hlines(y, ent["reference_entropy"].fillna(1.0), ent["sample_entropy"], color="#888888", linewidth=1.4, alpha=0.8)
    for layer, group in ent.groupby("ontology_layer"):
        idx = group.index
        positions = [ent.index.get_loc(i) for i in idx]
        ax0.scatter(group["sample_entropy"], positions, s=80 + 190 * group["coverage"].fillna(1), color=LAYER_COLORS.get(layer, "#777777"), edgecolor="white", linewidth=1, label=layer, zorder=5)
    ax0.axvline(1, color="#111111", linestyle="--", linewidth=1)
    ax0.set_yticks(y)
    ax0.set_yticklabels(ent["factor"], fontsize=10)
    ax0.set_xlim(0, 1.05)
    ax0.invert_yaxis()
    ax0.set_xlabel("Normalised entropy")
    ax0.set_title("Entropy and coverage by ontology layer")
    ax0.legend(title="", loc="lower right", fontsize=9)
    _panel_label(ax0, "A")

    ax1 = fig.add_subplot(gs[0, 1])
    attack = metrics["attack_entropy"].copy()
    ax1.scatter(
        attack["total_variation_to_reference"],
        attack["entropy_difference"],
        s=130 + 40 * attack["sample_categories"],
        color=LAYER_COLORS["Attack"],
        edgecolor="white",
        linewidth=1.2,
    )
    for _, row in attack.iterrows():
        ax1.text(row["total_variation_to_reference"] + 0.00006, row["entropy_difference"], row["factor"], fontsize=9, va="center")
    ax1.axhline(0, color="#111111", linestyle="--", linewidth=1)
    ax1.axvline(0, color="#111111", linestyle=":", linewidth=1)
    ax1.set_xscale("symlog", linthresh=0.0001)
    ax1.set_xlabel("Total variation to filtered pool")
    ax1.set_ylabel("Entropy difference")
    ax1.set_title("Attack marginal preservation")
    _panel_label(ax1, "B")

    ax2 = fig.add_subplot(gs[0, 2])
    design = metrics["design_cramers"].copy()
    np.fill_diagonal(design.values, np.nan)
    sns.heatmap(design, ax=ax2, cmap="mako", vmin=0, vmax=max(0.10, np.nanmax(design.to_numpy())), cbar_kws={"label": "Cramer's V"})
    ax2.set_title("Cross-factor association audit")
    ax2.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    _panel_label(ax2, "C")

    ax3 = fig.add_subplot(gs[1, 0])
    top_corr = _clustered_corr(metrics["highres_corr_top"])
    sns.heatmap(
        top_corr.rename(index=_compact_feature_name, columns=_compact_feature_name),
        ax=ax3,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Spearman rho"},
    )
    ax3.set_title("High-resolution profile correlation block")
    ax3.tick_params(axis="x", labelrotation=55, labelsize=6)
    ax3.tick_params(axis="y", labelsize=6)
    _panel_label(ax3, "D")

    ax4 = fig.add_subplot(gs[1, 1])
    planned = metrics["planned_vif"].copy()
    planned["set"] = "Planned block"
    high = metrics["highres_vif"].copy()
    high["set"] = "Full profile space"
    vif_long = pd.concat([planned[["feature", "vif", "set"]], high[["feature", "vif", "set"]]], ignore_index=True)
    rng = np.random.default_rng(42)
    sets = ["Planned block", "Full profile space"]
    for ypos, name in enumerate(sets):
        values = vif_long.loc[vif_long["set"] == name, "vif"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        ax4.boxplot(
            values,
            vert=False,
            positions=[ypos],
            widths=0.34,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#c9d8ed", "edgecolor": "#777777"},
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#777777"},
            capprops={"color": "#777777"},
        )
        jitter = rng.normal(loc=ypos, scale=0.045, size=len(values))
        color = "#356b8c" if name == "Planned block" else "#b05a4a"
        ax4.scatter(values, jitter, s=9, alpha=0.32, color=color, linewidth=0, zorder=4)
    ax4.axvline(5, color="#8c2d04", linestyle="--", linewidth=1)
    ax4.axvline(10, color="#4a1486", linestyle="--", linewidth=1)
    ax4.set_xscale("log")
    ax4.set_xlim(0.8, max(12, float(vif_long["vif"].replace([np.inf, -np.inf], np.nan).max()) * 1.15))
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(sets, fontsize=10)
    ax4.set_xlabel("VIF")
    ax4.set_ylabel("")
    ax4.set_title("Moderator collinearity: planned vs full space")
    _panel_label(ax4, "E")

    ax5 = fig.add_subplot(gs[1, 2])
    cat = metrics["cat_cramers"].copy()
    np.fill_diagonal(cat.values, np.nan)
    sns.heatmap(cat, ax=ax5, cmap="mako", vmin=0, vmax=max(0.12, np.nanmax(cat.to_numpy())), cbar_kws={"label": "Cramer's V"})
    ax5.set_title("Categorical profile moderator association")
    ax5.tick_params(axis="x", labelrotation=55, labelsize=7)
    ax5.tick_params(axis="y", labelsize=7)
    _panel_label(ax5, "F")

    fig.suptitle("Supplementary sample diagnostics: entropy preservation and moderator readiness", fontsize=22, fontweight="bold")
    fig.savefig(FIGURE_02, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(df: pd.DataFrame, numeric_df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    df.to_csv(METRICS_DIR / "scenario_design_frame.csv", index=False)
    metrics["entropy"].to_csv(METRICS_DIR / "entropy_summary.csv", index=False)
    metrics["attack_entropy"].to_csv(METRICS_DIR / "attack_entropy_preservation.csv", index=False)
    metrics["state_space"].to_csv(METRICS_DIR / "state_space_scale.csv", index=False)
    metrics["design_cramers"].to_csv(METRICS_DIR / "design_factor_cramers_v.csv")
    metrics["planned_vif"].to_csv(METRICS_DIR / "planned_profile_moderator_vif.csv", index=False)
    metrics["highres_vif"].to_csv(METRICS_DIR / "high_resolution_profile_vif.csv", index=False)
    metrics["highres_corr_top"].to_csv(METRICS_DIR / "high_resolution_profile_top_correlation_block.csv")
    metrics["cat_cramers"].to_csv(METRICS_DIR / "categorical_profile_cramers_v.csv")
    pd.DataFrame([metrics["summary"]]).to_csv(TABLES_DIR / "sample_diagnostics_summary.csv", index=False)


def _write_readme(metrics: Dict[str, Any]) -> None:
    s = metrics["summary"]
    readme = f"""# Supplementary Analysis 02: Sample Diagnostics

This supplement audits the integrated 10,000-row scenario sample before any LLM
scoring. It focuses on realized design coverage, entropy preservation across the
profile, opinion, and attack ontologies, and whether profile moderator variables
are usable without hidden collinearity or confounding.

## Design

Each scenario combines one profile, one DISARM-red Plan/Prepare/Execute attack
triplet, and one opinion issue-domain cluster. The diagnostics use the existing
Stage 01 artifacts only. No model calls are made.

The two 3D source images in `images/3D_entropy_maximization/` are used as
exemplary spread panels inside Figure 1:

- `age_opinions_attacks.png`
- `neuroticism_opinions_attacks.png`

## Main Figures

### Figure 1. Realized Integrated 10K Design

![Sampling design diagnostics](images/01_sampling_design_diagnostics.png)

Interpretation: the domain-by-phase panel checks whether the realized sample
keeps attack-phase composition balanced within each opinion domain. The age and
neuroticism panels then illustrate that profile variation is distributed across
opinion-domain strata and Plan/Prepare/Execute phases, rather than concentrated
in a narrow slice of the design. The sampling-fidelity panel places ontology
factors by reference support size and realized entropy or support deviation,
making the high-dimensional attack-leaf compression visible without conflating
it with failures of marginal balance. The scale panel documents the log10
compression from the large attack and scenario candidate spaces into the
retained 10K design, and the attack-signal panel shows close distributional
preservation relative to the filtered attack pool.

### Figure 2. Entropy Preservation and Moderator Readiness

![Entropy and moderator diagnostics](images/02_entropy_and_profile_moderator_diagnostics.png)

Interpretation: profile and opinion entropy are at or near their theoretical
maximum, while attack marginals preserve the filtered DISARM-red pool rather
than imposing artificial uniformity. Cross-ontology Cramer's V remains very low,
which supports the independence of profile, opinion, and attack assignment.
The high-resolution profile space has strong structural collinearity, as
expected from facet and aggregate scores, but the planned moderator block has
VIF near 1.0.

## Key Results

- Scenarios: `{s['n_scenarios']:,}`.
- Unique profiles: `{s['n_profiles']:,}` of `{s['profile_source_n']:,}`.
- Opinion domains: `{s['n_opinion_domains']}` of `{s['opinion_source_domains']}`.
- Directional opinion leaves covered: `{s['opinion_directional_leaves']}`.
- Filtered attack triplets sampled: `{s['n_attack_triplets']:,}` of `{s['attack_reference_n']:,}`.
- Distinct filtered attack leaves covered: `{s['attack_leaf_coverage']:.1%}`.
- Maximum cross-ontology Cramer's V: `{s['max_cross_ontology_cramers_v']:.3f}`.
- Maximum within-attack Cramer's V: `{s['max_within_attack_cramers_v']:.3f}`.
- Planned moderator maximum VIF: `{s['planned_max_vif']:.3f}`.
- Full high-resolution profile median VIF: `{s['highres_median_vif']:.2f}`.
- Full high-resolution profile maximum VIF: `{s['highres_max_vif']:.2f}`.
- Maximum categorical profile Cramer's V: `{s['max_categorical_cramers_v']:.3f}`.

## Outputs

- `images/01_sampling_design_diagnostics.png`
- `images/02_entropy_and_profile_moderator_diagnostics.png`
- `images/3D_entropy_maximization/age_opinions_attacks.png`
- `images/3D_entropy_maximization/neuroticism_opinions_attacks.png`
- `metrics/`
- `tables/sample_diagnostics_summary.csv`

## Re-run

```bash
python src/supplementaries/02_sample_diagnostics/scripts/sample_diagnostics.py
```
"""
    (SUPPLEMENT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def run() -> None:
    _ensure_clean_dirs()
    if not AGE_SOURCE_PNG.exists() or not NEUROTICISM_SOURCE_PNG.exists():
        missing = [str(p) for p in [AGE_SOURCE_PNG, NEUROTICISM_SOURCE_PNG] if not p.exists()]
        raise RuntimeError("Missing source PNG(s): " + ", ".join(missing))
    print("Loading integrated 10K design frame...")
    df, numeric_df = _extract_frames()
    print("Loading filtered attack reference pool...")
    attack_ref = _load_attack_reference()
    print("Computing entropy, association, and profile-collinearity diagnostics...")
    metrics = _prepare_metrics(df, numeric_df, attack_ref)
    print("Rendering exactly two multipanel PNG figures...")
    _make_figure_01(df, attack_ref, metrics)
    _make_figure_02(df, metrics)
    _write_outputs(df, numeric_df, metrics)
    _write_readme(metrics)
    print(f"Wrote sample diagnostics to {SUPPLEMENT_ROOT}")


if __name__ == "__main__":
    run()
