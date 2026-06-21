from __future__ import annotations

"""
Technical overview
------------------
Stage 06 is the main modeling stage. It takes the attacked long table and the
profile-level repeated-outcome panel from Stage 05 and produces three linked
outputs:

1. a repeated-outcome path SEM over attacked opinion-shift indicators
2. robust OLS / bootstrap summaries for profile-level attacked effectivity
3. a post hoc conditional susceptibility artifact and ranking

The SEM side answers:
    which profile features are associated with larger attacked shifts on each
    repeated opinion leaf?

The conditional susceptibility side answers:
    given the configured attack-leaf set and opinion-leaf set, which profiles
    are predicted to be more susceptible overall?

This distinction matters. The SEM/path estimates remain leaf-specific, while
the conditional susceptibility index aggregates fitted task-level profile
effects into a reusable target-conditional profile score.

This module therefore sits at the center of the research design:
- it preserves repeated attacked outcomes instead of collapsing too early
- it separates descriptive profile ranking from path-level moderation output
- it prepares reusable fitted artifacts for future runs and later nonlinear
  model replacements
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from semopy import Model, calc_stats
from semopy.inspector import inspect as sem_inspect
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.analysis.advanced_inferential import fit_mixed_effects_moderation, run_advanced_inferential
from src.backend.utils.analysis.scenario_ml import profile_distance_moderation_test, run_scenario_ml
from src.backend.utils.analysis.conditional_susceptibility import (
    HierarchicalDecomposition,
    build_conditional_weight_table,
    fit_blockwise_family_susceptibility,
    fit_conditional_susceptibility_index,
)
from src.backend.utils.analysis.individual_layer_statistics import run_individual_layer_statistics
from src.backend.utils.data_utils import infer_analysis_mode, zscore_series
from src.backend.utils.io import abs_path, ensure_dir, stage_manifest_path, write_json, write_text
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.analysis.methodology_audit import (
    build_assumption_register,
    build_peer_review_critique_notes,
    render_methodology_audit_text,
)
from src.backend.utils.schemas import SemCoefficient, SemFitResult, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


class Stage06Config(StageConfig):
    primary_moderator: str = "posthoc_profile_susceptibility_index"
    bootstrap_samples: int = 500


CORE_CONTINUOUS_MODERATORS = [
    "profile_cont_age_years",
    "profile_cont_big_five_neuroticism_mean_pct",
    "profile_cont_big_five_openness_to_experience_mean_pct",
    "profile_cont_big_five_conscientiousness_mean_pct",
]
EXPLORATORY_CONTINUOUS_MODERATORS = [
    "profile_cont_big_five_extraversion_mean_pct",
    "profile_cont_big_five_agreeableness_mean_pct",
]
SEX_COLUMNS = [
    "profile_cat__profile_cat_sex_Female",
    "profile_cat__profile_cat_sex_Other",
]
CONTROL_COLUMNS = [
    "mean_baseline_abs_score_z",
    "mean_exposure_quality_score_z",
]

# Inventories present in run_1 ontology that are excluded from all analyses.
# They are not mappable to standard survey instruments and pollute the feature space.
# current design uses Political_Psychology / Socioeconomic_Status / Social_Context instead.
_INVENTORY_EXCLUSION_PREFIXES: Tuple[str, ...] = (
    "profile_cont_dual_process_inventory_",
    "profile_cont_digital_literacy_inventory_",
    "profile_cont_political_engagement_inventory_",
)


def _pretty_moderator_label(column_name: str) -> str:
    label = column_name
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("_z", "")
    label = label.replace("__", " ")
    label = label.replace("_", " ").strip()
    return " ".join(part.capitalize() if part.lower() != "pct" else "%" for part in label.split())


def _normalize_fit_indices(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "Value" in value:
            inner = value["Value"]
            flat[key] = float(inner) if hasattr(inner, "__float__") else inner
        else:
            flat[key] = float(value) if hasattr(value, "__float__") else value
    for bounded_key in ["CFI", "TLI", "GFI", "AGFI", "NFI"]:
        if bounded_key in flat and flat[bounded_key] is not None:
            flat[bounded_key] = max(0.0, min(1.0, float(flat[bounded_key])))
    if "RMSEA" in flat and flat["RMSEA"] is not None:
        flat["RMSEA"] = max(0.0, float(flat["RMSEA"]))
    return flat


_MAX_SEM_INDICATORS = 8


def _indicator_columns(profile_df: pd.DataFrame) -> List[str]:
    """Prefer adversarially-aligned indicators (current design); fall back to abs_delta.

    The repeated-outcome path SEM was designed for the run_1 CROSSED panel, where
    every profile faced the same handful of opinion leaves. In the integrated
    NESTED design each profile faces only one issue domain, so the per-leaf
    indicators are block-disjoint (often >100 leaves, each observed for ~10
    profiles). A full path SEM over all of them produces thousands of covariance
    parameters that are neither estimable nor fast. We therefore cap the SEM to
    the best-covered indicators; the primary moderation evidence comes from the
    profile-level OLS/ridge/elastic-net/random-forest models and the conditional
    susceptibility index, not this secondary path SEM.
    """
    adversarial = [
        column
        for column in profile_df.columns
        if column.startswith("adversarial_delta_indicator__") and not column.endswith("_z")
    ]
    candidates = adversarial or [
        column
        for column in profile_df.columns
        if column.startswith("abs_delta_indicator__") and not column.endswith("_z")
    ]
    if len(candidates) <= _MAX_SEM_INDICATORS:
        return sorted(candidates)
    # Rank by coverage (non-NaN profiles) so the capped SEM uses the most
    # commonly observed outcomes and stays estimable.
    coverage = {col: int(profile_df[col].notna().sum()) for col in candidates}
    top = sorted(candidates, key=lambda c: (-coverage[c], c))[:_MAX_SEM_INDICATORS]
    return sorted(top)


def _primary_outcome_column(profile_df: pd.DataFrame) -> str:
    """Return the profile-level aggregate outcome column for OLS/SEM."""
    if "mean_adversarial_effectivity" in profile_df.columns and profile_df["mean_adversarial_effectivity"].notna().any():
        return "mean_adversarial_effectivity"
    return "mean_abs_delta_score"


def _safe_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _available(columns: Sequence[str], df: pd.DataFrame) -> List[str]:
    return [column for column in columns if column in df.columns and df[column].nunique(dropna=True) > 1]


def _apply_bh_qvalues(df: pd.DataFrame, p_column: str, q_column: str, *, exclude_terms: Sequence[str] | None = None) -> pd.DataFrame:
    work = df.copy()
    if p_column not in work.columns:
        return work
    exclude = set(exclude_terms or [])
    valid_mask = work[p_column].notna()
    if "term" in work.columns and exclude:
        valid_mask &= ~work["term"].astype(str).isin(exclude)
    if valid_mask.sum() == 0:
        work[q_column] = np.nan
        return work
    rejected, q_values, _, _ = multipletests(work.loc[valid_mask, p_column].astype(float), method="fdr_bh")
    work[q_column] = np.nan
    work.loc[valid_mask, q_column] = q_values
    if q_column.replace("_value", "_significant") != q_column:
        work[q_column.replace("_value", "_significant")] = False
        work.loc[valid_mask, q_column.replace("_value", "_significant")] = rejected
    return work


def _ensure_standardized_columns(df: pd.DataFrame, continuous_columns: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    for column in continuous_columns:
        if column in work.columns and work[column].nunique(dropna=True) > 1:
            work[f"{column}_z"] = zscore_series(work[column].astype(float))
    return work


def _core_structural_terms(df: pd.DataFrame) -> List[str]:
    terms: List[str] = []
    terms.extend(_available([f"{column}_z" for column in CORE_CONTINUOUS_MODERATORS], df))
    terms.extend(_available(SEX_COLUMNS, df))
    terms.extend(_available(CONTROL_COLUMNS, df))
    return terms


def _all_profile_terms(df: pd.DataFrame) -> List[str]:
    terms: List[str] = []
    terms.extend(_available([f"{column}_z" for column in CORE_CONTINUOUS_MODERATORS], df))
    terms.extend(_available([f"{column}_z" for column in EXPLORATORY_CONTINUOUS_MODERATORS], df))
    terms.extend(_available(SEX_COLUMNS, df))
    return terms


def _build_formula(target: str, terms: Sequence[str]) -> str:
    rhs = " + ".join(terms) if terms else "1"
    return f"{target} ~ {rhs}"


def _fit_sem(
    profile_df: pd.DataFrame,
    indicator_columns: List[str],
    structural_terms: List[str],
) -> Tuple[SemFitResult, pd.DataFrame]:
    warnings: List[str] = []
    if len(indicator_columns) < 3:
        return (
            SemFitResult(
                model_name="profile_panel_path_sem",
                model_formula="",
                converged=False,
                n_obs=len(profile_df),
                fit_indices={},
                coefficients=[],
                warnings=["Need at least three abs-delta indicators for the profile-level SEM."],
            ),
            pd.DataFrame(),
        )

    # Restrict to complete cases on the chosen indicators. Under the integrated
    # NESTED design each profile only faces one issue domain, so cross-leaf
    # indicators are usually block-disjoint and there may be too few profiles
    # observed on all indicators to fit a path SEM. Skip gracefully in that case
    # (fast, no fabrication) rather than fitting a degenerate covariance model.
    complete_df = profile_df.dropna(subset=indicator_columns)
    min_needed = max(10, 2 * len(indicator_columns))
    if len(complete_df) < min_needed:
        return (
            SemFitResult(
                model_name="profile_panel_path_sem",
                model_formula="",
                converged=False,
                n_obs=int(len(complete_df)),
                fit_indices={},
                coefficients=[],
                warnings=[
                    f"Path SEM skipped: only {len(complete_df)} profiles are jointly observed on the "
                    f"{len(indicator_columns)} repeated-outcome indicators (nested cluster design). "
                    "Primary moderation evidence is the profile-level OLS/ridge/elastic-net/random-forest "
                    "models and the conditional susceptibility index."
                ],
            ),
            pd.DataFrame(),
        )

    regression_blocks = [
        _build_formula(indicator, structural_terms)
        for indicator in indicator_columns
    ]
    covariance_blocks: List[str] = []
    for idx, left in enumerate(indicator_columns):
        for right in indicator_columns[idx + 1 :]:
            covariance_blocks.append(f"{left} ~~ {right}")
    model_formula = "\n".join([*regression_blocks, *covariance_blocks])
    model = Model(model_formula)

    try:
        model.fit(complete_df)
        converged = True
    except Exception as exc:
        warnings.append(f"semopy fit failed: {exc}")
        converged = False

    coefficients: List[SemCoefficient] = []
    fit_indices: Dict[str, Any] = {}
    factor_scores = pd.DataFrame(index=profile_df.index)
    if converged:
        est = sem_inspect(model)
        for _, row in est.iterrows():
            coefficients.append(
                SemCoefficient(
                    lhs=str(row.get("lval", "")),
                    op=str(row.get("op", "")),
                    rhs=_pretty_moderator_label(str(row.get("rval", ""))),
                    estimate=float(row.get("Estimate", 0.0)),
                    std_error=_safe_optional_float(row.get("Std. Err")),
                    z_value=_safe_optional_float(row.get("z-value")),
                    p_value=_safe_optional_float(row.get("p-value")),
                )
            )
        stats = calc_stats(model)
        if hasattr(stats, "to_dict"):
            fit_indices = _normalize_fit_indices(stats.to_dict())

    return (
        SemFitResult(
            model_name="profile_panel_path_sem",
            model_formula=model_formula,
            converged=converged,
            n_obs=len(profile_df),
            fit_indices=fit_indices,
            coefficients=coefficients,
            warnings=warnings,
        ),
        factor_scores,
    )


def _ridge_fit_matrix(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    xtx = x.T @ x
    xty = x.T @ y
    return np.linalg.pinv(xtx + alpha * penalty) @ xty


def _kfold_indices(n_obs: int, seed: int, n_splits: int = 5) -> List[np.ndarray]:
    n_splits = max(2, min(n_splits, n_obs))
    indices = np.arange(n_obs)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, n_splits) if len(fold) > 0]


def _compute_icc(long_df: pd.DataFrame, outcome: str = "abs_delta_score",
                 cluster_col: str = "profile_id") -> Dict[str, float]:
    """Compute ICC(1) for a nested outcome.

    ICC(1) = σ²_between / (σ²_between + σ²_within)
    Estimated via one-way random-effects ANOVA decomposition.
    """
    groups = [g[outcome].values for _, g in long_df.groupby(cluster_col) if len(g) > 0]
    if len(groups) < 2:
        return {"icc1": np.nan, "sigma2_between": np.nan, "sigma2_within": np.nan, "n_clusters": 0}

    k = len(groups)
    ns = np.array([len(g) for g in groups])
    N = ns.sum()
    grand_mean = long_df[outcome].mean()

    ss_between = float(sum(n * (g.mean() - grand_mean) ** 2 for g, n in zip(groups, ns)))
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))

    ms_between = ss_between / max(1, k - 1)
    ms_within = ss_within / max(1, N - k)

    n0 = (N - (ns ** 2).sum() / N) / max(1, k - 1)
    sigma2_between = max(0.0, (ms_between - ms_within) / max(1e-12, n0))
    sigma2_within = ms_within
    icc1 = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0.0

    return {
        "icc1": round(icc1, 4),
        "sigma2_between": round(sigma2_between, 4),
        "sigma2_within": round(sigma2_within, 4),
        "n_clusters": k,
        "mean_cluster_size": round(float(ns.mean()), 2),
    }


def _cross_validated_ridge(
    df: pd.DataFrame,
    outcome: str,
    predictor_terms: Sequence[str],
    seed: int,
    alpha_grid: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    alpha_grid = list(alpha_grid or np.logspace(-3, 3, 25))
    design_terms = ["Intercept", *predictor_terms]
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            *[df[term].astype(float).to_numpy() for term in predictor_terms],
        ]
    )
    y = df[outcome].astype(float).to_numpy()
    folds = _kfold_indices(len(df), seed=seed, n_splits=5)

    best_alpha = alpha_grid[0]
    best_cv_mse = float("inf")
    for alpha in alpha_grid:
        fold_mses: List[float] = []
        for fold in folds:
            mask = np.ones(len(df), dtype=bool)
            mask[fold] = False
            beta = _ridge_fit_matrix(x[mask], y[mask], alpha)
            preds = x[fold] @ beta
            fold_mses.append(float(np.mean((y[fold] - preds) ** 2)))
        cv_mse = float(np.mean(fold_mses))
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_alpha = alpha

    beta = _ridge_fit_matrix(x, y, best_alpha)
    coeff_df = pd.DataFrame(
        {
            "outcome": outcome,
            "term": design_terms,
            "estimate": beta,
            "alpha": best_alpha,
            "cv_mse": best_cv_mse,
        }
    )
    predictions = pd.Series(x @ beta, index=df.index, name=f"predicted__{outcome}")
    model_meta = {
        "outcome": outcome,
        "alpha": float(best_alpha),
        "cv_mse": float(best_cv_mse),
    }
    return coeff_df, model_meta, predictions


def _fit_ridge_path_models(
    profile_df: pd.DataFrame,
    indicator_columns: Sequence[str],
    predictor_terms: Sequence[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coeff_frames: List[pd.DataFrame] = []
    meta_rows: List[Dict[str, float]] = []
    for offset, outcome in enumerate(indicator_columns):
        coeff_df, meta, _ = _cross_validated_ridge(
            df=profile_df,
            outcome=outcome,
            predictor_terms=predictor_terms,
            seed=seed + offset,
        )
        coeff_frames.append(coeff_df)
        meta_rows.append(meta)
    return pd.concat(coeff_frames, ignore_index=True), pd.DataFrame(meta_rows)


def _moderator_group(term: str) -> str:
    t = term.lower()
    # Demographics
    if "chronological_age" in t or "age_years" in t:
        return "Demographics: Age"
    if "sex_" in t or "profile_cat_sex" in t:
        return "Demographics: Sex"
    if "education_level" in t or ("education" in t and "level" in t):
        return "Demographics: Education"
    if "news_diet" in t:
        return "Demographics: News Diet"
    # Big Five
    if "neuroticism" in t:
        return "Big Five: Neuroticism"
    if "openness_to_experience" in t or ("openness" in t and "experience" in t):
        return "Big Five: Openness"
    if "conscientiousness" in t:
        return "Big Five: Conscientiousness"
    if "extraversion" in t:
        return "Big Five: Extraversion"
    if "agreeableness" in t:
        return "Big Five: Agreeableness"
    # Dual Process Inventory (run_1 ontology)
    if "dual_process" in t:
        return "Dual Process"
    # Digital Literacy Inventory (run_1 ontology)
    if "digital_literacy" in t:
        return "Digital Literacy"
    # Political Engagement Inventory (run_1 ontology)
    if "political_engagement_inventory_institutional_trust" in t:
        return "Political Engagement: Institutional Trust"
    if "political_engagement_inventory_ideological_identity" in t:
        return "Political Engagement: Ideology"
    if "political_engagement_inventory_political_interest" in t:
        return "Political Engagement: Interest"
    if "political_engagement_inventory_collective_efficacy" in t:
        return "Political Engagement: Efficacy"
    if "political_engagement_inventory" in t:
        return "Political Engagement"
    # Political Psychology (current design ontology)
    if "political_psychology_institutional_trust" in t or "institutional_trust" in t:
        return "Political Psychology: Institutional Trust"
    if "political_psychology_ideological_positioning" in t or "ideological_positioning" in t:
        return "Political Psychology: Ideology"
    if "political_psychology_political_engagement" in t:
        return "Political Psychology: Engagement"
    if "political_psychology" in t:
        return "Political Psychology"
    # Socioeconomic Status (current design ontology)
    if "socioeconomic_status_employment_type" in t or "employed_" in t or "unemployed" in t or "retired" in t or "employment_type" in t:
        return "Socioeconomic Status: Employment"
    if (
        "socioeconomic_status_economic_standing" in t
        or "household_income" in t
        or "economic_anxiety" in t
        or "financial_security" in t
        or "upward_mobility" in t
        or "subjective_class" in t
        or "economic_standing" in t
    ):
        return "Socioeconomic Status: Economic"
    if "socioeconomic" in t:
        return "Socioeconomic Status"
    # Social Context (current design ontology)
    if (
        "social_context_online_behavior" in t
        or "social_media_hours" in t
        or "echo_chamber" in t
        or "online_political_discussion" in t
        or "platform_primary_type" in t
        or ("platform" in t and "dominant" in t)
    ):
        return "Social Context: Online Behavior"
    if (
        "social_context_social_capital" in t
        or "interpersonal_trust" in t
        or "social_network_diversity" in t
        or "community_belonging" in t
        or "social_isolation" in t
    ):
        return "Social Context: Social Capital"
    if "social_context" in t:
        return "Social Context"
    # Integrated production ontology (10K set) naming.
    if "moral_foundations" in t:
        return "Political: Moral Foundations"
    if "gal_tan" in t:
        return "Political: GAL/TAN"
    if "libertarian_authoritarian" in t:
        return "Political: Libertarian/Authoritarian"
    if "ideological_dimensions" in t or "ideological" in t:
        return "Political: Ideology"
    if "nationalism" in t or "cosmopolitan" in t:
        return "Political: Nationalism"
    if "collective_narcissism" in t:
        return "Political: Collective Narcissism"
    if "populis" in t:
        return "Political: Populism"
    if "system_justification" in t:
        return "Political: System Justification"
    if "political_participation" in t:
        return "Political: Participation"
    if "political_profile" in t or "political" in t:
        return "Political Profile"
    if "religion" in t or "spirituality" in t or "worldview" in t:
        return "Religion and Worldview"
    if "digital" in t or "media_literacy" in t:
        return "Digital and Media Literacy"
    if "socioeconomic" in t or "income" in t or "employment" in t or "education" in t:
        return "Socioeconomic and Demographics"
    if "demographic" in t or "geography" in t or "household" in t or "migration" in t or "language" in t:
        return "Demographics"
    return "Other"


def _moderator_feature_type(term: str) -> str:
    t = term.lower()
    if t.startswith("profile_cat__"):
        return "Categorical dummy"
    if "chronological_age" in t or "age_years" in t:
        return "Continuous demographic"
    if "_mean_pct" in t:
        if "big_five" in t:
            return "Trait aggregate"
        return "Scale aggregate"
    if "big_five" in t:
        return "Facet"
    return "Continuous subscale"


_MAX_CATEGORICAL_DUMMIES = 60


def _dynamic_profile_terms(df: pd.DataFrame) -> List[str]:
    """Return profile feature columns used by the moderation/network/ML models.

    All variance-bearing continuous constructs are kept (these are the
    psychometric / political-psychological / ideological / demographic
    inter-individual-difference moderators). Synthetic proxies and the three
    run_1 non-survey inventories are excluded.

    Categorical one-hot dummies are kept as-is for small ontologies (run_1). For
    the production profile, one-hotting ~272 categorical traits explodes into
    ~1500 mostly near-singleton administrative/clinical dummies, which are both
    statistically noise on a small panel and prohibitively slow for the network
    / random-forest stack. When the dummy space is large we therefore keep only
    the reasonably balanced dummies (prevalence in [0.10, 0.90]) and cap the
    count to the most balanced ones, so the moderation analysis stays
    interpretable and tractable. The full profile is still shown to the
    simulation agents; this filter only governs the analysis feature space.
    """
    _exclude = {
        "profile_cont_heuristic_shift_sensitivity_proxy",
        "profile_cont_resilience_index",
    }
    continuous_terms: List[str] = []
    categorical_terms: List[str] = []
    for col in sorted(df.columns):
        if col in _exclude:
            continue
        if any(col.startswith(p) for p in _INVENTORY_EXCLUSION_PREFIXES):
            continue
        if col.endswith("_z"):  # skip pre-standardised duplicates
            continue
        if col.startswith("profile_cont_"):
            if df[col].nunique(dropna=True) > 1:
                continuous_terms.append(col)
        elif col.startswith("profile_cat__"):
            if df[col].nunique(dropna=True) > 1:
                categorical_terms.append(col)

    if len(categorical_terms) > _MAX_CATEGORICAL_DUMMIES:
        prevalence = {c: float(df[c].astype(float).mean()) for c in categorical_terms}
        balanced = [c for c in categorical_terms if 0.10 <= prevalence[c] <= 0.90]
        balanced.sort(key=lambda c: abs(prevalence[c] - 0.5))
        categorical_terms = sorted(balanced[:_MAX_CATEGORICAL_DUMMIES])

    return continuous_terms + categorical_terms


def _conditional_feature_terms(df: pd.DataFrame) -> List[str]:
    """Construct-level continuous moderators for the conditional susceptibility index.

    Each per-(attack, opinion) task model is fit on only ~10-15 profiles, so the
    index uses the ~45 higher-order *_mean_pct psychometric / political-psychology
    aggregates (Big Five, ideological dimensions, moral foundations, RWA, SDO,
    populism, ...) plus age, rather than all ~250 fine facets. This keeps each
    per-task ridge well-posed and the resulting feature weights interpretable,
    while the empirical-Bayes shrinkage pools strength across the 106 opinion-leaf
    tasks. Using all facets would be both over-parameterised at the task level and
    far slower under the bootstrap.
    """
    _exclude = {
        "profile_cont_heuristic_shift_sensitivity_proxy",
        "profile_cont_resilience_index",
    }
    terms: List[str] = []
    for col in sorted(df.columns):
        if not col.startswith("profile_cont_") or col.endswith("_z") or col in _exclude:
            continue
        if any(col.startswith(p) for p in _INVENTORY_EXCLUSION_PREFIXES):
            continue
        if (col.endswith("mean_pct") or col == "profile_cont_age_years") and df[col].nunique(dropna=True) > 1:
            terms.append(col)
    # Fall back to the broader set if this profile has no construct-level aggregates.
    return terms or _dynamic_profile_terms(df)


def _analysis_feature_terms(df: pd.DataFrame) -> List[str]:
    """Moderator panel for the headline mixed-effects model and the ML moderation
    stack.

    For the high-dimensional production profile we use the ~46 construct-level
    aggregates (Big Five plus the political-psychology, ideological,
    moral-foundations, authoritarianism / SDO, populism and political-participation
    constructs) rather than the ~250 fine facets. At n ~ 100 scenarios this keeps
    the mixed-effects and regularized models well-posed and the moderators
    interpretable while still spanning the whole profile space, not just Big Five
    and age. Small ontologies (run_1) keep their full variance-bearing feature set
    unchanged.
    """
    broad = _dynamic_profile_terms(df)
    if len(broad) > 80:
        return _conditional_feature_terms(df)
    return broad


def _fit_elastic_net(
    df: pd.DataFrame,
    outcome: str,
    feature_terms: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    """Cross-validated Elastic Net on all profile features.

    Addresses the fundamental weakness of the aggregate OLS: using only 8 hardcoded
    Big Five predictors when 100+ profile features are available. ElasticNetCV handles
    high-dimensional small-n by regularisation, selecting informative features and
    shrinking noise to zero. CV-R² is reported as an honest fit estimate.
    """
    from sklearn.linear_model import ElasticNetCV  # type: ignore
    from sklearn.model_selection import KFold, cross_val_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    x_raw = df[list(feature_terms)].astype(float).fillna(0.0).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    col_std = np.std(x_raw, axis=0)
    valid_mask = col_std > 1e-8
    terms_used = [t for t, v in zip(feature_terms, valid_mask) if v]
    x = x_raw[:, valid_mask]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        cv=cv_inner,
        random_state=seed,
        max_iter=20000,
        alphas=50,
        n_jobs=-1,
    )
    enet.fit(x_scaled, y)

    y_hat = enet.predict(x_scaled)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_train = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cv_outer = KFold(n_splits=5, shuffle=True, random_state=seed + 1)
    cv_r2_scores = cross_val_score(
        ElasticNetCV(
            l1_ratio=float(enet.l1_ratio_),
            cv=5,
            random_state=seed,
            max_iter=20000,
            alphas=50,
        ),
        x_scaled,
        y,
        cv=cv_outer,
        scoring="r2",
        n_jobs=-1,
    )
    cv_r2 = float(np.mean(cv_r2_scores))
    cv_r2_std = float(np.std(cv_r2_scores))

    coeff_df = pd.DataFrame(
        {
            "term": terms_used,
            "label": [_pretty_moderator_label(t) for t in terms_used],
            "ontology_group": [_moderator_group(t) for t in terms_used],
            "elastic_net_estimate": enet.coef_,
        }
    )
    selected_df = (
        coeff_df[coeff_df["elastic_net_estimate"].abs() > 1e-8]
        .copy()
        .sort_values("elastic_net_estimate", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    return {
        "r2_train": r2_train,
        "cv_r2": cv_r2,
        "cv_r2_std": cv_r2_std,
        "alpha": float(enet.alpha_),
        "l1_ratio": float(enet.l1_ratio_),
        "n_features_total": len(terms_used),
        "n_features_selected": int((np.abs(enet.coef_) > 1e-8).sum()),
        "coeff_df": coeff_df,
        "selected_df": selected_df,
    }


def _fit_random_forest(
    df: pd.DataFrame,
    outcome: str,
    feature_terms: Sequence[str],
    seed: int,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """Random Forest regression for non-linear moderation detection.

    OOB R² provides an honest fit estimate without a 01_separated test set.
    Permutation importance (n_repeats=50) is preferred over MDI because it
    accounts for correlated features and is invariant to feature scale.
    """
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.inspection import permutation_importance as sk_perm_importance  # type: ignore

    x_raw = df[list(feature_terms)].astype(float).fillna(0.0).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    col_std = np.std(x_raw, axis=0)
    valid_mask = col_std > 1e-8
    terms_used = [t for t, v in zip(feature_terms, valid_mask) if v]
    x = x_raw[:, valid_mask]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=3,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x, y)

    perm = sk_perm_importance(rf, x, y, n_repeats=50, random_state=seed, n_jobs=-1)

    importance_df = (
        pd.DataFrame(
            {
                "term": terms_used,
                "label": [_pretty_moderator_label(t) for t in terms_used],
                "ontology_group": [_moderator_group(t) for t in terms_used],
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
                "mdi_importance": rf.feature_importances_,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "oob_r2": float(rf.oob_score_),
        "n_estimators": n_estimators,
        "n_features": len(terms_used),
        "importance_df": importance_df,
    }


def _fit_ridge_full_features(
    df: pd.DataFrame,
    outcome: str,
    feature_terms: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    """Pure Ridge regression on all profile features — the correct estimator for effect
    direction when p ~ n.

    Unlike LASSO/EN, Ridge does NOT zero out correlated features; it shrinks all
    coefficients proportionally. This gives continuous, interpretable estimates for
    every predictor (digital literacy, political engagement, dual process, Big Five
    facets) rather than arbitrarily selecting one representative from each correlated
    cluster.

    CV-R² will be near-zero when ICC≈0 (expected), but the coefficient *direction and
    relative magnitude* across features are still theoretically informative — they show
    which features the model consistently associates with higher or lower susceptibility.
    """
    from sklearn.linear_model import Ridge, RidgeCV  # type: ignore
    from sklearn.model_selection import KFold, cross_val_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    x_raw = df[list(feature_terms)].astype(float).fillna(0.0).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    col_std = np.std(x_raw, axis=0)
    valid_mask = col_std > 1e-8
    terms_used = [t for t, v in zip(feature_terms, valid_mask) if v]
    x = x_raw[:, valid_mask]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    alphas = np.logspace(-3, 4, 60)
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    ridge_cv = RidgeCV(alphas=alphas, cv=cv_inner)
    ridge_cv.fit(x_scaled, y)

    cv_outer = KFold(n_splits=5, shuffle=True, random_state=seed + 1)
    cv_r2_scores = cross_val_score(
        Ridge(alpha=float(ridge_cv.alpha_)),
        x_scaled,
        y,
        cv=cv_outer,
        scoring="r2",
        n_jobs=-1,
    )
    cv_r2 = float(np.mean(cv_r2_scores))
    cv_r2_std = float(np.std(cv_r2_scores))

    coeff_df = (
        pd.DataFrame(
            {
                "term": terms_used,
                "label": [_pretty_moderator_label(t) for t in terms_used],
                "ontology_group": [_moderator_group(t) for t in terms_used],
                "ridge_estimate": ridge_cv.coef_,
            }
        )
        .sort_values("ridge_estimate", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    return {
        "cv_r2": cv_r2,
        "cv_r2_std": cv_r2_std,
        "alpha": float(ridge_cv.alpha_),
        "coeff_df": coeff_df,
    }


def _fit_network_analysis(
    df: pd.DataFrame,
    feature_terms: Sequence[str],
    corr_threshold: float = 0.15,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build a Spearman correlation network of profile features and compute
    a comprehensive suite of local and global network metrics.

    Local metrics (per node):
      - degree_centrality
      - eigenvector_centrality
      - betweenness_centrality
      - closeness_centrality
      - clustering_coefficient
      - pagerank
      - community (Louvain via greedy_modularity_communities)

    Global metrics:
      - n_nodes, n_edges, density
      - avg_clustering, transitivity
      - n_communities, modularity_score
      - avg_degree, max_degree
    """
    import networkx as nx
    from scipy import stats as scipy_stats

    x = df[list(feature_terms)].astype(float).fillna(0.0)
    col_std = x.std(axis=0)
    valid = col_std[col_std > 1e-8].index.tolist()
    x = x[valid]

    # Spearman correlation matrix
    n_feats = len(valid)
    corr_mat = np.ones((n_feats, n_feats))
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            rho, _ = scipy_stats.spearmanr(x.iloc[:, i].values, x.iloc[:, j].values)
            corr_mat[i, j] = corr_mat[j, i] = float(rho) if not np.isnan(rho) else 0.0
    corr_df = pd.DataFrame(corr_mat, index=valid, columns=valid)

    # Build graph: undirected, edge weight = |rho|, only keep |rho| >= threshold
    G = nx.Graph()
    G.add_nodes_from(valid)
    node_group = {node: _moderator_group(node) for node in valid}
    node_family = {node: node_group[node].split(":", 1)[0] for node in valid}
    node_feature_type = {node: _moderator_feature_type(node) for node in valid}
    nx.set_node_attributes(G, node_group, "ontology_group")
    nx.set_node_attributes(G, node_family, "ontology_family")
    nx.set_node_attributes(G, node_feature_type, "feature_type")
    edge_rows: List[Dict[str, Any]] = []
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            rho = corr_mat[i, j]
            if abs(rho) >= corr_threshold:
                strength = float(abs(rho))
                G.add_edge(
                    valid[i],
                    valid[j],
                    weight=strength,
                    distance=float(1.0 / max(strength, 1e-6)),
                    rho=float(rho),
                )
                edge_rows.append({
                    "source": valid[i],
                    "target": valid[j],
                    "rho": float(rho),
                    "abs_rho": strength,
                    "source_label": _pretty_moderator_label(valid[i]),
                    "target_label": _pretty_moderator_label(valid[j]),
                })
    edge_df = pd.DataFrame(edge_rows)

    # Local centrality metrics
    deg_cent = nx.degree_centrality(G)
    try:
        eig_cent = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig_cent = {n: 0.0 for n in G.nodes()}
    try:
        bet_cent = nx.betweenness_centrality(G, weight="distance", normalized=True, seed=seed)
    except Exception:
        bet_cent = {n: 0.0 for n in G.nodes()}
    try:
        clo_cent = nx.closeness_centrality(G, distance="distance")
    except Exception:
        clo_cent = {n: 0.0 for n in G.nodes()}
    clust = nx.clustering(G, weight="weight")
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    # Community detection (greedy modularity communities = Louvain-like)
    communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    node_community: Dict[str, int] = {}
    for comm_idx, comm in enumerate(communities):
        for node in comm:
            node_community[node] = comm_idx
    community_sizes = {idx: len(comm) for idx, comm in enumerate(communities)}

    # Compute modularity
    try:
        modularity_score = float(nx.community.modularity(G, communities, weight="weight"))
    except Exception:
        modularity_score = float("nan")

    node_strength = {node: float(sum(data.get("weight", 0.0) for _, _, data in G.edges(node, data=True))) for node in G.nodes()}
    try:
        core_number = nx.core_number(G)
    except Exception:
        core_number = {node: 0 for node in G.nodes()}

    positive_degree: Dict[str, int] = {}
    negative_degree: Dict[str, int] = {}
    positive_strength: Dict[str, float] = {}
    negative_strength: Dict[str, float] = {}
    within_community_strength: Dict[str, float] = {}
    between_community_strength: Dict[str, float] = {}
    participation_coefficient: Dict[str, float] = {}
    bridge_ratio: Dict[str, float] = {}
    signed_balance: Dict[str, float] = {}
    same_family_strength_share: Dict[str, float] = {}
    within_module_zscore: Dict[str, float] = {}

    within_community_raw: Dict[str, float] = {}
    community_strength_lists: Dict[int, List[float]] = {}

    for node in G.nodes():
        comm_strengths: Dict[int, float] = {}
        total_strength = 0.0
        pos_degree = 0
        neg_degree = 0
        pos_strength = 0.0
        neg_strength = 0.0
        same_family_strength = 0.0
        node_comm = node_community.get(node, -1)
        node_node_family = node_family.get(node, "Other")
        for neighbor, edge_data in G[node].items():
            weight = float(edge_data.get("weight", 0.0))
            rho = float(edge_data.get("rho", 0.0))
            total_strength += weight
            if rho >= 0:
                pos_degree += 1
                pos_strength += weight
            else:
                neg_degree += 1
                neg_strength += weight
            neighbor_comm = node_community.get(neighbor, -1)
            comm_strengths[neighbor_comm] = comm_strengths.get(neighbor_comm, 0.0) + weight
            if node_family.get(neighbor, "Other") == node_node_family:
                same_family_strength += weight

        within_strength = comm_strengths.get(node_comm, 0.0)
        between_strength = max(0.0, total_strength - within_strength)
        positive_degree[node] = pos_degree
        negative_degree[node] = neg_degree
        positive_strength[node] = pos_strength
        negative_strength[node] = neg_strength
        within_community_strength[node] = within_strength
        between_community_strength[node] = between_strength
        participation_coefficient[node] = (
            1.0 - sum((value / total_strength) ** 2 for value in comm_strengths.values())
            if total_strength > 1e-12
            else 0.0
        )
        bridge_ratio[node] = between_strength / total_strength if total_strength > 1e-12 else 0.0
        signed_balance[node] = (pos_strength - neg_strength) / total_strength if total_strength > 1e-12 else 0.0
        same_family_strength_share[node] = same_family_strength / total_strength if total_strength > 1e-12 else 0.0
        within_community_raw[node] = within_strength
        community_strength_lists.setdefault(node_comm, []).append(within_strength)

    community_within_mean = {
        comm: float(np.mean(values)) if values else 0.0
        for comm, values in community_strength_lists.items()
    }
    community_within_std = {
        comm: float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
        for comm, values in community_strength_lists.items()
    }
    for node in G.nodes():
        comm = node_community.get(node, -1)
        std = community_within_std.get(comm, 0.0)
        mean = community_within_mean.get(comm, 0.0)
        within_module_zscore[node] = (
            (within_community_raw.get(node, 0.0) - mean) / std
            if std > 1e-12
            else 0.0
        )

    centrality_rows: List[Dict[str, Any]] = []
    for node in valid:
        centrality_rows.append({
            "term": node,
            "label": _pretty_moderator_label(node),
            "ontology_group": node_group.get(node, "Other"),
            "ontology_family": node_family.get(node, "Other"),
            "feature_type": node_feature_type.get(node, "Other"),
            "degree_centrality": float(deg_cent.get(node, 0.0)),
            "eigenvector_centrality": float(eig_cent.get(node, 0.0)),
            "betweenness_centrality": float(bet_cent.get(node, 0.0)),
            "closeness_centrality": float(clo_cent.get(node, 0.0)),
            "clustering_coefficient": float(clust.get(node, 0.0)),
            "pagerank": float(pr.get(node, 0.0)),
            "community": int(node_community.get(node, -1)),
            "community_size": int(community_sizes.get(node_community.get(node, -1), 0)),
            "degree": int(G.degree(node)),
            "strength": float(node_strength.get(node, 0.0)),
            "positive_degree": int(positive_degree.get(node, 0)),
            "negative_degree": int(negative_degree.get(node, 0)),
            "positive_strength": float(positive_strength.get(node, 0.0)),
            "negative_strength": float(negative_strength.get(node, 0.0)),
            "within_community_strength": float(within_community_strength.get(node, 0.0)),
            "between_community_strength": float(between_community_strength.get(node, 0.0)),
            "participation_coefficient": float(participation_coefficient.get(node, 0.0)),
            "bridge_ratio": float(bridge_ratio.get(node, 0.0)),
            "within_module_zscore": float(within_module_zscore.get(node, 0.0)),
            "same_family_strength_share": float(same_family_strength_share.get(node, 0.0)),
            "signed_balance": float(signed_balance.get(node, 0.0)),
            "k_core": int(core_number.get(node, 0)),
        })
    centrality_df = (
        pd.DataFrame(centrality_rows)
        .sort_values("eigenvector_centrality", ascending=False)
        .reset_index(drop=True)
    )

    # Global metrics
    abs_rho_values = edge_df["abs_rho"].astype(float) if not edge_df.empty else pd.Series(dtype=float)
    pos_edges = int((edge_df["rho"] > 0).sum()) if not edge_df.empty else 0
    neg_edges = int((edge_df["rho"] < 0).sum()) if not edge_df.empty else 0
    within_community_edges = 0
    between_community_edges = 0
    within_family_edges = 0
    between_family_edges = 0
    for source, target in G.edges():
        if node_community.get(source, -1) == node_community.get(target, -1):
            within_community_edges += 1
        else:
            between_community_edges += 1
        if node_family.get(source, "Other") == node_family.get(target, "Other"):
            within_family_edges += 1
        else:
            between_family_edges += 1
    try:
        ontology_group_assortativity = float(nx.attribute_assortativity_coefficient(G, "ontology_group"))
    except Exception:
        ontology_group_assortativity = float("nan")
    try:
        ontology_family_assortativity = float(nx.attribute_assortativity_coefficient(G, "ontology_family"))
    except Exception:
        ontology_family_assortativity = float("nan")
    try:
        feature_type_assortativity = float(nx.attribute_assortativity_coefficient(G, "feature_type"))
    except Exception:
        feature_type_assortativity = float("nan")
    global_metrics: Dict[str, Any] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": float(nx.density(G)),
        "avg_clustering": float(nx.average_clustering(G, weight="weight")),
        "transitivity": float(nx.transitivity(G)),
        "n_communities": len(communities),
        "modularity_score": modularity_score,
        "avg_degree": float(sum(d for _, d in G.degree()) / max(1, G.number_of_nodes())),
        "max_degree": int(max((d for _, d in G.degree()), default=0)),
        "avg_strength": float(np.mean(list(node_strength.values()))) if node_strength else 0.0,
        "max_strength": float(max(node_strength.values())) if node_strength else 0.0,
        "mean_abs_rho": float(abs_rho_values.mean()) if not abs_rho_values.empty else 0.0,
        "positive_edge_count": pos_edges,
        "negative_edge_count": neg_edges,
        "positive_edge_share": float(pos_edges / max(1, G.number_of_edges())),
        "negative_edge_share": float(neg_edges / max(1, G.number_of_edges())),
        "largest_community_size": int(max(community_sizes.values(), default=0)),
        "within_community_edge_share": float(within_community_edges / max(1, G.number_of_edges())),
        "between_community_edge_share": float(between_community_edges / max(1, G.number_of_edges())),
        "within_family_edge_share": float(within_family_edges / max(1, G.number_of_edges())),
        "between_family_edge_share": float(between_family_edges / max(1, G.number_of_edges())),
        "mean_participation_coefficient": float(np.mean(list(participation_coefficient.values()))) if participation_coefficient else 0.0,
        "mean_bridge_ratio": float(np.mean(list(bridge_ratio.values()))) if bridge_ratio else 0.0,
        "ontology_group_assortativity": ontology_group_assortativity,
        "ontology_family_assortativity": ontology_family_assortativity,
        "feature_type_assortativity": feature_type_assortativity,
        "corr_threshold": corr_threshold,
        "n_features_total": len(valid),
        "note": "Spearman correlation network of profile features. Strength-based metrics use |rho|; path-based metrics use distance = 1 / |rho|. Participation and within-module z-score 01_separated bridge-like nodes from community-local hubs.",
    }

    # Compute spring layout for visualization
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.5 / max(1, n_feats ** 0.5))
    layout_rows: List[Dict[str, Any]] = []
    for node, (x_pos, y_pos) in pos.items():
        layout_rows.append({"term": node, "x": float(x_pos), "y": float(y_pos)})
    layout_df = pd.DataFrame(layout_rows)

    return {
        "centrality_df": centrality_df,
        "edge_df": edge_df,
        "corr_df": corr_df,
        "layout_df": layout_df,
        "global_metrics": global_metrics,
        "n_communities": len(communities),
    }


def _build_expanded_moderator_table(
    ridge_coeff_df: pd.DataFrame,
    ols_exploratory_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Ridge (all features) with OLS estimates (Big Five only) into one table.

    For features present in both: keep OLS multivariate p-value and CI alongside
    ridge coefficient. For features only in ridge: show ridge estimate with NaN for
    OLS columns. Sorted by |ridge_estimate| descending.
    """
    ridge = ridge_coeff_df.copy()
    ols_lookup: Dict[str, Any] = {}
    if not ols_exploratory_df.empty and "moderator_column" in ols_exploratory_df.columns:
        for row in ols_exploratory_df.to_dict(orient="records"):
            ols_lookup[row["moderator_column"]] = row

    rows: List[Dict[str, Any]] = []
    for _, rrow in ridge.iterrows():
        term = rrow["term"]
        ols = ols_lookup.get(term, {})
        rows.append(
            {
                "moderator_column": term,
                "moderator_label": rrow["label"],
                "ontology_group": rrow["ontology_group"],
                "ridge_estimate": float(rrow["ridge_estimate"]),
                "multivariate_estimate": ols.get("multivariate_estimate", np.nan),
                "multivariate_p_value": ols.get("multivariate_p_value", np.nan),
                "multivariate_q_value": ols.get("multivariate_q_value", np.nan),
                "multivariate_conf_low": ols.get("multivariate_conf_low", np.nan),
                "multivariate_conf_high": ols.get("multivariate_conf_high", np.nan),
                "univariate_estimate": ols.get("univariate_estimate", np.nan),
                "univariate_conf_low": ols.get("univariate_conf_low", np.nan),
                "univariate_conf_high": ols.get("univariate_conf_high", np.nan),
                "elastic_net_estimate": rrow.get("elastic_net_estimate", np.nan),
                "rf_permutation_importance": ols.get("rf_permutation_importance", np.nan),
                "normalized_weight_pct": ols.get("normalized_weight_pct", np.nan),
                "ridge_mean_estimate": ols.get("ridge_mean_estimate", np.nan),
                "role": "core" if term in ols_lookup else "expanded",
            }
        )

    expanded = (
        pd.DataFrame(rows)
        .sort_values("ridge_estimate", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return expanded


def _build_moderator_weight_table(
    profile_df: pd.DataFrame,
    ridge_params: pd.DataFrame,
    moderator_terms: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for term in moderator_terms:
        if term not in profile_df.columns:
            continue
        match = ridge_params.loc[ridge_params["term"] == term]
        if match.empty:
            continue
        estimate = float(match["estimate"].mean())
        mean_abs_estimate = float(match["estimate"].abs().mean())
        term_sd = float(profile_df[term].astype(float).std(ddof=0))
        importance = mean_abs_estimate * term_sd
        rows.append(
            {
                "term": term,
                "moderator_label": _pretty_moderator_label(term),
                "ontology_group": _moderator_group(term),
                "estimate": estimate,
                "mean_abs_estimate": mean_abs_estimate,
                "term_sd": term_sd,
                "importance": importance,
                "direction": "higher_effectivity" if estimate >= 0 else "lower_effectivity",
                "n_outcomes": int(match["outcome"].nunique()) if "outcome" in match.columns else 1,
            }
        )
    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        return weight_df
    denom = float(weight_df["importance"].sum()) or 1.0
    weight_df["normalized_weight_pct"] = (weight_df["importance"] / denom) * 100.0
    return weight_df.sort_values(["normalized_weight_pct", "moderator_label"], ascending=[False, True]).reset_index(drop=True)


def _bootstrap_ols(
    df: pd.DataFrame,
    formula: str,
    terms: Sequence[str],
    n_bootstrap: int,
    seed: int,
    cluster_col: str = "profile_id",
) -> pd.DataFrame:
    """Cluster bootstrap OLS: resamples at the profile level to respect nesting.

    When *cluster_col* is present in *df*, entire clusters are resampled
    (preserving within-cluster dependence).  Falls back to IID pairs
    bootstrap when the column is absent.
    """
    rng = np.random.default_rng(seed)
    use_cluster = cluster_col in df.columns
    if use_cluster:
        cluster_ids = df[cluster_col].unique()
    records: List[Dict[str, float]] = []
    for _ in range(n_bootstrap):
        if use_cluster:
            sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sample = pd.concat(
                [df[df[cluster_col] == cid] for cid in sampled], ignore_index=True
            )
        else:
            sample_idx = rng.integers(0, len(df), size=len(df))
            sample = df.iloc[sample_idx].copy()
        try:
            model = smf.ols(formula, data=sample).fit()
        except Exception:
            continue
        records.append({term: float(model.params.get(term, np.nan)) for term in terms})

    if not records:
        return pd.DataFrame(
            {
                "term": list(terms),
                "bootstrap_mean": np.nan,
                "bootstrap_std": np.nan,
                "conf_low": np.nan,
                "conf_high": np.nan,
                "n_bootstrap_success": 0,
            }
        )

    boot_df = pd.DataFrame(records)
    summary_rows: List[Dict[str, object]] = []
    for term in terms:
        values = boot_df[term].dropna()
        summary_rows.append(
            {
                "term": term,
                "bootstrap_mean": float(values.mean()) if len(values) else np.nan,
                "bootstrap_std": float(values.std(ddof=0)) if len(values) else np.nan,
                "conf_low": float(values.quantile(0.025)) if len(values) else np.nan,
                "conf_high": float(values.quantile(0.975)) if len(values) else np.nan,
                "n_bootstrap_success": int(len(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def _fit_exploratory_models(
    profile_df: pd.DataFrame,
    multivariate_params: pd.DataFrame,
    control_terms: List[str],
    candidate_terms: List[str],
    outcome: str = "mean_abs_delta_score",
) -> pd.DataFrame:
    multivariate_lookup = {row["term"]: row for row in multivariate_params.to_dict(orient="records")}
    rows: List[Dict[str, object]] = []

    for term in candidate_terms:
        formula = _build_formula(outcome, [term, *control_terms])
        try:
            result = smf.ols(formula, data=profile_df).fit(cov_type="HC3")
        except Exception as exc:
            LOGGER.warning("Exploratory model failed for %s: %s", term, exc)
            continue
        conf = result.conf_int()
        multi = multivariate_lookup.get(term, {})
        rows.append(
            {
                "moderator_column": term,
                "moderator_label": _pretty_moderator_label(term),
                "multivariate_estimate": float(multi.get("estimate", np.nan)),
                "multivariate_std_error": float(multi.get("std_error", np.nan)),
                "multivariate_p_value": float(multi.get("p_value", np.nan)),
                "multivariate_conf_low": float(multi.get("conf_low", np.nan)),
                "multivariate_conf_high": float(multi.get("conf_high", np.nan)),
                "univariate_estimate": float(result.params.get(term, np.nan)),
                "univariate_std_error": float(result.bse.get(term, np.nan)),
                "univariate_p_value": float(result.pvalues.get(term, np.nan)),
                "univariate_conf_low": float(conf.loc[term, 0]),
                "univariate_conf_high": float(conf.loc[term, 1]),
                "role": "core" if term in _available([f"{column}_z" for column in CORE_CONTINUOUS_MODERATORS], profile_df) or term in _available(SEX_COLUMNS, profile_df) else "exploratory",
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    comparison = _apply_bh_qvalues(comparison, "univariate_p_value", "univariate_q_value")
    comparison = _apply_bh_qvalues(comparison, "multivariate_p_value", "multivariate_q_value")

    # Report Wald z instead of mislabelled effect-size surrogates.
    for prefix in ("univariate", "multivariate"):
        est_col = f"{prefix}_estimate"
        se_col = f"{prefix}_std_error"
        if est_col in comparison.columns and se_col in comparison.columns:
            comparison[f"{prefix}_wald_z"] = (
                comparison[est_col] / comparison[se_col].replace(0, np.nan)
            ).round(4)

    return comparison.sort_values(["role", "multivariate_p_value", "moderator_label"]).reset_index(drop=True)


def _compute_profile_susceptibility_outputs(
    profile_df: pd.DataFrame,
    ridge_params: pd.DataFrame,
    moderator_terms: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lookup = (
        ridge_params.groupby("term", as_index=False)["estimate"]
        .mean()
        .set_index("term")["estimate"]
        .to_dict()
    )
    intercept = float(lookup.get("Intercept", 0.0))
    scores = pd.Series(np.zeros(len(profile_df)), index=profile_df.index, dtype=float)
    contributions: Dict[str, pd.Series] = {}
    group_contributions: Dict[str, pd.Series] = {}
    breakdown_rows: List[Dict[str, object]] = []

    for term in moderator_terms:
        if term not in profile_df.columns:
            continue
        beta = float(lookup.get(term, 0.0))
        contribution = profile_df[term].astype(float) * beta
        scores = scores + contribution
        contributions[term] = contribution
        group = _moderator_group(term)
        if group not in group_contributions:
            group_contributions[group] = pd.Series(np.zeros(len(profile_df)), index=profile_df.index, dtype=float)
        group_contributions[group] = group_contributions[group] + contribution

    result = profile_df[["profile_id", "mean_abs_delta_score", "mean_signed_delta_score"]].copy()
    if "latent_attack_effectivity_factor_score" in profile_df.columns:
        result["latent_attack_effectivity_factor_score"] = profile_df["latent_attack_effectivity_factor_score"]
        result["latent_attack_effectivity_factor_score_z"] = zscore_series(profile_df["latent_attack_effectivity_factor_score"])
    result["profile_moderator_linear_score"] = scores
    result["profile_moderator_linear_score_z"] = zscore_series(scores)
    result["predicted_mean_abs_delta_from_moderators"] = intercept + scores
    result["susceptibility_index_pct"] = scores.rank(method="average", pct=True) * 100.0
    result["observed_effectivity_pct"] = result["mean_abs_delta_score"].rank(method="average", pct=True) * 100.0

    for term, contribution in contributions.items():
        result[f"contribution__{term}"] = contribution

    for group, contribution in group_contributions.items():
        slug = (
            group.lower()
            .replace(": ", "__")
            .replace(" ", "_")
            .replace("-", "_")
        )
        result[f"group_contribution__{slug}"] = contribution

    for row in result.to_dict(orient="records"):
        for term in contributions:
            breakdown_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "component_type": "term",
                    "component_name": _pretty_moderator_label(term),
                    "component_key": term,
                    "ontology_group": _moderator_group(term),
                    "contribution": row.get(f"contribution__{term}", np.nan),
                    "susceptibility_index_pct": row["susceptibility_index_pct"],
                }
            )
        for group in group_contributions:
            slug = (
                group.lower()
                .replace(": ", "__")
                .replace(" ", "_")
                .replace("-", "_")
            )
            breakdown_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "component_type": "group",
                    "component_name": group,
                    "component_key": slug,
                    "ontology_group": group,
                    "contribution": row.get(f"group_contribution__{slug}", np.nan),
                    "susceptibility_index_pct": row["susceptibility_index_pct"],
                }
            )

    result = result.sort_values("susceptibility_index_pct", ascending=False).reset_index(drop=True)
    breakdown_df = pd.DataFrame(breakdown_rows)
    return result, breakdown_df


def _build_quality_diagnostics(long_df: pd.DataFrame) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "n_rows": int(len(long_df)),
        "n_profiles": int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else None,
        "n_attack_opinion_tasks": (
            int(long_df[["attack_leaf", "opinion_leaf"]].drop_duplicates().shape[0])
            if {"attack_leaf", "opinion_leaf"}.issubset(long_df.columns)
            else None
        ),
    }
    bool_columns = [
        "baseline_fallback_used",
        "post_fallback_used",
        "attack_rewrite_required",
        "baseline_rewrite_required",
        "post_rewrite_required",
        "attack_heuristic_pass",
        "baseline_heuristic_pass",
        "post_heuristic_pass",
    ]
    for column in bool_columns:
        if column in long_df.columns:
            diagnostics[f"{column}_rate"] = float(long_df[column].fillna(False).astype(bool).mean())

    score_columns = [
        "attack_realism_score",
        "attack_coherence_score",
        "baseline_plausibility_score",
        "baseline_consistency_score",
        "post_plausibility_score",
        "post_consistency_score",
    ]
    for column in score_columns:
        if column in long_df.columns and long_df[column].notna().any():
            diagnostics[f"mean_{column}"] = float(long_df[column].dropna().mean())

    diagnostics["note"] = (
        "Run-quality diagnostics are computed directly from the attacked long table. "
        "High fallback rates or zero-valued review scores indicate execution problems rather than substantive moderator evidence."
    )
    return diagnostics


def _bootstrap_rank_table(conditional_fit) -> pd.DataFrame:
    if conditional_fit.bootstrap_ci is None:
        return pd.DataFrame(columns=["profile_id", "rank_ci_low", "rank_ci_high", "rank_sd", "n_bootstrap_samples"])
    rows: List[Dict[str, Any]] = []
    for profile_id, (low, high) in conditional_fit.bootstrap_ci.rank_ci.items():
        rows.append(
            {
                "profile_id": profile_id,
                "rank_ci_low": low,
                "rank_ci_high": high,
                "rank_sd": conditional_fit.bootstrap_ci.rank_sd.get(profile_id, np.nan),
                "n_bootstrap_samples": conditional_fit.bootstrap_ci.n_samples,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ci_high", ascending=False).reset_index(drop=True)


def _bootstrap_feature_sd_table(conditional_fit) -> pd.DataFrame:
    if conditional_fit.bootstrap_ci is None:
        return pd.DataFrame(columns=["term", "component_key", "coefficient_sd"])
    rows: List[Dict[str, Any]] = []
    for term, component_dict in conditional_fit.bootstrap_ci.coefficient_sd.items():
        for component_key, sd_value in component_dict.items():
            rows.append(
                {
                    "term": term,
                    "component_key": component_key,
                    "coefficient_sd": sd_value,
                }
            )
    return pd.DataFrame(rows).sort_values("coefficient_sd", ascending=False).reset_index(drop=True)


def _safe_ols_summary(ols_model) -> str:
    try:
        return str(ols_model.summary())
    except Exception as exc:
        fallback_lines = [
            "statsmodels summary unavailable",
            f"reason: {exc}",
            "",
            "parameter estimates",
            ols_model.params.to_string(),
            "",
            "p-values",
            ols_model.pvalues.to_string(),
        ]
        return "\n".join(fallback_lines)


def _render_report(
    long_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    sem_result: SemFitResult,
    multivariate_formula: str,
    ols_summary: str,
    ols_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    exploratory_table: pd.DataFrame,
    profile_index_df: pd.DataFrame,
    weight_table: pd.DataFrame,
    task_summary_df: pd.DataFrame,
    run_id: str,
    primary_outcome: str = "mean_abs_delta_score",
    hierarchical_decomposition: Any = None,
    enet_result: Dict[str, Any] | None = None,
    rf_result: Dict[str, Any] | None = None,
    ridge_full_result: Dict[str, Any] | None = None,
) -> str:
    fit_cfi = sem_result.fit_indices.get("CFI")
    fit_rmsea = sem_result.fit_indices.get("RMSEA")
    fit_line = (
        f"CFI={fit_cfi:.3f}, RMSEA={fit_rmsea:.3f}"
        if fit_cfi is not None and fit_rmsea is not None
        else "fit indices unavailable"
    )

    realism_text = "n/a"
    if "attack_realism_score" in long_df.columns:
        realism_vals = long_df["attack_realism_score"].dropna()
        if len(realism_vals) > 0:
            realism_text = f"{float(realism_vals.mean()):.3f}"

    plausibility_text = "n/a"
    if "post_plausibility_score" in long_df.columns:
        plausibility_vals = long_df["post_plausibility_score"].dropna()
        if len(plausibility_vals) > 0:
            plausibility_text = f"{float(plausibility_vals.mean()):.3f}"

    indicator_columns = _indicator_columns(profile_df)
    top_multivariate = exploratory_table.sort_values("multivariate_p_value").head(6) if not exploratory_table.empty else pd.DataFrame()
    top_weights = weight_table.head(6) if not weight_table.empty else pd.DataFrame()
    attack_leaves = sorted({str(value) for value in long_df["attack_leaf"].dropna().unique().tolist()}) if "attack_leaf" in long_df.columns else []
    opinion_leaves = sorted({str(value) for value in long_df["opinion_leaf"].dropna().unique().tolist()}) if "opinion_leaf" in long_df.columns else []
    task_meta_lines = []
    if not task_summary_df.empty:
        for row in task_summary_df.to_dict(orient="records"):
            task_meta_lines.append(
                f"{row['attack_leaf']} | {row['opinion_leaf']}: alpha={float(row['alpha']):.4f}, cv_mse={float(row['cv_mse']):.4f}, weight={float(row['reliability_weight']):.4f}"
            )

    bootstrap_lookup = {row["term"]: row for row in bootstrap_table.to_dict(orient="records")}

    adv_text = "n/a"
    if "adversarial_effectivity" in long_df.columns:
        adv_vals = long_df["adversarial_effectivity"].dropna()
        if len(adv_vals) > 0:
            pos_pct = float((adv_vals > 0).mean() * 100.0)
            adv_text = f"{float(adv_vals.mean()):.3f} (positive={pos_pct:.1f}%)"

    lines = [
        f"Moderation Report - {run_id}",
        "=========================",
        "",
        f"Profiles analyzed: {len(profile_df)}",
        f"Attacked opinion scenarios analyzed: {len(long_df)}",
        f"Repeated opinion indicators: {len(indicator_columns)}",
        f"Primary effectivity outcome: {primary_outcome}",
        f"Mean adversarial effectivity: {adv_text}",
        f"Mean absolute delta: {float(long_df['abs_delta_score'].mean()):.3f}",
        f"Mean signed delta: {float(long_df['delta_score'].mean()):.3f}",
        f"Mean attack realism score: {realism_text}",
        f"Mean post-exposure plausibility score: {plausibility_text}",
        "",
        "Profile-Level Path SEM",
        "----------------------",
        f"Converged: {sem_result.converged}",
        f"Fit indices: {fit_line}",
        f"Warnings: {', '.join(sem_result.warnings) if sem_result.warnings else 'none'}",
        f"Formula: {sem_result.model_formula}",
        "",
        "Regularized Susceptibility Model",
        "--------------------------------",
        "The empirical susceptibility index is conditional on the modeled attack-leaf set and opinion-leaf set. It is derived from task-specific cross-validated ridge models fit separately to each (attack leaf, opinion leaf) target, then aggregated back to the profile level using reliability weights.",
        f"Attack target set: {', '.join(attack_leaves) if attack_leaves else 'n/a'}",
        f"Opinion target set: {', '.join(opinion_leaves) if opinion_leaves else 'n/a'}",
    ]

    if task_meta_lines:
        lines.extend(task_meta_lines)

    lines.extend(
        [
            "",
            "Primary Multivariate Profile Model",
            "----------------------------------",
            f"Outcome: {primary_outcome}",
            f"Formula: {multivariate_formula}",
        ]
    )

    if hierarchical_decomposition is not None:
        lines.extend(["", "Hierarchical Variance Decomposition (Conditional Susceptibility)", "----------------------------------------------------------------"])
        lines.append(f"Full model CV-R\u00b2: {hierarchical_decomposition.full_model_cv_r2:.4f}")
        sorted_groups = sorted(
            hierarchical_decomposition.group_marginal_r2.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        for group, mar_r2 in sorted_groups:
            rel_pct = hierarchical_decomposition.group_relative_importance_pct.get(group, 0.0)
            lines.append(f"  {group}: marginal_R\u00b2={mar_r2:.4f}, relative_importance={rel_pct:.1f}%")

    for row in ols_table.to_dict(orient="records"):
        boot = bootstrap_lookup.get(row["term"], {})
        lines.append(
            f"{row['term']}: est={row['estimate']:.4f}, p={row['p_value']:.6f}, boot95=[{boot.get('conf_low', np.nan):.4f}, {boot.get('conf_high', np.nan):.4f}]"
        )

    if not top_multivariate.empty:
        lines.extend(["", "Moderator Highlights", "-------------------"])
        for row in top_multivariate.to_dict(orient="records"):
            lines.append(
                f"{row['moderator_label']}: mean univariate b={row['univariate_estimate']:.4f}, p={row['univariate_p_value']:.6f}; ridge mean b={row.get('ridge_mean_estimate', np.nan):.4f}, weight_pct={row.get('normalized_weight_pct', np.nan):.2f}"
            )

    if not profile_index_df.empty:
        top_profiles = profile_index_df.head(5)
        lines.extend(["", "Empirical Profile Susceptibility Index", "--------------------------------------"])
        lines.append(f"The post hoc susceptibility index is the percentile-ranked profile-only linear predictor under the configured attack/opinion target set. Primary outcome: {primary_outcome}.")
        for row in top_profiles.to_dict(orient="records"):
            adv_val = row.get("mean_adversarial_effectivity")
            adv_str = f", mean_adversarial_eff={adv_val:.2f}" if adv_val is not None else ""
            lines.append(
                f"{row['profile_id']}: susceptibility_index_pct={row['susceptibility_index_pct']:.2f}, mean_abs_delta={row.get('mean_abs_delta_score', float('nan')):.2f}{adv_str}"
            )

    if not top_weights.empty:
        lines.extend(["", "Moderator Weight Decomposition", "------------------------------"])
        for row in top_weights.to_dict(orient="records"):
            lines.append(
                f"{row['moderator_label']} [{row['ontology_group']}]: est={row['estimate']:.4f}, normalized_weight_pct={row['normalized_weight_pct']:.2f}"
            )

    # Ridge full-feature results (primary effect estimator)
    if ridge_full_result is not None:
        lines.extend(
            [
                "",
                "Ridge Regression — Full Feature Set (Primary Effect Estimator)",
                "----------------------------------------------------------------",
                "Ridge retains ALL ~100 profile features with continuous shrinkage.",
                "Unlike LASSO/EN, it does NOT zero correlated features — gives proper",
                "direction and relative magnitude for Big Five facets, political engagement,",
                "digital literacy, dual process, etc.",
                f"CV-R² (5-fold): {ridge_full_result.get('cv_r2', float('nan')):.4f} ± {ridge_full_result.get('cv_r2_std', float('nan')):.4f}",
                f"Best alpha: {ridge_full_result.get('alpha', float('nan')):.4f}",
                "Top 15 features by |ridge coefficient| (on std-scaled features):",
            ]
        )
        rc = ridge_full_result.get("coeff_df", pd.DataFrame())
        for row in rc.head(15).to_dict(orient="records"):
            lines.append(
                f"  {row.get('label', row.get('term'))}: ridge={row['ridge_estimate']:.4f}  [{row.get('ontology_group', '')}]"
            )

    # Elastic Net results (LASSO feature selector — secondary)
    if enet_result is not None:
        lines.extend(
            [
                "",
                "Elastic Net / LASSO (Feature Selector — Secondary)",
                "--------------------------------------------------",
                "LASSO zeroes correlated features; selected set = hardest survivors.",
                f"Features: {enet_result.get('n_features_total', 'n/a')} → selected {enet_result.get('n_features_selected', 'n/a')}",
                f"alpha={enet_result.get('alpha', float('nan')):.5f}, l1_ratio={enet_result.get('l1_ratio', float('nan')):.2f}",
                f"CV-R²: {enet_result.get('cv_r2', float('nan')):.4f} ± {enet_result.get('cv_r2_std', float('nan')):.4f}",
                "Selected features:",
            ]
        )
        sel = enet_result.get("selected_df", pd.DataFrame())
        if sel.empty:
            lines.append("  (none — LASSO collapses to null model, consistent with ICC≈0)")
        for row in sel.head(10).to_dict(orient="records"):
            lines.append(
                f"  {row.get('label', row.get('term'))}: coef={row['elastic_net_estimate']:.4f}  [{row.get('ontology_group', '')}]"
            )

    # Random Forest results
    if rf_result is not None:
        lines.extend(
            [
                "",
                "Random Forest Moderation Model (Non-linear, Full Feature Set)",
                "--------------------------------------------------------------",
                f"OOB R²: {rf_result.get('oob_r2', float('nan')):.4f}  (unbiased held-out estimate)",
                f"Trees: {rf_result.get('n_estimators', 'n/a')}, Features: {rf_result.get('n_features', 'n/a')}",
                "Top features by permutation importance:",
            ]
        )
        imp = rf_result.get("importance_df", pd.DataFrame())
        for row in imp.head(10).to_dict(orient="records"):
            lines.append(
                f"  {row.get('label', row.get('term'))}: perm_imp={row['permutation_importance_mean']:.4f} ± {row['permutation_importance_std']:.4f}  [{row.get('ontology_group', '')}]"
            )

    lines.extend(
        [
            "",
            "OLS Supplement (Conventional Benchmark — Big Five + Age + Sex Only)",
            "---------------------------------------------------------------------",
            "NOTE: The following OLS uses only the hardcoded Big Five domain means, age, and sex.",
            "It serves as a conventional benchmark. The Elastic Net and Random Forest above use",
            "the full feature set and are the primary moderation estimators.",
            "",
            ols_summary,
            "",
            "Caveat",
            "------",
            "This attacked-only testing run estimates heterogeneity in attacked opinion movement. It does not estimate a no-attack counterfactual effect, and the post hoc susceptibility index is descriptive because it is derived from the fitted profile moderation model rather than observed independently.",
        ]
    )
    return "\n".join(lines)


def run_stage(input_path: str, output_dir: str, config: Stage06Config) -> StageArtifactManifest:
    ensure_dir(output_dir)
    long_df = pd.read_csv(input_path)
    analysis_mode = infer_analysis_mode(long_df)
    if analysis_mode != "treated_only":
        raise RuntimeError("SEM stage is designed for attacked-only profile-panel data (attack_ratio=1.0).")

    # Attack-grouping policy. In the integrated design each scenario carries its
    # own (effectively unique) DISARM-red triplet identity, so per-(attack,opinion)
    # task models would be singletons and an attack one-hot would degenerate into
    # ~one dummy per scenario. When attacks are this granular we (a) POOL attacks
    # for the conditional-susceptibility index so each opinion leaf is one
    # well-populated task that estimates PROFILE moderation, and (b) collapse the
    # attack factor to the real DISARM *Execute tactic* (a small set of shared,
    # human-readable vectors such as "Deliver Content" / "Maximise Exposure") for
    # the scenario-ML stack and the per-attack effect summaries, falling back to
    # the coarser inclusion_route only when the tactic is unavailable. The stored
    # long table keeps the true per-scenario attack identity for provenance.
    # Few-attack runs (e.g. run_1) are left untouched.
    _n_attacks = int(long_df["attack_leaf"].nunique()) if "attack_leaf" in long_df.columns else 0
    _n_scenarios = int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else 0
    pool_attacks = _n_attacks > max(8, _n_scenarios // 4)
    if pool_attacks:
        LOGGER.info(
            "Stage 06: attacks are near-unique (%d distinct over %d scenarios); pooling attacks "
            "for the conditional-susceptibility index and grouping by DISARM Execute tactic for the ML stack.",
            _n_attacks, _n_scenarios,
        )
        # Conditional-susceptibility index. Keep the well-populated POOLED task
        # (one model per opinion leaf over all attacks, the primary profile-moderation
        # estimate) AND add per-DISARM-Execute-tactic tasks so the dashboard estimator
        # can be scoped to a specific attack vector rather than only the pooled view.
        # The two label spaces are disjoint ("DISARM_attacks_pooled" vs the tactic
        # names), so a task scope selects one regime without double-counting; the
        # reliability weights down-weight thinly-populated per-tactic cells.
        _pooled_csi = long_df.assign(attack_leaf="DISARM_attacks_pooled", attack_leaf_label="DISARM_attacks_pooled")
        if "attack_execute_tactic" in long_df.columns and long_df["attack_execute_tactic"].notna().any():
            _tactic = long_df["attack_execute_tactic"].fillna("Unspecified tactic").astype(str)
            long_df_csi = pd.concat(
                [_pooled_csi, long_df.assign(attack_leaf=_tactic, attack_leaf_label=_tactic)],
                ignore_index=True,
            )
            long_df_sml = long_df.assign(attack_leaf=_tactic, attack_leaf_label=_tactic)
        elif "attack_inclusion_route" in long_df.columns:
            _route = long_df["attack_inclusion_route"].fillna("unknown_route").astype(str)
            long_df_csi = _pooled_csi
            long_df_sml = long_df.assign(attack_leaf="route::" + _route, attack_leaf_label="route::" + _route)
        else:
            long_df_csi = _pooled_csi
            long_df_sml = long_df.assign(attack_leaf="DISARM_attacks_pooled", attack_leaf_label="DISARM_attacks_pooled")
    else:
        long_df_csi = long_df
        long_df_sml = long_df

    stage05_dir = Path(input_path).resolve().parent
    profile_summary_path = stage05_dir / "profile_level_effectivity.csv"
    profile_wide_path = stage05_dir / "profile_sem_wide.csv"
    if not profile_summary_path.exists() or not profile_wide_path.exists():
        raise RuntimeError("Stage 06 requires profile_level_effectivity.csv and profile_sem_wide.csv from Stage 05.")

    profile_summary_df = pd.read_csv(profile_summary_path)
    profile_df = pd.read_csv(profile_wide_path)
    profile_df = _ensure_standardized_columns(
        profile_df,
        [*CORE_CONTINUOUS_MODERATORS, *EXPLORATORY_CONTINUOUS_MODERATORS],
    )

    indicator_columns = _indicator_columns(profile_df)
    structural_terms = _core_structural_terms(profile_df)
    control_terms = _available(CONTROL_COLUMNS, profile_df)
    profile_terms = _all_profile_terms(profile_df)
    primary_outcome = _primary_outcome_column(profile_df)

    sem_result, factor_scores = _fit_sem(profile_df, indicator_columns=indicator_columns, structural_terms=structural_terms)

    multivariate_terms = [*profile_terms, *control_terms]
    multivariate_formula = _build_formula(primary_outcome, multivariate_terms)
    ols_model = smf.ols(multivariate_formula, data=profile_df).fit(cov_type="HC3")
    conf_int = ols_model.conf_int()
    ols_table = pd.DataFrame(
        {
            "term": ols_model.params.index,
            "estimate": ols_model.params.values,
            "std_error": ols_model.bse.values,
            "p_value": ols_model.pvalues.values,
            "conf_low": conf_int[0].values,
            "conf_high": conf_int[1].values,
        }
    )
    ols_table = _apply_bh_qvalues(ols_table, "p_value", "q_value", exclude_terms=["Intercept"])

    bootstrap_table = _bootstrap_ols(
        df=profile_df,
        formula=multivariate_formula,
        terms=list(ols_model.params.index),
        n_bootstrap=config.bootstrap_samples,
        seed=config.seed,
    )
    exploratory_table = _fit_exploratory_models(
        profile_df=profile_df,
        multivariate_params=ols_table,
        control_terms=control_terms,
        candidate_terms=profile_terms,
        outcome=primary_outcome,
    )
    # Use all available profile features (means + facets + demographics) for the
    # conditional susceptibility index — richer than SEM/OLS which use means only.
    # feature_columns=None triggers _default_feature_columns which picks all profile_cont_/profile_cat__ columns.
    conditional_outcome = "adversarial_effectivity" if "adversarial_effectivity" in long_df.columns and long_df["adversarial_effectivity"].notna().any() else "abs_delta_score"
    # Exclude synthetic proxies and the three run_1 non-survey-mappable inventories.
    # _INVENTORY_EXCLUSION_PREFIXES are prefix patterns; conditional_susceptibility
    # uses an exact list, so expand them against the actual long_df columns.
    _inv_excl_exact = [
        col for col in long_df.columns
        if any(col.startswith(p) for p in _INVENTORY_EXCLUSION_PREFIXES)
    ]
    # Construct-level moderator set (~46 *_mean_pct aggregates + age). Each
    # per-(attack, opinion) task ridge sees only ~10-15 profiles, so using the
    # higher-order constructs (not ~250 facets or ~1500 categorical dummies)
    # keeps every task model well-posed and fast, with EB shrinkage pooling
    # across the 106 opinion-leaf tasks.
    conditional_feature_terms = _conditional_feature_terms(long_df_csi)
    conditional_fit = fit_conditional_susceptibility_index(
        long_df=long_df_csi,
        outcome_metric=conditional_outcome,
        feature_columns=conditional_feature_terms,
        excluded_feature_columns=[
            "profile_cont_heuristic_shift_sensitivity_proxy",
            "profile_cont_resilience_index",
            *_inv_excl_exact,
        ],
        seed=config.seed,
        compute_hierarchy=True,
        # The per-task rank-CI block bootstrap dominates Stage 06 runtime; 120
        # resamples give stable enough rankings without the full bootstrap budget.
        bootstrap_samples=min(int(config.bootstrap_samples or 0), 120),
        shrinkage_strength=0.20,
    )
    # Scalable, overfitting-resistant block-wise family model. Each ontology family
    # (Big Five, the political-psychology inventories, demographics, ...) is fit as
    # its own regularized sub-model on the profile-level mean effect, then combined
    # by a heavily-regularized meta-learner under nested CV. This stays well-posed no
    # matter how wide the profile ontology is, unlike a single full-feature ridge.
    blockwise_result = None
    try:
        _bw_feats = [c for c in long_df.columns if c.startswith("profile_cont_")]
        _bw_profile = (
            long_df.dropna(subset=[conditional_outcome])
            .groupby("profile_id")
            .agg(_y=(conditional_outcome, "mean"), **{c: (c, "first") for c in _bw_feats})
            .reset_index()
        )
        blockwise_result = fit_blockwise_family_susceptibility(
            _bw_profile, outcome_column="_y", feature_columns=_bw_feats, seed=config.seed
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Stage 06 block-wise family susceptibility model skipped: %s", exc)

    # Inferential statistics: significance-test every main individual-layer claim
    # (attack works, tactic / domain / complexity differences, dose-response,
    # direction, between-profile heterogeneity), clustering-aware and BH-FDR
    # corrected. Saved as CSV + JSON + a report section.
    stats_results = None
    stats_summary = ""
    try:
        stats_results, stats_summary = run_individual_layer_statistics(long_df)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Stage 06 individual-layer statistics skipped: %s", exc)

    weight_table = build_conditional_weight_table(conditional_fit.artifact)
    weight_table["moderator_label"] = weight_table["term"].map(_pretty_moderator_label)
    weight_table["ontology_group"] = weight_table["term"].map(_moderator_group)
    weight_table["direction"] = np.where(
        weight_table["weighted_mean_estimate"] >= 0,
        "higher_effectivity",
        "lower_effectivity",
    )
    weight_table["estimate"] = weight_table["weighted_mean_estimate"]
    weight_table["mean_abs_estimate"] = weight_table["weighted_mean_abs_estimate"]

    merge_cols = ["profile_id", "mean_abs_delta_score", "mean_signed_delta_score"]
    if "mean_adversarial_effectivity" in profile_df.columns:
        merge_cols.append("mean_adversarial_effectivity")
    profile_index_df = conditional_fit.profile_scores.merge(
        profile_df[merge_cols],
        on="profile_id",
        how="left",
    )
    obs_col = "mean_adversarial_effectivity" if "mean_adversarial_effectivity" in profile_index_df.columns else "mean_abs_delta_score"
    profile_index_df["observed_effectivity_pct"] = (
        profile_index_df[obs_col].rank(method="average", pct=True) * 100.0
    )
    if conditional_fit.bootstrap_ci is not None:
        profile_index_df["rank_ci_low"] = profile_index_df["profile_id"].map(
            lambda pid: conditional_fit.bootstrap_ci.rank_ci.get(pid, (np.nan, np.nan))[0]
        )
        profile_index_df["rank_ci_high"] = profile_index_df["profile_id"].map(
            lambda pid: conditional_fit.bootstrap_ci.rank_ci.get(pid, (np.nan, np.nan))[1]
        )
        profile_index_df["rank_sd"] = profile_index_df["profile_id"].map(
            conditional_fit.bootstrap_ci.rank_sd
        )
    contribution_breakdown_df = conditional_fit.contribution_breakdown
    ridge_summary_lookup = (
        conditional_fit.task_coefficients.loc[conditional_fit.task_coefficients["term"] != "Intercept"]
        .groupby("term", as_index=False)
        .agg(
            ridge_mean_estimate=("estimate", "mean"),
            ridge_mean_abs_estimate=("estimate", lambda s: float(np.mean(np.abs(s)))),
        )
    )
    exploratory_table = exploratory_table.merge(
        ridge_summary_lookup,
        left_on="moderator_column",
        right_on="term",
        how="left",
    ).drop(columns=["term"], errors="ignore")
    exploratory_table = exploratory_table.merge(
        weight_table[["term", "normalized_weight_pct"]],
        left_on="moderator_column",
        right_on="term",
        how="left",
    ).drop(columns=["term"], errors="ignore")

    # ------------------------------------------------------------------
    # Full-feature moderation models
    #
    # The aggregate OLS uses only 8 hardcoded Big Five / demographics
    # predictors — it ignores ~92 theoretically motivated features.
    #
    # Three complementary estimators on all available profile features:
    #
    # 1. Ridge (primary effect estimator): does NOT zero out correlated
    #    features. Gives continuous, interpretable coefficient estimates
    #    for ALL predictors. Ridge is correct here because Big Five facets
    #    are collinear; LASSO/EN arbitrarily zeroes correlated features
    #    keeping only one representative (e.g. "Extraversion·Gregariousness"
    #    and nothing else), which is misleading. Ridge shows the full
    #    theoretical picture with effect sizes for political engagement,
    #    digital literacy, dual process, etc.
    #
    # 2. Elastic Net / LASSO (selector): identifies features that survive
    #    strict penalisation. With ICC≈0, may select very few features —
    #    this is a data quality signal, not a model failure.
    #
    # 3. Random Forest (non-linear check): OOB R² as upper bound for
    #    non-linear profile moderation.
    #
    # CV-R² near-zero (or negative) when ICC≈0 is EXPECTED — the model
    # correctly finds no aggregate signal. Coefficient directions from
    # ridge are still theoretically interpretable.
    # ------------------------------------------------------------------
    all_feature_terms = _analysis_feature_terms(profile_df)
    LOGGER.info("Fitting Ridge (all %d features) ...", len(all_feature_terms))
    ridge_full_result = _fit_ridge_full_features(
        df=profile_df,
        outcome=primary_outcome,
        feature_terms=all_feature_terms,
        seed=config.seed,
    )
    LOGGER.info(
        "Ridge full-feature: CV-R²=%.3f (±%.3f), α=%.4f",
        ridge_full_result["cv_r2"], ridge_full_result["cv_r2_std"], ridge_full_result["alpha"],
    )

    LOGGER.info("Fitting Elastic Net (LASSO selector) on %d features ...", len(all_feature_terms))
    enet_result = _fit_elastic_net(
        df=profile_df,
        outcome=primary_outcome,
        feature_terms=all_feature_terms,
        seed=config.seed,
    )
    LOGGER.info(
        "Elastic Net: CV-R²=%.3f (±%.3f), α=%.4f, l1_ratio=%.2f, %d/%d features selected",
        enet_result["cv_r2"], enet_result["cv_r2_std"],
        enet_result["alpha"], enet_result["l1_ratio"],
        enet_result["n_features_selected"], enet_result["n_features_total"],
    )

    LOGGER.info("Fitting Random Forest on %d features ...", len(all_feature_terms))
    rf_result = _fit_random_forest(
        df=profile_df,
        outcome=primary_outcome,
        feature_terms=all_feature_terms,
        seed=config.seed,
    )
    LOGGER.info("Random Forest OOB R²=%.3f", rf_result["oob_r2"])

    LOGGER.info("Building profile correlation network on %d features ...", len(all_feature_terms))
    network_result = _fit_network_analysis(
        df=profile_df,
        feature_terms=all_feature_terms,
        corr_threshold=0.15,
        seed=config.seed,
    )
    LOGGER.info(
        "Network: %d nodes, %d edges, density=%.3f, %d communities, modularity=%.3f",
        network_result["global_metrics"]["n_nodes"],
        network_result["global_metrics"]["n_edges"],
        network_result["global_metrics"]["density"],
        network_result["global_metrics"]["n_communities"],
        network_result["global_metrics"]["modularity_score"],
    )

    # Merge ridge and RF estimates into exploratory_table (Big Five features only)
    ridge_lookup = ridge_full_result["coeff_df"].set_index("term")["ridge_estimate"].to_dict()
    rf_lookup = rf_result["importance_df"].set_index("term")["permutation_importance_mean"].to_dict()
    if not exploratory_table.empty:
        exploratory_table["ridge_estimate"] = exploratory_table["moderator_column"].map(ridge_lookup).astype(float)
        exploratory_table["rf_permutation_importance"] = exploratory_table["moderator_column"].map(rf_lookup).astype(float)

    # Build expanded moderator table: ALL ~100 features with ridge estimates
    # + OLS estimates where available. This is the primary moderator forest input.
    ridge_full_with_enet = ridge_full_result["coeff_df"].copy()
    enet_lookup = enet_result["coeff_df"].set_index("term")["elastic_net_estimate"].to_dict()
    ridge_full_with_enet["elastic_net_estimate"] = ridge_full_with_enet["term"].map(enet_lookup)
    expanded_moderator_table = _build_expanded_moderator_table(
        ridge_coeff_df=ridge_full_with_enet,
        ols_exploratory_df=exploratory_table,
    )

    out = Path(output_dir)
    spec_txt = out / "sem_model_spec.txt"
    profile_formula_txt = out / "profile_multivariate_model_spec.txt"
    sem_json = out / "sem_result.json"
    sem_coeff_csv = out / "sem_coefficients.csv"
    sem_fit_json = out / "sem_fit_indices.json"
    ols_txt = out / "ols_robust_summary.txt"
    ols_params_csv = out / "ols_robust_params.csv"
    bootstrap_csv = out / "bootstrap_primary_params.csv"
    exploratory_csv = out / "exploratory_moderator_comparison.csv"
    weight_table_csv = out / "moderator_weight_table.csv"
    profile_index_csv = out / "profile_susceptibility_index.csv"
    contribution_breakdown_csv = out / "profile_susceptibility_breakdown.csv"
    latent_scores_csv = out / "latent_attack_effectivity_scores.csv"
    profile_summary_copy_csv = out / "profile_level_effectivity.csv"
    profile_wide_copy_csv = out / "profile_sem_wide.csv"
    ridge_coeff_csv = out / "conditional_susceptibility_task_coefficients.csv"
    ridge_summary_csv = out / "conditional_susceptibility_task_summary.csv"
    conditional_artifact_json = out / "conditional_susceptibility_artifact.json"
    report_txt = out / "moderation_report.txt"
    assumptions_json = out / "assumption_register.json"
    critiques_json = out / "peer_review_critiques.json"
    methodology_txt = out / "methodology_audit.txt"
    enet_coeff_csv = out / "elastic_net_coefficients.csv"
    enet_selected_csv = out / "elastic_net_selected.csv"
    enet_summary_json = out / "elastic_net_summary.json"
    rf_importance_csv = out / "rf_feature_importance.csv"
    rf_summary_json = out / "rf_summary.json"
    ridge_full_coeff_csv = out / "ridge_full_coefficients.csv"
    ridge_full_summary_json = out / "ridge_full_summary.json"
    expanded_moderator_csv = out / "expanded_moderator_comparison.csv"
    bootstrap_rank_csv = out / "conditional_susceptibility_bootstrap_ranks.csv"
    bootstrap_feature_sd_csv = out / "conditional_susceptibility_bootstrap_feature_sd.csv"
    group_contribution_csv = out / "conditional_susceptibility_group_contributions.csv"
    quality_diagnostics_json = out / "analysis_quality_diagnostics.json"
    network_centrality_csv = out / "profile_network_centrality.csv"
    network_edges_csv = out / "profile_network_edges.csv"
    network_layout_csv = out / "profile_network_layout.csv"
    network_global_json = out / "profile_network_global_metrics.json"

    write_text(spec_txt, sem_result.model_formula)
    write_text(profile_formula_txt, multivariate_formula)
    write_json(sem_json, sem_result.model_dump())
    sem_coeff_df = pd.DataFrame([coeff.model_dump() for coeff in sem_result.coefficients])
    if not sem_coeff_df.empty and "p_value" in sem_coeff_df.columns:
        sem_coeff_df = _apply_bh_qvalues(sem_coeff_df, "p_value", "q_value")
    sem_coeff_df.to_csv(sem_coeff_csv, index=False)
    write_json(sem_fit_json, sem_result.fit_indices)
    ols_summary_text = _safe_ols_summary(ols_model)
    write_text(ols_txt, ols_summary_text)
    ols_table.to_csv(ols_params_csv, index=False)
    bootstrap_table.to_csv(bootstrap_csv, index=False)
    exploratory_table.to_csv(exploratory_csv, index=False)
    weight_table.to_csv(weight_table_csv, index=False)
    profile_index_df.to_csv(profile_index_csv, index=False)
    contribution_breakdown_df.to_csv(contribution_breakdown_csv, index=False)
    conditional_fit.task_coefficients.to_csv(ridge_coeff_csv, index=False)
    conditional_fit.task_summary.to_csv(ridge_summary_csv, index=False)
    write_json(conditional_artifact_json, conditional_fit.artifact.model_dump())
    bootstrap_rank_df = _bootstrap_rank_table(conditional_fit)
    bootstrap_feature_sd_df = _bootstrap_feature_sd_table(conditional_fit)
    bootstrap_rank_df.to_csv(bootstrap_rank_csv, index=False)
    bootstrap_feature_sd_df.to_csv(bootstrap_feature_sd_csv, index=False)
    if conditional_fit.group_contribution_breakdown is not None:
        conditional_fit.group_contribution_breakdown.to_csv(group_contribution_csv, index=False)
    # Individual-layer inferential statistics (significance tests + effect sizes).
    if stats_results is not None and not stats_results.empty:
        stats_results.to_csv(out / "individual_layer_statistical_tests.csv", index=False)
        write_json(
            out / "individual_layer_statistical_tests.json",
            {
                "n_tests": int(len(stats_results)),
                "n_significant": int(stats_results["significant"].sum()),
                "tests": stats_results.to_dict(orient="records"),
                "notes": [
                    "Unit of analysis respects clustering: scenario-level means for attack and "
                    "opinion-domain contrasts, opinion-leaf means for the direction contrast.",
                    "Pairwise families are Benjamini-Hochberg FDR corrected (q_value).",
                    "Effect sizes: Cohen d_z (Wilcoxon), epsilon^2 (Kruskal-Wallis), rank-biserial "
                    "(Mann-Whitney), Spearman rho (trend), ICC(1) (heterogeneity).",
                ],
            },
        )

    # Block-wise family susceptibility model (scalable, overfitting-resistant).
    if blockwise_result is not None:
        blockwise_result.family_table.to_csv(out / "blockwise_family_susceptibility.csv", index=False)
        blockwise_result.profile_scores.to_csv(out / "blockwise_profile_susceptibility.csv", index=False)
        write_json(
            out / "blockwise_family_susceptibility.json",
            {
                "stacked_oos_r2": blockwise_result.stacked_oos_r2,
                "n_profiles": blockwise_result.n_profiles,
                "n_families": blockwise_result.n_families,
                "family_table": blockwise_result.family_table.to_dict(orient="records"),
                "notes": blockwise_result.notes,
            },
        )
    if conditional_fit.hierarchical_decomposition is not None:
        hier = conditional_fit.hierarchical_decomposition
        write_json(
            out / "conditional_susceptibility_hierarchical_decomposition.json",
            {
                "full_model_cv_r2": hier.full_model_cv_r2,
                "group_marginal_r2": hier.group_marginal_r2,
                "group_relative_importance_pct": hier.group_relative_importance_pct,
                "task_group_r2": {k: v for k, v in hier.task_group_r2.items()},
                "notes": [
                    "marginal_r2 = full model CV-R2 minus CV-R2 of model with that group removed (leave-one-group-out).",
                    "Positive marginal_r2 means removing the group reduces predictive accuracy.",
                    "relative_importance_pct = |marginal_r2| / sum(|marginal_r2|) * 100.",
                ],
            },
        )
    if "latent_attack_effectivity_factor_score" in profile_df.columns:
        profile_df[["profile_id", "latent_attack_effectivity_factor_score"]].to_csv(latent_scores_csv, index=False)
    profile_summary_df.to_csv(profile_summary_copy_csv, index=False)
    profile_df.to_csv(profile_wide_copy_csv, index=False)
    enet_result["coeff_df"].to_csv(enet_coeff_csv, index=False)
    enet_result["selected_df"].to_csv(enet_selected_csv, index=False)
    rf_result["importance_df"].to_csv(rf_importance_csv, index=False)
    ridge_full_result["coeff_df"].to_csv(ridge_full_coeff_csv, index=False)
    expanded_moderator_table.to_csv(expanded_moderator_csv, index=False)
    quality_diagnostics = _build_quality_diagnostics(long_df)
    write_json(quality_diagnostics_json, quality_diagnostics)
    network_result["centrality_df"].to_csv(network_centrality_csv, index=False)
    if not network_result["edge_df"].empty:
        network_result["edge_df"].to_csv(network_edges_csv, index=False)
    network_result["layout_df"].to_csv(network_layout_csv, index=False)
    write_json(network_global_json, network_result["global_metrics"])
    write_json(enet_summary_json, {
        "r2_train": enet_result["r2_train"],
        "cv_r2": enet_result["cv_r2"],
        "cv_r2_std": enet_result["cv_r2_std"],
        "alpha": enet_result["alpha"],
        "l1_ratio": enet_result["l1_ratio"],
        "n_features_total": enet_result["n_features_total"],
        "n_features_selected": enet_result["n_features_selected"],
        "note": (
            "Elastic Net selector. CV-R² via nested 5-fold CV on standardised features. "
            "Higher l1_ratio values behave more like LASSO; use ridge_full for interpretable "
            "direction estimates across all retained predictors."
        ),
    })
    write_json(rf_summary_json, {
        "oob_r2": rf_result["oob_r2"],
        "n_estimators": rf_result["n_estimators"],
        "n_features": rf_result["n_features"],
        "note": "OOB R² is an unbiased estimate of held-out accuracy (each tree scored on samples it never saw during training).",
    })
    write_json(ridge_full_summary_json, {
        "cv_r2": ridge_full_result["cv_r2"],
        "cv_r2_std": ridge_full_result["cv_r2_std"],
        "alpha": ridge_full_result["alpha"],
        "n_features": len(all_feature_terms),
        "note": (
            "Ridge on all profile features (StandardScaler). CV-R² via 5-fold. "
            "Unlike LASSO/EN, ridge retains all coefficients — use for effect direction "
            "and relative magnitude interpretation across ALL predictors."
        ),
    })

    # ICC computation for hierarchical nesting diagnostics
    icc_results: Dict[str, object] = {}
    for icc_outcome in ["abs_delta_score", "delta_score", "adversarial_effectivity"]:
        if icc_outcome in long_df.columns and long_df[icc_outcome].notna().sum() > 0:
            icc_results[icc_outcome] = _compute_icc(long_df, outcome=icc_outcome)
    if icc_results:
        write_json(out / "intraclass_correlation.json", icc_results)
        LOGGER.info("ICC results: %s", {k: v.get("icc1") for k, v in icc_results.items()})

    # ── current design: advanced inferential layer (multilevel ICC, mixed-effects,
    # permutation FDR, BCa cluster bootstrap, Bayesian rank stability,
    # signed-network diagnostics).  All sub-results are non-fatal.
    LOGGER.info("Running advanced inferential layer ...")
    bootstrap_long_for_advanced = bootstrap_rank_df.copy() if isinstance(bootstrap_rank_df, pd.DataFrame) else pd.DataFrame()
    if not bootstrap_long_for_advanced.empty and "rank" not in bootstrap_long_for_advanced.columns:
        # Try to derive a long form from the wide bootstrap output
        rank_cols = [c for c in bootstrap_long_for_advanced.columns if c.startswith("rank_iter_")]
        if rank_cols and "profile_id" in bootstrap_long_for_advanced.columns:
            bootstrap_long_for_advanced = bootstrap_long_for_advanced.melt(
                id_vars=["profile_id"], value_vars=rank_cols,
                var_name="iteration", value_name="rank",
            )
    # Headline linear mixed-effects moderation (random intercept per scenario)
    # needs a parsimonious fixed-effect panel: with ~100 between-scenario units a
    # 47-term LMM is rank-deficient and will not converge. Use the Elastic-Net
    # selected moderators (data-driven, not hand-picked) as the LMM/advanced fixed
    # effects; fall back to the full construct panel only if EN selected too few.
    _en_selected = (
        enet_result.get("selected_df", pd.DataFrame())["term"].tolist()
        if isinstance(enet_result.get("selected_df"), pd.DataFrame) and not enet_result["selected_df"].empty
        else []
    )
    headline_feature_terms = [t for t in _en_selected if t in long_df.columns and long_df[t].nunique(dropna=True) > 1]
    if len(headline_feature_terms) < 3:
        # Data-driven fallback for small integrated test panels: choose the
        # strongest construct-level moderators by profile-level univariate
        # association, keeping the LMM estimable without hand-picking age /
        # Big Five or drowning it in hundreds of profile facets.
        candidate_terms = [t for t in all_feature_terms if t in profile_df.columns and profile_df[t].nunique(dropna=True) > 1]
        scored_terms: List[Tuple[float, str]] = []
        for term in candidate_terms:
            try:
                corr = abs(float(np.corrcoef(profile_df[term].astype(float), profile_df[primary_outcome].astype(float))[0, 1]))
            except Exception:
                corr = 0.0
            if np.isfinite(corr):
                scored_terms.append((corr, term))
        scored_terms.sort(reverse=True)
        headline_feature_terms = [term for _, term in scored_terms[:12]]
    headline_feature_terms = [t for t in headline_feature_terms if t in long_df.columns][:12]
    LOGGER.info("Advanced/mixed-effects layer fixed effects: %d moderators", len(headline_feature_terms))
    advanced_bundle = run_advanced_inferential(
        long_df=long_df,
        feature_cols=headline_feature_terms,
        bootstrap_ranks_long=bootstrap_long_for_advanced,
        network_edge_df=network_result.get("edge_df"),
        network_centrality_df=network_result.get("centrality_df"),
        n_permutations=200,
        n_bootstrap=min(int(getattr(config, "bootstrap_samples", 300) or 300), 400),
        seed=config.seed,
    )

    multilevel_icc_path = out / "advanced_multilevel_icc.json"
    mixed_effects_csv = out / "advanced_mixed_effects_coefficients.csv"
    perm_csv = out / "advanced_permutation_importance.csv"
    bca_csv = out / "advanced_bca_bootstrap_ridge.csv"
    rank_stability_csv = out / "advanced_rank_stability.csv"
    network_diagnostics_path = out / "advanced_network_diagnostics.json"

    write_json(multilevel_icc_path, {
        "icc_profile": advanced_bundle.multilevel_icc.icc_profile,
        "icc_attack": advanced_bundle.multilevel_icc.icc_attack,
        "icc_opinion": advanced_bundle.multilevel_icc.icc_opinion,
        "icc_residual": advanced_bundle.multilevel_icc.icc_residual,
        "n_profiles": advanced_bundle.multilevel_icc.n_profiles,
        "n_attacks": advanced_bundle.multilevel_icc.n_attacks,
        "n_opinions": advanced_bundle.multilevel_icc.n_opinions,
        "n_obs": advanced_bundle.multilevel_icc.n_obs,
        "method": advanced_bundle.multilevel_icc.method,
        "converged": advanced_bundle.multilevel_icc.converged,
        "notes": advanced_bundle.multilevel_icc.notes,
    })
    mixed_effects_result = advanced_bundle.mixed_effects
    if mixed_effects_result.coefficients.empty and headline_feature_terms:
        LOGGER.warning(
            "Advanced bundle returned no mixed-effects coefficients; refitting direct LMM with %d terms.",
            len(headline_feature_terms),
        )
        mixed_effects_result = fit_mixed_effects_moderation(
            long_df,
            outcome_col=conditional_outcome,
            feature_cols=headline_feature_terms,
            seed=config.seed,
        )
        if mixed_effects_result.coefficients.empty and len(headline_feature_terms) > 6:
            mixed_effects_result = fit_mixed_effects_moderation(
                long_df,
                outcome_col=conditional_outcome,
                feature_cols=headline_feature_terms[:6],
                seed=config.seed,
            )
    if not mixed_effects_result.coefficients.empty:
        mixed_effects_result.coefficients.to_csv(mixed_effects_csv, index=False)
    if not advanced_bundle.permutation_importance.table.empty:
        advanced_bundle.permutation_importance.table.to_csv(perm_csv, index=False)
    if not advanced_bundle.bca_bootstrap.table.empty:
        advanced_bundle.bca_bootstrap.table.to_csv(bca_csv, index=False)
    if not advanced_bundle.rank_stability.table.empty:
        advanced_bundle.rank_stability.table.to_csv(rank_stability_csv, index=False)
    write_json(network_diagnostics_path, {
        "n_nodes": advanced_bundle.network_diagnostics.n_nodes,
        "n_edges": advanced_bundle.network_diagnostics.n_edges,
        "n_communities": advanced_bundle.network_diagnostics.n_communities,
        "modularity": advanced_bundle.network_diagnostics.modularity,
        "sbm_n_blocks": advanced_bundle.network_diagnostics.sbm_n_blocks,
        "structural_balance_index": advanced_bundle.network_diagnostics.structural_balance_index,
        "triangle_balance_share": advanced_bundle.network_diagnostics.triangle_balance_share,
        "signed_modularity": advanced_bundle.network_diagnostics.signed_modularity,
        "notes": advanced_bundle.network_diagnostics.notes,
    })

    # ── current design: scenario-level ML moderation stack ──────────────────────────
    # Primary conditional analysis: GroupKFold-by-profile model ladder,
    # held-out boosted-tree importance, cluster-robust feature x context
    # moderation scans with BH-FDR, and cluster-bootstrap effect summaries.
    LOGGER.info("Running scenario-level ML moderation stack ...")
    scenario_feature_terms = _analysis_feature_terms(long_df)
    scenario_outcome = conditional_outcome
    scenario_bundle = run_scenario_ml(
        long_df=long_df_sml,
        outcome_col=scenario_outcome,
        feature_cols=scenario_feature_terms,
        n_bootstrap=min(int(config.bootstrap_samples or 500), 600),
        seed=config.seed,
    )

    def _decorate_terms(df_in: pd.DataFrame, term_col: str) -> pd.DataFrame:
        if df_in.empty or term_col not in df_in.columns:
            return df_in
        out_df = df_in.copy()
        out_df["label"] = out_df[term_col].map(_pretty_moderator_label)
        out_df["ontology_group"] = out_df[term_col].map(_moderator_group)
        return out_df

    scenario_ladder_csv = out / "scenario_model_ladder.csv"
    scenario_ladder_folds_csv = out / "scenario_model_ladder_folds.csv"
    scenario_importance_csv = out / "scenario_heldout_importance.csv"
    scan_by_attack_csv = out / "moderation_scan_by_attack.csv"
    scan_by_opinion_csv = out / "moderation_scan_by_opinion.csv"
    scan_pooled_csv = out / "moderation_scan_pooled.csv"
    effect_attacks_csv = out / "effect_summary_attacks.csv"
    effect_opinions_csv = out / "effect_summary_opinions.csv"
    effect_cells_csv = out / "effect_summary_cells.csv"
    effect_profiles_csv = out / "effect_summary_profiles.csv"
    scenario_meta_json = out / "scenario_ml_summary.json"

    if not scenario_bundle.model_ladder.table.empty:
        scenario_bundle.model_ladder.table.to_csv(scenario_ladder_csv, index=False)
        scenario_bundle.model_ladder.fold_table.to_csv(scenario_ladder_folds_csv, index=False)
    if not scenario_bundle.heldout_importance.table.empty:
        _decorate_terms(scenario_bundle.heldout_importance.table, "term").to_csv(scenario_importance_csv, index=False)
    if not scenario_bundle.moderation_scan.by_attack.empty:
        _decorate_terms(scenario_bundle.moderation_scan.by_attack, "feature").to_csv(scan_by_attack_csv, index=False)
    if not scenario_bundle.moderation_scan.by_opinion.empty:
        _decorate_terms(scenario_bundle.moderation_scan.by_opinion, "feature").to_csv(scan_by_opinion_csv, index=False)
    if not scenario_bundle.moderation_scan.pooled.empty:
        _decorate_terms(scenario_bundle.moderation_scan.pooled, "feature").to_csv(scan_pooled_csv, index=False)
    if not scenario_bundle.effect_summaries.attack_effects.empty:
        scenario_bundle.effect_summaries.attack_effects.to_csv(effect_attacks_csv, index=False)
        scenario_bundle.effect_summaries.opinion_effects.to_csv(effect_opinions_csv, index=False)
        scenario_bundle.effect_summaries.cell_effects.to_csv(effect_cells_csv, index=False)
        scenario_bundle.effect_summaries.profile_effects.to_csv(effect_profiles_csv, index=False)

    ladder_meta: Dict[str, Any] = {
        "outcome": scenario_outcome,
        "n_obs": scenario_bundle.model_ladder.n_obs,
        "n_profiles": scenario_bundle.model_ladder.n_profiles,
        "n_profile_features": scenario_bundle.model_ladder.n_features_profile,
        "cv_scheme": "GroupKFold by profile_id (generalization to unseen profiles)",
        "notes": [
            *scenario_bundle.model_ladder.notes,
            *scenario_bundle.effect_summaries.notes,
        ],
    }
    if not scenario_bundle.model_ladder.table.empty:
        ladder_meta["models"] = scenario_bundle.model_ladder.table.to_dict(orient="records")
    write_json(scenario_meta_json, ladder_meta)

    # supplementary: profile-configuration distance vs moderation-pattern distance
    try:
        mantel = profile_distance_moderation_test(
            long_df, feature_cols=scenario_feature_terms,
            outcome_col=scenario_outcome,
            n_permutations=2000, seed=config.seed,
        )
        if not mantel.pair_table.empty:
            mantel.pair_table.to_csv(out / "supplementary_profile_distance_pairs.csv", index=False)
        write_json(out / "supplementary_profile_distance_mantel.json", {
            "mantel_r": mantel.mantel_r, "mantel_p": mantel.mantel_p,
            "n_profiles": mantel.n_profiles, "n_pairs": mantel.n_pairs,
            "n_permutations": mantel.n_permutations, "notes": mantel.notes,
        })
    except Exception as exc:
        LOGGER.warning("Mantel supplementary failed: %s", exc)
    if not scenario_bundle.model_ladder.table.empty:
        LOGGER.info("Scenario model ladder:\n%s", scenario_bundle.model_ladder.table.to_string(index=False))

    write_text(
        report_txt,
        _render_report(
            long_df=long_df,
            profile_df=profile_df,
            sem_result=sem_result,
            multivariate_formula=multivariate_formula,
            ols_summary=ols_summary_text,
            ols_table=ols_table,
            bootstrap_table=bootstrap_table,
            exploratory_table=exploratory_table,
            profile_index_df=profile_index_df,
            weight_table=weight_table,
            task_summary_df=conditional_fit.task_summary,
            run_id=config.run_id,
            primary_outcome=primary_outcome,
            hierarchical_decomposition=conditional_fit.hierarchical_decomposition,
            enet_result=enet_result,
            rf_result=rf_result,
            ridge_full_result=ridge_full_result,
        ),
    )

    if not scenario_bundle.model_ladder.table.empty:
        ladder_lines = [
            "",
            "Scenario-Level Model Ladder (Primary Conditional Analysis, current design)",
            "--------------------------------------------------------------------",
            "GroupKFold by profile: CV-R2 measures generalization to UNSEEN profiles.",
            f"Outcome: {scenario_outcome}; n_obs={scenario_bundle.model_ladder.n_obs}; "
            f"n_profiles={scenario_bundle.model_ladder.n_profiles}; "
            f"profile features={scenario_bundle.model_ladder.n_features_profile}",
        ]
        for row in scenario_bundle.model_ladder.table.to_dict(orient="records"):
            ladder_lines.append(
                f"  {row['model']}: CV-R2={row['cv_r2_mean']:.4f} (sd {row['cv_r2_sd']:.4f}), "
                f"profile increment vs context+baseline={row['delta_r2_vs_context_baseline']:.4f}"
            )
        if not scenario_bundle.moderation_scan.pooled.empty:
            ladder_lines.append("Top pooled moderation effects (cluster-robust, BH-FDR):")
            for row in scenario_bundle.moderation_scan.pooled.head(8).to_dict(orient="records"):
                ladder_lines.append(
                    f"  {_pretty_moderator_label(row['feature'])}: b={row['estimate']:.3f}, "
                    f"p={row['p_value']:.4f}, q={row['q_value']:.4f}"
                )
        with open(report_txt, "a", encoding="utf-8") as handle:
            handle.write("\n" + "\n".join(ladder_lines) + "\n")

    if stats_results is not None and not stats_results.empty and stats_summary:
        with open(report_txt, "a", encoding="utf-8") as handle:
            handle.write("\n" + stats_summary + "\n")

    if blockwise_result is not None:
        bw_lines = [
            "",
            "Block-Wise Family Susceptibility Model (scalable, overfitting-resistant)",
            "----------------------------------------------------------------------",
            "Each ontology family is fit as its own regularized sub-model on the profile-level mean",
            "effect; a heavily-regularized meta-learner combines them under nested cross-validation.",
            f"Profiles={blockwise_result.n_profiles}; families={blockwise_result.n_families}; "
            f"stacked OOS R2 (nested, 3x)={blockwise_result.stacked_oos_r2:.4f}.",
            "Per-family standalone out-of-fold R2 (which trait family carries predictive signal):",
        ]
        for row in blockwise_result.family_table.to_dict(orient="records"):
            bw_lines.append(
                f"  {row['family']}: standalone_oof_R2={row['standalone_oof_r2']:+.4f}, "
                f"meta_weight={row['meta_weight']:+.3f}, n_features={row['n_features']}"
            )
        bw_lines.append(
            "Note: a near-zero stacked OOS R2 at small n reflects a genuinely weak stable-trait signal; "
            "the per-family decomposition isolates where signal concentrates and strengthens at production scale."
        )
        with open(report_txt, "a", encoding="utf-8") as handle:
            handle.write("\n" + "\n".join(bw_lines) + "\n")

    assumptions = build_assumption_register(long_df, sem_result)
    critiques = build_peer_review_critique_notes(long_df, sem_result)
    write_json(assumptions_json, assumptions)
    write_json(critiques_json, critiques)
    write_text(methodology_txt, render_methodology_audit_text(assumptions, critiques))

    output_files = [
        abs_path(spec_txt),
        abs_path(profile_formula_txt),
        abs_path(sem_json),
        abs_path(sem_coeff_csv),
        abs_path(sem_fit_json),
        abs_path(ols_txt),
        abs_path(ols_params_csv),
        abs_path(bootstrap_csv),
        abs_path(exploratory_csv),
        abs_path(weight_table_csv),
        abs_path(profile_index_csv),
        abs_path(contribution_breakdown_csv),
        abs_path(profile_summary_copy_csv),
        abs_path(profile_wide_copy_csv),
        abs_path(ridge_coeff_csv),
        abs_path(ridge_summary_csv),
        abs_path(conditional_artifact_json),
        abs_path(report_txt),
        abs_path(assumptions_json),
        abs_path(critiques_json),
        abs_path(methodology_txt),
        abs_path(enet_coeff_csv),
        abs_path(enet_selected_csv),
        abs_path(enet_summary_json),
        abs_path(rf_importance_csv),
        abs_path(rf_summary_json),
        abs_path(ridge_full_coeff_csv),
        abs_path(ridge_full_summary_json),
        abs_path(expanded_moderator_csv),
        abs_path(bootstrap_rank_csv),
        abs_path(bootstrap_feature_sd_csv),
        abs_path(quality_diagnostics_json),
        abs_path(network_centrality_csv),
        abs_path(network_layout_csv),
        abs_path(network_global_json),
    ]
    if network_edges_csv.exists():
        output_files.append(abs_path(network_edges_csv))
    if group_contribution_csv.exists():
        output_files.append(abs_path(group_contribution_csv))
    # advanced inferential outputs (current design)
    output_files.append(abs_path(multilevel_icc_path))
    output_files.append(abs_path(network_diagnostics_path))
    if mixed_effects_csv.exists():
        output_files.append(abs_path(mixed_effects_csv))
    if perm_csv.exists():
        output_files.append(abs_path(perm_csv))
    if bca_csv.exists():
        output_files.append(abs_path(bca_csv))
    if rank_stability_csv.exists():
        output_files.append(abs_path(rank_stability_csv))
    if latent_scores_csv.exists():
        output_files.append(abs_path(latent_scores_csv))
    # scenario-level ML outputs (current design)
    for candidate in [
        scenario_ladder_csv,
        scenario_ladder_folds_csv,
        scenario_importance_csv,
        scan_by_attack_csv,
        scan_by_opinion_csv,
        scan_pooled_csv,
        effect_attacks_csv,
        effect_opinions_csv,
        effect_cells_csv,
        effect_profiles_csv,
        scenario_meta_json,
        out / "supplementary_profile_distance_pairs.csv",
        out / "supplementary_profile_distance_mantel.json",
    ]:
        if candidate.exists():
            output_files.append(abs_path(candidate))

    manifest = StageArtifactManifest(
        stage_id="06",
        stage_name="construct_structural_equation_model",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(report_txt),
        output_files=output_files,
        record_count=len(profile_df),
        metadata={
            "sem_converged": sem_result.converged,
            "n_coefficients": len(sem_result.coefficients),
            "primary_moderator": config.primary_moderator,
            "bootstrap_samples": config.bootstrap_samples,
            "analysis_mode": analysis_mode,
            "indicator_columns": indicator_columns,
            "structural_terms": structural_terms,
            "multivariate_formula": multivariate_formula,
            "n_profile_moderators": len(profile_terms),
            "conditional_susceptibility_attack_leaves": conditional_fit.artifact.attack_leaves,
            "conditional_susceptibility_opinion_leaves": conditional_fit.artifact.opinion_leaves,
            "conditional_susceptibility_tasks": [task.task_key for task in conditional_fit.artifact.task_models],
            "conditional_shrinkage_strength": 0.20,
            "conditional_bootstrap_samples": config.bootstrap_samples,
            "ridge_full_cv_r2": ridge_full_result["cv_r2"],
            "elastic_net_cv_r2": enet_result["cv_r2"],
            "elastic_net_n_selected": enet_result["n_features_selected"],
            "elastic_net_n_total": enet_result["n_features_total"],
            "rf_oob_r2": rf_result["oob_r2"],
            "baseline_fallback_rate": quality_diagnostics.get("baseline_fallback_used_rate"),
            "post_fallback_rate": quality_diagnostics.get("post_fallback_used_rate"),
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 06 - SEM construction")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-moderator", default="posthoc_profile_susceptibility_index")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage06Config(
        stage_name="construct_structural_equation_model",
        run_id=args.run_id,
        seed=args.seed,
        primary_moderator=args.primary_moderator,
        bootstrap_samples=args.bootstrap_samples,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 06 completed: %s profiles", manifest.record_count)


if __name__ == "__main__":
    main()
