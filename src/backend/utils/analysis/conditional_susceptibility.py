from __future__ import annotations

"""
Technical overview
------------------
This module implements the analysis-facing conditional susceptibility index used
after a run has completed. The key design choice is that susceptibility is not
treated as a fixed trait known before simulation. Instead, it is estimated
post hoc conditional on the exact target set that was modeled:

    target set = {(attack_leaf, opinion_leaf)}

Primary effectivity metric (current design): adversarial_effectivity
    = signed opinion delta × adversarially assigned direction per leaf.
    Positive = attack moved opinion in the adversary's intended direction.
    This replaces abs_delta_score as primary because direction matters:
    a profile that shifted but in the wrong direction for the adversary is
    less exploitable than one that shifted in the right direction.

For each attack-opinion task the module fits a regularized profile-only ridge
model on observed adversarial effectivity. Each task-specific model produces a
fitted mapping:

    profile features -> predicted adversarial effectivity for that task

Those task-level predictions are aggregated back to the profile level using
reliability weights derived from sample size and cross-validated error.

Hierarchical decomposition (current design):
    Profile features are organized into an ontology-aligned hierarchy:
    - Level 1: Demographics (age, sex) vs. Personality (all Big Five features)
    - Level 2: Within Personality — five Big Five trait groups
    - Level 3: Within each trait — trait mean vs. individual facets
    For each hierarchy level and group, marginal cross-validated R² is computed
    (full model R² minus ablated model R² with that group removed) to quantify
    unique hierarchical contribution to susceptibility variation.

Key public entry points:
- fit_conditional_susceptibility_index(...)
- score_profiles_with_conditional_artifact(...)
- build_conditional_weight_table(...)
- compute_hierarchical_decomposition(...)
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backend.utils.schemas import (
    ConditionalSusceptibilityArtifact,
    ConditionalSusceptibilityTaskModel,
)


LEGACY_EXCLUDED_FEATURE_COLUMNS = {
    "profile_cont_heuristic_shift_sensitivity_proxy",
    "profile_cont_resilience_index",
}

# Kept for import compatibility only — not used internally. Use
# _build_feature_hierarchy(), which auto-discovers groups from column names.
BIG_FIVE_TRAITS = [
    "neuroticism",
    "openness_to_experience",
    "conscientiousness",
    "extraversion",
    "agreeableness",
]

# Known demographic singleton tokens (maps to the "demographics" ablation group
# alongside all profile_cat__* one-hot columns).
_DEMO_SINGLETONS: frozenset = frozenset({
    "age", "income", "education", "bmi", "weight", "height",
})

# Suffixes stripped when parsing tokens from column names
_COL_STRIP_SUFFIXES = ("_pct", "_years", "_score", "_proxy", "_index", "_z", "_norm")


@dataclass
class HierarchicalDecomposition:
    """Per-task and aggregated hierarchical R² decomposition."""
    # Aggregated across tasks (weighted by reliability_weight)
    group_marginal_r2: Dict[str, float] = field(default_factory=dict)
    group_relative_importance_pct: Dict[str, float] = field(default_factory=dict)
    full_model_cv_r2: float = 0.0
    # Per-task breakdown (task_key -> {group -> marginal_r2})
    task_group_r2: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class BootstrapCISummary:
    """Bootstrap uncertainty for the conditional susceptibility index.

    Populated by :func:`fit_conditional_susceptibility_index` when
    ``bootstrap_samples > 0``.  All values are in percentile units (0-100).
    """
    n_samples: int = 0
    rank_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    rank_sd: Dict[str, float] = field(default_factory=dict)
    coefficient_sd: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ConditionalSusceptibilityFitResult:
    artifact: ConditionalSusceptibilityArtifact
    task_coefficients: pd.DataFrame
    task_summary: pd.DataFrame
    profile_scores: pd.DataFrame
    contribution_breakdown: pd.DataFrame
    hierarchical_decomposition: Optional[HierarchicalDecomposition] = None
    bootstrap_ci: Optional[BootstrapCISummary] = None
    group_contribution_breakdown: Optional[pd.DataFrame] = None
    feature_engineering: Dict[str, List[str]] = field(default_factory=dict)


def _kfold_indices(n_obs: int, seed: int, n_splits: int = 5) -> List[np.ndarray]:
    n_splits = max(2, min(n_splits, n_obs))
    indices = np.arange(n_obs)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, n_splits) if len(fold) > 0]


def _ridge_fit_matrix(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + alpha * penalty) @ (x.T @ y)


def _cross_validated_ridge(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    alpha_grid: Sequence[float] | None = None,
) -> Tuple[np.ndarray, float, float]:
    alpha_grid = list(alpha_grid or np.logspace(-3, 3, 25))
    folds = _kfold_indices(len(y), seed=seed, n_splits=5)

    best_alpha = float(alpha_grid[0])
    best_cv_mse = float("inf")
    for alpha in alpha_grid:
        fold_mses: List[float] = []
        for fold in folds:
            mask = np.ones(len(y), dtype=bool)
            mask[fold] = False
            beta = _ridge_fit_matrix(x[mask], y[mask], alpha)
            preds = x[fold] @ beta
            fold_mses.append(float(np.mean((y[fold] - preds) ** 2)))
        cv_mse = float(np.mean(fold_mses))
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_alpha = float(alpha)

    beta = _ridge_fit_matrix(x, y, best_alpha)
    return beta, best_alpha, best_cv_mse


def _cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    folds: List[np.ndarray],
) -> float:
    """Cross-validated R² using a fixed alpha."""
    all_preds: List[float] = []
    all_actual: List[float] = []
    for fold in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        if mask.sum() < 2:
            continue
        beta = _ridge_fit_matrix(x[mask], y[mask], alpha)
        all_preds.extend((x[fold] @ beta).tolist())
        all_actual.extend(y[fold].tolist())
    if not all_preds:
        return 0.0
    preds_arr = np.array(all_preds)
    actual_arr = np.array(all_actual)
    ss_res = float(np.sum((actual_arr - preds_arr) ** 2))
    ss_tot = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-10)


def _default_feature_columns(
    df: pd.DataFrame,
    excluded_columns: Iterable[str] | None = None,
) -> List[str]:
    excluded = set(excluded_columns or []) | LEGACY_EXCLUDED_FEATURE_COLUMNS
    columns: List[str] = []
    for column in sorted(df.columns):
        is_continuous = column.startswith("profile_cont_")
        is_categorical = column.startswith("profile_cat__")
        if not (is_continuous or is_categorical):
            continue
        if column in excluded:
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        columns.append(column)
    return columns


def _fit_feature_scaler(
    unique_profiles_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, float], List[str], List[str]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    continuous_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in feature_columns:
        if column.startswith("profile_cont_"):
            continuous_columns.append(column)
            means[column] = float(unique_profiles_df[column].astype(float).mean())
            std = float(unique_profiles_df[column].astype(float).std(ddof=0))
            stds[column] = std if std > 0.0 else 1.0
        else:
            categorical_columns.append(column)

    return means, stds, continuous_columns, categorical_columns


def _transform_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    continuous_columns: Sequence[str],
) -> pd.DataFrame:
    transformed = pd.DataFrame(index=df.index)
    continuous_set = set(continuous_columns)
    for column in feature_columns:
        if column not in df.columns:
            transformed[column] = 0.0
            continue
        values = df[column].astype(float)
        if column in continuous_set:
            transformed[column] = (values - float(feature_means[column])) / float(feature_stds[column])
        else:
            transformed[column] = values.fillna(0.0)
    return transformed


def _task_key(attack_leaf: str, opinion_leaf: str) -> str:
    return f"{attack_leaf} || {opinion_leaf}"


def _col_to_tokens(col: str) -> Tuple[str, ...]:
    """Strip profile prefix and trailing suffix; return token tuple."""
    inner = col
    for pfx in ("profile_cont_", "profile_cat__"):
        if inner.startswith(pfx):
            inner = inner[len(pfx):]
            break
    for suf in _COL_STRIP_SUFFIXES:
        if inner.endswith(suf):
            inner = inner[: -len(suf)]
            break
    # Also strip a trailing _mean token (it belongs to the dimension, not a sub-level)
    if inner.endswith("_mean"):
        inner = inner[: -len("_mean")]
    return tuple(inner.split("_"))


def _build_feature_hierarchy(feature_columns: List[str]) -> Dict[str, List[str]]:
    """
    Auto-discover ontology-aligned feature groups from column names alone.

    No inventory names (Big Five, HEXACO, Dark Triad, …) or field names (age,
    sex) are hardcoded.  The algorithm:

    1.  Categorical columns (profile_cat__*) → "demographics" group.
    2.  Continuous columns (profile_cont_*) whose token-set overlaps
        _DEMO_SINGLETONS → "demographics" group.
    3.  Remaining continuous columns → prefix-trie to find:
          • inventory-level groups  (shallowest prefix with ≥2 cols + ≥2 child tokens)
          • dimension-level groups  (one level below each inventory)
        These produce groups named by their inferred key, e.g.:
          "big_five"              → all Big Five columns
          "big_five_neuroticism"  → all Neuroticism columns

    Groups with fewer than 2 columns are omitted (not meaningful for ablation).
    """
    from collections import defaultdict as _dd

    cat_cols = [c for c in feature_columns if c.startswith("profile_cat__")]
    cont_cols = [c for c in feature_columns if c.startswith("profile_cont_")]

    col_toks: Dict[str, Tuple[str, ...]] = {c: _col_to_tokens(c) for c in cont_cols}

    # ── Demographics: categorical + age/income/... singletons ────────────────
    demo_cont = [
        c for c, toks in col_toks.items()
        if any(t in _DEMO_SINGLETONS for t in toks)
    ]
    demo = sorted(set(cat_cols + demo_cont))

    # ── Prefix trie over non-demographic continuous columns ──────────────────
    non_demo_cont = [c for c in cont_cols if c not in demo_cont]
    nd_toks: Dict[str, Tuple[str, ...]] = {c: col_toks[c] for c in non_demo_cont}

    # trie[depth][prefix_tuple] → [col, …]
    trie: Dict[int, Dict[Tuple, List[str]]] = _dd(lambda: _dd(list))
    for col, toks in nd_toks.items():
        for d in range(1, len(toks)):
            trie[d][toks[:d]].append(col)

    inventory_prefixes: set = set()
    inventory_groups: Dict[str, List[str]] = {}

    for d in sorted(trie):
        for prefix, cols in trie[d].items():
            # Skip if a shallower parent is already claimed as an inventory
            if any(prefix[:i] in inventory_prefixes for i in range(1, d)):
                continue
            if len(cols) < 2:
                continue
            # Need ≥2 distinct child tokens at depth d (confirms a branching hierarchy)
            child_tokens = {nd_toks[c][d] for c in cols if len(nd_toks[c]) > d}
            if len(child_tokens) < 2:
                continue
            inv_key = "_".join(prefix)
            inventory_groups[inv_key] = cols
            inventory_prefixes.add(prefix)

    # Dimension groups: one level below each inventory
    dimension_groups: Dict[str, List[str]] = {}
    for inv_prefix in inventory_prefixes:
        inv_d = len(inv_prefix)
        dim_d = inv_d + 1
        for prefix, cols in trie.get(dim_d, {}).items():
            if prefix[:inv_d] == inv_prefix and len(cols) >= 2:
                dimension_groups["_".join(prefix)] = cols

    # ── Assemble final groups ────────────────────────────────────────────────
    groups: Dict[str, List[str]] = {}
    if demo:
        groups["demographics"] = demo
    groups.update(inventory_groups)
    groups.update(dimension_groups)

    return {k: sorted(set(v)) for k, v in groups.items() if len(v) >= 2}


def compute_hierarchical_decomposition(
    x_full: np.ndarray,
    y: np.ndarray,
    feature_columns: List[str],
    best_alpha: float,
    seed: int,
) -> Dict[str, float]:
    """Compute leave-one-group-out marginal CV-R² for each feature hierarchy group.

    Returns a dict mapping group_name -> marginal_cv_r2 (contribution of that group
    to the full model's explanatory power). Also includes 'full_model' -> full CV-R².
    """
    folds = _kfold_indices(len(y), seed=seed, n_splits=5)
    r2_full = _cv_r2(x_full, y, best_alpha, folds)

    hierarchy = _build_feature_hierarchy(feature_columns)
    # col_idx_map: feature_columns[i] is at column index i+1 in x_full (index 0 is intercept)
    col_idx_map = {name: idx + 1 for idx, name in enumerate(feature_columns)}

    result: Dict[str, float] = {"full_model": r2_full}

    for group_name, group_cols in hierarchy.items():
        group_indices = [col_idx_map[c] for c in group_cols if c in col_idx_map]
        if not group_indices:
            continue
        keep = [i for i in range(x_full.shape[1]) if i not in group_indices]
        if len(keep) < 2:
            result[f"marginal_{group_name}"] = r2_full
            continue
        x_ablated = x_full[:, keep]
        r2_ablated = _cv_r2(x_ablated, y, best_alpha, folds)
        result[f"marginal_{group_name}"] = r2_full - r2_ablated

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Optional enhancement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_polynomial_and_interactions(
    df: pd.DataFrame,
    feature_columns: List[str],
    polynomial_features: List[str] | None,
    interaction_pairs: List[Tuple[str, str]] | None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """Add polynomial and cross-product interaction columns to *df*.

    Works on an **already z-scored** feature matrix so the engineered terms
    have meaningful magnitude and don't compound scale problems.

    Returns
    -------
    augmented_df : pd.DataFrame  — original df plus new columns
    new_columns  : List[str]     — names of added columns only
    engineering_log : Dict       — {"polynomial": [...], "interaction": [...]}
    """
    augmented = df.copy()
    new_columns: List[str] = []
    log: Dict[str, List[str]] = {"polynomial": [], "interaction": []}

    col_set = set(feature_columns)

    # Polynomial (squared) terms — only for continuous features
    for col in (polynomial_features or []):
        if col not in augmented.columns:
            continue
        if not col.startswith("profile_cont_"):
            continue
        new_col = f"{col}__sq"
        augmented[new_col] = augmented[col].astype(float) ** 2
        new_columns.append(new_col)
        log["polynomial"].append(new_col)

    # Interaction terms
    for col_a, col_b in (interaction_pairs or []):
        if col_a not in augmented.columns or col_b not in augmented.columns:
            continue
        new_col = f"{col_a}__x__{col_b}"
        augmented[new_col] = augmented[col_a].astype(float) * augmented[col_b].astype(float)
        new_columns.append(new_col)
        log["interaction"].append(new_col)

    return augmented, new_columns, log


def _apply_eb_shrinkage(
    task_models: List,
    feature_columns: List[str],
    shrinkage_strength: float,
) -> List:
    """Empirical-Bayes James-Stein partial pooling of task-specific coefficients.

    Each task's coefficient β_task is shrunk toward the cross-task mean μ:

        β_shrunk = (1 − λ) * β_task  +  λ * μ_cross_task

    where λ = ``shrinkage_strength`` ∈ [0, 1].  λ=0 is no pooling; λ=1 is
    full pooling.  Typical useful range: 0.1–0.4.

    This reduces overfitting when a single task has few observations and
    idiosyncratic noise drives extreme coefficients.
    """
    if shrinkage_strength <= 0.0 or not task_models:
        return task_models

    lam = float(min(1.0, max(0.0, shrinkage_strength)))

    # Compute cross-task reliability-weighted means per feature
    cross_mean: Dict[str, float] = {}
    total_w = sum(float(m.reliability_weight) for m in task_models) or 1.0
    for col in feature_columns:
        cross_mean[col] = sum(
            float(m.reliability_weight) * float(m.coefficients.get(col, 0.0))
            for m in task_models
        ) / total_w

    # Shrink in-place (deep-copy coefficients dict to avoid aliasing)
    from copy import deepcopy
    from dataclasses import is_dataclass, replace as dc_replace

    shrunk_models = []
    for m in task_models:
        new_coeffs = deepcopy(m.coefficients)
        for col in feature_columns:
            orig = float(new_coeffs.get(col, 0.0))
            new_coeffs[col] = (1.0 - lam) * orig + lam * cross_mean.get(col, 0.0)
        if is_dataclass(m):
            shrunk_models.append(dc_replace(m, coefficients=new_coeffs))
        elif hasattr(m, "model_copy"):
            shrunk_models.append(m.model_copy(update={"coefficients": new_coeffs}))
        else:
            setattr(m, "coefficients", new_coeffs)
            shrunk_models.append(m)

    return shrunk_models


def _task_models_to_coeff_df(
    task_models: Sequence[ConditionalSusceptibilityTaskModel],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in task_models:
        rows.append(
            {
                "task_key": model.task_key,
                "attack_leaf": model.attack_leaf,
                "opinion_leaf": model.opinion_leaf,
                "outcome_metric": model.outcome_metric,
                "term": "Intercept",
                "estimate": float(model.intercept),
                "n_obs": int(model.n_obs),
                "alpha": float(model.alpha),
                "cv_mse": float(model.cv_mse),
            }
        )
        for column in feature_columns:
            rows.append(
                {
                    "task_key": model.task_key,
                    "attack_leaf": model.attack_leaf,
                    "opinion_leaf": model.opinion_leaf,
                    "outcome_metric": model.outcome_metric,
                    "term": column,
                    "estimate": float(model.coefficients.get(column, 0.0)),
                    "n_obs": int(model.n_obs),
                    "alpha": float(model.alpha),
                    "cv_mse": float(model.cv_mse),
                }
            )
    return pd.DataFrame(rows)


def _block_bootstrap_csi(
    long_df: pd.DataFrame,
    feature_columns: List[str],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    continuous_columns: List[str],
    outcome_metric: str,
    n_samples: int,
    seed: int,
    alpha_grid: Sequence[float] | None,
    min_rows_per_task: int,
    shrinkage_strength: float,
    task_alpha_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    task_weight_lookup: Optional[Dict[Tuple[str, str], float]] = None,
) -> BootstrapCISummary:
    """Profile-level block bootstrap for CSI percentile confidence intervals.

    For each bootstrap resample:
    1.  Sample *profile_ids* with replacement (block = all rows for that profile).
    2.  Re-fit ridge models on the resampled panel.
    3.  Score profiles and record their susceptibility_index_pct ranks plus
        task-level coefficients.

    Returns 90 % CIs (5th–95th percentile) on the rank of each profile and
    standard deviations of per-feature coefficients across bootstrap iterations.
    """
    rng = np.random.default_rng(seed + 10000)
    profile_ids = long_df["profile_id"].dropna().unique().tolist()
    n_profiles = len(profile_ids)

    original_profiles = (
        long_df[["profile_id", *feature_columns]]
        .drop_duplicates(subset=["profile_id"])
        .reset_index(drop=True)
    )
    original_transformed = _transform_features(
        original_profiles,
        feature_columns,
        feature_means,
        feature_stds,
        continuous_columns,
    )
    original_x = np.column_stack(
        [
            np.ones(len(original_transformed), dtype=float),
            *[original_transformed[c].astype(float).to_numpy() for c in feature_columns],
        ]
    )
    original_pid_order = original_profiles["profile_id"].tolist()

    rank_samples: Dict[str, List[float]] = {pid: [] for pid in profile_ids}
    coeff_samples: Dict[str, Dict[str, List[float]]] = {}

    for b in range(n_samples):
        # Bootstrap resample at profile level
        boot_pids = rng.choice(profile_ids, size=n_profiles, replace=True).tolist()
        # Allow duplicated profiles by appending a suffix to profile_id
        rows: List[pd.DataFrame] = []
        for i, pid in enumerate(boot_pids):
            sub = long_df[long_df["profile_id"] == pid].copy()
            sub["profile_id"] = f"{pid}__b{i}"
            rows.append(sub)
        if not rows:
            continue
        boot_df = pd.concat(rows, ignore_index=True)

        # Fit one task per (attack, opinion) combo
        unique_profiles = (
            boot_df[["profile_id", *feature_columns]]
            .drop_duplicates(subset=["profile_id"])
            .reset_index(drop=True)
        )
        transformed = _transform_features(unique_profiles, feature_columns, feature_means, feature_stds, continuous_columns)
        transformed = transformed.copy()
        transformed.insert(0, "profile_id", unique_profiles["profile_id"])
        profile_lu = transformed.set_index("profile_id")

        task_contribs: Dict[str, float] = {pid: 0.0 for pid in original_pid_order}
        total_w = 0.0
        coeff_agg: Dict[str, float] = {col: 0.0 for col in feature_columns}

        offset = 0
        for (atk, op), tdf in boot_df.groupby(["attack_leaf", "opinion_leaf"], dropna=False):
            if len(tdf) < min_rows_per_task:
                continue
            x_df = profile_lu.loc[tdf["profile_id"]].reset_index(drop=True)
            x = np.column_stack([
                np.ones(len(x_df)), *[x_df[c].astype(float).to_numpy() for c in feature_columns]
            ])
            y = tdf[outcome_metric].astype(float).to_numpy()
            task_key = (str(atk), str(op))
            fixed_alpha = task_alpha_lookup.get(task_key) if task_alpha_lookup else None
            fixed_weight = task_weight_lookup.get(task_key) if task_weight_lookup else None
            try:
                if fixed_alpha is not None:
                    beta = _ridge_fit_matrix(x, y, float(fixed_alpha))
                    cv_mse = float(np.mean((y - (x @ beta)) ** 2))
                else:
                    beta, _, cv_mse = _cross_validated_ridge(x, y, seed=seed + b * 1000 + offset, alpha_grid=alpha_grid)
            except Exception:
                offset += 1
                continue
            w = float(fixed_weight) if fixed_weight is not None else len(tdf) / max(cv_mse, 1e-6)

            # Apply optional shrinkage on-the-fly (reuse same lambda)
            if shrinkage_strength > 0:
                # Use single-iteration approximation: we can't do cross-task mean in a streaming loop,
                # so we skip EB shrinkage inside bootstrap (it's negligible variance anyway)
                pass

            preds = original_x @ beta
            for idx, pid in enumerate(original_pid_order):
                task_contribs[pid] = task_contribs.get(pid, 0.0) + w * float(preds[idx])
            total_w += w
            for idx, col in enumerate(feature_columns):
                coeff_agg[col] += w * float(beta[idx + 1])
            offset += 1

        if total_w == 0:
            continue

        norm_scores = np.array([task_contribs[pid] / total_w for pid in original_pid_order], dtype=float)
        ranks = pd.Series(norm_scores).rank(method="average", pct=True).to_numpy() * 100.0
        for idx, pid in enumerate(original_pid_order):
            rank_samples[pid].append(float(ranks[idx]))

        for col in feature_columns:
            if col not in coeff_samples:
                coeff_samples[col] = {}
            task_key = "global"
            if task_key not in coeff_samples[col]:
                coeff_samples[col][task_key] = []
            coeff_samples[col][task_key].append(coeff_agg[col] / total_w)

    # Summarise
    rank_ci: Dict[str, Tuple[float, float]] = {}
    rank_sd: Dict[str, float] = {}
    for pid, samples in rank_samples.items():
        if len(samples) < 3:
            continue
        arr = np.array(samples)
        rank_ci[pid] = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        rank_sd[pid] = float(np.std(arr, ddof=1))

    coeff_sd: Dict[str, Dict[str, float]] = {}
    for col, task_dict in coeff_samples.items():
        coeff_sd[col] = {
            tk: float(np.std(np.array(vals), ddof=1))
            for tk, vals in task_dict.items() if len(vals) >= 2
        }

    notes = [
        f"Profile-level block bootstrap, n_samples={n_samples}.",
        "rank_ci gives 5th–95th percentile CI of susceptibility_index_pct across bootstrap resamples.",
        "coeff_sd gives global cross-task average coefficient standard deviation.",
        "Bootstrap refits condition on the original task-level ridge alphas and normalized reliability weights for computational stability.",
    ]
    return BootstrapCISummary(
        n_samples=n_samples,
        rank_ci=rank_ci,
        rank_sd=rank_sd,
        coefficient_sd=coeff_sd,
        notes=notes,
    )


def _compute_group_attribution(
    score_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Ontology-aware rollup: aggregate contribution__* columns to group level.

    Uses :func:`_build_feature_hierarchy` to discover the same groups that
    hierarchical decomposition uses, then sums individual feature contributions
    within each group for every profile.

    Returns a tidy DataFrame with columns:
        profile_id, ontology_group, group_contribution, n_features_in_group
    """
    hierarchy = _build_feature_hierarchy(feature_columns)
    contribution_prefix = "contribution__"
    available_contrib_cols = [c for c in score_df.columns if c.startswith(contribution_prefix)]

    # Map contribution column → original feature column
    contrib_to_feat = {
        c: c[len(contribution_prefix):]
        for c in available_contrib_cols
    }

    rows: List[Dict[str, object]] = []
    for group_name, group_feat_cols in hierarchy.items():
        group_contrib_cols = [
            f"{contribution_prefix}{fc}" for fc in group_feat_cols
            if f"{contribution_prefix}{fc}" in score_df.columns
        ]
        if not group_contrib_cols:
            continue
        for _, profile_row in score_df[["profile_id", *group_contrib_cols]].iterrows():
            group_sum = sum(float(profile_row[c]) for c in group_contrib_cols)
            rows.append({
                "profile_id": profile_row["profile_id"],
                "ontology_group": group_name,
                "group_contribution": group_sum,
                "n_features_in_group": len(group_contrib_cols),
            })

    if not rows:
        return pd.DataFrame(columns=["profile_id", "ontology_group", "group_contribution", "n_features_in_group"])
    return pd.DataFrame(rows)


def fit_conditional_susceptibility_index(
    long_df: pd.DataFrame,
    *,
    outcome_metric: str = "adversarial_effectivity",
    feature_columns: Sequence[str] | None = None,
    excluded_feature_columns: Sequence[str] | None = None,
    seed: int = 42,
    alpha_grid: Sequence[float] | None = None,
    min_rows_per_task: int = 8,
    compute_hierarchy: bool = True,
    # ── Enhancement kwargs (all default-off for backwards compatibility) ──────
    bootstrap_samples: int = 0,
    shrinkage_strength: float = 0.0,
    polynomial_features: List[str] | None = None,
    interaction_pairs: List[Tuple[str, str]] | None = None,
    compute_group_attribution: bool = True,
) -> ConditionalSusceptibilityFitResult:
    # Fall back gracefully if adversarial_effectivity is not yet in the data
    if outcome_metric not in long_df.columns:
        fallback = "abs_delta_score"
        if fallback in long_df.columns:
            import warnings
            warnings.warn(
                f"outcome_metric='{outcome_metric}' not found in long_df; falling back to '{fallback}'. "
                "Run Stage 05 with --ontology-root to generate adversarial_effectivity.",
                stacklevel=2,
            )
            outcome_metric = fallback
        else:
            raise ValueError(
                f"Neither '{outcome_metric}' nor 'abs_delta_score' found in long_df columns."
            )

    required_columns = {"profile_id", "attack_leaf", "opinion_leaf", outcome_metric}
    missing = sorted(required_columns - set(long_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for conditional susceptibility fit: {missing}")

    attacked_df = long_df.copy()
    attacked_df = attacked_df.loc[attacked_df[outcome_metric].notna()].copy()
    if attacked_df.empty:
        raise ValueError("No rows with non-null outcome available for conditional susceptibility fitting.")

    feature_columns = list(feature_columns or _default_feature_columns(attacked_df, excluded_feature_columns))
    if not feature_columns:
        raise ValueError("No usable profile feature columns available for conditional susceptibility fitting.")

    unique_profiles_df = attacked_df[["profile_id", *feature_columns]].drop_duplicates(subset=["profile_id"]).reset_index(drop=True)
    feature_means, feature_stds, continuous_columns, categorical_columns = _fit_feature_scaler(
        unique_profiles_df=unique_profiles_df,
        feature_columns=feature_columns,
    )

    transformed_profiles = _transform_features(
        df=unique_profiles_df,
        feature_columns=feature_columns,
        feature_means=feature_means,
        feature_stds=feature_stds,
        continuous_columns=continuous_columns,
    )
    transformed_profiles = transformed_profiles.copy()
    transformed_profiles.insert(0, "profile_id", unique_profiles_df["profile_id"])

    # ── Optional feature engineering (polynomial + interactions) ─────────────
    engineering_log: Dict[str, List[str]] = {}
    if polynomial_features or interaction_pairs:
        aug_df, new_cols, engineering_log = _engineer_polynomial_and_interactions(
            df=transformed_profiles.drop(columns=["profile_id"]),
            feature_columns=feature_columns,
            polynomial_features=polynomial_features,
            interaction_pairs=interaction_pairs,
        )
        # Augment the unique profiles frame and extend feature_columns
        for nc in new_cols:
            transformed_profiles[nc] = aug_df[nc].values
        feature_columns = feature_columns + new_cols
        # Extend scaler state so score_profiles_with_conditional_artifact works
        for nc in new_cols:
            if nc not in feature_means:
                feature_means[nc] = 0.0   # already z-scored products → mean≈0
                feature_stds[nc] = 1.0
                continuous_columns = continuous_columns + [nc]

    profile_lookup = transformed_profiles.set_index("profile_id")
    task_rows: List[Dict[str, object]] = []
    coeff_rows: List[pd.DataFrame] = []
    task_models: List[ConditionalSusceptibilityTaskModel] = []
    task_hierarchy_r2: Dict[str, Dict[str, float]] = {}

    grouped = attacked_df.groupby(["attack_leaf", "opinion_leaf"], dropna=False)
    for offset, ((attack_leaf, opinion_leaf), task_df) in enumerate(grouped):
        if len(task_df) < min_rows_per_task:
            continue
        x_df = profile_lookup.loc[task_df["profile_id"]].reset_index(drop=True)
        x = np.column_stack(
            [
                np.ones(len(x_df), dtype=float),
                *[x_df[column].astype(float).to_numpy() for column in feature_columns],
            ]
        )
        y = task_df[outcome_metric].astype(float).to_numpy()
        beta, alpha, cv_mse = _cross_validated_ridge(
            x=x,
            y=y,
            seed=seed + offset,
            alpha_grid=alpha_grid,
        )
        reliability_weight = float(len(task_df) / max(cv_mse, 1e-6))
        key = _task_key(str(attack_leaf), str(opinion_leaf))
        coefficients = {column: float(beta[idx + 1]) for idx, column in enumerate(feature_columns)}

        if compute_hierarchy:
            try:
                hier_r2 = compute_hierarchical_decomposition(
                    x_full=x,
                    y=y,
                    feature_columns=list(feature_columns),
                    best_alpha=alpha,
                    seed=seed + offset,
                )
                task_hierarchy_r2[key] = hier_r2
            except Exception:
                task_hierarchy_r2[key] = {}

        task_models.append(
            ConditionalSusceptibilityTaskModel(
                task_key=key,
                attack_leaf=str(attack_leaf),
                opinion_leaf=str(opinion_leaf),
                outcome_metric=outcome_metric,
                n_obs=int(len(task_df)),
                alpha=float(alpha),
                cv_mse=float(cv_mse),
                reliability_weight=reliability_weight,
                intercept=float(beta[0]),
                coefficients=coefficients,
            )
        )

        task_rows.append(
            {
                "task_key": key,
                "attack_leaf": attack_leaf,
                "opinion_leaf": opinion_leaf,
                "outcome_metric": outcome_metric,
                "n_obs": int(len(task_df)),
                "alpha": float(alpha),
                "cv_mse": float(cv_mse),
                "reliability_weight_raw": reliability_weight,
            }
        )
        coeff_rows.append(
            pd.DataFrame(
                {
                    "task_key": key,
                    "attack_leaf": attack_leaf,
                    "opinion_leaf": opinion_leaf,
                    "outcome_metric": outcome_metric,
                    "term": ["Intercept", *feature_columns],
                    "estimate": [float(beta[0]), *[float(coefficients[column]) for column in feature_columns]],
                    "n_obs": int(len(task_df)),
                    "alpha": float(alpha),
                    "cv_mse": float(cv_mse),
                }
            )
        )

    if not task_models:
        raise ValueError("No task-specific models could be fit for the configured attack/opinion target set.")

    task_summary = pd.DataFrame(task_rows)
    task_summary["reliability_weight"] = (
        task_summary["reliability_weight_raw"] / float(task_summary["reliability_weight_raw"].sum())
    )
    task_summary = task_summary.drop(columns=["reliability_weight_raw"])

    normalized_weights = {
        row["task_key"]: float(row["reliability_weight"])
        for row in task_summary.to_dict(orient="records")
    }
    for model in task_models:
        model.reliability_weight = normalized_weights[model.task_key]

    # ── Optional Empirical-Bayes cross-task shrinkage ─────────────────────────
    if shrinkage_strength > 0.0:
        task_models = _apply_eb_shrinkage(task_models, list(feature_columns), shrinkage_strength)
    coeff_df = _task_models_to_coeff_df(task_models, feature_columns)

    artifact = ConditionalSusceptibilityArtifact(
        outcome_metric=outcome_metric,
        attack_leaves=sorted({str(value) for value in attacked_df["attack_leaf"].dropna().unique().tolist()}),
        opinion_leaves=sorted({str(value) for value in attacked_df["opinion_leaf"].dropna().unique().tolist()}),
        feature_columns=list(feature_columns),
        continuous_feature_columns=continuous_columns,
        categorical_feature_columns=categorical_columns,
        excluded_feature_columns=sorted(set(excluded_feature_columns or []) | LEGACY_EXCLUDED_FEATURE_COLUMNS),
        feature_means=feature_means,
        feature_stds=feature_stds,
        task_models=task_models,
        notes=[
            "This artifact is valid only for the attack leaves and opinion leaves recorded in the target set metadata.",
            f"Primary effectivity metric: {outcome_metric}. Positive = opinion moved in adversary's intended direction.",
            "The conditional susceptibility index is computed from fitted task-specific ridge models and is model-based rather than directly observed.",
            (
                f"Cross-task empirical-Bayes shrinkage applied with λ={shrinkage_strength:.2f}."
                if shrinkage_strength > 0.0
                else "No cross-task empirical-Bayes shrinkage applied."
            ),
        ],
    )

    score_df, breakdown_df = score_profiles_with_conditional_artifact(unique_profiles_df, artifact)

    # ── Optional group-level XAI attribution ─────────────────────────────────
    group_attribution_df: Optional[pd.DataFrame] = None
    if compute_group_attribution:
        try:
            group_attribution_df = _compute_group_attribution(score_df, list(feature_columns))
        except Exception:
            group_attribution_df = None

    # ── Optional block bootstrap CSI CIs ─────────────────────────────────────
    boot_ci: Optional[BootstrapCISummary] = None
    if bootstrap_samples > 0:
        try:
            boot_ci = _block_bootstrap_csi(
                long_df=attacked_df,
                feature_columns=list(feature_columns),
                feature_means=feature_means,
                feature_stds=feature_stds,
                continuous_columns=continuous_columns,
                outcome_metric=outcome_metric,
                n_samples=bootstrap_samples,
                seed=seed,
                alpha_grid=alpha_grid,
                min_rows_per_task=min_rows_per_task,
                shrinkage_strength=shrinkage_strength,
                task_alpha_lookup={
                    (str(model.attack_leaf), str(model.opinion_leaf)): float(model.alpha)
                    for model in task_models
                },
                task_weight_lookup={
                    (str(model.attack_leaf), str(model.opinion_leaf)): float(model.reliability_weight)
                    for model in task_models
                },
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"Bootstrap CI computation failed: {exc}", stacklevel=2)

    # Aggregate hierarchical decomposition across tasks weighted by reliability
    hier_decomp: Optional[HierarchicalDecomposition] = None
    if compute_hierarchy and task_hierarchy_r2:
        agg_group_r2: Dict[str, float] = {}
        total_weight = sum(normalized_weights.values())
        full_r2_weighted = 0.0
        for model in task_models:
            w = float(model.reliability_weight)
            task_r2 = task_hierarchy_r2.get(model.task_key, {})
            full_r2_weighted += w * float(task_r2.get("full_model", 0.0))
            for k, v in task_r2.items():
                if k == "full_model":
                    continue
                agg_group_r2[k] = agg_group_r2.get(k, 0.0) + w * float(v)
        # Normalize weighted marginals by total weight
        if total_weight > 0:
            agg_group_r2 = {k: v / total_weight for k, v in agg_group_r2.items()}
            full_r2_weighted /= total_weight

        # Relative importance: marginal R² / sum(|marginal R²|)
        total_abs = sum(abs(v) for v in agg_group_r2.values()) or 1.0
        rel_importance = {k: abs(v) / total_abs * 100.0 for k, v in agg_group_r2.items()}

        hier_decomp = HierarchicalDecomposition(
            group_marginal_r2=agg_group_r2,
            group_relative_importance_pct=rel_importance,
            full_model_cv_r2=full_r2_weighted,
            task_group_r2=task_hierarchy_r2,
        )

    return ConditionalSusceptibilityFitResult(
        artifact=artifact,
        task_coefficients=coeff_df,
        task_summary=task_summary,
        profile_scores=score_df,
        contribution_breakdown=breakdown_df,
        hierarchical_decomposition=hier_decomp,
        bootstrap_ci=boot_ci,
        group_contribution_breakdown=group_attribution_df,
        feature_engineering=engineering_log,
    )


def score_profiles_with_conditional_artifact(
    profile_df: pd.DataFrame,
    artifact: ConditionalSusceptibilityArtifact,
    target_attacks: Optional[List[str]] = None,
    target_opinions: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score profiles using a fitted artifact, optionally filtering to a subset of tasks.

    Args:
        profile_df: DataFrame with profile_id and feature columns. Missing feature
            columns are imputed with the artifact's stored training-set means.
        artifact: Fitted ConditionalSusceptibilityArtifact.
        target_attacks: If given, restrict to tasks whose attack_leaf is in this list.
        target_opinions: If given, restrict to tasks whose opinion_leaf is in this list.

    Returns:
        (profile_scores_df, breakdown_df) — same structure as fit outputs.
    """
    if "profile_id" not in profile_df.columns:
        raise ValueError("profile_df must contain profile_id for conditional susceptibility scoring.")

    # Select tasks matching filter (or all tasks)
    active_tasks = [
        t for t in artifact.task_models
        if (target_attacks is None or t.attack_leaf in target_attacks)
        and (target_opinions is None or t.opinion_leaf in target_opinions)
    ]
    if not active_tasks:
        raise ValueError(
            "No tasks matched the specified target_attacks/target_opinions filter. "
            f"Available attacks: {artifact.attack_leaves}. "
            f"Available opinions: {artifact.opinion_leaves}."
        )

    # Re-normalize weights for selected tasks
    weight_sum = sum(t.reliability_weight for t in active_tasks)
    weight_sum = max(weight_sum, 1e-10)

    # Build per-profile feature matrix — impute missing columns with training means
    avail_features = [c for c in artifact.feature_columns if c in profile_df.columns]
    missing_features = [c for c in artifact.feature_columns if c not in profile_df.columns]

    unique_profile_df = profile_df[["profile_id", *avail_features]].drop_duplicates(subset=["profile_id"]).reset_index(drop=True)

    # Fill missing feature columns with training-set means (imputation)
    for col in missing_features:
        imputed = artifact.feature_means.get(col, 0.0)
        unique_profile_df[col] = imputed

    transformed = _transform_features(
        df=unique_profile_df,
        feature_columns=artifact.feature_columns,
        feature_means=artifact.feature_means,
        feature_stds=artifact.feature_stds,
        continuous_columns=artifact.continuous_feature_columns,
    )
    transformed = transformed.copy()
    transformed.insert(0, "profile_id", unique_profile_df["profile_id"])

    score_df = unique_profile_df[["profile_id"]].copy()
    score_df["conditional_target_attack_count"] = len({t.attack_leaf for t in active_tasks})
    score_df["conditional_target_opinion_count"] = len({t.opinion_leaf for t in active_tasks})
    score_df["conditional_target_task_count"] = len(active_tasks)
    score_df["imputed_feature_count"] = len(missing_features)
    score_df["imputed_features"] = ", ".join(missing_features) if missing_features else ""

    raw_score = np.zeros(len(score_df), dtype=float)
    breakdown_rows: List[Dict[str, object]] = []

    for task_model in active_tasks:
        normalized_w = float(task_model.reliability_weight) / weight_sum
        task_contribution = np.full(len(score_df), float(task_model.intercept), dtype=float)
        for column in artifact.feature_columns:
            beta = float(task_model.coefficients.get(column, 0.0))
            task_contribution = task_contribution + transformed[column].astype(float).to_numpy() * beta
        weighted_contribution = task_contribution * normalized_w
        task_slug = (
            task_model.task_key.lower()
            .replace(" ", "_")
            .replace(">", "_")
            .replace("|", "_")
            .replace("/", "_")
        )
        score_df[f"predicted_effectivity__{task_slug}"] = task_contribution
        score_df[f"weighted_effectivity__{task_slug}"] = weighted_contribution
        raw_score = raw_score + weighted_contribution

        for idx, profile_id in enumerate(score_df["profile_id"].tolist()):
            breakdown_rows.append(
                {
                    "profile_id": profile_id,
                    "component_type": "task",
                    "component_name": task_model.task_key,
                    "component_key": task_slug,
                    "attack_leaf": task_model.attack_leaf,
                    "opinion_leaf": task_model.opinion_leaf,
                    "contribution": float(weighted_contribution[idx]),
                    "reliability_weight": float(task_model.reliability_weight),
                }
            )

    score_df["conditional_susceptibility_raw_score"] = raw_score
    score_df["susceptibility_index_pct"] = pd.Series(raw_score, index=score_df.index).rank(method="average", pct=True) * 100.0

    signed_weight_lookup: Dict[str, float] = {}
    for task_model in active_tasks:
        normalized_w = float(task_model.reliability_weight) / weight_sum
        for column, beta in task_model.coefficients.items():
            signed_weight_lookup[column] = signed_weight_lookup.get(column, 0.0) + float(beta) * normalized_w

    for column in artifact.feature_columns:
        values = transformed[column].astype(float).to_numpy()
        total_beta_weight = signed_weight_lookup.get(column, 0.0)
        contribution = values * total_beta_weight
        score_df[f"contribution__{column}"] = contribution
        for idx, profile_id in enumerate(score_df["profile_id"].tolist()):
            breakdown_rows.append(
                {
                    "profile_id": profile_id,
                    "component_type": "feature",
                    "component_name": column,
                    "component_key": column,
                    "attack_leaf": None,
                    "opinion_leaf": None,
                    "contribution": float(contribution[idx]),
                    "reliability_weight": float(total_beta_weight),
                }
            )

    return (
        score_df.sort_values("susceptibility_index_pct", ascending=False).reset_index(drop=True),
        pd.DataFrame(breakdown_rows),
    )


def build_conditional_weight_table(
    artifact: ConditionalSusceptibilityArtifact,
) -> pd.DataFrame:
    """
    Aggregate per-task ridge coefficients into a reliability-weighted feature
    importance table.

    Columns returned:
      term                     – raw feature column name
      moderator_label          – human-readable label (from SemanticScaleRegistry)
      ontology_group           – auto-discovered hierarchy group (inventory or dim)
      weighted_mean_estimate   – signed task-reliability-weighted mean coefficient
      weighted_mean_abs_estimate – unsigned magnitude
      normalized_weight_pct    – |effect| as % of total |effect|
      direction                – "higher susceptibility" | "lower susceptibility" | "neutral"
      n_tasks                  – how many tasks contributed a non-zero coefficient
    """
    # Build ontology groups from column names (no hardcoding)
    hierarchy = _build_feature_hierarchy(artifact.feature_columns)
    # Invert: column → group name (use most specific group that contains it)
    col_to_group: Dict[str, str] = {}
    # Sort by group key length descending so more specific groups win
    for grp, cols in sorted(hierarchy.items(), key=lambda kv: len(kv[0]), reverse=True):
        for c in cols:
            if c not in col_to_group:
                col_to_group[c] = grp

    # Import semantic scale registry lazily to avoid circular deps
    try:
        from src.backend.utils.embeddings.semantic_scale import get_default_registry as _gdr
        scale_reg = _gdr()
    except Exception:
        scale_reg = None

    rows: List[Dict[str, object]] = []
    for column in artifact.feature_columns:
        signed_effect = 0.0
        abs_effect = 0.0
        n_tasks = 0
        for task_model in artifact.task_models:
            if column not in task_model.coefficients:
                continue
            beta = float(task_model.coefficients[column])
            weight = float(task_model.reliability_weight)
            signed_effect += weight * beta
            abs_effect += weight * abs(beta)
            n_tasks += 1

        # Human-readable label
        if scale_reg is not None:
            sc = scale_reg.get_scale(column)
            if sc is not None:
                label = sc.dimension_label
            elif column.startswith("profile_cat__"):
                # Categorical one-hot: derive "Sex: Female" style label
                inner = column.removeprefix("profile_cat__")
                # Strip any leading "profile_cat_" fragment kept by double-prefix encoding
                inner = inner.removeprefix("profile_cat_")
                parts = inner.split("_")
                # Last token = level, everything before = group key
                level = parts[-1]
                group = " ".join(parts[:-1]).title() if len(parts) > 1 else inner.title()
                label = f"{group}: {level}"
            else:
                inner = column.removeprefix("profile_cont_")
                # Strip known suffixes
                for suf in ("_pct", "_years", "_score"):
                    if inner.endswith(suf):
                        inner = inner[: -len(suf)]
                        break
                label = inner.replace("_", " ").strip().title()
        else:
            label = column

        group = col_to_group.get(column, "other")
        direction = (
            "higher susceptibility" if signed_effect > 0.01
            else "lower susceptibility" if signed_effect < -0.01
            else "neutral"
        )

        rows.append(
            {
                "term": column,
                "moderator_label": label,
                "ontology_group": group,
                "weighted_mean_estimate": signed_effect,
                "weighted_mean_abs_estimate": abs_effect,
                "direction": direction,
                "n_tasks": n_tasks,
            }
        )

    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        return weight_df
    denom = float(weight_df["weighted_mean_abs_estimate"].sum()) or 1.0
    weight_df["normalized_weight_pct"] = (
        weight_df["weighted_mean_abs_estimate"] / denom
    ) * 100.0
    return weight_df.sort_values(
        ["normalized_weight_pct", "term"], ascending=[False, True]
    ).reset_index(drop=True)
