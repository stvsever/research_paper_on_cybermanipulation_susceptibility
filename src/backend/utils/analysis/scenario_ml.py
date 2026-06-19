from __future__ import annotations

"""
Scenario-level machine-learning moderation stack (current design).

Why this module exists
----------------------
The historical stage 06 analyses operate on the profile-aggregate panel
(n = number of profiles). That panel is the right place for SEM-style path
diagnostics, but it throws away the scenario-level factorial structure:
each row of the long table is one (profile, attack, opinion) cell, and the
research question is explicitly conditional:

    Which profile features moderate adversarial effectivity,
    conditional on the attack vector and the opinion target?

The right primary analysis for that question is scenario-level, with
generalization measured across held-out PROFILES (GroupKFold by profile),
because the scientific claim is about transfer to new people, not new rows.

Components
----------
1. Model ladder with grouped cross-validation:
     M0  context only            (attack + opinion fixed effects)
     M0b context + baseline      (adds pre-exposure position)
     M1  M0b + profile, linear   (ridge)
     M2  M0b + profile, boosted  (HistGradientBoosting; non-linear,
                                  interaction-capable)
   The profile predictive increment (M1 - M0b, M2 - M0b) is the honest
   effect-size statement for "profiles matter beyond context".

2. Held-out permutation importance for the boosted model, restricted to
   profile features, averaged over folds (importance measured only on
   profiles the model never saw).

3. Conditional moderation scan: for every (profile feature, attack leaf)
   pair, a within-attack OLS of AE on the standardized feature plus opinion
   fixed effects, with CR1 cluster-robust standard errors clustered on
   profile, Benjamini-Hochberg corrected across the full scan. The same
   scan is repeated per opinion leaf. This yields the feature-by-context
   moderation surface the dashboard renders.

4. Effect summaries with profile-cluster bootstrap CIs: attack ranking,
   opinion ranking, attack x opinion cell means, and empirical-Bayes
   shrunken per-profile susceptibility means.

All functions are pure (no file I/O); stage 06 persists the outputs.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.nanstd(x))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.nanmean(x))) / sd


def _one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(series.astype(str), prefix=prefix, dtype=float)


def _valid_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    out: List[str] = []
    for col in feature_cols:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            out.append(col)
    return out


def _cluster_robust_ols(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """OLS with CR1 cluster-robust covariance. Returns (beta, se, n_clusters)."""
    n, p = x.shape
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.pinv(xtx)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), np.full(p, np.nan), 0
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    unique = np.unique(clusters)
    g = len(unique)
    meat = np.zeros((p, p))
    for cid in unique:
        mask = clusters == cid
        xg = x[mask]
        ug = resid[mask]
        sg = xg.T @ ug
        meat += np.outer(sg, sg)
    if g > 1 and n > p:
        correction = (g / (g - 1)) * ((n - 1) / (n - p))
    else:
        correction = 1.0
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, g


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty(m)
    cum_min = 1.0
    for i in range(m - 1, -1, -1):
        q = ranked[i] * m / (i + 1)
        cum_min = min(cum_min, q)
        adj[i] = cum_min
    q_vals = np.empty(m)
    q_vals[order] = adj
    out[mask] = q_vals
    return out


# ---------------------------------------------------------------------------
# 1. model ladder with GroupKFold by profile
# ---------------------------------------------------------------------------


@dataclass
class ModelLadderResult:
    table: pd.DataFrame
    fold_table: pd.DataFrame
    n_obs: int
    n_profiles: int
    n_features_profile: int
    notes: List[str] = field(default_factory=list)


def fit_model_ladder(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    feature_cols: Sequence[str],
    profile_col: str = "profile_id",
    attack_col: str = "attack_leaf",
    opinion_col: str = "opinion_leaf",
    baseline_col: str = "baseline_score",
    n_splits: int = 5,
    seed: int = 0,
) -> ModelLadderResult:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    notes: List[str] = []
    needed = [outcome_col, profile_col, attack_col, opinion_col]
    if any(c not in long_df.columns for c in needed):
        return ModelLadderResult(pd.DataFrame(), pd.DataFrame(), 0, 0, 0, ["missing columns"])

    feats = _valid_feature_columns(long_df, feature_cols)
    work = long_df.dropna(subset=[outcome_col]).copy()
    y = work[outcome_col].astype(float).to_numpy()
    groups = work[profile_col].astype(str).to_numpy()
    n_profiles = len(np.unique(groups))
    if n_profiles < n_splits:
        n_splits = max(2, n_profiles)
        notes.append(f"reduced folds to {n_splits} (few profiles)")

    context = pd.concat(
        [_one_hot(work[attack_col], "atk"), _one_hot(work[opinion_col], "op")], axis=1
    )
    baseline = (
        work[[baseline_col]].astype(float).fillna(0.0)
        if baseline_col in work.columns
        else pd.DataFrame(index=work.index)
    )
    profile_x = work[feats].astype(float).fillna(0.0) if feats else pd.DataFrame(index=work.index)
    profile_x = (profile_x - profile_x.mean()) / profile_x.std(ddof=0).replace(0, 1.0)

    designs: Dict[str, pd.DataFrame] = {
        "M0_context": context,
        "M0b_context_baseline": pd.concat([context, baseline], axis=1),
        "M1_profile_linear": pd.concat([context, baseline, profile_x], axis=1),
        "M2_profile_boosted": pd.concat([context, baseline, profile_x], axis=1),
    }

    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, Any]] = []
    alphas = np.logspace(-2, 4, 25)

    for model_name, design in designs.items():
        xmat = design.to_numpy(dtype=float)
        for fold_idx, (tr, te) in enumerate(gkf.split(xmat, y, groups=groups)):
            if model_name == "M2_profile_boosted":
                est = HistGradientBoostingRegressor(
                    max_iter=350,
                    learning_rate=0.07,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=seed,
                )
            else:
                est = RidgeCV(alphas=alphas)
            est.fit(xmat[tr], y[tr])
            pred = est.predict(xmat[te])
            ss_res = float(np.sum((y[te] - pred) ** 2))
            ss_tot = float(np.sum((y[te] - np.mean(y[tr])) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "cv_r2": r2,
                    "rmse": float(np.sqrt(np.mean((y[te] - pred) ** 2))),
                    "n_test": int(len(te)),
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    summary = (
        fold_table.groupby("model", as_index=False)
        .agg(cv_r2_mean=("cv_r2", "mean"), cv_r2_sd=("cv_r2", "std"), rmse_mean=("rmse", "mean"))
        .reset_index(drop=True)
    )
    base_r2 = float(
        summary.loc[summary["model"] == "M0b_context_baseline", "cv_r2_mean"].iloc[0]
    ) if (summary["model"] == "M0b_context_baseline").any() else np.nan
    ctx_r2 = float(
        summary.loc[summary["model"] == "M0_context", "cv_r2_mean"].iloc[0]
    ) if (summary["model"] == "M0_context").any() else np.nan
    summary["delta_r2_vs_context_baseline"] = summary["cv_r2_mean"] - base_r2
    summary["delta_r2_vs_context"] = summary["cv_r2_mean"] - ctx_r2
    order = ["M0_context", "M0b_context_baseline", "M1_profile_linear", "M2_profile_boosted"]
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)

    return ModelLadderResult(
        table=summary,
        fold_table=fold_table,
        n_obs=int(len(work)),
        n_profiles=n_profiles,
        n_features_profile=len(feats),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 2. held-out permutation importance for the boosted model
# ---------------------------------------------------------------------------


@dataclass
class HeldOutImportanceResult:
    table: pd.DataFrame
    notes: List[str] = field(default_factory=list)


def heldout_gbm_importance(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    feature_cols: Sequence[str],
    profile_col: str = "profile_id",
    attack_col: str = "attack_leaf",
    opinion_col: str = "opinion_leaf",
    baseline_col: str = "baseline_score",
    n_splits: int = 5,
    n_repeats: int = 8,
    seed: int = 0,
) -> HeldOutImportanceResult:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold

    feats = _valid_feature_columns(long_df, feature_cols)
    if not feats or outcome_col not in long_df.columns:
        return HeldOutImportanceResult(pd.DataFrame(), ["no features or outcome"])

    work = long_df.dropna(subset=[outcome_col]).copy()
    y = work[outcome_col].astype(float).to_numpy()
    groups = work[profile_col].astype(str).to_numpy()
    n_profiles = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_profiles))

    context = pd.concat(
        [_one_hot(work[attack_col], "atk"), _one_hot(work[opinion_col], "op")], axis=1
    )
    baseline = (
        work[[baseline_col]].astype(float).fillna(0.0)
        if baseline_col in work.columns
        else pd.DataFrame(index=work.index)
    )
    profile_x = work[feats].astype(float).fillna(0.0)
    profile_x = (profile_x - profile_x.mean()) / profile_x.std(ddof=0).replace(0, 1.0)
    design = pd.concat([context, baseline, profile_x], axis=1)
    xmat = design.to_numpy(dtype=float)
    col_names = list(design.columns)
    profile_positions = [col_names.index(f) for f in feats]

    gkf = GroupKFold(n_splits=n_splits)
    acc: Dict[str, List[float]] = {f: [] for f in feats}
    for tr, te in gkf.split(xmat, y, groups=groups):
        est = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.07, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed,
        )
        est.fit(xmat[tr], y[tr])
        perm = permutation_importance(
            est, xmat[te], y[te], n_repeats=n_repeats, random_state=seed, n_jobs=-1
        )
        for feat, pos in zip(feats, profile_positions):
            acc[feat].append(float(perm.importances_mean[pos]))

    rows = [
        {
            "term": feat,
            "importance_mean": float(np.mean(vals)),
            "importance_sd": float(np.std(vals, ddof=0)),
            "n_folds": len(vals),
        }
        for feat, vals in acc.items()
    ]
    table = (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return HeldOutImportanceResult(table=table)


# ---------------------------------------------------------------------------
# 3. conditional moderation scan (feature x attack, feature x opinion)
# ---------------------------------------------------------------------------


@dataclass
class ModerationScanResult:
    by_attack: pd.DataFrame
    by_opinion: pd.DataFrame
    pooled: pd.DataFrame
    notes: List[str] = field(default_factory=list)


def conditional_moderation_scan(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    feature_cols: Sequence[str],
    profile_col: str = "profile_id",
    attack_col: str = "attack_leaf",
    opinion_col: str = "opinion_leaf",
) -> ModerationScanResult:
    from scipy import stats as sps

    feats = _valid_feature_columns(long_df, feature_cols)
    work = long_df.dropna(subset=[outcome_col]).copy()
    if work.empty or not feats:
        empty = pd.DataFrame()
        return ModerationScanResult(empty, empty, empty, ["no data or features"])

    clusters_all = work[profile_col].astype(str).to_numpy()
    y_all = work[outcome_col].astype(float).to_numpy()

    def _scan(context_col: str, control_col: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for ctx_value, sub_idx in work.groupby(context_col).groups.items():
            sub = work.loc[sub_idx]
            controls = _one_hot(sub[control_col], "c")
            # drop one control column to avoid collinearity with intercept
            if controls.shape[1] > 1:
                controls = controls.iloc[:, 1:]
            y = sub[outcome_col].astype(float).to_numpy()
            clusters = sub[profile_col].astype(str).to_numpy()
            base_design = np.column_stack(
                [np.ones(len(sub)), controls.to_numpy(dtype=float)]
            )
            for feat in feats:
                fx = _zscore(sub[feat].astype(float).fillna(0.0).to_numpy())
                if float(np.std(fx)) < 1e-12:
                    continue
                x = np.column_stack([base_design, fx])
                beta, se, g = _cluster_robust_ols(x, y, clusters)
                est = float(beta[-1])
                s = float(se[-1])
                if not np.isfinite(s) or s <= 0 or g < 3:
                    continue
                t = est / s
                p = float(2 * sps.t.sf(abs(t), df=max(1, g - 1)))
                rows.append(
                    {
                        "feature": feat,
                        "context": str(ctx_value),
                        "estimate": est,
                        "std_error": s,
                        "t_value": float(t),
                        "p_value": p,
                        "n_obs": int(len(sub)),
                        "n_clusters": int(g),
                    }
                )
        out = pd.DataFrame(rows)
        if not out.empty:
            out["q_value"] = _bh_fdr(out["p_value"].to_numpy())
        return out

    by_attack = _scan(attack_col, opinion_col)
    by_opinion = _scan(opinion_col, attack_col)

    # pooled per-feature regression across all contexts
    pooled_rows: List[Dict[str, Any]] = []
    controls = pd.concat(
        [_one_hot(work[attack_col], "atk").iloc[:, 1:], _one_hot(work[opinion_col], "op").iloc[:, 1:]],
        axis=1,
    )
    base_design = np.column_stack([np.ones(len(work)), controls.to_numpy(dtype=float)])
    for feat in feats:
        fx = _zscore(work[feat].astype(float).fillna(0.0).to_numpy())
        if float(np.std(fx)) < 1e-12:
            continue
        x = np.column_stack([base_design, fx])
        beta, se, g = _cluster_robust_ols(x, y_all, clusters_all)
        est = float(beta[-1])
        s = float(se[-1])
        if not np.isfinite(s) or s <= 0 or g < 3:
            continue
        t = est / s
        p = float(2 * sps.t.sf(abs(t), df=max(1, g - 1)))
        pooled_rows.append(
            {
                "feature": feat,
                "estimate": est,
                "std_error": s,
                "t_value": float(t),
                "p_value": p,
                "n_obs": int(len(work)),
                "n_clusters": int(g),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    if not pooled.empty:
        pooled["q_value"] = _bh_fdr(pooled["p_value"].to_numpy())
        pooled = pooled.sort_values("p_value").reset_index(drop=True)

    return ModerationScanResult(by_attack=by_attack, by_opinion=by_opinion, pooled=pooled)


# ---------------------------------------------------------------------------
# 4. effect summaries with profile-cluster bootstrap CIs
# ---------------------------------------------------------------------------


@dataclass
class EffectSummaryResult:
    attack_effects: pd.DataFrame
    opinion_effects: pd.DataFrame
    cell_effects: pd.DataFrame
    profile_effects: pd.DataFrame
    notes: List[str] = field(default_factory=list)


def _bootstrap_column_means(
    per_profile: pd.DataFrame, n_bootstrap: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """per_profile: rows = profiles, cols = conditions, values = profile-cell means.

    Returns (ci_low, ci_high) for the across-profile mean of each column,
    resampling profiles with replacement (cluster bootstrap, profile level).
    """
    rng = np.random.default_rng(seed)
    mat = per_profile.to_numpy(dtype=float)
    n = mat.shape[0]
    boots = np.full((n_bootstrap, mat.shape[1]), np.nan)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.nanmean(mat[idx], axis=0)
    return np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0)


def compute_effect_summaries(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    profile_col: str = "profile_id",
    attack_col: str = "attack_leaf",
    opinion_col: str = "opinion_leaf",
    attack_label_col: str = "attack_leaf_label",
    opinion_label_col: str = "opinion_leaf_label",
    n_bootstrap: int = 500,
    seed: int = 0,
) -> EffectSummaryResult:
    work = long_df.dropna(subset=[outcome_col]).copy()
    if work.empty:
        empty = pd.DataFrame()
        return EffectSummaryResult(empty, empty, empty, empty, ["no rows"])

    def _labels(col: str, label_col: str) -> Dict[str, str]:
        if label_col in work.columns:
            return (
                work[[col, label_col]]
                .dropna()
                .drop_duplicates()
                .set_index(col)[label_col]
                .astype(str)
                .to_dict()
            )
        return {}

    atk_labels = _labels(attack_col, attack_label_col)
    op_labels = _labels(opinion_col, opinion_label_col)

    def _condition_table(cond_col: str, labels: Dict[str, str], offset: int) -> pd.DataFrame:
        per_profile = work.pivot_table(
            index=profile_col, columns=cond_col, values=outcome_col, aggfunc="mean"
        )
        ci_low, ci_high = _bootstrap_column_means(per_profile, n_bootstrap, seed + offset)
        cell_counts = work.groupby(cond_col)[outcome_col].count()
        pos_share = work.groupby(cond_col)[outcome_col].apply(lambda s: float((s > 0).mean()))
        rows: List[Dict[str, Any]] = []
        for j, cond in enumerate(per_profile.columns):
            mean_of_profile_means = float(np.nanmean(per_profile[cond].to_numpy()))
            rows.append(
                {
                    cond_col: str(cond),
                    "label": labels.get(cond, str(cond)),
                    "mean_ae": mean_of_profile_means,
                    "ci_low": float(ci_low[j]),
                    "ci_high": float(ci_high[j]),
                    "share_positive": float(pos_share.get(cond, np.nan)),
                    "n_obs": int(cell_counts.get(cond, 0)),
                    "n_profiles": int(per_profile[cond].notna().sum()),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("mean_ae", ascending=False)
            .reset_index(drop=True)
        )

    attack_effects = _condition_table(attack_col, atk_labels, 11)
    opinion_effects = _condition_table(opinion_col, op_labels, 23)

    # attack x opinion cells
    work["_cell"] = work[attack_col].astype(str) + " || " + work[opinion_col].astype(str)
    per_profile_cell = work.pivot_table(
        index=profile_col, columns="_cell", values=outcome_col, aggfunc="mean"
    )
    ci_low, ci_high = _bootstrap_column_means(per_profile_cell, max(200, n_bootstrap // 2), seed + 37)
    cell_rows: List[Dict[str, Any]] = []
    cell_counts = work.groupby("_cell")[outcome_col].count()
    for j, cell in enumerate(per_profile_cell.columns):
        atk, op = str(cell).split(" || ", 1)
        cell_rows.append(
            {
                "attack_leaf": atk,
                "opinion_leaf": op,
                "attack_label": atk_labels.get(atk, atk),
                "opinion_label": op_labels.get(op, op),
                "mean_ae": float(np.nanmean(per_profile_cell[cell].to_numpy())),
                "ci_low": float(ci_low[j]),
                "ci_high": float(ci_high[j]),
                "n_obs": int(cell_counts.get(cell, 0)),
            }
        )
    cell_effects = pd.DataFrame(cell_rows)

    # empirical-Bayes shrunken profile means
    grp = work.groupby(profile_col)[outcome_col]
    m_i = grp.mean()
    n_i = grp.count()
    v_i = grp.var(ddof=1).fillna(0.0)
    grand = float(work[outcome_col].mean())
    sigma2_within = float(np.average(v_i.to_numpy(), weights=np.clip(n_i.to_numpy() - 1, 1, None)))
    var_means = float(np.var(m_i.to_numpy(), ddof=1)) if len(m_i) > 1 else 0.0
    tau2 = max(0.0, var_means - sigma2_within * float(np.mean(1.0 / n_i.to_numpy())))
    shrink = tau2 / (tau2 + sigma2_within / n_i.to_numpy()) if (tau2 + sigma2_within) > 0 else np.zeros(len(m_i))
    eb_means = grand + shrink * (m_i.to_numpy() - grand)
    profile_effects = pd.DataFrame(
        {
            "profile_id": m_i.index.astype(str),
            "raw_mean_ae": m_i.to_numpy(),
            "eb_mean_ae": eb_means,
            "shrinkage_weight": shrink,
            "n_obs": n_i.to_numpy(),
            "within_sd": np.sqrt(np.clip(v_i.to_numpy(), 0, None)),
        }
    )
    profile_effects["eb_rank"] = (
        profile_effects["eb_mean_ae"].rank(ascending=False, method="average").astype(float)
    )
    profile_effects["eb_percentile"] = (
        profile_effects["eb_mean_ae"].rank(pct=True, method="average") * 100.0
    )
    profile_effects = profile_effects.sort_values("eb_mean_ae", ascending=False).reset_index(drop=True)

    notes = [
        f"tau2={tau2:.4f}, sigma2_within={sigma2_within:.4f}, grand_mean={grand:.4f}",
        "Condition means are means of per-profile means (each profile weighted equally).",
        "CIs are 95% percentile cluster bootstrap intervals resampling profiles.",
    ]
    return EffectSummaryResult(
        attack_effects=attack_effects,
        opinion_effects=opinion_effects,
        cell_effects=cell_effects,
        profile_effects=profile_effects,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 4b. supplementary: profile-space distance vs moderation-pattern distance
# ---------------------------------------------------------------------------


@dataclass
class ProfileDistanceModerationResult:
    pair_table: pd.DataFrame
    mantel_r: float
    mantel_p: float
    n_profiles: int
    n_pairs: int
    n_permutations: int
    notes: List[str] = field(default_factory=list)


def profile_distance_moderation_test(
    long_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    outcome_col: str = "adversarial_effectivity",
    profile_col: str = "profile_id",
    attack_col: str = "attack_leaf",
    opinion_col: str = "opinion_leaf",
    n_permutations: int = 2000,
    seed: int = 0,
) -> ProfileDistanceModerationResult:
    """Mantel-style test: is closeness of two profiles in configuration space
    predictive of similarity in their susceptibility (moderation) patterns?

    Distance in profile space = Euclidean over z-scored features.
    Distance in moderation space = Euclidean over the per-task (attack x
    opinion) mean-AE vectors. Significance via permutation of profile labels
    on the moderation matrix (the standard Mantel correction for the
    non-independence of pairwise distances).
    """
    feats = _valid_feature_columns(long_df, feature_cols)
    work = long_df.dropna(subset=[outcome_col]).copy()
    if work.empty or not feats:
        return ProfileDistanceModerationResult(pd.DataFrame(), float("nan"), float("nan"), 0, 0, 0, ["no data"])

    prof_feats = work.groupby(profile_col)[feats].first()
    z = (prof_feats - prof_feats.mean()) / prof_feats.std(ddof=0).replace(0, 1.0)
    work["_task"] = work[attack_col].astype(str) + " || " + work[opinion_col].astype(str)
    ae_mat = work.pivot_table(index=profile_col, columns="_task", values=outcome_col, aggfunc="mean")
    ae_mat = ae_mat.loc[z.index]

    ids = list(z.index)
    n = len(ids)
    if n < 8:
        return ProfileDistanceModerationResult(pd.DataFrame(), float("nan"), float("nan"), n, 0, 0,
                                               ["too few profiles for Mantel test (<8)"])
    fz = z.to_numpy(dtype=float)
    am = ae_mat.to_numpy(dtype=float)
    col_means = np.nanmean(am, axis=0)
    inds = np.where(np.isnan(am))
    am[inds] = np.take(col_means, inds[1])

    def _pairwise(mat: np.ndarray) -> np.ndarray:
        sq = np.sum(mat ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2 * mat @ mat.T
        return np.sqrt(np.clip(d2, 0, None))

    d_feat = _pairwise(fz)
    d_ae = _pairwise(am)
    iu = np.triu_indices(n, k=1)
    x = d_feat[iu]
    y = d_ae[iu]

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    r_obs = _pearson(x, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        r_perm = _pearson(x, d_ae[np.ix_(perm, perm)][iu])
        if np.isfinite(r_perm) and abs(r_perm) >= abs(r_obs):
            count += 1
    p = (count + 1) / (n_permutations + 1)

    pair_rows = [
        {"profile_a": ids[i], "profile_b": ids[j],
         "feature_distance": float(d_feat[i, j]), "moderation_distance": float(d_ae[i, j])}
        for i, j in zip(*iu)
    ]
    return ProfileDistanceModerationResult(
        pair_table=pd.DataFrame(pair_rows),
        mantel_r=float(r_obs), mantel_p=float(p),
        n_profiles=n, n_pairs=len(pair_rows), n_permutations=n_permutations,
        notes=[
            "Positive Mantel r = profiles that are close in configuration space show similar susceptibility patterns.",
            "Permutation p from relabeling profiles on the moderation matrix (two-sided on |r|).",
        ],
    )


# ---------------------------------------------------------------------------
# 5. bundle runner
# ---------------------------------------------------------------------------


@dataclass
class ScenarioMlBundle:
    model_ladder: ModelLadderResult
    heldout_importance: HeldOutImportanceResult
    moderation_scan: ModerationScanResult
    effect_summaries: EffectSummaryResult


def run_scenario_ml(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    feature_cols: Sequence[str],
    n_bootstrap: int = 500,
    seed: int = 0,
) -> ScenarioMlBundle:
    """End-to-end scenario-level analysis. Each component degrades gracefully."""

    def _safe(factory, fallback):
        try:
            return factory()
        except Exception as exc:
            LOGGER.warning("scenario_ml component failed: %s", exc)
            return fallback

    ladder = _safe(
        lambda: fit_model_ladder(long_df, outcome_col=outcome_col, feature_cols=feature_cols, seed=seed),
        ModelLadderResult(pd.DataFrame(), pd.DataFrame(), 0, 0, 0, ["failed"]),
    )
    importance = _safe(
        lambda: heldout_gbm_importance(long_df, outcome_col=outcome_col, feature_cols=feature_cols, seed=seed),
        HeldOutImportanceResult(pd.DataFrame(), ["failed"]),
    )
    scan = _safe(
        lambda: conditional_moderation_scan(long_df, outcome_col=outcome_col, feature_cols=feature_cols),
        ModerationScanResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["failed"]),
    )
    summaries = _safe(
        lambda: compute_effect_summaries(long_df, outcome_col=outcome_col, n_bootstrap=n_bootstrap, seed=seed),
        EffectSummaryResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["failed"]),
    )
    return ScenarioMlBundle(
        model_ladder=ladder,
        heldout_importance=importance,
        moderation_scan=scan,
        effect_summaries=summaries,
    )
