from __future__ import annotations

"""Condition-level statistical models and robustness checks for H3/H4.

All inferential functions operate at the condition-cell level; profile rows are used only to construct condition summaries.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - statsmodels is present in the project venv
    smf = None

from .data import _condition_table
from .formatting import _as_bool


def _fit_fe_model(
    condition: pd.DataFrame,
    outcome: str,
    *,
    predictor: str = "achieved_alignment_z",
    model_id: str,
    model_label: str,
    robust_cov: str | None = None,
) -> dict[str, Any]:
    base = {
        "outcome": outcome,
        "predictor": predictor,
        "model_id": model_id,
        "model": model_label,
        "covariance": robust_cov or "classical",
        "estimate": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "n": int(len(condition)),
        "df_resid": np.nan,
    }
    if smf is None or condition.empty:
        return base
    model = smf.ols(
        f"{outcome} ~ {predictor} + C(opinion_label) + C(attack_label)",
        data=condition,
    ).fit()
    result = model.get_robustcov_results(cov_type=robust_cov) if robust_cov else model
    params = pd.Series(result.params, index=model.params.index)
    bse = pd.Series(result.bse, index=model.params.index)
    pvalues = pd.Series(result.pvalues, index=model.params.index)
    return {
        **base,
        "estimate": float(params.get(predictor, np.nan)),
        "std_error": float(bse.get(predictor, np.nan)),
        "p_value": float(pvalues.get(predictor, np.nan)),
        "r_squared": float(model.rsquared),
        "n": int(model.nobs),
        "df_resid": float(model.df_resid),
    }


def _fit_models(condition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in ["mean_pn_increment_effectivity", "mean_ae_total_network"]:
        rows.append(
            _fit_fe_model(
                condition,
                outcome,
                model_id="main_fe_ols",
                model_label="OLS with opinion and attack fixed effects",
            )
        )
        rows.append(
            _fit_fe_model(
                condition,
                outcome,
                model_id="main_fe_hc3",
                model_label="OLS with opinion and attack fixed effects",
                robust_cov="HC3",
            )
        )
    return pd.DataFrame(rows)


def _bh_fdr_q_values(p_values: list[float]) -> list[float]:
    valid = [(idx, float(p)) for idx, p in enumerate(p_values) if not math.isnan(float(p))]
    q_values = [np.nan] * len(p_values)
    if not valid:
        return q_values
    m = len(valid)
    ranked = sorted(valid, key=lambda item: item[1])
    adjusted: list[tuple[int, float]] = []
    running_min = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(ranked), start=1):
        rank = m - rank_from_end + 1
        running_min = min(running_min, p_value * m / rank)
        adjusted.append((idx, min(running_min, 1.0)))
    for idx, q_value in adjusted:
        q_values[idx] = q_value
    return q_values


def _hc3_alignment_model_summary(condition: pd.DataFrame, outcome: str) -> dict[str, Any]:
    base = {
        "beta": np.nan,
        "hc3_se": np.nan,
        "hc3_ci_low": np.nan,
        "hc3_ci_high": np.nan,
        "hc3_p": np.nan,
        "n_conditions": int(condition.shape[0]),
    }
    if smf is None or condition.empty:
        return base
    model = smf.ols(
        f"{outcome} ~ achieved_alignment_z + C(opinion_label) + C(attack_label)",
        data=condition,
    ).fit()
    result = model.get_robustcov_results(cov_type="HC3")
    names = list(model.params.index)
    params = pd.Series(result.params, index=names)
    bse = pd.Series(result.bse, index=names)
    pvalues = pd.Series(result.pvalues, index=names)
    ci = pd.DataFrame(result.conf_int(), index=names, columns=["low", "high"])
    predictor = "achieved_alignment_z"
    return {
        **base,
        "beta": float(params.get(predictor, np.nan)),
        "hc3_se": float(bse.get(predictor, np.nan)),
        "hc3_ci_low": float(ci.loc[predictor, "low"]) if predictor in ci.index else np.nan,
        "hc3_ci_high": float(ci.loc[predictor, "high"]) if predictor in ci.index else np.nan,
        "hc3_p": float(pvalues.get(predictor, np.nan)),
        "n_conditions": int(model.nobs),
    }


def _figure4_model_summary(condition: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        ("mean_pn_increment_effectivity", "primary_h3_h4_mechanism"),
        ("mean_ae_total_network", "secondary_final_endpoint"),
    ]
    rows: list[dict[str, Any]] = []
    for outcome, role in outcomes:
        summary = _hc3_alignment_model_summary(condition, outcome)
        permutation = robustness[
            robustness["outcome"].eq(outcome) & robustness["model_id"].eq("within_attack_permutation")
        ]
        permutation_p = (
            float(permutation["p_value"].iloc[0]) if not permutation.empty and "p_value" in permutation.columns else np.nan
        )
        rows.append(
            {
                "outcome": outcome,
                "role": role,
                **summary,
                "bh_fdr_q": np.nan,
                "permutation_p": permutation_p,
                "effect_from_minus_0_9_to_plus_0_9": float(summary["beta"]) * 1.8
                if not math.isnan(float(summary["beta"]))
                else np.nan,
            }
        )
    q_values = _bh_fdr_q_values([float(row["hc3_p"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["bh_fdr_q"] = q_value
    return pd.DataFrame(rows)


def _permutation_p_value(
    condition: pd.DataFrame,
    outcome: str,
    observed_estimate: float,
    *,
    n_perm: int = 5000,
    seed: int = 120,
) -> float:
    if smf is None or math.isnan(float(observed_estimate)):
        return np.nan
    rng = np.random.default_rng(seed)
    count = 0
    usable = condition.dropna(subset=["achieved_alignment_z", outcome, "attack_label", "opinion_label"]).copy()
    if usable.empty:
        return np.nan
    for _ in range(n_perm):
        permuted = usable.copy()
        permuted["permuted_alignment_z"] = np.nan
        for _, idx in permuted.groupby("attack_label").groups.items():
            values = permuted.loc[idx, "achieved_alignment_z"].to_numpy()
            permuted.loc[idx, "permuted_alignment_z"] = rng.permutation(values)
        model = smf.ols(
            f"{outcome} ~ permuted_alignment_z + C(opinion_label) + C(attack_label)",
            data=permuted,
        ).fit()
        estimate = float(model.params.get("permuted_alignment_z", np.nan))
        if estimate >= observed_estimate:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _fit_robustness_models(sem: pd.DataFrame, condition: pd.DataFrame, design: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fallback_cols = [
        col
        for col in ["baseline_fallback_used", "post_fallback_used", "network_exposure_fallback_used"]
        if col in sem.columns
    ]
    fallback_mask = pd.Series(False, index=sem.index)
    for col in fallback_cols:
        fallback_mask = fallback_mask | _as_bool(sem[col])
    fallback_condition = _condition_table(sem.loc[~fallback_mask].copy(), design) if fallback_cols else condition.copy()

    if "post_heuristic_pass" in sem.columns:
        warning_conditions = set(
            sem.loc[~_as_bool(sem["post_heuristic_pass"].fillna(True)), "alignment_condition_id"].dropna()
        )
    else:
        warning_conditions = set()
    heuristic_condition = condition.loc[~condition["alignment_condition_id"].isin(warning_conditions)].copy()

    for outcome in ["mean_pn_increment_effectivity", "mean_ae_total_network"]:
        main = _fit_fe_model(
            condition,
            outcome,
            model_id="main_fe_ols",
            model_label="OLS with opinion and attack fixed effects",
        )
        hc3 = _fit_fe_model(
            condition,
            outcome,
            model_id="main_fe_hc3",
            model_label="OLS with opinion and attack fixed effects",
            robust_cov="HC3",
        )
        fallback = _fit_fe_model(
            fallback_condition,
            outcome,
            model_id="exclude_upstream_fallback_rows",
            model_label="Sensitivity excluding private or BN fallback-flagged rows",
        )
        heuristic = _fit_fe_model(
            heuristic_condition,
            outcome,
            model_id="exclude_post_heuristic_warning_conditions",
            model_label="Sensitivity excluding conditions containing post-attack heuristic warnings",
        )
        permutation_p = _permutation_p_value(condition, outcome, float(main["estimate"]))
        rows.extend(
            [
                {**main, "excluded_rows": 0, "excluded_conditions": 0, "permutation_count": np.nan},
                {**hc3, "excluded_rows": 0, "excluded_conditions": 0, "permutation_count": np.nan},
                {
                    **main,
                    "model_id": "within_attack_permutation",
                    "model": "Directional permutation of achieved alignment within attack",
                    "covariance": "permutation",
                    "p_value": permutation_p,
                    "excluded_rows": 0,
                    "excluded_conditions": 0,
                    "permutation_count": 5000,
                },
                {
                    **fallback,
                    "excluded_rows": int(fallback_mask.sum()),
                    "excluded_conditions": 0,
                    "permutation_count": np.nan,
                },
                {
                    **heuristic,
                    "excluded_rows": 0,
                    "excluded_conditions": int(len(warning_conditions)),
                    "permutation_count": np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def _original_vs_branch_comparison(branch_root: Path, condition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources: list[tuple[str, pd.DataFrame, str]] = [
        ("alignment_gradient_branch", condition.copy(), "achieved_alignment_z"),
    ]
    original_path = (
        branch_root.parent
        / "network_exposure_analysis"
        / "tables"
        / "centrality_alignment_outcome_link.csv"
    )
    if original_path.exists():
        original = pd.read_csv(original_path).copy()
        original["achieved_alignment_z"] = pd.to_numeric(
            original["sender_reach_susceptibility_alignment_z"], errors="coerce"
        )
        sources.insert(0, ("original_fixed_assignment_run_2", original, "achieved_alignment_z"))

    for source, df, predictor in sources:
        for outcome in ["mean_pn_increment_effectivity", "mean_ae_total_network"]:
            fit = _fit_fe_model(
                df,
                outcome,
                predictor=predictor,
                model_id=f"{source}_fe_ols",
                model_label="OLS with opinion and attack fixed effects",
            )
            rows.append(
                {
                    "source": source,
                    "outcome": outcome,
                    "n_conditions": int(df.shape[0]),
                    "alignment_min": float(pd.to_numeric(df[predictor], errors="coerce").min()),
                    "alignment_max": float(pd.to_numeric(df[predictor], errors="coerce").max()),
                    "alignment_sd": float(pd.to_numeric(df[predictor], errors="coerce").std(ddof=1)),
                    "estimate": fit["estimate"],
                    "std_error": fit["std_error"],
                    "p_value": fit["p_value"],
                    "r_squared": fit["r_squared"],
                    "interpretation": (
                        "Natural observed assignment; limited alignment range."
                        if source == "original_fixed_assignment_run_2"
                        else "Counterfactual assignment; experimentally widened alignment range."
                    ),
                }
            )
    return pd.DataFrame(rows)
