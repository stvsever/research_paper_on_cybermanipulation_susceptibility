#!/usr/bin/env python3
"""Build profile-trait direct-vs-network susceptibility analysis for Run 2."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-profile-trait-direct-vs-network")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


SEED = 120
N_BOOT = 5000

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIMARY_PATH = (
    ROOT
    / "counterfactual_alignment_gradient"
    / "merged_outputs"
    / "stage_outputs"
    / "05_compute_effectivity_deltas"
    / "sem_long_encoded.csv"
)
SENSITIVITY_PATH = ROOT / "stage_outputs" / "05_compute_effectivity_deltas" / "sem_long_encoded.csv"

OUTCOME_DIRECT = "mean_ae_private"
OUTCOME_NETWORK = "mean_pn_increment_effectivity"

FOCAL_PREDICTORS = {
    "age": "profile_cont_chronological_age",
    "openness": "profile_cont_big_five_openness_to_experience_mean_pct",
    "conscientiousness": "profile_cont_big_five_conscientiousness_mean_pct",
    "extraversion": "profile_cont_big_five_extraversion_mean_pct",
    "agreeableness": "profile_cont_big_five_agreeableness_mean_pct",
    "neuroticism": "profile_cont_big_five_neuroticism_mean_pct",
}
SEX_COVARIATES = {
    "female": "profile_cat__profile_cat_sex_Female",
    "other_gender": "profile_cat__profile_cat_sex_Other",
}
DERIVED_SUPPLEMENT = {
    "heuristic_shift_sensitivity": "profile_cont_heuristic_shift_sensitivity_proxy",
    "resilience_index": "profile_cont_resilience_index",
}

PREDICTOR_LABELS = {
    OUTCOME_DIRECT: "Direct private susceptibility",
    "age": "Age",
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
    "female": "Female",
    "other_gender": "Other gender",
    "heuristic_shift_sensitivity": "Heuristic-shift sensitivity",
    "resilience_index": "Resilience index",
}


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(q, index=p_values.index)

    idx = np.where(finite)[0]
    order = idx[np.argsort(p[finite])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q[order] = adjusted
    return pd.Series(q, index=p_values.index)


@dataclass(frozen=True)
class ModelOutputs:
    profile: pd.DataFrame
    coefficients: pd.DataFrame
    bootstrap: pd.DataFrame
    summary: dict[str, float | int | str]


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd


def _ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (math.nan, math.nan)
    return tuple(np.quantile(values, [alpha / 2, 1 - alpha / 2]).astype(float))


def _read_sem(path: Path, *, exclude_post_network_fallback: bool) -> pd.DataFrame:
    needed = {
        "scenario_id",
        "profile_id",
        "ae_private",
        "pn_increment_effectivity",
        "post_attack_network_fallback_used",
        *FOCAL_PREDICTORS.values(),
        *SEX_COVARIATES.values(),
        *DERIVED_SUPPLEMENT.values(),
    }
    df = pd.read_csv(path, usecols=lambda c: c in needed)
    if exclude_post_network_fallback:
        df = df[~_bool_series(df["post_attack_network_fallback_used"])].copy()
    return df


def _profile_summary(sem: pd.DataFrame) -> pd.DataFrame:
    first_cols = {**FOCAL_PREDICTORS, **SEX_COVARIATES, **DERIVED_SUPPLEMENT}
    agg = {col: "first" for col in first_cols.values()}
    agg.update(
        {
            "scenario_id": "count",
            "ae_private": "mean",
            "pn_increment_effectivity": "mean",
        }
    )
    profile = sem.groupby("profile_id", as_index=False).agg(agg)
    profile = profile.rename(
        columns={
            "scenario_id": "n_profile_rows",
            "ae_private": OUTCOME_DIRECT,
            "pn_increment_effectivity": OUTCOME_NETWORK,
            **{v: k for k, v in first_cols.items()},
        }
    )
    keep = ["profile_id", "n_profile_rows", OUTCOME_DIRECT, OUTCOME_NETWORK, *first_cols.keys()]
    profile = profile[keep].copy()
    required = [OUTCOME_DIRECT, OUTCOME_NETWORK, *FOCAL_PREDICTORS.keys(), *SEX_COVARIATES.keys()]
    return profile.dropna(subset=required).reset_index(drop=True)


def _fit_models(profile: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, sm.regression.linear_model.RegressionResultsWrapper]]:
    predictors = [*FOCAL_PREDICTORS.keys(), *SEX_COVARIATES.keys()]
    X = profile[predictors].apply(_z)
    X = X.loc[:, X.std(ddof=0) > 0]
    X = sm.add_constant(X)

    outputs = []
    models = {}
    for outcome, role in [
        (OUTCOME_DIRECT, "direct_private_susceptibility"),
        (OUTCOME_NETWORK, "post_network_amplification"),
    ]:
        y = _z(profile[outcome])
        model = sm.OLS(y, X).fit(cov_type="HC3")
        models[outcome] = model
        for predictor in X.columns:
            if predictor == "const":
                continue
            ci_low, ci_high = model.conf_int().loc[predictor].tolist()
            outputs.append(
                {
                    "outcome": outcome,
                    "role": role,
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS[predictor],
                    "is_focal_predictor": predictor in FOCAL_PREDICTORS,
                    "beta": float(model.params[predictor]),
                    "hc3_se": float(model.bse[predictor]),
                    "hc3_ci_low": float(ci_low),
                    "hc3_ci_high": float(ci_high),
                    "hc3_p": float(model.pvalues[predictor]),
                    "n_profiles": int(model.nobs),
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                }
            )
    return pd.DataFrame(outputs), models


def _fit_direct_adjusted_network_model(profile: pd.DataFrame) -> pd.DataFrame:
    predictors = [OUTCOME_DIRECT, *FOCAL_PREDICTORS.keys(), *SEX_COVARIATES.keys()]
    X = profile[predictors].apply(_z)
    X = X.loc[:, X.std(ddof=0) > 0]
    X = sm.add_constant(X)
    y = _z(profile[OUTCOME_NETWORK])
    model = sm.OLS(y, X).fit(cov_type="HC3")

    rows = []
    for predictor in X.columns:
        if predictor == "const":
            continue
        ci_low, ci_high = model.conf_int().loc[predictor].tolist()
        rows.append(
            {
                "model": "traits_plus_direct",
                "outcome": OUTCOME_NETWORK,
                "predictor": predictor,
                "predictor_label": PREDICTOR_LABELS[predictor],
                "predictor_role": "direct_susceptibility"
                if predictor == OUTCOME_DIRECT
                else "focal_trait"
                if predictor in FOCAL_PREDICTORS
                else "adjustment_covariate",
                "beta": float(model.params[predictor]),
                "hc3_se": float(model.bse[predictor]),
                "hc3_ci_low": float(ci_low),
                "hc3_ci_high": float(ci_high),
                "hc3_p": float(model.pvalues[predictor]),
                "n_profiles": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
            }
        )
    out = pd.DataFrame(rows)
    tested = out["predictor_role"].isin({"direct_susceptibility", "focal_trait"})
    out["bh_fdr_q"] = np.nan
    out.loc[tested, "bh_fdr_q"] = _bh_fdr(out.loc[tested, "hc3_p"])
    return out


def _fit_model_ladder(profile: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("direct_only", "Direct susceptibility only", [OUTCOME_DIRECT]),
        ("traits_only", "Profile traits + sex", [*FOCAL_PREDICTORS.keys(), *SEX_COVARIATES.keys()]),
        (
            "traits_plus_direct",
            "Profile traits + sex + direct susceptibility",
            [OUTCOME_DIRECT, *FOCAL_PREDICTORS.keys(), *SEX_COVARIATES.keys()],
        ),
    ]
    y = _z(profile[OUTCOME_NETWORK])
    rows = []
    previous_r2 = 0.0
    for model_name, label, predictors in specs:
        X = profile[predictors].apply(_z)
        X = X.loc[:, X.std(ddof=0) > 0]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit(cov_type="HC3")
        rows.append(
            {
                "model": model_name,
                "model_label": label,
                "outcome": OUTCOME_NETWORK,
                "predictors": ",".join(predictors),
                "n_profiles": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "delta_r_squared_from_previous": float(model.rsquared - previous_r2),
                "aic": float(model.aic),
            }
        )
        previous_r2 = float(model.rsquared)
    return pd.DataFrame(rows)


def _multiple_testing_sensitivity(
    coefficients: pd.DataFrame, direct_adjusted: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for _, row in coefficients[coefficients["is_focal_predictor"]].iterrows():
        rows.append(
            {
                "model": row["role"],
                "outcome": row["outcome"],
                "predictor": row["predictor"],
                "predictor_label": row["predictor_label"],
                "hc3_p": row["hc3_p"],
            }
        )
    for _, row in direct_adjusted[
        direct_adjusted["predictor_role"].isin({"direct_susceptibility", "focal_trait"})
    ].iterrows():
        rows.append(
            {
                "model": row["model"],
                "outcome": row["outcome"],
                "predictor": row["predictor"],
                "predictor_label": row["predictor_label"],
                "hc3_p": row["hc3_p"],
            }
        )
    out = pd.DataFrame(rows)
    out["bh_fdr_q_within_model"] = np.nan
    for model, idx in out.groupby("model").groups.items():
        out.loc[idx, "bh_fdr_q_within_model"] = _bh_fdr(out.loc[idx, "hc3_p"])
    return out


def _fit_beta_vector(profile: pd.DataFrame, predictors: list[str], outcome: str) -> pd.Series:
    X = profile[predictors].apply(_z)
    X = X.loc[:, X.std(ddof=0) > 0]
    X = sm.add_constant(X)
    y = _z(profile[outcome])
    return sm.OLS(y, X).fit().params.drop("const")


def _bootstrap(profile: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    focal = list(FOCAL_PREDICTORS.keys())
    model_predictors = [*FOCAL_PREDICTORS.keys(), *SEX_COVARIATES.keys()]
    rows = []
    n = len(profile)

    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample = profile.iloc[sample_idx].reset_index(drop=True)
        try:
            r = sample[OUTCOME_DIRECT].corr(sample[OUTCOME_NETWORK])
            beta_direct = _fit_beta_vector(sample, model_predictors, OUTCOME_DIRECT)
            beta_network = _fit_beta_vector(sample, model_predictors, OUTCOME_NETWORK)
            direct_focal = beta_direct.reindex(focal)
            network_focal = beta_network.reindex(focal)
            coef_similarity = direct_focal.corr(network_focal)
        except Exception:
            continue

        rows.append(
            {
                "bootstrap_id": i,
                "metric": "profile_outcome_correlation",
                "predictor": "",
                "value": float(r),
            }
        )
        rows.append(
            {
                "bootstrap_id": i,
                "metric": "focal_coefficient_vector_similarity",
                "predictor": "",
                "value": float(coef_similarity),
            }
        )
        for predictor in focal:
            rows.append(
                {
                    "bootstrap_id": i,
                    "metric": "beta_direct",
                    "predictor": predictor,
                    "value": float(beta_direct[predictor]),
                }
            )
            rows.append(
                {
                    "bootstrap_id": i,
                    "metric": "beta_network",
                    "predictor": predictor,
                    "value": float(beta_network[predictor]),
                }
            )
            rows.append(
                {
                    "bootstrap_id": i,
                    "metric": "beta_network_minus_direct",
                    "predictor": predictor,
                    "value": float(beta_network[predictor] - beta_direct[predictor]),
                }
            )
    return pd.DataFrame(rows)


def _summarize_bootstrap(profile: pd.DataFrame, coefficients: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    focal = list(FOCAL_PREDICTORS.keys())
    rows = []
    direct = coefficients[coefficients["outcome"].eq(OUTCOME_DIRECT)].set_index("predictor")
    network = coefficients[coefficients["outcome"].eq(OUTCOME_NETWORK)].set_index("predictor")

    observed_r = float(profile[OUTCOME_DIRECT].corr(profile[OUTCOME_NETWORK]))
    lo, hi = _ci(boot.loc[boot["metric"].eq("profile_outcome_correlation"), "value"].to_numpy())
    rows.append(
        {
            "metric": "profile_outcome_correlation",
            "predictor": "",
            "predictor_label": "",
            "observed": observed_r,
            "ci_low": lo,
            "ci_high": hi,
            "n_bootstrap": int(boot["bootstrap_id"].nunique()),
        }
    )

    observed_similarity = float(direct.loc[focal, "beta"].corr(network.loc[focal, "beta"]))
    lo, hi = _ci(boot.loc[boot["metric"].eq("focal_coefficient_vector_similarity"), "value"].to_numpy())
    rows.append(
        {
            "metric": "focal_coefficient_vector_similarity",
            "predictor": "",
            "predictor_label": "",
            "observed": observed_similarity,
            "ci_low": lo,
            "ci_high": hi,
            "n_bootstrap": int(boot["bootstrap_id"].nunique()),
        }
    )

    for predictor in focal:
        for metric, observed in [
            ("beta_direct", float(direct.loc[predictor, "beta"])),
            ("beta_network", float(network.loc[predictor, "beta"])),
            (
                "beta_network_minus_direct",
                float(network.loc[predictor, "beta"] - direct.loc[predictor, "beta"]),
            ),
        ]:
            vals = boot.loc[(boot["metric"].eq(metric)) & (boot["predictor"].eq(predictor)), "value"].to_numpy()
            lo, hi = _ci(vals)
            rows.append(
                {
                    "metric": metric,
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS[predictor],
                    "observed": observed,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_bootstrap": int(np.isfinite(vals).sum()),
                }
            )
    return pd.DataFrame(rows)


def _univariate_supplement(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = {**FOCAL_PREDICTORS, **SEX_COVARIATES, **DERIVED_SUPPLEMENT}
    for predictor in predictors:
        for outcome in [OUTCOME_DIRECT, OUTCOME_NETWORK]:
            r, p = stats.pearsonr(profile[predictor], profile[outcome])
            rows.append(
                {
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS[predictor],
                    "predictor_type": "derived_index" if predictor in DERIVED_SUPPLEMENT else "model_predictor",
                    "outcome": outcome,
                    "pearson_r": float(r),
                    "p_value": float(p),
                    "n_profiles": int(len(profile)),
                }
            )
    return pd.DataFrame(rows)


def _run_analysis(label: str, path: Path, *, exclude_post_network_fallback: bool) -> ModelOutputs:
    sem = _read_sem(path, exclude_post_network_fallback=exclude_post_network_fallback)
    profile = _profile_summary(sem)
    coefs, _ = _fit_models(profile)
    boot = _bootstrap(profile)
    summary = {
        "source_label": label,
        "source_path": str(path),
        "row_count": int(len(sem)),
        "profile_count": int(profile["profile_id"].nunique()),
        "min_rows_per_profile": int(profile["n_profile_rows"].min()),
        "max_rows_per_profile": int(profile["n_profile_rows"].max()),
        "exclude_post_network_fallback": bool(exclude_post_network_fallback),
        "post_network_fallback_count": int(_bool_series(sem.get("post_attack_network_fallback_used", pd.Series([], dtype=bool))).sum())
        if not exclude_post_network_fallback
        else 0,
        "profile_outcome_correlation": float(profile[OUTCOME_DIRECT].corr(profile[OUTCOME_NETWORK])),
    }
    return ModelOutputs(profile=profile, coefficients=coefs, bootstrap=boot, summary=summary)


def _format_coef_for_plot(coefficients: pd.DataFrame, boot_summary: pd.DataFrame) -> pd.DataFrame:
    focal = list(FOCAL_PREDICTORS.keys())
    rows = []
    for outcome, metric in [(OUTCOME_DIRECT, "beta_direct"), (OUTCOME_NETWORK, "beta_network")]:
        coef = coefficients[coefficients["outcome"].eq(outcome)].set_index("predictor")
        ci = boot_summary[boot_summary["metric"].eq(metric)].set_index("predictor")
        for predictor in focal:
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS[predictor],
                    "beta": float(coef.loc[predictor, "beta"]),
                    "ci_low": float(ci.loc[predictor, "ci_low"]),
                    "ci_high": float(ci.loc[predictor, "ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_main_figure(
    profile: pd.DataFrame,
    coefficients: pd.DataFrame,
    boot_summary: pd.DataFrame,
    direct_adjusted: pd.DataFrame,
    model_ladder: pd.DataFrame,
) -> dict[str, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c7ced8",
            "axes.labelcolor": "#172033",
            "xtick.color": "#172033",
            "ytick.color": "#172033",
            "text.color": "#172033",
        }
    )

    direct_color = "#4C78A8"
    network_color = "#E07A5F"
    grid_color = "#d8dde6"
    adjusted_color = "#1f2a44"
    text_muted = "#5d6b82"

    fig = plt.figure(figsize=(17.2, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel A: profile scatter with bootstrap line band.
    x = profile[OUTCOME_DIRECT].to_numpy()
    y = profile[OUTCOME_NETWORK].to_numpy()
    ax_a.scatter(x, y, s=42, color="#7FA6D9", alpha=0.72, edgecolor="white", linewidth=0.5)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 120)
    ax_a.plot(xs, intercept + slope * xs, color="#1f2a44", linewidth=2.0)

    r_row = boot_summary[boot_summary["metric"].eq("profile_outcome_correlation")].iloc[0]
    rng = np.random.default_rng(SEED)
    line_samples = []
    for _ in range(1000):
        idx = rng.integers(0, len(profile), size=len(profile))
        sx, sy = x[idx], y[idx]
        if np.unique(sx).size < 2:
            continue
        bs, bi = np.polyfit(sx, sy, 1)
        line_samples.append(bi + bs * xs)
    if line_samples:
        arr = np.vstack(line_samples)
        ax_a.fill_between(xs, np.quantile(arr, 0.025, axis=0), np.quantile(arr, 0.975, axis=0), color="#1f2a44", alpha=0.12, linewidth=0)

    ax_a.set_title("A. Profile-level association", loc="left", fontsize=13, weight="bold")
    ax_a.set_xlabel("Mean direct private attack effect\nmean(AE_private), score points")
    ax_a.set_ylabel("Mean post-network amplification\nmean(PN_increment_effectivity), score points")
    ax_a.grid(True, color=grid_color, linewidth=0.8)
    ax_a.text(
        0.04,
        0.96,
        f"r = {r_row.observed:.2f}\n95% bootstrap CI [{r_row.ci_low:.2f}, {r_row.ci_high:.2f}]\nn = {len(profile)} profiles",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c7ced8", alpha=0.95),
    )

    # Panel B: paired coefficient forest.
    coef_plot = _format_coef_for_plot(coefficients, boot_summary)
    order = (
        coef_plot[coef_plot["outcome"].eq(OUTCOME_NETWORK)]
        .assign(abs_beta=lambda d: d["beta"].abs())
        .sort_values("abs_beta", ascending=True)["predictor"]
        .tolist()
    )
    y_pos = np.arange(len(order))
    offsets = {OUTCOME_DIRECT: -0.13, OUTCOME_NETWORK: 0.13}
    colors = {OUTCOME_DIRECT: direct_color, OUTCOME_NETWORK: network_color}
    labels = {OUTCOME_DIRECT: "Direct private effect", OUTCOME_NETWORK: "Post-network increment"}
    for outcome in [OUTCOME_DIRECT, OUTCOME_NETWORK]:
        sub = coef_plot[coef_plot["outcome"].eq(outcome)].set_index("predictor").loc[order].reset_index()
        ypos = y_pos + offsets[outcome]
        ax_b.hlines(ypos, sub["ci_low"], sub["ci_high"], color=colors[outcome], alpha=0.85, linewidth=2)
        ax_b.scatter(sub["beta"], ypos, s=62, color=colors[outcome], edgecolor="white", linewidth=0.8, label=labels[outcome], zorder=3)
    direct_points = coef_plot[coef_plot["outcome"].eq(OUTCOME_DIRECT)].set_index("predictor").loc[order]
    network_points = coef_plot[coef_plot["outcome"].eq(OUTCOME_NETWORK)].set_index("predictor").loc[order]
    for i, predictor in enumerate(order):
        ax_b.plot(
            [direct_points.loc[predictor, "beta"], network_points.loc[predictor, "beta"]],
            [i, i],
            color="#9aa4b2",
            linewidth=1.1,
            alpha=0.55,
            zorder=1,
        )
    ax_b.axvline(0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([PREDICTOR_LABELS[p] for p in order])
    ax_b.set_xlabel("Standardized beta with 95% paired-profile bootstrap CI")
    ax_b.set_title("B. Trait predictors in parallel models", loc="left", fontsize=13, weight="bold")
    ax_b.grid(True, axis="x", color=grid_color, linewidth=0.8)
    ax_b.legend(frameon=False, loc="lower right", fontsize=10)

    # Panel C: direct-adjusted network amplification test.
    c_order = [
        OUTCOME_DIRECT,
        "age",
        "openness",
        "agreeableness",
        "extraversion",
        "conscientiousness",
        "neuroticism",
    ]
    c_plot = direct_adjusted.set_index("predictor").loc[c_order].reset_index()
    c_plot["plot_label"] = c_plot["predictor_label"]
    c_plot.loc[c_plot["predictor"].eq(OUTCOME_DIRECT), "plot_label"] = "Direct susceptibility"
    c_y = np.arange(len(c_order))
    point_colors = [direct_color if p == OUTCOME_DIRECT else adjusted_color for p in c_plot["predictor"]]
    ax_c.hlines(c_y, c_plot["hc3_ci_low"], c_plot["hc3_ci_high"], color="#5f6b7a", alpha=0.85, linewidth=2.1)
    ax_c.scatter(c_plot["beta"], c_y, s=64, color=point_colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax_c.axvline(0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax_c.axhline(0.5, color="#d0d5de", linewidth=1.0)
    ax_c.set_yticks(c_y)
    ax_c.set_yticklabels(c_plot["plot_label"])
    ax_c.set_xlim(-0.55, 0.82)
    ax_c.set_xlabel("Standardized beta predicting post-network amplification\n95% HC3 CI")
    ax_c.set_title("C. Network model adjusted for direct susceptibility", loc="left", fontsize=13, weight="bold")
    ax_c.grid(True, axis="x", color=grid_color, linewidth=0.8)

    direct_row = direct_adjusted[direct_adjusted["predictor"].eq(OUTCOME_DIRECT)].iloc[0]
    ladder = model_ladder.set_index("model").loc[["direct_only", "traits_only", "traits_plus_direct"]].reset_index()
    ladder_values = dict(zip(ladder["model"], ladder["r_squared"]))
    ax_c.text(
        0.97,
        0.04,
        "Direct after traits\n"
        f"β = {direct_row.beta:.3f}, p = {direct_row.hc3_p:.3f}\n"
        f"95% CI [{direct_row.hc3_ci_low:.3f}, {direct_row.hc3_ci_high:.3f}]\n\n"
        "R² ladder\n"
        f"direct only  {ladder_values['direct_only']:.3f}\n"
        f"traits only  {ladder_values['traits_only']:.3f}\n"
        f"traits + direct  {ladder_values['traits_plus_direct']:.3f}",
        transform=ax_c.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.8,
        color=adjusted_color,
        linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#c7ced8", alpha=0.96),
    )

    fig.suptitle(
        "Network-conditioned susceptibility is not reducible to direct private susceptibility",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.12, wspace=0.42)

    base = OUT / "main_figure_profile_trait_direct_vs_network"
    outputs = {
        "png": base.with_suffix(".png"),
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(
        outputs["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return outputs


def _plot_supplementary_coefficient_overlap(
    coefficients: pd.DataFrame, boot_summary: pd.DataFrame
) -> dict[str, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c7ced8",
            "axes.labelcolor": "#172033",
            "xtick.color": "#172033",
            "ytick.color": "#172033",
            "text.color": "#172033",
        }
    )
    grid_color = "#d8dde6"
    text_muted = "#5d6b82"
    direct = coefficients[coefficients["outcome"].eq(OUTCOME_DIRECT)].set_index("predictor")
    network = coefficients[coefficients["outcome"].eq(OUTCOME_NETWORK)].set_index("predictor")
    focal = list(FOCAL_PREDICTORS.keys())
    cx = direct.loc[focal, "beta"]
    cy = network.loc[focal, "beta"]
    lim = max(abs(cx).max(), abs(cy).max()) + 0.08

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.axhline(0, color="#d0d5de", linewidth=1)
    ax.axvline(0, color="#d0d5de", linewidth=1)
    ax.plot([-lim, lim], [-lim, lim], color="#6b7280", linewidth=1.1, linestyle="--")
    ax.scatter(cx, cy, s=78, color="#8F6BAE", edgecolor="white", linewidth=0.9, zorder=3)
    label_offsets = {
        "age": (0.02, -0.03),
        "openness": (0.02, 0.01),
        "conscientiousness": (0.02, -0.03),
        "extraversion": (0.03, 0.04),
        "agreeableness": (0.03, -0.035),
        "neuroticism": (0.02, 0.02),
    }
    for predictor in focal:
        dx, dy = label_offsets[predictor]
        ax.text(cx[predictor] + dx, cy[predictor] + dy, PREDICTOR_LABELS[predictor], fontsize=9)
    sim_row = boot_summary[boot_summary["metric"].eq("focal_coefficient_vector_similarity")].iloc[0]
    ax.text(
        0.04,
        0.96,
        f"coefficient-vector r = {sim_row.observed:.2f}\n95% bootstrap CI [{sim_row.ci_low:.2f}, {sim_row.ci_high:.2f}]",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c7ced8", alpha=0.95),
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Direct effect model beta")
    ax.set_ylabel("Network increment model beta")
    ax.set_title("Supplementary Figure. Predictor-profile overlap", loc="left", fontsize=13, weight="bold")
    ax.grid(True, color=grid_color, linewidth=0.8)
    fig.text(
        0.02,
        0.01,
        "Note. Each point is one focal continuous profile trait. The dashed line marks identical standardized coefficients in the direct and network models.",
        ha="left",
        va="bottom",
        fontsize=9,
        color=text_muted,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.16)

    base = OUT / "supplementary_profile_trait_coefficient_overlap_map"
    outputs = {
        "png": base.with_suffix(".png"),
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def _write_outputs(primary: ModelOutputs, sensitivity: ModelOutputs) -> dict[str, str]:
    boot_summary = _summarize_bootstrap(primary.profile, primary.coefficients, primary.bootstrap)
    sensitivity_boot_summary = _summarize_bootstrap(
        sensitivity.profile, sensitivity.coefficients, sensitivity.bootstrap
    )
    direct_adjusted = _fit_direct_adjusted_network_model(primary.profile)
    model_ladder = _fit_model_ladder(primary.profile)
    multiple_testing = _multiple_testing_sensitivity(primary.coefficients, direct_adjusted)
    univariate = _univariate_supplement(primary.profile)

    profile_path = OUT / "profile_trait_direct_vs_network_profile_summary.csv"
    coefficients_path = OUT / "profile_trait_direct_vs_network_coefficients.csv"
    bootstrap_path = OUT / "profile_trait_direct_vs_network_bootstrap_summary.csv"
    direct_adjusted_path = OUT / "profile_trait_direct_vs_network_direct_adjusted_coefficients.csv"
    model_ladder_path = OUT / "profile_trait_direct_vs_network_model_ladder.csv"
    multiple_testing_path = OUT / "profile_trait_direct_vs_network_multiple_testing.csv"
    sensitivity_path = OUT / "profile_trait_direct_vs_network_fixed_position_sensitivity.csv"
    univariate_path = OUT / "profile_trait_direct_vs_network_univariate_supplement.csv"
    manifest_path = OUT / "profile_trait_direct_vs_network_manifest.json"

    primary.profile.to_csv(profile_path, index=False)
    primary.coefficients.to_csv(coefficients_path, index=False)
    boot_summary.to_csv(bootstrap_path, index=False)
    direct_adjusted.to_csv(direct_adjusted_path, index=False)
    model_ladder.to_csv(model_ladder_path, index=False)
    multiple_testing.to_csv(multiple_testing_path, index=False)

    sensitivity_rows = []
    for row in sensitivity_boot_summary.to_dict(orient="records"):
        sensitivity_rows.append(row)
    for row in sensitivity.coefficients.to_dict(orient="records"):
        row = {"metric": "fixed_position_model_coefficient", **row}
        sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    univariate.to_csv(univariate_path, index=False)

    figures = _plot_main_figure(
        primary.profile,
        primary.coefficients,
        boot_summary,
        direct_adjusted,
        model_ladder,
    )
    supplementary_figures = _plot_supplementary_coefficient_overlap(primary.coefficients, boot_summary)

    direct_adjusted_direct_row = direct_adjusted[direct_adjusted["predictor"].eq(OUTCOME_DIRECT)].iloc[0]
    ladder_lookup = model_ladder.set_index("model")

    manifest = {
        "analysis": "profile_trait_direct_vs_network",
        "created_by": "build_profile_trait_direct_vs_network.py",
        "seed": SEED,
        "n_bootstrap_requested": N_BOOT,
        "primary_source": primary.summary,
        "sensitivity_source": sensitivity.summary,
        "tables": {
            "profile_summary": str(profile_path),
            "coefficients": str(coefficients_path),
            "bootstrap_summary": str(bootstrap_path),
            "direct_adjusted_coefficients": str(direct_adjusted_path),
            "model_ladder": str(model_ladder_path),
            "multiple_testing_sensitivity": str(multiple_testing_path),
            "fixed_position_sensitivity": str(sensitivity_path),
            "univariate_supplement": str(univariate_path),
        },
        "figures": {k: str(v) for k, v in figures.items()},
        "supplementary_figures": {
            f"coefficient_overlap_{k}": str(v) for k, v in supplementary_figures.items()
        },
        "key_results": {
            "profile_outcome_correlation": float(
                boot_summary.loc[
                    boot_summary["metric"].eq("profile_outcome_correlation"), "observed"
                ].iloc[0]
            ),
            "profile_outcome_correlation_ci": [
                float(boot_summary.loc[boot_summary["metric"].eq("profile_outcome_correlation"), "ci_low"].iloc[0]),
                float(boot_summary.loc[boot_summary["metric"].eq("profile_outcome_correlation"), "ci_high"].iloc[0]),
            ],
            "focal_coefficient_vector_similarity": float(
                boot_summary.loc[
                    boot_summary["metric"].eq("focal_coefficient_vector_similarity"), "observed"
                ].iloc[0]
            ),
            "focal_coefficient_vector_similarity_ci": [
                float(
                    boot_summary.loc[
                        boot_summary["metric"].eq("focal_coefficient_vector_similarity"), "ci_low"
                    ].iloc[0]
                ),
                float(
                    boot_summary.loc[
                        boot_summary["metric"].eq("focal_coefficient_vector_similarity"), "ci_high"
                    ].iloc[0]
                ),
            ],
            "direct_adjusted_direct_susceptibility_beta": float(direct_adjusted_direct_row["beta"]),
            "direct_adjusted_direct_susceptibility_hc3_ci": [
                float(direct_adjusted_direct_row["hc3_ci_low"]),
                float(direct_adjusted_direct_row["hc3_ci_high"]),
            ],
            "direct_adjusted_direct_susceptibility_hc3_p": float(direct_adjusted_direct_row["hc3_p"]),
            "network_model_r_squared": {
                "direct_only": float(ladder_lookup.loc["direct_only", "r_squared"]),
                "traits_only": float(ladder_lookup.loc["traits_only", "r_squared"]),
                "traits_plus_direct": float(ladder_lookup.loc["traits_plus_direct", "r_squared"]),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "profile_summary": str(profile_path),
        "coefficients": str(coefficients_path),
        "bootstrap_summary": str(bootstrap_path),
        "direct_adjusted_coefficients": str(direct_adjusted_path),
        "model_ladder": str(model_ladder_path),
        "multiple_testing_sensitivity": str(multiple_testing_path),
        "fixed_position_sensitivity": str(sensitivity_path),
        "univariate_supplement": str(univariate_path),
        "manifest": str(manifest_path),
        **{f"figure_{k}": str(v) for k, v in figures.items()},
        **{f"supplementary_coefficient_overlap_{k}": str(v) for k, v in supplementary_figures.items()},
    }


def _validate_primary(primary: ModelOutputs) -> None:
    if primary.summary["row_count"] != 3500:
        raise ValueError(f"Expected 3500 primary rows, got {primary.summary['row_count']}")
    if primary.summary["profile_count"] != 100:
        raise ValueError(f"Expected 100 primary profiles, got {primary.summary['profile_count']}")
    if primary.summary["min_rows_per_profile"] != 35 or primary.summary["max_rows_per_profile"] != 35:
        raise ValueError(
            "Expected exactly 35 profile x condition measurements per primary profile, "
            f"got min={primary.summary['min_rows_per_profile']} max={primary.summary['max_rows_per_profile']}"
        )
    if primary.summary["post_network_fallback_count"] != 0:
        raise ValueError(f"Expected 0 primary post-network fallbacks, got {primary.summary['post_network_fallback_count']}")


def main() -> None:
    primary = _run_analysis(
        "alignment_gradient_branch",
        PRIMARY_PATH,
        exclude_post_network_fallback=False,
    )
    _validate_primary(primary)
    sensitivity = _run_analysis(
        "fixed_position_main_run_valid_post_network_rows",
        SENSITIVITY_PATH,
        exclude_post_network_fallback=True,
    )
    outputs = _write_outputs(primary, sensitivity)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
