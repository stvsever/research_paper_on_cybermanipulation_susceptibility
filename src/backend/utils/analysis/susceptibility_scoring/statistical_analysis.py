"""
Statistical Analysis Module
-----------------------------
Runs a comprehensive suite of statistical tests on the run's long-format
effectivity data. Automatically selects parametric vs. non-parametric tests
based on Shapiro-Wilk normality results.

Tests performed
---------------
Per opinion leaf (and per opinion×attack):
  1. Shapiro-Wilk normality test on adversarial_effectivity and abs_delta_score
  2. Sex group differences:
       normal  → one-way ANOVA + Tukey HSD post-hoc
       non-normal → Kruskal-Wallis + Dunn post-hoc (Bonferroni)
  3. Age × effectivity:
       normal  → Pearson r
       non-normal → Spearman ρ
  4. Big Five correlations with adversarial_effectivity:
       Pearson (normal) or Spearman (non-normal)
  5. Baseline score as covariate: partial correlation controlling for baseline
  6. Effect sizes:
       group differences → eta-squared (ANOVA) or epsilon-squared (Kruskal-Wallis)
       correlations → r² or ρ²

Output
------
  statistical_tests/
    normality_results.csv
    sex_group_tests.csv
    age_correlations.csv
    big_five_correlations.csv
    full_statistical_report.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05
NORMALITY_ALPHA = 0.05

BIG_FIVE_MEANS = {
    "profile_cont_big_five_agreeableness_mean_pct": "Agreeableness",
    "profile_cont_big_five_conscientiousness_mean_pct": "Conscientiousness",
    "profile_cont_big_five_extraversion_mean_pct": "Extraversion",
    "profile_cont_big_five_neuroticism_mean_pct": "Neuroticism",
    "profile_cont_big_five_openness_to_experience_mean_pct": "Openness",
}

BIG_FIVE_FACETS = {
    # Neuroticism facets
    "profile_cont_big_five_neuroticism_anxiety_pct": "N_Anxiety",
    "profile_cont_big_five_neuroticism_anger_hostility_pct": "N_Anger",
    "profile_cont_big_five_neuroticism_depression_pct": "N_Depression",
    "profile_cont_big_five_neuroticism_impulsiveness_pct": "N_Impulsiveness",
    "profile_cont_big_five_neuroticism_self_consciousness_pct": "N_SelfConsciousness",
    "profile_cont_big_five_neuroticism_stress_vulnerability_pct": "N_StressVulnerability",
    # Conscientiousness facets
    "profile_cont_big_five_conscientiousness_competence_pct": "C_Competence",
    "profile_cont_big_five_conscientiousness_order_pct": "C_Order",
    "profile_cont_big_five_conscientiousness_dutifulness_pct": "C_Dutifulness",
    "profile_cont_big_five_conscientiousness_achievement_striving_pct": "C_AchievementStriving",
    "profile_cont_big_five_conscientiousness_self_discipline_pct": "C_SelfDiscipline",
    "profile_cont_big_five_conscientiousness_deliberation_pct": "C_Deliberation",
}

OUTCOME_COLS = ["adversarial_effectivity", "abs_delta_score", "delta_score"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _leaf_short(leaf: str) -> str:
    return leaf.split(">")[-1].strip()


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return "ns"


def _effect_size_label(r2: float) -> str:
    r = abs(r2) ** 0.5
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    return "large"


def _eta_squared(groups: List[np.ndarray]) -> float:
    """Eta-squared from group arrays."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _epsilon_squared(h_stat: float, n: int) -> float:
    """Epsilon-squared from Kruskal-Wallis H statistic."""
    return float((h_stat - 1) / (n - 1)) if n > 1 else 0.0


def _partial_correlation(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> Tuple[float, float]:
    """Partial correlation of x and y controlling for covariate."""
    from scipy.stats import pearsonr
    resid_x = x - np.polyval(np.polyfit(covariate, x, 1), covariate)
    resid_y = y - np.polyval(np.polyfit(covariate, y, 1), covariate)
    r, p = pearsonr(resid_x, resid_y)
    return float(r), float(p)


def _dunn_posthoc(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Simple Dunn post-hoc test with Bonferroni correction."""
    groups = df[group_col].unique()
    results = []
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i+1:]]
    n_comp = len(pairs)
    for g1, g2 in pairs:
        v1 = df[df[group_col] == g1][value_col].dropna().values
        v2 = df[df[group_col] == g2][value_col].dropna().values
        if len(v1) < 2 or len(v2) < 2:
            continue
        u, p_raw = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        p_adj = min(p_raw * n_comp, 1.0)  # Bonferroni
        results.append({
            "group_1": g1, "group_2": g2,
            "n1": len(v1), "n2": len(v2),
            "U": float(u), "p_raw": float(p_raw),
            "p_bonferroni": float(p_adj),
            "sig": _stars(p_adj),
        })
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────────────────────

def run_normality_tests(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Shapiro-Wilk normality test per outcome × opinion leaf (× attack leaf)."""
    rows = []
    for outcome in OUTCOME_COLS:
        if outcome not in df.columns:
            continue
        for opinion, grp in df.groupby("opinion_leaf"):
            vals = grp[outcome].dropna().values
            if len(vals) < 3:
                continue
            stat, p = stats.shapiro(vals)
            rows.append({
                "outcome": outcome,
                "opinion_leaf": _leaf_short(str(opinion)),
                "attack_leaf": _leaf_short(str(grp["attack_leaf"].iloc[0])) if "attack_leaf" in grp else "all",
                "n": len(vals),
                "shapiro_W": round(float(stat), 4),
                "shapiro_p": round(float(p), 4),
                "is_normal": p >= NORMALITY_ALPHA,
                "sig": _stars(p),
            })
    return pd.DataFrame(rows)


def run_sex_group_tests(df: pd.DataFrame, normality: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group difference tests (ANOVA or Kruskal-Wallis) by sex × opinion leaf."""
    # decode sex from one-hot
    if "profile_cat__profile_cat_sex_Female" in df.columns:
        df = df.copy()
        df["sex"] = "Other"
        df.loc[df["profile_cat__profile_cat_sex_Female"] == 1, "sex"] = "Female"
        df.loc[df["profile_cat__profile_cat_sex_Male"] == 1, "sex"] = "Male"
    elif "sex" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    main_rows = []
    posthoc_rows = []

    for outcome in ["adversarial_effectivity", "abs_delta_score"]:
        if outcome not in df.columns:
            continue
        for opinion, grp in df.groupby("opinion_leaf"):
            opinion_short = _leaf_short(str(opinion))
            attack_short = _leaf_short(str(grp["attack_leaf"].iloc[0])) if "attack_leaf" in grp else "all"

            # Determine normality for this outcome × opinion
            norm_row = normality[
                (normality["outcome"] == outcome) &
                (normality["opinion_leaf"] == opinion_short)
            ]
            is_normal = bool(norm_row["is_normal"].iloc[0]) if len(norm_row) > 0 else False

            sex_groups = {s: grp[grp["sex"] == s][outcome].dropna().values
                          for s in ["Female", "Male", "Other"]}
            sex_groups = {s: v for s, v in sex_groups.items() if len(v) >= 2}

            if len(sex_groups) < 2:
                continue

            group_arrays = list(sex_groups.values())

            if is_normal:
                f_stat, p_val = stats.f_oneway(*group_arrays)
                eta2 = _eta_squared(group_arrays)
                main_rows.append({
                    "outcome": outcome, "opinion_leaf": opinion_short, "attack_leaf": attack_short,
                    "test": "one-way ANOVA", "statistic": round(float(f_stat), 4),
                    "p_value": round(float(p_val), 4), "sig": _stars(float(p_val)),
                    "effect_size_name": "eta_squared", "effect_size": round(eta2, 4),
                    "effect_size_label": _effect_size_label(eta2),
                    "n_total": sum(len(v) for v in group_arrays),
                })
            else:
                h_stat, p_val = stats.kruskal(*group_arrays)
                n_total = sum(len(v) for v in group_arrays)
                eps2 = _epsilon_squared(float(h_stat), n_total)
                main_rows.append({
                    "outcome": outcome, "opinion_leaf": opinion_short, "attack_leaf": attack_short,
                    "test": "Kruskal-Wallis", "statistic": round(float(h_stat), 4),
                    "p_value": round(float(p_val), 4), "sig": _stars(float(p_val)),
                    "effect_size_name": "epsilon_squared", "effect_size": round(eps2, 4),
                    "effect_size_label": _effect_size_label(eps2),
                    "n_total": n_total,
                })
                # Post-hoc Dunn
                ph = _dunn_posthoc(grp.assign(sex=df.loc[grp.index, "sex"]) if "sex" not in grp.columns else grp,
                                   "sex", outcome)
                ph["outcome"] = outcome
                ph["opinion_leaf"] = opinion_short
                ph["attack_leaf"] = attack_short
                posthoc_rows.append(ph)

    main_df = pd.DataFrame(main_rows)
    posthoc_df = pd.concat(posthoc_rows, ignore_index=True) if posthoc_rows else pd.DataFrame()
    return main_df, posthoc_df


def run_continuous_correlations(
    df: pd.DataFrame,
    normality: pd.DataFrame,
    feature_cols: Dict[str, str],
    outcome: str = "adversarial_effectivity",
) -> pd.DataFrame:
    """Pearson or Spearman correlations of continuous features with outcome, per opinion leaf."""
    if outcome not in df.columns:
        return pd.DataFrame()

    rows = []
    for opinion, grp in df.groupby("opinion_leaf"):
        opinion_short = _leaf_short(str(opinion))
        attack_short = _leaf_short(str(grp["attack_leaf"].iloc[0])) if "attack_leaf" in grp else "all"

        norm_row = normality[
            (normality["outcome"] == outcome) &
            (normality["opinion_leaf"] == opinion_short)
        ]
        is_normal = bool(norm_row["is_normal"].iloc[0]) if len(norm_row) > 0 else False

        y = grp[outcome].dropna().values

        for col, label in feature_cols.items():
            if col not in grp.columns:
                continue
            x = grp[col].dropna().values
            # Align on common index
            valid = grp[[col, outcome]].dropna()
            if len(valid) < 5:
                continue
            x_v = valid[col].values
            y_v = valid[outcome].values

            if is_normal:
                r, p = stats.pearsonr(x_v, y_v)
                test_name = "Pearson"
            else:
                r, p = stats.spearmanr(x_v, y_v)
                test_name = "Spearman"

            # Also compute partial correlation controlling for baseline
            partial_r, partial_p = float("nan"), float("nan")
            if "baseline_score" in grp.columns:
                bl_valid = grp[[col, outcome, "baseline_score"]].dropna()
                if len(bl_valid) >= 5:
                    partial_r, partial_p = _partial_correlation(
                        bl_valid[col].values,
                        bl_valid[outcome].values,
                        bl_valid["baseline_score"].values,
                    )

            rows.append({
                "outcome": outcome,
                "opinion_leaf": opinion_short,
                "attack_leaf": attack_short,
                "feature": label,
                "feature_col": col,
                "test": test_name,
                "r": round(float(r), 4),
                "r_squared": round(float(r) ** 2, 4),
                "p_value": round(float(p), 4),
                "sig": _stars(float(p)),
                "effect_label": _effect_size_label(float(r) ** 2),
                "partial_r_controlling_baseline": round(float(partial_r), 4) if not np.isnan(partial_r) else None,
                "partial_p": round(float(partial_p), 4) if not np.isnan(partial_p) else None,
                "n": len(valid),
                "is_normal": is_normal,
            })

    return pd.DataFrame(rows)


def run_opinion_leaf_comparisons(df: pd.DataFrame, normality: pd.DataFrame) -> pd.DataFrame:
    """Compare adversarial effectivity across opinion leaves (is one leaf more vulnerable?)."""
    outcome = "adversarial_effectivity"
    if outcome not in df.columns:
        return pd.DataFrame()

    groups = {_leaf_short(str(k)): v[outcome].dropna().values
              for k, v in df.groupby("opinion_leaf")}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return pd.DataFrame()

    arrs = list(groups.values())
    # Global normality: all leaves combined
    all_vals = np.concatenate(arrs)
    _, p_norm = stats.shapiro(all_vals)
    is_normal = p_norm >= NORMALITY_ALPHA

    rows = []
    if is_normal:
        f_stat, p_val = stats.f_oneway(*arrs)
        eta2 = _eta_squared(arrs)
        rows.append({
            "comparison": "opinion_leaf_effect_on_adversarial_effectivity",
            "test": "one-way ANOVA", "statistic": round(float(f_stat), 4),
            "p_value": round(float(p_val), 4), "sig": _stars(float(p_val)),
            "effect_size_name": "eta_squared", "effect_size": round(eta2, 4),
            "n_groups": len(groups), "n_total": len(all_vals),
        })
    else:
        h_stat, p_val = stats.kruskal(*arrs)
        eps2 = _epsilon_squared(float(h_stat), len(all_vals))
        rows.append({
            "comparison": "opinion_leaf_effect_on_adversarial_effectivity",
            "test": "Kruskal-Wallis", "statistic": round(float(h_stat), 4),
            "p_value": round(float(p_val), 4), "sig": _stars(float(p_val)),
            "effect_size_name": "epsilon_squared", "effect_size": round(eps2, 4),
            "n_groups": len(groups), "n_total": len(all_vals),
        })

    # Pairwise Mann-Whitney between opinion leaves
    leaf_names = list(groups.keys())
    for i, a in enumerate(leaf_names):
        for b in leaf_names[i+1:]:
            u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            p_adj = min(p * (len(leaf_names) * (len(leaf_names) - 1) / 2), 1.0)
            r_eff = 1.0 - (2 * float(u)) / (len(groups[a]) * len(groups[b])) if len(groups[a]) * len(groups[b]) > 0 else 0.0
            rows.append({
                "comparison": f"{a} vs {b}",
                "test": "Mann-Whitney U (pairwise)", "statistic": round(float(u), 2),
                "p_value": round(float(p), 4), "sig": _stars(float(p_adj)),
                "effect_size_name": "r_effect", "effect_size": round(abs(r_eff), 4),
                "n_groups": 2, "n_total": len(groups[a]) + len(groups[b]),
            })

    return pd.DataFrame(rows)


def run_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics per outcome × opinion leaf × attack leaf."""
    rows = []
    for outcome in OUTCOME_COLS:
        if outcome not in df.columns:
            continue
        for (opinion, attack), grp in df.groupby(["opinion_leaf", "attack_leaf"]):
            vals = grp[outcome].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                "outcome": outcome,
                "opinion_leaf": _leaf_short(str(opinion)),
                "attack_leaf": _leaf_short(str(attack)),
                "n": len(vals),
                "mean": round(float(np.mean(vals)), 3),
                "median": round(float(np.median(vals)), 3),
                "std": round(float(np.std(vals, ddof=1)), 3),
                "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 3),
                "min": round(float(np.min(vals)), 3),
                "max": round(float(np.max(vals)), 3),
                "pct_positive": round(float(np.mean(vals > 0)) * 100, 1),
                "skewness": round(float(stats.skew(vals)), 3),
                "kurtosis": round(float(stats.kurtosis(vals)), 3),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Report renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_statistical_report(
    normality: pd.DataFrame,
    sex_tests: pd.DataFrame,
    sex_posthoc: pd.DataFrame,
    age_corr: pd.DataFrame,
    bf_means_corr: pd.DataFrame,
    bf_facets_corr: pd.DataFrame,
    leaf_comparisons: pd.DataFrame,
    descriptives: pd.DataFrame,
    run_id: str,
) -> str:
    lines = [
        "=" * 76,
        f"STATISTICAL ANALYSIS REPORT — {run_id.upper()}",
        "Normality: Shapiro-Wilk (SW). Normal → parametric; non-normal → non-parametric.",
        f"Significance threshold: α = {ALPHA}. Effect sizes reported throughout.",
        "Partial correlations control for baseline opinion score.",
        "=" * 76,
        "",
    ]

    # Descriptive stats
    lines += ["DESCRIPTIVE STATISTICS", "─" * 60]
    if not descriptives.empty:
        for outcome in descriptives["outcome"].unique():
            lines.append(f"\n  Outcome: {outcome}")
            sub = descriptives[descriptives["outcome"] == outcome]
            for _, r in sub.iterrows():
                lines.append(
                    f"    {r['opinion_leaf']:<45} n={r['n']:>3}  "
                    f"mean={r['mean']:>8.2f}  sd={r['std']:>7.2f}  "
                    f"median={r['median']:>8.2f}  IQR={r['iqr']:>6.2f}  "
                    f"skew={r['skewness']:>5.2f}  pct+={r['pct_positive']:.0f}%"
                )
    lines.append("")

    # Normality
    lines += ["NORMALITY TESTS (Shapiro-Wilk)", "─" * 60]
    if not normality.empty:
        for outcome in normality["outcome"].unique():
            lines.append(f"\n  Outcome: {outcome}")
            sub = normality[normality["outcome"] == outcome]
            for _, r in sub.iterrows():
                status = "NORMAL" if r["is_normal"] else "NON-NORMAL"
                lines.append(
                    f"    {r['opinion_leaf']:<45} W={r['shapiro_W']:.4f}  "
                    f"p={r['shapiro_p']:.4f} {r['sig']}  → {status}"
                )
    lines.append("")

    # Opinion leaf comparison
    lines += ["OPINION LEAF DIFFERENCES IN ADVERSARIAL EFFECTIVITY", "─" * 60]
    if not leaf_comparisons.empty:
        for _, r in leaf_comparisons.iterrows():
            lines.append(
                f"  {r['comparison']:<55} {r['test']}  "
                f"stat={r['statistic']:.3f}  p={r['p_value']:.4f}{r['sig']}  "
                f"{r['effect_size_name']}={r['effect_size']:.4f}"
            )
    lines.append("")

    # Sex group tests
    lines += ["SEX GROUP DIFFERENCES", "─" * 60]
    if not sex_tests.empty:
        for outcome in sex_tests["outcome"].unique():
            lines.append(f"\n  Outcome: {outcome}")
            sub = sex_tests[sex_tests["outcome"] == outcome]
            for _, r in sub.iterrows():
                lines.append(
                    f"    {r['opinion_leaf']:<45} {r['test']}  "
                    f"stat={r['statistic']:.3f}  p={r['p_value']:.4f}{r['sig']}  "
                    f"{r['effect_size_name']}={r['effect_size']:.4f} ({r['effect_size_label']})"
                )
    if not sex_posthoc.empty:
        lines.append("\n  Post-hoc pairwise (Bonferroni-corrected Mann-Whitney):")
        for _, r in sex_posthoc[sex_posthoc["p_bonferroni"] < 0.10].iterrows():
            lines.append(
                f"    {r.get('opinion_leaf','')} | {r['group_1']} vs {r['group_2']}  "
                f"U={r['U']:.1f}  p_adj={r['p_bonferroni']:.4f}{r['sig']}"
            )
    lines.append("")

    # Age correlations
    lines += ["AGE CORRELATIONS WITH ADVERSARIAL EFFECTIVITY", "─" * 60]
    if not age_corr.empty:
        for _, r in age_corr.iterrows():
            partial_str = (f"  partial_r={r['partial_r_controlling_baseline']:.3f}(p={r['partial_p']:.3f})"
                           if r.get("partial_r_controlling_baseline") is not None else "")
            lines.append(
                f"  {r['opinion_leaf']:<45} {r['test']} r={r['r']:+.3f}  r²={r['r_squared']:.3f}  "
                f"p={r['p_value']:.4f}{r['sig']}  {r['effect_label']}{partial_str}"
            )
    lines.append("")

    # Big Five means correlations
    lines += ["BIG FIVE TRAIT MEANS × ADVERSARIAL EFFECTIVITY", "─" * 60]
    if not bf_means_corr.empty:
        for opinion in bf_means_corr["opinion_leaf"].unique():
            lines.append(f"\n  Opinion: {opinion}")
            sub = bf_means_corr[bf_means_corr["opinion_leaf"] == opinion]
            for _, r in sub.sort_values("p_value").iterrows():
                partial_str = (f"  partial_r={r['partial_r_controlling_baseline']:.3f}"
                               if r.get("partial_r_controlling_baseline") is not None else "")
                lines.append(
                    f"    {r['feature']:<30} {r['test']} r={r['r']:+.3f}  r²={r['r_squared']:.3f}  "
                    f"p={r['p_value']:.4f}{r['sig']}  {r['effect_label']}{partial_str}"
                )
    lines.append("")

    # Big Five facets correlations (significant only)
    lines += ["BIG FIVE FACETS × ADVERSARIAL EFFECTIVITY (significant p<.10 only)", "─" * 60]
    if not bf_facets_corr.empty:
        sig_facets = bf_facets_corr[bf_facets_corr["p_value"] < 0.10].sort_values("p_value")
        for _, r in sig_facets.iterrows():
            lines.append(
                f"  {r['opinion_leaf']:<30} {r['feature']:<30} r={r['r']:+.3f}  "
                f"p={r['p_value']:.4f}{r['sig']}"
            )
    lines.append("")
    lines.append("=" * 76)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_full_analysis(sem_long_path: Path, output_dir: Path, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sem_long_path)
    print(f"  Loaded {len(df)} rows from {sem_long_path.name}")

    print("  Running descriptive stats ...")
    descriptives = run_descriptive_stats(df)
    descriptives.to_csv(output_dir / "descriptive_statistics.csv", index=False)

    print("  Running normality tests ...")
    normality = run_normality_tests(df, ["opinion_leaf"])
    normality.to_csv(output_dir / "normality_results.csv", index=False)

    print("  Running opinion leaf comparisons ...")
    leaf_comparisons = run_opinion_leaf_comparisons(df, normality)
    leaf_comparisons.to_csv(output_dir / "opinion_leaf_comparisons.csv", index=False)

    print("  Running sex group tests ...")
    sex_tests, sex_posthoc = run_sex_group_tests(df, normality)
    sex_tests.to_csv(output_dir / "sex_group_tests.csv", index=False)
    sex_posthoc.to_csv(output_dir / "sex_posthoc_tests.csv", index=False)

    print("  Running age correlations ...")
    age_col = {"profile_cont_age_years": "Age"}
    age_corr = run_continuous_correlations(df, normality, age_col)
    age_corr.to_csv(output_dir / "age_correlations.csv", index=False)

    print("  Running Big Five means correlations ...")
    bf_means_corr = run_continuous_correlations(df, normality, BIG_FIVE_MEANS)
    bf_means_corr.to_csv(output_dir / "big_five_means_correlations.csv", index=False)

    print("  Running Big Five facets correlations ...")
    bf_facets_corr = run_continuous_correlations(df, normality, BIG_FIVE_FACETS)
    bf_facets_corr.to_csv(output_dir / "big_five_facets_correlations.csv", index=False)

    print("  Rendering report ...")
    report = render_statistical_report(
        normality, sex_tests, sex_posthoc,
        age_corr, bf_means_corr, bf_facets_corr,
        leaf_comparisons, descriptives, run_id
    )
    (output_dir / "full_statistical_report.txt").write_text(report, encoding="utf-8")

    print(f"  Statistical analysis complete → {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sem_long = None
    for sub in [run_dir / "stage_outputs" / "05_compute_effectivity_deltas", run_dir]:
        candidate = sub / "sem_long_encoded.csv"
        if candidate.exists():
            sem_long = candidate
            break
    if sem_long is None:
        print("sem_long_encoded.csv not found", file=sys.stderr)
        sys.exit(1)

    run_full_analysis(sem_long, Path(args.output_dir).resolve(), args.run_id or run_dir.name)
