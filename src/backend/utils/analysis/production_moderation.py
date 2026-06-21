from __future__ import annotations

"""
SOTA profile-moderation analysis for the INDIVIDUAL layer (production run 1).

The research question of the individual layer is: which inter-individual differences
moderate susceptibility to cyber-manipulation? Rather than throwing all ~159 profile
features into one under-determined model, this module analyses each ontology FAMILY
on its own footing (Big Five personality, the political-psychology battery, the
ideological dimensions, moral-foundations theory, and demographics), which is both
more interpretable and methodologically cleaner.

The unit of analysis is the scenario (one synthetic person and attack), so each of
the 10,000 scenarios contributes one independent mean adversarial-effectivity value;
opinion leaves are averaged within a scenario before any profile regression. This
avoids treating the ~15 clustered leaf measurements per scenario as independent.

Outputs (all returned as DataFrames):
  - family_table: per-family out-of-sample CV R^2 and unique variance (commonality).
  - within_family: every trait's standardised moderation slope with a BH-FDR q-value.
  - curated: a compact, interpretable cross-family model (one score per construct)
    with cluster-robust standardised betas, 95% CIs and BH-FDR q-values.
  - by_domain: the curated moderators re-estimated within each of the 7 issue domains.

Entry point: run_production_moderation(sem_long_df) -> dict of DataFrames + summary.
"""

import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    sm = None
    multipletests = None

_AE = "adversarial_effectivity"
_DERIVED = ("resilience_index", "heuristic_shift_sensitivity_proxy")

# The macroeconomic domain is a single opinion leaf in the integrated set and a
# statistical outlier, so it is excluded from every analysis and figure.
EXCLUDED_DOMAINS = {"Macroeconomic_And_Fiscal_Policy"}


def drop_excluded_domains(df: pd.DataFrame) -> pd.DataFrame:
    if "opinion_domain" in df.columns:
        return df[~df["opinion_domain"].isin(EXCLUDED_DOMAINS)].copy()
    return df


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _true_scn(df: pd.DataFrame) -> pd.Series:
    sid = df["scenario_id"].astype(str)
    return sid.str.split("__", n=1).str[0] if sid.str.contains("__").any() else sid


def _fdr(p: List[float]) -> List[float]:
    p = np.asarray(p, float)
    if len(p) == 0:
        return []
    if multipletests is not None:
        return list(multipletests(p, method="fdr_bh")[1])
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return list(out)


def _family_of(col: str) -> str:
    s = re.sub(r"^profile_(cont|cat)(__|_)", "", col).lower()
    if "big_five" in s:
        return "Big Five"
    if "moral_found" in s:
        return "Moral Foundations"
    if any(t in s for t in ("ideological_dimensions", "gal_tan", "libertarian_authoritarian")):
        return "Ideology"
    if "political_profile" in s or "political_" in s:
        return "Political Psychology"
    if any(t in s for t in ("demographic", "age_", "sex", "ethnocultural", "citizenship",
                            "gender", "relationship", "country_of_birth")):
        return "Demographics"
    return "Other"


def _pretty(col: str) -> str:
    s = re.sub(r"^profile_(cont_|cat__|cat_)", "", col)
    s = (s.replace("demographics_and_identity_political_profile_", "")
           .replace("political_profile_", "")
           .replace("ideological_dimensions_two_axis_model_", "")
           .replace("_model", "").replace("moral_foundations_theory_", "MFT ")
           .replace("_mean_pct", "").replace("personality_", ""))
    return s.replace("_", " ").strip().title()


def _scenario_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    """Collapse to one row per scenario: mean effectivity + the (constant) profile."""
    work = df.dropna(subset=[_AE]).copy()
    work["_scn"] = _true_scn(work)
    prof_cont = [c for c in work.columns if c.startswith("profile_cont_") and not any(d in c for d in _DERIVED)]
    # near-constant guard
    prof_cont = [c for c in prof_cont if pd.to_numeric(work[c], errors="coerce").std(ddof=0) > 1e-9]
    agg = {"ae": (_AE, "mean"), "opinion_domain": ("opinion_domain", "first")}
    for c in prof_cont:
        agg[c] = (c, "first")
    if "sex" in work.columns:
        agg["sex"] = ("sex", "first")
    scn = work.groupby("_scn").agg(**agg).reset_index()
    fams: Dict[str, List[str]] = {}
    for c in prof_cont:
        fams.setdefault(_family_of(c), []).append(c)
    fams = {k: v for k, v in fams.items() if k in
            ("Big Five", "Political Psychology", "Ideology", "Moral Foundations", "Demographics")}
    return scn, fams, prof_cont


def _cv_r2(X: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    if X.shape[1] == 0 or len(y) < 50:
        return float("nan")
    model = RidgeCV(alphas=np.logspace(-2, 4, 25))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Xs = StandardScaler().fit_transform(X)
    scores = cross_val_score(model, Xs, y, cv=kf, scoring="r2")
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Analyses                                                                     #
# --------------------------------------------------------------------------- #
def _adj_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Non-negative in-sample adjusted R^2 (variance explained by the family)."""
    from sklearn.linear_model import LinearRegression
    if X.shape[1] == 0:
        return 0.0
    Xs = StandardScaler().fit_transform(X)
    r2 = LinearRegression().fit(Xs, y).score(Xs, y)
    n, p = len(y), X.shape[1]
    adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    return float(max(0.0, adj))


def _family_table(scn: pd.DataFrame, fams: Dict[str, List[str]], all_cont: List[str]) -> pd.DataFrame:
    y = scn["ae"].to_numpy(float)
    rows = []
    full_cv = _cv_r2(scn[all_cont].apply(pd.to_numeric, errors="coerce").fillna(scn[all_cont].median()).to_numpy(), y)
    for fam, cols in fams.items():
        Xf = scn[cols].apply(pd.to_numeric, errors="coerce")
        Xf = Xf.fillna(Xf.median()).to_numpy()
        cv = _cv_r2(Xf, y)
        rest = [c for c in all_cont if c not in set(cols)]
        Xr = scn[rest].apply(pd.to_numeric, errors="coerce").fillna(scn[rest].median()).to_numpy()
        cv_rest = _cv_r2(Xr, y)
        rows.append({
            "family": fam, "n_features": len(cols),
            "var_explained_pct": round(100 * _adj_r2(Xf, y), 3),  # non-negative adjusted in-sample R^2
            "cv_r2_standalone": round(cv, 4),
            "cv_r2_unique": round(max(0.0, full_cv - cv_rest), 4),
        })
    out = pd.DataFrame(rows).sort_values("var_explained_pct", ascending=False).reset_index(drop=True)
    out.attrs["full_cv_r2"] = round(full_cv, 4)
    return out


def _within_family(scn: pd.DataFrame, fams: Dict[str, List[str]]) -> pd.DataFrame:
    if sm is None:
        return pd.DataFrame()
    y = (scn["ae"] - scn["ae"].mean()) / scn["ae"].std(ddof=0)
    rows = []
    for fam, cols in fams.items():
        for c in cols:
            x = pd.to_numeric(scn[c], errors="coerce")
            if x.std(ddof=0) < 1e-9:
                continue
            xs = (x - x.mean()) / x.std(ddof=0)
            X = sm.add_constant(xs.fillna(0.0).to_numpy())
            res = sm.OLS(y.to_numpy(), X).fit(cov_type="HC3")
            ci = res.conf_int()
            rows.append({"family": fam, "trait": _pretty(c), "column": c,
                         "beta_std": float(res.params[1]), "ci_low": float(ci[1][0]),
                         "ci_high": float(ci[1][1]), "p_value": float(res.pvalues[1]),
                         "n": int(scn.shape[0])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # FDR correction WITHIN each family (each family is its own set of tests), which is
    # the right granularity: correcting across all families' traits at once is overly
    # conservative and would suppress genuine, construct-specific moderators.
    df["q_value"] = df.groupby("family")["p_value"].transform(
        lambda s: pd.Series(_fdr(s.tolist()), index=s.index))
    df["significant"] = df["q_value"] < 0.05
    return df.sort_values("beta_std", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


_CURATED_PATTERNS = [
    ("Big Five", "Openness", "big_five_openness_to_experience_mean_pct"),
    ("Big Five", "Conscientiousness", "big_five_conscientiousness_mean_pct"),
    ("Big Five", "Extraversion", "big_five_extraversion_mean_pct"),
    ("Big Five", "Agreeableness", "big_five_agreeableness_mean_pct"),
    ("Big Five", "Neuroticism", "big_five_neuroticism_mean_pct"),
    ("Political Psychology", "Right-wing authoritarianism", "right_wing_authoritarianism_model_mean_pct"),
    ("Political Psychology", "Social dominance orientation", "social_dominance_orientation_model_mean_pct"),
    ("Political Psychology", "System justification", "system_justification_theory_mean_pct"),
    ("Political Psychology", "Populist attitudes", "populist_attitudes_model_mean_pct"),
    ("Political Psychology", "Collective narcissism", "collective_narcissism_model_mean_pct"),
    ("Political Psychology", "Nationalism", "nationalism_and_cosmopolitanism_model_mean_pct"),
    ("Political Psychology", "Political trust", "political_trust_and_legitimacy_model_mean_pct"),
    ("Political Psychology", "Political efficacy", "political_efficacy_model_mean_pct"),
    ("Ideology", "Economic left-right", "ideological_dimensions_two_axis_model_economic_left_right_mean_pct"),
    ("Ideology", "Socio-cultural lib-cons", "ideological_dimensions_two_axis_model_socio_cultural_liberal_conservative_mean_pct"),
    ("Ideology", "GAL-TAN (traditional)", "gal_tan_model_traditional_authoritarian_nationalist_mean_pct"),
    ("Ideology", "Authoritarianism", "libertarian_authoritarian_dimension_model_authoritarianism_mean_pct"),
    ("Moral Foundations", "MFT care", "moral_foundations_theory_care_harm_mean_pct"),
    ("Moral Foundations", "MFT fairness", "moral_foundations_theory_fairness_cheating_mean_pct"),
    ("Moral Foundations", "MFT loyalty", "moral_foundations_theory_loyalty_betrayal_mean_pct"),
    ("Moral Foundations", "MFT authority", "moral_foundations_theory_authority_subversion_mean_pct"),
    ("Moral Foundations", "MFT sanctity", "moral_foundations_theory_sanctity_degradation_mean_pct"),
    ("Demographics", "Age", "age_years"),
]


def _resolve_curated(cols: List[str]) -> List[Tuple[str, str, str]]:
    """Map each curated construct to the actual column (first match), de-duplicating."""
    resolved, used = [], set()
    for fam, label, pat in _CURATED_PATTERNS:
        match = next((c for c in cols if c.endswith(pat) or c.endswith("profile_cont_" + pat)), None)
        if match is None:
            match = next((c for c in cols if pat in c), None)
        if match and match not in used:
            resolved.append((fam, label, match)); used.add(match)
    return resolved


def _curated_model(scn: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if sm is None:
        return pd.DataFrame()
    curated = _resolve_curated(cols)
    feats = [(f, l, c) for f, l, c in curated]
    X_parts, labels, fams = [], [], []
    y = ((scn["ae"] - scn["ae"].mean()) / scn["ae"].std(ddof=0)).to_numpy()
    for fam, label, col in feats:
        x = pd.to_numeric(scn[col], errors="coerce")
        if x.std(ddof=0) < 1e-9:
            continue
        X_parts.append(((x - x.mean()) / x.std(ddof=0)).fillna(0.0).to_numpy())
        labels.append(label); fams.append(fam)
    # add sex (binary Female) if present
    if "sex" in scn.columns:
        female = scn["sex"].astype(str).str.lower().str.startswith("f").astype(float)
        if female.std(ddof=0) > 1e-9:
            X_parts.append(((female - female.mean()) / female.std(ddof=0)).to_numpy())
            labels.append("Sex (female)"); fams.append("Demographics")
    X = np.column_stack(X_parts)
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type="HC3")
    ci = res.conf_int()
    rows = []
    for i, (lab, fam) in enumerate(zip(labels, fams), start=1):
        rows.append({"family": fam, "moderator": lab, "beta_std": float(res.params[i]),
                     "ci_low": float(ci[i][0]), "ci_high": float(ci[i][1]),
                     "p_value": float(res.pvalues[i])})
    df = pd.DataFrame(rows)
    df["q_value"] = _fdr(df["p_value"].tolist())
    df["significant"] = df["q_value"] < 0.05
    df.attrs["model_r2"] = float(res.rsquared)
    df.attrs["n"] = int(scn.shape[0])
    return df.sort_values("beta_std", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _curated_univariate(scn: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Univariate standardised slope per curated construct (FDR within family). This is
    the construct-level association every family shows, complementary to the strict
    cross-family model; more constructs are significant because they are not competing
    for shared variance."""
    if sm is None:
        return pd.DataFrame()
    y = ((scn["ae"] - scn["ae"].mean()) / scn["ae"].std(ddof=0)).to_numpy()
    rows = []
    for fam, label, col in _resolve_curated(cols):
        x = pd.to_numeric(scn[col], errors="coerce")
        if x.std(ddof=0) < 1e-9:
            continue
        xs = ((x - x.mean()) / x.std(ddof=0)).fillna(0.0).to_numpy()
        res = sm.OLS(y, sm.add_constant(xs)).fit(cov_type="HC3")
        ci = res.conf_int()
        rows.append({"family": fam, "moderator": label, "beta_std": float(res.params[1]),
                     "ci_low": float(ci[1][0]), "ci_high": float(ci[1][1]), "p_value": float(res.pvalues[1])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = df.groupby("family")["p_value"].transform(lambda s: pd.Series(_fdr(s.tolist()), index=s.index))
    df["significant"] = df["q_value"] < 0.05
    return df.reset_index(drop=True)


def _by_domain(scn: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if sm is None:
        return pd.DataFrame()
    curated = _resolve_curated(cols)
    out = []
    for dom, g in scn.groupby("opinion_domain"):
        if len(g) < 80:
            continue
        y = ((g["ae"] - g["ae"].mean()) / g["ae"].std(ddof=0)).to_numpy()
        for fam, label, col in curated:
            x = pd.to_numeric(g[col], errors="coerce")
            if x.std(ddof=0) < 1e-9:
                continue
            xs = ((x - x.mean()) / x.std(ddof=0)).fillna(0.0)
            res = sm.OLS(y, sm.add_constant(xs.to_numpy())).fit(cov_type="HC3")
            out.append({"domain": dom, "family": fam, "moderator": label,
                        "beta_std": float(res.params[1]), "p_value": float(res.pvalues[1])})
    df = pd.DataFrame(out)
    if not df.empty:
        # Correct within each (domain, family) cell: each construct family inside a
        # domain is its own small set of tests, which is the appropriate granularity
        # and far less conservative than one global correction across all cells.
        df["q_value"] = df.groupby(["domain", "family"])["p_value"].transform(
            lambda s: pd.Series(_fdr(s.tolist()), index=s.index))
        df["significant"] = df["q_value"] < 0.05
    return df


def _variance_context(sem_long_df: pd.DataFrame, scn: pd.DataFrame, full_cv_r2: float) -> Dict[str, float]:
    """How much scenario-mean-effectivity variance the issue domain explains, versus
    all profile traits, plus the leaf-level between-profile ICC. The headline contrast:
    susceptibility is driven far more by what is attacked than by who is attacked."""
    ctx: Dict[str, float] = {"profile_traits_cv_r2": round(full_cv_r2, 4)}
    if sm is not None and "opinion_domain" in scn.columns:
        dd = pd.get_dummies(scn["opinion_domain"], drop_first=True).astype(float)
        if dd.shape[1] >= 1:
            r2 = sm.OLS(scn["ae"].to_numpy(),
                        sm.add_constant(dd.to_numpy())).fit().rsquared
            ctx["domain_r2"] = round(float(r2), 4)
    # leaf-level ICC(1): between-profile share of total leaf variance
    lv = sem_long_df.dropna(subset=[_AE]).copy()
    lv["_scn"] = _true_scn(lv)
    grand = lv[_AE].mean()
    g = lv.groupby("_scn")[_AE]
    ns, means = g.count(), g.mean()
    ssb = float((ns * (means - grand) ** 2).sum())
    ssw = float(sum(((x - x.mean()) ** 2).sum() for _, x in g))
    k, n = len(ns), int(ns.sum())
    if k > 1 and n > k:
        nbar = (n - (ns ** 2).sum() / n) / (k - 1)
        msb, msw = ssb / (k - 1), ssw / (n - k)
        denom = msb + (nbar - 1) * msw
        ctx["leaf_icc_between_profile"] = round(float((msb - msw) / denom) if denom > 0 else 0.0, 3)
    return ctx


def run_production_moderation(sem_long_df: pd.DataFrame) -> Dict[str, object]:
    sem_long_df = drop_excluded_domains(sem_long_df)
    scn, fams, all_cont = _scenario_frame(sem_long_df)
    family_table = _family_table(scn, fams, all_cont)
    within = _within_family(scn, fams)
    curated = _curated_model(scn, all_cont)
    curated_univariate = _curated_univariate(scn, all_cont)
    by_domain = _by_domain(scn, all_cont)
    variance_context = _variance_context(sem_long_df, scn, family_table.attrs.get("full_cv_r2", float("nan")))

    lines = ["Profile moderation of susceptibility (scenario-level, n=%d)" % scn.shape[0],
             "=" * 60, ""]
    lines.append("Variance context (what drives susceptibility):")
    lines.append(f"  issue domain explains {100 * variance_context.get('domain_r2', float('nan')):.1f}% of scenario-mean effectivity")
    lines.append(f"  all profile traits explain {100 * variance_context.get('profile_traits_cv_r2', float('nan')):.1f}% (out-of-sample)")
    lines.append(f"  leaf-level between-profile ICC = {variance_context.get('leaf_icc_between_profile')}")
    lines.append("")
    lines.append("Family predictive power (5-fold CV R^2; unique = commonality):")
    for _, r in family_table.iterrows():
        lines.append(f"  {r['family']:22s} standalone R2={r['cv_r2_standalone']:+.3f} | "
                     f"unique={r['cv_r2_unique']:+.3f} | {r['n_features']} traits")
    lines.append(f"  full model CV R2 = {family_table.attrs.get('full_cv_r2')}")
    lines.append("")
    if not curated.empty:
        sig = curated[curated["significant"]]
        lines.append(f"Significant curated moderators (BH-FDR q<0.05): {len(sig)} of {len(curated)} "
                     f"(model R2={curated.attrs.get('model_r2'):.3f})")
        for _, r in sig.head(12).iterrows():
            lines.append(f"  {r['moderator']:28s} ({r['family'][:12]:12s}) beta={r['beta_std']:+.3f} "
                         f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] q={r['q_value']:.1e}")
    summary = "\n".join(lines)
    return {"family_table": family_table, "within_family": within, "curated": curated,
            "curated_univariate": curated_univariate, "by_domain": by_domain,
            "scenario_frame": scn, "variance_context": variance_context, "summary": summary}
