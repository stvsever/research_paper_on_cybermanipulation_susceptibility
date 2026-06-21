from __future__ import annotations

"""
Inferential statistics for the INDIVIDUAL layer.

Every descriptive claim the figures and report make is backed here by an explicit
significance test with an effect size, chosen to respect the nested structure of
the data (opinion-leaf measurements are clustered within scenarios and profiles).
The unit of analysis is therefore the scenario-level mean effect for attack and
opinion-domain contrasts (one independent value per scenario), and the leaf-level
mean for the adversarial-direction contrast (one value per opinion leaf). Families
of pairwise comparisons are corrected with Benjamini-Hochberg FDR.

Entry point: run_individual_layer_statistics(sem_long_df) -> (results_df, summary_text).
"""

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as st

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

_AE = "adversarial_effectivity"
_TIER_ORDER = {"T1_atomic": 1, "T2_campaign": 2, "T3_synthetic": 3, "T4_orchestrated": 4}


def _pfmt(p: float) -> str:
    """Format a p-value for prose, guarding against float underflow to exactly 0."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p <= 0.0:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.2g}"


def _fdr(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    if len(p) == 0:
        return []
    if multipletests is not None:
        return list(multipletests(p, method="fdr_bh")[1])
    # Manual Benjamini-Hochberg fallback.
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return list(out)


def _rank_biserial(a: np.ndarray, b: np.ndarray, U: float) -> float:
    """Rank-biserial correlation effect size for Mann-Whitney U (a vs b)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(1.0 - (2.0 * U) / (n1 * n2))


def _cliffs_or_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _true_scenario_key(df: pd.DataFrame) -> pd.Series:
    """Recover the independent analysis unit.

    The analysis scenario is one (profile x attack x opinion-domain) cell. In sem_long
    the scenario_id is encoded as '<scenario>__<opinion_leaf>', so it is unique per leaf
    row; grouping on it directly would treat clustered leaves as independent observations
    and inflate every test. Stripping the leaf suffix recovers the true scenario. Falls
    back to scenario_id (or profile_id) when no '__' suffix is present.
    """
    if "scenario_id" in df.columns:
        sid = df["scenario_id"].astype(str)
        if sid.str.contains("__").any():
            return sid.str.split("__", n=1).str[0]
        return sid
    return df.get("profile_id", pd.Series(range(len(df)), index=df.index)).astype(str)


def _scenario_level(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["profile_id", "opinion_domain", "attack_execute_tactic",
                        "attack_plan_tactic", "attack_prepare_tactic", "attack_complexity_tier"] if c in df.columns]
    work = df.dropna(subset=[_AE]).copy()
    work["_scn"] = _true_scenario_key(work)
    g = work.groupby("_scn")
    agg = g.agg(ae=(_AE, "mean"), **{c: (c, "first") for c in keep})
    return agg.reset_index().rename(columns={"_scn": "scenario_id"})


def _omnibus_pairwise(scn: pd.DataFrame, group_col: str, analysis: str, min_n: int = 8) -> List[Dict]:
    """Kruskal-Wallis omnibus + BH-FDR-corrected pairwise Mann-Whitney over scenario-level means."""
    sub = scn.dropna(subset=[group_col, "ae"])
    groups = {k: v["ae"].to_numpy() for k, v in sub.groupby(sub[group_col].astype(str)) if len(v) >= min_n}
    if len(groups) < 2:
        return []
    rows: List[Dict] = []
    H, p = st.kruskal(*groups.values())
    k = len(groups)
    eps2 = float(max(0.0, (H - k + 1) / (len(sub) - k))) if len(sub) > k else float("nan")  # epsilon-squared effect size (>=0)
    rows.append({
        "analysis": analysis, "contrast": f"omnibus across {k} groups", "test": "Kruskal-Wallis",
        "unit": "scenario mean", "n": int(sum(len(v) for v in groups.values())),
        "statistic": round(float(H), 3), "p_value": float(p), "q_value": float(p),
        "effect_size": round(eps2, 4), "effect_size_name": "epsilon^2",
        "significant": bool(p < 0.05),
        "summary": f"{analysis}: groups differ (H={H:.1f}, p={_pfmt(p)}, eps2={eps2:.3f})" if p < 0.05
        else f"{analysis}: no significant group differences (p={_pfmt(p)})",
    })
    pairs = list(itertools.combinations(sorted(groups), 2))
    praw = []
    for a, b in pairs:
        U, pu = st.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        praw.append((a, b, U, pu))
    q = _fdr([x[3] for x in praw])
    for (a, b, U, pu), qv in zip(praw, q):
        rb = _rank_biserial(groups[a], groups[b], U)
        rows.append({
            "analysis": analysis, "contrast": f"{a} vs {b}", "test": "Mann-Whitney U (BH-FDR)",
            "unit": "scenario mean", "n": int(len(groups[a]) + len(groups[b])),
            "statistic": round(float(U), 1), "p_value": float(pu), "q_value": float(qv),
            "effect_size": round(rb, 3), "effect_size_name": "rank-biserial",
            "significant": bool(qv < 0.05),
            "summary": f"{a} vs {b}: {'differs' if qv < 0.05 else 'n.s.'} (q={_pfmt(qv)}, rb={rb:+.2f})",
        })
    return rows


def run_individual_layer_statistics(sem_long_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = sem_long_df.copy()
    if _AE not in df.columns:
        return pd.DataFrame(), "adversarial_effectivity not available; statistics skipped."
    scn = _scenario_level(df)
    rows: List[Dict] = []

    # H1: the attack moves opinions (mean effect > 0), tested on independent scenario means.
    ae = scn["ae"].dropna().to_numpy()
    if len(ae) >= 8:
        W, p = st.wilcoxon(ae, alternative="greater")
        d = float(ae.mean() / ae.std(ddof=1)) if ae.std(ddof=1) > 0 else 0.0
        rows.append({
            "analysis": "H1 attack effectiveness", "contrast": "mean effect > 0",
            "test": "Wilcoxon signed-rank (one-sided)", "unit": "scenario mean", "n": int(len(ae)),
            "statistic": round(float(W), 1), "p_value": float(p), "q_value": float(p),
            "effect_size": round(d, 3), "effect_size_name": "Cohen d_z",
            "significant": bool(p < 0.05),
            "summary": f"Attacks move opinions toward the goal (median {np.median(ae):+.1f}, "
                       f"{(ae > 0).mean() * 100:.0f}% of scenarios positive, p={_pfmt(p)}, d={d:.2f}).",
        })

    # H2: DISARM tactic effectiveness differs (Execute, Plan) and rises with complexity.
    for col, name in [("attack_execute_tactic", "H2 Execute-tactic effectiveness"),
                      ("attack_plan_tactic", "H2 Plan-tactic effectiveness")]:
        rows.extend(_omnibus_pairwise(scn, col, name))
    if "attack_complexity_tier" in scn.columns and scn["attack_complexity_tier"].notna().any():
        rows.extend(_omnibus_pairwise(scn, "attack_complexity_tier", "H2 complexity-tier effectiveness"))
        t = scn.dropna(subset=["attack_complexity_tier", "ae"]).copy()
        t["tier"] = t["attack_complexity_tier"].map(_TIER_ORDER)
        t = t.dropna(subset=["tier"])
        if t["tier"].nunique() >= 3:
            rho, pr = st.spearmanr(t["tier"], t["ae"])
            rows.append({
                "analysis": "H2 complexity-tier dose-response", "contrast": "monotonic trend with tier",
                "test": "Spearman rank correlation", "unit": "scenario mean", "n": int(len(t)),
                "statistic": round(float(rho), 3), "p_value": float(pr), "q_value": float(pr),
                "effect_size": round(float(rho), 3), "effect_size_name": "Spearman rho",
                "significant": bool(pr < 0.05),
                "summary": f"Effect {'rises' if rho > 0 else 'falls'} monotonically with operation complexity "
                           f"(rho={rho:+.2f}, p={_pfmt(pr)})." if pr < 0.05 else f"No complexity dose-response (p={_pfmt(pr)}).",
            })

    # H3: opinion domains differ in movability.
    rows.extend(_omnibus_pairwise(scn, "opinion_domain", "H3 opinion-domain susceptibility"))

    # H4: erode vs amplify leaves differ (leaf-level units).
    if "adversarial_direction" in df.columns and "opinion_leaf_label" in df.columns:
        leaf = df.dropna(subset=[_AE]).groupby("opinion_leaf_label").agg(
            ae=(_AE, "mean"), d=("adversarial_direction", "first")).dropna()
        erode = leaf.loc[leaf["d"] == -1, "ae"].to_numpy()
        ampl = leaf.loc[leaf["d"] == 1, "ae"].to_numpy()
        if len(erode) >= 5 and len(ampl) >= 5:
            U, pu = st.mannwhitneyu(erode, ampl, alternative="two-sided")
            rows.append({
                "analysis": "H4 adversarial direction", "contrast": "erode (d=-1) vs amplify (d=+1) leaves",
                "test": "Mann-Whitney U", "unit": "opinion leaf mean", "n": int(len(erode) + len(ampl)),
                "statistic": round(float(U), 1), "p_value": float(pu), "q_value": float(pu),
                "effect_size": round(_rank_biserial(erode, ampl, U), 3), "effect_size_name": "rank-biserial",
                "significant": bool(pu < 0.05),
                "summary": f"Erode and amplify leaves {'differ' if pu < 0.05 else 'do not differ'} in effectivity "
                           f"(p={_pfmt(pu)}).",
            })

    # H5: susceptibility is heterogeneous between profiles (ICC + label-permutation test).
    aedf = df.dropna(subset=[_AE])[["profile_id", _AE]].copy()
    pg = aedf.groupby("profile_id")[_AE]
    grand = float(aedf[_AE].mean())
    ns = pg.count()
    means = pg.mean()
    if len(ns) >= 5:
        ss_between = float(sum(n * (m - grand) ** 2 for n, m in zip(ns, means)))
        ss_within = float(sum((((grp - grp.mean()) ** 2).sum()) for _, grp in pg))
        k = len(ns)
        n_tot = int(ns.sum())
        nbar = (n_tot - (ns ** 2).sum() / n_tot) / (k - 1) if k > 1 else 1.0
        ms_b = ss_between / (k - 1) if k > 1 else 0.0
        ms_w = ss_within / (n_tot - k) if n_tot > k else 1.0
        icc = float((ms_b - ms_w) / (ms_b + (nbar - 1) * ms_w)) if (ms_b + (nbar - 1) * ms_w) > 0 else 0.0
        # Permutation null: shuffle profile labels, recompute SS_between.
        rng = np.random.default_rng(7)
        obs = ss_between
        vals = df.dropna(subset=[_AE])[_AE].to_numpy()
        sizes = ns.to_numpy()
        perm_ge = 0
        B = 1000
        for _ in range(B):
            sh = rng.permutation(vals)
            idx = np.cumsum(sizes)[:-1]
            parts = np.split(sh, idx)
            ssb = sum(len(p) * (p.mean() - grand) ** 2 for p in parts)
            perm_ge += int(ssb >= obs)
        p_perm = (perm_ge + 1) / (B + 1)
        rows.append({
            "analysis": "H5 inter-individual heterogeneity", "contrast": "between-profile variance > chance",
            "test": "label-permutation on SS_between (B=1000)", "unit": "profile", "n": int(k),
            "statistic": round(obs, 1), "p_value": float(p_perm), "q_value": float(p_perm),
            "effect_size": round(icc, 4), "effect_size_name": "ICC(1)",
            "significant": bool(p_perm < 0.05),
            "summary": f"Profiles differ in susceptibility beyond chance (ICC={icc:.3f}, permutation p={_pfmt(p_perm)}).",
        })

    results = pd.DataFrame(rows)
    # Human-readable summary of the headline (significant, omnibus / single tests).
    lines = ["Individual-layer inferential statistics", "=" * 39, ""]
    for _, r in results.iterrows():
        star = "  [sig]" if r["significant"] else ""
        lines.append(f"- {r['summary']}{star}")
    summary = "\n".join(lines)
    return results, summary
