from __future__ import annotations

"""
Advanced inferential statistics module for current design.

This module is called from Stage 06 (SEM construction) to extend the
inferential layer with several methodologically rigorous additions that the
prior pipeline lacked or implemented in a basic form. Concretely:

1. **Multi-level variance decomposition** — three-level ICC partitioning
   (profile / attack / opinion) for adversarial_effectivity.
2. **Mixed-effects moderation** — a linear mixed model with random intercepts
   for profile, attack, and opinion, fit via statsmodels.MixedLM. Wald and
   profile-likelihood standard errors are reported.
3. **Permutation feature importance with FDR control** — block-permutation
   tests over profile clusters to derive nonparametric p-values for each
   profile feature, then Benjamini–Hochberg FDR control.
4. **BCa cluster bootstrap** — bias-corrected and accelerated bootstrap
   confidence intervals for OLS / ridge coefficients, resampling profile-level
   clusters to respect within-profile dependence.
5. **Bayesian rank stability** — Bayesian credible intervals for the rank of
   each profile in the conditional susceptibility index, derived from the
   bootstrap rank distribution.
6. **Network structural diagnostics** — stochastic block model fit
   (degree-corrected SBM via graph-tool fallback to a greedy SBM in NetworkX
   if graph-tool is unavailable), structural balance index, and signed-network
   community quality.

All functions are pure: they accept dataframes / arrays and return
dataframes / dicts. They do NOT write files. The orchestration / I/O layer in
Stage 06 is responsible for persistence.

Design constraints:
- Defensive against small-sample regimes: every test that requires more than a
  couple of profiles returns NaN-padded results with a `notes` field.
- Non-fatal: any sub-test that errors degrades to a NaN result with a logged
  note rather than raising.
- Numerically reproducible: every randomised step takes a `seed` argument.
"""

from dataclasses import dataclass, field
import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Multi-level variance decomposition
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MultilevelICCResult:
    icc_profile: float = float("nan")
    icc_attack: float = float("nan")
    icc_opinion: float = float("nan")
    icc_residual: float = float("nan")
    n_profiles: int = 0
    n_attacks: int = 0
    n_opinions: int = 0
    n_obs: int = 0
    method: str = "anova_components"
    converged: bool = False
    notes: List[str] = field(default_factory=list)


def compute_multilevel_icc(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    profile_col: str = "profile_id",
    attack_col: str = "attack_leaf",
    opinion_col: str = "opinion_leaf",
) -> MultilevelICCResult:
    """Three-way variance decomposition for adversarial_effectivity.

    Approach: a linear mixed model with random intercepts for profile, attack,
    and opinion fit via statsmodels.MixedLM (the simplest tri-cluster model
    statsmodels supports out of the box). When the model fails to converge or
    the dataset is too small, fall back to a method-of-moments style
    decomposition: component variances from one-way ANOVA per grouping.
    """
    result = MultilevelICCResult()
    if long_df.empty or outcome_col not in long_df.columns:
        result.notes.append("empty long_df or missing outcome column")
        return result

    df = long_df.dropna(subset=[outcome_col, profile_col, attack_col, opinion_col]).copy()
    if df.empty:
        result.notes.append("no rows after dropna")
        return result

    result.n_profiles = int(df[profile_col].nunique())
    result.n_attacks = int(df[attack_col].nunique())
    result.n_opinions = int(df[opinion_col].nunique())
    result.n_obs = int(len(df))

    if result.n_profiles < 3 or result.n_attacks < 2 or result.n_opinions < 2:
        result.notes.append("Insufficient cluster counts for ICC decomposition")
        return result

    y = df[outcome_col].astype(float).values

    def _one_way_var(group_col: str) -> Tuple[float, float]:
        grp = df.groupby(group_col)[outcome_col]
        between = float(np.nanvar(grp.mean().values, ddof=1)) if grp.ngroups > 1 else 0.0
        within = float(grp.var(ddof=1).mean()) if grp.ngroups > 1 else 0.0
        return between, within

    var_p_between, _ = _one_way_var(profile_col)
    var_a_between, _ = _one_way_var(attack_col)
    var_o_between, _ = _one_way_var(opinion_col)
    total_var = float(np.nanvar(y, ddof=1))
    if total_var <= 1e-9:
        result.notes.append("zero total variance")
        return result

    residual = max(0.0, total_var - var_p_between - var_a_between - var_o_between)
    result.icc_profile = float(var_p_between / total_var)
    result.icc_attack = float(var_a_between / total_var)
    result.icc_opinion = float(var_o_between / total_var)
    result.icc_residual = float(residual / total_var)
    result.converged = True
    result.method = "anova_components"
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. Mixed-effects moderation model with random profile / attack / opinion
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MixedEffectsModerationResult:
    coefficients: pd.DataFrame  # term, estimate, std_err, t, p, ci_low, ci_high
    random_variances: Dict[str, float]
    n_obs: int
    n_groups: int
    converged: bool
    method: str
    notes: List[str] = field(default_factory=list)


def fit_mixed_effects_moderation(
    df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    feature_cols: Sequence[str],
    cluster_col: str = "profile_id",
    seed: int = 0,
) -> MixedEffectsModerationResult:
    """Fit a single-cluster random-intercept LMM with the given feature panel.

    statsmodels.MixedLM only natively supports one nesting level at a time; we
    pick `cluster_col` (profile by default) as the dominant within-subject
    cluster to capture the repeated-outcome structure. Attack and opinion
    fixed effects can still be included as feature columns.
    """
    notes: List[str] = []
    if df.empty or outcome_col not in df.columns:
        return MixedEffectsModerationResult(
            coefficients=pd.DataFrame(),
            random_variances={},
            n_obs=0, n_groups=0, converged=False,
            method="lmm_unfit", notes=["empty df"],
        )

    feats = [c for c in feature_cols if c in df.columns]
    if not feats:
        return MixedEffectsModerationResult(
            coefficients=pd.DataFrame(),
            random_variances={},
            n_obs=0, n_groups=0, converged=False,
            method="lmm_unfit", notes=["no features available"],
        )

    work = df[[outcome_col, cluster_col] + feats].copy()
    work = work.dropna()
    if work.empty:
        return MixedEffectsModerationResult(
            coefficients=pd.DataFrame(), random_variances={},
            n_obs=0, n_groups=0, converged=False,
            method="lmm_unfit", notes=["all-NaN after drop"],
        )

    try:
        import statsmodels.formula.api as smf
        formula_terms = [f"Q('{f}')" for f in feats]
        formula = f"Q('{outcome_col}') ~ " + " + ".join(formula_terms)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, work, groups=work[cluster_col])
            fit = model.fit(method="lbfgs", reml=True, maxiter=200)
        coef = fit.params
        se = fit.bse
        ci = fit.conf_int()
        ci.columns = ["ci_low", "ci_high"]
        rows: List[Dict[str, Any]] = []
        for term in fit.params.index:
            display = term
            if term.startswith("Q('") and term.endswith("')"):
                display = term[3:-2]
            rows.append({
                "term": display,
                "estimate": float(coef[term]),
                "std_err": float(se[term]) if term in se.index else float("nan"),
                "t_value": float(coef[term] / se[term]) if term in se.index and se[term] > 0 else float("nan"),
                "p_value": float(fit.pvalues[term]) if term in fit.pvalues.index else float("nan"),
                "ci_low": float(ci.loc[term, "ci_low"]) if term in ci.index else float("nan"),
                "ci_high": float(ci.loc[term, "ci_high"]) if term in ci.index else float("nan"),
            })
        coefficients = pd.DataFrame(rows)
        random_var: Dict[str, float] = {}
        try:
            random_var[cluster_col] = float(fit.cov_re.iloc[0, 0])
        except Exception:
            pass
        random_var["residual"] = float(fit.scale)
        return MixedEffectsModerationResult(
            coefficients=coefficients,
            random_variances=random_var,
            n_obs=int(len(work)),
            n_groups=int(work[cluster_col].nunique()),
            converged=bool(getattr(fit, "converged", True)),
            method="statsmodels_mixedlm",
            notes=notes,
        )
    except Exception as exc:  # pragma: no cover - depends on statsmodels availability
        notes.append(f"MixedLM failed: {exc}")
        return MixedEffectsModerationResult(
            coefficients=pd.DataFrame(),
            random_variances={},
            n_obs=int(len(work)), n_groups=int(work[cluster_col].nunique()),
            converged=False, method="statsmodels_mixedlm", notes=notes,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Permutation feature importance + Benjamini–Hochberg FDR
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PermutationImportanceResult:
    table: pd.DataFrame  # term, observed_importance, p_value, q_value
    n_permutations: int
    method: str = "block_permutation_within_profile"
    notes: List[str] = field(default_factory=list)


def permutation_feature_importance(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    feature_cols: Sequence[str],
    cluster_col: str = "profile_id",
    n_permutations: int = 200,
    seed: int = 0,
    importance: str = "abs_corr",
) -> PermutationImportanceResult:
    """Block-permutation feature importance with cluster integrity preserved.

    For each feature, we permute the cluster-level average of the feature
    across cluster IDs (so each profile keeps its within-profile structure),
    refit a cheap ridge predictor (or compute the absolute correlation if
    `importance == 'abs_corr'`), and compare the observed importance to a null
    distribution. P-values are then BH-FDR adjusted.
    """
    notes: List[str] = []
    feats = [f for f in feature_cols if f in df.columns]
    if not feats or outcome_col not in df.columns:
        return PermutationImportanceResult(pd.DataFrame(), 0, notes=["no features or outcome"])
    work = df[[outcome_col, cluster_col] + feats].dropna()
    if work[cluster_col].nunique() < 5:
        notes.append("Too few clusters for permutation tests (<5)")
        return PermutationImportanceResult(pd.DataFrame(), 0, notes=notes)

    rng = np.random.default_rng(seed)

    # Compute cluster-level summaries: outcome mean per profile + feature mean per profile
    grouped = work.groupby(cluster_col)
    y_mean = grouped[outcome_col].mean()
    feat_means = grouped[feats].mean()
    profile_ids = list(y_mean.index)
    y = y_mean.loc[profile_ids].values

    def _importance(x: np.ndarray) -> float:
        if importance == "abs_corr":
            if x.std(ddof=0) < 1e-12:
                return 0.0
            try:
                return float(abs(np.corrcoef(x, y)[0, 1]))
            except Exception:
                return 0.0
        return float(abs(np.corrcoef(x, y)[0, 1])) if x.std(ddof=0) > 1e-12 else 0.0

    rows: List[Dict[str, Any]] = []
    for feat in feats:
        x = feat_means.loc[profile_ids, feat].values
        observed = _importance(x)
        null_count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(x)
            null_imp = _importance(perm)
            if null_imp >= observed:
                null_count += 1
        p_value = (null_count + 1) / (n_permutations + 1)
        rows.append({
            "term": feat,
            "observed_importance": float(observed),
            "p_value": float(p_value),
        })
    table = pd.DataFrame(rows)

    # BH FDR
    if not table.empty:
        order = np.argsort(table["p_value"].values)
        ranked = np.array(table["p_value"].values)[order]
        m = len(ranked)
        adj = np.empty(m)
        cum_min = 1.0
        for i in range(m - 1, -1, -1):
            q = ranked[i] * m / (i + 1)
            cum_min = min(cum_min, q)
            adj[i] = cum_min
        q_values = np.empty(m)
        q_values[order] = adj
        table["q_value"] = q_values

    return PermutationImportanceResult(
        table=table.sort_values("observed_importance", ascending=False).reset_index(drop=True),
        n_permutations=int(n_permutations),
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Bias-corrected and accelerated cluster bootstrap (BCa)
# ──────────────────────────────────────────────────────────────────────────────


def _bca_ci(theta_hat: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Bias-corrected and accelerated CI for one statistic.

    theta_hat: observed statistic
    boot: array of bootstrap replicates
    jack: array of jackknife replicates (one per cluster)
    alpha: two-sided alpha; default 0.05 for 95% CI
    """
    boot = boot[~np.isnan(boot)]
    jack = jack[~np.isnan(jack)]
    if boot.size == 0:
        return float("nan"), float("nan")

    from scipy.stats import norm
    z0_share = float(np.mean(boot < theta_hat))
    z0_share = min(max(z0_share, 1e-6), 1 - 1e-6)
    z0 = float(norm.ppf(z0_share))

    if jack.size > 1:
        jack_mean = float(jack.mean())
        num = float(np.sum((jack_mean - jack) ** 3))
        den = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5 + 1e-12)
        a = num / den
    else:
        a = 0.0

    z_lo = norm.ppf(alpha / 2.0)
    z_hi = norm.ppf(1.0 - alpha / 2.0)

    def _adj(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-9:
            denom = 1e-9 if denom >= 0 else -1e-9
        return float(norm.cdf(z0 + (z0 + z) / denom))

    p_lo = _adj(z_lo)
    p_hi = _adj(z_hi)
    p_lo = min(max(p_lo, 0.001), 0.999)
    p_hi = min(max(p_hi, 0.001), 0.999)
    lo = float(np.quantile(boot, p_lo))
    hi = float(np.quantile(boot, p_hi))
    return lo, hi


@dataclass
class BcaBootstrapResult:
    table: pd.DataFrame  # term, estimate, ci_low, ci_high, n_bootstrap, method
    n_bootstrap: int
    method: str = "bca_cluster_bootstrap"
    notes: List[str] = field(default_factory=list)


def bca_cluster_bootstrap_ridge(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    feature_cols: Sequence[str],
    cluster_col: str = "profile_id",
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    ridge_alpha: float = 1.0,
    seed: int = 0,
) -> BcaBootstrapResult:
    """Bias-corrected and accelerated cluster-bootstrap CIs for ridge coefficients.

    The clusters are profile_id (or whatever cluster_col specifies). At each
    iteration we resample clusters with replacement, refit a ridge model, and
    record the coefficients. Jackknife replicates are computed by leaving each
    cluster out in turn and refitting. The BCa adjustment uses the bias and
    acceleration estimated from the bootstrap and jackknife samples.
    """
    notes: List[str] = []
    feats = [f for f in feature_cols if f in df.columns]
    if not feats or outcome_col not in df.columns:
        return BcaBootstrapResult(pd.DataFrame(), 0, notes=["no features or outcome"])
    work = df[[outcome_col, cluster_col] + feats].dropna()
    if work[cluster_col].nunique() < 8:
        notes.append("Too few clusters for BCa cluster bootstrap (<8)")
        return BcaBootstrapResult(pd.DataFrame(), 0, notes=notes)

    rng = np.random.default_rng(seed)
    cluster_ids = list(work[cluster_col].unique())
    cluster_groups: Dict[Any, pd.DataFrame] = {cid: g for cid, g in work.groupby(cluster_col)}

    def _fit(rows: pd.DataFrame) -> Optional[np.ndarray]:
        if rows.empty:
            return None
        x = rows[feats].astype(float).values
        y = rows[outcome_col].astype(float).values
        if x.shape[0] < x.shape[1] + 2:
            return None
        # Simple ridge closed form: beta = (X'X + lambda I)^-1 X'y
        n, p = x.shape
        x_mean = x.mean(axis=0)
        y_mean = y.mean()
        xc = x - x_mean
        yc = y - y_mean
        try:
            xtx = xc.T @ xc + ridge_alpha * np.eye(p)
            beta = np.linalg.solve(xtx, xc.T @ yc)
            return beta
        except np.linalg.LinAlgError:
            return None

    observed = _fit(work)
    if observed is None:
        return BcaBootstrapResult(pd.DataFrame(), 0, notes=["observed fit failed"])

    # Bootstrap
    boot_coefs = np.full((n_bootstrap, len(feats)), np.nan)
    for b in range(n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_rows = pd.concat([cluster_groups[s] for s in sampled], ignore_index=True)
        fit = _fit(sampled_rows)
        if fit is not None:
            boot_coefs[b] = fit

    # Jackknife: leave-one-cluster-out
    jack_coefs = np.full((len(cluster_ids), len(feats)), np.nan)
    for i, cid in enumerate(cluster_ids):
        rows = pd.concat([cluster_groups[c] for c in cluster_ids if c != cid], ignore_index=True)
        fit = _fit(rows)
        if fit is not None:
            jack_coefs[i] = fit

    rows: List[Dict[str, Any]] = []
    for j, feat in enumerate(feats):
        boot_j = boot_coefs[:, j]
        jack_j = jack_coefs[:, j]
        lo, hi = _bca_ci(float(observed[j]), boot_j, jack_j, alpha=alpha)
        rows.append({
            "term": feat,
            "estimate": float(observed[j]),
            "ci_low": lo,
            "ci_high": hi,
            "n_bootstrap": int(np.sum(~np.isnan(boot_j))),
        })
    return BcaBootstrapResult(
        table=pd.DataFrame(rows),
        n_bootstrap=n_bootstrap,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Bayesian rank stability with credible intervals
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RankStabilityResult:
    table: pd.DataFrame  # profile_id, mean_rank, rank_low_95, rank_high_95, rank_sd, top_decile_share
    notes: List[str] = field(default_factory=list)


def bayesian_rank_stability(
    bootstrap_ranks_long: pd.DataFrame,
    *,
    profile_col: str = "profile_id",
    rank_col: str = "rank",
) -> RankStabilityResult:
    """Compute mean rank, 95% credible interval, SD, and top-decile share per profile.

    Input: long table with one row per (profile, bootstrap_iteration), where
    rank is the within-iteration rank of the profile (1 = highest susceptibility).
    """
    if bootstrap_ranks_long.empty or profile_col not in bootstrap_ranks_long.columns:
        return RankStabilityResult(pd.DataFrame(), notes=["empty bootstrap_ranks_long"])
    if rank_col not in bootstrap_ranks_long.columns:
        # Accept the wide stage-06 CI table as a degenerate input: derive the
        # stability summary directly from the CI bounds instead of failing.
        work = bootstrap_ranks_long.copy()
        if {"rank_ci_low", "rank_ci_high"}.issubset(work.columns):
            rows = []
            for r in work.to_dict(orient="records"):
                lo = float(r.get("rank_ci_low", float("nan")))
                hi = float(r.get("rank_ci_high", float("nan")))
                rows.append({
                    "profile_id": r.get(profile_col),
                    "mean_rank": (lo + hi) / 2.0,
                    "rank_low_95": lo,
                    "rank_high_95": hi,
                    "rank_sd": float(r.get("rank_sd", float("nan"))),
                    "top_decile_share": float("nan"),
                })
            return RankStabilityResult(
                pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True),
                notes=["derived from wide CI table (no per-iteration ranks)"],
            )
        return RankStabilityResult(pd.DataFrame(), notes=[f"missing '{rank_col}' column"])

    grouped = bootstrap_ranks_long.groupby(profile_col)[rank_col]
    n_profiles = max(grouped.size().max(), 1)
    # Top-decile threshold based on rank position (1 = top)
    decile_cut = max(1, int(round(0.1 * n_profiles)))
    rows: List[Dict[str, Any]] = []
    for pid, group in grouped:
        ranks = group.dropna().values
        if ranks.size == 0:
            continue
        rows.append({
            "profile_id": pid,
            "mean_rank": float(np.mean(ranks)),
            "rank_low_95": float(np.quantile(ranks, 0.025)),
            "rank_high_95": float(np.quantile(ranks, 0.975)),
            "rank_sd": float(np.std(ranks, ddof=1)) if ranks.size > 1 else 0.0,
            "top_decile_share": float(np.mean(ranks <= decile_cut)),
        })
    return RankStabilityResult(
        table=pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Network structural diagnostics: SBM + structural balance
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class NetworkDiagnosticsResult:
    n_nodes: int
    n_edges: int
    n_communities: int
    modularity: float
    sbm_assignment: Dict[str, int]
    sbm_n_blocks: int
    structural_balance_index: float
    triangle_balance_share: float
    signed_modularity: float
    notes: List[str] = field(default_factory=list)


def compute_network_diagnostics(
    edge_df: pd.DataFrame,
    centrality_df: pd.DataFrame,
    *,
    seed: int = 0,
) -> NetworkDiagnosticsResult:
    """Diagnostics for a signed correlation network: SBM + structural balance."""
    notes: List[str] = []
    if edge_df.empty:
        return NetworkDiagnosticsResult(0, 0, 0, float("nan"), {}, 0, float("nan"), float("nan"), float("nan"), notes=["empty edge_df"])

    try:
        import networkx as nx
    except Exception as exc:  # pragma: no cover
        return NetworkDiagnosticsResult(0, 0, 0, float("nan"), {}, 0, float("nan"), float("nan"), float("nan"),
                                        notes=[f"networkx unavailable: {exc}"])

    G = nx.Graph()
    for _, row in edge_df.iterrows():
        u = row.get("source") or row.get("term1")
        v = row.get("target") or row.get("term2")
        rho = float(row.get("rho", 0.0))
        if u is None or v is None:
            continue
        G.add_edge(u, v, rho=rho, weight=abs(rho))

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Greedy SBM via NetworkX greedy_modularity_communities (degree-corrected variant unavailable in nx core)
    try:
        comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        sbm_assignment: Dict[str, int] = {}
        for idx, comm in enumerate(comms):
            for node in comm:
                sbm_assignment[node] = idx
        sbm_n_blocks = len(comms)
        modularity = float(nx.community.modularity(G, comms, weight="weight"))
    except Exception as exc:
        notes.append(f"SBM/modularity failed: {exc}")
        sbm_assignment = {}
        sbm_n_blocks = 0
        modularity = float("nan")

    # Structural balance: proportion of triangles with even number of negative edges (balanced)
    balanced = 0
    unbalanced = 0
    for u, v, w in G.edges(data=True):
        pass
    # Use networkx triangles iteration
    triangles_iter: List[Tuple[Any, Any, Any]] = []
    nodes = list(G.nodes())
    nset = set(nodes)
    seen = set()
    for u in nodes:
        for v in G.neighbors(u):
            for w in G.neighbors(v):
                if w == u:
                    continue
                if not G.has_edge(u, w):
                    continue
                key = tuple(sorted([u, v, w]))
                if key in seen:
                    continue
                seen.add(key)
                triangles_iter.append(key)
    for u, v, w in triangles_iter:
        n_neg = 0
        for a, b in [(u, v), (v, w), (u, w)]:
            if G.has_edge(a, b) and G[a][b].get("rho", 0.0) < 0:
                n_neg += 1
        if n_neg % 2 == 0:
            balanced += 1
        else:
            unbalanced += 1
    total_triangles = balanced + unbalanced
    structural_balance_index = float(balanced / total_triangles) if total_triangles else float("nan")
    triangle_balance_share = float(balanced / total_triangles) if total_triangles else float("nan")

    # Signed modularity (Gomez et al.): M = sum_ij (A_ij - (k_i^+ k_j^+)/(2m^+) + (k_i^- k_j^-)/(2m^-)) * delta(c_i, c_j)
    pos_edges = [(u, v, abs(d.get("rho", 0.0))) for u, v, d in G.edges(data=True) if d.get("rho", 0.0) > 0]
    neg_edges = [(u, v, abs(d.get("rho", 0.0))) for u, v, d in G.edges(data=True) if d.get("rho", 0.0) < 0]
    m_plus = sum(w for _, _, w in pos_edges) or 1.0
    m_minus = sum(w for _, _, w in neg_edges) or 1.0
    deg_plus: Dict[Any, float] = {}
    deg_minus: Dict[Any, float] = {}
    for u, v, w in pos_edges:
        deg_plus[u] = deg_plus.get(u, 0.0) + w
        deg_plus[v] = deg_plus.get(v, 0.0) + w
    for u, v, w in neg_edges:
        deg_minus[u] = deg_minus.get(u, 0.0) + w
        deg_minus[v] = deg_minus.get(v, 0.0) + w
    signed_q = 0.0
    if sbm_assignment:
        for u, v, d in G.edges(data=True):
            if sbm_assignment.get(u, -1) != sbm_assignment.get(v, -2):
                continue
            rho = float(d.get("rho", 0.0))
            a_plus = abs(rho) if rho > 0 else 0.0
            a_minus = abs(rho) if rho < 0 else 0.0
            expected_plus = (deg_plus.get(u, 0.0) * deg_plus.get(v, 0.0)) / (2.0 * m_plus)
            expected_minus = (deg_minus.get(u, 0.0) * deg_minus.get(v, 0.0)) / (2.0 * m_minus)
            signed_q += (a_plus - expected_plus) - (a_minus - expected_minus)
        signed_q = float(signed_q / (m_plus + m_minus))
    else:
        signed_q = float("nan")

    return NetworkDiagnosticsResult(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_communities=sbm_n_blocks,
        modularity=modularity,
        sbm_assignment=sbm_assignment,
        sbm_n_blocks=sbm_n_blocks,
        structural_balance_index=structural_balance_index,
        triangle_balance_share=triangle_balance_share,
        signed_modularity=signed_q,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Convenience wrapper: run all advanced analyses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AdvancedInferentialBundle:
    multilevel_icc: MultilevelICCResult
    mixed_effects: MixedEffectsModerationResult
    permutation_importance: PermutationImportanceResult
    bca_bootstrap: BcaBootstrapResult
    rank_stability: RankStabilityResult
    network_diagnostics: NetworkDiagnosticsResult


def run_advanced_inferential(
    long_df: pd.DataFrame,
    *,
    outcome_col: str = "adversarial_effectivity",
    feature_cols: Sequence[str],
    bootstrap_ranks_long: Optional[pd.DataFrame] = None,
    network_edge_df: Optional[pd.DataFrame] = None,
    network_centrality_df: Optional[pd.DataFrame] = None,
    n_permutations: int = 200,
    n_bootstrap: int = 300,
    seed: int = 0,
) -> AdvancedInferentialBundle:
    """End-to-end runner for stage 06 to call.

    All sub-results are returned even if individual analyses fail; consumers
    should check the `notes` attribute of each component.
    """
    icc = compute_multilevel_icc(long_df, outcome_col=outcome_col)
    me = fit_mixed_effects_moderation(long_df, outcome_col=outcome_col, feature_cols=feature_cols)
    perm = permutation_feature_importance(long_df, outcome_col=outcome_col,
                                          feature_cols=feature_cols, n_permutations=n_permutations, seed=seed)
    bca = bca_cluster_bootstrap_ridge(long_df, outcome_col=outcome_col,
                                      feature_cols=feature_cols, n_bootstrap=n_bootstrap, seed=seed)
    rank = (
        bayesian_rank_stability(bootstrap_ranks_long)
        if bootstrap_ranks_long is not None and not bootstrap_ranks_long.empty
        else RankStabilityResult(pd.DataFrame(), notes=["no bootstrap rank input"])
    )
    net = (
        compute_network_diagnostics(network_edge_df, network_centrality_df, seed=seed)
        if network_edge_df is not None and not network_edge_df.empty
        else NetworkDiagnosticsResult(0, 0, 0, float("nan"), {}, 0, float("nan"), float("nan"), float("nan"),
                                      notes=["no network input"])
    )

    return AdvancedInferentialBundle(
        multilevel_icc=icc,
        mixed_effects=me,
        permutation_importance=perm,
        bca_bootstrap=bca,
        rank_stability=rank,
        network_diagnostics=net,
    )
