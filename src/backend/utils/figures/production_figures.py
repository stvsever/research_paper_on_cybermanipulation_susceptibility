from __future__ import annotations

"""
Production run 1 paper figures (PNG). Design rules:
  - no main figure titles (left-aligned panel sub-titles only).
  - full-resolution distributions: half-violin KDE + raw jittered scatter + a median
    marker with a scenario-clustered bootstrap CI (medians are the primary summary).
  - group differences carry significance brackets (ns / * / ** / ***) placed compactly
    on the value axis, from rank tests on the independent scenario means.
  - perceptually-uniform colour maps (mako / viridis / rocket), no harsh reds.
  - the single-leaf outlier domain "Macroeconomic And Fiscal Policy" is excluded.
"""

import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats as st
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

from src.backend.utils.analysis.production_moderation import drop_excluded_domains

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#3a3f4a",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#eaedf2", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 9.5,
    "savefig.dpi": 200, "figure.dpi": 120,
})
_DPI = 200
_AE = "adversarial_effectivity"
_PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#64B5CD"]
_FAM_COLOR = {
    "Big Five": "#4C72B0", "Political Psychology": "#C44E52", "Ideology": "#55A868",
    "Moral Foundations": "#8172B3", "Demographics": "#DD8452", "Other": "#8C8C8C",
}
_MAKO = sns.color_palette("mako", as_cmap=True)
_ROCKET = sns.color_palette("rocket_r", as_cmap=True)


def _scn(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.str.split("__", n=1).str[0] if s.str.contains("__").any() else s


def _short_dom(s: str) -> str:
    return str(s).split(" > ")[-1].replace("_", " ").strip()


def _short_leaf(s: str) -> str:
    return str(s).split(" > ")[-1].replace("_", " ").strip()


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _med_ci(vals: np.ndarray, clusters: np.ndarray, n: int = 500, seed: int = 0):
    s = pd.Series(np.asarray(vals, float)).groupby(np.asarray(clusters)).mean().to_numpy()
    if len(s) < 3:
        m = float(np.median(s)) if len(s) else 0.0
        return m, m, m
    rng = np.random.default_rng(seed)
    boots = [np.median(rng.choice(s, size=len(s), replace=True)) for _ in range(n)]
    return float(np.median(s)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "ns"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def _raincloud_v(ax, arrays, clusters, colors, seed=0, width=0.34, point_alpha=0.16):
    """Vertical raincloud: categories on x, value on y. Half-violin to the right, raw
    jittered scatter to the left, median marker with scenario-clustered CI."""
    rng = np.random.default_rng(seed)
    meds = []
    for i, (v, clu, col) in enumerate(zip(arrays, clusters, colors)):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v) == 0:
            meds.append(np.nan); continue
        if len(v) > 8 and v.std() > 0:
            kde = gaussian_kde(v)
            ys = np.linspace(np.percentile(v, 0.5), np.percentile(v, 99.5), 160)
            dens = kde(ys); dens = dens / dens.max() * width
            ax.fill_betweenx(ys, i + 0.06, i + 0.06 + dens, color=col, alpha=0.38, linewidth=0)
        jit = i - 0.08 - rng.uniform(0, 0.28, size=len(v))
        idx = rng.choice(len(v), size=min(len(v), 1200), replace=False)
        ax.scatter(jit[idx], v[idx], s=4.5, color=col, alpha=point_alpha, linewidth=0, zorder=2)
        med, lo, hi = _med_ci(v, np.asarray(clu), seed=seed + i)
        ax.errorbar(i - 0.01, med, yerr=[[med - lo], [hi - med]], fmt="o", color="#11151c",
                    ecolor="#11151c", elinewidth=1.7, capsize=3, markersize=5.5, zorder=4)
        meds.append(med)
    return meds


def _brackets_top(ax, pairs_q: List[Tuple[int, int, float]], y0: float, dy: float):
    """Compact significance brackets above the data (only the given pairs), stacked."""
    level = y0
    for i, j, q in pairs_q:
        star = _stars(q)
        lo, hi = sorted((i, j))
        ax.plot([lo, lo, hi, hi], [level, level + dy * 0.28, level + dy * 0.28, level],
                color="#3a3f4a", lw=1.1, clip_on=False)
        ax.text((lo + hi) / 2, level + dy * 0.30, star, ha="center", va="bottom",
                fontsize=10 if star != "ns" else 8,
                fontweight="bold" if star != "ns" else "normal", color="#11151c", clip_on=False)
        level += dy


def _scn_level(sem: pd.DataFrame) -> pd.DataFrame:
    w = sem.dropna(subset=[_AE]).copy(); w["_scn"] = _scn(w["scenario_id"])
    keep = [c for c in ["opinion_domain", "attack_execute_tactic", "attack_plan_tactic",
                        "attack_prepare_tactic", "attack_complexity_tier"] if c in w.columns]
    agg = {"ae": (_AE, "mean")}
    for c in keep:
        agg[c] = (c, "first")
    return w.groupby("_scn").agg(**agg).reset_index()


def _adjacent_q(groups: Dict[str, np.ndarray], order: List[str]):
    """Adjacent-rank pairwise Mann-Whitney with BH-FDR; returns Kruskal H,p and the
    list of (i,j,q) for adjacent pairs in display order."""
    from statsmodels.stats.multitest import multipletests
    keys = [k for k in order if k in groups]
    if len(keys) < 2:
        return np.nan, np.nan, []
    H, p = st.kruskal(*[groups[k] for k in keys])
    praw = [st.mannwhitneyu(groups[keys[i]], groups[keys[i + 1]], alternative="two-sided")[1]
            for i in range(len(keys) - 1)]
    q = list(multipletests(praw, method="fdr_bh")[1]) if praw else []
    return H, p, [(i, i + 1, q[i]) for i in range(len(keys) - 1)]


# --------------------------------------------------------------------------- #
def _overview(sem: pd.DataFrame, mod: Dict, out: Path) -> Optional[Path]:
    ae = pd.to_numeric(sem[_AE], errors="coerce").dropna()
    w = sem.dropna(subset=[_AE]).copy(); w["_scn"] = _scn(w["scenario_id"])
    scn_mean = w.groupby("_scn")[_AE].mean()
    ctx = mod.get("variance_context", {})
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.8))

    a = ax[0, 0]
    hi, lo = float(np.percentile(ae, 99)), float(min(np.percentile(ae, 1), 0))
    a.hist(ae, bins=48, range=(lo, hi), color="#cdd9ef", edgecolor="white", linewidth=0.3, density=True)
    xs = np.linspace(lo, hi, 200); a.plot(xs, gaussian_kde(ae.clip(lo, hi))(xs), color="#3b5ba8", lw=1.8)
    a.axvline(float(ae.median()), color="#1b7f5e", lw=1.8, label=f"median {ae.median():+.0f}")
    a.axvline(float(ae.mean()), color="#c44e52", lw=1.6, ls="--", label=f"mean {ae.mean():+.1f}")
    a.axvline(0, color="#1F2430", lw=0.9); a.set_xlim(lo, hi); a.set_yticks([])
    a.set_title(f"(a) Per-leaf effectivity  |  {(ae > 0).mean()*100:.0f}% toward goal  |  scenario d_z = 1.23")
    a.set_xlabel("Adversarial effectivity"); a.legend(frameon=False, fontsize=8)

    b = ax[0, 1]
    dom_r2 = 100 * ctx.get("domain_r2", float("nan")); trait_r2 = 100 * max(0.0, ctx.get("profile_traits_cv_r2", 0.0))
    bars = b.barh(["What is attacked\n(issue domain)", "Who is attacked\n(159 profile traits)"],
                  [dom_r2, trait_r2], color=["#1b7f5e", "#9aa1ab"], edgecolor="#1F2430", linewidth=0.5)
    for r, v in zip(bars, [dom_r2, trait_r2]):
        b.text(r.get_width() + max(dom_r2, 1) * 0.02, r.get_y() + r.get_height() / 2, f"{v:.1f}%",
               va="center", fontsize=10, fontweight="bold")
    b.set_xlim(0, max(dom_r2, trait_r2) * 1.3 + 0.5)
    b.set_title(f"(b) The target drives susceptibility, not the trait  (ICC = {ctx.get('leaf_icc_between_profile')})")
    b.set_xlabel("Between-scenario variance of effectivity explained (%)")

    c = ax[1, 0]
    s = np.sort(scn_mean.to_numpy()); med = float(np.median(s)); xx = np.arange(len(s))
    c.fill_between(xx, s, med, where=(s >= med), color="#e76f51", alpha=0.3, linewidth=0)
    c.fill_between(xx, s, med, where=(s < med), color="#457b9d", alpha=0.3, linewidth=0)
    c.plot(xx, s, color="#1d3557", lw=1.3)
    c.axhline(med, color="#c44e52", ls="--", lw=1.2, label=f"median {med:.0f}")
    c.set_yscale("symlog", linthresh=10)
    c.set_title(f"(c) Inter-individual heterogeneity  (IQR {np.percentile(s,25):.0f} to {np.percentile(s,75):.0f}, symlog y)")
    c.set_xlabel("Synthetic individuals ranked by mean susceptibility")
    c.set_ylabel("Mean effectivity (symlog)"); c.legend(frameon=False, fontsize=8)

    d = ax[1, 1]
    dd = w.dropna(subset=["baseline_score", "post_score"])
    hb = d.hexbin(dd["baseline_score"], dd["post_score"], gridsize=44, bins="log", cmap=_MAKO, mincnt=1)
    lim = [dd["baseline_score"].min(), dd["baseline_score"].max()]
    d.plot(lim, lim, color="#e9c46a", lw=1.3, ls="--", label="no change (P = B)")
    fig.colorbar(hb, ax=d, fraction=0.046, pad=0.02).set_label("leaf measurements (log)", fontsize=8)
    d.set_title("(d) Baseline to post-attack movement"); d.set_xlabel("Baseline opinion B"); d.set_ylabel("Post-attack P")
    d.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, out / "susceptibility_overview.png")


# --------------------------------------------------------------------------- #
def _domain_susceptibility(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    w = sem.dropna(subset=[_AE]).copy(); w["_scn"] = _scn(w["scenario_id"]); w["dom"] = w["opinion_domain"]
    order = w.groupby("dom")[_AE].median().sort_values().index.tolist()
    if len(order) < 2:
        return None
    arrays = [w.loc[w["dom"] == d, _AE].to_numpy() for d in order]
    clusters = [w.loc[w["dom"] == d, "_scn"].to_numpy() for d in order]
    colors = _MAKO(np.linspace(0.25, 0.85, len(order)))
    scn = _scn_level(w)
    groups = {d: scn.loc[scn["opinion_domain"] == d, "ae"].to_numpy() for d in order}
    H, p, adj = _adjacent_q(groups, order)

    fig, ax = plt.subplots(figsize=(2.0 * len(order) + 1.5, 7.6))
    _raincloud_v(ax, arrays, clusters, colors, seed=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["\n".join(textwrap.wrap(_short_dom(d), 16)) for d in order], fontsize=8.5)
    ax.axhline(0, color="#1F2430", lw=0.9)
    bulk = float(np.percentile(w[_AE], 98)); base = float(min(np.percentile(w[_AE], 1), 0))
    dy = bulk * 0.085
    _brackets_top(ax, adj, y0=bulk * 1.02, dy=dy)
    ax.set_ylim(base - 2, bulk * 1.02 + dy * (len(adj) + 1))
    eps2 = max(0.0, (H - len(order) + 1) / (len(scn) - len(order)))
    ax.set_title(f"Adversarial effectivity by issue domain (raw points, half-violin density, scenario-clustered median 95% CI)\n"
                 f"Kruskal-Wallis H={H:.0f}, p={p:.1e}, epsilon^2={eps2:.3f}; brackets = adjacent-domain Mann-Whitney BH-FDR "
                 f"(* q<.05  ** q<.01  *** q<.001)", fontsize=10)
    ax.set_ylabel("Adversarial effectivity")
    fig.tight_layout()
    return _save(fig, out / "domain_susceptibility.png")


# --------------------------------------------------------------------------- #
def _attack_phase_diagnostics(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    levels = [("attack_plan_tactic", "DISARM Plan tactic"),
              ("attack_prepare_tactic", "DISARM Prepare tactic"),
              ("attack_execute_tactic", "DISARM Execute tactic"),
              ("attack_complexity_tier", "Operation complexity tier")]
    levels = [(c, t) for c, t in levels if c in sem.columns and sem[c].notna().sum() > 50 and sem[c].nunique() >= 2]
    if not levels:
        return None
    w = sem.dropna(subset=[_AE]).copy(); w["_scn"] = _scn(w["scenario_id"]); scn = _scn_level(w)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), squeeze=False)
    for k, (col, title) in enumerate(levels):
        ax = axes[k // 2][k % 2]; sub = w.dropna(subset=[col])
        order = sub.groupby(col)[_AE].median().sort_values().index.tolist()
        order = [o for o in order if (sub[col] == o).sum() >= 30]
        if len(order) < 2:
            ax.set_axis_off(); continue
        arrays = [sub.loc[sub[col] == o, _AE].to_numpy() for o in order]
        clusters = [sub.loc[sub[col] == o, "_scn"].to_numpy() for o in order]
        cols = [_PAL[i % len(_PAL)] for i in range(len(order))]
        _raincloud_v(ax, arrays, clusters, cols, seed=k, width=0.34)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["\n".join(textwrap.wrap(str(o).replace("_", " "), 14)) for o in order], fontsize=8)
        ax.axhline(0, color="#1F2430", lw=0.8)
        groups = {o: scn.loc[scn[col] == o, "ae"].to_numpy() for o in order}
        groups = {o: v for o, v in groups.items() if len(v) >= 8}
        bulk = float(np.percentile(sub[_AE], 97)); base = float(min(np.percentile(sub[_AE], 1), 0))
        H, p, adj = _adjacent_q(groups, list(groups))
        dy = bulk * 0.12
        _brackets_top(ax, adj, y0=bulk * 1.02, dy=dy)
        ax.set_ylim(base - 2, bulk * 1.02 + dy * (len(adj) + 1.2))
        ax.set_title(f"{title}   (Kruskal p = {p:.2g})", fontsize=10)
        if k % 2 == 0:
            ax.set_ylabel("Adversarial effectivity")
    fig.tight_layout()
    return _save(fig, out / "attack_phase_diagnostics.png")


# --------------------------------------------------------------------------- #
def _profile_moderation(mod: Dict, out: Path) -> Optional[Path]:
    fam = mod.get("family_table"); uni = mod.get("curated_univariate"); cur = mod.get("curated")
    within = mod.get("within_family")
    if fam is None or fam.empty or uni is None or uni.empty:
        return None
    fig = plt.figure(figsize=(14.8, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.3], wspace=0.36)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])

    f = fam.sort_values("var_explained_pct"); yy = np.arange(len(f))
    cols = [_FAM_COLOR.get(x, "#888") for x in f["family"]]
    axa.barh(yy, f["var_explained_pct"], color=cols, edgecolor="#1F2430", linewidth=0.4)
    for i, r in enumerate(f.itertuples()):
        axa.text(r.var_explained_pct + 0.005, i, f"{r.var_explained_pct:.2f}%", va="center", fontsize=8.5)
    axa.set_yticks(yy); axa.set_yticklabels([f"{r.family}\n({r.n_features} traits)" for r in f.itertuples()], fontsize=8)
    axa.set_xlim(0, f["var_explained_pct"].max() * 1.25 + 0.02)
    axa.set_title("(a) Variance of susceptibility explained per profile family\n(adjusted in-sample R^2)")
    axa.set_xlabel("Variance explained (%)")

    c = uni.copy()
    fam_order = ["Big Five", "Political Psychology", "Ideology", "Moral Foundations", "Demographics"]
    c["_f"] = c["family"].map({k: i for i, k in enumerate(fam_order)}).fillna(9)
    c = c.sort_values(["_f", "beta_std"])
    for i, r in enumerate(c.itertuples()):
        col = _FAM_COLOR.get(r.family, "#8C8C8C")
        axb.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=2.0, alpha=0.85, zorder=2,
                 solid_capstyle="round")
        axb.scatter([r.beta_std], [i], s=52, color=col if r.significant else "white",
                    edgecolors=col, linewidths=1.6, zorder=3)
        if r.significant:
            axb.text(r.ci_high + 0.004, i, _stars(r.q_value), va="center", fontsize=10, fontweight="bold")
    axb.axvline(0, color="#1F2430", lw=1.0)
    axb.set_yticks(range(len(c))); axb.set_yticklabels(c["moderator"], fontsize=8); axb.set_ylim(-0.8, len(c) - 0.2)
    axb.set_xlabel("Standardised moderation slope (beta, 95% cluster-robust CI; univariate, FDR within family)")
    n_within = int(within["significant"].sum()) if within is not None and not within.empty else 0
    n_cross = int(cur["significant"].sum()) if cur is not None and not cur.empty else 0
    axb.set_title(f"(b) Trait moderators of susceptibility (filled = q<0.05)\n"
                  f"openness +, conscientiousness -, neuroticism + are the significant constructs; "
                  f"{n_within} traits significant at facet level, {n_cross} survive the strict cross-family model")
    handles = [Patch(facecolor=_FAM_COLOR[k], label=k) for k in fam_order if k in set(c["family"])]
    axb.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False, title="Family", title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out / "profile_moderation.png")


# --------------------------------------------------------------------------- #
def _moderator_by_domain(mod: Dict, out: Path) -> Optional[Path]:
    bd = mod.get("by_domain")
    if bd is None or bd.empty:
        return None
    mat = bd.pivot_table(index="moderator", columns="domain", values="beta_std", aggfunc="mean")
    sig = bd.pivot_table(index="moderator", columns="domain", values="significant", aggfunc="max").fillna(False)
    mat = mat.dropna(how="all")
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return None
    try:
        order = dendrogram(linkage(mat.fillna(0).to_numpy(), method="average"), no_plot=True)["leaves"]
    except Exception:
        order = list(range(mat.shape[0]))
    mat = mat.iloc[order]; sig = sig.reindex(index=mat.index, columns=mat.columns).fillna(False)
    col_order = mat.mean(axis=0).sort_values().index; mat = mat[col_order]; sig = sig[col_order]
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) or 0.1
    fig, ax = plt.subplots(figsize=(2.4 + 1.0 * mat.shape[1], 1.6 + 0.34 * mat.shape[0]))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(["\n".join(textwrap.wrap(_short_dom(c), 16)) for c in mat.columns], fontsize=8)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if bool(sig.to_numpy()[i, j]):
                ax.text(j, i, "*", ha="center", va="center", color="#111", fontsize=13, fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("standardised slope (beta)", fontsize=8)
    nsig = int(bd["significant"].sum())
    ax.set_title(f"Trait moderators by issue domain (within-domain, FDR per family; * q<0.05, {nsig} significant)",
                 fontsize=10.5, pad=8)
    fig.tight_layout()
    return _save(fig, out / "moderator_by_domain.png")


# --------------------------------------------------------------------------- #
def _opinion_clusters(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """2x2 on the opinion-cluster hierarchy: a domain dendrogram-ordered domain x
    Execute-tactic matrix, a domain x complexity matrix, the within-domain leaf
    heterogeneity, and the most-movable leaves coloured by parent domain."""
    w = sem.dropna(subset=[_AE, "opinion_leaf_label"]).copy()
    w["dom"] = w["opinion_domain"].map(_short_dom); w["leaf"] = w["opinion_leaf_label"].map(_short_leaf)
    if "attack_execute_tactic" not in w.columns:
        return None
    fig = plt.figure(figsize=(15.5, 11))
    outer = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.24)
    inner = outer[0, 0].subgridspec(1, 2, width_ratios=[0.2, 1.0], wspace=0.03)
    axdend = fig.add_subplot(inner[0, 0]); a = fig.add_subplot(inner[0, 1])
    b = fig.add_subplot(outer[0, 1]); c = fig.add_subplot(outer[1, 0]); d_ax = fig.add_subplot(outer[1, 1])

    # (a) domain x Execute tactic with a domain dendrogram mounted on the left
    m1 = w.pivot_table(index="dom", columns=w["attack_execute_tactic"].astype(str), values=_AE, aggfunc="median")
    link = linkage(m1.to_numpy(), method="average")
    order = dendrogram(link, ax=axdend, orientation="left", color_threshold=0, above_threshold_color="#6b7280")["leaves"]
    axdend.set_axis_off()
    m1 = m1.iloc[order]
    arr = m1.to_numpy()
    im = a.imshow(arr, aspect="auto", cmap=_MAKO, origin="lower")  # origin lower aligns with the left dendrogram
    a.set_xticks(range(m1.shape[1])); a.set_xticklabels(["\n".join(textwrap.wrap(cc.replace("_", " "), 11)) for cc in m1.columns], fontsize=7.5)
    a.set_yticks(range(m1.shape[0])); a.set_yticklabels(["\n".join(textwrap.wrap(d, 20)) for d in m1.index], fontsize=8)
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            a.text(j, i, f"{arr[i,j]:.0f}", ha="center", va="center", fontsize=7,
                   color="white" if arr[i, j] < np.nanmedian(arr) else "#11151c")
    a.grid(False); fig.colorbar(im, ax=a, fraction=0.046, pad=0.02).set_label("median effectivity", fontsize=8)
    a.set_title("(a) Issue domain x DISARM Execute tactic (domain dendrogram)")

    # (b) domain x complexity tier
    if "attack_complexity_tier" in w.columns:
        m2 = w.pivot_table(index=m1.index.name, columns=w["attack_complexity_tier"].astype(str), values=_AE, aggfunc="median").reindex(m1.index)
        im2 = b.imshow(m2.to_numpy(), aspect="auto", cmap=_MAKO)
        b.set_xticks(range(m2.shape[1])); b.set_xticklabels([c.replace("_", " ") for c in m2.columns], fontsize=8)
        b.set_yticks(range(m2.shape[0])); b.set_yticklabels(["\n".join(textwrap.wrap(d, 22)) for d in m2.index], fontsize=8)
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                if np.isfinite(m2.to_numpy()[i, j]):
                    b.text(j, i, f"{m2.to_numpy()[i,j]:.0f}", ha="center", va="center", fontsize=7,
                           color="white" if m2.to_numpy()[i, j] < np.nanmedian(m2.to_numpy()) else "#11151c")
        b.grid(False); fig.colorbar(im2, ax=b, fraction=0.046, pad=0.02).set_label("median effectivity", fontsize=8)
    b.set_title("(b) Issue domain x operation complexity tier")

    # (c) within-domain leaf heterogeneity
    leaf_med = w.groupby(["dom", "leaf"])[_AE].median().reset_index()
    dom_order = leaf_med.groupby("dom")["adversarial_effectivity"].median().sort_values().index.tolist()
    parts = [leaf_med.loc[leaf_med["dom"] == d, _AE].to_numpy() for d in dom_order]
    vp = c.violinplot(parts, vert=False, showmedians=True, widths=0.85)
    for pc, col in zip(vp["bodies"], _MAKO(np.linspace(0.25, 0.85, len(dom_order)))):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    for d_i, d in enumerate(dom_order):
        vals = leaf_med.loc[leaf_med["dom"] == d, _AE].to_numpy()
        c.scatter(vals, np.full(len(vals), d_i + 1) + np.random.default_rng(d_i).uniform(-0.12, 0.12, len(vals)),
                  s=14, color="#11151c", alpha=0.5, zorder=3)
    c.set_yticks(range(1, len(dom_order) + 1)); c.set_yticklabels(["\n".join(textwrap.wrap(d, 22)) for d in dom_order], fontsize=8)
    c.set_xlabel("Per-leaf median effectivity"); c.set_title("(c) Within-domain spread across opinion leaves")

    # (d) most movable leaves coloured by domain
    top = leaf_med.sort_values(_AE, ascending=False).head(16).iloc[::-1]
    dom_set = sorted(w["dom"].unique()); dcol = {dm: _PAL[i % len(_PAL)] for i, dm in enumerate(dom_set)}
    d_ax.barh(range(len(top)), top[_AE], color=[dcol[dm] for dm in top["dom"]], edgecolor="#1F2430", linewidth=0.4)
    d_ax.set_yticks(range(len(top))); d_ax.set_yticklabels([l[:34] for l in top["leaf"]], fontsize=7.5)
    d_ax.set_xlabel("Median effectivity"); d_ax.set_title("(d) Most movable opinion leaves (colour = parent domain)")
    handles = [Patch(facecolor=dcol[dm], label=dm) for dm in dom_set]
    d_ax.legend(handles=handles, fontsize=6.5, loc="lower right", frameon=False)
    for a_ in (a, b, c, d_ax):
        a_.grid(False)
    fig.tight_layout()
    return _save(fig, out / "opinion_clusters.png")


# --------------------------------------------------------------------------- #
def _opinion_treemap(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    w = sem.dropna(subset=[_AE, "opinion_leaf_label"]).copy()
    w["domain"] = w["opinion_domain"].map(_short_dom); w["leaf"] = w["opinion_leaf_label"].map(_short_leaf)
    agg = w.groupby(["domain", "leaf"])[_AE].agg(["median", "count"]).reset_index()
    if agg.empty:
        return None
    dom_med = agg.groupby("domain").apply(lambda g: np.average(g["median"], weights=g["count"])).sort_values(ascending=False)
    domains = list(dom_med.index); n = len(domains)
    vmax = float(np.nanpercentile(agg["median"], 97)); vmin = float(np.nanpercentile(agg["median"], 3))
    cmap = _MAKO; norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(2.1 * n + 0.6, 9.2)); cw = 1.0 / n; pad = 0.06 * cw
    for di, dom in enumerate(domains):
        sub = agg[agg["domain"] == dom].sort_values("median", ascending=False).reset_index(drop=True)
        m = len(sub); x0 = di * cw + pad; ww = cw - 2 * pad
        for li, r in enumerate(sub.itertuples()):
            hh = 1.0 / m; y0 = 1.0 - (li + 1) * hh; rgba = cmap(norm(r.median))
            ax.add_patch(plt.Rectangle((x0, y0), ww, hh, facecolor=rgba, edgecolor="white", linewidth=0.6))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt = (r.leaf[:30] + "...") if len(r.leaf) > 33 else r.leaf
            fs = 7.2 if m <= 16 else (6.4 if m <= 20 else 5.6)
            ax.text(x0 + ww / 2, y0 + hh / 2, txt, ha="center", va="center", fontsize=fs,
                    color="white" if lum < 0.5 else "#11151c", clip_on=True)
        ax.text(di * cw + cw / 2, 1.012, "\n".join(textwrap.wrap(dom, 18)), ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.10); ax.axis("off")
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm); smap.set_array([])
    fig.colorbar(smap, ax=ax, fraction=0.022, pad=0.01).set_label("median adversarial effectivity (movability)", fontsize=9)
    ax.set_title("Opinion-susceptibility hierarchy: issue domains and their leaves (sorted by movability within domain)",
                 fontsize=11, loc="left")
    fig.tight_layout()
    return _save(fig, out / "opinion_susceptibility_treemap.png")


# --------------------------------------------------------------------------- #
def generate_production_figures(sem_long_csv_path: str, moderation: Dict, output_dir: str) -> List[str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sem = drop_excluded_domains(pd.read_csv(sem_long_csv_path))
    produced: List[Path] = []
    for fn in (
        lambda: _overview(sem, moderation, out),
        lambda: _domain_susceptibility(sem, out),
        lambda: _attack_phase_diagnostics(sem, out),
        lambda: _profile_moderation(moderation, out),
        lambda: _moderator_by_domain(moderation, out),
        lambda: _opinion_clusters(sem, out),
        lambda: _opinion_treemap(sem, out),
    ):
        try:
            p = fn()
            if p is not None:
                produced.append(Path(p))
        except Exception as exc:
            import traceback
            print(f"  figure failed: {type(exc).__name__}: {exc}"); traceback.print_exc()
    return [str(p) for p in produced]
