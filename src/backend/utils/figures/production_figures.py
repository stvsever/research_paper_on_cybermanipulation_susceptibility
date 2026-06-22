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
from scipy.spatial.distance import pdist, squareform
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
    c.set_ylim(bottom=0, top=float(s.max()) * 1.15)  # effectivity is non-negative; drop the empty negative axis
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
    """Two panels. (A) Per-domain rainclouds with the scenario-clustered median and a 95%
    CI, the scenario-level Kruskal-Wallis omnibus, and significance stars for each domain
    against the least-movable reference domain (Mann-Whitney, BH-FDR). (B) A radar of the
    per-domain median effectivity, gradient-coloured, with the grand-median reference ring."""
    from statsmodels.stats.multitest import multipletests
    w = sem.dropna(subset=[_AE]).copy(); w["_scn"] = _scn(w["scenario_id"]); w["dom"] = w["opinion_domain"]
    scn = _scn_level(w)
    # Order by the scenario-mean median (the analysis unit), consistent with the radar.
    order = scn.groupby("opinion_domain")["ae"].median().sort_values().index.tolist()
    if len(order) < 2:
        return None
    arrays = [w.loc[w["dom"] == d, _AE].to_numpy() for d in order]
    clusters = [w.loc[w["dom"] == d, "_scn"].to_numpy() for d in order]
    colors = _MAKO(np.linspace(0.25, 0.85, len(order)))
    sgroups = {d: scn.loc[scn["opinion_domain"] == d, "ae"].to_numpy() for d in order}
    H, p = st.kruskal(*[sgroups[d] for d in order])
    k = len(order); eps2 = max(0.0, (H - k + 1) / (len(scn) - k))
    ref = order[0]  # least-movable reference (scenario-mean)
    praw, rbmap = [], {}
    for d in order[1:]:
        U, pv = st.mannwhitneyu(sgroups[d], sgroups[ref], alternative="two-sided")
        praw.append(pv)
        rbmap[d] = abs(1.0 - (2.0 * U) / (len(sgroups[d]) * len(sgroups[ref])))  # |rank-biserial|
    q = list(multipletests(praw, method="fdr_bh")[1]) if praw else []
    qmap = {d: q[i] for i, d in enumerate(order[1:])}

    fig = plt.figure(figsize=(16.5, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.16)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1], projection="polar")

    # (A) rainclouds, ordered by movability.
    _raincloud_v(axA, arrays, clusters, colors, seed=3)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(["\n".join(textwrap.wrap(_short_dom(d), 16)) for d in order], fontsize=8.5)
    axA.axhline(0, color="#1F2430", lw=0.9)
    bulk = float(np.percentile(w[_AE], 98))
    for i, d in enumerate(order):
        if d == ref:
            axA.text(i, bulk * 1.03, "ref.", ha="center", va="bottom", fontsize=8, color="#6F768A")
            continue
        # A domain is starred only when it both clears FDR and exceeds a small-effect floor,
        # so trivially-significant (large-n) but negligible differences are not flagged.
        star = _stars(qmap.get(d, np.nan)) if rbmap.get(d, 0.0) >= 0.10 else "ns"
        axA.text(i, bulk * 1.03, star, ha="center", va="bottom",
                 fontsize=12 if star != "ns" else 8, fontweight="bold" if star != "ns" else "normal",
                 color="#11151c")
    axA.set_ylim(-1, bulk * 1.14)
    axA.set_ylabel("Adversarial effectivity")
    axA.set_title(f"(A) Per-domain effectivity   (Kruskal-Wallis $H$ = {H:.0f}, $p$ = {p:.1e}, "
                  f"$\\varepsilon^2$ = {eps2:.3f})", fontsize=10.5)

    # (B) radar of per-domain median effectivity, most-movable first.
    dom_desc = order[::-1]
    rmed = np.array([float(np.median(sgroups[d])) for d in dom_desc])
    ang = np.linspace(0, 2 * np.pi, len(dom_desc), endpoint=False)
    grad = _MAKO(np.linspace(0.85, 0.30, len(dom_desc)))
    axB.set_theta_offset(np.pi / 2); axB.set_theta_direction(-1)
    rmax = float(rmed.max()) * 1.18; axB.set_ylim(0, rmax)
    gm = float(np.median(scn["ae"]))
    axB.plot(np.linspace(0, 2 * np.pi, 220), np.full(220, gm), color="#c44e52", ls="--", lw=1.2,
             label=f"grand median {gm:.1f}")
    aa = np.concatenate([ang, ang[:1]]); rr = np.concatenate([rmed, rmed[:1]])
    axB.plot(aa, rr, color="#1d3557", lw=1.9, zorder=3); axB.fill(aa, rr, color="#457b9d", alpha=0.18, zorder=2)
    for a_, r_, col in zip(ang, rmed, grad):
        axB.scatter(a_, r_, s=95, color=col, edgecolor="#11151c", linewidth=0.8, zorder=5)
        axB.text(a_, r_ + rmax * 0.075, f"{r_:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
    axB.set_xticks(ang); axB.set_xticklabels(["\n".join(textwrap.wrap(_short_dom(d), 14)) for d in dom_desc], fontsize=7.5)
    axB.set_yticks(np.linspace(0, rmax, 4)); axB.set_yticklabels([f"{v:.0f}" for v in np.linspace(0, rmax, 4)], fontsize=6, color="#8a90a0")
    axB.set_rlabel_position(90); axB.grid(color="#e6e8ec", lw=0.6)
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8)
    axB.set_title("(B) Median effectivity by issue domain", fontsize=10.5, pad=22)
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
    gm = float(np.median(scn["ae"]))  # scenario-level grand median (the analysis unit)
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
        # Grand-median reference line, so each tactic's own scenario-clustered median (the
        # black dot from the raincloud) can be read directly against the overall median.
        ax.axhline(gm, color="#c44e52", ls="--", lw=1.2, zorder=1,
                   label=f"grand median {gm:.1f}")
        # Scenario-level Kruskal-Wallis omnibus, drawn as a single span bracket carrying the
        # significance level (ns / * / ** / ***), so a detectable phase (here only Plan) is
        # explicitly flagged rather than left implicit in the title.
        groups = [scn.loc[scn[col] == o, "ae"].to_numpy() for o in order]
        groups = [v for v in groups if len(v) >= 8]
        try:
            H, p = st.kruskal(*groups) if len(groups) >= 2 else (float("nan"), float("nan"))
        except Exception:
            H, p = float("nan"), float("nan")
        bulk = float(np.percentile(sub[_AE], 97)); base = float(min(np.percentile(sub[_AE], 1), 0))
        star = _stars(p); y0 = bulk * 1.02; tick = bulk * 0.04
        ax.plot([0, 0, len(order) - 1, len(order) - 1], [y0, y0 + tick, y0 + tick, y0],
                color="#3a3f4a", lw=1.1, clip_on=False, zorder=5)
        ax.text((len(order) - 1) / 2, y0 + tick * 1.25, star, ha="center", va="bottom",
                fontsize=12 if star != "ns" else 9,
                fontweight="bold" if star != "ns" else "normal", color="#11151c", clip_on=False)
        ax.set_ylim(base - 2, bulk * 1.22)
        ax.set_title(f"({chr(65 + k)}) {title}   (Kruskal-Wallis $p$ = {p:.2g})", fontsize=10.5)
        if k % 2 == 0:
            ax.set_ylabel("Adversarial effectivity")
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
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
    axa.set_title("(a) Variance explained per trait family")
    axa.set_xlabel("Adjusted in-sample R^2 (%)")

    c = uni.copy()
    fam_order = ["Big Five", "Political Psychology", "Ideology", "Moral Foundations", "Demographics"]
    # Order by absolute slope so the strongest moderators sit at the top; family is
    # carried by colour (filled marker = q<.05, so the star annotation is dropped).
    c = c.reindex(c["beta_std"].abs().sort_values(ascending=True).index)
    for i, r in enumerate(c.itertuples()):
        col = _FAM_COLOR.get(r.family, "#8C8C8C")
        axb.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=2.3, alpha=0.85, zorder=2,
                 solid_capstyle="round")
        axb.scatter([r.beta_std], [i], s=58, color=col if r.significant else "white",
                    edgecolors=col, linewidths=1.7, zorder=3)
    axb.axvline(0, color="#1F2430", lw=1.0)
    axb.set_yticks(range(len(c))); axb.set_yticklabels(c["moderator"], fontsize=8); axb.set_ylim(-0.8, len(c) - 0.2)
    axb.set_xlabel("Standardised moderation slope (beta, 95% cluster-robust CI; univariate, FDR within family)")
    axb.set_title("(b) Construct-level moderators")
    handles = [Patch(facecolor=_FAM_COLOR[k], label=k) for k in fam_order if k in set(c["family"])]
    axb.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False, title="Family", title_fontsize=8)
    fig.tight_layout()
    return _save(fig, out / "profile_moderation.png")


# --------------------------------------------------------------------------- #
def _moderator_by_domain(mod: Dict, out: Path) -> Optional[Path]:
    """Seven-panel domain-conditional moderation figure. (A) The full construct-by-domain
    standardised-slope heatmap with a family-grouped dendrogram and a factor-family colour
    strip, mounted tall on the left. (B to G) One radar per issue domain showing the same
    within-domain slopes as a moderation fingerprint, spokes grouped and coloured by family,
    with significant constructs (FDR q<.05) starred and a dashed zero reference ring."""
    from matplotlib.colors import to_rgba
    bd = mod.get("by_domain")
    if bd is None or bd.empty:
        return None
    fam_order = ["Big Five", "Political Psychology", "Ideology", "Moral Foundations", "Demographics"]
    fam_of = bd.drop_duplicates("moderator").set_index("moderator")["family"].to_dict()
    mat = bd.pivot_table(index="moderator", columns="domain", values="beta_std", aggfunc="mean")
    sig = bd.pivot_table(index="moderator", columns="domain", values="significant", aggfunc="max")
    mat = mat.dropna(how="all")
    if mat.shape[0] < 3 or mat.shape[1] < 2:
        return None
    move = ["Critical_Infrastructure_And_Energy_Sovereignty", "Supranational_And_Regional_Integration",
            "Democratic_Resilience_And_Institutions", "Defense_And_National_Security",
            "Information_Integrity_And_Platforms", "Foreign_Policy_And_Geopolitics"]
    cols = [c for c in move if c in mat.columns] + [c for c in mat.columns if c not in move]
    mat = mat[cols]; sig = sig.reindex(index=mat.index, columns=cols).fillna(False)
    # Family-grouped clustering (family penalty forces families to cluster first).
    fams0 = [fam_of.get(m, "Other") for m in mat.index]
    fam_idx = np.array([fam_order.index(f) if f in fam_order else 99 for f in fams0])
    D = squareform(pdist(mat.fillna(0.0).to_numpy(), metric="euclidean"))
    D = D + (fam_idx[:, None] != fam_idx[None, :]) * (D.max() * 3.0 + 1.0)
    link = linkage(squareform(D, checks=False), method="average")
    leaves = dendrogram(link, no_plot=True)["leaves"]
    mat = mat.iloc[leaves]; sig = sig.iloc[leaves]
    fams = [fam_of.get(m, "Other") for m in mat.index]; nrow = mat.shape[0]

    fig = plt.figure(figsize=(21, 13.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.3], wspace=0.10)

    # (A) tall dendrogram-mounted heatmap on the left.
    gsA = outer[0, 0].subgridspec(1, 3, width_ratios=[0.16, 0.035, 1.0], wspace=0.015)
    axd = fig.add_subplot(gsA[0, 0]); axstrip = fig.add_subplot(gsA[0, 1]); axm = fig.add_subplot(gsA[0, 2])
    dendrogram(link, ax=axd, orientation="left", color_threshold=0, above_threshold_color="#6b7280"); axd.set_axis_off()
    strip = np.array([[list(to_rgba(_FAM_COLOR.get(f, "#888")))] for f in fams])
    axstrip.imshow(strip, aspect="auto", origin="lower"); axstrip.set_xticks([]); axstrip.set_yticks([])
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) or 0.1
    im = axm.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    axm.set_xticks(range(mat.shape[1]))
    axm.set_xticklabels(
        [_short_dom(c).replace(" And ", "\n& ").replace(" and ", "\n& ") for c in mat.columns],
        fontsize=7.6, rotation=38, ha="right", rotation_mode="anchor",
    )
    axm.set_yticks(range(nrow)); axm.set_yticklabels(mat.index, fontsize=8); axm.yaxis.tick_right()
    for i in range(nrow):
        for j in range(mat.shape[1]):
            if bool(sig.to_numpy()[i, j]):
                axm.text(j, i, "*", ha="center", va="center", color="#11151c", fontsize=13, fontweight="bold")
    for sp in axm.spines.values():
        sp.set_visible(False)
    axm.grid(False); axm.tick_params(left=False)
    fig.colorbar(im, ax=axm, fraction=0.028, pad=0.30).set_label("standardised slope (beta)", fontsize=8)
    axm.set_title("(A) Construct x issue-domain moderation map", fontsize=11.5, loc="left", pad=8)

    # (B to G) one trait-moderation radar per issue domain.
    cons = bd[["family", "moderator"]].drop_duplicates().copy()
    cons["_f"] = cons["family"].map({k: i for i, k in enumerate(fam_order)}).fillna(9)
    cons = cons.sort_values(["_f", "moderator"]).reset_index(drop=True)
    labels = cons["moderator"].tolist(); rfams = cons["family"].tolist(); nC = len(labels)
    angles = np.linspace(0, 2 * np.pi, nC, endpoint=False)
    bmax = max(0.06, float(np.nanpercentile(np.abs(bd["beta_std"].to_numpy()), 98)))
    domains = bd.groupby("domain")["significant"].sum().sort_values(ascending=False).index.tolist()[:6]
    gsR = outer[0, 1].subgridspec(3, 2, hspace=0.62, wspace=0.30)
    for di, dom in enumerate(domains):
        ax = fig.add_subplot(gsR[di // 2, di % 2], projection="polar")
        sub = bd[bd["domain"] == dom].set_index("moderator")
        vals = np.array([float(sub.loc[m, "beta_std"]) if m in sub.index else 0.0 for m in labels])
        sigf = np.array([bool(sub.loc[m, "significant"]) if m in sub.index else False for m in labels])
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(-bmax, bmax)
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.zeros(200), color="#5b6270", ls="--", lw=0.9, zorder=1)
        aa = np.concatenate([angles, angles[:1]]); vv = np.concatenate([vals, vals[:1]])
        ax.plot(aa, vv, color="#11151c", lw=1.2, zorder=3); ax.fill(aa, vv, color="#1d3557", alpha=0.10, zorder=2)
        for kk in range(nC):
            col = _FAM_COLOR.get(rfams[kk], "#888")
            if sigf[kk]:
                ax.scatter(angles[kk], vals[kk], s=85, marker="*", color=col, edgecolor="#11151c", linewidth=0.6, zorder=6)
            else:
                ax.scatter(angles[kk], vals[kk], s=9, color=col, alpha=0.7, zorder=4)
        ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=5.0)
        for lab, fam in zip(ax.get_xticklabels(), rfams):
            lab.set_color(_FAM_COLOR.get(fam, "#333"))
        ax.tick_params(axis="x", pad=0.5)
        ax.set_yticks([0]); ax.set_yticklabels([]); ax.grid(color="#eceef2", lw=0.5)
        ns = int(sigf.sum())
        ax.set_title(f"({chr(66 + di)}) {_short_dom(dom)}  ({ns} sig.)", fontsize=9.5, pad=16)
    handles = [Patch(facecolor=_FAM_COLOR[k], label=k) for k in fam_order]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.004), title="Trait family   (* within-domain q < .05; dashed ring = zero slope)",
               title_fontsize=9.5)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    return _save(fig, out / "moderator_by_domain.png")


# --------------------------------------------------------------------------- #
def _leaf_rainclouds(ax, w: pd.DataFrame, leaf_df: pd.DataFrame, dcol: Dict, ae_col: str, seed: int = 0) -> None:
    """Horizontal per-leaf rainclouds: for each (domain, leaf) row a half-violin of the
    leaf's raw effectivity above the row, a jittered subsample below, and a median dot,
    coloured by parent domain."""
    rng = np.random.default_rng(seed)
    for i, r in enumerate(leaf_df.itertuples()):
        vals = w.loc[(w["dom"] == r.dom) & (w["leaf"] == r.leaf), ae_col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        col = dcol.get(r.dom, "#888")
        if len(vals) > 8 and vals.std() > 0:
            kde = gaussian_kde(vals)
            xs = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 120)
            dens = kde(xs); dens = dens / dens.max() * 0.42
            ax.fill_between(xs, i + 0.05, i + 0.05 + dens, color=col, alpha=0.45, lw=0, zorder=2)
        idx = rng.choice(len(vals), size=min(len(vals), 220), replace=False)
        ax.scatter(vals[idx], i - 0.10 - rng.uniform(0, 0.18, size=len(idx)), s=5, color=col, alpha=0.30, lw=0, zorder=1)
        ax.scatter(float(np.median(vals)), i, s=44, color="#11151c", zorder=5)
    ax.set_ylim(-0.6, len(leaf_df) - 0.3)
    ax.grid(True, axis="x", color="#eaedf2", lw=0.6)


def _opinion_clusters(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """2x2 on opinion structure, all panels data-varying: within-domain leaf
    heterogeneity, the most-movable and most-resistant opinion leaves as per-leaf
    rainclouds coloured by parent domain, and the amplify-versus-erode contrast that tests
    for a directional asymmetry. Leaf rainclouds carry their labels on the right."""
    w = sem.dropna(subset=[_AE, "opinion_leaf_label"]).copy()
    w["dom"] = w["opinion_domain"].map(_short_dom); w["leaf"] = w["opinion_leaf_label"].map(_short_leaf)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    a, b, c, d_ax = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    leaf_med = w.groupby(["dom", "leaf"])[_AE].median().reset_index()
    dom_set = sorted(w["dom"].unique()); dcol = {dm: _PAL[i % len(_PAL)] for i, dm in enumerate(dom_set)}

    # (a) within-domain spread of per-leaf median effectivity.
    dom_order = leaf_med.groupby("dom")[_AE].median().sort_values().index.tolist()
    parts = [leaf_med.loc[leaf_med["dom"] == d, _AE].to_numpy() for d in dom_order]
    vp = a.violinplot(parts, vert=False, showmedians=True, widths=0.85)
    for pc, col in zip(vp["bodies"], _MAKO(np.linspace(0.25, 0.85, len(dom_order)))):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    for d_i, d in enumerate(dom_order):
        vals = leaf_med.loc[leaf_med["dom"] == d, _AE].to_numpy()
        a.scatter(vals, np.full(len(vals), d_i + 1) + np.random.default_rng(d_i).uniform(-0.12, 0.12, len(vals)),
                  s=14, color="#11151c", alpha=0.5, zorder=3)
    a.set_yticks(range(1, len(dom_order) + 1)); a.set_yticklabels(["\n".join(textwrap.wrap(d, 22)) for d in dom_order], fontsize=8)
    a.set_xlabel("Per-leaf median effectivity"); a.set_title("(a) Within-domain spread across opinion leaves")

    # (b) most-movable opinion leaves as per-leaf rainclouds (distribution + median dot).
    top = leaf_med.sort_values(_AE, ascending=False).head(12).iloc[::-1].reset_index(drop=True)
    _leaf_rainclouds(b, w, top, dcol, _AE, seed=7)
    b.set_xlim(right=80)
    b.set_yticks(range(len(top))); b.set_yticklabels([l[:30] for l in top["leaf"]], fontsize=7.5); b.yaxis.tick_right()
    b.set_xlabel("Per-leaf effectivity (black = median)"); b.set_title("(b) Most movable opinion leaves")

    # (c) amplify (d=+1) versus erode (d=-1) leaf effectivity (the direction test).
    if "adversarial_direction" in w.columns:
        leaf = w.groupby("opinion_leaf_label").agg(ae=(_AE, "mean"), d=("adversarial_direction", "first")).dropna()
        erode = leaf.loc[leaf["d"] == -1, "ae"].to_numpy(); ampl = leaf.loc[leaf["d"] == 1, "ae"].to_numpy()
        groups = [("Erode\n(d = -1)", erode, _PAL[3]), ("Amplify\n(d = +1)", ampl, _PAL[0])]
        rng = np.random.default_rng(5)
        for i, (lab, v, col) in enumerate(groups):
            v = v[np.isfinite(v)]
            if len(v) > 5 and v.std() > 0:
                kde = gaussian_kde(v); ys = np.linspace(v.min(), v.max(), 140)
                dens = kde(ys); dens = dens / dens.max() * 0.34
                c.fill_betweenx(ys, i + 0.06, i + 0.06 + dens, color=col, alpha=0.40, lw=0)
            jit = i - 0.08 - rng.uniform(0, 0.26, size=len(v))
            c.scatter(jit, v, s=16, color=col, alpha=0.5, lw=0)
            med = float(np.median(v))
            c.errorbar(i - 0.01, med, yerr=[[med - np.percentile(v, 25)], [np.percentile(v, 75) - med]],
                       fmt="o", color="#11151c", elinewidth=1.6, capsize=3, ms=6, zorder=5)
        try:
            _, pu = st.mannwhitneyu(erode, ampl, alternative="two-sided")
        except Exception:
            pu = float("nan")
        c.set_xticks([0, 1]); c.set_xticklabels([g[0] for g in groups], fontsize=9)
        c.set_xlim(-0.6, 1.5); c.axhline(0, color="#1F2430", lw=0.8)
        c.set_ylabel("Per-leaf mean effectivity")
        c.set_title(f"(c) Effectivity by adversarial direction (Mann-Whitney $p$ = {pu:.2g})")

    # (d) most-resistant opinion leaves as per-leaf rainclouds (distribution + median dot).
    bot = leaf_med.sort_values(_AE, ascending=True).head(12).reset_index(drop=True)
    _leaf_rainclouds(d_ax, w, bot, dcol, _AE, seed=9)
    d_ax.set_xlim(right=80)
    d_ax.set_yticks(range(len(bot))); d_ax.set_yticklabels([l[:30] for l in bot["leaf"]], fontsize=7.5); d_ax.yaxis.tick_right()
    d_ax.set_xlabel("Per-leaf effectivity (black = median)"); d_ax.set_title("(d) Most resistant opinion leaves")
    handles = [Patch(facecolor=dcol[dm], label=dm) for dm in dom_set]
    a.grid(False)
    c.grid(True, axis="y", color="#eaedf2", lw=0.7)
    fig.legend(handles=handles, loc="lower center", ncol=len(dom_set), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.012), title="Parent issue domain", title_fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    return _save(fig, out / "opinion_clusters.png")


# --------------------------------------------------------------------------- #
def _clipped_mean(s: pd.Series, lo_pct: float = 5.0, hi_pct: float = 95.0) -> float:
    lo, hi = float(np.percentile(s, lo_pct)), float(np.percentile(s, hi_pct))
    return float(np.mean(np.clip(s, lo, hi)))


def _opinion_polar_hierarchy(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """One radar per issue domain (panels A to F): each domain's opinion leaves are the
    spokes and the radius is the leaf's outlier-clipped mean adversarial effectivity
    (5th–95th percentile clip), so the movability fingerprint of each domain's opinion
    set is legible at a glance. The dashed ring marks the grand median (unclipped),
    and the radial scale is shared across domains for comparability."""
    w = sem.dropna(subset=[_AE, "opinion_leaf_label"]).copy()
    w["domain"] = w["opinion_domain"].map(_short_dom); w["leaf"] = w["opinion_leaf_label"].map(_short_leaf)
    leaf_med = (
        w.groupby(["domain", "leaf"])[_AE]
        .agg(_clipped_mean)
        .reset_index()
    )
    if leaf_med.empty:
        return None
    domains = leaf_med.groupby("domain")[_AE].median().sort_values(ascending=False).index.tolist()
    dcol = {dm: _MAKO(x) for dm, x in zip(domains, np.linspace(0.80, 0.30, len(domains)))}
    gm = float(np.median(w[_AE]))  # leaf-level grand median (= +14), the design-wide reference ring
    rmax = float(np.nanpercentile(leaf_med[_AE], 99)) * 1.12
    theta = np.linspace(0, 2 * np.pi, 220)
    ncol = 3; nrow = int(np.ceil(len(domains) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16.5, 5.4 * nrow), subplot_kw=dict(projection="polar"))
    axes = np.atleast_1d(axes).ravel()
    for di, dom in enumerate(domains):
        ax = axes[di]
        sub = leaf_med[leaf_med["domain"] == dom].sort_values(_AE, ascending=False)
        leaves = sub["leaf"].tolist(); vals = sub[_AE].to_numpy()
        dmed = float(np.median(vals))  # this domain's median leaf movability
        n = len(leaves); ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, rmax)
        # Two reference rings: the shared grand median (red dashed) and this domain's own
        # median (dark dotted), so each domain's leaves can be read against both at once.
        ax.plot(theta, np.full(220, gm), color="#c44e52", ls="--", lw=1.0, zorder=1)
        ax.plot(theta, np.full(220, dmed), color="#11151c", ls=":", lw=1.3, zorder=1)
        aa = np.concatenate([ang, ang[:1]]); rr = np.concatenate([vals, vals[:1]])
        ax.plot(aa, rr, color=dcol[dom], lw=1.7, zorder=3); ax.fill(aa, rr, color=dcol[dom], alpha=0.25, zorder=2)
        ax.scatter(ang, vals, s=22, color=dcol[dom], edgecolor="#11151c", linewidth=0.4, zorder=5)
        ax.set_xticks(ang); ax.set_xticklabels([l[:22] for l in leaves], fontsize=5.2)
        ax.set_yticks(np.linspace(0, rmax, 3)); ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(0, rmax, 3)], fontsize=6, color="#8a90a0")
        ax.set_rlabel_position(0); ax.grid(color="#eceef2", lw=0.5)
        ax.text(1.0, 1.10, f"domain median {dmed:.1f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.2, color="#11151c", fontweight="bold")
        ax.set_title(f"({chr(65 + di)}) {dom}", fontsize=9.5, pad=18, loc="left")
    for di in range(len(domains), len(axes)):
        axes[di].set_axis_off()
    handles = [plt.Line2D([0], [0], color="#c44e52", ls="--", lw=1.1, label=f"grand median ({gm:.0f})"),
               plt.Line2D([0], [0], color="#11151c", ls=":", lw=1.3, label="domain median")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, out / "opinion_polar_hierarchy.png")


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
        lambda: _opinion_polar_hierarchy(sem, out),
    ):
        try:
            p = fn()
            if p is not None:
                produced.append(Path(p))
        except Exception as exc:
            import traceback
            print(f"  figure failed: {type(exc).__name__}: {exc}"); traceback.print_exc()
    return [str(p) for p in produced]
