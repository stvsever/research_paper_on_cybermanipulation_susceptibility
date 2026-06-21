from __future__ import annotations

"""
Paper-ready static figures for the INDIVIDUAL layer.

These are high-resolution PNGs aimed at the manuscript, complementary to the
interactive HTML dashboard. They are deliberately dense but kept UX-clean:
hierarchical structure is shown with dendrograms mounted on the effectiveness
matrices, labels are wrapped and sized so nothing overlaps, and the sparse
sampling of the full ontological state space is handled explicitly (thinly
observed cells are masked rather than imputed as zero).

Entry point: generate_individual_layer_paper_figures(...).
"""

import logging
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde as st_gaussian_kde

LOGGER = logging.getLogger(__name__)

_DPI = 200
_AE = "adversarial_effectivity"
# Minimum observations for an (opinion leaf x Execute tactic) cell to be trusted.
_MIN_CELL_N = 5

# A qualitative palette that is colour-blind-aware and prints well.
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#3a3f4a",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e6e8ec",
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#3a3f4a",
        "ytick.color": "#3a3f4a",
        "figure.autolayout": False,
    })


def _raincloud_h(ax, labels, arrays, colors, point_alpha: float = 0.35, seed: int = 0, clusters=None):
    """Horizontal raincloud per category: a half-violin density (upper), a jittered
    raw-point strip (lower), and a mean marker with a 95% bootstrap CI. This shows
    the full distribution and the raw data, not just a single summary bar. When
    *clusters* (a list parallel to *arrays* of scenario ids) is given, the mean
    marker and CI are bootstrapped over scenarios so clustered leaf measurements are
    not treated as independent."""
    rng = np.random.default_rng(seed)
    for i, (lab, vals, col) in enumerate(zip(labels, arrays, colors)):
        vals = np.asarray(vals, float)
        clu = None if clusters is None else np.asarray(clusters[i])
        finite = np.isfinite(vals)
        if clu is not None:
            clu = clu[finite]
        vals = vals[finite]
        if len(vals) == 0:
            continue
        # Half-violin (KDE) above the row centre.
        if len(vals) > 5 and vals.std() > 0:
            try:
                kde = st_gaussian_kde(vals)
                xs = np.linspace(vals.min(), vals.max(), 120)
                dens = kde(xs)
                dens = dens / dens.max() * 0.36
                ax.fill_between(xs, i + 0.06, i + 0.06 + dens, color=col, alpha=0.32, linewidth=0)
            except Exception:
                pass
        # Raw jittered strip below the row centre.
        jitter = i - 0.18 - rng.uniform(0, 0.16, size=len(vals))
        ax.scatter(vals, jitter, s=7, color=col, alpha=point_alpha, linewidth=0, zorder=2)
        # Mean with bootstrap CI (scenario-clustered when cluster ids are supplied).
        if clu is not None and len(clu) == len(vals):
            m, lo, hi = _cluster_bootstrap_ci(vals, clu, seed=seed + i)
        else:
            m, lo, hi = _bootstrap_ci(vals, seed=seed + i)
        ax.errorbar(m, i - 0.02, xerr=[[m - lo], [hi - m]], fmt="o", color="#1F2430",
                    ecolor="#1F2430", elinewidth=1.6, capsize=3, markersize=5, zorder=4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    # Robust x-limits focused on the bulk so a few extreme leaves do not compress it.
    pooled = np.concatenate([np.asarray(a, float)[np.isfinite(a)] for a in arrays if len(a)]) if arrays else np.array([0.0])
    if pooled.size:
        lo_x, hi_x = np.percentile(pooled, 1), np.percentile(pooled, 99)
        pad = max((hi_x - lo_x) * 0.08, 1.0)
        ax.set_xlim(min(lo_x, 0) - pad, hi_x + pad)


def _scn_key(series: pd.Series) -> pd.Series:
    """True analysis scenario id (strip the '__<leaf>' suffix from sem_long scenario_id)."""
    s = series.astype(str)
    return s.str.split("__", n=1).str[0] if s.str.contains("__").any() else s


def _short_leaf(label: str) -> str:
    return str(label).split(" > ")[-1].replace("_", " ").strip()


def _short_domain(label: str) -> str:
    return str(label).split(" > ")[-1].replace("_", " ").strip()


def _wrap(label: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _ordered_linkage(matrix: np.ndarray) -> Optional[np.ndarray]:
    """Average-linkage order over rows of *matrix*, NaN-robust."""
    if matrix.shape[0] < 3:
        return None
    filled = np.where(np.isfinite(matrix), matrix, np.nanmean(matrix))
    filled = np.nan_to_num(filled, nan=float(np.nanmean(filled)) if np.isfinite(np.nanmean(filled)) else 0.0)
    try:
        d = pdist(filled, metric="correlation")
        d = np.nan_to_num(d, nan=1.0)
        return linkage(d, method="average")
    except Exception:
        return None


def _attack_opinion_clustermap(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Opinion leaves (rows) x DISARM Execute tactics (cols) mean-effectivity matrix,
    with dendrograms mounted on both axes and a domain colour strip on the rows."""
    if "attack_execute_tactic" not in sem.columns or sem["attack_execute_tactic"].isna().all():
        return None
    work = sem.dropna(subset=[_AE, "attack_execute_tactic", "opinion_leaf_label"]).copy()
    if work.empty:
        return None
    work["tactic"] = work["attack_execute_tactic"].astype(str)
    work["leaf"] = work["opinion_leaf_label"].map(_short_leaf)
    work["domain"] = work.get("opinion_domain", "").map(_short_domain)

    mat = work.pivot_table(index="leaf", columns="tactic", values=_AE, aggfunc="mean")
    cnt = work.pivot_table(index="leaf", columns="tactic", values=_AE, aggfunc="count")
    mat = mat.where(cnt >= _MIN_CELL_N)  # mask thin cells
    mat = mat.dropna(how="all").dropna(axis=1, how="all")
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return None
    leaf_domain = work.drop_duplicates("leaf").set_index("leaf")["domain"].to_dict()

    row_link = _ordered_linkage(mat.to_numpy())
    col_link = _ordered_linkage(mat.to_numpy().T)
    row_order = dendrogram(row_link, no_plot=True)["leaves"] if row_link is not None else list(range(mat.shape[0]))
    col_order = dendrogram(col_link, no_plot=True)["leaves"] if col_link is not None else list(range(mat.shape[1]))
    mat = mat.iloc[row_order, col_order]

    n_rows, n_cols = mat.shape
    fig = plt.figure(figsize=(2.4 + 0.95 * n_cols, 1.8 + 0.34 * n_rows))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[0.10, 0.04, 1.0], height_ratios=[0.10, 1.0],
        wspace=0.02, hspace=0.02,
    )
    ax_col = fig.add_subplot(gs[0, 2])
    ax_row = fig.add_subplot(gs[1, 0])
    ax_strip = fig.add_subplot(gs[1, 1])
    ax_main = fig.add_subplot(gs[1, 2])

    if col_link is not None:
        dendrogram(col_link, ax=ax_col, color_threshold=0, above_threshold_color="#6b7280")
    ax_col.set_axis_off()
    if row_link is not None:
        dendrogram(row_link, ax=ax_row, orientation="left", color_threshold=0, above_threshold_color="#6b7280")
    ax_row.set_axis_off()

    # Domain colour strip (rows).
    domains = sorted(set(leaf_domain.values()))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(domains), 1)))
    dom_color = {d: palette[i] for i, d in enumerate(domains)}
    strip = np.array([[list(dom_color.get(leaf_domain.get(lf, ""), (0.8, 0.8, 0.8, 1.0)))] for lf in mat.index])
    ax_strip.imshow(strip, aspect="auto")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])

    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax_main.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
    ax_main.set_xticks(range(n_cols))
    ax_main.set_xticklabels([_wrap(c, 14) for c in mat.columns], fontsize=8, rotation=30, ha="right")
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels([_short_leaf(lf) for lf in mat.index], fontsize=7)
    ax_main.tick_params(left=False)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02)
    cbar.set_label("Mean adversarial effectivity (toward goal)", fontsize=8)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=dom_color[d], markersize=8, label=d) for d in domains]
    ax_main.legend(handles=handles, title="Issue domain", fontsize=7, title_fontsize=8,
                   loc="upper left", bbox_to_anchor=(1.16, 1.0), frameon=False)
    fig.suptitle("Attack effectiveness by opinion leaf and DISARM Execute tactic", fontsize=13, y=0.99)
    fig.text(0.5, 0.955, "Rows and columns hierarchically clustered; thinly-observed cells masked white.",
             ha="center", color="#6F768A", fontsize=9)
    return _save(fig, out / "attack_opinion_effectiveness_clustermap.png")


def _opinion_susceptibility_by_domain(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Per-opinion-leaf mean effectivity, grouped and dividered by issue domain."""
    work = sem.dropna(subset=[_AE, "opinion_leaf_label"]).copy()
    if work.empty:
        return None
    work["leaf"] = work["opinion_leaf_label"].map(_short_leaf)
    work["domain"] = work.get("opinion_domain", "").map(_short_domain)
    agg = (
        work.groupby(["domain", "leaf"])[_AE]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["domain", "mean"], ascending=[True, False])
    )
    agg = agg[agg["count"] >= _MIN_CELL_N]
    if agg.empty:
        return None
    domains = sorted(agg["domain"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(domains), 1)))
    dom_color = {d: palette[i] for i, d in enumerate(domains)}

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.26 * len(agg))))
    y = 0
    yticks, ylabels = [], []
    for d in domains:
        sub = agg[agg["domain"] == d]
        for _, r in sub.iterrows():
            err = (r["std"] / np.sqrt(max(r["count"], 1))) if np.isfinite(r["std"]) else 0.0
            ax.barh(y, r["mean"], color=dom_color[d], edgecolor="#3a3f4a", linewidth=0.4, height=0.75)
            ax.errorbar(r["mean"], y, xerr=err, color="#3a3f4a", elinewidth=0.8, capsize=2, fmt="none")
            yticks.append(y)
            ylabels.append(r["leaf"])
            y += 1
        y += 0.8  # domain gap
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean adversarial effectivity (toward attacker goal)")
    ax.axvline(0, color="#1F2430", linewidth=1.0)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=dom_color[d], markersize=8, label=d) for d in domains]
    ax.legend(handles=handles, title="Issue domain", fontsize=7, title_fontsize=8, loc="lower right", frameon=False)
    ax.set_title("Opinion-leaf susceptibility within the issue-domain hierarchy", fontsize=12)
    ax.margins(y=0.005)
    return _save(fig, out / "opinion_leaf_susceptibility_by_domain.png")


def _blockwise_family_figure(stage06_dir: Path, out: Path) -> Optional[Path]:
    """Per-ontology-family standalone out-of-fold R2 and meta weight from the
    block-wise scalable susceptibility model."""
    csv = stage06_dir / "blockwise_family_susceptibility.csv"
    if not csv.exists():
        return None
    fam = pd.read_csv(csv)
    if fam.empty:
        return None
    fam = fam.sort_values("standalone_oof_r2", ascending=True)
    labels = [_wrap(str(f).replace("_", " ").title(), 22) for f in fam["family"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.0, 0.42 * len(fam))), sharey=True)
    yy = np.arange(len(fam))
    c1 = np.where(fam["standalone_oof_r2"] >= 0, "#2a9d8f", "#bcc3cc")
    axes[0].barh(yy, fam["standalone_oof_r2"], color=c1, edgecolor="#3a3f4a", linewidth=0.5)
    axes[0].axvline(0, color="#1F2430", linewidth=1.0)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Standalone out-of-fold R2")
    axes[0].set_title("Predictive signal carried by each trait family")
    c2 = np.where(fam["meta_weight"] >= 0, "#e76f51", "#8ecae6")
    axes[1].barh(yy, fam["meta_weight"], color=c2, edgecolor="#3a3f4a", linewidth=0.5)
    axes[1].axvline(0, color="#1F2430", linewidth=1.0)
    axes[1].set_xlabel("Meta-learner weight in the stack")
    axes[1].set_title("Contribution to the combined model")
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Block-wise family susceptibility model (scalable across the full profile ontology)", fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, out / "blockwise_family_susceptibility.png")


def _execute_tactic_profile(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Mean adversarial effectivity per DISARM Execute tactic, with bootstrap CIs."""
    if "attack_execute_tactic" not in sem.columns or sem["attack_execute_tactic"].isna().all():
        return None
    work = sem.dropna(subset=[_AE, "attack_execute_tactic"]).copy()
    rng = np.random.default_rng(0)
    rows = []
    for tac, g in work.groupby(work["attack_execute_tactic"].astype(str)):
        vals = g[_AE].to_numpy()
        if len(vals) < _MIN_CELL_N:
            continue
        boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(400)]
        rows.append({"tactic": tac, "mean": float(vals.mean()), "lo": float(np.percentile(boots, 2.5)),
                     "hi": float(np.percentile(boots, 97.5)), "n": len(vals)})
    if not rows:
        return None
    order = pd.DataFrame(rows).sort_values("mean")["tactic"].tolist()
    masks = [work["attack_execute_tactic"].astype(str) == t for t in order]
    arrays = [work.loc[m, _AE].to_numpy() for m in masks]
    clusters = ([_scn_key(work.loc[m, "scenario_id"]).to_numpy() for m in masks]
                if "scenario_id" in work.columns else None)
    counts = [len(a) for a in arrays]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(order))]
    _setup_style()
    fig, ax = plt.subplots(figsize=(10.0, 0.95 * len(order) + 2.0))
    _raincloud_h(ax, [f"{t}\n(n={n})" for t, n in zip(order, counts)], arrays, colors, seed=3, clusters=clusters)
    ax.set_yticklabels([f"{t}  (n={n})" for t, n in zip(order, counts)], fontsize=9)
    ax.axvline(0, color="#1F2430", linewidth=1.0, zorder=1)
    ax.set_xlabel("Adversarial effectivity (per leaf measurement; black = mean with 95% scenario-clustered CI)")
    ax.set_title("DISARM Execute-tactic effectiveness: full distribution, raw points and mean CI", fontsize=12)
    ax.set_ylim(-0.6, len(order) - 0.3)
    fig.tight_layout()
    return _save(fig, out / "execute_tactic_effectiveness.png")


def _bootstrap_ci(vals: np.ndarray, n: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    if len(vals) < 3:
        m = float(np.mean(vals)) if len(vals) else 0.0
        return m, m, m
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _cluster_bootstrap_ci(vals: np.ndarray, clusters: np.ndarray, n: int = 400, seed: int = 0) -> tuple:
    """Mean and 95% CI bootstrapping over clusters (scenarios), so clustered leaf
    measurements are not treated as independent. The point estimate is the mean of
    the per-scenario means; resampling is at the scenario level."""
    cmeans = pd.Series(np.asarray(vals, float)).groupby(np.asarray(clusters)).mean().to_numpy()
    if len(cmeans) < 3:
        m = float(cmeans.mean()) if len(cmeans) else 0.0
        return m, m, m
    rng = np.random.default_rng(seed)
    boots = [rng.choice(cmeans, size=len(cmeans), replace=True).mean() for _ in range(n)]
    return float(cmeans.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _disarm_hierarchy_effectiveness(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Attack effectiveness across the DISARM hierarchy: not only the high-level
    Execute tactic but also the Plan and Prepare tactics, the complexity tier and
    the inclusion route. One ranked-with-CI panel per level."""
    levels = [
        ("attack_plan_tactic", "DISARM Plan tactic"),
        ("attack_prepare_tactic", "DISARM Prepare tactic"),
        ("attack_execute_tactic", "DISARM Execute tactic"),
        ("attack_complexity_tier", "Operation complexity tier"),
    ]
    levels = [(c, t) for c, t in levels if c in sem.columns and sem[c].notna().any() and sem[c].nunique() >= 2]
    if not levels:
        return None
    n = len(levels)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 0.5 + 2.6 * nrows), squeeze=False)
    _setup_style()
    work = sem.dropna(subset=[_AE]).copy()
    for k, (col, title) in enumerate(levels):
        ax = axes[k // ncols][k % ncols]
        sub = work.dropna(subset=[col])
        gd = {}
        for lev, g in sub.groupby(sub[col].astype(str)):
            if len(g) >= _MIN_CELL_N:
                gd[str(lev)] = (g[_AE].to_numpy(),
                                _scn_key(g["scenario_id"]).to_numpy() if "scenario_id" in g.columns else None)
        if not gd:
            ax.set_axis_off()
            continue
        order = sorted(gd, key=lambda kk: np.mean(gd[kk][0]))
        arrays = [gd[o][0] for o in order]
        clusters = [gd[o][1] for o in order] if all(gd[o][1] is not None for o in order) else None
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(order))]
        _raincloud_h(ax, order, arrays, colors, point_alpha=0.28, seed=k, clusters=clusters)
        ax.set_yticklabels(["\n".join(textwrap.wrap(f"{o}  (n={len(gd[o][0])})", 24)) for o in order], fontsize=8)
        ax.axvline(0, color="#1F2430", linewidth=1.0, zorder=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Adversarial effectivity")
        ax.set_ylim(-0.6, len(order) - 0.3)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    fig.suptitle("Attack effectiveness across the DISARM operation hierarchy (95% bootstrap CI)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, out / "disarm_attack_hierarchy_effectiveness.png")


def _moderator_opinion_heatmap(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Construct-level moderator (rows) by opinion domain (cols) association matrix:
    the standardized correlation of each construct score with per-domain mean
    susceptibility, with dendrograms when the grid is large enough."""
    constructs = [c for c in sem.columns if c.startswith("profile_cont_") and c.endswith("_mean_pct")]
    if len(constructs) < 3 or "opinion_domain" not in sem.columns:
        return None
    work = sem.dropna(subset=[_AE]).copy()
    work["domain"] = work["opinion_domain"].map(_short_domain)
    # Profile x domain mean AE, plus one profile-level value per construct.
    pdm = work.groupby(["profile_id", "domain"])[_AE].mean().reset_index()
    prof = work.drop_duplicates("profile_id").set_index("profile_id")[constructs]
    domains = sorted(pdm["domain"].unique())
    rows = []
    for c in constructs:
        cvals = prof[c]
        row = {}
        for d in domains:
            sub = pdm[pdm["domain"] == d].set_index("profile_id")[_AE]
            join = pd.concat([cvals, sub], axis=1, join="inner").dropna()
            row[d] = float(join.iloc[:, 0].corr(join.iloc[:, 1])) if len(join) > 5 and join.iloc[:, 0].std() > 0 else np.nan
        rows.append(row)
    mat = pd.DataFrame(rows, index=[_pretty_moderator(c) for c in constructs])
    mat = mat.dropna(how="all").dropna(axis=1, how="all")
    # Keep the most variable constructs so the figure stays readable.
    if mat.shape[0] > 28:
        mat = mat.loc[mat.abs().max(axis=1).sort_values(ascending=False).head(28).index]
    if mat.shape[0] < 3 or mat.shape[1] < 1:
        return None
    row_link = _ordered_linkage(mat.to_numpy())
    if row_link is not None:
        order = dendrogram(row_link, no_plot=True)["leaves"]
        mat = mat.iloc[order]

    fig = plt.figure(figsize=(2.0 + 1.2 * mat.shape[1], 1.6 + 0.34 * mat.shape[0]))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.12, 1.0], wspace=0.02)
    axd = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])
    if row_link is not None:
        dendrogram(row_link, ax=axd, orientation="left", color_threshold=0, above_threshold_color="#6b7280")
    axd.set_axis_off()
    _vm = float(np.nanmax(np.abs(mat.to_numpy()))) if np.isfinite(mat.to_numpy()).any() else 0.0
    vmax = _vm if (np.isfinite(_vm) and _vm > 1e-3) else 0.2
    im = axm.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axm.set_xticks(range(mat.shape[1]))
    axm.set_xticklabels([_wrap(c, 14) for c in mat.columns], fontsize=8, rotation=30, ha="right")
    axm.set_yticks(range(mat.shape[0]))
    axm.set_yticklabels(mat.index, fontsize=7)
    for sp in axm.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=axm, fraction=0.025, pad=0.02)
    cbar.set_label("Correlation with per-domain susceptibility", fontsize=8)
    fig.suptitle("Which trait constructs moderate susceptibility, by opinion domain", fontsize=12, y=1.01)
    return _save(fig, out / "moderator_by_opinion_domain.png")


def _susceptibility_structure(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Coherent multipanel on the structure of susceptibility: baseline to post
    movement, the direction-aware effect split by amplify vs erode leaves, the
    per-profile susceptibility spread, and the effect by operation complexity."""
    work = sem.dropna(subset=[_AE]).copy()
    if work.empty or "baseline_score" not in work.columns:
        return None
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # (a) Baseline -> post movement as a 2D density (hexbin), with the no-change diagonal.
    ax = axes[0, 0]
    d = work.dropna(subset=["baseline_score", "post_score"])
    hb = ax.hexbin(d["baseline_score"], d["post_score"], gridsize=42, bins="log",
                   cmap="magma_r", mincnt=1, linewidths=0.0)
    lim = [min(d["baseline_score"].min(), d["post_score"].min()), max(d["baseline_score"].max(), d["post_score"].max())]
    ax.plot(lim, lim, color="#2a9d8f", linewidth=1.2, linestyle="--", label="no change (P = B)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("measurements (log)", fontsize=8)
    ax.set_xlabel("Baseline opinion B")
    ax.set_ylabel("Post-attack opinion P")
    ax.set_title("(a) Baseline to post-attack movement (density)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # (b) Direction-aware effect split by amplify (+1) vs erode (-1) leaves:
    #     raw jittered points behind a narrow box, y clipped to the bulk.
    ax = axes[0, 1]
    if "adversarial_direction" in work.columns:
        groups = [("Erode (d=-1)", work[work["adversarial_direction"] == -1][_AE].dropna().to_numpy()),
                  ("Amplify (d=+1)", work[work["adversarial_direction"] == 1][_AE].dropna().to_numpy())]
        groups = [(lab, v) for lab, v in groups if len(v) > 3]
        if groups:
            rng = np.random.default_rng(3)
            cols = [_PALETTE[3], _PALETTE[0]]
            pooled = np.concatenate([v for _, v in groups])
            hi = float(np.percentile(pooled, 99)); lo = min(float(np.percentile(pooled, 1)), 0.0)
            for i, (lab, v) in enumerate(groups):
                jx = (i + 1) + rng.uniform(-0.13, 0.13, size=len(v))
                ax.scatter(jx, np.clip(v, lo, hi), s=6, alpha=0.22, color=cols[i % len(cols)],
                           edgecolors="none", zorder=1)
            ax.boxplot([np.clip(v, lo, hi) for _, v in groups], vert=True, patch_artist=True,
                       widths=0.34, showfliers=False, zorder=2,
                       boxprops=dict(facecolor="white", alpha=0.8),
                       medianprops=dict(color="#e63946", linewidth=1.7))
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([g[0] for g in groups], fontsize=9)
            pad = (hi - lo) * 0.05 + 1
            ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0, color="#1F2430", linewidth=0.9)
    ax.set_ylabel("Adversarial effectivity (clipped to 99th pct)")
    ax.set_title("(b) Effect by leaf adversarial direction", fontsize=11)

    # (c) Per-profile susceptibility spread (sorted mean AE with within-profile SD band).
    ax = axes[1, 0]
    pp = work.groupby("profile_id")[_AE].agg(["mean", "std"]).sort_values("mean").reset_index()
    xx = np.arange(len(pp))
    ax.fill_between(xx, pp["mean"] - pp["std"].fillna(0), pp["mean"] + pp["std"].fillna(0),
                    color="#bcd4f0", alpha=0.6, linewidth=0)
    ax.plot(xx, pp["mean"], color="#1d3557", linewidth=1.4)
    ax.axhline(float(work[_AE].mean()), color="#e63946", linestyle="--", linewidth=1.2, label="population mean")
    ax.set_xlabel("Profiles ranked by mean susceptibility")
    ax.set_ylabel("Mean adversarial effectivity")
    ax.set_title(f"(c) Inter-individual heterogeneity\n(between-profile SD = {pp['mean'].std():.1f})", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    # (d) Effect by operation complexity tier (dose-response): raw points + mean bootstrap CI.
    ax = axes[1, 1]
    if "attack_complexity_tier" in work.columns and work["attack_complexity_tier"].notna().any():
        td = work.dropna(subset=["attack_complexity_tier"]).copy()
        td["tier"] = td["attack_complexity_tier"].astype(str)
        tiers = sorted(td["tier"].unique())
        rng = np.random.default_rng(11)
        pooled = td[_AE].dropna().to_numpy()
        hi = float(np.percentile(pooled, 99)); lo = min(float(np.percentile(pooled, 1)), 0.0)
        for i, t in enumerate(tiers):
            sel = td[td["tier"] == t].dropna(subset=[_AE])
            v = sel[_AE].to_numpy()
            jx = i + rng.uniform(-0.18, 0.18, size=len(v))
            ax.scatter(jx, np.clip(v, lo, hi), s=6, alpha=0.20,
                       color=_PALETTE[i % len(_PALETTE)], edgecolors="none", zorder=1)
            if "scenario_id" in sel.columns:
                m, l, h = _cluster_bootstrap_ci(v, _scn_key(sel["scenario_id"]).to_numpy(), seed=i)
            else:
                m, l, h = _bootstrap_ci(v, seed=i)
            ax.errorbar(i, m, yerr=[[m - l], [h - m]], fmt="o", color="#1F2430",
                        ecolor="#1F2430", elinewidth=1.7, capsize=3, markersize=6, zorder=3)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([t.replace("_", "\n") for t in tiers], fontsize=8)
        pad = (hi - lo) * 0.05 + 1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("Adversarial effectivity")
        ax.set_title("(d) Effect by operation complexity tier\n(raw points, mean with 95% CI)", fontsize=11)
    else:
        ax.set_axis_off()
    for a in axes.flat:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Structure of individual susceptibility", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, out / "susceptibility_structure.png")


def _interactive_attack_opinion_html(sem: pd.DataFrame, out: Path) -> Optional[Path]:
    """Interactive (hover-able) opinion-leaf x Execute-tactic effectiveness heatmap."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    if "attack_execute_tactic" not in sem.columns or sem["attack_execute_tactic"].isna().all():
        return None
    work = sem.dropna(subset=[_AE, "attack_execute_tactic", "opinion_leaf_label"]).copy()
    work["leaf"] = work["opinion_leaf_label"].map(_short_leaf)
    mat = work.pivot_table(index="leaf", columns=work["attack_execute_tactic"].astype(str), values=_AE, aggfunc="mean")
    cnt = work.pivot_table(index="leaf", columns=work["attack_execute_tactic"].astype(str), values=_AE, aggfunc="count")
    mat = mat.where(cnt >= _MIN_CELL_N)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return None
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=mat.to_numpy(), x=list(mat.columns), y=list(mat.index),
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="mean AE"),
        hovertemplate="opinion: %{y}<br>tactic: %{x}<br>mean AE: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(template="plotly_white", height=max(420, 22 * mat.shape[0]),
                      title="Attack effectiveness: opinion leaf x DISARM Execute tactic (hover for values)",
                      margin=dict(l=10, r=10, t=60, b=10))
    p = out / "attack_opinion_effectiveness_interactive.html"
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
    return p


def _pretty_moderator(term: str) -> str:
    t = term.replace("profile_cont_", "").replace("profile_cat__profile_cat_", "")
    t = t.replace("_mean_pct_z", "").replace("_z", "").replace("big_five_", "").replace("_", " ")
    return t.strip().title()


def _individual_layer_overview(sem: pd.DataFrame, stage06_dir: Path, out: Path) -> Optional[Path]:
    """Coherent 2x2 manuscript overview: effect distribution, curated moderators,
    block-wise family signal, and the most-movable opinion leaves."""
    ae = pd.to_numeric(sem.get(_AE), errors="coerce").dropna()
    if ae.empty:
        return None
    s06 = Path(stage06_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) Effect distribution (x focused on the bulk; long positive tail summarized).
    ax = axes[0, 0]
    hi = float(np.percentile(ae, 99)); lo = float(min(np.percentile(ae, 1), 0.0))
    n_tail = int((ae > hi).sum())
    ax.hist(ae, bins=44, range=(lo, hi), color="#5b8def", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#1F2430", linewidth=1.0)
    ax.axvline(float(ae.mean()), color="#e63946", linewidth=1.6, linestyle="--",
               label=f"mean = {ae.mean():+.1f}")
    ax.set_xlim(lo, hi)
    if n_tail:
        ax.text(0.97, 0.72, f"+{n_tail} extreme leaves > {hi:.0f}\n(beyond axis)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#5a6068")
    ax.set_title(f"(a) Adversarial effectivity distribution\n{(ae > 0).mean() * 100:.1f}% moved toward the attacker goal", fontsize=11)
    ax.set_xlabel("Adversarial effectivity (per leaf measurement)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)

    # (b) Curated multivariate moderators (OLS robust, bootstrap CI).
    ax = axes[0, 1]
    ols = s06 / "ols_robust_params.csv"
    if ols.exists():
        o = pd.read_csv(ols)
        o = o[o["term"] != "Intercept"].copy()
        o["lab"] = o["term"].map(_pretty_moderator)
        o["sig"] = (o["conf_low"] > 0) | (o["conf_high"] < 0)
        o = o.sort_values("estimate")
        yy = np.arange(len(o))
        cols = np.where(o["sig"], "#2a9d8f", "#9aa1ab")
        ax.barh(yy, o["estimate"], color=cols, edgecolor="#3a3f4a", linewidth=0.4)
        ax.errorbar(o["estimate"], yy, xerr=[o["estimate"] - o["conf_low"], o["conf_high"] - o["estimate"]],
                    fmt="none", ecolor="#3a3f4a", elinewidth=0.8, capsize=2)
        ax.axvline(0, color="#1F2430", linewidth=1.0)
        ax.set_yticks(yy)
        ax.set_yticklabels(o["lab"], fontsize=8)
        ax.set_xlabel("Standardized effect on susceptibility")
        ax.set_title("(b) Curated multivariate moderators\n(filled = bootstrap CI excludes 0)", fontsize=11)
    else:
        ax.set_axis_off()

    # (c) Block-wise family signal.
    ax = axes[1, 0]
    bw = s06 / "blockwise_family_susceptibility.csv"
    if bw.exists():
        f = pd.read_csv(bw).sort_values("standalone_oof_r2")
        yy = np.arange(len(f))
        cols = np.where(f["standalone_oof_r2"] >= 0, "#2a9d8f", "#bcc3cc")
        ax.barh(yy, f["standalone_oof_r2"], color=cols, edgecolor="#3a3f4a", linewidth=0.4)
        ax.axvline(0, color="#1F2430", linewidth=1.0)
        ax.set_yticks(yy)
        ax.set_yticklabels([str(x).replace("_", " ").title() for x in f["family"]], fontsize=8)
        ax.set_xlabel("Standalone out-of-fold R2")
        ax.set_title("(c) Block-wise family predictive signal", fontsize=11)
    else:
        ax.set_axis_off()

    # (d) Most-movable opinion leaves.
    ax = axes[1, 1]
    work = sem.dropna(subset=[_AE, "opinion_leaf_label"]).copy()
    lk = (work.assign(leaf=work["opinion_leaf_label"].map(_short_leaf))
          .groupby("leaf")[_AE].agg(["mean", "count"]).reset_index())
    lk = lk[lk["count"] >= _MIN_CELL_N].sort_values("mean", ascending=False).head(12).iloc[::-1]
    yy = np.arange(len(lk))
    ax.barh(yy, lk["mean"], color="#e76f51", edgecolor="#3a3f4a", linewidth=0.4)
    ax.set_yticks(yy)
    ax.set_yticklabels(lk["leaf"], fontsize=8)
    ax.set_xlabel("Mean adversarial effectivity")
    ax.set_title("(d) Most-movable opinion leaves", fontsize=11)
    for a in axes.flat:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Individual-layer susceptibility overview", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, out / "individual_layer_overview.png")


def generate_individual_layer_paper_figures(
    sem_long_csv_path: str,
    stage06_dir: str,
    output_dir: str,
    run_id: str = "run",
) -> List[str]:
    """Generate the paper-ready individual-layer PNGs into output_dir/paper_figures.

    Returns the list of written file paths. Each figure is guarded so one failure
    never blocks the others or the rest of the pipeline.
    """
    out = Path(output_dir) / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    _setup_style()
    sem = pd.read_csv(sem_long_csv_path)
    s06 = Path(stage06_dir)
    written: List[str] = []
    for fn in (
        lambda: _individual_layer_overview(sem, s06, out),
        lambda: _attack_opinion_clustermap(sem, out),
        lambda: _disarm_hierarchy_effectiveness(sem, out),
        lambda: _moderator_opinion_heatmap(sem, out),
        lambda: _susceptibility_structure(sem, out),
        lambda: _opinion_susceptibility_by_domain(sem, out),
        lambda: _execute_tactic_profile(sem, out),
        lambda: _blockwise_family_figure(s06, out),
        lambda: _interactive_attack_opinion_html(sem, out),
    ):
        try:
            p = fn()
            if p is not None:
                written.append(str(p))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Individual-layer paper figure skipped: %s", exc)
    return written
