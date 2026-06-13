"""
Research-grade Visualization Suite — Dynamic / Scalable
---------------------------------------------------------
Zero hardcoded column names. Feature hierarchy is auto-discovered via
FeatureRegistry from whatever profile_cont_* / profile_cat__* columns exist.

Scales transparently to:
  • Any personality inventory (Big Five, Dark Triad, HEXACO, …)
  • Any number of opinion leaves / domains
  • Any number of attack vectors (attack × opinion interaction matrix)
  • Any run_N directory structure

Key research questions answered:
  RQ1  Who is susceptible / resilient?                (05, 06, 08)
  RQ2  Which opinion targets are most vulnerable?     (01, 02, 04)
  RQ3  Which attack vectors are most effective?       (01b)
  RQ4  Which profile features drive susceptibility?   (03, 06, 08c)
  RQ5  Does baseline extremity moderate shifts?       (02b)
  RQ6  Are demographic variables significant?         (03b, 03c)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_registry import FeatureRegistry, build_row_color_annotations

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

DPI          = 300
CMAP_AE      = "RdBu"      # red = adversary wins, blue = resilient
CMAP_CORR    = "PiYG"
ADV_RED      = "#d73027"
RES_BLUE     = "#2166ac"
NEUTRAL_GREY = "#7f8c8d"

OUTCOME = "adversarial_effectivity"


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path.name}")


def _mkdir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def _short(s: str, n: int = 4) -> str:
    """Last '>' segment, max n words."""
    leaf = s.split(">")[-1].strip().replace("_", " ")
    w = leaf.split()
    return " ".join(w[:n]) if len(w) > n else leaf


def _short_attack(s: str) -> str:
    parts = [p.strip() for p in s.split(">")]
    return "\n".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _add_categorical_cols(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """
    For every categorical dimension discovered by the registry, add a convenience
    string column (e.g. "sex") decoded from the one-hot level columns.
    Fully dynamic — no hardcoded column names.
    """
    df = df.copy()
    for dim_label, _inv_label, levels in registry.categorical_group_info():
        col_name = dim_label.lower()
        if col_name in df.columns:
            continue  # already present as a string column
        if not levels:
            continue
        df[col_name] = levels[0][1]  # default = last level (acts as fallback)
        # Decode from one-hot: last matching level wins (consistent with one-hot encoding)
        for level_label, level_col in levels:
            if level_col in df.columns:
                df.loc[df[level_col] == 1, col_name] = level_label
    return df


def _cat_palette(registry: FeatureRegistry) -> Dict[str, Dict[str, str]]:
    """
    Returns {dim_label_lower: {level_label: hex_color}} for all categorical dims.
    Used in place of hardcoded PALETTE_SEX etc.
    """
    result: Dict[str, Dict[str, str]] = {}
    for dim_label, _inv_label, levels in registry.categorical_group_info():
        pal = sns.color_palette("Set2", len(levels))
        result[dim_label.lower()] = {
            lbl: matplotlib.colors.to_hex(c) for (lbl, _), c in zip(levels, pal)
        }
    return result


def _profile_meta(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    df = _add_categorical_cols(df, registry)
    feat_cols = [c for c in registry.all_continuous_cols() if c in df.columns]
    cat_cols  = [d.lower() for d, _, _ in registry.categorical_group_info() if d.lower() in df.columns]
    keep = ["profile_id"] + cat_cols + feat_cols
    return df[keep].drop_duplicates("profile_id").set_index("profile_id")


def _ae_wide(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="profile_id", columns="opinion_leaf_label",
        values=OUTCOME, aggfunc="mean")


def _opinion_pal(opinions: List[str]) -> Dict[str, str]:
    pal = sns.color_palette("tab10", len(opinions))
    return {op: matplotlib.colors.to_hex(c) for op, c in zip(sorted(opinions), pal)}


def _assign_domain_colors(domains: List[str]) -> Dict[str, str]:
    pal = sns.color_palette("Set2", len(domains))
    return {d: matplotlib.colors.to_hex(c) for d, c in zip(domains, pal)}


# ── Statistics ────────────────────────────────────────────────────────────────

def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.6,
            n_eval: int = 80,
            x_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xe = x_eval if x_eval is not None else np.linspace(xs.min(), xs.max(), n_eval)
    ye = np.full(len(xe), np.nan)
    bw = frac * (xs.max() - xs.min())
    if bw < 1e-12:
        return xe, np.full(len(xe), ys.mean())
    for i, xv in enumerate(xe):
        w = np.exp(-0.5 * ((xs - xv) / bw) ** 2)
        s = w.sum()
        if s > 0:
            mx, my = np.average(xs, weights=w), np.average(ys, weights=w)
            sxx = np.average((xs - mx) ** 2, weights=w)
            sxy = np.average((xs - mx) * (ys - my), weights=w)
            b1 = sxy / sxx if sxx > 1e-12 else 0
            ye[i] = my + b1 * (xv - mx)
    return xe, ye


def _loess_ci(x: np.ndarray, y: np.ndarray, frac: float = 0.6,
              n_boot: int = 150, n_eval: int = 80
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Shared x_eval grid — same as what _lowess will use on the full data
    xg = np.linspace(x.min(), x.max(), n_eval)
    rng = np.random.default_rng(42)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        try:
            _, yh = _lowess(x[idx], y[idx], frac=frac, n_eval=n_eval, x_eval=xg)
            curves.append(yh)
        except Exception:
            pass
    if not curves:
        return xg, np.full(n_eval, np.nan), np.full(n_eval, np.nan)
    arr = np.array(curves)
    return xg, np.nanpercentile(arr, 2.5, 0), np.nanpercentile(arr, 97.5, 0)


def _sig(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .10 else "ns"


def _bracket(ax, x1, x2, y, p, h=None):
    if h is None:
        h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
    lbl, color = _sig(p), "#333" if p < .10 else "#aaa"
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, color=color)
    ax.text((x1 + x2) / 2, y + h * 1.2, lbl, ha="center", va="bottom",
            fontsize=8, color=color)


def _mwu_pairs(groups: Dict[str, np.ndarray]):
    keys = sorted(groups)
    results = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            a, b = groups[k1], groups[k2]
            if len(a) < 3 or len(b) < 3:
                continue
            if np.std(a) < 1e-10 and np.std(b) < 1e-10:
                continue  # both zero-variance — MWU undefined
            try:
                p = mannwhitneyu(a, b, alternative="two-sided")[1]
                results.append((k1, k2, p))
            except Exception:
                pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ontology helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_leaf_domain_map(opinion_json: dict) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for top_k, top_v in opinion_json.items():
        if top_k.startswith("_") or not isinstance(top_v, dict):
            continue
        for dom_k, dom_v in top_v.items():
            if dom_k.startswith("_") or not isinstance(dom_v, dict):
                continue
            for leaf_k in dom_v:
                if not leaf_k.startswith("_"):
                    mapping[leaf_k] = (dom_k, f"{top_k} > {dom_k} > {leaf_k}")
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Ontology-aware state space
# ─────────────────────────────────────────────────────────────────────────────

def plot_opinion_hierarchy_panel(df, leaf_domain, out_dir):
    opinions = df["opinion_leaf_label"].unique()
    domains  = sorted({leaf_domain.get(lf, ("Unknown",))[0] for lf in opinions})
    dom_colors = _assign_domain_colors(domains)

    rows = []
    for lf in opinions:
        sub = df[df["opinion_leaf_label"] == lf][OUTCOME].dropna()
        dom = leaf_domain.get(lf, ("Unknown",))[0]
        rows.append(dict(leaf=lf, label=_short(lf, 5), domain=dom,
                         mean=sub.mean(), sem=sub.sem(),
                         pct_pos=(sub > 0).mean() * 100, color=dom_colors[dom]))
    plot_df = pd.DataFrame(rows).sort_values(["domain", "mean"], ascending=[True, False])
    yp = np.arange(len(plot_df))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(plot_df) * 0.75 + 2)),
                                   gridspec_kw={"width_ratios": [3, 1]})
    ax.barh(yp, plot_df["mean"], xerr=plot_df["sem"], color=plot_df["color"],
            edgecolor="white", lw=0.5, error_kw={"capsize": 3})
    ax.axvline(0, color="#333", lw=1.0, linestyle="--", alpha=0.6)
    ax.set_yticks(yp); ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.set_xlabel("Mean Adversarial Effectivity (±SEM)"); ax.grid(True, alpha=0.25)
    ax.set_title("Opinion Leaf Effectivity by Domain", fontweight="bold")

    # domain separators
    prev = None
    for i, (_, r) in enumerate(plot_df.iterrows()):
        if r["domain"] != prev and prev is not None:
            ax.axhline(i - 0.5, color="#ccc", lw=0.8)
        prev = r["domain"]

    ax2.barh(yp, plot_df["pct_pos"], color=plot_df["color"], alpha=0.75, edgecolor="white")
    ax2.axvline(50, color="#333", lw=0.8, linestyle="--", alpha=0.5)
    ax2.set_yticks(yp); ax2.set_yticklabels([])
    ax2.set_xlabel("% Positive AE"); ax2.set_xlim(0, 100)

    patches = [mpatches.Patch(color=dom_colors[d], label=d.replace("_", " ")) for d in domains]
    fig.legend(handles=patches, title="Domain", loc="lower right", fontsize=8)
    fig.suptitle("Hierarchical Opinion Effectivity — Grouped by Ontology Domain",
                 fontweight="bold")
    _save(fig, out_dir / "01a_opinion_hierarchy_effectivity.png")


def plot_attack_opinion_matrix(df, leaf_domain, out_dir):
    attacks  = sorted(df["attack_leaf"].unique())
    opinions = sorted(df["opinion_leaf_label"].unique(),
                      key=lambda op: (leaf_domain.get(op, ("",))[0], op))
    mat = np.full((len(attacks), len(opinions)), np.nan)
    n_mat = np.zeros_like(mat, dtype=int)
    for i, atk in enumerate(attacks):
        for j, op in enumerate(opinions):
            sub = df[(df["attack_leaf"] == atk) & (df["opinion_leaf_label"] == op)][OUTCOME].dropna()
            if len(sub):
                mat[i, j] = sub.mean(); n_mat[i, j] = len(sub)

    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 50
    fig, ax = plt.subplots(figsize=(max(6, len(opinions)*1.9+2), max(3, len(attacks)*1.5+2)))
    im = ax.imshow(mat, cmap=CMAP_AE, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean AE", shrink=0.8)
    for i in range(len(attacks)):
        for j in range(len(opinions)):
            if not np.isnan(mat[i, j]):
                c = "white" if abs(mat[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, f"{mat[i,j]:+.1f}\n(n={n_mat[i,j]})",
                        ha="center", va="center", fontsize=7.5, color=c)
    ax.set_xticks(range(len(opinions)))
    ax.set_xticklabels([_short(op, 4) for op in opinions], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([_short_attack(a) for a in attacks], fontsize=8)
    ax.set_title("Attack × Opinion Adversarial Effectivity (multi-attack compatible)",
                 fontweight="bold")
    # domain dividers
    prev = None
    for j, op in enumerate(opinions):
        dom = leaf_domain.get(op, ("",))[0]
        if dom != prev and prev is not None:
            ax.axvline(j - 0.5, color="white", lw=1.5)
        prev = dom
    _save(fig, out_dir / "01b_attack_opinion_matrix.png")


def plot_state_space_pca(df, registry, out_dir):
    df = _add_categorical_cols(df, registry)
    bw = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                         values="baseline_score", aggfunc="mean").dropna()
    if len(bw) < 4:
        return
    X = StandardScaler().fit_transform(bw.values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    meta = _profile_meta(df, registry)
    mean_ae = df.groupby("profile_id")[OUTCOME].mean()
    ae_v = [mean_ae.get(p, np.nan) for p in bw.index]
    vmax = np.nanpercentile(np.abs(ae_v), 95)

    cat_pals = _cat_palette(registry)
    # Use first categorical dimension for coloring (e.g. sex)
    cat_dims = list(cat_pals.keys())
    first_cat = cat_dims[0] if cat_dims else None

    n_panels = 1 + (1 if first_cat else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=ae_v, cmap=CMAP_AE,
                         vmin=-vmax, vmax=vmax, s=60, alpha=0.85, edgecolors="white", lw=0.4)
    plt.colorbar(sc, ax=axes[0], label="Mean AE", shrink=0.85)
    axes[0].set_title("Baseline Opinion Space — Coloured by Mean AE", fontweight="bold")

    if first_cat and first_cat in meta.columns:
        palette = cat_pals[first_cat]
        cat_v = [meta.loc[p, first_cat] if p in meta.index else list(palette.keys())[0]
                 for p in bw.index]
        for level, color in palette.items():
            m = np.array([s == level for s in cat_v])
            axes[1].scatter(coords[m, 0], coords[m, 1], c=color, label=level, s=60,
                            alpha=0.85, edgecolors="white", lw=0.4)
        axes[1].legend(title=first_cat.title())
        axes[1].set_title(f"Baseline Opinion Space — Coloured by {first_cat.title()}",
                          fontweight="bold")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("N-Dimensional Opinion State Space (PCA)", fontweight="bold")
    _save(fig, out_dir / "01c_baseline_pca.png")


def plot_perturbation_pca(df, out_dir):
    bw = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                         values="baseline_score", aggfunc="mean").dropna()
    pw = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                         values="post_score", aggfunc="mean")
    if len(bw) < 4:
        return
    sc = StandardScaler()
    Xb = sc.fit_transform(bw.values)
    # Fill missing post scores with baseline (no-shift assumption, not zero)
    pw_aligned = pw.reindex(bw.index).reindex(columns=bw.columns)
    pw_filled = pw_aligned.where(pw_aligned.notna(), bw)
    Xp = sc.transform(pw_filled.values)
    pca = PCA(n_components=2, random_state=42)
    cb = pca.fit_transform(Xb); cp = pca.transform(Xp)
    var = pca.explained_variance_ratio_
    mean_ae = df.groupby("profile_id")[OUTCOME].mean()
    ae_v = np.array([mean_ae.get(p, 0) for p in bw.index])
    vmax = np.percentile(np.abs(ae_v), 95)
    norm = plt.Normalize(-vmax, vmax); cmap_obj = plt.get_cmap(CMAP_AE)

    fig, ax = plt.subplots(figsize=(9, 8))
    for i in range(len(cb)):
        ax.annotate("", xy=cp[i], xytext=cb[i],
                    arrowprops=dict(arrowstyle="->", color=cmap_obj(norm(ae_v[i])),
                                    lw=0.9, alpha=0.65))
    sc_ = ax.scatter(cb[:, 0], cb[:, 1], c=ae_v, cmap=CMAP_AE, vmin=-vmax, vmax=vmax,
                     s=45, alpha=0.9, zorder=4, edgecolors="#333", lw=0.4, label="Pre")
    plt.colorbar(sc_, ax=ax, label="Mean AE"); ax.scatter(cp[:, 0], cp[:, 1], c="#333", s=18,
                 alpha=0.5, zorder=3, marker="^", label="Post")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title("Opinion Space: Pre→Post Perturbation Vectors\n"
                 "(arrow colour = AE; red = adversary succeeds)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.25)
    _save(fig, out_dir / "01d_perturbation_vectors_pca.png")


def plot_state_space_transition(df, psi_df, out_dir):
    """Paired heatmap: baseline vs post-attack opinion scores, sorted by susceptibility."""
    bw = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                         values="baseline_score", aggfunc="mean")
    pw = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                         values="post_score", aggfunc="mean")
    if len(bw) < 3:
        return
    # Align and sort by CSI
    common = sorted(set(bw.columns) & set(pw.columns))
    if not common:
        return
    bw, pw = bw[common], pw[common]
    common_idx = sorted(set(bw.index) & set(pw.index))
    bw, pw = bw.loc[common_idx], pw.loc[common_idx]
    if psi_df is not None and "csi_percentile" in psi_df.columns:
        csi = psi_df.set_index("profile_id")["csi_percentile"].reindex(common_idx).fillna(50)
        sort_order = csi.sort_values(ascending=False).index
    else:
        mean_ae = df.groupby("profile_id")[OUTCOME].mean()
        sort_order = mean_ae.reindex(common_idx).fillna(0).sort_values(ascending=False).index
    bw, pw = bw.loc[sort_order], pw.loc[sort_order]

    vmin = min(bw.min().min(), pw.min().min())
    vmax = max(bw.max().max(), pw.max().max())
    abs_max = max(abs(vmin), abs(vmax))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, max(6, len(bw) * 0.12 + 2)),
                                          gridspec_kw={"width_ratios": [1, 1, 0.8]})
    sns.heatmap(bw.values, ax=ax1, cmap="RdBu_r", center=0, vmin=-abs_max, vmax=abs_max,
                xticklabels=[c.replace("_", " ") for c in common], yticklabels=False, cbar_kws={"label": "Score"})
    ax1.set_title("Baseline Opinion Scores", fontweight="bold")
    ax1.set_xlabel("Opinion Leaf"); ax1.set_ylabel("Profiles (sorted by susceptibility →)")
    ax1.tick_params(axis="x", rotation=45)

    sns.heatmap(pw.values, ax=ax2, cmap="RdBu_r", center=0, vmin=-abs_max, vmax=abs_max,
                xticklabels=[c.replace("_", " ") for c in common], yticklabels=False, cbar_kws={"label": "Score"})
    ax2.set_title("Post-Attack Opinion Scores", fontweight="bold")
    ax2.set_xlabel("Opinion Leaf")
    ax2.tick_params(axis="x", rotation=45)

    delta = pw.values - bw.values
    d_abs = max(1, np.nanpercentile(np.abs(delta), 97))
    sns.heatmap(delta, ax=ax3, cmap=CMAP_AE, center=0, vmin=-d_abs, vmax=d_abs,
                xticklabels=[c.replace("_", " ") for c in common], yticklabels=False, cbar_kws={"label": "Δ Score"})
    ax3.set_title("Opinion Shift (Post − Baseline)", fontweight="bold")
    ax3.set_xlabel("Opinion Leaf")
    ax3.tick_params(axis="x", rotation=45)

    fig.suptitle("Baseline → Perturbed Opinion State Space", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "01e_state_space_transition.png")


def plot_attack_comparison_panel(df, out_dir):
    """Per-attack violin comparison of adversarial effectivity across opinion leaves."""
    attacks = sorted(df["attack_leaf"].dropna().unique())
    if len(attacks) < 2:
        return
    ops = sorted(df["opinion_leaf_label"].unique())
    n_attacks = len(attacks)
    fig, axes = plt.subplots(1, n_attacks, figsize=(5 * n_attacks, 6), sharey=True)
    if n_attacks == 1:
        axes = [axes]
    pal = _opinion_pal(ops)

    for ax, attack in zip(axes, attacks):
        sub = df[df["attack_leaf"] == attack]
        attack_label = attack.split(" > ")[-1].replace("_", " ") if " > " in attack else attack.replace("_", " ")
        if sub.empty:
            ax.set_title(attack_label); continue
        sns.violinplot(data=sub, x="opinion_leaf_label", y=OUTCOME, palette=pal,
                       inner="quartile", ax=ax, cut=0, linewidth=0.8)
        ax.axhline(0, color="#555", ls="--", lw=0.7)
        ax.set_title(attack_label, fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        if ax == axes[0]:
            ax.set_ylabel("Adversarial Effectivity")
        else:
            ax.set_ylabel("")

    fig.suptitle("Adversarial Effectivity by Attack Vector × Opinion", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "01f_attack_comparison_panel.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Perturbation analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_violins(df, out_dir):
    ops = sorted(df["opinion_leaf_label"].unique())
    pal = _opinion_pal(ops)
    groups = {op: df[df["opinion_leaf_label"] == op][OUTCOME].dropna().values for op in ops}

    fig, ax = plt.subplots(figsize=(max(8, len(ops) * 2.2), 7))
    parts = ax.violinplot([groups[op] for op in ops], positions=range(len(ops)),
                          showmedians=True, showextrema=False)
    for pc, op in zip(parts["bodies"], ops):
        pc.set_facecolor(pal[op]); pc.set_alpha(0.6)
    parts["cmedians"].set_color("#333"); parts["cmedians"].set_linewidth(1.8)
    for i, op in enumerate(ops):
        y = groups[op]
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(y))
        ax.scatter(np.full(len(y), i) + jitter, y, c=pal[op], s=14, alpha=0.55, edgecolors="none", zorder=3)
    ax.axhline(0, color="#555", lw=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(range(len(ops))); ax.set_xticklabels([_short(op, 4) for op in ops], rotation=25, ha="right")
    ax.set_ylabel("Adversarial Effectivity")
    ax.set_title("AE Distribution per Opinion Leaf\n(+ve = adversary succeeded)", fontweight="bold")

    pairs = _mwu_pairs(groups)
    y_max = max(v.max() for v in groups.values() if len(v))
    y_rng = y_max - min(v.min() for v in groups.values() if len(v))
    for rank, (k1, k2, p) in enumerate([x for x in pairs if x[2] < .05][:6]):
        xi, xj = ops.index(k1), ops.index(k2)
        _bracket(ax, xi, xj, y_max + 0.07 * y_rng * (rank + 1), p, 0.012 * y_rng)
    _save(fig, out_dir / "02a_ae_violin_significance.png")


def plot_baseline_extremity(df, out_dir):
    ops = sorted(df["opinion_leaf_label"].unique())
    pal = _opinion_pal(ops)
    ncols = min(len(ops), 2); nrows = (len(ops) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False)
    xcol = "baseline_extremity_norm" if "baseline_extremity_norm" in df.columns else "baseline_abs_score"
    for idx, op in enumerate(ops):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["opinion_leaf_label"] == op].dropna(subset=[xcol, OUTCOME])
        if len(sub) < 5:
            continue
        x, y = sub[xcol].values, sub[OUTCOME].values
        ax.scatter(x, y, c=pal[op], s=28, alpha=0.55, edgecolors="none")
        try:
            xg, yg = _lowess(x, y)
            xg2, lo, hi = _loess_ci(x, y)
            ax.plot(xg, yg, color=pal[op], lw=2.0)
            ax.fill_between(xg2, lo, hi, color=pal[op], alpha=0.18)
        except Exception:
            pass
        r, p = spearmanr(x, y)
        ax.axhline(0, color="#555", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{_short(op, 4)}  ρ={r:+.2f} ({_sig(p)})", fontsize=9)
        ax.set_xlabel("Baseline Extremity"); ax.set_ylabel("AE")
    for idx in range(len(ops), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("RQ5 — Baseline Extremity × AE (LOESS + 95% CI)", fontweight="bold")
    _save(fig, out_dir / "02b_baseline_extremity_loess.png")


def plot_ae_density_cdf(df, out_dir):
    ops = sorted(df["opinion_leaf_label"].unique()); pal = _opinion_pal(ops)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for op in ops:
        v = df[df["opinion_leaf_label"] == op][OUTCOME].dropna()
        if len(v) < 5:
            continue
        kde = stats.gaussian_kde(v, bw_method=0.4)
        xg = np.linspace(v.min() - 5, v.max() + 5, 300)
        ax1.plot(xg, kde(xg), color=pal[op], lw=2.0, label=_short(op, 4))
        ax1.axvline(v.mean(), color=pal[op], lw=0.8, linestyle=":", alpha=0.7)
        sv = np.sort(v.values)
        ax2.plot(sv, np.arange(1, len(sv) + 1) / len(sv), color=pal[op], lw=2.0, label=_short(op, 4))
    for ax in (ax1, ax2):
        ax.axvline(0, color="#333", lw=1.0, linestyle="--", alpha=0.6)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    ax1.set_xlabel("AE"); ax1.set_ylabel("Density"); ax1.set_title("KDE — AE per Opinion", fontweight="bold")
    ax2.set_xlabel("AE"); ax2.set_ylabel("CDF"); ax2.set_title("CDF — AE per Opinion", fontweight="bold")
    ax2.axhline(0.5, color="#888", lw=0.7, linestyle=":", alpha=0.5)
    fig.suptitle("RQ2 — AE Distribution by Opinion Target", fontweight="bold")
    _save(fig, out_dir / "02c_ae_density_cdf.png")


def plot_direction_effect(df, out_dir):
    ops = sorted(df["opinion_leaf_label"].unique()); pal = _opinion_pal(ops)
    x = np.arange(len(ops)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(ops) * 2.5), 5.5))
    for di, (direction, label, hatch) in enumerate([(-1, "d=-1 (goal: decrease)", "//"),
                                                     (+1, "d=+1 (goal: increase)", "")]):
        means, sems = [], []
        for op in ops:
            sub = df[(df["opinion_leaf_label"] == op) &
                     (df["adversarial_direction"] == direction)][OUTCOME].dropna()
            means.append(sub.mean() if len(sub) else 0)
            sems.append(sub.sem() if len(sub) > 1 else 0)
        bars = ax.bar(x + (di - 0.5) * w, means, w, yerr=sems, label=label,
                      alpha=0.75, hatch=hatch, error_kw={"capsize": 3})
        for bar, op in zip(bars, ops): bar.set_color(pal[op])
    ax.axhline(0, color="#333", lw=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([_short(op, 4) for op in ops], rotation=20, ha="right")
    ax.set_ylabel("Mean AE (±SEM)"); ax.legend()
    ax.set_title("Adversarial Direction Effect per Opinion Leaf", fontweight="bold")
    _save(fig, out_dir / "02d_direction_effect_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Moderation (profile features → AE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_moderation_grid(df, registry, moderator_weights, out_dir):
    """
    Dynamic scatter grid: top-K features × N opinions.
    Features ranked by moderator weight (falls back to Spearman |r|).
    One panel per (feature, opinion) with LOESS + first categorical dim colouring.
    """
    df = _add_categorical_cols(df, registry)
    ops = sorted(df["opinion_leaf_label"].unique()); pal = _opinion_pal(ops)
    top_feats = registry.top_features_by_weight(moderator_weights, n=10)
    # Filter to continuous features only (skip one-hot categoricals for scatter)
    top_feats = [(col, lbl) for col, lbl in top_feats
                 if col in df.columns and df[col].nunique() > 5][:8]

    if not top_feats:
        return
    n_feat, n_op = len(top_feats), len(ops)
    fig, axes = plt.subplots(n_feat, n_op,
                             figsize=(3.4 * n_op, 3.2 * n_feat),
                             sharex="row", sharey=False)
    if n_feat == 1: axes = axes[np.newaxis, :]
    if n_op == 1:   axes = axes[:, np.newaxis]

    cat_pals = _cat_palette(registry)
    first_cat = next(iter(cat_pals), None)

    for ri, (fcol, flbl) in enumerate(top_feats):
        for ci, op in enumerate(ops):
            ax = axes[ri][ci]
            sub = df[df["opinion_leaf_label"] == op].dropna(subset=[fcol, OUTCOME])
            if len(sub) < 4:
                continue
            x, y = sub[fcol].values, sub[OUTCOME].values
            if first_cat and first_cat in sub.columns:
                palette = cat_pals[first_cat]
                for level, color in palette.items():
                    m = sub[first_cat] == level
                    ax.scatter(sub.loc[m, fcol], sub.loc[m, OUTCOME],
                               c=color, s=12, alpha=0.55, edgecolors="none", zorder=3)
            else:
                ax.scatter(sub[fcol], sub[OUTCOME], c=pal[op], s=12, alpha=0.55,
                           edgecolors="none", zorder=3)
            try:
                xg, yg = _lowess(x, y, frac=0.7)
                xg2, lo, hi = _loess_ci(x, y, frac=0.7, n_boot=100)
                ax.plot(xg, yg, color=pal[op], lw=1.8)
                ax.fill_between(xg2, lo, hi, color=pal[op], alpha=0.18)
            except Exception:
                pass
            r, p = spearmanr(x, y)
            ax.axhline(0, color="#888", lw=0.6, linestyle="--", alpha=0.5)
            title = f"ρ={r:+.2f}{_sig(p)}"
            if ri == 0:
                title = f"{_short(op, 3)}\n{title}"
            ax.set_title(title, fontsize=8, pad=2)
            if ci == 0:
                ax.set_ylabel(flbl[:22], fontsize=8)
    cat_title = first_cat.title() if first_cat else "Category"
    fig.suptitle(f"RQ4 — Feature × Opinion Moderation Grid (LOESS + 95% CI; dots by {cat_title})",
                 fontweight="bold", y=1.01)
    if first_cat and first_cat in cat_pals:
        handles = [mpatches.Patch(color=c, label=lv)
                   for lv, c in cat_pals[first_cat].items()]
        fig.legend(handles=handles, title=cat_title, loc="upper right", fontsize=7)
    _save(fig, out_dir / "03a_feature_moderation_grid.png")


def plot_categorical_violins(df, registry, out_dir):
    """
    For every categorical dimension in the registry × every opinion:
    violin + MWU brackets.
    """
    df = _add_categorical_cols(df, registry)
    ops = sorted(df["opinion_leaf_label"].unique())

    for inv in registry.inventories.values():
        for dim in inv.categorical_dimensions:
            level_cols = [(lf.col, lf.label) for lf in dim.leaves]
            if not level_cols:
                continue
            n_op = len(ops); ncols = min(n_op, 2); nrows = (n_op + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
            for idx, op in enumerate(ops):
                ax = axes[idx // ncols][idx % ncols]
                sub = df[df["opinion_leaf_label"] == op]
                groups_data: Dict[str, np.ndarray] = {}
                for col, level in level_cols:
                    if col in sub.columns:
                        vals = sub[sub[col] == 1][OUTCOME].dropna().values
                        if len(vals) >= 3:
                            groups_data[level] = vals
                if not groups_data:
                    ax.set_visible(False); continue
                pal_lvl = sns.color_palette("Set2", len(groups_data))
                levels = sorted(groups_data)
                parts = ax.violinplot([groups_data[lv] for lv in levels],
                                      positions=range(len(levels)),
                                      showmedians=True, showextrema=False)
                for pc, (lv, color) in zip(parts["bodies"], zip(levels, pal_lvl)):
                    pc.set_facecolor(matplotlib.colors.to_hex(color)); pc.set_alpha(0.6)
                parts["cmedians"].set_color("#333"); parts["cmedians"].set_linewidth(1.8)
                for i, lv in enumerate(levels):
                    y = groups_data[lv]
                    jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(y))
                    ax.scatter(np.full(len(y), i) + jitter, y,
                               c=matplotlib.colors.to_hex(pal_lvl[i]), s=12, alpha=0.55,
                               edgecolors="none", zorder=3)
                ax.axhline(0, color="#555", lw=0.8, linestyle="--", alpha=0.6)
                ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, fontsize=9)
                ax.set_ylabel("AE"); ax.set_title(_short(op, 4), fontsize=9, fontweight="bold")
                pairs = _mwu_pairs(groups_data)
                y_top = max(v.max() for v in groups_data.values())
                y_rng = y_top - min(v.min() for v in groups_data.values())
                for rank, (k1, k2, p) in enumerate(pairs):
                    _bracket(ax, levels.index(k1), levels.index(k2),
                             y_top + 0.08 * y_rng * (rank + 1), p, 0.012 * y_rng)
            for idx in range(len(ops), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fname = dim.label.replace(" ", "_").lower()
            fig.suptitle(f"RQ6 — {dim.label} × Opinion AE (MWU significance)",
                         fontweight="bold")
            _save(fig, out_dir / f"03b_{fname}_violin_significance.png")



def plot_top_pair_surface(df, registry, moderator_weights, out_dir):
    """
    2D kernel surface for the top-2 continuous features from moderator weights.
    """
    top = [(col, lbl) for col, lbl in
           registry.top_features_by_weight(moderator_weights, n=6)
           if col in df.columns and df[col].nunique() > 5]
    if len(top) < 2:
        return
    xcol, xlbl = top[0]; ycol, ylbl = top[1]

    sub = df.dropna(subset=[xcol, ycol, OUTCOME])
    x, y, z = sub[xcol].values, sub[ycol].values, sub[OUTCOME].values
    vmax = np.percentile(np.abs(z), 95)

    n_g = 35; bwx = 0.25 * (x.max()-x.min()); bwy = 0.25 * (y.max()-y.min())
    xi = np.linspace(x.min(), x.max(), n_g); yi = np.linspace(y.min(), y.max(), n_g)
    XG, YG = np.meshgrid(xi, yi); Z = np.full((n_g, n_g), np.nan)
    for i in range(n_g):
        for j in range(n_g):
            w = np.exp(-0.5*((x-xi[j])/bwx)**2) * np.exp(-0.5*((y-yi[i])/bwy)**2)
            if w.sum() > 1e-10:
                Z[i, j] = np.average(z, weights=w)

    ops = sorted(df["opinion_leaf_label"].unique()); pal = _opinion_pal(ops)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    im = ax1.contourf(XG, YG, Z, levels=16, cmap=CMAP_AE, vmin=-vmax, vmax=vmax, alpha=0.85)
    ax1.contour(XG, YG, Z, levels=8, colors="white", linewidths=0.4, alpha=0.5)
    plt.colorbar(im, ax=ax1, label="Mean AE (kernel-weighted)", shrink=0.85)
    ax1.scatter(x, y, c=z, cmap=CMAP_AE, vmin=-vmax, vmax=vmax,
                s=15, alpha=0.5, edgecolors="none", zorder=4)
    ax1.set_xlabel(xlbl); ax1.set_ylabel(ylbl)
    ax1.set_title(f"{xlbl} × {ylbl} → AE (pooled, kernel surface)", fontweight="bold")
    for op in ops:
        s = df[df["opinion_leaf_label"] == op].dropna(subset=[xcol, ycol, OUTCOME])
        sc = ax2.scatter(s[xcol], s[ycol], c=s[OUTCOME], cmap=CMAP_AE, vmin=-vmax, vmax=vmax,
                         s=22, alpha=0.6, edgecolors=pal[op], lw=0.4, label=_short(op, 3))
    plt.colorbar(sc, ax=ax2, label="AE", shrink=0.85)
    ax2.set_xlabel(xlbl); ax2.set_ylabel(ylbl)
    ax2.set_title("Per-Opinion Scatter", fontweight="bold"); ax2.legend(fontsize=7)
    fig.suptitle(f"RQ4 — Top-2 Feature Interaction Surface: {xlbl} × {ylbl}", fontweight="bold")
    _save(fig, out_dir / "03d_top_pair_surface.png")


def plot_hierarchical_r2(moderator_weights, out_dir):
    if moderator_weights is None or len(moderator_weights) == 0:
        return
    df = moderator_weights.copy()
    df["abs_w"] = df["normalized_weight_pct"].abs()
    df = df.sort_values("abs_w", ascending=True)
    ogs = df["ontology_group"].unique()
    og_pal = _assign_domain_colors(sorted(ogs))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(df)*0.38+2)),
                                   gridspec_kw={"width_ratios": [3, 1]})
    colors = [og_pal.get(og, "#888") for og in df["ontology_group"]]
    ax.barh(df["moderator_label"], df["abs_w"], color=colors, edgecolor="white", lw=0.4)
    ax.set_xlabel("Normalised |Weight| (%)"); ax.set_title("Hierarchical Feature Importance", fontweight="bold")
    bar_colors = [ADV_RED if w > 0 else RES_BLUE for w in df["weighted_mean_estimate"]]
    ax2.barh(df["moderator_label"], df["weighted_mean_estimate"],
             color=bar_colors, edgecolor="white", lw=0.4, alpha=0.8)
    ax2.axvline(0, color="#333", lw=0.8); ax2.set_yticks(range(len(df))); ax2.set_yticklabels([])
    ax2.set_xlabel("Signed Effect"); ax2.set_title("Direction\n(+= susceptible)", fontsize=9, fontweight="bold")
    patches = [mpatches.Patch(color=og_pal[og], label=og) for og in sorted(ogs)]
    fig.legend(handles=patches, title="Feature Group", loc="lower right", fontsize=7.5)
    fig.suptitle("RQ4 — Hierarchical Feature Decomposition (task-reliability weighted)", fontweight="bold", y=1.01)
    _save(fig, out_dir / "03e_hierarchical_r2_decomp.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_outcome_violin_panel(df, out_dir):
    outcomes = [(OUTCOME, "Adversarial Effectivity"),
                ("delta_score", "Signed Δ Score"),
                ("abs_delta_score", "|Δ| Score")]
    ops = sorted(df["opinion_leaf_label"].unique()); pal = _opinion_pal(ops)
    short = {op: _short(op, 3) for op in ops}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (col, title) in zip(axes, outcomes):
        if col not in df.columns:
            ax.set_visible(False); continue
        data = [df[df["opinion_leaf_label"] == op][col].dropna().values for op in ops]
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc, op in zip(parts["bodies"], ops):
            pc.set_facecolor(pal[op]); pc.set_alpha(0.65)
        for k in ["cmedians","cmins","cmaxes","cbars"]:
            if k in parts: parts[k].set_color("#333")
        ax.axhline(0, color="#555", lw=0.9, linestyle="--", alpha=0.6)
        ax.set_xticks(range(1, len(ops)+1))
        ax.set_xticklabels([short[op] for op in ops], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(title); ax.set_title(title, fontweight="bold")
    fig.suptitle("Outcome Distributions by Opinion Leaf", fontweight="bold")
    _save(fig, out_dir / "04a_outcome_violin_panel.png")


def plot_positive_rate_bars(df, leaf_domain, out_dir):
    ops = sorted(df["opinion_leaf_label"].unique())
    domains = sorted({leaf_domain.get(op, ("Unknown",))[0] for op in ops})
    dom_colors = _assign_domain_colors(domains)
    ops_sorted = sorted(ops, key=lambda op: (leaf_domain.get(op, ("",))[0],
                                              -(df[df["opinion_leaf_label"]==op][OUTCOME].dropna()>0).mean()))
    colors = [dom_colors[leaf_domain.get(op, ("Unknown",))[0]] for op in ops_sorted]
    pos_r = [(df[df["opinion_leaf_label"]==op][OUTCOME].dropna()>0).mean()*100 for op in ops_sorted]
    means = [df[df["opinion_leaf_label"]==op][OUTCOME].dropna().mean() for op in ops_sorted]
    sems  = [df[df["opinion_leaf_label"]==op][OUTCOME].dropna().sem() for op in ops_sorted]
    x = np.arange(len(ops_sorted))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(max(9, len(ops)*2.5), 5.5))
    ax.bar(x, pos_r, color=colors, edgecolor="white", lw=0.5, alpha=0.82)
    ax.axhline(50, color="#333", lw=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([_short(op, 4) for op in ops_sorted], rotation=25, ha="right")
    ax.set_ylabel("% Positive AE"); ax.set_ylim(0, 105)
    ax.set_title("% Profiles with Positive AE", fontweight="bold")
    ax2.bar(x, means, yerr=sems, color=colors, edgecolor="white", lw=0.5, alpha=0.82,
            error_kw={"capsize": 4})
    ax2.axhline(0, color="#333", lw=1.0, linestyle="--", alpha=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels([_short(op, 4) for op in ops_sorted], rotation=25, ha="right")
    ax2.set_ylabel("Mean AE (±SEM)"); ax2.set_title("Mean AE per Opinion", fontweight="bold")
    patches = [mpatches.Patch(color=dom_colors[d], label=d.replace("_"," ")) for d in domains]
    fig.legend(handles=patches, title="Domain", loc="upper right", fontsize=8)
    fig.suptitle("RQ2 — Which Opinions Are Most Susceptible?", fontweight="bold")
    _save(fig, out_dir / "04b_positive_rate_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
# 05 — Hierarchical clustering
# ─────────────────────────────────────────────────────────────────────────────

def plot_clustermap(df, registry, psi_df, out_dir):
    """
    Seaborn clustermap: profiles × opinion leaves.
    Row colour annotations are fully dynamic — one bar per inventory/dimension
    discovered by FeatureRegistry (scales to any future personality subtrees).
    """
    df = _add_categorical_cols(df, registry)
    ae_mat = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                             values=OUTCOME, aggfunc="mean").dropna()
    if ae_mat.empty or ae_mat.shape[0] < 5:
        return
    meta = _profile_meta(df, registry)
    profiles = ae_mat.index.tolist()

    row_colors = build_row_color_annotations(registry, profiles, meta)

    # Column annotation: adversarial direction
    dir_map = df.groupby("opinion_leaf_label")["adversarial_direction"].first()
    col_dir = pd.Series(
        [ADV_RED if dir_map.get(op, 0) == -1 else
         RES_BLUE if dir_map.get(op, 0) == 1 else NEUTRAL_GREY
         for op in ae_mat.columns],
        index=ae_mat.columns, name="Adv. Direction"
    )
    vmax = np.nanpercentile(np.abs(ae_mat.values), 95)

    try:
        g = sns.clustermap(
            ae_mat,
            cmap=CMAP_AE, center=0, vmin=-vmax, vmax=vmax,
            row_colors=row_colors if not row_colors.empty else None,
            col_colors=col_dir.to_frame().T,
            figsize=(max(8, ae_mat.shape[1]*1.8+5),
                     max(12, ae_mat.shape[0]*0.13+4)),
            dendrogram_ratio=(0.12, 0.08),
            cbar_pos=(0.02, 0.05, 0.03, 0.35),
            xticklabels=[_short(op, 4) for op in ae_mat.columns],
            yticklabels=False,
            linewidths=0, method="ward", metric="euclidean",
        )
        g.fig.suptitle("RQ1 — Profile × Opinion AE Clustermap\n"
                       "(row annotations auto-built from discovered feature inventory; "
                       "col annotation: adversarial direction)",
                       y=1.01, fontsize=11, fontweight="bold")
        dir_patches = [mpatches.Patch(color=ADV_RED,  label="d=-1 (goal: decrease)"),
                       mpatches.Patch(color=RES_BLUE,  label="d=+1 (goal: increase)"),
                       mpatches.Patch(color=NEUTRAL_GREY, label="d=0 (ambiguous)")]
        g.fig.legend(handles=dir_patches, loc="upper right", fontsize=7.5,
                     title="Direction", framealpha=0.9, bbox_to_anchor=(1.02, 0.98))
        _save(g.fig, out_dir / "05a_clustermap_profiles_opinions.png")
    except Exception as e:
        print(f"    [warn] clustermap failed: {e}")


def _radar_ax(ax, vals, labels, color, label, alpha=0.25, lw=1.8):
    n = len(vals)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    vp = list(vals) + [vals[0]]; ap = angles + [angles[0]]
    ax.plot(ap, vp, color=color, linewidth=lw, label=label)
    ax.fill(ap, vp, color=color, alpha=alpha)
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=7.5)


def plot_radar_by_categorical_dim(df, registry, out_dir):
    """
    For each categorical dimension in the registry, draw one radar subplot
    per level showing pre vs post opinion scores. Fully dynamic — no hardcoded
    sex column or palette.
    """
    df = _add_categorical_cols(df, registry)
    ops = sorted(df["opinion_leaf_label"].unique()); labels = [_short(op, 3) for op in ops]
    if not ops:
        return
    cat_pals = _cat_palette(registry)

    for dim_label, _inv_label, levels in registry.categorical_group_info():
        col_name = dim_label.lower()
        if col_name not in df.columns or not levels:
            continue
        palette = cat_pals.get(col_name, {})
        level_labels = [lbl for lbl, _ in levels]
        n_levels = len(level_labels)
        fig = plt.figure(figsize=(5 * n_levels, 5))
        for si, lv in enumerate(level_labels):
            ax = fig.add_subplot(1, n_levels, si + 1, projection="polar")
            sub = df[df[col_name] == lv]
            if not len(sub):
                ax.set_visible(False); continue
            pre  = [sub[sub["opinion_leaf_label"]==op]["baseline_score"].mean() for op in ops]
            post = [sub[sub["opinion_leaf_label"]==op]["post_score"].mean() for op in ops]
            _radar_ax(ax, pre,  labels, RES_BLUE, "Pre",  alpha=0.2)
            _radar_ax(ax, post, labels, ADV_RED,  "Post", alpha=0.2)
            ax.set_title(lv, fontsize=11, fontweight="bold", pad=15)
            ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fname = dim_label.replace(" ", "_").lower()
        fig.suptitle(f"Radar: Pre vs Post Opinion Scores by {dim_label}", fontweight="bold")
        _save(fig, out_dir / f"05b_radar_pre_post_{fname}.png")


def plot_radar_susceptibility_quartile(df, psi_df, out_dir):
    ops = sorted(df["opinion_leaf_label"].unique()); labels = [_short(op, 3) for op in ops]
    if psi_df is not None and "susceptibility_index_pct" in psi_df.columns:
        q_map = psi_df.set_index("profile_id")["susceptibility_index_pct"]
        q_map = pd.qcut(q_map, 4, labels=[1,2,3,4], duplicates="drop")
        df = df.copy(); df["q"] = df["profile_id"].map(q_map)
    else:
        mae = df.groupby("profile_id")[OUTCOME].mean()
        q_map = pd.qcut(mae, 4, labels=[1,2,3,4], duplicates="drop")
        df = df.copy(); df["q"] = df["profile_id"].map(q_map)
    pal_q = {1: RES_BLUE, 2: "#74add1", 3: "#fdae61", 4: ADV_RED}
    fig = plt.figure(figsize=(14, 5))
    for qi in [1,2,3,4]:
        ax = fig.add_subplot(1, 4, qi, projection="polar")
        sub = df[df["q"]==qi]
        ae_v = [sub[sub["opinion_leaf_label"]==op][OUTCOME].mean() for op in ops]
        offset = abs(min(ae_v, default=0)) + 1
        _radar_ax(ax, [v+offset for v in ae_v], labels, pal_q[qi], f"Q{qi}", alpha=0.28, lw=2.0)
        ax.set_title(f"CSI Q{qi}\n(n={sub['profile_id'].nunique()})",
                     fontsize=9, fontweight="bold", pad=12)
    fig.suptitle("RQ1 — AE by Susceptibility Quartile (Radar)", fontweight="bold")
    _save(fig, out_dir / "05c_radar_ae_by_csi_quartile.png")


# ─────────────────────────────────────────────────────────────────────────────
# 06 — Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_profile_opinion_heatmap(df, psi_df, leaf_domain, out_dir):
    ae_mat = df.pivot_table(index="profile_id", columns="opinion_leaf_label",
                             values=OUTCOME, aggfunc="mean")
    col_order = sorted(ae_mat.columns,
                       key=lambda op: (leaf_domain.get(op,("",))[0], op))
    ae_mat = ae_mat[col_order]
    if psi_df is not None and "susceptibility_index_pct" in psi_df.columns:
        order = psi_df.sort_values("susceptibility_index_pct", ascending=False)["profile_id"].values
        ae_mat = ae_mat.reindex([p for p in order if p in ae_mat.index])
    vmax = np.nanpercentile(np.abs(ae_mat.values), 95)
    fig, ax = plt.subplots(figsize=(max(6, ae_mat.shape[1]*1.7+3),
                                    max(8, ae_mat.shape[0]*0.13+3)))
    sns.heatmap(ae_mat, ax=ax, cmap=CMAP_AE, center=0, vmin=-vmax, vmax=vmax,
                xticklabels=[_short(op,3) for op in ae_mat.columns], yticklabels=False,
                linewidths=0, cbar_kws={"label":"Mean AE","shrink":0.7})
    prev = None
    for j, op in enumerate(ae_mat.columns):
        dom = leaf_domain.get(op, ("",))[0]
        if dom != prev and prev is not None:
            ax.axvline(j, color="white", lw=2.0)
        prev = dom
    ax.set_xlabel("Opinion Leaf (sorted by domain)")
    ax.set_ylabel(f"Profile (sorted by CSI ↑ susceptible)")
    ax.set_title("RQ1+RQ2 — Profile × Opinion AE Heatmap\n"
                 "(top = most susceptible; red = adversary succeeds)", fontweight="bold")
    _save(fig, out_dir / "06a_profile_opinion_ae_heatmap.png")


def plot_facet_opinion_heatmap(df, registry, out_dir):
    """
    Dynamic: all discovered continuous facets × opinion leaves.
    Grouped by inventory/dimension with separator lines.
    """
    ops = sorted(df["opinion_leaf_label"].unique())

    # Collect (col, dim_label, inv_label) for all facets + means
    all_feats = []
    for inv in registry.inventories.values():
        for dim in inv.continuous_dimensions:
            for lf in dim.leaves:
                if lf.col in df.columns:
                    all_feats.append((lf.col, lf.label, dim.label, inv.label, dim.color))

    if not all_feats:
        return

    corr_mat = np.full((len(all_feats), len(ops)), np.nan)
    pval_mat = np.full_like(corr_mat, np.nan)
    for ri, (fcol, *_) in enumerate(all_feats):
        for ci, op in enumerate(ops):
            sub = df[df["opinion_leaf_label"]==op].dropna(subset=[fcol, OUTCOME])
            if len(sub) >= 5:
                r, p = spearmanr(sub[fcol], sub[OUTCOME])
                corr_mat[ri, ci] = r; pval_mat[ri, ci] = p

    fig_h = max(8, len(all_feats)*0.32+2)
    fig, ax = plt.subplots(figsize=(max(6, len(ops)*1.7+2), fig_h))
    im = ax.imshow(corr_mat, cmap=CMAP_CORR, vmin=-0.5, vmax=0.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.7)
    for ri in range(len(all_feats)):
        for ci in range(len(ops)):
            p = pval_mat[ri, ci]
            if not np.isnan(p) and p < 0.10:
                lbl = "**" if p < .01 else ("*" if p < .05 else ".")
                ax.text(ci, ri, lbl, ha="center", va="center", fontsize=7.5,
                        color="white" if abs(corr_mat[ri,ci]) > 0.3 else "black")
    ax.set_xticks(range(len(ops))); ax.set_xticklabels([_short(op,3) for op in ops], rotation=30, ha="right", fontsize=9)
    # Row labels: short feature label; dim separators
    row_labels = [f[1] for f in all_feats]
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=6.5)
    # Dimension/inventory separator lines
    prev_dim = None
    for ri, (_, _, dim_lbl, inv_lbl, _) in enumerate(all_feats):
        key = (inv_lbl, dim_lbl)
        if key != prev_dim and prev_dim is not None:
            ax.axhline(ri-0.5, color="white", lw=1.5)
        prev_dim = key
    # Right-side inventory labels (auto-discovered)
    inv_starts: Dict[str, int] = {}
    for ri, (_, _, _, inv_lbl, _) in enumerate(all_feats):
        if inv_lbl not in inv_starts: inv_starts[inv_lbl] = ri
    for inv_lbl, start in inv_starts.items():
        end = start + sum(1 for f in all_feats if f[3] == inv_lbl)
        mid = (start + end - 1) / 2
        inv = registry.inventories.get(inv_lbl.lower().replace(" ", "_"))
        color = inv.color if inv else "#333"
        ax.text(len(ops)+0.3, mid, inv_lbl, ha="left", va="center",
                fontsize=7.5, color=color, fontweight="bold")
    ax.set_title("RQ4 — Feature Facets × Opinion Leaf Spearman Correlations\n"
                 "(grouped by inventory/dimension; * p<.05 ** p<.01 · p<.10)", fontweight="bold")
    _save(fig, out_dir / "06b_facet_opinion_correlation_heatmap.png")


def plot_feature_contribution_heatmap(psi_df, out_dir, n_top=30):
    if psi_df is None:
        return
    contrib_cols = [c for c in psi_df.columns if c.startswith("contribution__")]
    if not contrib_cols:
        return
    if "susceptibility_index_pct" in psi_df.columns:
        srt = psi_df.sort_values("susceptibility_index_pct", ascending=False)
        half = n_top // 2
        sel = pd.concat([srt.head(half), srt.tail(half)]).drop_duplicates()
    else:
        sel = psi_df.head(n_top)
    mat = sel.set_index("profile_id")[contrib_cols]

    def _shorten(col):
        col = col.replace("contribution__profile_", "")
        col = col.replace("cont_big_five_", "")
        col = col.replace("cat__profile_cat_sex_", "Sex_")
        return "_".join(col.split("_")[:4])

    fig, ax = plt.subplots(figsize=(max(12, len(contrib_cols)*0.3+3),
                                    max(7, len(sel)*0.26+2)))
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    sns.heatmap(mat, ax=ax, cmap="RdBu", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=[_shorten(c) for c in contrib_cols], yticklabels=True,
                linewidths=0, cbar_kws={"label":"Feature Contribution", "shrink":0.6})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha="right", fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6.5)
    ax.axhline(len(sel)//2, color="yellow", lw=2.0, linestyle="--")
    ax.text(len(contrib_cols)//2, len(sel)//2-0.5,
            "↑ Most Susceptible  |  Most Resilient ↓",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round", fc="#333a", ec="none"))
    ax.set_title(f"Top {n_top} Profiles — Per-Feature CSI Ridge Contributions", fontweight="bold")
    _save(fig, out_dir / "06c_feature_contribution_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 07 — Gradient scatters (dynamic top-feature pairs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_scatters(df, registry, moderator_weights, out_dir):
    """
    Auto-select top feature pairs from moderator weights.
    One figure per pair, faceted by opinion.
    """
    top = [(col, lbl) for col, lbl in
           registry.top_features_by_weight(moderator_weights, n=8)
           if col in df.columns and df[col].nunique() > 5]
    if len(top) < 2:
        return
    # Generate consecutive pairs of top features
    pairs = [(top[i], top[i+1]) for i in range(min(3, len(top)-1))]
    ops = sorted(df["opinion_leaf_label"].unique())
    vmax = np.nanpercentile(np.abs(df[OUTCOME].dropna()), 95)

    for (xcol, xlbl), (ycol, ylbl) in pairs:
        fig, axes = plt.subplots(1, len(ops), figsize=(5*len(ops), 5), squeeze=False)
        for ci, op in enumerate(ops):
            ax = axes[0][ci]
            sub = df[df["opinion_leaf_label"]==op].dropna(subset=[xcol, ycol, OUTCOME])
            sc = ax.scatter(sub[xcol], sub[ycol], c=sub[OUTCOME], cmap=CMAP_AE,
                            vmin=-vmax, vmax=vmax, s=35, alpha=0.8, edgecolors="white", lw=0.3)
            plt.colorbar(sc, ax=ax, label="AE" if ci==len(ops)-1 else "", shrink=0.85)
            # density contour
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([sub[xcol], sub[ycol]])
                if xy.shape[1] > 10:
                    kde = gaussian_kde(xy)
                    xg = np.linspace(sub[xcol].min(), sub[xcol].max(), 35)
                    yg = np.linspace(sub[ycol].min(), sub[ycol].max(), 35)
                    XG, YG = np.meshgrid(xg, yg)
                    Z = kde(np.vstack([XG.ravel(), YG.ravel()])).reshape(XG.shape)
                    ax.contour(XG, YG, Z, levels=4, colors="white", linewidths=0.4, alpha=0.35)
            except Exception:
                pass
            ax.set_xlabel(xlbl, fontsize=9); ax.set_ylabel(ylbl if ci==0 else "", fontsize=9)
            ax.set_title(_short(op, 4), fontsize=9, fontweight="bold")
        fname = f"{xlbl[:12]}_{ylbl[:12]}".replace(" ","_").lower()
        fig.suptitle(f"RQ4 — {xlbl} × {ylbl} gradient scatter (fill=AE, contours=density)",
                     fontweight="bold")
        _save(fig, out_dir / f"07_{fname}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 08 — Susceptibility (CSI)
# ─────────────────────────────────────────────────────────────────────────────

def plot_csi_distribution(psi_df, out_dir):
    if psi_df is None or "susceptibility_index_pct" not in psi_df.columns:
        return
    vals = psi_df["susceptibility_index_pct"].dropna().values
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    n, bins, patches = ax.hist(vals, bins=25, edgecolor="white", density=True, alpha=0.75)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(RES_BLUE if left < 25 else ADV_RED if left > 75 else NEUTRAL_GREY)
    xg = np.linspace(0, 100, 300)
    ax.plot(xg, stats.gaussian_kde(vals, bw_method=0.35)(xg), color="#333", lw=2.0)
    for q, lbl, c in [(25,"Q1",RES_BLUE),(50,"Median",NEUTRAL_GREY),(75,"Q3",ADV_RED)]:
        ax.axvline(q, color=c, lw=1.2, linestyle="--"); ax.text(q+1, ax.get_ylim()[1]*0.85, lbl, fontsize=8, color=c)
    ax.set_xlabel("CSI Percentile"); ax.set_ylabel("Density")
    ax.set_title("CSI Distribution", fontweight="bold")
    sv = np.sort(vals); cdf = np.arange(1, len(sv)+1)/len(sv)*100
    ax2.plot(sv, cdf, color="#333", lw=2.2)
    ax2.fill_betweenx(cdf, sv, 100, where=(sv>75), alpha=0.15, color=ADV_RED)
    ax2.fill_betweenx(cdf, 0, sv, where=(sv<25), alpha=0.15, color=RES_BLUE)
    ax2.set_xlabel("CSI Percentile"); ax2.set_ylabel("Cum. % Profiles")
    ax2.set_title("CSI CDF", fontweight="bold")
    fig.suptitle("RQ1 — Conditional Susceptibility Index Distribution", fontweight="bold")
    _save(fig, out_dir / "08a_csi_distribution.png")


def plot_csi_vs_observed(psi_df, df, out_dir):
    if psi_df is None or "susceptibility_index_pct" not in psi_df.columns:
        return
    obs = df.groupby(["profile_id","opinion_leaf_label"])[OUTCOME].mean().reset_index()
    obs = obs.merge(psi_df[["profile_id","susceptibility_index_pct"]], on="profile_id", how="left")
    ops = sorted(df["opinion_leaf_label"].unique()); pal = _opinion_pal(ops)
    ncols = min(len(ops), 2); nrows = (len(ops)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 5*nrows), squeeze=False)
    for idx, op in enumerate(ops):
        ax = axes[idx//ncols][idx%ncols]
        sub = obs[obs["opinion_leaf_label"]==op].dropna(subset=["susceptibility_index_pct", OUTCOME])
        if len(sub) < 5: continue
        x, y = sub["susceptibility_index_pct"].values, sub[OUTCOME].values
        ax.scatter(x, y, c=pal[op], s=30, alpha=0.65, edgecolors="none")
        try:
            xg, yg = _lowess(x, y); xg2, lo, hi = _loess_ci(x, y)
            ax.plot(xg, yg, color=pal[op], lw=2.0)
            ax.fill_between(xg2, lo, hi, color=pal[op], alpha=0.18)
        except Exception: pass
        r, p = spearmanr(x, y)
        ax.axhline(0, color="#555", lw=0.8, linestyle="--")
        ax.set_xlabel("CSI Percentile"); ax.set_ylabel("Observed AE")
        ax.set_title(f"{_short(op,4)}  ρ={r:+.2f} ({_sig(p)})", fontsize=9)
    for idx in range(len(ops), nrows*ncols): axes[idx//ncols][idx%ncols].set_visible(False)
    fig.suptitle("RQ1 — CSI Predicted vs Observed AE (LOESS + 95% CI)", fontweight="bold")
    _save(fig, out_dir / "08b_csi_vs_observed.png")


def plot_forest_plot(moderator_weights, bootstrap_params, registry, out_dir):
    if moderator_weights is None or len(moderator_weights) == 0:
        return
    df = moderator_weights.copy().sort_values("normalized_weight_pct", ascending=True)
    ci_map = {}
    if bootstrap_params is not None:
        for _, row in bootstrap_params.iterrows():
            ci_map[row["term"]] = (row["conf_low"], row["conf_high"])

    ogs = df["ontology_group"].unique()
    og_pal = _assign_domain_colors(sorted(ogs))
    n = len(df)
    fig, ax = plt.subplots(figsize=(13, max(7, n*0.42+2)))
    y = np.arange(n)
    bar_colors = [ADV_RED if "higher" in str(d) else RES_BLUE if "lower" in str(d) else NEUTRAL_GREY
                  for d in (df["direction"].values if "direction" in df.columns else ["unknown"]*n)]

    # Background shading by group
    prev_g = None; shade = False
    for i, g in enumerate(df["ontology_group"]):
        if g != prev_g: shade = not shade; prev_g = g
        if shade: ax.axhspan(i-0.5, i+0.5, color="#f5f5f5", zorder=0)

    ax.barh(y, df["normalized_weight_pct"].abs(), color=bar_colors, alpha=0.72, edgecolor="white", lw=0.4, height=0.7)

    # CI lines where available
    for i, (_, row) in enumerate(df.iterrows()):
        term = row.get("term", "")
        if term in ci_map:
            lo, hi = ci_map[term]; est = row["weighted_mean_estimate"]
            w = row["normalized_weight_pct"]
            if abs(est) > 0:
                scale = abs(w) / abs(est)
                ax.plot([w+(lo-est)*scale, w+(hi-est)*scale], [i,i], color="#333", lw=1.5)

    ax.set_yticks(y); ax.set_yticklabels(df["moderator_label"], fontsize=7.5)
    ax.set_xlabel("Normalised |Weight| (%)"); ax.axvline(0, color="#333", lw=0.8)
    ax.set_title("RQ4 — Feature Forest Plot (red=↑susceptibility, blue=↑resilience, CI=bootstrap)",
                 fontweight="bold")

    # Group labels auto from ontology_group
    prev_g = None; gs_start = {}
    for i, g in enumerate(df["ontology_group"]):
        if g != prev_g: gs_start[g] = i; prev_g = g
    for g, start in gs_start.items():
        end = start + sum(1 for gg in df["ontology_group"] if gg == g)
        ax.text(ax.get_xlim()[1]*1.02, (start+end-1)/2, g,
                ha="left", va="center", fontsize=7, color=og_pal.get(g, "#333"), fontweight="bold")
    legend_handles = [mpatches.Patch(color=ADV_RED, label="→ Higher susceptibility"),
                      mpatches.Patch(color=RES_BLUE, label="→ Lower susceptibility")]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    _save(fig, out_dir / "08c_feature_forest_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _find(base: Path, *candidates: str) -> Optional[Path]:
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    return None


def run_all(run_dir: Path, visuals_root: Path) -> None:
    run_dir = Path(run_dir).resolve(); visuals_root = Path(visuals_root)

    sem_path = _find(run_dir,
        "stage_outputs/07_generate_research_visuals/data_snapshots/sem_long_encoded_snapshot.csv",
        "sem_long_encoded.csv")
    if sem_path is None:
        raise FileNotFoundError(f"sem_long CSV not found under {run_dir}")

    psi_path  = _find(run_dir, "stage_outputs/06_construct_structural_equation_model/profile_susceptibility_index.csv")
    mw_path   = _find(run_dir, "stage_outputs/06_construct_structural_equation_model/moderator_weight_table.csv")
    boot_path = _find(run_dir, "stage_outputs/06_construct_structural_equation_model/bootstrap_primary_params.csv")

    opinion_json = attack_json = None
    cfg_path = _find(run_dir, "config/pipeline_config.json")
    if cfg_path:
        cfg = json.loads(cfg_path.read_text())
        ont_root = cfg.get("ontology_root", "")
        if ont_root:
            op = Path(ont_root) / "OPINION" / "opinion.json"
            at = Path(ont_root) / "ATTACK"  / "attack.json"
            if op.exists(): opinion_json = json.loads(op.read_text())
            if at.exists(): attack_json  = json.loads(at.read_text())

    # Fallback: search
    for cand in [run_dir.parents[2] / "src/backend/ontology/separate/test/OPINION/opinion.json",
                 run_dir.parents[2] / "src/backend/ontology/separate/production/OPINION/opinion.json"]:
        if cand.exists() and opinion_json is None:
            opinion_json = json.loads(cand.read_text())

    print(f"  Loading {sem_path.name} ...")
    df = pd.read_csv(sem_path)
    if "opinion_leaf_label" not in df.columns and "opinion_leaf" in df.columns:
        df["opinion_leaf_label"] = df["opinion_leaf"].apply(lambda x: x.split(">")[-1].strip())
    if "baseline_extremity_norm" not in df.columns and "baseline_abs_score" in df.columns:
        df["baseline_extremity_norm"] = df["baseline_abs_score"] / 500.0

    # Build registry from actual data columns (zero hardcoding)
    registry = FeatureRegistry(df)
    print(f"  Feature registry:\n{registry.summary()}\n")
    # Decode categorical one-hot cols into convenience string cols once
    df = _add_categorical_cols(df, registry)

    psi_df  = pd.read_csv(psi_path)  if psi_path  else None
    mw_df   = pd.read_csv(mw_path)   if mw_path   else None
    boot_df = pd.read_csv(boot_path) if boot_path  else None

    leaf_domain: Dict[str, Tuple[str, str]] = {}
    if opinion_json:
        leaf_domain = _build_leaf_domain_map(opinion_json)
    for _, row in df.drop_duplicates("opinion_leaf_label").iterrows():
        lf = row["opinion_leaf_label"]
        if lf not in leaf_domain:
            leaf_domain[lf] = (row.get("opinion_domain", "Unknown"), row.get("opinion_leaf", lf))

    def D(n, name): return _mkdir(visuals_root / name)

    d1 = D(1, "01_state_space")
    d2 = D(2, "02_perturbation")
    d3 = D(3, "03_moderation")
    d4 = D(4, "04_distributions")
    d5 = D(5, "05_clustering")
    d6 = D(6, "06_heatmaps")
    d7 = D(7, "07_gradient")
    d8 = D(8, "08_susceptibility")

    print("  [1/8] State space ...")
    plot_opinion_hierarchy_panel(df, leaf_domain, d1)
    plot_attack_opinion_matrix(df, leaf_domain, d1)
    plot_state_space_pca(df, registry, d1)
    plot_perturbation_pca(df, d1)
    plot_state_space_transition(df, psi_df, d1)
    plot_attack_comparison_panel(df, d1)

    print("  [2/8] Perturbation ...")
    plot_delta_violins(df, d2)
    plot_baseline_extremity(df, d2)
    plot_ae_density_cdf(df, d2)
    plot_direction_effect(df, d2)

    print("  [3/8] Moderation ...")
    plot_feature_moderation_grid(df, registry, mw_df, d3)
    plot_categorical_violins(df, registry, d3)
    plot_top_pair_surface(df, registry, mw_df, d3)
    plot_hierarchical_r2(mw_df, d3)

    print("  [4/8] Distributions ...")
    plot_outcome_violin_panel(df, d4)
    plot_positive_rate_bars(df, leaf_domain, d4)

    print("  [5/8] Clustering + radar ...")
    plot_clustermap(df, registry, psi_df, d5)
    plot_radar_by_categorical_dim(df, registry, d5)
    plot_radar_susceptibility_quartile(df, psi_df, d5)

    print("  [6/8] Heatmaps ...")
    plot_profile_opinion_heatmap(df, psi_df, leaf_domain, d6)
    plot_facet_opinion_heatmap(df, registry, d6)
    plot_feature_contribution_heatmap(psi_df, d6)

    print("  [7/8] Gradient scatters ...")
    plot_gradient_scatters(df, registry, mw_df, d7)

    print("  [8/8] Susceptibility ...")
    plot_csi_distribution(psi_df, d8)
    plot_csi_vs_observed(psi_df, df, d8)
    plot_forest_plot(mw_df, boot_df, registry, d8)

    print(f"\n  All visualizations → {visuals_root}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",    required=True)
    parser.add_argument("--visuals-dir", required=True)
    args = parser.parse_args()
    run_all(Path(args.run_dir), Path(args.visuals_dir))
