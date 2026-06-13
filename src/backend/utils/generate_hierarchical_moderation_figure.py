"""
Standalone script: generate supplementary_figure_s5_hierarchical_moderation.
Saves PDF + PNG to research_report/assets/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "research_report" / "assets" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
C_SIG     = "#1a6896"   # FDR-significant — deep blue
C_GROUP   = "#444444"   # aggregate group  — dark grey
C_MARG    = "#6aabcf"   # marginal / moderate importance — mid blue
C_NS      = "#c0c0c0"   # not significant / small — light grey

# ── data ─────────────────────────────────────────────────────────────────────
# Columns: (display_label, importance_pct, bar_color, q_fdr_label, indent)
entries = [
    # ─── group-level ─────────────────────────────────
    ("Big Five Personality",        32.0, C_GROUP, "",              0),
    ("Demographics",                28.8, C_GROUP, "",              0),
    # ─── Big Five trait decomposition ────────────────
    ("  Conscientiousness",         23.1, C_SIG,   " q = .030*",   1),
    ("  Extraversion",               7.1, C_MARG,  "",              1),
    ("  Openness",                   3.8, C_NS,    "",              1),
    ("  Neuroticism",                3.0, C_NS,    "",              1),
    ("  Agreeableness",              2.2, C_NS,    "",              1),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.8, 4.4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

labels    = [e[0] for e in entries]
values    = [e[1] for e in entries]
colors    = [e[2] for e in entries]
q_labels  = [e[3] for e in entries]
indents   = [e[4] for e in entries]

n = len(entries)
y_pos = np.arange(n)[::-1]   # top-to-bottom reading order

bars = ax.barh(
    y_pos, values,
    color=colors, height=0.52,
    edgecolor="white", linewidth=0.5,
    zorder=3,
)

# ── value + q-value labels ────────────────────────────────────────────────────
for bar, val, ql, col in zip(bars, values, q_labels, colors):
    yi = bar.get_y() + bar.get_height() / 2
    label = f"{val:.1f}%{ql}"
    ax.text(
        val + 0.5, yi, label,
        va="center", ha="left",
        fontsize=8.5,
        color=col if col != C_NS else "#888888",
        fontweight="bold" if ql else "normal",
    )

# ── separator between group-level and trait-level blocks ─────────────────────
# Draw a subtle horizontal line below "Demographics" (y_pos index 5 = indented items start)
# Group rows: indices 0 and 1 → y_pos[-1] and y_pos[-2]  (top two bars)
# Trait rows: indices 2-6 → remainder
sep_y = (y_pos[1] + y_pos[2]) / 2   # between demographics and conscientiousness
ax.axhline(sep_y, color="#dddddd", linewidth=0.8, zorder=2)

# ── y-axis ────────────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.tick_params(axis="y", length=0, pad=4)

# Bold the group-level labels
for tick, indent in zip(ax.get_yticklabels(), indents):
    tick.set_fontweight("bold" if indent == 0 else "normal")
    tick.set_color(C_GROUP if indent == 0 else "#333333")

# ── x-axis ───────────────────────────────────────────────────────────────────
ax.set_xlabel("Relative importance (%)", fontsize=9.5)
ax.set_xlim(0, 44)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#aaaaaa")
ax.tick_params(axis="x", color="#aaaaaa", labelsize=8.5)
ax.grid(axis="x", color="#eeeeee", linewidth=0.6, zorder=0)

# ── section annotations ───────────────────────────────────────────────────────
ax.text(
    43.5, (y_pos[0] + y_pos[1]) / 2,
    "Group\nlevel",
    va="center", ha="right", fontsize=7.5, color="#888888", style="italic",
)
ax.text(
    43.5, (y_pos[2] + y_pos[-1]) / 2,
    "Within\nBig Five",
    va="center", ha="right", fontsize=7.5, color="#888888", style="italic",
)

# ── legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_GROUP,  label="Aggregate group"),
    mpatches.Patch(facecolor=C_SIG,    label="FDR significant (q < .05)"),
    mpatches.Patch(facecolor=C_MARG,   label="Moderate importance"),
    mpatches.Patch(facecolor=C_NS,     label="Low importance / n.s."),
]
ax.legend(
    handles=legend_elements,
    loc="lower right", fontsize=7.5, frameon=False,
    bbox_to_anchor=(1.0, 0.0),
)

# ── footnote ──────────────────────────────────────────────────────────────────
fig.text(
    0.01, -0.03,
    (
        "* Benjamini–Hochberg FDR corrected q-value from the multivariate profile-moderator model. "
        "Relative importance = |marginal CV-R²| / Σ|marginal CV-R²| × 100 "
        "(leave-one-group-out decomposition of the conditional susceptibility index)."
    ),
    fontsize=6.8, color="#666666", style="italic", wrap=True,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ── save ──────────────────────────────────────────────────────────────────────
stem = "supplementary_figure_s5_hierarchical_moderation"
for fmt in ("pdf", "png"):
    out = OUTPUT_DIR / f"{stem}.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")

plt.close()
print("Done.")
