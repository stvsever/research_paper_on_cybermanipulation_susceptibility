"""
Regenerate figure_2_absolute_delta_distribution as gradient-density violin + jitter scatter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LONG_CSV = (
    PROJECT_ROOT
    / "evaluation" / "run_1" / "stage_outputs"
    / "05_compute_effectivity_deltas" / "sem_long_encoded.csv"
)
OUT_DIRS = [
    PROJECT_ROOT / "research_report" / "assets" / "figures",
    PROJECT_ROOT / "evaluation" / "run_1" / "publication_assets" / "figures",
]

OPINION_ORDER = [
    "Alliance_Commitment_Support",
    "Civil_Defense_Preparedness_Support",
    "Domestic_Security_Powers_Expansion_Support",
    "Strategic_Autonomy_Support",
]
OPINION_LABELS = [
    "Alliance\nCommitment\nSupport",
    "Civil Defense\nPreparedness\nSupport",
    "Domestic Security\nPowers Expansion\nSupport",
    "Strategic\nAutonomy\nSupport",
]
COLORS = ["#1a6896", "#2e9dc7", "#c85a2a", "#e8961f"]

df = pd.read_csv(LONG_CSV)
leaf_col = "opinion_leaf_label" if "opinion_leaf_label" in df.columns else "opinion_leaf"

fig, ax = plt.subplots(figsize=(11.5, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafbfe")

positions = np.arange(len(OPINION_ORDER))

for i, (op, label, col) in enumerate(zip(OPINION_ORDER, OPINION_LABELS, COLORS)):
    data = df[df[leaf_col] == op]["abs_delta_score"].dropna().values

    # violin
    parts = ax.violinplot(
        [data], positions=[i], widths=0.72,
        showmedians=True, showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.9)
    parts["cmedians"].set_color("#111111")
    parts["cmedians"].set_linewidth(2.2)

    # IQR bar
    q25, q75 = np.percentile(data, [25, 75])
    ax.plot([i, i], [q25, q75], color=col, linewidth=4, alpha=0.85, solid_capstyle="round", zorder=4)

    # jitter scatter
    rng = np.random.RandomState(42 + i)
    jitter = rng.uniform(-0.17, 0.17, len(data))
    ax.scatter(
        np.full_like(data, float(i)) + jitter,
        data,
        color=col, alpha=0.40, s=22, zorder=5,
        edgecolors="white", linewidths=0.4,
    )

    # mean diamond
    ax.scatter(
        [i], [data.mean()],
        marker="D", color="white", edgecolors=col,
        s=55, zorder=6, linewidths=1.6,
    )

ax.set_xticks(positions)
ax.set_xticklabels(OPINION_LABELS, fontsize=10.5)
ax.set_ylabel("Absolute post–baseline opinion shift", fontsize=11.5)
ax.set_xlabel("")
ax.set_title(
    "Absolute attacked opinion shift by opinion leaf\n"
    "100 profiles × 4 attack vectors per leaf  ·  diamond = mean  ·  bar = IQR",
    fontsize=11.5, fontweight="bold", pad=12,
)
ax.grid(axis="y", alpha=0.28, linestyle="--", color="#9aaac8")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

legend_elements = [
    mpatches.Patch(facecolor=c, alpha=0.7, label=l.replace("\n", " "))
    for c, l in zip(COLORS, OPINION_LABELS)
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=True, framealpha=0.85)

plt.tight_layout()

stem = "figure_2_absolute_delta_distribution"
for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = out_dir / f"{stem}.{fmt}"
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")

plt.close()
print("Violin figure done.")
