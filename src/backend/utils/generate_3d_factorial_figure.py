"""
Generate figure_5_3d_factorial_effectivity:
A 3D bar chart of mean adversarial effectivity across the 4 attack × 4 opinion factorial.
Saved to both research_report/assets/figures/ and evaluation/tests/run_1/publication_assets/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D axes
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LONG_CSV = PROJECT_ROOT / "evaluation" / "run_1" / "stage_outputs" / "05_compute_effectivity_deltas" / "sem_long_encoded.csv"

OUT_DIRS = [
    PROJECT_ROOT / "research_report" / "assets" / "figures",
    PROJECT_ROOT / "evaluation" / "run_1" / "publication_assets" / "figures",
]

# ── short labels ─────────────────────────────────────────────────────────────
ATTACK_SHORT = {
    "Misleading_Narrative_Framing":     "Misleading\nNarrative",
    "Fear_Appeal_Scapegoating_Post":     "Fear\nAppeal",
    "Astroturf_Comment_Wave":           "Astroturf\nWave",
    "Pseudo_Expert_Authority_Cue":      "Pseudo\nExpert",
}
OPINION_SHORT = {
    "Alliance_Commitment_Support":                "Alliance\nCommitment",
    "Civil_Defense_Preparedness_Support":          "Civil\nDefense",
    "Domestic_Security_Powers_Expansion_Support":  "Domestic\nSecurity",
    "Strategic_Autonomy_Support":                  "Strategic\nAutonomy",
}

ATTACK_ORDER   = list(ATTACK_SHORT.keys())
OPINION_ORDER  = list(OPINION_SHORT.keys())

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(LONG_CSV)

# identify attack and opinion columns
attack_col  = "attack_leaf_label"  if "attack_leaf_label"  in df.columns else "attack_leaf"
opinion_col = "opinion_leaf_label" if "opinion_leaf_label" in df.columns else "opinion_leaf"
ae_col      = "adversarial_effectivity"

def _last_segment(s: str) -> str:
    if isinstance(s, str) and " > " in s:
        return s.rsplit(" > ", 1)[-1]
    return str(s)

if attack_col == "attack_leaf":
    df["_atk"] = df[attack_col].apply(_last_segment)
else:
    df["_atk"] = df[attack_col]

if opinion_col == "opinion_leaf_label":
    df["_op"] = df[opinion_col]
else:
    df["_op"] = df[opinion_col].apply(_last_segment)

matrix = (
    df.groupby(["_atk", "_op"])[ae_col]
    .mean()
    .unstack("_op")
    .reindex(index=ATTACK_ORDER, columns=OPINION_ORDER)
)

# ── 3D figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.5, 7.5))
ax  = fig.add_subplot(111, projection="3d")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n_atk  = len(ATTACK_ORDER)
n_op   = len(OPINION_ORDER)

x_ticks = np.arange(n_op)
y_ticks = np.arange(n_atk)

dx = dy = 0.65

POS_COLOR = "#1a6896"   # positive AE — deep blue
NEG_COLOR = "#c0392b"   # negative AE — red

for i, atk in enumerate(ATTACK_ORDER):
    for j, op in enumerate(OPINION_ORDER):
        val = float(matrix.loc[atk, op])
        color = POS_COLOR if val >= 0 else NEG_COLOR
        zbot  = min(val, 0)
        zht   = abs(val)
        ax.bar3d(
            j - dx / 2,
            i - dy / 2,
            zbot,
            dx, dy, zht,
            color=color, alpha=0.82, edgecolor="white", linewidth=0.4,
            shade=True,
        )
        ax.text(
            j, i, val + (2.5 if val >= 0 else -5),
            f"{val:.1f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=7, color="#333333",
        )

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x_ticks)
ax.set_xticklabels(
    [OPINION_SHORT[o] for o in OPINION_ORDER],
    fontsize=8, ha="right", va="top",
)
ax.set_yticks(y_ticks)
ax.set_yticklabels(
    [ATTACK_SHORT[a] for a in ATTACK_ORDER],
    fontsize=8, ha="left", va="center",
)
ax.set_zlabel("Mean adversarial effectivity", fontsize=9, labelpad=8)
ax.zaxis.set_tick_params(labelsize=7.5)

# zero plane
z0 = 0
ax.plot_surface(
    np.array([[x_ticks[0]-0.5, x_ticks[-1]+0.5], [x_ticks[0]-0.5, x_ticks[-1]+0.5]]),
    np.array([[y_ticks[0]-0.5, y_ticks[0]-0.5], [y_ticks[-1]+0.5, y_ticks[-1]+0.5]]),
    np.zeros((2, 2)),
    alpha=0.10, color="#888888",
)

ax.view_init(elev=28, azim=-55)
ax.grid(True, alpha=0.25)

# ── legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=POS_COLOR, label="Positive AE (attack succeeded)"),
    mpatches.Patch(facecolor=NEG_COLOR, label="Negative AE (backfire / resistance)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8, frameon=False)

ax.set_title(
    "Adversarial Effectivity Across the 4 × 4 Attack–Opinion Factorial\n"
    "Test run 1 — 100 Profiles × 4 Attacks × 4 Opinions",
    fontsize=10, fontweight="bold", pad=14,
)

plt.tight_layout()

# ── save ──────────────────────────────────────────────────────────────────────
stem = "figure_5_3d_factorial_effectivity"
for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = out_dir / f"{stem}.{fmt}"
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")

plt.close()
print("3D figure done.")
