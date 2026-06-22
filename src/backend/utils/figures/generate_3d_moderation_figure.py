"""
Generate figure_5_3d_susceptibility_isd:
3D bar chart of inter-individual standard deviation of adversarial effectivity across the
4 attack × 4 opinion factorial. Higher Z = more individual-difference variability in that cell
(stronger potential for moderation by profile features).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LONG_CSV = (
    PROJECT_ROOT
    / "evaluation" / "run_1" / "stage_outputs"
    / "05_compute_effectivity_deltas" / "sem_long_encoded.csv"
)
OUT_DIRS = [
    PROJECT_ROOT / "report" / "assets" / "figures",
    PROJECT_ROOT / "evaluation" / "run_1" / "publication_assets" / "figures",
]

ATTACK_SHORT = {
    "Misleading_Narrative_Framing":     "Misleading\nNarrative",
    "Fear_Appeal_Scapegoating_Post":     "Fear\nAppeal",
    "Astroturf_Comment_Wave":            "Astroturf\nWave",
    "Pseudo_Expert_Authority_Cue":       "Pseudo\nExpert",
}
OPINION_SHORT = {
    "Alliance_Commitment_Support":                 "Alliance\nCommitment",
    "Civil_Defense_Preparedness_Support":           "Civil\nDefense",
    "Domestic_Security_Powers_Expansion_Support":   "Domestic\nSecurity",
    "Strategic_Autonomy_Support":                   "Strategic\nAutonomy",
}

ATTACK_ORDER  = list(ATTACK_SHORT.keys())
OPINION_ORDER = list(OPINION_SHORT.keys())

df = pd.read_csv(LONG_CSV)

attack_col  = "attack_leaf_label"  if "attack_leaf_label"  in df.columns else "attack_leaf"
opinion_col = "opinion_leaf_label" if "opinion_leaf_label" in df.columns else "opinion_leaf"
ae_col      = "adversarial_effectivity"


def _last(s: str) -> str:
    return s.rsplit(" > ", 1)[-1] if isinstance(s, str) and " > " in s else str(s)


df["_atk"] = df[attack_col].apply(_last) if attack_col == "attack_leaf" else df[attack_col]
df["_op"]  = df[opinion_col] if opinion_col == "opinion_leaf_label" else df[opinion_col].apply(_last)

# Inter-individual standard deviation per cell
isd_matrix = (
    df.groupby(["_atk", "_op"])[ae_col]
    .std()
    .unstack("_op")
    .reindex(index=ATTACK_ORDER, columns=OPINION_ORDER)
)

# ── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7.5))
ax  = fig.add_subplot(111, projection="3d")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n_atk = len(ATTACK_ORDER)
n_op  = len(OPINION_ORDER)
x_ticks = np.arange(n_op)
y_ticks = np.arange(n_atk)
dx = dy = 0.66

# colormap: low ISD = cool blue, high ISD = warm red  (diverging = more drama)
cmap = cm.get_cmap("YlOrRd")
vals_flat = isd_matrix.values.flatten()
vmin, vmax = np.nanmin(vals_flat), np.nanmax(vals_flat)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

for i, atk in enumerate(ATTACK_ORDER):
    for j, op in enumerate(OPINION_ORDER):
        val = float(isd_matrix.loc[atk, op])
        color = cmap(norm(val))
        ax.bar3d(
            j - dx / 2, i - dy / 2, 0,
            dx, dy, val,
            color=color, alpha=0.88, edgecolor="white", linewidth=0.4,
            shade=True,
        )
        ax.text(
            j, i, val + 1.5,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=7.5, color="#222222",
        )

# ── axes ───────────────────────────────────────────────────────────────────
ax.set_xticks(x_ticks)
ax.set_xticklabels([OPINION_SHORT[o] for o in OPINION_ORDER], fontsize=8.5, ha="right", va="top")
ax.set_yticks(y_ticks)
ax.set_yticklabels([ATTACK_SHORT[a] for a in ATTACK_ORDER], fontsize=8.5, ha="left", va="center")
ax.set_zlabel("SD of adversarial effectivity across profiles", fontsize=9, labelpad=9)
ax.zaxis.set_tick_params(labelsize=7.5)

ax.view_init(elev=30, azim=-50)
ax.grid(True, alpha=0.22)

# colorbar proxy
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.55, aspect=18)
cbar.set_label("ISD of AE (profile variability)", fontsize=8.5)
cbar.ax.tick_params(labelsize=7.5)

ax.set_title(
    "Inter-individual Moderation Strength Across the 4 × 4 Attack–Opinion Factorial\n"
    "SD of Adversarial Effectivity  ·  Test run 1  ·  100 Profiles × 4 Attacks × 4 Opinions",
    fontsize=10, fontweight="bold", pad=14,
)

plt.tight_layout()

stem = "figure_5_3d_susceptibility_isd"
for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = out_dir / f"{stem}.{fmt}"
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")

plt.close()
print("3D moderation ISD figure done.")
