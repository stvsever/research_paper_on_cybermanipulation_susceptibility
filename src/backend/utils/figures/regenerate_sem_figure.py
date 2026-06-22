"""
Regenerate figure_4_annotated_sem_path_diagram WITHOUT the Note. text at the bottom.
Reads from run_1 stage output CSVs directly.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE6 = PROJECT_ROOT / "evaluation" / "run_1" / "stage_outputs" / "06_construct_structural_equation_model"
SEM_CSV = STAGE6 / "sem_coefficients.csv"
EXPL_CSV = STAGE6 / "exploratory_moderator_comparison.csv"

OUT_DIRS = [
    PROJECT_ROOT / "report" / "assets" / "figures",
    PROJECT_ROOT / "evaluation" / "run_1" / "publication_assets" / "figures",
]

INDICATOR_ORDER = [
    "adversarial_delta_indicator__alliance_commitment_support",
    "adversarial_delta_indicator__civil_defense_preparedness_support",
    "adversarial_delta_indicator__domestic_security_powers_expansion_support",
    "adversarial_delta_indicator__strategic_autonomy_support",
]

INDICATOR_LABELS = [
    "Alliance\nCommitment Support",
    "Civil Defense\nPreparedness Support",
    "Domestic Security\nPowers Expansion",
    "Strategic\nAutonomy Support",
]

def _pretty(s: str) -> str:
    return s.replace("_", " ").replace("  ", " ").strip().title()

def _wrap(s: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(s, width))

sem = pd.read_csv(SEM_CSV)
expl = pd.read_csv(EXPL_CSV)

sem = sem.loc[sem["op"].astype(str) == "~"].copy()
sem["estimate"] = pd.to_numeric(sem["estimate"], errors="coerce")
sem["p_value"]  = pd.to_numeric(sem["p_value"],  errors="coerce")
sem = sem.loc[sem["lhs"].isin(INDICATOR_ORDER) & sem["estimate"].notna() & np.isfinite(sem["estimate"])]

heatmap_df = sem.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")

# row order: by normalized weight if available
if "normalized_weight_pct" in expl.columns and "moderator_label" in expl.columns:
    ordered_rows = (
        expl.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
    )
    ordered_rows = [r for r in ordered_rows if r in heatmap_df.index]
    heatmap_df = heatmap_df.loc[ordered_rows]

# keep only indicator columns that exist
available = [c for c in INDICATOR_ORDER if c in heatmap_df.columns]
heatmap_df = heatmap_df[available]

# annotation with stars
annot = heatmap_df.copy().astype(str)
for rhs in heatmap_df.index:
    for lhs in available:
        row = sem.loc[(sem["lhs"] == lhs) & (sem["rhs"] == rhs)]
        if row.empty:
            annot.loc[rhs, lhs] = ""
            continue
        first = row.iloc[0]
        p = first["p_value"]
        stars = "***" if pd.notna(p) and p < 0.001 else "**" if pd.notna(p) and p < 0.01 else "*" if pd.notna(p) and p < 0.05 else "†" if pd.notna(p) and p < 0.10 else ""
        annot.loc[rhs, lhs] = f"{first['estimate']:.2f}{stars}"

fig, ax = plt.subplots(figsize=(12.6, 7.4))
fig.patch.set_facecolor("white")

x_labels = []
for c in available:
    idx = INDICATOR_ORDER.index(c)
    x_labels.append(INDICATOR_LABELS[idx])

sns.heatmap(
    heatmap_df,
    cmap="RdBu_r",
    center=0.0,
    linewidths=0.6,
    linecolor="white",
    annot=annot,
    fmt="",
    cbar_kws={"label": "SEM path coefficient"},
    ax=ax,
)
ax.set_title("Path-SEM coefficients from profile moderators to attacked opinion shifts", fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Repeated attacked opinion outcome", fontsize=10.5, labelpad=8)
ax.set_ylabel("Profile moderator", fontsize=10.5)
ax.set_yticklabels([_wrap(_pretty(r), 24) for r in heatmap_df.index], rotation=0, fontsize=9.5)
ax.set_xticklabels(x_labels, rotation=18, ha="right", fontsize=9.5)

plt.tight_layout()

stem = "figure_4_annotated_sem_path_diagram"
for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = out_dir / f"{stem}.{fmt}"
        plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p}")

plt.close()
print("SEM figure (no note) done.")
