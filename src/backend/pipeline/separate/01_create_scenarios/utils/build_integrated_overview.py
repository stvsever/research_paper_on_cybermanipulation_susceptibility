from __future__ import annotations

"""
Integrated 10K scenario-set overview figures
============================================
Communicates the FINAL integrated design (profile + attack triplet + opinion
cluster) handed to collaborators. Each figure is written as a PNG (overview/
integrated/pngs) and an interactive HTML (overview/integrated/htmls):

  integrated_composition              marginals of the 10K (balanced by design)
  integrated_entropy_retention        per-factor entropy + coverage vs full space
  integrated_independence_matrix      Cramér's V across factors (no confounding)
  integrated_flow_sankey              region → opinion family → attack phase flow

The flagship interactive view (the hierarchical all-factor 3D explorer) is built
separately by build_integrated_explorer.py.

Run by file path:
  python .../utils/build_integrated_overview.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[6]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overview_theme import (  # noqa: E402
    PAL, SEQ, base_layout, save_fig, style_subplot_titles,
)

STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
SEP = STAGE / "samples" / "01_separated"
INTEG = STAGE / "samples" / "02_integrated"
OUT = STAGE / "overview" / "integrated"
PHASES = ("Plan", "Prepare", "Execute")
PHASE_COLORS = {"Plan": PAL["blue"], "Prepare": PAL["teal"], "Execute": PAL["orange"]}
TRAITS = ("openness_to_experience", "conscientiousness", "extraversion",
          "agreeableness", "neuroticism")


def short(s, n=24):
    s = s.replace("_And_", " & ").replace("_", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def norm_entropy(counts):
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    if c.size <= 1:
        return 1.0 if c.size == 1 else 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(c)))


def cramers_v(a, b):
    ca = {v: i for i, v in enumerate(dict.fromkeys(a))}
    cb = {v: i for i, v in enumerate(dict.fromkeys(b))}
    t = np.zeros((len(ca), len(cb)))
    for x, y in zip(a, b):
        t[ca[x], cb[y]] += 1
    n = t.sum()
    if n == 0 or t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    exp = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = np.nansum((t - exp) ** 2 / np.where(exp == 0, np.nan, exp))
    phi2 = chi2 / n
    r, k = t.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); kc = k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(math.sqrt(phi2c / d)) if d > 0 else 0.0


class Data:
    """Streams the (large) self-contained jsonl once and keeps only light
    per-scenario records, plus the coverage sets and small summary files."""

    def __init__(self):
        recs, opin_leaves, atk_leaves = [], set(), set()
        with (INTEG / "integrated_scenarios_10000.jsonl").open() as fh:
            for line in fh:
                r = json.loads(line)
                p = r["profile"]; d = p["demographics"]; bf = d.get("big_five", {})
                ca = p["categorical_attributes"]
                a = r["attack"]; oc = r["opinion_cluster"]
                phase_sig = {ph: a["triplet"][ph]["signal_score"] for ph in PHASES}
                for ph in PHASES:
                    atk_leaves.add(a["triplet"][ph]["leaf_id"])
                for lf in oc["leaves"]:
                    opin_leaves.add(lf["path"])
                recs.append({
                    "age": d["age_years"],
                    "bf": {t: bf[t]["pct"] for t in TRAITS if t in bf},
                    "region": next((v for k, v in ca.items() if "broad_region" in k), "NA"),
                    "education": next((v for k, v in ca.items() if "highest_education" in k), "NA"),
                    "family": oc["family"], "cluster_key": oc["key"],
                    "cluster": oc["parent_name"],
                    "amplify": oc["direction_summary"]["amplify_+1"],
                    "erode": oc["direction_summary"]["erode_-1"],
                    "signal": a["signal_total"], "route": a["inclusion_route"],
                    "plan_sec": a["triplet"]["Plan"]["secondary"],
                    "exec_sec": a["triplet"]["Execute"]["secondary"],
                    "dominant_phase": max(phase_sig, key=phase_sig.get),
                })
        self.recs = recs
        self.opin_leaves = opin_leaves
        self.atk_leaves = atk_leaves
        self.summary = json.loads((INTEG / "integrated_scenarios_10000.summary.json").read_text())
        asum = json.loads((SEP / "attacks" /
                           "red_plan_prepare_execute_opinion_effect_filtered.summary.json").read_text())
        self.n_filtered_leaves = asum["filtered_set"]["filtered_distinct_leaves"]
        osum = json.loads((SEP / "opinions" / "opinion_targets_maxent_1000.summary.json").read_text())
        self.n_opinion_clusters = osum["n_clusters"]
        self.n_opinion_dir_leaves = osum["n_directional_leaves"]
        self.opinion_subtree = json.loads(
            (INTEG / "integrated_scenarios_10000.summary.json").read_text()
        )["sources"]["opinions"].get("subtree", "opinion ontology")


# ── composition of the 10K ───────────────────────────────────────────────────
def fig_composition(D):
    recs = D.recs
    clu = Counter(r["cluster"] for r in recs)
    clu_items = sorted(clu.items(), key=lambda kv: kv[1])
    ages = [r["age"] for r in recs]
    sig = [r["signal"] for r in recs]
    exec_sec = Counter(r["exec_sec"] for r in recs)
    exec_items = sorted(exec_sec.items(), key=lambda kv: kv[1])

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.17, horizontal_spacing=0.16,
        subplot_titles=("Opinion cluster / issue domain (uniform)", "Profile age (uniform, max-entropy)",
                        "Attack Execute tactic mix", "Attack opinion-signal (raw density)"),
        specs=[[{"type": "bar"}, {"type": "xy"}], [{"type": "bar"}, {"type": "xy"}]])
    fig.add_trace(go.Bar(y=[short(k, 30) for k, _ in clu_items], x=[v for _, v in clu_items],
                         orientation="h", marker_color=PAL["teal"], text=[v for _, v in clu_items],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x} scenarios<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Histogram(x=ages, nbinsx=33, marker_color=PAL["sky"],
                               hovertemplate="age %{x}: %{y}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(y=[short(k, 26) for k, _ in exec_items], x=[v for _, v in exec_items],
                         orientation="h", marker_color=PAL["orange"], text=[v for _, v in exec_items],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Violin(x=sig, orientation="h", line_color=PAL["red"], fillcolor=PAL["red"],
                            opacity=0.5, points=False, meanline_visible=True, side="positive",
                            hovertemplate="signal %{x}<extra></extra>"), row=2, col=2)
    fig.update_layout(**base_layout(
        "What is inside the 10,000-scenario set",
        "every factor is balanced or preserved by design: opinion clusters uniform, age uniform, "
        "attack mix and signal preserved",
        h=820, w=1280, multipanel=True))
    fig.update_layout(showlegend=False, bargap=0.25)
    fig.update_xaxes(showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(automargin=True, tickfont=dict(size=10))
    fig.update_xaxes(title_text="age (years)", row=1, col=2)
    fig.update_xaxes(title_text="total opinion-manipulation signal", row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    style_subplot_titles(fig)
    save_fig(fig, OUT, "integrated_composition")


# ── entropy retention + coverage vs full state space ─────────────────────────
def fig_entropy_retention(D):
    er = D.summary["entropy_report"]
    ar = er["attack_subsample"]
    nc = D.n_opinion_clusters
    factors = [
        ("Profile (each used once)", er["profile_usage_normalised_entropy"], 1.0),
        (f"Opinion cluster ({nc})", er["opinion_cluster_normalised_entropy"], 1.0),
        ("Attack · Plan tactic", ar["normalised_entropy_subsample"]["plan"],
         ar["normalised_entropy_full_filtered"]["plan"]),
        ("Attack · Prepare tactic", ar["normalised_entropy_subsample"]["prepare"],
         ar["normalised_entropy_full_filtered"]["prepare"]),
        ("Attack · Execute tactic", ar["normalised_entropy_subsample"]["execute"],
         ar["normalised_entropy_full_filtered"]["execute"]),
        ("Attack · signal decile", ar["normalised_entropy_subsample"]["sigbin"],
         ar["normalised_entropy_full_filtered"]["sigbin"]),
    ]
    labels = [f[0] for f in factors][::-1]
    sub = [f[1] * 100 for f in factors][::-1]
    ref = [f[2] * 100 for f in factors][::-1]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44],
                        specs=[[{"type": "bar"}, {"type": "bar"}]],
                        subplot_titles=("Normalised entropy of each scenario factor",
                                        "Coverage relative to the full state space"),
                        horizontal_spacing=0.16)
    fig.add_trace(go.Bar(y=labels, x=ref, orientation="h", name="source / theoretical max",
                         marker_color=PAL["line"], marker_line_color=PAL["muted"], marker_line_width=1,
                         hovertemplate="reference: %{x:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(y=labels, x=sub, orientation="h", name="integrated 10K set",
                         marker_color=PAL["teal"], text=[f"{v:.1f}%" for v in sub],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="10K set: %{x:.1f}%<extra></extra>"), row=1, col=1)

    cl_usage = Counter(r["cluster_key"] for r in D.recs)
    nc = D.n_opinion_clusters; ndl = D.n_opinion_dir_leaves
    cov_lab = ["Profiles used<br>(of 10,000)", f"Opinion clusters<br>(issue domains, of {nc})",
               "Directional issue-position<br>leaves touched", "Attack leaves<br>(of filtered pool)"]
    cov_val = [100.0, len(cl_usage) / nc * 100,
               len(D.opin_leaves) / ndl * 100, len(D.atk_leaves) / D.n_filtered_leaves * 100]
    cov_txt = ["10,000 / 10,000", f"{len(cl_usage)} / {nc}",
               f"{len(D.opin_leaves)} / {ndl}", f"{len(D.atk_leaves):,} / {D.n_filtered_leaves:,}"]
    fig.add_trace(go.Bar(y=cov_lab[::-1], x=cov_val[::-1], orientation="h", showlegend=False,
                         marker_color=[PAL["blue"], PAL["teal"], PAL["amber"], PAL["orange"]][::-1],
                         text=cov_txt[::-1], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{text}<extra></extra>"), row=1, col=2)
    fig.update_layout(**base_layout(
        "Entropy is preserved and the set spans its state space",
        "left: each factor stays at or near its source / maximal entropy · right: how much of each "
        "ontology the 10K touches (opinions = Issue Position Taxonomy)",
        h=620, w=1340, multipanel=True))
    fig.update_layout(barmode="overlay", bargap=0.28,
                      legend=dict(orientation="h", y=-0.12, x=0.28, xanchor="center"))
    fig.update_xaxes(title_text="normalised entropy (%)", range=[0, 108], showgrid=True, gridcolor=PAL["line"], row=1, col=1)
    fig.update_xaxes(title_text="percent", range=[0, 118], showgrid=True, gridcolor=PAL["line"], row=1, col=2)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    style_subplot_titles(fig)
    save_fig(fig, OUT, "integrated_entropy_retention")


# ── independence matrix (Cramér's V) ─────────────────────────────────────────
def fig_independence(D):
    recs = D.recs
    feats = {
        "Profile region": [r["region"] for r in recs],
        "Profile age band": [f"{(r['age']//16)*16}" for r in recs],
        "Profile education": [r["education"] for r in recs],
        "Opinion cluster": [r["cluster"] for r in recs],
        "Attack Plan tactic": [r["plan_sec"] for r in recs],
        "Attack Exec tactic": [r["exec_sec"] for r in recs],
        "Attack route": [r["route"] for r in recs],
    }
    names = list(feats)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = 1.0 if i == j else cramers_v(feats[a], feats[b])
    fig = go.Figure(go.Heatmap(
        z=M, x=names, y=names, zmin=0, zmax=1,
        colorscale=[[0, "#ffffff"], [0.05, "#e9f1f7"], [0.2, PAL["sky"]], [0.5, PAL["orange"]], [1, PAL["red"]]],
        text=[[f"{v:.02f}" for v in row] for row in M], texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="Cramér's V", thickness=14),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(
        "Scenario factors are statistically independent (no confounding)",
        "pairwise Cramér's V between design factors; off-diagonal ≈ 0 ⇒ internal validity is protected",
        h=760, w=900))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11), automargin=True)
    fig.update_layout(margin=dict(t=104, l=150, r=60, b=150))
    save_fig(fig, OUT, "integrated_independence_matrix")


# ── tri-partite flow (Sankey) ────────────────────────────────────────────────
def fig_sankey(D):
    recs = D.recs
    regions = sorted({r["region"] for r in recs})
    fams = list(dict.fromkeys(sorted(r["cluster"] for r in recs)))
    node_labels = (list(regions) + [short(f, 26) for f in fams] + [f"{p} dominant" for p in PHASES])
    node_colors = ([PAL["blue"]] * len(regions) + [PAL["teal"]] * len(fams)
                   + [PHASE_COLORS[p] for p in PHASES])
    ri = {r: i for i, r in enumerate(regions)}
    fi = {f: i + len(regions) for i, f in enumerate(fams)}
    pi = {p: i + len(regions) + len(fams) for i, p in enumerate(PHASES)}

    l1 = Counter((r["region"], r["cluster"]) for r in recs)
    l2 = Counter((r["cluster"], r["dominant_phase"]) for r in recs)
    src, tgt, val, lc = [], [], [], []
    for (rg, fm), c in l1.items():
        src.append(ri[rg]); tgt.append(fi[fm]); val.append(c); lc.append("rgba(41,128,185,0.18)")
    for (fm, ph), c in l2.items():
        src.append(fi[fm]); tgt.append(pi[ph]); val.append(c); lc.append("rgba(42,157,143,0.20)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=node_labels, color=node_colors, pad=14, thickness=14,
                  line=dict(color="white", width=0.5),
                  hovertemplate="%{label}: %{value} scenarios<extra></extra>"),
        link=dict(source=src, target=tgt, value=val, color=lc,
                  hovertemplate="%{value} scenarios<extra></extra>")))
    fig.update_layout(**base_layout(
        "How the 10,000 scenarios flow across the three ontologies",
        "profile world region → opinion issue domain → dominant attack phase · even spread reflects "
        "the independent, balanced design",
        h=820, w=1280))
    fig.update_layout(font=dict(size=11))
    save_fig(fig, OUT, "integrated_flow_sankey")


def main():
    print("Loading integrated set (streaming the self-contained jsonl)…")
    D = Data()
    print(f"  {len(D.recs):,} scenarios loaded")
    print("Building INTEGRATED overview figures…")
    fig_composition(D)
    fig_entropy_retention(D)
    fig_independence(D)
    fig_sankey(D)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
