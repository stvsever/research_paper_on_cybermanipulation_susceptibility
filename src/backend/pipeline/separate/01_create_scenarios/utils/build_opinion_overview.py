from __future__ import annotations

"""
OPINION sample-set overview figures
===================================
Visual overview of the directional opinion-target panel and its relation to the
OPINION Issue_Position_Taxonomy subtree (the issue positions an adversary tries
to shift), scoped to that subtree's issue domains:

  opinion_ontology_sunburst.html/.png    full state space, rings family→…→leaf
  opinion_state_space_treemap.png         treemap of the state space by family
  opinion_direction_sunburst.html/.png    radial map: leaves coloured by the baked
                                           adversarial direction (+1 amplify / −1 erode)
  opinion_direction_landscape.png         diverging amplify/erode bars per family
  opinion_sampling_structure.png          two-layer max-entropy + cluster allocation
  opinion_coverage_vs_statespace.png      sample ↔ full state-space coverage cascade

Run by file path:
  python .../utils/build_opinion_overview.py
"""

import json
import sys
from collections import Counter, defaultdict
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
    PAL, SEQ, FONT, DIR_COLORS, DIR_LABELS, base_layout, save_fig,
    style_subplot_titles, walk_hierarchy, domain_color_map, is_branch, leaf_count,
)

STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
ONT = PROJECT_ROOT / "src" / "backend" / "ontology" / "separate" / "production" / "OPINION" / "opinion.json"
SAMPLES = STAGE / "samples" / "01_separated" / "opinions"
OUT = STAGE / "overview" / "separate" / "opinions"

NEUTRAL_LEAF = "#c4cdde"
INNER = "#eef2f8"


def short(name: str, n: int = 26) -> str:
    s = name.replace("_And_", " & ").replace("_", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ── collect leaves with direction ────────────────────────────────────────────
def families(ont):
    return [k for k in ont if is_branch(k)]


def collect_leaves(ont):
    """Return list of (family, direction) for every terminal opinion leaf."""
    out = []

    def walk(node, fam):
        kids = [k for k in node if is_branch(k)]
        if not kids:
            d = node.get("adversarial_direction", 0) if isinstance(node, dict) else 0
            out.append((fam, int(d or 0)))
            return
        for k in kids:
            walk(node[k], fam)

    for f in families(ont):
        walk(ont[f], f)
    return out


# ── 1. full-hierarchy sunburst ───────────────────────────────────────────────
def fig_sunburst(ont):
    root = {k: ont[k] for k in families(ont)}
    h = walk_hierarchy(root, "ISSUE POSITIONS", max_depth=3)
    cmap = domain_color_map(h["domains"])
    colors = ["#ffffff" if d == "root" else cmap[d] for d in h["domains"]]
    fig = go.Figure(go.Sunburst(
        ids=h["ids"], parents=h["parents"], labels=h["labels"], values=h["values"],
        branchvalues="total", marker=dict(colors=colors, line=dict(color="white", width=1.2)),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value} leaf states<br>%{percentRoot:.1%} of state space<extra></extra>",
        maxdepth=3))
    total = leaf_count(root)
    fig.update_layout(**base_layout(
        "OPINION targets: Issue Position Taxonomy state space",
        f"{total:,} issue-position leaves · rings: issue domain → sub-issue → position · arc ∝ leaf count",
        h=820, w=920))
    save_fig(fig, OUT, "opinion_ontology_sunburst", html=True)


# ── 2. treemap by family ─────────────────────────────────────────────────────
def fig_treemap(ont):
    root = {k: ont[k] for k in families(ont)}
    h = walk_hierarchy(root, "ISSUE POSITIONS", max_depth=2)
    cmap = domain_color_map(h["domains"])
    colors = ["#ffffff" if d == "root" else cmap[d] for d in h["domains"]]
    fig = go.Figure(go.Treemap(
        ids=h["ids"], parents=h["parents"], labels=h["labels"], values=h["values"],
        branchvalues="total", marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+value", tiling=dict(pad=3),
        hovertemplate="<b>%{label}</b><br>%{value} leaf states (%{percentRoot:.1%})<extra></extra>"))
    fig.update_layout(**base_layout(
        "OPINION targets state space by issue domain",
        "tile area ∝ number of opinion leaves in the deployment ontology", h=620, w=1180, t=78))
    save_fig(fig, OUT, "opinion_state_space_treemap")


# ── 3. direction sunburst (family → leaf, coloured by adversarial direction) ──
def fig_direction_sunburst(ont):
    leaves = collect_leaves(ont)
    fam_dir = defaultdict(Counter)
    for fam, d in leaves:
        fam_dir[fam][d] += 1
    fam_order = sorted(families(ont), key=lambda f: -(fam_dir[f][-1] + fam_dir[f][0] + fam_dir[f][1]))
    ids, parents, labels, values, colors, custom = [], [], [], [], [], []
    ids.append("ISSUE POSITIONS"); parents.append(""); labels.append("ISSUE POSITIONS")
    values.append(len(leaves)); colors.append("#ffffff"); custom.append("")
    for f in fam_order:
        fam_n = fam_dir[f][-1] + fam_dir[f][0] + fam_dir[f][1]
        ids.append(f); parents.append("ISSUE POSITIONS"); labels.append(short(f, 20))
        values.append(fam_n); colors.append(INNER); custom.append(f"{fam_n} leaves")
        for d in (-1, 0, 1):
            n = fam_dir[f][d]
            if n == 0:
                continue
            ids.append(f"{f}|{d}"); parents.append(f); labels.append(str(n))
            values.append(n); colors.append(DIR_COLORS[d]); custom.append(DIR_LABELS[d])
    fig = go.Figure(go.Sunburst(
        ids=ids, parents=parents, labels=labels, values=values, branchvalues="total",
        marker=dict(colors=colors, line=dict(color="white", width=1)),
        sort=False, customdata=custom, insidetextorientation="radial",
        texttemplate="%{label}", hovertemplate="<b>%{customdata}</b> · %{value}<extra></extra>",
        maxdepth=3))
    # legend proxy via annotations
    fig.update_layout(**base_layout(
        "Adversarial-direction map of the issue-position state space",
        "inner ring = issue domain · outer ring = erode / neutral / amplify leaf counts (arc ∝ count)",
        h=860, w=980))
    leg = " &nbsp; ".join(
        f"<span style='color:{DIR_COLORS[k]}'>■</span> {DIR_LABELS[k]}" for k in (-1, 0, 1))
    fig.add_annotation(x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False,
                       text=leg, font=dict(size=14, color=PAL["ink"]))
    save_fig(fig, OUT, "opinion_direction_sunburst", html=True)


# ── 4. diverging direction landscape per family ──────────────────────────────
def fig_direction_landscape(ont):
    leaves = collect_leaves(ont)
    fam_dir = defaultdict(Counter)
    for fam, d in leaves:
        fam_dir[fam][d] += 1
    fams = sorted(families(ont), key=lambda f: (fam_dir[f][1] + fam_dir[f][-1]))
    labels = [short(f, 30) for f in fams]
    erode = [-fam_dir[f][-1] for f in fams]
    amplify = [fam_dir[f][1] for f in fams]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                        specs=[[{"type": "bar"}, {"type": "domain"}]],
                        subplot_titles=("Amplify vs erode targets per issue domain",
                                        "Whole state space"), horizontal_spacing=0.10)
    fig.add_trace(go.Bar(y=labels, x=erode, orientation="h", name="−1 erode",
                         marker_color=DIR_COLORS[-1], text=[abs(v) or "" for v in erode],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="erode: %{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(y=labels, x=amplify, orientation="h", name="+1 amplify",
                         marker_color=DIR_COLORS[1], text=[v or "" for v in amplify],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="amplify: %{x}<extra></extra>"), row=1, col=1)
    tot = Counter(d for _, d in leaves)
    fig.add_trace(go.Pie(labels=[DIR_LABELS[-1], DIR_LABELS[0], DIR_LABELS[1]],
                         values=[tot[-1], tot[0], tot[1]], hole=0.55, sort=False, showlegend=False,
                         marker=dict(colors=[DIR_COLORS[-1], NEUTRAL_LEAF, DIR_COLORS[1]]),
                         textinfo="label+value", textfont=dict(size=12),
                         hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"), row=1, col=2)
    fig.update_layout(**base_layout(
        "Adversarial-direction encoding of the issue-position taxonomy",
        "+1 = adversary amplifies the construct · −1 = adversary erodes it · 0 = retained for diversity",
        h=720, w=1240, multipanel=True))
    fig.update_layout(barmode="relative", bargap=0.28,
                      legend=dict(orientation="h", y=-0.08, x=0.36, xanchor="center"))
    fig.update_xaxes(title_text="← erode    leaves    amplify →", zeroline=True,
                     zerolinecolor=PAL["muted"], zerolinewidth=1.5,
                     showgrid=True, gridcolor=PAL["line"], row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=11), automargin=True, row=1, col=1)
    style_subplot_titles(fig)
    save_fig(fig, OUT, "opinion_direction_landscape")


# ── 5. sampling structure (two-layer entropy + cluster allocation) ───────────
def fig_sampling_structure(sample, summ):
    clusters = sample["clusters"]
    n_sampled = np.array([c["n_sampled"] for c in clusters.values()])
    n_leaves = np.array([c["n_leaves"] for c in clusters.values()])
    fam_of = [c["parent_name"] for c in clusters.values()]
    fam_order = list(dict.fromkeys(sorted(fam_of)))
    cmap = {f: SEQ[i % len(SEQ)] for i, f in enumerate(fam_order)}
    tl = summ["two_layer_max_entropy"]
    db = summ["direction_balance"]

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.27, 0.34, 0.39],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Two-layer normalised entropy", "Leaves per parent cluster",
                        "Allocation logic: draws-per-leaf vs cluster size"), horizontal_spacing=0.12)

    ent_lab = ["Leaf layer<br>(within cluster)", "Cluster layer<br>(across clusters)"]
    ent_val = [tl["leaf_layer_normalised_entropy"] * 100, tl["cluster_layer_normalised_entropy"] * 100]
    fig.add_trace(go.Bar(y=ent_lab, x=ent_val, orientation="h", width=0.5,
                         marker_color=[PAL["blue"], PAL["teal"]],
                         text=[f"{v:.2f}%" for v in ent_val], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x:.3f}%<extra></extra>"), row=1, col=1)
    fig.add_vline(x=100, line=dict(color=PAL["muted"], dash="dot"), row=1, col=1)

    # leaves-per-cluster heterogeneity (the structure the sampler must balance)
    fig.add_trace(go.Histogram(x=n_leaves, xbins=dict(start=0.5, end=n_leaves.max() + 0.5, size=1),
                               marker_color=PAL["sky"],
                               hovertemplate="%{x} leaves: %{y} clusters<extra></extra>"), row=1, col=2)

    # uniform per-cluster draws ⇒ draws-per-leaf ∝ 1/size; reference hyperbola
    for f in fam_order:
        xs = np.array([c["n_leaves"] for c in clusters.values() if c["parent_name"] == f])
        dr = np.array([c["n_sampled"] for c in clusters.values() if c["parent_name"] == f])
        fig.add_trace(go.Scatter(x=xs, y=dr / xs, mode="markers", name=short(f, 22),
                                 marker=dict(size=11, color=cmap[f], opacity=0.85,
                                             line=dict(color="white", width=1)),
                                 customdata=np.stack([dr, xs], axis=-1),
                                 hovertemplate=(f"{short(f,30)}<br>%{{customdata[1]}} leaves · "
                                                "%{customdata[0]} draws → %{y:.2f} draws/leaf<extra></extra>")),
                      row=1, col=3)
    xr = np.linspace(1, n_leaves.max(), 60)
    fig.add_trace(go.Scatter(x=xr, y=n_sampled.mean() / xr, mode="lines", showlegend=False,
                             line=dict(color=PAL["muted"], dash="dash"),
                             hovertemplate="≈ %{y:.2f} draws/leaf<extra></extra>"), row=1, col=3)
    fig.update_layout(**base_layout(
        "Two-layer maximal-entropy sampling of the opinion panel",
        f"1,000 draws · 69 parent clusters · 14 families · direction balance of draws "
        f"−1:{db['-1']} / +1:{db['1']}",
        h=640, w=1320, multipanel=True))
    fig.update_layout(showlegend=False, bargap=0.08)
    fig.update_xaxes(title_text="normalised entropy (%)", range=[80, 101.5], row=1, col=1,
                     showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(tickfont=dict(size=11), automargin=True, row=1, col=1)
    fig.update_xaxes(title_text="leaves in cluster", row=1, col=2, showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(title_text="parent clusters", row=1, col=2, showgrid=True, gridcolor=PAL["line"])
    fig.update_xaxes(title_text="leaves in cluster", row=1, col=3, showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(title_text="draws per leaf", row=1, col=3, showgrid=True, gridcolor=PAL["line"])
    style_subplot_titles(fig)
    save_fig(fig, OUT, "opinion_sampling_structure")


# ── 6. coverage vs full state space ──────────────────────────────────────────
def fig_coverage(ont, sample, summ):
    leaves = collect_leaves(ont)
    fam_total = Counter(f for f, _ in leaves)
    fam_dir = Counter(f for f, d in leaves if d != 0)
    clusters = sample["clusters"]
    fam_draws, fam_cov = Counter(), Counter()
    for c in clusters.values():
        fam_draws[c["parent_name"]] += c["n_sampled"]
        fam_cov[c["parent_name"]] += sum(1 for lf in c["leaves"] if lf["count"] > 0)
    covered = sum(fam_cov.values())
    fams = sorted(fam_total, key=lambda f: -fam_total[f])
    labels = [short(f, 30) for f in fams][::-1]
    total = [fam_total[f] for f in fams][::-1]
    direc = [fam_dir[f] for f in fams][::-1]
    draws = [fam_draws.get(f, 0) for f in fams][::-1]
    n_total = sum(fam_total.values()); n_dir = sum(fam_dir.values())

    fig = make_subplots(rows=1, cols=2, column_widths=[0.34, 0.66],
                        specs=[[{"type": "bar"}, {"type": "bar"}]],
                        subplot_titles=("State space → frame → covered", "Per issue domain: leaves, frame, draws"),
                        horizontal_spacing=0.17)
    casc_lab = ["Opinion leaves<br>(full state space)", "Adversarially directional<br>(sampling frame)",
                "Directional leaves<br>covered by the panel"]
    casc_val = [n_total, n_dir, covered]
    casc_pct = ["100%", f"{n_dir/n_total:.0%} of leaves", f"{covered/n_dir:.0%} of frame"]
    casc_col = [PAL["slate"], PAL["blue"], PAL["teal"]]
    fig.add_trace(go.Bar(y=casc_lab[::-1], x=casc_val[::-1], orientation="h", showlegend=False,
                         marker_color=casc_col[::-1],
                         text=[f"{v}  ·  {p}" for v, p in zip(casc_val, casc_pct)][::-1],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(y=labels, x=total, orientation="h", name="leaves (state space)",
                         marker_color=PAL["line"], marker_line_color=PAL["muted"], marker_line_width=1,
                         hovertemplate="%{y}: %{x} leaves<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(y=labels, x=direc, orientation="h", name="directional (frame)",
                         marker_color=PAL["blue"], hovertemplate="%{y}: %{x} directional<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(y=labels, x=draws, mode="markers", name="draws allocated",
                             marker=dict(color=PAL["orange"], size=11, symbol="diamond",
                                         line=dict(color="white", width=1)),
                             hovertemplate="%{y}: %{x} draws<extra></extra>"), row=1, col=2)
    fig.update_layout(**base_layout(
        "Opinion panel coverage of the Issue Position Taxonomy",
        f"{n_dir}/{n_total} issue-position leaves are adversarially directional ({n_dir/n_total:.0%}); "
        f"draws across {summ['n_clusters']} issue domains cover {covered}/{n_dir} of them ({covered/n_dir:.0%})",
        h=680, w=1280, multipanel=True))
    fig.update_layout(barmode="overlay", bargap=0.25,
                      legend=dict(orientation="h", y=-0.09, x=0.66, xanchor="center"))
    fig.update_xaxes(showgrid=True, gridcolor=PAL["line"], range=[0, n_total * 1.18], row=1, col=1)
    fig.update_xaxes(title_text="count", showgrid=True, gridcolor=PAL["line"], row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), automargin=True)
    style_subplot_titles(fig)
    save_fig(fig, OUT, "opinion_coverage_vs_statespace")


def main():
    full_ont = json.loads(ONT.read_text())
    sample = json.loads((SAMPLES / "opinion_targets_maxent_1000.json").read_text())
    summ = json.loads((SAMPLES / "opinion_targets_maxent_1000.summary.json").read_text())

    # Opinion targets are now restricted to the Issue_Position_Taxonomy subtree, so
    # scope the state-space figures to it; its issue domains become the grouping.
    root = sample.get("_meta", {}).get("root_subtree") or "Issue_Position_Taxonomy"
    ont = full_ont.get(root, full_ont)

    print("Building OPINION overview figures…")
    fig_sunburst(ont)
    fig_treemap(ont)
    fig_direction_sunburst(ont)
    fig_direction_landscape(ont)
    fig_sampling_structure(sample, summ)
    fig_coverage(ont, sample, summ)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
