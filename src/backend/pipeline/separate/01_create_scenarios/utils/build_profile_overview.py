from __future__ import annotations

"""
PROFILE sample-set overview figures
===================================
A polished visual overview of the 10K high-resolution profile set and its
relationship to the full PROFILE deployment state space:

  profile_ontology_sunburst.html/.png   hierarchical sunburst of the state space
                                          (rings: domain → subconstruct → variable)
  profile_state_space_treemap.png        treemap of the state space by domain
  profile_basic_demographics.png         sex / gender / age / education / region / …
  profile_power_compromise.png           Kish q=0.6: realistic vs achieved + weights
  profile_maxentropy_fundamentals.png    age + Big Five (continuous, full-range)
  profile_coverage_and_power.png         state-space coverage + per-variable min cell
  profile_personality_manifold_3d.*      3D Big-Five hypercube with wall shadows
                                          (demonstrates joint max-entropy fill)

Run by file path:
  python .../utils/build_profile_overview.py
"""

import json
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

from production_profile_sampling import load_population  # noqa: E402
from overview_theme import (  # noqa: E402
    PAL, SEQ, FONT, base_layout, save_fig, style_subplot_titles,
    walk_hierarchy, domain_color_map, leaf_count, human,
)

STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
ONT = PROJECT_ROOT / "src" / "backend" / "ontology" / "separate" / "production" / "PROFILE" / "profile.json"
SAMPLES = STAGE / "samples" / "01_separated" / "profiles"
OUT = STAGE / "overview" / "separate" / "profiles"


# ── ontology hierarchy figures ───────────────────────────────────────────────
def fig_sunburst(person):
    h = walk_hierarchy(person, "PERSON", max_depth=3)
    cmap = domain_color_map(h["domains"])
    colors = ["#ffffff" if d == "root" else cmap[d] for d in h["domains"]]
    fig = go.Figure(go.Sunburst(
        ids=h["ids"], parents=h["parents"], labels=h["labels"], values=h["values"],
        branchvalues="total", marker=dict(colors=colors, line=dict(color="white", width=1.2)),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value} leaf states<br>%{percentRoot:.1%} of state space<extra></extra>",
        maxdepth=3,
    ))
    total = leaf_count(person)
    fig.update_layout(**base_layout(
        "PROFILE deployment state space",
        f"{total:,} leaf states · rings: domain → subconstruct → variable · arc ∝ state count",
        h=820, w=920))
    save_fig(fig, OUT, "profile_ontology_sunburst", html=True)


def fig_treemap(person):
    h = walk_hierarchy(person, "PERSON", max_depth=2)
    cmap = domain_color_map(h["domains"])
    colors = ["#ffffff" if d == "root" else cmap[d] for d in h["domains"]]
    fig = go.Figure(go.Treemap(
        ids=h["ids"], parents=h["parents"], labels=h["labels"], values=h["values"],
        branchvalues="total", marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+value", tiling=dict(pad=3),
        hovertemplate="<b>%{label}</b><br>%{value} leaf states (%{percentRoot:.1%})<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        "PROFILE state space by domain",
        "tile area ∝ number of leaf states in the deployment ontology", h=620, w=1180, t=78))
    save_fig(fig, OUT, "profile_state_space_treemap")


# ── distributions ────────────────────────────────────────────────────────────
def _cv(p, frag):
    for k, v in p["categorical_attributes"].items():
        if frag in k and v is not None:
            return v
    return None


def fig_demographics(profs, summ):
    def dist(frag, top=10):
        c = Counter(_cv(p, frag) for p in profs if _cv(p, frag) is not None)
        items = c.most_common(top)
        return [x[0].replace("_", " ") for x in items][::-1], [x[1] for x in items][::-1]
    ages = [p["demographics"]["age_years"] for p in profs]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Sex assigned at birth", "Gender identity (top 10)", "Chronological age",
                        "Highest education", "World region", "Sexual orientation"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "xy"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.17, horizontal_spacing=0.12)

    def hbar(frag, r, c, color, top=10):
        labs, vals = dist(frag, top)
        fig.add_trace(go.Bar(y=labs, x=vals, orientation="h", marker_color=color,
                             text=vals, textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x}<extra></extra>"), row=r, col=c)

    hbar("sex_assigned_at_birth", 1, 1, PAL["blue"], 5)
    hbar("gender_identity", 1, 2, PAL["teal"], 10)
    fig.add_trace(go.Histogram(x=ages, nbinsx=32, marker_color=PAL["sky"],
                               hovertemplate="age %{x}: %{y}<extra></extra>"), row=1, col=3)
    hbar("highest_education", 2, 1, PAL["amber"], 9)
    hbar("broad_region_continent", 2, 2, PAL["orange"], 7)
    hbar("sexual_orientation_identity", 2, 3, PAL["muted"], 8)

    fig.update_layout(**base_layout(
        "Basic demographic composition of the 10K profile set",
        "categorical variables use Kish power-compromise allocation (q=0.6); age is uniform / max-entropy",
        h=800, w=1240, multipanel=True))
    fig.update_layout(showlegend=False, bargap=0.25)
    fig.update_xaxes(showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=11))
    style_subplot_titles(fig)
    save_fig(fig, OUT, "profile_basic_demographics")


def fig_power_compromise(summ):
    sa = summ["entropy_report"]["subgroup_allocation"]
    per = sa["per_variable"]
    q = sa["compromise_exponent_q"]

    def grouped(varname, top):
        cats = per[varname]["categories"]
        items = sorted(cats.items(), key=lambda kv: -kv[1]["achieved"])[:top]
        labs = [k.replace("_", " ") for k, _ in items]
        ach = [v["achieved"] * 100 for _, v in items]
        real = [v["realistic"] * 100 for _, v in items]
        dw = [v["design_weight"] for _, v in items]
        return labs[::-1], ach[::-1], real[::-1], dw[::-1]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Sex assigned at birth", "Gender identity (top 8)"),
                        horizontal_spacing=0.18)
    for col, (var, top) in enumerate([("Sex_Assigned_At_Birth", 3), ("Gender_Identity", 8)], start=1):
        labs, ach, real, dw = grouped(var, top)
        fig.add_trace(go.Bar(y=labs, x=real, orientation="h", name="population (realistic)",
                             marker_color=PAL["line"], marker_line_color=PAL["muted"], marker_line_width=1,
                             showlegend=(col == 1), hovertemplate="population: %{x:.1f}%<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Bar(y=labs, x=ach, orientation="h", name="sampled (q=%.1f)" % q,
                             marker_color=PAL["teal"], showlegend=(col == 1),
                             text=[f"×{w:.2f}" if w else "" for w in dw], textposition="outside", cliponaxis=False,
                             hovertemplate="sampled: %{x:.1f}%<br>design weight %{text}<extra></extra>"), row=1, col=col)
    fig.update_layout(**base_layout(
        "Prevalence ↔ statistical-power compromise (Kish power allocation, q=0.6)",
        "bars: population prevalence vs achieved sample share · labels: design weight to reweight to population",
        h=560, w=1180, multipanel=True))
    fig.update_layout(barmode="group", bargap=0.3, legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    fig.update_xaxes(title_text="share of sample (%)", showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(automargin=True, tickfont=dict(size=12))
    style_subplot_titles(fig)
    save_fig(fig, OUT, "profile_power_compromise")


def fig_maxent(profs):
    ages = [p["demographics"]["age_years"] for p in profs]
    traits = ["openness_to_experience", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    vals = {t: [] for t in traits}
    for p in profs:
        bf = p["demographics"].get("big_five", {})
        for t in traits:
            if t in bf:
                vals[t].append(bf[t]["pct"])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=("Chronological age (16–80, uniform)",
                                        "Big Five trait levels (0–100, full-range)"),
                        specs=[[{"type": "xy"}, {"type": "xy"}]], horizontal_spacing=0.12)
    fig.add_trace(go.Histogram(x=ages, nbinsx=33, marker_color=PAL["sky"],
                               hovertemplate="age %{x}: %{y}<extra></extra>"), row=1, col=1)
    colors = [PAL["teal"], PAL["blue"], PAL["amber"], PAL["orange"], PAL["red"]]
    for i, t in enumerate(traits):
        fig.add_trace(go.Violin(x=vals[t], name=t.split("_")[0].title(), orientation="h",
                                line_color=colors[i], fillcolor=colors[i], opacity=0.65,
                                points=False, meanline_visible=True,
                                hovertemplate=f"{t.split('_')[0]}: %{{x:.0f}}<extra></extra>"), row=1, col=2)
    fig.update_layout(**base_layout(
        "Max-entropy fundamentals: age & personality span their full range",
        "the focal variables are deliberately uniform / full-spread to avoid range restriction",
        h=560, w=1200, multipanel=True))
    fig.update_layout(showlegend=False, violingap=0.25)
    fig.update_xaxes(showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(title_text="age (years)", row=1, col=1)
    fig.update_xaxes(title_text="percentile", row=1, col=2, range=[0, 100])
    style_subplot_titles(fig)
    save_fig(fig, OUT, "profile_maxentropy_fundamentals")


def fig_coverage_power(summ, meta):
    er = summ["entropy_report"]
    cov = summ.get("coverage_report", meta.get("coverage_report", {}))
    sa = er["subgroup_allocation"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=("State-space coverage", "Min cell size per curated variable"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]], horizontal_spacing=0.16)
    cov_labels = ["Categorical variables<br>fully covered", "Mean categorical<br>option coverage",
                  "Continuous range<br>visited", "Leaves resolved<br>per profile"]
    cov_vals = [cov.get("categorical_variables_fully_covered", 0) / max(1, cov.get("categorical_variable_count", 1)) * 100,
                cov.get("categorical_option_coverage_mean", 0) * 100,
                cov.get("continuous_mean_range_visited_pct", 0),
                meta.get("leaves_resolved_per_profile", 0)]
    cov_disp = [cov_vals[0], cov_vals[1], cov_vals[2], 100]
    cov_text = [f"{cov_vals[0]:.0f}%", f"{cov_vals[1]*1:.1f}%" if cov_vals[1] <= 100 else f"{cov_vals[1]:.0f}",
                f"{cov_vals[2]:.0f}%", f"{int(cov_vals[3])} leaves"]
    fig.add_trace(go.Bar(y=cov_labels[::-1], x=cov_disp[::-1], orientation="h",
                         marker_color=[PAL["teal"], PAL["sky"], PAL["blue"], PAL["amber"]][::-1],
                         text=cov_text[::-1], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{text}<extra></extra>"), row=1, col=1)
    per = sa["per_variable"]
    mins = sorted(((v.replace("_", " "), d["min_cell_count"]) for v, d in per.items()), key=lambda x: x[1])[:14]
    labs = [m[0] for m in mins][::-1]
    vals = [m[1] for m in mins][::-1]
    thr = sa["min_cell_count_threshold"]
    bar_colors = [PAL["red"] if v < thr else PAL["teal"] for v in vals]
    fig.add_trace(go.Bar(y=labs, x=vals, orientation="h", marker_color=bar_colors,
                         text=vals, textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x} samples<extra></extra>"), row=1, col=2)
    fig.add_vline(x=thr, line=dict(color=PAL["muted"], dash="dash"), row=1, col=2)
    fig.update_layout(**base_layout(
        "Full-state-space coverage & per-subgroup statistical power",
        f"every cell stays analysable; dashed line = reporting threshold ({thr}); red = below it",
        h=560, w=1240, multipanel=True))
    fig.update_layout(showlegend=False, bargap=0.3)
    fig.update_xaxes(showgrid=True, gridcolor=PAL["line"])
    fig.update_xaxes(title_text="rarest-category sample size", row=1, col=2)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    style_subplot_titles(fig)
    save_fig(fig, OUT, "profile_coverage_and_power")


# ── 3D personality manifold ──────────────────────────────────────────────────
def fig_personality_3d(profs):
    """All 10K configurations placed in the Big-Five hypercube (O,C,E axes,
    colour = Neuroticism), with faint projected 'shadows' on the three back
    walls. The uniform cube fill + uniform wall shadows show that the sampler
    fills the joint personality space at maximal entropy (no range restriction)."""
    def trait(p, t):
        return p["demographics"].get("big_five", {}).get(t, {}).get("pct")

    O, C, E, N, A = [], [], [], [], []
    for p in profs:
        o, c, e, n, a = (trait(p, t) for t in
                         ("openness_to_experience", "conscientiousness", "extraversion",
                          "neuroticism", "agreeableness"))
        if None in (o, c, e, n, a):
            continue
        O.append(o); C.append(c); E.append(e); N.append(n); A.append(a)
    O, C, E, N, A = map(np.array, (O, C, E, N, A))
    rng = np.random.default_rng(7)
    shadow = rng.choice(len(O), size=min(2600, len(O)), replace=False)

    fig = go.Figure()
    # wall shadows (marginals projected onto the cube faces)
    wall = dict(mode="markers", marker=dict(size=2.2, color=PAL["line"], opacity=0.55),
                hoverinfo="skip", showlegend=False)
    fig.add_trace(go.Scatter3d(x=np.zeros_like(O[shadow]), y=C[shadow], z=E[shadow], **wall))
    fig.add_trace(go.Scatter3d(x=O[shadow], y=np.full_like(C[shadow], 100), z=E[shadow], **wall))
    fig.add_trace(go.Scatter3d(x=O[shadow], y=C[shadow], z=np.zeros_like(E[shadow]), **wall))
    # main cloud
    fig.add_trace(go.Scatter3d(
        x=O, y=C, z=E, mode="markers",
        marker=dict(size=2.7, color=N, colorscale=[[0, PAL["blue"]], [0.5, "#eef2f8"], [1, PAL["red"]]],
                    cmin=0, cmax=100, opacity=0.78, line=dict(width=0),
                    colorbar=dict(title="Neuroticism", thickness=14, len=0.55, x=0.98, tickvals=[0, 50, 100])),
        customdata=np.stack([N, A], axis=-1),
        hovertemplate=("Openness %{x:.0f} · Conscientiousness %{y:.0f}<br>"
                       "Extraversion %{z:.0f} · Neuroticism %{customdata[0]:.0f}<br>"
                       "Agreeableness %{customdata[1]:.0f}<extra></extra>"),
        showlegend=False))

    axis = dict(backgroundcolor="#fbfcfe", gridcolor=PAL["line"], zerolinecolor=PAL["line"],
                showbackground=True, range=[0, 100], tickvals=[0, 25, 50, 75, 100],
                titlefont=dict(size=13))
    fig.update_layout(**base_layout(
        "Profile-configuration sampling — the Big-Five personality hypercube",
        f"{len(O):,} profiles · axes O·C·E, colour = Neuroticism · wall shadows are the marginals "
        "→ uniform fill = joint maximal entropy",
        h=820, w=1060))
    fig.update_layout(margin=dict(t=104, l=10, r=10, b=10), scene=dict(
        domain=dict(x=[0.0, 0.92], y=[0.0, 1.0]),
        xaxis=dict(title="Openness", **axis), yaxis=dict(title="Conscientiousness", **axis),
        zaxis=dict(title="Extraversion", **axis),
        camera=dict(eye=dict(x=1.42, y=1.32, z=0.86), center=dict(x=0, y=0, z=-0.08)),
        aspectmode="cube"))
    save_fig(fig, OUT, "profile_personality_manifold_3d", html=True)


def _prune(node, names):
    """Drop branch children whose key is in `names` so the state-space figures
    match the sample (e.g. the excluded Issue_Position_Taxonomy subtree)."""
    if isinstance(node, dict):
        for k in list(node.keys()):
            if k and k[0].isupper():
                if k in names:
                    del node[k]
                else:
                    _prune(node[k], names)


def main():
    ontology = json.loads(ONT.read_text())
    bulk = json.loads((SAMPLES / "production_profiles_maxent_10000.json").read_text())
    summ = json.loads((SAMPLES / "production_profiles_maxent_10000.summary.json").read_text())
    profs = load_population(bulk)
    meta = bulk.get("_meta", {})

    person = ontology["PERSON"]
    _prune(person, set(meta.get("excluded_subtrees", [])))  # match the sample's scope

    print("Building PROFILE overview figures…")
    fig_sunburst(person)
    fig_treemap(person)
    fig_demographics(profs, summ)
    fig_power_compromise(summ)
    fig_maxent(profs)
    fig_coverage_power(summ, meta)
    fig_personality_3d(profs)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
