from __future__ import annotations

"""
ATTACK sample-set overview figures
==================================
Visual overview of the DISARM-red Plan/Prepare/Execute opinion-manipulation
vector set.  The sample comes from an EXTERNAL attack ontology (not this
repository's attack ontology): 100K coherent triplets were sampled at maximal
marginal entropy, then heuristically filtered to 48,991 opinion-relevant
vectors.  Figures:

  attack_state_space_sunburst.html/.png   phase → tactic → sub-tactic hierarchy
  attack_state_space_treemap.png          leaf pool by phase and tactic
  attack_filter_funnel.png                100K → 48,991 retention + removal reasons
  attack_signal_distributions.png         included vs excluded opinion-signal
  attack_evidence_criteria.png            9 opinion-manipulation mechanisms × phase
  attack_entropy_retention.png            post-filter range-retention per phase
  attack_tactic_retention.png             how the filter reshaped tactic shares
  attack_phase_signal_cube_3d.html/.png   triplet signal in Plan×Prepare×Execute

Run by file path:
  python .../utils/build_attack_overview.py
"""

import json
import sys
from collections import defaultdict
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
    PAL, SEQ, FONT, base_layout, save_fig, style_subplot_titles, human,
)

STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
ATTACKS = STAGE / "samples" / "01_separated" / "attacks"
FILTERED = ATTACKS / "red_plan_prepare_execute_opinion_effect_filtered.json"
OUT = STAGE / "overview" / "separate" / "attacks"

PHASES = ("Plan", "Prepare", "Execute")
PHASE_COLORS = {"Plan": PAL["blue"], "Prepare": PAL["teal"], "Execute": PAL["orange"]}


def short(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ── path-based hierarchy for sunburst / treemap ──────────────────────────────
def path_hierarchy(leaves, max_depth=3):
    """Aggregate leaf-catalog paths into a node hierarchy sized by leaf count."""
    counts = defaultdict(int)         # tuple(prefix) -> leaf count
    labels = {}
    for lf in leaves:
        p = lf["path"][:max_depth]
        for d in range(1, len(p) + 1):
            pref = tuple(p[:d])
            counts[pref] += 1
            labels[pref] = p[d - 1]
    ids, parents, lab, vals, phase = [], [], [], [], []
    for pref, c in counts.items():
        ids.append(" > ".join(pref))
        parents.append(" > ".join(pref[:-1]) if len(pref) > 1 else "")
        lab.append(short(labels[pref], 26))
        vals.append(c)
        phase.append(pref[0])
    return ids, parents, lab, vals, phase


def fig_sunburst(leaves):
    ids, parents, lab, vals, phase = path_hierarchy(leaves, max_depth=3)
    colors = [PHASE_COLORS.get(ph, PAL["muted"]) for ph in phase]
    fig = go.Figure(go.Sunburst(
        ids=ids, parents=parents, labels=lab, values=vals, branchvalues="total",
        marker=dict(colors=colors, line=dict(color="white", width=1.1)),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value} leaves<br>%{percentRoot:.1%} of pool<extra></extra>",
        maxdepth=3))
    fig.update_layout(**base_layout(
        "DISARM-red attack state space (external ontology)",
        f"{len(leaves):,} leaves · rings: phase → tactic → sub-tactic · "
        "blue Plan · teal Prepare · orange Execute",
        h=840, w=940))
    save_fig(fig, OUT, "attack_state_space_sunburst", html=True)


def fig_treemap(leaves):
    ids, parents, lab, vals, phase = path_hierarchy(leaves, max_depth=2)
    colors = [PHASE_COLORS.get(ph, PAL["muted"]) for ph in phase]
    fig = go.Figure(go.Treemap(
        ids=ids, parents=parents, labels=lab, values=vals, branchvalues="total",
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+value", tiling=dict(pad=3),
        hovertemplate="<b>%{label}</b><br>%{value} leaves (%{percentRoot:.1%})<extra></extra>"))
    fig.update_layout(**base_layout(
        "Attack leaf pool by phase and tactic",
        "tile area ∝ leaves in the raw DISARM-red Plan/Prepare/Execute pool", h=640, w=1180, t=78))
    save_fig(fig, OUT, "attack_state_space_treemap")


# ── filter funnel + removal reasons ──────────────────────────────────────────
def fig_funnel(man):
    s = man["summary"]
    raw = s["raw_configurations"]
    kept = s["configurations_remaining_after_filter"]
    excl = man["diagnostics_exclusion"]
    fc = excl["failed_criteria_counts"]
    sc = excl["support_exclusion_criteria_counts_in_excluded_configs"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.46, 0.54],
                        specs=[[{"type": "funnel"}, {"type": "bar"}]],
                        subplot_titles=("Heuristic opinion-effect filter",
                                        "Why configurations were removed (non-exclusive)"),
                        horizontal_spacing=0.16)
    fig.add_trace(go.Funnel(
        y=["Raw coherent triplets", "Opinion-relevant (retained)"], x=[raw, kept],
        textposition="inside", textinfo="value+percent initial",
        marker=dict(color=[PAL["slate"], PAL["teal"]]),
        connector=dict(line=dict(color=PAL["line"])),
        hovertemplate="%{y}: %{x:,}<extra></extra>"), row=1, col=1)

    reasons = {
        "Too few relevant phases": fc["insufficient_relevant_phases"],
        "Total signal < 5.0": fc["total_signal_below_threshold"],
        "Support-only leaf present": fc["support_only_leaf_present"],
        "Too few distinct mechanisms": fc["insufficient_distinct_opinion_manipulation_evidence"],
    }
    reasons = dict(sorted(reasons.items(), key=lambda kv: kv[1]))
    fig.add_trace(go.Bar(
        y=list(reasons), x=list(reasons.values()), orientation="h", marker_color=PAL["red"],
        text=[f"{v:,}" for v in reasons.values()], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} configs<extra></extra>"), row=1, col=2)
    # support-only sub-reasons as a second, lighter series
    sup = dict(sorted(sc.items(), key=lambda kv: kv[1]))
    fig.add_trace(go.Bar(
        y=[short(k, 26) for k in sup], x=list(sup.values()), orientation="h",
        marker_color=PAL["amber"], opacity=0.85,
        text=[f"{v:,}" for v in sup.values()], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} configs<extra></extra>"), row=1, col=2)

    fig.update_layout(**base_layout(
        "From 100K coherent triplets to 48,991 opinion-manipulation vectors",
        f"retention {s['configuration_retention_rate']:.0%} · entropy is never used for inclusion "
        "(reported only afterwards as a range diagnostic)",
        h=640, w=1320, multipanel=True))
    fig.update_layout(showlegend=False, bargap=0.35)
    fig.update_xaxes(title_text="configurations", showgrid=True, gridcolor=PAL["line"], row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), automargin=True, row=1, col=2)
    style_subplot_titles(fig)
    save_fig(fig, OUT, "attack_filter_funnel")


# ── included vs excluded signal ──────────────────────────────────────────────
def fig_signal(configs, man):
    sig = np.array([c["opinion_manipulation_evidence"]["signal_total"] for c in configs])
    inc = man["summary"]["included_signal_score_distribution"]
    exc = man["summary"]["excluded_signal_score_distribution"]
    thr = man["filter_method"]["thresholds"]["config_min_total_signal"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        specs=[[{"type": "xy"}, {"type": "xy"}]],
                        subplot_titles=("Opinion-signal of retained vectors (density + raw points)",
                                        "Retained vs removed (p10–p90 range, ● median)"),
                        horizontal_spacing=0.14)
    rng = np.random.default_rng(3)
    pts = sig[rng.choice(len(sig), size=min(6000, len(sig)), replace=False)]
    counts, edges = np.histogram(sig, bins=70)
    dens = counts[np.clip(np.searchsorted(edges, pts, "right") - 1, 0, len(counts) - 1)]
    fig.add_trace(go.Violin(x=sig, orientation="h", side="positive", width=1.7, points=False,
                            line=dict(color=PAL["teal"], width=2), fillcolor="rgba(42,157,143,0.20)",
                            meanline_visible=True, spanmode="hard", name="retained",
                            hovertemplate="signal %{x}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pts, y=-0.05 - rng.random(len(pts)) * 0.62, mode="markers",
                             marker=dict(size=3.0, color=dens, colorscale=[[0, "#cfe6df"], [0.5, PAL["teal"]],
                                         [1, PAL["navy"]]], opacity=0.55, line=dict(width=0)),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_vline(x=thr, line=dict(color=PAL["red"], dash="dash"), row=1, col=1,
                  annotation_text=f"inclusion ≥ {thr}", annotation_position="top right",
                  annotation_font_color=PAL["red"])
    fig.add_vline(x=float(sig.mean()), line=dict(color=PAL["navy"], dash="dot"), row=1, col=1)

    for i, (lab, d, col) in enumerate([("retained", inc, PAL["teal"]), ("removed", exc, PAL["muted"])]):
        y = lab
        fig.add_trace(go.Scatter(x=[d["p10"], d["p90"]], y=[y, y], mode="lines",
                                 line=dict(color=col, width=10), opacity=0.45, showlegend=False,
                                 hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[d["p25"], d["p75"]], y=[y, y], mode="lines",
                                 line=dict(color=col, width=18), showlegend=False,
                                 hovertemplate=f"{lab} IQR %{{x}}<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[d["median"]], y=[y], mode="markers",
                                 marker=dict(color="white", size=11, line=dict(color=col, width=3)),
                                 showlegend=False, hovertemplate=f"{lab} median %{{x}}<extra></extra>"),
                      row=1, col=2)
    fig.add_vline(x=thr, line=dict(color=PAL["red"], dash="dash"), row=1, col=2)

    fig.update_layout(**base_layout(
        "The filter cleanly separates opinion-relevant attack vectors",
        f"retained median signal {inc['median']} vs removed median {exc['median']}; "
        "threshold is a floor, not an entropy objective",
        h=600, w=1300, multipanel=True))
    fig.update_layout(showlegend=False, bargap=0.04)
    fig.update_xaxes(title_text="total opinion-manipulation signal", showgrid=True, gridcolor=PAL["line"], row=1, col=1)
    fig.update_yaxes(title_text="", showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="signal score", showgrid=True, gridcolor=PAL["line"], row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=2)
    style_subplot_titles(fig)
    save_fig(fig, OUT, "attack_signal_distributions")


# ── evidence criteria × phase ────────────────────────────────────────────────
def fig_evidence(diag):
    rows = sorted(diag["evidence_criteria_profile"]["rows"],
                  key=lambda r: r["Plan_leaf_occurrences"] + r["Prepare_leaf_occurrences"]
                  + r["Execute_leaf_occurrences"])
    labels = [short(r["criterion"], 34) for r in rows]
    fig = go.Figure()
    for ph in PHASES:
        fig.add_trace(go.Bar(y=labels, x=[r[f"{ph}_leaf_occurrences"] for r in rows],
                             orientation="h", name=ph, marker_color=PHASE_COLORS[ph],
                             hovertemplate=f"{ph}: %{{x:,}} leaf occurrences<extra></extra>"))
    fig.update_layout(**base_layout(
        "Opinion-manipulation mechanisms present in the retained vectors",
        "leaf occurrences per evidence criterion, split by attack phase (stacked)",
        h=720, w=1180))
    fig.update_layout(barmode="stack", bargap=0.3,
                      legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
    fig.update_xaxes(title_text="leaf occurrences across configurations", showgrid=True, gridcolor=PAL["line"])
    fig.update_yaxes(tickfont=dict(size=12), automargin=True)
    save_fig(fig, OUT, "attack_evidence_criteria")


# ── entropy range-retention per phase ────────────────────────────────────────
def fig_entropy(diag):
    leaf = diag["entropy_impact"]["leaf_level"]
    metrics = [("relative_entropy_retained", "Entropy retained", "impact"),
               ("category_coverage", "Category coverage", "filtered"),
               ("pielou_evenness_against_raw_pool", "Evenness (Pielou)", "filtered")]
    fig = go.Figure()
    colors = [PAL["teal"], PAL["blue"], PAL["amber"]]
    for i, (key, name, where) in enumerate(metrics):
        vals = [leaf[ph][where][key] * 100 for ph in PHASES]
        fig.add_trace(go.Bar(x=list(PHASES), y=vals, name=name, marker_color=colors[i],
                             text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
                             hovertemplate=f"{name} %{{x}}: %{{y:.2f}}%<extra></extra>"))
    fig.update_layout(**base_layout(
        "Range-retention diagnostic: the filter preserves attack diversity",
        "computed post-filter against the raw ontological ceiling · entropy is a diagnostic, not optimized",
        h=620, w=1120))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    fig.update_yaxes(title_text="percent of raw ceiling", range=[80, 103], showgrid=True, gridcolor=PAL["line"])
    fig.update_xaxes(tickfont=dict(size=13))
    save_fig(fig, OUT, "attack_entropy_retention")


# ── tactic retention / share shift ───────────────────────────────────────────
def fig_tactic(diag):
    rows = sorted(diag["tactic_retention"], key=lambda r: r["share_shift"])
    labels = [f"{short(r['secondary'], 26)}  ({r['phase'][:2]})" for r in rows]
    shifts = [r["share_shift"] * 100 for r in rows]
    phases = [r["phase"] for r in rows]
    rates = [r["occurrence_retention_rate"] for r in rows]
    colors = [PHASE_COLORS[p] for p in phases]
    span = max(abs(min(shifts)), abs(max(shifts)))
    fig = go.Figure(go.Bar(
        y=labels, x=shifts, orientation="h", marker_color=colors, showlegend=False,
        text=[f"{s:+.1f}pp" for s in shifts],
        textposition="outside", cliponaxis=False,
        customdata=[f"{rt:.0%}" for rt in rates],
        hovertemplate="%{y}<br>share shift %{x:+.2f} pp · %{customdata} of occurrences kept<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=PAL["muted"], width=1.5))
    # phase legend proxies
    for ph in PHASES:
        fig.add_trace(go.Bar(y=[None], x=[None], orientation="h", name=ph,
                             marker_color=PHASE_COLORS[ph], showlegend=True))
    fig.update_layout(**base_layout(
        "How the opinion filter reshaped the tactic mix",
        "within-phase share shift (filtered − raw); the filter favours audience, content and "
        "priming tactics over support tactics",
        h=720, w=1200))
    fig.update_layout(bargap=0.3, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
    fig.update_xaxes(title_text="← de-emphasised   share shift (pp)   emphasised →",
                     range=[-span * 1.35, span * 1.35],
                     showgrid=True, gridcolor=PAL["line"], zeroline=False)
    fig.update_yaxes(tickfont=dict(size=11), automargin=True)
    save_fig(fig, OUT, "attack_tactic_retention")


# ── 3D phase-signal cube ─────────────────────────────────────────────────────
def fig_phase_cube(configs):
    rng = np.random.default_rng(11)
    idx = rng.choice(len(configs), size=min(7000, len(configs)), replace=False)
    P, R, E, T = [], [], [], []
    for i in idx:
        ps = configs[i]["opinion_manipulation_evidence"]["phase_scores"]
        P.append(ps["Plan"]["net_score"]); R.append(ps["Prepare"]["net_score"])
        E.append(ps["Execute"]["net_score"]); T.append(configs[i]["opinion_manipulation_evidence"]["signal_total"])
    P, R, E, T = map(np.array, (P, R, E, T))
    jit = lambda a: a + rng.normal(0, 0.12, size=len(a))  # noqa: E731
    fig = go.Figure(go.Scatter3d(
        x=jit(P), y=jit(R), z=jit(E), mode="markers",
        marker=dict(size=2.6, color=T, colorscale=[[0, PAL["blue"]], [0.5, PAL["teal"]], [1, PAL["red"]]],
                    opacity=0.72, line=dict(width=0),
                    colorbar=dict(title="total<br>signal", thickness=14, len=0.55, x=0.98)),
        hovertemplate="Plan %{x:.1f} · Prepare %{y:.1f} · Execute %{z:.1f}<extra></extra>"))
    axis = dict(backgroundcolor="#fbfcfe", gridcolor=PAL["line"], zerolinecolor=PAL["line"],
                showbackground=True, titlefont=dict(size=13))
    fig.update_layout(**base_layout(
        "Where each triplet carries its opinion-signal — Plan × Prepare × Execute",
        f"{len(P):,} retained configurations · axis = per-phase net opinion-manipulation score · "
        "colour = total signal",
        h=820, w=1060))
    fig.update_layout(margin=dict(t=104, l=10, r=10, b=10), scene=dict(
        domain=dict(x=[0.0, 0.92], y=[0.0, 1.0]),
        xaxis=dict(title="Plan signal", **axis), yaxis=dict(title="Prepare signal", **axis),
        zaxis=dict(title="Execute signal", **axis),
        camera=dict(eye=dict(x=1.5, y=1.4, z=0.95), center=dict(x=0, y=0, z=-0.08))))
    save_fig(fig, OUT, "attack_phase_signal_cube_3d", html=True)


def main():
    data = json.loads(FILTERED.read_text())
    leaves = data["leaf_catalog"]
    configs = data["configurations"]
    man = data["manifest"]
    diag = data["diagnostics"]
    man = {**man, "diagnostics_exclusion": diag["exclusion_summary"]}

    print("Building ATTACK overview figures…")
    fig_sunburst(leaves)
    fig_treemap(leaves)
    fig_funnel(man)
    fig_signal(configs, man)
    fig_evidence(diag)
    fig_entropy(diag)
    fig_tactic(diag)
    fig_phase_cube(configs)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
