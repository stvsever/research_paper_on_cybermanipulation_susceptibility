from __future__ import annotations

"""
Integrated scenario explorer (bespoke interactive HTML)
=======================================================
Builds the flagship interactive view of the 10,000-scenario set:

  overview/integrated/htmls/integrated_factor_explorer_3d.html   interactive
  overview/integrated/pngs/integrated_factor_explorer_3d.png     static default

A hand-built HTML page with three axis selectors (X, Y, Z). Each selector is a
THREE-level hierarchical cascade that mirrors the REAL production ontologies:

  ontology · domain   →   construct   →   factor

  Profile : the real profile.json tree (domain → construct → ... → leaf), read
            from build_variable_plan so every factor keeps its true ontology path
  Attack  : the DISARM-red sample structure (signal/scores, tactics, route) since
            the attack ontology is external and not yet wired in
  Opinion : the Issue_Position_Taxonomy structure (issue domain, leaf count,
            adversarial-direction balance)

Each factor is typed (continuous / ordinal / categorical) and rendered properly:
continuous/interval axes are linear, ordinal axes use the construct order,
categorical axes are encoded to integer positions with the real category labels
on the axis plus a small jitter. Colour is fixed on the opinion cluster (the
issue domain) and the legend is click-toggleable.

Run by file path:
  python .../utils/build_integrated_explorer.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[6]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overview_theme import PAL, SEQ  # noqa: E402
from production_profile_sampling import build_variable_plan  # noqa: E402
from sample_high_res_profiles import prune_subtrees  # noqa: E402

STAGE = PROJECT_ROOT / "src" / "backend" / "pipeline" / "separate" / "01_create_scenarios"
INTEG = STAGE / "samples" / "02_integrated"
OUT = STAGE / "overview" / "integrated"
JSONL = INTEG / "integrated_scenarios_10000.jsonl"
PROFILE_ONT = PROJECT_ROOT / "src" / "backend" / "ontology" / "separate" / "production" / "PROFILE" / "profile.json"
PHASES = ("Plan", "Prepare", "Execute")
TRAITS = ("openness_to_experience", "conscientiousness", "extraversion",
          "agreeableness", "neuroticism")
EDU_ORDER = ["No_Formal_Education", "Primary", "Lower_Secondary", "Upper_Secondary",
             "Post_Secondary_Non_Tertiary", "Short_Cycle_Tertiary", "Bachelors", "Masters", "Doctoral"]


def humanize(s: str) -> str:
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


# ── real profile ontology path map (key -> ontology path) ────────────────────
def profile_path_map():
    """key -> (domain, construct_levels[list], leaf_label) from the real ontology,
    matching the sample's scope (Issue_Position_Taxonomy pruned)."""
    ont = json.loads(PROFILE_ONT.read_text())
    person = ont["PERSON"]
    prune_subtrees(person, {"Issue_Position_Taxonomy"})
    plan = build_variable_plan(person)
    key2path = {}
    for c in plan.categoricals:
        key2path[c.key] = c.path
    for s in plan.continuous_scales:
        for ck, lp in s.leaves:
            key2path[ck] = lp
    for o in plan.ordinals:
        key2path[o.column_key] = o.path
    for t in plan.trait_scalars:
        key2path[t.column_key] = t.path

    def split(key):
        path = key2path.get(key)
        if not path:
            return None
        parts = [p for p in path.split(">")]
        parts = [p.strip() for p in parts if p.strip() and p.strip() != "PERSON"]
        if not parts:
            return None
        domain = parts[0]
        leaf = parts[-1]
        construct = parts[1:-1]
        return domain, construct, leaf
    return split


# ── load + extract every factor as a column ──────────────────────────────────
def load_columns(limit=None):
    split = profile_path_map()
    cluster_list = []
    num_cols = defaultdict(list)   # (a, b, label, key) -> [floats]
    cat_raw = defaultdict(list)    # (a, b, label, key, ordinal) -> [strings]

    def add_num(a, b, label, key, val):
        num_cols[(a, b, label, key)].append(float(val) if val is not None and val == val else np.nan)

    def add_cat(a, b, label, key, val, ordinal=False):
        cat_raw[(a, b, label, key, ordinal)].append(str(val) if val is not None else "NA")

    def prof_group(key, fallback_a, fallback_b):
        s = split(key)
        if not s:
            return fallback_a, fallback_b, None
        domain, construct, leaf = s
        a = f"Profile · {humanize(domain)}"
        b = " › ".join(humanize(x) for x in construct) or "(direct)"
        return a, b, humanize(leaf)

    n = 0
    with JSONL.open() as fh:
        for line in fh:
            if limit and n >= limit:
                break
            r = json.loads(line); n += 1
            p = r["profile"]; d = p["demographics"]; bf = d.get("big_five", {})
            cluster_list.append(r["opinion_cluster"]["parent_name"])

            add_num("Profile · Demographics and Identity", "Chronological Age", "Age (years)",
                    "age_years", d["age_years"])
            for t in TRAITS:
                add_num("Profile · Personality", "Big Five", humanize(t), f"bf_{t}",
                        bf.get(t, {}).get("pct", np.nan))
            for k, v in p["numeric_attributes"].items():
                if k.endswith("_mean_pct"):
                    continue  # scale-mean aggregate, redundant with its facets
                a, b, lab = prof_group(k, "Profile · Other (numeric)", "(unmapped)")
                add_num(a, b, lab or humanize(k), k, v)
            for k, v in p["categorical_attributes"].items():
                if "administrative_and_data_context" in k or "_raw" in k:
                    continue
                a, b, lab = prof_group(k, "Profile · Other (categorical)", "(unmapped)")
                add_cat(a, b, lab or humanize(k), k, v, ordinal="highest_education" in k)

            a = r["attack"]
            add_num("Attack · DISARM-red", "Signal & scores", "Opinion-manipulation signal (total)",
                    "atk_signal", a["signal_total"])
            confs = []
            for ph in PHASES:
                add_num("Attack · DISARM-red", "Signal & scores", f"{ph} signal score",
                        f"atk_{ph}_sig", a["triplet"][ph]["signal_score"])
                confs.append(a["triplet"][ph]["confidence"])
            add_num("Attack · DISARM-red", "Signal & scores", "Mean leaf confidence",
                    "atk_conf", sum(confs) / len(confs))
            for ph in PHASES:
                add_cat("Attack · DISARM-red", "Tactics & route", f"{ph} tactic",
                        f"atk_{ph}_tac", a["triplet"][ph]["secondary"])
            add_cat("Attack · DISARM-red", "Tactics & route", "Inclusion route", "atk_route",
                    a["inclusion_route"])
            ps = {ph: a["triplet"][ph]["signal_score"] for ph in PHASES}
            add_cat("Attack · DISARM-red", "Tactics & route", "Dominant phase", "atk_dom",
                    max(ps, key=ps.get))

            oc = r["opinion_cluster"]
            add_cat("Opinion · Issue Position Taxonomy", "Issue domain", "Issue domain (cluster)",
                    "op_cluster", oc["parent_name"])
            add_num("Opinion · Issue Position Taxonomy", "Issue domain", "Leaf count", "op_nleaves",
                    oc["n_leaves"])
            amp = oc["direction_summary"]["amplify_+1"]; ero = oc["direction_summary"]["erode_-1"]
            add_num("Opinion · Issue Position Taxonomy", "Adversarial direction", "Amplify (+1) leaves",
                    "op_amp", amp)
            add_num("Opinion · Issue Position Taxonomy", "Adversarial direction", "Erode (−1) leaves",
                    "op_ero", ero)
            add_num("Opinion · Issue Position Taxonomy", "Adversarial direction",
                    "Net direction (amplify − erode)", "op_net", amp - ero)

    return cluster_list, num_cols, cat_raw, n


# ── build the typed/encoded factor registry ──────────────────────────────────
def build_factors(num_cols, cat_raw):
    factors = []
    fid = 0
    for (a, b, label, key), vals in num_cols.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0 or finite.min() == finite.max():
            continue
        dec = 0 if (finite.max() - finite.min()) >= 12 else 1
        factors.append({"id": f"f{fid}", "a": a, "b": b, "label": label, "type": "continuous",
                        "values": [None if not np.isfinite(v) else round(float(v), dec) for v in arr]})
        fid += 1
    for (a, b, label, key, ordinal), vals in cat_raw.items():
        counts = Counter(vals)
        if len(counts) <= 1:
            continue
        if ordinal and "highest_education" in key:
            cats = [c for c in EDU_ORDER if c in counts] + [c for c in counts if c not in EDU_ORDER]
            typ = "ordinal"
        else:
            cats = [c for c, _ in counts.most_common()]
            typ = "categorical"
        code = {c: i for i, c in enumerate(cats)}
        tick = [humanize(c) for c in cats] if len(cats) <= 16 else None
        factors.append({"id": f"f{fid}", "a": a, "b": b, "label": label, "type": typ,
                        "values": [code[v] for v in vals],
                        "ticktext": tick, "tickvals": list(range(len(cats))) if tick else None,
                        "labels": [humanize(c) for c in cats]})
        fid += 1
    return factors


def order_factors(factors):
    def akey(a):
        return (0 if a.startswith("Profile") else 1 if a.startswith("Attack") else 2, a)
    factors.sort(key=lambda f: (akey(f["a"]), f["b"], f["label"]))
    return factors


# ── static PNG (default view, plotly) ────────────────────────────────────────
def static_png(cluster_list, factors, clus, colors):
    import plotly.graph_objects as go
    from overview_theme import base_layout
    by_label = {f["label"]: f for f in factors}
    fx = by_label["Age (years)"]
    fy = by_label["Opinion-manipulation signal (total)"]
    fz = by_label["Openness to experience"]
    cidx = {c: i for i, c in enumerate(clus)}
    fig = go.Figure()
    for ci, cname in enumerate(clus):
        pts = [i for i, cl in enumerate(cluster_list) if cidx[cl] == ci]
        fig.add_trace(go.Scatter3d(
            x=[fx["values"][i] for i in pts], y=[fy["values"][i] for i in pts],
            z=[fz["values"][i] for i in pts], mode="markers",
            name=cname.replace("_And_", " & ").replace("_", " ")[:26],
            marker=dict(size=2.4, color=colors[ci], opacity=0.74, line=dict(width=0)), hoverinfo="skip"))
    axis = dict(backgroundcolor="#fbfcfe", gridcolor=PAL["line"], zerolinecolor=PAL["line"],
                showbackground=True, titlefont=dict(size=12))
    fig.update_layout(**base_layout(
        "Integrated scenario factor explorer (static default view)",
        "interactive HTML: any axis = any factor of the real profile / attack / opinion ontologies · "
        "colour = opinion issue domain · here age × attack signal × openness",
        h=820, w=1120))
    fig.update_layout(margin=dict(t=120, l=10, r=10, b=10),
                      legend=dict(itemsizing="constant", font=dict(size=9), y=0.5,
                                  bordercolor=PAL["line"], borderwidth=1),
                      scene=dict(domain=dict(x=[0.0, 0.82], y=[0.0, 1.0]),
                                 xaxis=dict(title="Profile age", **axis),
                                 yaxis=dict(title="Attack signal", **axis),
                                 zaxis=dict(title="Openness", **axis),
                                 camera=dict(eye=dict(x=1.55, y=1.45, z=0.92),
                                             center=dict(x=0, y=0, z=-0.05))))
    (OUT / "pngs").mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT / "pngs" / "integrated_factor_explorer_3d.png"), scale=2)
    print("  wrote pngs/integrated_factor_explorer_3d.png")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Integrated scenario factor explorer</title>
<script>__PLOTLYJS__</script>
<style>
  body { font-family:'Helvetica Neue',Arial,sans-serif; color:#14213d; margin:0; padding:16px 22px; background:#fff; }
  h1 { font-size:21px; margin:0 0 2px 0; }
  .sub { color:#5a6b86; font-size:13px; margin:0 0 12px 0; max-width:1150px; }
  #controls { display:flex; flex-direction:column; gap:7px; }
  .axis-row { display:flex; align-items:center; gap:8px; }
  .axis-badge { font-weight:700; width:18px; text-align:center; color:#fff; border-radius:4px; padding:3px 0; font-size:13px; }
  .bx { background:#1d4e89; } .by { background:#2a9d8f; } .bz { background:#e76f51; }
  select { font-size:12px; padding:5px 6px; border:1px solid #dbe3ef; border-radius:5px; background:#fff; color:#14213d; }
  select.sa { width:250px; } select.sb { width:300px; } select.sc { width:300px; }
  .arrow { color:#9aa7bd; font-size:12px; }
  .hint { font-size:11.5px; color:#5a6b86; margin-top:7px; max-width:1140px; line-height:1.45; }
  #plot { width:1180px; height:740px; }
</style></head>
<body>
<h1>Joint factor explorer: 10,000 opinion cognitive-warfare scenarios</h1>
<p class="sub">Each axis drills the real ontology: pick <b>ontology · domain → construct → factor</b>.
Colour = opinion issue domain (click the legend to toggle). Max-entropy-targeted factors (age, the Big
Five) fill their range evenly; others show their sampled distribution. The cloud stays independent
across ontologies.</p>
<div id="controls"></div>
<div class="hint" id="hint"></div>
<div id="plot"></div>
<script>
const CLUS = __CLUS__;            // opinion issue-domain names (colour groups)
const CLUS_COLORS = __CLUS_COLORS__;
const CL = __CL__;                // cluster index per scenario
const FACTORS = __FACTORS__;      // [{id,a,b,label,type,ticktext,tickvals,labels}]
const VALUES = __VALUES__;        // {id:[..N..]}
const N = CL.length;
const DEFAULTS = __DEFAULTS__;

const JIT = new Array(N);
let s = 1234567; for (let i=0;i<N;i++){ s=(1103515245*s+12345)&0x7fffffff; JIT[i]=(s/0x7fffffff-0.5)*0.6; }

const F_BY_ID = {}; FACTORS.forEach(f => F_BY_ID[f.id]=f);
// hierarchy: A -> [B], (A,B) -> [factor]
const A_LIST = []; const A2B = {}; const AB2F = {};
FACTORS.forEach(f => {
  if(!(f.a in A2B)){ A2B[f.a]=[]; A_LIST.push(f.a); }
  if(A2B[f.a].indexOf(f.b)<0) A2B[f.a].push(f.b);
  const kb = f.a+'|||'+f.b; if(!(kb in AB2F)) AB2F[kb]=[]; AB2F[kb].push(f);
});

const CL_IDX = CLUS.map(()=>[]);
for (let i=0;i<N;i++) CL_IDX[CL[i]].push(i);

function axisData(fid){
  const f=F_BY_ID[fid], v=VALUES[fid], cat=(f.type!=='continuous');
  return CL_IDX.map(idxs => idxs.map(i => { let x=v[i]; if(x===null) return null; return cat? x+JIT[i] : x; }));
}
function labelData(fid){
  const f=F_BY_ID[fid], v=VALUES[fid], cat=(f.type!=='continuous');
  return CL_IDX.map(idxs => idxs.map(i => cat ? (f.labels[v[i]] ?? '?') : v[i]));
}
function sceneAxis(fid){
  const f=F_BY_ID[fid];
  const ax={ title:{text:f.label,font:{size:12}}, backgroundcolor:'#fbfcfe', gridcolor:'#dbe3ef',
             zerolinecolor:'#dbe3ef', showbackground:true };
  if(f.type!=='continuous' && f.ticktext){ ax.tickmode='array'; ax.tickvals=f.tickvals; ax.ticktext=f.ticktext; }
  return ax;
}
function currentIds(){ return ['x','y','z'].map(a => document.getElementById('sc-'+a).value); }

function render(){
  const [xi,yi,zi]=currentIds();
  if(!xi||!yi||!zi) return;
  const xd=axisData(xi),yd=axisData(yi),zd=axisData(zi);
  const xl=labelData(xi),yl=labelData(yi),zl=labelData(zi);
  const fx=F_BY_ID[xi],fy=F_BY_ID[yi],fz=F_BY_ID[zi];
  const traces=CLUS.map((name,t)=>({
    type:'scatter3d', mode:'markers', name:(name.replace(/_And_/g,' & ').replace(/_/g,' ')).slice(0,30),
    x:xd[t], y:yd[t], z:zd[t],
    marker:{size:2.5, color:CLUS_COLORS[t], opacity:0.76, line:{width:0}},
    text:xd[t].map((_,k)=>`<b>${name.replace(/_/g,' ')}</b><br>${fx.label}: ${xl[t][k]}<br>${fy.label}: ${yl[t][k]}<br>${fz.label}: ${zl[t][k]}`),
    hovertemplate:'%{text}<extra></extra>'
  }));
  const layout={ margin:{t:6,l:6,r:6,b:6}, paper_bgcolor:'#fff', showlegend:true,
    legend:{itemsizing:'constant', font:{size:9}, y:0.5, title:{text:'Issue domain',font:{size:10}},
            bordercolor:'#dbe3ef', borderwidth:1},
    scene:{ domain:{x:[0,0.80],y:[0,1]}, aspectmode:'cube', xaxis:sceneAxis(xi), yaxis:sceneAxis(yi),
            zaxis:sceneAxis(zi), camera:{eye:{x:1.55,y:1.45,z:0.92}, center:{x:0,y:0,z:-0.05}} } };
  Plotly.react('plot', traces, layout, {responsive:true, displaylogo:false});
  document.getElementById('hint').innerHTML =
    '<b>X</b> '+fx.a+' › '+fx.b+' › '+fx.label+' ('+fx.type+')<br><b>Y</b> '+fy.a+' › '+fy.b+' › '+fy.label+' ('+fy.type+')<br><b>Z</b> '+fz.a+' › '+fz.b+' › '+fz.label+' ('+fz.type+')'
    + '<br>“Opinion-manipulation signal (total)” = heuristic strength (~5 to 26) of an attack triplet\\'s relation to opinion change; preserved from the filtered attack pool, not maximised.';
}

function buildAxis(letter, cls, defId){
  const def=F_BY_ID[defId];
  const row=document.createElement('div'); row.className='axis-row';
  const badge=document.createElement('span'); badge.className='axis-badge '+cls; badge.textContent=letter.toUpperCase();
  const sa=document.createElement('select'); sa.className='sa'; sa.id='sa-'+letter;
  const sb=document.createElement('select'); sb.className='sb'; sb.id='sb-'+letter;
  const sc=document.createElement('select'); sc.className='sc'; sc.id='sc-'+letter;
  A_LIST.forEach(a=>{ const o=document.createElement('option'); o.value=a; o.textContent=a; sa.appendChild(o); });
  function fillB(selB){ sb.innerHTML=''; A2B[sa.value].forEach(b=>{ const o=document.createElement('option'); o.value=b; o.textContent=b; sb.appendChild(o); }); if(selB) sb.value=selB; }
  function fillC(selC){ sc.innerHTML=''; (AB2F[sa.value+'|||'+sb.value]||[]).forEach(f=>{ const o=document.createElement('option'); o.value=f.id; o.textContent=f.label; sc.appendChild(o); }); if(selC) sc.value=selC; }
  sa.value=def.a; fillB(def.b); fillC(def.id);
  sa.addEventListener('change',()=>{ fillB(null); fillC(null); render(); });
  sb.addEventListener('change',()=>{ fillC(null); render(); });
  sc.addEventListener('change',render);
  [sa,sb,sc].forEach(el=>el.addEventListener('wheel',()=>el.blur()));
  const ar1=document.createElement('span'); ar1.className='arrow'; ar1.textContent='›';
  const ar2=document.createElement('span'); ar2.className='arrow'; ar2.textContent='›';
  row.append(badge,sa,ar1,sb,ar2,sc);
  document.getElementById('controls').appendChild(row);
}
buildAxis('x','bx',DEFAULTS[0]); buildAxis('y','by',DEFAULTS[1]); buildAxis('z','bz',DEFAULTS[2]);
render();
</script>
</body></html>
"""


def build_html(cluster_list, factors, clus, colors):
    cidx = {c: i for i, c in enumerate(clus)}
    cl = [cidx[c] for c in cluster_list]
    values = {f["id"]: f["values"] for f in factors}
    meta = [{k: f[k] for k in ("id", "a", "b", "label", "type", "ticktext", "tickvals", "labels") if k in f}
            for f in factors]
    by_label = {f["label"]: f["id"] for f in factors}
    defaults = [by_label["Age (years)"], by_label["Opinion-manipulation signal (total)"],
                by_label["Openness to experience"]]
    from plotly.offline import get_plotlyjs

    def j(x):
        return json.dumps(x, separators=(",", ":"))

    html = (HTML_TEMPLATE
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__CLUS__", j([c for c in clus]))
            .replace("__CLUS_COLORS__", j(colors))
            .replace("__CL__", j(cl))
            .replace("__FACTORS__", j(meta))
            .replace("__VALUES__", j(values))
            .replace("__DEFAULTS__", j(defaults)))
    (OUT / "htmls").mkdir(parents=True, exist_ok=True)
    (OUT / "htmls" / "integrated_factor_explorer_3d.html").write_text(html)
    print(f"  wrote htmls/integrated_factor_explorer_3d.html  ({len(html)/1e6:.1f} MB, "
          f"{len(factors)} factors, {len(cluster_list):,} points)")


def main():
    print("Loading integrated set (streaming the self-contained jsonl)…")
    cluster_list, num_cols, cat_raw, n = load_columns()
    factors = order_factors(build_factors(num_cols, cat_raw))
    clus = sorted(set(cluster_list))
    colors = [SEQ[i % len(SEQ)] for i in range(len(clus))]
    na = len({f["a"] for f in factors})
    print(f"  {n:,} scenarios · {len(factors)} factors in {na} ontology·domain groups · "
          f"colour = {len(clus)} opinion issue domains")
    print("Building explorer…")
    build_html(cluster_list, factors, clus, colors)
    static_png(cluster_list, factors, clus, colors)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
