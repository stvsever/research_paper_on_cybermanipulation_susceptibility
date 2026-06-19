"""
Interactive attack-effectivity dashboard — next-level visualization.

Tabs (generic for any run):
  🗂 Ontologies        → Ontology Explorer
  📡 Factorial Space   → Factorial 3D Surface, Factorial Heat + Contour
  🧠 SEM Analysis      → SEM Network (interactive), SEM Heatmap
  🔬 Estimation        → Conditional Susceptibility Estimator ★ (new), Perturbation Explorer
  👤 Profiles          → Susceptibility Map, Profile Heatmap
  📊 Moderators        → Moderator Forest, Hierarchical Importance
  📈 Raw Data          → Violin Distributions, Baseline vs Post
"""
from __future__ import annotations

import re
from collections import Counter
import json
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from src.backend.utils.data_utils import infer_analysis_mode


# ─── palette & helpers ────────────────────────────────────────────────────────

PALETTE = dict(
    navy="#0f2240", blue="#1d4e89", sky="#2980b9",
    teal="#2a9d8f", orange="#e76f51", red="#c0392b",
    amber="#c89b3c", ink="#14213d", muted="#4a5d7a",
    panel="#ffffff", line="#dbe3ef", gold="#f0c040",
)


def _leaf(s: str) -> str:
    raw = s.rsplit(">", 1)[-1].strip() if ">" in str(s) else str(s)
    return raw.replace("_", " ").strip()


def _pretty(s: str) -> str:
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_z", "").replace("_", " ").replace("  ", " ").strip().title()


def _pretty_indicator(s: str) -> str:
    for prefix in ["adversarial_delta_indicator__", "abs_delta_indicator__"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.replace("_", " ").replace("  ", " ").strip().title()


def _wrap_label(s: str, width: int = 18) -> str:
    text = str(s).replace("<br>", " ").strip()
    lines = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return "<br>".join(lines) if lines else text


def _clip_label(s: str, max_len: int = 42) -> str:
    text = re.sub(r"\s+", " ", str(s)).strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _path_parts(s: str) -> List[str]:
    raw = str(s)
    if ">" in raw:
        return [_leaf(part) for part in raw.split(">") if str(part).strip()]
    return [_leaf(raw)]


def _path_context(s: str, keep: int = 2) -> str:
    parts = _path_parts(s)
    if len(parts) <= 1:
        return ""
    trimmed = [
        part for idx, part in enumerate(parts[:-1])
        if idx > 0 or part.lower() not in {"attack vectors", "issue position taxonomy", "profile"}
    ]
    return " / ".join(trimmed[-keep:])


def _unique_display_map(values: List[str]) -> Dict[str, str]:
    leaves = [_leaf(v) for v in values]
    counts = Counter(leaves)
    labels: Dict[str, str] = {}
    for value, leaf_name in zip(values, leaves):
        if counts[leaf_name] <= 1:
            labels[value] = leaf_name
            continue
        context = _path_context(value, keep=1)
        labels[value] = f"{context} • {leaf_name}" if context else leaf_name
    return labels


def _moderator_hierarchy(label: str, ontology_group: str | None = None) -> List[str]:
    group_parts = [part.strip() for part in str(ontology_group or "").split(":") if part.strip()]
    clean_label = re.sub(r"\s+", " ", str(label)).strip()
    leaf_label = clean_label
    if group_parts:
        last = group_parts[-1]
        leaf_label = re.sub(rf"(?i)\b{re.escape(last)}\b", "", leaf_label, count=1)
        leaf_label = re.sub(r"(?i)\bBig Five\b", "", leaf_label)
        leaf_label = re.sub(r"\s+", " ", leaf_label).strip(" -:%")
    if not leaf_label:
        leaf_label = clean_label

    segments = ["Profile Features"]
    segments.extend(group_parts if group_parts else ["Other Moderators"])
    if not segments or segments[-1] != leaf_label:
        segments.append(leaf_label)
    deduped: List[str] = []
    for seg in segments:
        if not deduped or deduped[-1] != seg:
            deduped.append(seg)
    return deduped


def _network_ontology_family(ontology_group: str) -> str:
    text = str(ontology_group or "").strip()
    return text.split(":", 1)[0].strip() if ":" in text else (text or "Other")


def _network_feature_type(term: str) -> str:
    raw = str(term or "").lower()
    if raw.startswith("profile_cat__"):
        return "Categorical dummy"
    if "chronological_age" in raw or "age_years" in raw:
        return "Continuous demographic"
    if "_mean_pct" in raw:
        if "big_five" in raw:
            return "Trait aggregate"
        return "Scale aggregate"
    if "big_five" in raw:
        return "Facet"
    return "Continuous subscale"


def _infer_sem_moderator_groups(label: str) -> Tuple[str, str]:
    txt = re.sub(r"\s+", " ", str(label)).strip()
    if txt.startswith("Big Five "):
        remainder = txt[len("Big Five "):].replace("%", "").strip()
        subgroup = remainder.split(" Mean")[0].strip()
        return "Profile Traits", subgroup or "Big Five"
    if txt.startswith("Sex "):
        return "Demographics", "Sex"
    if "Age" in txt:
        return "Demographics", "Age"
    if "Baseline" in txt:
        return "Model Controls", "Baseline"
    if "Exposure" in txt or "Realism" in txt or "Plausibility" in txt:
        return "Model Controls", "Exposure / Quality"
    return "Other Moderators", txt.split(" ")[0]


def _humanize_ontology_label(label: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(label).replace("_", " ").strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text.lower() == text:
        specials = {"ai", "ui", "id", "uk", "us", "eu", "vat", "ngo", "api"}
        parts = [
            part.upper() if part.lower() in specials else part.capitalize()
            for part in text.split(" ")
        ]
        return " ".join(parts)
    return text


def _split_ontology_children(node: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(node, dict):
        return {}, {"value": node}
    structural: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for key, value in node.items():
        if str(key).startswith("_"):
            metadata[str(key)] = value
        elif isinstance(value, dict):
            structural[str(key)] = value
        else:
            metadata[str(key)] = value
    return structural, metadata


def _build_ontology_payload(
    tree: Dict[str, Any],
    *,
    env: str,
    ontology_key: str,
    sampled_paths: set[str],
    sampled_leaf_names: set[str],
) -> Dict[str, Any]:
    root_label = {
        "ATTACK": "Attack Ontology",
        "OPINION": "Opinion Ontology",
        "PROFILE": "Profile Ontology",
    }.get(ontology_key, ontology_key.title())

    nodes: List[Dict[str, Any]] = []
    node_counter = 0

    def _walk(
        label: str,
        payload: Any,
        parent_id: Optional[str],
        raw_parts: Tuple[str, ...],
        depth: int,
    ) -> Tuple[str, int, int, int, int, int, int]:
        nonlocal node_counter
        node_id = f"{env}:{ontology_key}:{node_counter}"
        node_counter += 1

        structural, metadata = _split_ontology_children(payload)
        metadata_preview: List[List[str]] = []
        for key, value in list(metadata.items())[:6]:
            if isinstance(value, (dict, list)):
                value_txt = json.dumps(value, ensure_ascii=False)
            else:
                value_txt = str(value)
            metadata_preview.append([
                _humanize_ontology_label(key),
                _clip_label(value_txt.replace("\n", " "), 120),
            ])

        raw_path = " > ".join(raw_parts)
        human_path = " > ".join(_humanize_ontology_label(part) for part in raw_parts) if raw_parts else root_label
        display_label = _humanize_ontology_label(label)

        node: Dict[str, Any] = {
            "id": node_id,
            "parent": parent_id,
            "name": str(label),
            "label": display_label,
            "short": _clip_label(display_label, 30),
            "tiny": _clip_label(display_label, 18),
            "depth": depth,
            "path": raw_path,
            "path_label": human_path,
            "children": [],
            "metadata_preview": metadata_preview,
            "metadata_count": len(metadata),
            "kind": "branch",
            "sample_exact": False,
            "sample_aligned": False,
        }
        nodes.append(node)

        subtree_nodes = 1
        subtree_leaves = 0
        subtree_metadata_leaves = 0
        subtree_exact = 0
        subtree_aligned = 0
        max_depth = depth

        if not structural:
            node["kind"] = "leaf_meta" if metadata else "leaf"
            subtree_leaves = 1
            if metadata:
                subtree_metadata_leaves = 1
            leaf_token = raw_parts[-1] if raw_parts else str(label)
            node["sample_exact"] = bool(raw_path and raw_path in sampled_paths)
            node["sample_aligned"] = bool(
                raw_path and not node["sample_exact"] and leaf_token in sampled_leaf_names
            )
            subtree_exact = 1 if node["sample_exact"] else 0
            subtree_aligned = 1 if node["sample_aligned"] else 0
        else:
            for child_name, child_payload in structural.items():
                child_id, n_nodes, n_leaves, n_meta_leaves, child_max_depth, n_exact, n_aligned = _walk(
                    child_name,
                    child_payload,
                    node_id,
                    raw_parts + (child_name,),
                    depth + 1,
                )
                node["children"].append(child_id)
                subtree_nodes += n_nodes
                subtree_leaves += n_leaves
                subtree_metadata_leaves += n_meta_leaves
                subtree_exact += n_exact
                subtree_aligned += n_aligned
                max_depth = max(max_depth, child_max_depth)

        node["child_count"] = len(node["children"])
        node["leaf_count"] = subtree_leaves
        node["subtree_node_count"] = subtree_nodes
        node["metadata_leaf_count"] = subtree_metadata_leaves
        node["max_subtree_depth"] = max_depth
        node["sample_exact_subtree"] = subtree_exact
        node["sample_aligned_subtree"] = subtree_aligned

        return (
            node_id,
            subtree_nodes,
            subtree_leaves,
            subtree_metadata_leaves,
            max_depth,
            subtree_exact,
            subtree_aligned,
        )

    root_id, node_count, leaf_count, metadata_leaf_count, max_depth, exact_count, aligned_count = _walk(
        root_label,
        tree,
        None,
        (),
        0,
    )

    branch_count = sum(1 for node in nodes if node["kind"] == "branch")
    recommended_depth = 2 if leaf_count > 1500 else 3 if leaf_count > 200 else min(4, max_depth)

    return {
        "root_id": root_id,
        "nodes": nodes,
        "summary": {
            "node_count": int(node_count),
            "leaf_count": int(leaf_count),
            "branch_count": int(branch_count),
            "metadata_leaf_count": int(metadata_leaf_count),
            "max_depth": int(max_depth),
            "recommended_depth": int(max(recommended_depth, 1)),
            "sample_exact_count": int(exact_count),
            "sample_aligned_count": int(aligned_count),
        },
    }


def _load_dashboard_ontology_payload(ontology_catalog: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve the repo root by walking up to the directory that actually holds
    # src/backend/ontology, so the explorer keeps finding the ontology JSONs
    # regardless of how deep this module is nested under src/backend/utils/...
    here = Path(__file__).resolve()
    project_root = next(
        (parent for parent in here.parents if (parent / "src" / "backend" / "ontology").is_dir()),
        here.parents[4] if len(here.parents) > 4 else here.parents[-1],
    )
    ontology_root = str(ontology_catalog.get("ontology_root", ""))
    run_source = "test" if (("separate/test" in ontology_root) or ("01_separated/test" in ontology_root)) else "production"

    selected_attack_paths = {str(v) for v in ontology_catalog.get("selected_attack_leaves", [])}
    if not selected_attack_paths and ontology_catalog.get("selected_attack_leaf"):
        selected_attack_paths.add(str(ontology_catalog["selected_attack_leaf"]))
    selected_opinion_paths = {str(v) for v in ontology_catalog.get("selected_opinion_leaves", [])}

    sampled_paths_by_key: Dict[str, set[str]] = {
        "ATTACK": selected_attack_paths,
        "OPINION": selected_opinion_paths,
        "PROFILE": set(),
    }
    sampled_leaf_names_by_key: Dict[str, set[str]] = {
        key: {path.split(">")[-1].strip() for path in paths if str(path).strip()}
        for key, paths in sampled_paths_by_key.items()
    }

    sources: Dict[str, Dict[str, Any]] = {}
    for env in ("production", "test"):
        # Directory is `separate` (legacy runs used `01_separated`); support both.
        env_root = project_root / "src" / "backend" / "ontology" / "separate" / env
        if not env_root.exists():
            env_root = project_root / "src" / "backend" / "ontology" / "01_separated" / env
        env_sources: Dict[str, Any] = {}
        for ontology_key, rel in {
            "ATTACK": env_root / "ATTACK" / "attack.json",
            "OPINION": env_root / "OPINION" / "opinion.json",
            "PROFILE": env_root / "PROFILE" / "profile.json",
        }.items():
            # The integrated run draws attacks from an external DISARM ontology, so
            # a given ontology JSON may be absent; skip it rather than crashing the
            # whole dashboard build.
            if not rel.exists():
                continue
            raw_tree = json.loads(rel.read_text(encoding="utf-8"))
            env_sources[ontology_key] = _build_ontology_payload(
                raw_tree,
                env=env,
                ontology_key=ontology_key,
                sampled_paths=sampled_paths_by_key.get(ontology_key, set()),
                sampled_leaf_names=sampled_leaf_names_by_key.get(ontology_key, set()),
            )
        sources[env] = env_sources

    return {
        "current_run_source": run_source,
        "selected_paths": {
            "ATTACK": sorted(selected_attack_paths),
            "OPINION": sorted(selected_opinion_paths),
            "PROFILE": [],
        },
        "sources": sources,
    }


def _build_tree_nodes(paths_with_values: List[Tuple[List[str], float]]) -> Tuple[List[str], List[str], List[str], List[float], List[str]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for path, value in paths_with_values:
        for idx, segment in enumerate(path):
            node_id = " | ".join(path[: idx + 1])
            parent_id = " | ".join(path[:idx]) if idx else ""
            if node_id not in nodes:
                nodes[node_id] = {
                    "label": segment,
                    "parent": parent_id,
                    "value": 0.0,
                    "path": " → ".join(path[: idx + 1]),
                }
            nodes[node_id]["value"] += float(value)
    ids = list(nodes.keys())
    labels = [nodes[node_id]["label"] for node_id in ids]
    parents = [nodes[node_id]["parent"] for node_id in ids]
    values = [nodes[node_id]["value"] for node_id in ids]
    paths = [nodes[node_id]["path"] for node_id in ids]
    return ids, labels, parents, values, paths


def _pretty_moderator_label(column_name: str) -> str:
    """Human-readable label for a profile feature column (mirror of stage 06)."""
    label = str(column_name)
    for prefix in ["profile_cont_", "profile_cat__profile_cat_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = label.replace("_z", "").replace("__", " ").replace("_", " ").strip()
    return " ".join(part.capitalize() if part.lower() != "pct" else "%" for part in label.split())


def _p_stars(p: Any) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""


def _safe_col(df: pd.DataFrame, col: str, fallback: Any = None):
    return df[col] if col in df.columns else (fallback if fallback is not None else pd.Series(dtype=float))


def _apply_style(fig: go.Figure, height: int = 520) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font=dict(family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif", size=12),
        height=height, margin=dict(l=60, r=30, t=52, b=50),
        dragmode="pan",
    )
    try:
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
    except Exception:
        pass
    return fig


PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoom2d", "autoScale2d"],
}


def _save_figure_html(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True, config=PLOTLY_CONFIG)
    return str(path)


def _save_html_block(content: str, path: Path, title: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plotly_js = get_plotlyjs()
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<style>
body{{margin:0;padding:16px;background:#ffffff;font-family:"IBM Plex Sans","Avenir Next","Segoe UI",sans-serif;color:#14213d;}}
</style>
<script>{plotly_js}</script>
</head>
<body>
{content}
</body>
</html>""",
        encoding="utf-8",
    )
    return str(path)


# ─── figure builders ──────────────────────────────────────────────────────────

def _html_ontology_explorer(ontology_payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(ontology_payload, ensure_ascii=False)
    run_source = str(ontology_payload.get("current_run_source", "production")).lower()
    run_source = run_source if run_source in ("production", "test") else "production"
    other_source = "test" if run_source == "production" else "production"
    source_btn_html = "".join(
        f'<button class="ontx-btn{" active" if name == run_source else ""}" data-source="{name}">{name.capitalize()}</button>'
        for name in ("production", "test")
    )
    source_sub = (
        f"This run uses the <strong>{run_source}</strong> ontology, shown by default. "
        f"The toggle lets you compare it against the {other_source} hierarchy."
    )
    return f"""
<div id="ontx-root">
  <style>
    #ontx-root .ontx-shell{{display:grid;grid-template-columns:minmax(300px,340px) minmax(0,1fr);gap:16px;align-items:start}}
    #ontx-root .ontx-card,#ontx-root .ontx-panel,#ontx-root .ontx-canvas-card{{background:#f7faff;border:1px solid #dbe3ef;border-radius:14px;box-shadow:0 3px 14px rgba(20,33,61,0.05)}}
    #ontx-root .ontx-card{{padding:12px 13px}}
    #ontx-root .ontx-card + .ontx-card{{margin-top:10px}}
    #ontx-root .ontx-title{{font-weight:800;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px}}
    #ontx-root .ontx-sub{{font-size:0.76rem;line-height:1.45;color:{PALETTE['muted']};margin-bottom:8px}}
    #ontx-root .ontx-segment{{display:flex;flex-wrap:wrap;gap:6px}}
    #ontx-root .ontx-btn{{padding:6px 10px;border-radius:999px;border:1px solid #c8d7ec;background:#fff;color:{PALETTE['ink']};cursor:pointer;font-size:0.75rem;font-weight:700}}
    #ontx-root .ontx-btn.active{{background:{PALETTE['blue']};border-color:{PALETTE['blue']};color:#fff}}
    #ontx-root .ontx-focus{{display:flex;flex-direction:column;gap:8px}}
    #ontx-root .ontx-focus-item{{padding:10px 11px;border-radius:12px;border:1px solid #dbe3ef;background:#fff;cursor:pointer;transition:transform 120ms ease,border-color 120ms ease,box-shadow 120ms ease}}
    #ontx-root .ontx-focus-item:hover{{transform:translateY(-1px);box-shadow:0 6px 14px rgba(20,33,61,0.08)}}
    #ontx-root .ontx-focus-item.active{{border-color:{PALETTE['blue']};box-shadow:0 0 0 2px rgba(29,78,137,0.08)}}
    #ontx-root .ontx-focus-top{{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:5px}}
    #ontx-root .ontx-focus-top strong{{font-size:0.82rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-focus-pill{{display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;font-size:0.66rem;font-weight:800;letter-spacing:0.02em}}
    #ontx-root .ontx-focus-meta{{display:flex;gap:10px;flex-wrap:wrap;font-size:0.72rem;color:{PALETTE['muted']}}}
    #ontx-root .ontx-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    #ontx-root .ontx-select{{width:100%;padding:7px 8px;border-radius:9px;border:1px solid #dbe3ef;background:#fff;color:{PALETTE['ink']};font-size:0.80rem}}
    #ontx-root .ontx-slider-wrap{{background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:9px 10px}}
    #ontx-root .ontx-slider-meta{{display:flex;justify-content:space-between;gap:10px;align-items:center;font-size:0.76rem;color:{PALETTE['muted']};font-weight:700;margin-bottom:6px}}
    #ontx-root input[type="range"]{{width:100%;accent-color:{PALETTE['blue']}}}
    #ontx-root .ontx-toggle{{display:flex;align-items:center;gap:7px;font-size:0.77rem;color:{PALETTE['ink']};font-weight:600}}
    #ontx-root .ontx-toggle-list{{display:flex;flex-direction:column;gap:8px}}
    #ontx-root .ontx-search{{width:100%;padding:9px 10px;border-radius:10px;border:1px solid #dbe3ef;background:#fff;font-size:0.82rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-results{{display:flex;flex-direction:column;gap:7px;max-height:220px;overflow:auto;margin-top:10px;padding-right:4px}}
    #ontx-root .ontx-result{{padding:8px 9px;border-radius:10px;background:#fff;border:1px solid #dbe3ef;cursor:pointer}}
    #ontx-root .ontx-result strong{{display:block;font-size:0.77rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-result span{{display:block;font-size:0.71rem;color:{PALETTE['muted']};line-height:1.4;margin-top:3px}}
    #ontx-root .ontx-stage{{display:flex;flex-direction:column;gap:12px}}
    #ontx-root .ontx-banner{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%);border:1px solid #dbe3ef;border-radius:12px;padding:11px 13px}}
    #ontx-root .ontx-status{{font-size:0.79rem;line-height:1.5;color:{PALETTE['muted']}}}
    #ontx-root .ontx-status strong{{color:{PALETTE['ink']}}}
    #ontx-root .ontx-legend{{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}}
    #ontx-root .ontx-legend-item{{display:flex;align-items:center;gap:6px;font-size:0.72rem;color:{PALETTE['muted']}}}
    #ontx-root .ontx-swatch{{display:inline-block;width:28px;height:10px;border-radius:999px}}
    #ontx-root .ontx-swatch.attack{{background:linear-gradient(90deg,rgba(231,111,81,0.22),rgba(231,111,81,0.92))}}
    #ontx-root .ontx-swatch.opinion{{background:linear-gradient(90deg,rgba(42,157,143,0.22),rgba(42,157,143,0.92))}}
    #ontx-root .ontx-swatch.profile{{background:linear-gradient(90deg,rgba(29,78,137,0.22),rgba(29,78,137,0.92))}}
    #ontx-root .ontx-compare{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}
    #ontx-root .ontx-compare-card{{padding:11px 12px;border-radius:12px;border:1px solid #dbe3ef;background:#fff}}
    #ontx-root .ontx-compare-card.active{{border-color:{PALETTE['blue']};box-shadow:0 0 0 2px rgba(29,78,137,0.08)}}
    #ontx-root .ontx-compare-top{{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:7px}}
    #ontx-root .ontx-compare-top strong{{font-size:0.82rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-chip{{display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;font-size:0.66rem;font-weight:800;letter-spacing:0.02em}}
    #ontx-root .ontx-compare-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}}
    #ontx-root .ontx-metric{{padding:7px 8px;border-radius:10px;background:#f8fbff;border:1px solid #e2eaf5}}
    #ontx-root .ontx-metric .k{{font-size:0.67rem;text-transform:uppercase;letter-spacing:0.06em;color:{PALETTE['muted']}}}
    #ontx-root .ontx-metric .v{{font-size:0.90rem;font-weight:800;color:{PALETTE['ink']};margin-top:2px}}
    #ontx-root .ontx-canvas-card{{padding:0;overflow:hidden}}
    #ontx-root .ontx-canvas-head{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:12px 14px;border-bottom:1px solid #dbe3ef;background:linear-gradient(180deg,#ffffff 0%,#fbfdff 100%)}}
    #ontx-root .ontx-canvas-head strong{{display:block;font-size:0.94rem;color:{PALETTE['navy']}}}
    #ontx-root .ontx-canvas-head span{{display:block;font-size:0.74rem;color:{PALETTE['muted']};margin-top:3px;line-height:1.45}}
    #ontx-root .ontx-chipline{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}}
    #ontx-root .ontx-chipline .ontx-chip{{background:rgba(29,78,137,0.08);color:{PALETTE['blue']}}}
    #ontx-root #ontx-canvas-wrap{{background:
      radial-gradient(circle at 12% 16%, rgba(42,157,143,0.08), transparent 28%),
      radial-gradient(circle at 88% 14%, rgba(231,111,81,0.08), transparent 26%),
      linear-gradient(180deg,#ffffff 0%,#f9fbff 100%);
      height:780px;overflow:auto;padding:16px;
      cursor:grab;utils-select:none;-webkit-utils-select:none}}
    #ontx-root #ontx-svg{{display:block;margin:auto}}
    #ontx-root #ontx-canvas-wrap{{display:flex;align-items:safe center;justify-content:safe center}}
    #ontx-root .ontx-bottom{{display:grid;grid-template-columns:1.15fr 1fr 0.95fr;gap:12px}}
    #ontx-root .ontx-panel{{padding:12px 13px}}
    #ontx-root .ontx-panel h4{{margin:0 0 8px;font-size:0.82rem;color:{PALETTE['navy']}}}
    #ontx-root .ontx-kv{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}
    #ontx-root .ontx-kv .ontx-metric{{padding:8px 9px}}
    #ontx-root .ontx-meta-list{{display:flex;flex-direction:column;gap:6px;margin-top:10px}}
    #ontx-root .ontx-meta-item{{padding:7px 8px;border-radius:10px;background:#fff;border:1px solid #dbe3ef}}
    #ontx-root .ontx-meta-item strong{{display:block;font-size:0.75rem;color:{PALETTE['ink']};margin-bottom:2px}}
    #ontx-root .ontx-meta-item span{{font-size:0.72rem;color:{PALETTE['muted']};line-height:1.4;word-break:break-word}}
    #ontx-root .ontx-path{{padding:8px 9px;border-radius:10px;background:#fff;border:1px solid #dbe3ef;font-size:0.73rem;line-height:1.45;color:{PALETTE['muted']};word-break:break-word}}
    #ontx-root .ontx-highlight-list{{display:flex;flex-direction:column;gap:7px}}
    #ontx-root .ontx-highlight-item{{padding:8px 9px;border-radius:10px;background:#fff;border:1px solid #dbe3ef}}
    #ontx-root .ontx-highlight-item strong{{display:block;font-size:0.76rem;color:{PALETTE['ink']}}}
    #ontx-root .ontx-highlight-item span{{display:block;font-size:0.71rem;color:{PALETTE['muted']};line-height:1.4;margin-top:3px}}
    #ontx-root .ontx-note{{font-size:0.73rem;line-height:1.45;color:{PALETTE['muted']}}}
    #ontx-root .ontx-legend-stack{{display:flex;flex-direction:column;gap:8px}}
    #ontx-root .ontx-legend-row{{display:flex;align-items:center;gap:9px;font-size:0.74rem;color:{PALETTE['muted']}}}
    #ontx-root .ontx-node-demo{{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:999px;flex:0 0 auto}}
    #ontx-root .ontx-actions{{display:flex;flex-wrap:wrap;gap:6px}}
    @media (max-width: 1180px) {{
      #ontx-root .ontx-shell{{grid-template-columns:1fr}}
      #ontx-root .ontx-bottom{{grid-template-columns:1fr}}
      #ontx-root .ontx-compare{{grid-template-columns:1fr}}
    }}
  </style>

  <div class="ontx-shell">
    <div>
      <div class="ontx-card">
        <div class="ontx-title">Ontology Source</div>
        <div class="ontx-sub">{source_sub}</div>
        <div class="ontx-segment" id="ontx-source">
          {source_btn_html}
        </div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Ontology Focus</div>
        <div class="ontx-sub">Choose which hierarchical state space to inspect: cybermanipulation attacks, opinion targets, or profile space.</div>
        <div id="ontx-focus" class="ontx-focus"></div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Layout</div>
        <div class="ontx-sub">Switch between directional tree flow, top-down structure, and radial orbit when the hierarchy gets dense.</div>
        <div class="ontx-segment" id="ontx-layout">
          <button class="ontx-btn active" data-layout="flow">Left → Right</button>
          <button class="ontx-btn" data-layout="vertical">Top ↓ Bottom</button>
          <button class="ontx-btn" data-layout="radial">Radial</button>
        </div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Depth & Labels</div>
        <div class="ontx-slider-wrap">
          <div class="ontx-slider-meta"><span>Visible depth</span><span id="ontx-depth-display">3</span></div>
          <input type="range" id="ontx-depth" min="1" max="8" step="1" value="3">
        </div>
        <div class="ontx-sub" style="margin:10px 0 6px">Label density</div>
        <select id="ontx-label-mode" class="ontx-select">
          <option value="compact" selected>Compact</option>
          <option value="branches">Branches only</option>
          <option value="full">Full labels</option>
        </select>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Visibility</div>
        <div class="ontx-toggle-list">
          <label class="ontx-toggle"><input type="checkbox" id="ontx-show-meta" checked> Show metadata halos for annotated leaves</label>
          <label class="ontx-toggle"><input type="checkbox" id="ontx-highlight-run" checked> Highlight run-aligned leaves and branch matches</label>
          <label class="ontx-toggle"><input type="checkbox" id="ontx-relevant-only"> Restrict view to run-aligned branches</label>
        </div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Search</div>
        <input id="ontx-search" class="ontx-search" type="text" placeholder="Search branches, leaves, or paths">
        <div id="ontx-results" class="ontx-results"></div>
      </div>

      <div class="ontx-card">
        <div class="ontx-title">Actions</div>
        <div class="ontx-actions">
          <button class="ontx-btn" id="ontx-expand-all">Expand all</button>
          <button class="ontx-btn" id="ontx-collapse-all">Collapse all</button>
          <button class="ontx-btn" id="ontx-reset-depth">Reset to depth</button>
          <button class="ontx-btn" id="ontx-zoom-out">Zoom −</button>
          <button class="ontx-btn" id="ontx-zoom-in">Zoom +</button>
          <button class="ontx-btn" id="ontx-zoom-reset">100%</button>
        </div>
      </div>
    </div>

    <div class="ontx-stage">
      <div class="ontx-banner">
        <div class="ontx-status" id="ontx-status"></div>
        <div class="ontx-legend">
          <div class="ontx-legend-item"><span class="ontx-swatch attack"></span><span>attack ontology accent</span></div>
          <div class="ontx-legend-item"><span class="ontx-swatch opinion"></span><span>opinion ontology accent</span></div>
          <div class="ontx-legend-item"><span class="ontx-swatch profile"></span><span>profile ontology accent</span></div>
        </div>
      </div>

      <div id="ontx-compare" class="ontx-compare"></div>

      <div class="ontx-canvas-card">
        <div class="ontx-canvas-head">
          <div>
            <strong id="ontx-active-title"></strong>
            <span id="ontx-active-sub"></span>
          </div>
          <div class="ontx-chipline" id="ontx-chipline"></div>
        </div>
        <div id="ontx-canvas-wrap"></div>
      </div>

      <div class="ontx-bottom">
        <div class="ontx-panel">
          <h4>Selected Node</h4>
          <div id="ontx-inspector"></div>
        </div>
        <div class="ontx-panel">
          <h4>Path Highlights</h4>
          <div id="ontx-highlights" class="ontx-highlight-list"></div>
        </div>
        <div class="ontx-panel">
          <h4>Legend & Use</h4>
          <div class="ontx-legend-stack">
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px solid {PALETTE['ink']}"></span><span>Branch node. Click to expand or collapse the subtree.</span></div>
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px dashed {PALETTE['amber']}"></span><span>Dashed halo marks annotated test leaves with metadata such as descriptions or adversarial direction.</span></div>
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px solid {PALETTE['gold']};box-shadow:0 0 0 3px rgba(240,192,64,0.16)"></span><span>Gold ring marks exact run-aligned leaves in the ontology source used by the run.</span></div>
            <div class="ontx-legend-row"><span class="ontx-node-demo" style="background:{PALETTE['panel']};border:2px solid {PALETTE['amber']}"></span><span>Amber ring marks leaf-name alignment across sources, useful when the production ontology extends the test ontology.</span></div>
            <div class="ontx-note">Zoom with ctrl+wheel or cmd+wheel (trackpad pinch works), double-click empty canvas to zoom in (alt+double-click out), drag empty canvas to pan, drag any node to reposition it. Clicking a node lights its root path gold with flowing particles; every traversal takes the same time, so longer paths flow faster.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
  (function(){{
    const DATA = {payload_json};
    const root = document.getElementById('ontx-root');
    const COLORS = {{
      ATTACK:  {{ hue: 18,  accent: '{PALETTE['orange']}', soft: 'rgba(231,111,81,0.14)', edge: 'rgba(231,111,81,0.24)' }},
      OPINION: {{ hue: 168, accent: '{PALETTE['teal']}',   soft: 'rgba(42,157,143,0.14)', edge: 'rgba(42,157,143,0.24)' }},
      PROFILE: {{ hue: 214, accent: '{PALETTE['blue']}',   soft: 'rgba(29,78,137,0.14)', edge: 'rgba(29,78,137,0.22)' }},
    }};
    const expanded = {{}};
    const depthStore = {{}};
    const state = {{
      source: '{run_source}',
      ontology: 'ATTACK',
      layout: 'flow',
      maxDepth: 3,
      labelMode: 'compact',
      showMetadata: true,
      highlightRun: true,
      relevantOnly: false,
      zoom: 1,
      search: '',
      selectedId: null,
    }};

    // Self-heal the starting selection: pick a source that actually carries
    // ontologies and a focus that exists within it. This keeps the explorer
    // rendering even when a run ships a partial set (e.g. attacks drawn from an
    // external DISARM ontology that is not vendored into this repo).
    (function normalizeState() {{
      const srcs = DATA.sources || {{}};
      const nonEmpty = Object.keys(srcs).filter(s => srcs[s] && Object.keys(srcs[s]).length);
      if (!nonEmpty.includes(state.source)) state.source = nonEmpty[0] || state.source;
      const focusKeys = Object.keys(srcs[state.source] || {{}});
      if (focusKeys.length && !focusKeys.includes(state.ontology)) state.ontology = focusKeys[0];
    }})();

    function escapeHtml(txt) {{
      return String(txt ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}
    function datasetKey() {{
      return `${{state.source}}:${{state.ontology}}`;
    }}
    function dataset() {{
      return DATA.sources[state.source][state.ontology];
    }}
    function activeSpec() {{
      return COLORS[state.ontology];
    }}
    function initDatasets() {{
      Object.entries(DATA.sources).forEach(([env, envObj]) => {{
        Object.entries(envObj).forEach(([ontology, ds]) => {{
          ds.nodeMap = Object.fromEntries(ds.nodes.map(n => [n.id, n]));
          const key = `${{env}}:${{ontology}}`;
          depthStore[key] = ds.summary.recommended_depth || 3;
          /* Always open layers 1 & 2: expand all branch nodes at depth 0 and 1 */
          expanded[key] = new Set(
            ds.nodes
              .filter(n => n.kind === 'branch' && n.depth <= 1)
              .map(n => n.id)
          );
        }});
      }});
      state.maxDepth = depthStore[datasetKey()] || 3;
      state.selectedId = dataset().root_id;
    }}
    function badgeHtml(text, bg, fg) {{
      return `<span class="ontx-chip" style="background:${{bg}};color:${{fg}}">${{escapeHtml(text)}}</span>`;
    }}
    function setSource(source) {{
      state.source = source;
      state.maxDepth = depthStore[datasetKey()] || dataset().summary.recommended_depth || 3;
      root.querySelector('#ontx-depth').value = state.maxDepth;
      root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
      state.selectedId = dataset().root_id;
      syncButtons();
      renderAll();
    }}
    function setOntology(ontology) {{
      state.ontology = ontology;
      state.maxDepth = depthStore[datasetKey()] || dataset().summary.recommended_depth || 3;
      root.querySelector('#ontx-depth').value = state.maxDepth;
      root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
      state.selectedId = dataset().root_id;
      renderAll();
    }}
    function setLayout(layout) {{
      state.layout = layout;
      syncButtons();
      renderGraph();
    }}
    function syncButtons() {{
      root.querySelectorAll('#ontx-source .ontx-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.source === state.source));
      root.querySelectorAll('#ontx-layout .ontx-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.layout === state.layout));
    }}
    function relevantAvailable(ds) {{
      return (ds.summary.sample_exact_count || 0) + (ds.summary.sample_aligned_count || 0);
    }}
    function currentExpanded() {{
      return expanded[datasetKey()];
    }}
    function collapseAll() {{
      expanded[datasetKey()] = new Set();
      renderGraph();
      renderInspector();
    }}
    function expandAll() {{
      const ds = dataset();
      expanded[datasetKey()] = new Set(ds.nodes.filter(n => n.kind === 'branch').map(n => n.id));
      renderGraph();
      renderInspector();
    }}
    function resetToDepth() {{
      const ds = dataset();
      expanded[datasetKey()] = new Set(ds.nodes.filter(n => n.kind === 'branch' && n.depth < state.maxDepth).map(n => n.id));
      renderGraph();
      renderInspector();
    }}
    function expandAncestors(id) {{
      const ds = dataset();
      const nodeMap = ds.nodeMap;
      let cur = nodeMap[id];
      while (cur && cur.parent) {{
        currentExpanded().add(cur.parent);
        cur = nodeMap[cur.parent];
      }}
    }}
    /* BM25-inspired lexical search */
    function bm25Score(tokens, node) {{
      if (!tokens.length) return 0;
      const k1 = 1.5, b = 0.75, AVG_LEN = 14;
      const docText = `${{node.label}} ${{node.path_label||''}} ${{node.name||''}}`.toLowerCase();
      const docTokens = docText.split(/[^a-z0-9]+/).filter(Boolean);
      const docLen = Math.max(docTokens.length, 1);
      let score = 0;
      for (const term of tokens) {{
        const tf = docTokens.reduce((a, t) => a + (t.startsWith(term) ? 1 : 0), 0);
        if (!tf) continue;
        const labelExact = node.label.toLowerCase().includes(term);
        const labelBoost = labelExact ? 2.5 : 1.0;
        const tf_n = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLen / AVG_LEN));
        score += tf_n * labelBoost;
      }}
      return score;
    }}
    function searchMatches(ds) {{
      const q = state.search.trim().toLowerCase();
      if (!q) return [];
      const tokens = q.split(/ +/).filter(t => t.length >= 2);
      if (!tokens.length) {{
        return ds.nodes.filter(n => n.label.toLowerCase().startsWith(q)).slice(0, 18);
      }}
      return ds.nodes
        .map(n => ({{ node: n, score: bm25Score(tokens, n) }}))
        .filter(r => r.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 18)
        .map(r => r.node);
    }}
    function allowRelevant(node) {{
      if (!state.relevantOnly) return true;
      return (node.sample_exact_subtree || 0) > 0 || (node.sample_aligned_subtree || 0) > 0;
    }}
    function collectVisible(ds) {{
      const nodeMap = ds.nodeMap;
      const visibleNodes = [];
      const visibleEdges = [];
      const visibleChildren = {{}};
      function walk(id) {{
        const node = nodeMap[id];
        if (!node) return;
        if (id !== ds.root_id && !allowRelevant(node)) return;
        visibleNodes.push(node);
        const childIds = node.children
          .filter(childId => nodeMap[childId] && allowRelevant(nodeMap[childId]));
        const open = id === ds.root_id || currentExpanded().has(id);
        const descend = open && node.depth < state.maxDepth;
        visibleChildren[id] = descend ? childIds : [];
        if (descend) {{
          childIds.forEach(childId => {{
            visibleEdges.push([id, childId]);
            walk(childId);
          }});
        }}
      }}
      walk(ds.root_id);
      return {{ visibleNodes, visibleEdges, visibleChildren }};
    }}
    function layoutTree(ds, visible) {{
      const order = {{}};
      const children = visible.visibleChildren;
      let leafIndex = 0;
      function place(id) {{
        const kids = children[id] || [];
        if (!kids.length) {{
          order[id] = leafIndex++;
          return order[id];
        }}
        const vals = kids.map(place);
        order[id] = vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
        return order[id];
      }}
      place(ds.root_id);
      const maxDepth = Math.max(...visible.visibleNodes.map(n => n.depth), 0);
      if (state.layout === 'flow') {{
        const width = Math.max(860, 160 + maxDepth * 220 + 340);
        const height = Math.max(440, leafIndex * 54 + 120);
        const pos = {{}};
        visible.visibleNodes.forEach(node => {{
          pos[node.id] = {{
            x: 84 + node.depth * 220,
            y: 60 + (order[node.id] || 0) * 54,
          }};
        }});
        return {{ pos, width, height, centerX: width / 2, centerY: height / 2 }};
      }}
      if (state.layout === 'vertical') {{
        const width = Math.max(760, leafIndex * 60 + 160);
        const height = Math.max(520, 120 + maxDepth * 180 + 160);
        const pos = {{}};
        visible.visibleNodes.forEach(node => {{
          pos[node.id] = {{
            x: 80 + (order[node.id] || 0) * 60,
            y: 72 + node.depth * 180,
          }};
        }});
        return {{ pos, width, height, centerX: width / 2, centerY: height / 2 }};
      }}

      const maxVisibleDepth = Math.max(...visible.visibleNodes.map(n => n.depth), 1);
      const angleById = {{}};
      leafIndex = 0;
      function placeRadial(id) {{
        const kids = children[id] || [];
        if (!kids.length) {{
          const angle = ((leafIndex++) / Math.max(visible.visibleNodes.filter(n => !(children[n.id] || []).length).length, 1)) * Math.PI * 2;
          angleById[id] = angle;
          return angle;
        }}
        const vals = kids.map(placeRadial);
        angleById[id] = vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
        return angleById[id];
      }}
      placeRadial(ds.root_id);
      const radiusStep = 128;
      const radiusMax = Math.max(2, maxVisibleDepth) * radiusStep;
      const width = Math.max(780, radiusMax * 2 + 280);
      const height = Math.max(780, radiusMax * 2 + 280);
      const centerX = width / 2;
      const centerY = height / 2;
      const pos = {{}};
      visible.visibleNodes.forEach(node => {{
        const angle = angleById[node.id] || 0;
        const radius = node.depth * radiusStep;
        pos[node.id] = {{
          x: centerX + Math.cos(angle - Math.PI / 2) * radius,
          y: centerY + Math.sin(angle - Math.PI / 2) * radius,
        }};
      }});
      return {{ pos, width, height, centerX, centerY }};
    }}
    function edgePath(p0, p1) {{
      if (state.layout === 'flow') {{
        const dx = (p1.x - p0.x) * 0.38;
        return `M ${{p0.x}} ${{p0.y}} C ${{p0.x + dx}} ${{p0.y}}, ${{p1.x - dx}} ${{p1.y}}, ${{p1.x}} ${{p1.y}}`;
      }}
      if (state.layout === 'vertical') {{
        const dy = (p1.y - p0.y) * 0.4;
        return `M ${{p0.x}} ${{p0.y}} C ${{p0.x}} ${{p0.y + dy}}, ${{p1.x}} ${{p1.y - dy}}, ${{p1.x}} ${{p1.y}}`;
      }}
      return `M ${{p0.x}} ${{p0.y}} L ${{p1.x}} ${{p1.y}}`;
    }}
    function nodeFill(node) {{
      const spec = activeSpec();
      const leaf = node.kind !== 'branch';
      const sat = leaf ? 74 : 56;
      const light = Math.max(30, (leaf ? 88 : 92) - node.depth * 6.2);
      return `hsl(${{spec.hue}}, ${{sat}}%, ${{light}}%)`;
    }}
    function nodeRadius(node, dense) {{
      /* Root is the biggest node; every level down shrinks by 30%
         (size = base x 0.7^depth), with a readable floor. */
      const base = dense ? 16 : 24;
      const scaled = base * Math.pow(0.7, Math.max(0, node.depth));
      const floor_ = node.kind === 'branch' ? (dense ? 4.5 : 6) : (dense ? 3.2 : 4.4);
      return Math.max(floor_, scaled);
    }}
    function labelText(node, dense) {{
      if (state.labelMode === 'branches' && node.kind !== 'branch') return '';
      if (state.labelMode === 'full') return node.label;
      if (state.labelMode === 'compact') {{
        /* Always show at least tiny label on the right; suppress only in extreme density (>600 nodes) */
        if (dense && visible_count > 600 && node.kind !== 'branch') return '';
        return dense ? node.tiny : (node.kind === 'branch' ? node.short : node.tiny);
      }}
      return node.short;
    }}
    function labelAttrs(node, radius, position, layoutInfo) {{
      if (state.layout === 'flow') {{
        return {{ x: radius + 9, y: 4, anchor: 'start' }};
      }}
      if (state.layout === 'vertical') {{
        /* Rotate every node label consistently in the top-down layout. */
        return {{ x: radius * 0.8 + 4, y: radius + 12, anchor: 'start', rotate: 45 }};
      }}
      const toRight = position.x >= layoutInfo.centerX;
      return {{ x: toRight ? radius + 9 : -radius - 9, y: 4, anchor: toRight ? 'start' : 'end' }};
    }}
    function renderFocusCards() {{
      const envData = DATA.sources[state.source];
      const html = ['ATTACK', 'OPINION', 'PROFILE'].map(ontology => {{
        const ds = envData[ontology];
        const spec = COLORS[ontology];
        return `
          <div class="ontx-focus-item ${{ontology === state.ontology ? 'active' : ''}}" data-ontology="${{ontology}}">
            <div class="ontx-focus-top">
              <strong>${{ontology === 'ATTACK' ? 'Attack Space' : ontology === 'OPINION' ? 'Opinion Space' : 'Profile Space'}}</strong>
              <span class="ontx-focus-pill" style="background:${{spec.soft}};color:${{spec.accent}}">${{ds.summary.leaf_count}} leaves</span>
            </div>
            <div class="ontx-focus-meta">
              <span>${{ds.summary.branch_count}} branches</span>
              <span>depth ${{ds.summary.max_depth}}</span>
              <span>${{ds.summary.node_count}} total nodes</span>
            </div>
          </div>`;
      }}).join('');
      root.querySelector('#ontx-focus').innerHTML = html;
      root.querySelectorAll('.ontx-focus-item').forEach(el => el.addEventListener('click', () => setOntology(el.dataset.ontology)));
    }}
    function renderCompare() {{
      const current = state.ontology;
      const html = ['production', 'test'].map(env => {{
        const ds = DATA.sources[env][current];
        const active = env === state.source;
        const isRun = env === DATA.current_run_source;
        return `
          <div class="ontx-compare-card ${{active ? 'active' : ''}}">
            <div class="ontx-compare-top">
              <strong>${{env === 'production' ? 'Production ontology' : 'Test ontology'}}</strong>
              <div style="display:flex;gap:6px;flex-wrap:wrap">
                ${{active ? badgeHtml('Visible source', 'rgba(29,78,137,0.08)', '{PALETTE['blue']}') : ''}}
                ${{isRun ? badgeHtml('Run source', 'rgba(231,111,81,0.12)', '{PALETTE['orange']}') : ''}}
              </div>
            </div>
            <div class="ontx-compare-grid">
              <div class="ontx-metric"><div class="k">Leaves</div><div class="v">${{ds.summary.leaf_count}}</div></div>
              <div class="ontx-metric"><div class="k">Branches</div><div class="v">${{ds.summary.branch_count}}</div></div>
              <div class="ontx-metric"><div class="k">Depth</div><div class="v">${{ds.summary.max_depth}}</div></div>
              <div class="ontx-metric"><div class="k">Annotated leaves</div><div class="v">${{ds.summary.metadata_leaf_count}}</div></div>
            </div>
          </div>`;
      }}).join('');
      root.querySelector('#ontx-compare').innerHTML = html;
    }}
    function renderStatus(ds, visible) {{
      const dense = visible.visibleNodes.length > 260;
      const runNote = DATA.current_run_source === state.source
        ? 'Current source matches the ontology used in this run.'
        : 'Current source is a comparison surface; switch to Test to inspect the exact test ontology used in this run.';
      const relevantCount = ds.summary.sample_exact_count + ds.summary.sample_aligned_count;
      root.querySelector('#ontx-status').innerHTML =
        `<strong>Visible structure:</strong> ${{visible.visibleNodes.length}} of ${{ds.summary.node_count}} nodes, depth ≤ ${{state.maxDepth}}, ${{state.layout}} layout<br>` +
        `<strong>Run alignment:</strong> ${{relevantCount}} highlighted leaves available for this ontology across exact or leaf-name matching. ${{runNote}}<br>` +
        `<strong>Readability:</strong> ${{dense ? 'Dense mode is active; leaf labels are softened. Use search or reduce depth for cleaner local inspection.' : 'Local labels are fully readable at the current depth.'}}`;
    }}
    function renderCanvasHeader(ds, visible) {{
      root.querySelector('#ontx-active-title').textContent =
        `${{state.source === 'production' ? 'Production' : 'Test'}} · ${{state.ontology === 'ATTACK' ? 'Attack Ontology' : state.ontology === 'OPINION' ? 'Opinion Ontology' : 'Profile Ontology'}}`;
      root.querySelector('#ontx-active-sub').textContent =
        `Mixed hierarchical state space with ${{ds.summary.leaf_count}} terminal leaves, ${{ds.summary.branch_count}} branching nodes, and a maximum depth of ${{ds.summary.max_depth}}.`;
      root.querySelector('#ontx-chipline').innerHTML =
        badgeHtml(`Zoom ${{Math.round(state.zoom * 100)}}%`, 'rgba(20,33,61,0.06)', '{PALETTE['muted']}') +
        badgeHtml(`Visible ${{visible.visibleNodes.length}} nodes`, 'rgba(29,78,137,0.08)', '{PALETTE['blue']}') +
        badgeHtml(`Depth ${{state.maxDepth}}`, activeSpec().soft, activeSpec().accent);
    }}
    function renderSearchResults(ds) {{
      const q = state.search.trim();
      let items = [];
      if (q) {{
        items = searchMatches(ds);
      }} else {{
        items = ds.nodes
          .filter(n => n.depth === 1 || n.sample_exact || n.sample_aligned)
          .slice(0, 12);
      }}
      if (!items.length) {{
        root.querySelector('#ontx-results').innerHTML = '<div class="ontx-note">No ontology paths match the current search or relevance filter.</div>';
        return;
      }}
      root.querySelector('#ontx-results').innerHTML = items.map(node => `
        <div class="ontx-result" data-node-id="${{node.id}}">
          <strong>${{escapeHtml(node.label)}}</strong>
          <span>${{escapeHtml(node.path_label || node.label)}}</span>
        </div>`).join('');
      root.querySelectorAll('.ontx-result').forEach(el => el.addEventListener('click', () => {{
        const id = el.dataset.nodeId;
        expandAncestors(id);
        state.selectedId = id;
        renderAll();
      }}));
    }}
    function renderHighlights(ds, visible) {{
      let items = ds.nodes.filter(n => n.kind !== 'branch' && (n.sample_exact || n.sample_aligned));
      if (state.relevantOnly) {{
        const visibleSet = new Set(visible.visibleNodes.map(n => n.id));
        items = items.filter(n => visibleSet.has(n.id));
      }}
      items = items
        .sort((a, b) => (Number(b.sample_exact) - Number(a.sample_exact)) || a.path_label.localeCompare(b.path_label))
        .slice(0, 9);
      if (!items.length) {{
        root.querySelector('#ontx-highlights').innerHTML =
          `<div class="ontx-note">${{state.ontology === 'PROFILE'
            ? 'The profile ontology is shown structurally; this run does not carry a leaf-level selected profile subset like the attack and opinion factorial leaves.'
            : 'No run-aligned leaf highlights are available for the current source. Switch source or disable the relevance-only filter to inspect the full ontology.'}}</div>`;
        return;
      }}
      root.querySelector('#ontx-highlights').innerHTML = items.map(node => `
        <div class="ontx-highlight-item">
          <strong>${{escapeHtml(node.label)}} ${{node.sample_exact ? '· exact run leaf' : '· leaf-name aligned'}}</strong>
          <span>${{escapeHtml(node.path_label)}}</span>
        </div>`).join('');
    }}
    function renderInspector() {{
      const ds = dataset();
      const node = ds.nodeMap[state.selectedId] || ds.nodeMap[ds.root_id];
      if (!node) return;
      const badges = [
        badgeHtml(node.kind === 'branch' ? 'Branch' : (node.kind === 'leaf_meta' ? 'Leaf + metadata' : 'Leaf'), activeSpec().soft, activeSpec().accent),
        badgeHtml(`Depth ${{node.depth}}`, 'rgba(20,33,61,0.06)', '{PALETTE['muted']}'),
      ];
      if (node.sample_exact) badges.push(badgeHtml('Exact run-aligned', 'rgba(240,192,64,0.18)', '{PALETTE['amber']}'));
      if (node.sample_aligned) badges.push(badgeHtml('Leaf-name aligned', 'rgba(200,155,60,0.12)', '{PALETTE['amber']}'));
      const children = (node.children || []).slice(0, 8).map(id => ds.nodeMap[id]).filter(Boolean);
      const metaHtml = node.metadata_preview && node.metadata_preview.length
        ? `<div class="ontx-meta-list">${{node.metadata_preview.map(([k,v]) => `
            <div class="ontx-meta-item">
              <strong>${{escapeHtml(k)}}</strong>
              <span>${{escapeHtml(v)}}</span>
            </div>`).join('')}}</div>`
        : `<div class="ontx-note">No leaf-level metadata stored on this node.</div>`;
      const childHtml = children.length
        ? `<div class="ontx-meta-list">${{children.map(child => `
            <div class="ontx-meta-item">
              <strong>${{escapeHtml(child.label)}}</strong>
              <span>${{child.leaf_count}} leaves below · ${{child.child_count}} direct children</span>
            </div>`).join('')}}</div>`
        : '';

      root.querySelector('#ontx-inspector').innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:10px">
          <div>
            <div style="font-weight:800;font-size:0.92rem;color:{PALETTE['ink']}">${{escapeHtml(node.label)}}</div>
            <div class="ontx-sub" style="margin:3px 0 0">${{node.parent ? 'Selected ontology node' : 'Synthetic ontology root used to organize the JSON hierarchy for visualization'}}</div>
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">${{badges.join('')}}</div>
        </div>
        <div class="ontx-path">${{escapeHtml(node.path_label || node.label)}}</div>
        <div class="ontx-kv" style="margin-top:10px">
          <div class="ontx-metric"><div class="k">Direct children</div><div class="v">${{node.child_count}}</div></div>
          <div class="ontx-metric"><div class="k">Leaves below</div><div class="v">${{node.leaf_count}}</div></div>
          <div class="ontx-metric"><div class="k">Subtree nodes</div><div class="v">${{node.subtree_node_count}}</div></div>
          <div class="ontx-metric"><div class="k">Metadata leaves below</div><div class="v">${{node.metadata_leaf_count}}</div></div>
        </div>
        <div style="margin-top:10px;font-weight:700;font-size:0.77rem;color:{PALETTE['navy']};margin-bottom:6px">Metadata preview</div>
        ${{metaHtml}}
        ${{children.length ? '<div style="margin-top:10px;font-weight:700;font-size:0.77rem;color:{PALETTE['navy']};margin-bottom:6px">Visible child branches</div>' + childHtml : ''}}
      `;
    }}
    let visible_count = 0;
    let lastLayout = null;
    let lastVisible = null;
    const manualOffsets = {{}};
    function offsetKey(nodeId) {{
      return `${{datasetKey()}}:${{state.layout}}:${{nodeId}}`;
    }}
    function applyManualOffsets(layout) {{
      Object.keys(layout.pos).forEach(id => {{
        const off = manualOffsets[offsetKey(id)];
        if (off) {{
          layout.pos[id] = {{ x: layout.pos[id].x + off.dx, y: layout.pos[id].y + off.dy }};
        }}
      }});
      return layout;
    }}
    /* Collision relaxation: nodes repel when overlapping, with strength
       proportional to their radii so high-level (bigger) nodes push harder,
       like a conventional force-directed ontology explorer. */
    function relaxCollisions(layout, visibleNodes, dense, iterations) {{
      const nodes = visibleNodes.map(n => ({{ id: n.id, r: nodeRadius(n, dense) + (state.labelMode === 'full' ? 8 : 5) }}));
      for (let it = 0; it < (iterations || 3); it++) {{
        for (let i = 0; i < nodes.length; i++) {{
          for (let j = i + 1; j < nodes.length; j++) {{
            const a = layout.pos[nodes[i].id], b = layout.pos[nodes[j].id];
            if (!a || !b) continue;
            let dx = b.x - a.x, dy = b.y - a.y;
            let d = Math.hypot(dx, dy);
            const minD = nodes[i].r + nodes[j].r + 4;
            if (d >= minD) continue;
            if (d < 1e-3) {{ dx = (Math.random() - 0.5); dy = (Math.random() - 0.5); d = 1; }}
            const push = (minD - d) / 2;
            const wi = nodes[i].r / (nodes[i].r + nodes[j].r);
            const wj = 1 - wi;
            /* bigger node moves less (it has more 'power') */
            a.x -= (dx / d) * push * wj; a.y -= (dy / d) * push * wj;
            b.x += (dx / d) * push * wi; b.y += (dy / d) * push * wi;
          }}
        }}
      }}
      return layout;
    }}
    function renderGraph() {{
      const ds = dataset();
      const visible = collectVisible(ds);
      visible_count = visible.visibleNodes.length;
      renderStatus(ds, visible);
      renderCanvasHeader(ds, visible);
      const searchIds = new Set(searchMatches(ds).map(n => n.id));
      const layoutBase = applyManualOffsets(layoutTree(ds, visible));
      const layout = relaxCollisions(layoutBase, visible.visibleNodes, visible.visibleNodes.length > 260, 3);
      lastLayout = layout;
      lastVisible = visible;
      const dense = visible_count > 260;
      const svgWidth = layout.width * state.zoom;
      const svgHeight = layout.height * state.zoom;
      let svg = `<svg id="ontx-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${{layout.width}} ${{layout.height}}" width="${{svgWidth}}" height="${{svgHeight}}">`;
      svg += `<defs>
        <filter id="ontx-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="rgba(20,33,61,0.12)"/>
        </filter>
        <filter id="ontx-glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="2.6" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>`;

      /* root-to-selection chain: drawn last, glowing, while other edges dim */
      const chainPairs = new Set();
      if (state.selectedId) {{
        let cur = ds.nodeMap[state.selectedId];
        while (cur && cur.parent) {{
          chainPairs.add(`${{cur.parent}}>>${{cur.id}}`);
          cur = ds.nodeMap[cur.parent];
        }}
      }}
      const hasChain = Array.from(chainPairs).some(key => {{
        const [a, b] = key.split('>>');
        return visible.visibleEdges.some(([x, y]) => x === a && y === b);
      }});
      const chainSvg = [];
      visible.visibleEdges.forEach(([a, b]) => {{
        const p0 = layout.pos[a];
        const p1 = layout.pos[b];
        const onChain = chainPairs.has(`${{a}}>>${{b}}`);
        if (onChain) {{
          chainSvg.push(
            `<path class="ontx-edge ontx-path-edge" data-edge-from="${{a}}" data-edge-to="${{b}}" d="${{edgePath(p0, p1)}}" fill="none" stroke="{PALETTE['gold']}" stroke-width="3.1" stroke-linecap="round" filter="url(#ontx-glow)"/>`
          );
          return;
        }}
        const accent = activeSpec().edge;
        const boosted = searchIds.has(a) || searchIds.has(b);
        const dimmed = hasChain && !boosted;
        svg += `<path class="ontx-edge" data-edge-from="${{a}}" data-edge-to="${{b}}" d="${{edgePath(p0, p1)}}" fill="none" stroke="${{boosted ? activeSpec().accent : accent}}" stroke-width="${{boosted ? 2.2 : 1.4}}" stroke-linecap="round"${{dimmed ? ' opacity="0.34"' : ''}}/>`;
      }});
      svg += chainSvg.join('');

      visible.visibleNodes.forEach(node => {{
        const pos = layout.pos[node.id];
        const selected = node.id === state.selectedId;
        /* Children are visible iff this branch is in expanded set AND depth < maxDepth */
        const childrenVisible = node.kind === 'branch' && node.child_count > 0 &&
          (currentExpanded().has(node.id) || node.id === ds.root_id) &&
          node.depth < state.maxDepth;
        const canExpand = node.kind === 'branch' && node.child_count > 0 && !childrenVisible;
        const radius = nodeRadius(node, dense);
        const fill = nodeFill(node);
        const stroke = selected ? '{PALETTE['gold']}' : activeSpec().accent;
        const label = labelText(node, dense);
        const attrs = label ? labelAttrs(node, radius, pos, layout) : null;
        const searchMatch = searchIds.has(node.id);
        svg += `<g class="ontx-node" data-node-id="${{node.id}}" style="cursor:${{node.kind==='branch'&&node.child_count>0?'pointer':'default'}}">`;
        if (state.highlightRun && (node.sample_exact || node.sample_aligned)) {{
          const ringColor = node.sample_exact ? '{PALETTE['gold']}' : '{PALETTE['amber']}';
          const dash = node.sample_exact ? '' : ' stroke-dasharray="4 3"';
          svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius + 4.3}}" fill="none" stroke="${{ringColor}}" stroke-width="2.2"${{dash}} opacity="0.98"/>`;
        }}
        if (state.showMetadata && node.kind === 'leaf_meta') {{
          svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius + 2.6}}" fill="none" stroke="{PALETTE['muted']}" stroke-width="1.2" stroke-dasharray="3 2" opacity="0.72"/>`;
        }}
        if (searchMatch) {{
          svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius + 6.6}}" fill="none" stroke="{PALETTE['sky']}" stroke-width="2" opacity="0.55"/>`;
        }}
        svg += `<circle cx="${{pos.x}}" cy="${{pos.y}}" r="${{radius}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="${{selected ? 2.6 : 1.4}}" filter="url(#ontx-shadow)"/>`;
        /* Show + (expandable) or − (children visible); nothing for leaf nodes */
        if (node.kind === 'branch' && node.child_count > 0 && node.id !== ds.root_id) {{
          const sym = canExpand ? '+' : '\u2212';
          const symFill = canExpand ? '#ffffff' : '{PALETTE['ink']}';
          svg += `<text x="${{pos.x}}" y="${{pos.y + 3.2}}" text-anchor="middle" style="font:700 8px IBM Plex Sans,sans-serif;fill:${{symFill}};pointer-events:none">${{sym}}</text>`;
        }}
        if (label) {{
          const rot = attrs.rotate ? ` transform="rotate(${{attrs.rotate}} ${{pos.x + attrs.x}} ${{pos.y + attrs.y}})"` : '';
          svg += `<text x="${{pos.x + attrs.x}}" y="${{pos.y + attrs.y}}" text-anchor="${{attrs.anchor}}"${{rot}} style="font:600 ${{dense ? 8.6 : 9.4}}px IBM Plex Sans, sans-serif;fill:{PALETTE['ink']};paint-order:stroke;stroke:white;stroke-width:3;pointer-events:none">${{escapeHtml(label)}}</text>`;
        }}
        svg += `</g>`;
      }});
      svg += `<g id="ontx-flow-layer" style="pointer-events:none"></g>`;
      svg += `</svg>`;
      root.querySelector('#ontx-canvas-wrap').innerHTML = svg;
      bindNodeInteractions(ds);
      startFlowAnimation(ds);
      renderHighlights(ds, visible);
    }}

    /* ── node drag + click (drag a node and connected edges follow live) ── */
    function bindNodeInteractions(ds) {{
      root.querySelectorAll('.ontx-node').forEach(el => {{
        el.addEventListener('mousedown', ev => {{
          ev.stopPropagation();
          ev.preventDefault();
          const id = el.dataset.nodeId;
          const node = ds.nodeMap[id];
          if (!node) return;
          const startX = ev.clientX, startY = ev.clientY;
          const key = offsetKey(id);
          const startOff = manualOffsets[key] ? {{...manualOffsets[key]}} : {{dx: 0, dy: 0}};
          let dragged = false;
          /* a drag carries the node's entire visible subtree along */
          const subtreeIds = [id];
          (function collect(cur) {{
            ((lastVisible && lastVisible.visibleChildren[cur]) || []).forEach(c => {{
              subtreeIds.push(c); collect(c);
            }});
          }})(id);
          const startOffs = subtreeIds.map(sid => {{
            const k = offsetKey(sid);
            return [sid, manualOffsets[k] ? {{...manualOffsets[k]}} : {{dx: 0, dy: 0}}];
          }});
          function onMove(e) {{
            const dxc = e.clientX - startX, dyc = e.clientY - startY;
            if (!dragged && Math.hypot(dxc, dyc) < 4) return;
            dragged = true;
            startOffs.forEach(([sid, off0]) => {{
              manualOffsets[offsetKey(sid)] = {{ dx: off0.dx + dxc / state.zoom, dy: off0.dy + dyc / state.zoom }};
            }});
            subtreeIds.forEach(sid => dragNodeDom(sid));
          }}
          function onUp(e) {{
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            if (dragged) {{ renderGraph(); renderInspector(); return; }}
            nodeClicked(ds, id);
          }}
          window.addEventListener('mousemove', onMove);
          window.addEventListener('mouseup', onUp);
        }});
      }});
    }}
    function dragNodeDom(id) {{
      if (!lastLayout) return;
      const base = layoutTree(dataset(), lastVisible);
      const off = manualOffsets[offsetKey(id)] || {{dx: 0, dy: 0}};
      const nx = base.pos[id].x + off.dx;
      const ny = base.pos[id].y + off.dy;
      const oldPos = lastLayout.pos[id];
      const dx = nx - oldPos.x, dy = ny - oldPos.y;
      lastLayout.pos[id] = {{ x: nx, y: ny }};
      const g = root.querySelector(`.ontx-node[data-node-id="${{id}}"]`);
      if (g) {{
        g.querySelectorAll('circle').forEach(c => {{
          c.setAttribute('cx', parseFloat(c.getAttribute('cx')) + dx);
          c.setAttribute('cy', parseFloat(c.getAttribute('cy')) + dy);
        }});
        g.querySelectorAll('text').forEach(t => {{
          t.setAttribute('x', parseFloat(t.getAttribute('x')) + dx);
          t.setAttribute('y', parseFloat(t.getAttribute('y')) + dy);
        }});
      }}
      root.querySelectorAll(`.ontx-edge[data-edge-from="${{id}}"],.ontx-edge[data-edge-to="${{id}}"]`).forEach(p => {{
        const a = p.dataset.edgeFrom, b = p.dataset.edgeTo;
        const p0 = lastLayout.pos[a], p1 = lastLayout.pos[b];
        if (p0 && p1) p.setAttribute('d', edgePath(p0, p1));
      }});
      refreshFlowChain();
    }}
    function nodeClicked(ds, id) {{
      const node = ds.nodeMap[id];
      if (!node) return;
      state.selectedId = id;
      let didExpand = false;
      if (node.kind === 'branch' && node.child_count > 0 && id !== ds.root_id) {{
        if (currentExpanded().has(id)) {{
          currentExpanded().delete(id);
        }} else {{
          currentExpanded().add(id);
          didExpand = true;
          if (node.depth >= state.maxDepth) {{
            state.maxDepth = node.depth + 1;
            depthStore[datasetKey()] = state.maxDepth;
            const depthEl = root.querySelector('#ontx-depth');
            if (depthEl) {{ depthEl.value = state.maxDepth; }}
            const dispEl = root.querySelector('#ontx-depth-display');
            if (dispEl) {{ dispEl.textContent = state.maxDepth; }}
          }}
        }}
      }}
      renderGraph();
      renderInspector();
      if (didExpand) {{
        centerOnSubtree(id);
      }} else {{
        const selEl = root.querySelector(`[data-node-id="${{id}}"] circle`);
        if (selEl) selEl.scrollIntoView({{block: 'nearest', inline: 'nearest'}});
      }}
    }}
    /* center the canvas on a freshly expanded node plus its visible children */
    function centerOnSubtree(id) {{
      if (!lastLayout || !lastVisible) return;
      const wrap = root.querySelector('#ontx-canvas-wrap');
      const ids = [id, ...(lastVisible.visibleChildren[id] || [])];
      const pts = ids.map(n => lastLayout.pos[n]).filter(Boolean);
      if (!pts.length) return;
      const minX = Math.min(...pts.map(p => p.x)), maxX = Math.max(...pts.map(p => p.x));
      const minY = Math.min(...pts.map(p => p.y)), maxY = Math.max(...pts.map(p => p.y));
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      wrap.scrollTo({{
        left: Math.max(0, cx * state.zoom - wrap.clientWidth / 2),
        top: Math.max(0, cy * state.zoom - wrap.clientHeight / 2),
        behavior: 'smooth',
      }});
    }}

    /* ── animated flow particles along the root-to-selected path ────────── */
    let flowRaf = null;
    let flowChain = [];
    function refreshFlowChain() {{
      /* Chain is rebuilt from the DOM path-edge elements ordered root-to-node,
         matching by dataset attributes via direct comparison (no CSS attribute
         selector, so ids containing '>' or quotes can never break matching;
         this was the cause of particles skipping nodes on some branches). */
      const ds = dataset();
      flowChain = [];
      if (!state.selectedId || !lastLayout) return;
      const nodeMap = ds.nodeMap;
      let cur = nodeMap[state.selectedId];
      const ids = [];
      while (cur) {{
        ids.unshift(cur.id);
        cur = cur.parent ? nodeMap[cur.parent] : null;
      }}
      const allEdges = Array.from(root.querySelectorAll('.ontx-edge'));
      for (let i = 0; i + 1 < ids.length; i++) {{
        const el = allEdges.find(e => e.dataset.edgeFrom === ids[i] && e.dataset.edgeTo === ids[i + 1]);
        if (el) flowChain.push(el);
      }}
    }}
    function startFlowAnimation(ds) {{
      if (flowRaf) {{ cancelAnimationFrame(flowRaf); flowRaf = null; }}
      refreshFlowChain();
      const layer = root.querySelector('#ontx-flow-layer');
      if (!layer) return;
      /* Constant traversal time: every root-to-node journey takes PERIOD_MS
         regardless of path length, so particles on long paths visibly speed
         up. Each particle is a comet: bright head, glow halo, fading tail. */
      const N_DOTS = 3;
      const TAIL = 3;
      const PERIOD_MS = 1500;
      const accent = activeSpec().accent;
      const gold = '{PALETTE['gold']}';
      let parts = '';
      for (let i = 0; i < N_DOTS; i++) {{
        for (let t = TAIL; t >= 1; t--) {{
          parts += `<circle class="ontx-flow-tail" data-dot="${{i}}" data-tail="${{t}}" r="${{(2.6 - t * 0.55).toFixed(2)}}" fill="${{accent}}" opacity="0"></circle>`;
        }}
        parts += `<circle class="ontx-flow-halo" data-dot="${{i}}" r="7.4" fill="none" stroke="${{gold}}" stroke-width="1.1" opacity="0"></circle>`;
        parts += `<circle class="ontx-flow-dot" data-dot="${{i}}" r="3.6" fill="${{gold}}" opacity="0" stroke="white" stroke-width="0.9" filter="url(#ontx-glow)"></circle>`;
      }}
      layer.innerHTML = parts;
      const dots = Array.from(layer.querySelectorAll('.ontx-flow-dot'));
      const halos = Array.from(layer.querySelectorAll('.ontx-flow-halo'));
      const tails = Array.from(layer.querySelectorAll('.ontx-flow-tail'));
      let t0 = null;
      function pointAt(phase, lengths, total) {{
        let dist = Math.max(0, Math.min(0.9999, phase)) * total;
        let seg = 0;
        while (seg < flowChain.length && dist > lengths[seg]) {{ dist -= lengths[seg]; seg++; }}
        if (seg >= flowChain.length) seg = flowChain.length - 1;
        try {{
          return flowChain[seg].getPointAtLength(Math.max(0, Math.min(dist, lengths[seg])));
        }} catch (e) {{ return null; }}
      }}
      function frame(ts) {{
        if (!t0) t0 = ts;
        if (!flowChain.length) {{
          dots.forEach(d => d.setAttribute('opacity', 0));
          halos.forEach(h => h.setAttribute('opacity', 0));
          tails.forEach(h => h.setAttribute('opacity', 0));
          flowRaf = requestAnimationFrame(frame);
          return;
        }}
        let lengths, total;
        try {{
          lengths = flowChain.map(p => p.getTotalLength());
          total = lengths.reduce((a, b) => a + b, 0);
        }} catch (e) {{ flowRaf = requestAnimationFrame(frame); return; }}
        if (total <= 0) {{ flowRaf = requestAnimationFrame(frame); return; }}
        const tailPhase = Math.min(0.035, 9 / Math.max(total, 1));
        for (let i = 0; i < N_DOTS; i++) {{
          const phase = (((ts - t0) / PERIOD_MS) + i / N_DOTS) % 1;
          const pt = pointAt(phase, lengths, total);
          if (!pt) continue;
          const pulse = 0.92 + 0.08 * Math.sin(ts / 130 + i * 2.1);
          dots[i].setAttribute('cx', pt.x); dots[i].setAttribute('cy', pt.y);
          dots[i].setAttribute('opacity', pulse.toFixed(2));
          halos[i].setAttribute('cx', pt.x); halos[i].setAttribute('cy', pt.y);
          halos[i].setAttribute('r', (6.6 + 1.6 * Math.sin(ts / 180 + i)).toFixed(2));
          halos[i].setAttribute('opacity', (0.34 + 0.18 * Math.sin(ts / 200 + i)).toFixed(2));
          tails.filter(el => +el.dataset.dot === i).forEach(el => {{
            const back = +el.dataset.tail * tailPhase;
            const tp = pointAt((phase - back + 1) % 1, lengths, total);
            if (!tp) return;
            el.setAttribute('cx', tp.x); el.setAttribute('cy', tp.y);
            el.setAttribute('opacity', (0.5 - 0.13 * (+el.dataset.tail)).toFixed(2));
          }});
        }}
        flowRaf = requestAnimationFrame(frame);
      }}
      flowRaf = requestAnimationFrame(frame);
    }}
    function renderAll() {{
      renderFocusCards();
      renderCompare();
      renderSearchResults(dataset());
      renderGraph();
      renderInspector();
    }}

    const ZOOM_MIN = 0.25;
    const ZOOM_MAX = 3.0;
    const ZOOM_STEP = 0.22;
    const ZOOM_WHEEL_SENSITIVITY = 0.0024;
    function setZoom(nextZoom, anchorClientX = null, anchorClientY = null) {{
      const wrap = root.querySelector('#ontx-canvas-wrap');
      if (!wrap) return;
      const clamped = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, nextZoom));
      if (Math.abs(clamped - state.zoom) < 1e-4) return;
      const prevZoom = state.zoom;

      const rect = wrap.getBoundingClientRect();
      let viewportX = wrap.clientWidth / 2;
      let viewportY = wrap.clientHeight / 2;
      if (anchorClientX !== null && anchorClientY !== null) {{
        viewportX = Math.max(0, Math.min(wrap.clientWidth, anchorClientX - rect.left));
        viewportY = Math.max(0, Math.min(wrap.clientHeight, anchorClientY - rect.top));
      }}

      const worldX = (wrap.scrollLeft + viewportX) / prevZoom;
      const worldY = (wrap.scrollTop + viewportY) / prevZoom;

      state.zoom = clamped;

      /* Instant zoom: the SVG keeps its viewBox, so resizing the element
         rescales every node, edge, and label without a re-render. */
      const svg = root.querySelector('#ontx-svg');
      if (svg && lastLayout) {{
        svg.setAttribute('width', lastLayout.width * state.zoom);
        svg.setAttribute('height', lastLayout.height * state.zoom);
      }} else {{
        renderGraph();
      }}
      wrap.scrollLeft = worldX * state.zoom - viewportX;
      wrap.scrollTop = worldY * state.zoom - viewportY;

      const zoomChip = root.querySelector('#ontx-chipline');
      if (zoomChip && zoomChip.firstElementChild) {{
        zoomChip.firstElementChild.textContent = `Zoom ${{Math.round(state.zoom * 100)}}%`;
      }}
    }}

    root.querySelectorAll('#ontx-source .ontx-btn').forEach(btn => btn.addEventListener('click', () => setSource(btn.dataset.source)));
    root.querySelectorAll('#ontx-layout .ontx-btn').forEach(btn => btn.addEventListener('click', () => setLayout(btn.dataset.layout)));
    root.querySelector('#ontx-depth').addEventListener('input', ev => {{
      state.maxDepth = parseInt(ev.target.value, 10);
      depthStore[datasetKey()] = state.maxDepth;
      root.querySelector('#ontx-depth-display').textContent = state.maxDepth;
      renderGraph();
      renderInspector();
    }});
    root.querySelector('#ontx-label-mode').addEventListener('change', ev => {{
      state.labelMode = ev.target.value;
      renderGraph();
    }});
    root.querySelector('#ontx-show-meta').addEventListener('change', ev => {{
      state.showMetadata = ev.target.checked;
      renderGraph();
    }});
    root.querySelector('#ontx-highlight-run').addEventListener('change', ev => {{
      state.highlightRun = ev.target.checked;
      renderGraph();
      renderInspector();
    }});
    root.querySelector('#ontx-relevant-only').addEventListener('change', ev => {{
      state.relevantOnly = ev.target.checked;
      renderAll();
    }});
    root.querySelector('#ontx-search').addEventListener('input', ev => {{
      state.search = ev.target.value;
      renderSearchResults(dataset());
      renderGraph();
    }});
    root.querySelector('#ontx-expand-all').addEventListener('click', expandAll);
    root.querySelector('#ontx-collapse-all').addEventListener('click', collapseAll);
    root.querySelector('#ontx-reset-depth').addEventListener('click', resetToDepth);
    root.querySelector('#ontx-zoom-in').addEventListener('click', () => {{
      setZoom(state.zoom + ZOOM_STEP);
    }});
    root.querySelector('#ontx-zoom-out').addEventListener('click', () => {{
      setZoom(state.zoom - ZOOM_STEP);
    }});
    root.querySelector('#ontx-zoom-reset').addEventListener('click', () => {{
      setZoom(1);
    }});

    /* ── Drag / pan on canvas-wrap ─────────────────────────────────────── */
    (function() {{
      const wrap = root.querySelector('#ontx-canvas-wrap');
      let drag = null;
      wrap.style.cursor = 'grab';
      wrap.addEventListener('mousedown', e => {{
        if (e.target.closest('.ontx-node')) return;
        drag = {{ sx: e.clientX + wrap.scrollLeft, sy: e.clientY + wrap.scrollTop }};
        wrap.style.cursor = 'grabbing';
        e.preventDefault();
      }});
      window.addEventListener('mousemove', e => {{
        if (!drag) return;
        wrap.scrollLeft = drag.sx - e.clientX;
        wrap.scrollTop  = drag.sy - e.clientY;
      }});
      window.addEventListener('mouseup', () => {{ drag = null; wrap.style.cursor = 'grab'; }});
      /* touch support */
      wrap.addEventListener('touchstart', e => {{
        if (e.touches.length !== 1 || e.target.closest('.ontx-node')) return;
        const t = e.touches[0];
        drag = {{ sx: t.clientX + wrap.scrollLeft, sy: t.clientY + wrap.scrollTop }};
      }}, {{passive: true}});
      wrap.addEventListener('touchmove', e => {{
        if (!drag || e.touches.length !== 1) return;
        const t = e.touches[0];
        wrap.scrollLeft = drag.sx - t.clientX;
        wrap.scrollTop  = drag.sy - t.clientY;
        e.preventDefault();
      }}, {{passive: false}});
      wrap.addEventListener('touchend', () => {{ drag = null; }});
      /* Zoom only on pinch / ctrl+wheel / cmd+wheel so plain wheel keeps
         scrolling the canvas naturally; this is what made zoom feel broken. */
      wrap.addEventListener('wheel', e => {{
        if (!(e.ctrlKey || e.metaKey)) return;
        e.preventDefault();
        const factor = Math.exp(-e.deltaY * ZOOM_WHEEL_SENSITIVITY);
        setZoom(state.zoom * factor, e.clientX, e.clientY);
      }}, {{passive: false}});
      wrap.addEventListener('dblclick', e => {{
        if (e.target.closest('.ontx-node')) return;
        e.preventDefault();
        setZoom(e.altKey ? state.zoom / 1.5 : state.zoom * 1.5, e.clientX, e.clientY);
      }});
    }})();

    initDatasets();
    syncButtons();
    renderAll();
  }})();
  </script>
</div>"""

def _fig_factorial_3d(long_df: pd.DataFrame) -> go.Figure:
    """Dual go.Surface: mean AE (RdBu_r) + ISD of AE (YlOrRd)."""
    atk_col = next((c for c in ("attack_execute_tactic", "attack_leaf_label", "attack_leaf") if c in long_df.columns), "attack_leaf")
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    for c in (atk_col, op_col, ae_col):
        if c not in long_df.columns:
            return go.Figure().add_annotation(text=f"Column '{c}' missing", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    attack_labels = _unique_display_map(attacks)
    opinion_labels = _unique_display_map(opinions)

    def _matrix(func):
        return (
            long_df.groupby([atk_col, op_col])[ae_col].agg(func)
            .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0)
        )

    mean_mat = _matrix("mean")
    isd_mat  = _matrix("std")
    atk_l    = [_wrap_label(attack_labels[a], 18) for a in attacks]
    op_l     = [_wrap_label(opinion_labels[o], 18) for o in opinions]

    # Use make_subplots to properly allocate non-overlapping scene domains.
    # Do NOT manually override domain in update_layout — it conflicts with the
    # auto-computed subplot annotations (titles) and causes visual overlap.
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Mean Adversarial Effectivity (AE)", "Inter-individual Variability (SD of AE)"],
        horizontal_spacing=0.06,
    )

    def _surf(mat, cscale, zlabel):
        z = mat.values.astype(float)
        return go.Surface(
            z=z, x=op_l, y=atk_l,
            colorscale=cscale, cmid=0,
            lighting=dict(ambient=0.75, diffuse=0.9, specular=0.4, roughness=0.4),
            lightposition=dict(x=100, y=200, z=500),
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
            hovertemplate=f"<b>%{{y}}</b> → <b>%{{x}}</b><br>{zlabel}: %{{z:.2f}}<extra></extra>",
        )

    fig.add_trace(_surf(mean_mat, "RdBu_r", "Mean AE"), row=1, col=1)
    fig.add_trace(_surf(isd_mat,  "YlOrRd",  "SD AE"),   row=1, col=2)

    cam = dict(eye=dict(x=1.55, y=-1.55, z=1.05))
    # Use update_scenes() instead of update_layout(scene=...) so the domain
    # computed by make_subplots is preserved — prevents left/right surface overlap.
    fig.update_scenes(
        xaxis=dict(title="Opinion leaf", tickfont=dict(size=8), gridcolor="#ccd8ee"),
        yaxis=dict(title="Attack vector", tickfont=dict(size=8), gridcolor="#ccd8ee"),
        zaxis=dict(gridcolor="#ccd8ee", title="AE"),
        camera=cam,
        bgcolor="rgba(248,250,255,1)",
        aspectmode="cube",
    )
    fig.update_layout(
        paper_bgcolor="white",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=660,
        showlegend=False,
        title=dict(
            text="3D Factorial Surface — Mean AE (left) · Inter-individual Variability SD (right)",
            font_size=13,
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(thickness=14),
    )
    return fig


def _fig_factorial_2d(long_df: pd.DataFrame) -> go.Figure:
    """Side-by-side 2D annotated heatmaps (mean AE | ISD)."""
    atk_col = next((c for c in ("attack_execute_tactic", "attack_leaf_label", "attack_leaf") if c in long_df.columns), "attack_leaf")
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    if ae_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    attack_labels = _unique_display_map(attacks)
    opinion_labels = _unique_display_map(opinions)
    atk_l    = [_wrap_label(attack_labels[a], 18) for a in attacks]
    op_l     = [_wrap_label(opinion_labels[o], 18) for o in opinions]

    mean_m = (long_df.groupby([atk_col, op_col])[ae_col].mean()
              .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0))
    isd_m  = (long_df.groupby([atk_col, op_col])[ae_col].std()
              .unstack(op_col).reindex(index=attacks, columns=opinions).fillna(0))

    total_cells = len(attacks) * len(opinions)
    # Always side-by-side; let Plotly auto-manage the axis tick labels
    rows, cols = 1, 2
    fig = make_subplots(
        rows,
        cols,
        subplot_titles=["Mean AE (red = manipulation succeeded)", "Inter-individual SD of AE"],
        horizontal_spacing=0.20,
    )
    text_size = 11 if total_cells <= 25 else 9 if total_cells <= 49 else 0
    show_text = text_size > 0

    common = dict(
        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>%{z:.1f}<extra></extra>",
        xgap=4,
        ygap=4,
    )
    mean_kwargs = {
        "z": mean_m.values,
        "x": op_l,
        "y": atk_l,
        "colorscale": "RdBu_r",
        "zmid": 0,
        "colorbar": dict(
            x=0.44, y=0.50, len=0.82,
            thickness=12, title="AE", title_side="right",
        ),
        **common,
    }
    isd_kwargs = {
        "z": isd_m.values,
        "x": op_l,
        "y": atk_l,
        "colorscale": "YlOrRd",
        "colorbar": dict(
            x=1.02, y=0.50, len=0.82,
            thickness=12, title="SD", title_side="right",
        ),
        **common,
    }
    if show_text:
        mean_kwargs.update(
            text=[[f"{v:.1f}" for v in row] for row in mean_m.values],
            texttemplate="%{text}",
            textfont=dict(size=text_size, color="white"),
        )
        isd_kwargs.update(
            text=[[f"{v:.1f}" for v in row] for row in isd_m.values],
            texttemplate="%{text}",
            textfont=dict(size=text_size, color="white"),
        )

    fig.add_trace(go.Heatmap(**mean_kwargs), row=1, col=1)
    fig.add_trace(go.Heatmap(**isd_kwargs), row=1, col=2)

    fig.update_xaxes(tickangle=-30, tickfont_size=9, automargin=True)
    fig.update_yaxes(tickfont_size=9, automargin=True, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_annotations(font=dict(size=12.5))
    # Generous left margin keeps y-axis attack labels from overlapping the left heatmap.
    # The colorbar for the left panel is pinned just left of centre (x=0.44); the right
    # colorbar sits at x=1.01.  horizontal_spacing=0.20 gives each panel room to breathe.
    n_attacks  = len(attacks)
    n_opinions = len(opinions)
    row_h = max(28, min(52, 380 // max(n_attacks, 1)))
    col_w = max(22, min(55, 480 // max(n_opinions, 1)))
    dynamic_height = max(420, row_h * n_attacks + 160)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=dynamic_height,
        margin=dict(l=170, r=90, t=70, b=140),
        title=dict(
            text="Factorial Heatmap — Mean AE (left) · Inter-individual SD of AE (right)",
            font_size=13,
            x=0.5,
            xanchor="center",
        ),
    )
    return fig


def _fig_sem_network(
    sem_coeff_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """3D hierarchical SEM map with profile ontology, attack scope, and opinion space."""
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return go.Figure().add_annotation(text="No SEM path data", showarrow=False)

    opinion_lookup: Dict[str, str] = {}
    opinion_group_lookup: Dict[str, str] = {}
    attack_stats = pd.DataFrame()
    if long_df is not None and not long_df.empty:
        opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
        attack_col = next((c for c in ("attack_execute_tactic", "attack_leaf_label", "attack_leaf") if c in long_df.columns), "attack_leaf")
        if opinion_col in long_df.columns:
            opinion_values = sorted(long_df[opinion_col].dropna().unique())
            opinion_display = _unique_display_map(opinion_values)
            for opinion_value in opinion_values:
                leaf_name = _leaf(opinion_value)
                opinion_lookup.setdefault(leaf_name, opinion_display[opinion_value])
                opinion_group_lookup.setdefault(leaf_name, _path_context(opinion_value, keep=1) or "Opinion Targets")
        if attack_col in long_df.columns:
            attack_stats = (
                long_df.groupby(attack_col, as_index=False)
                .agg(
                    mean_ae=("adversarial_effectivity", "mean"),
                    mean_abs=("abs_delta_score", "mean"),
                    sd_ae=("adversarial_effectivity", "std"),
                    n_rows=("scenario_id", "count"),
                )
            )

    df["rhs_label"] = df["rhs"].astype(str).map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
    df["lhs_leaf"] = df["lhs"].astype(str).map(_pretty_indicator)
    df["lhs_label"] = df["lhs_leaf"].map(lambda leaf: opinion_lookup.get(leaf, leaf))
    df["mod_root"] = df["rhs_label"].map(lambda s: _infer_sem_moderator_groups(s)[0])
    df["mod_group"] = df["rhs_label"].map(lambda s: _infer_sem_moderator_groups(s)[1])
    df["ind_group"] = df["lhs_leaf"].map(lambda s: opinion_group_lookup.get(s, "Opinion Targets"))

    mod_rank = (
        df.groupby(["mod_root", "rhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["mod_root", "abs_est", "rhs_label"], ascending=[True, False, True])
    )
    ind_rank = (
        df.groupby(["ind_group", "lhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["ind_group", "abs_est", "lhs_label"], ascending=[True, False, True])
    )

    mod_groups = (
        mod_rank.groupby("mod_root", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["mod_root"]
        .tolist()
    )
    ind_groups = (
        ind_rank.groupby("ind_group", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["ind_group"]
        .tolist()
    )
    mod_group_members = {
        group: mod_rank.loc[mod_rank["mod_root"] == group, "rhs_label"].tolist()
        for group in mod_groups
    }
    ind_group_members = {
        group: ind_rank.loc[ind_rank["ind_group"] == group, "lhs_label"].tolist()
        for group in ind_groups
    }

    def _lane_layout(group_members: Dict[str, List[str]], group_order: List[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        n_groups = max(len(group_order), 1)
        z_vals = np.linspace((n_groups - 1) * 2.6 / 2, -(n_groups - 1) * 2.6 / 2, n_groups) if n_groups > 1 else np.array([0.0])
        group_pos: Dict[str, Tuple[float, float]] = {}
        leaf_pos: Dict[str, Tuple[float, float]] = {}
        for z_lane, group in zip(z_vals, group_order):
            labels = group_members.get(group, [])
            if len(labels) <= 1:
                y_vals = np.array([0.0] * max(len(labels), 1))
            else:
                y_vals = np.linspace((len(labels) - 1) * 1.15 / 2, -(len(labels) - 1) * 1.15 / 2, len(labels))
            group_pos[group] = (0.0, float(z_lane))
            for y_val, label in zip(y_vals, labels):
                leaf_pos[label] = (float(y_val), float(z_lane))
        return group_pos, leaf_pos

    mod_group_pos, mod_leaf_pos = _lane_layout(mod_group_members, mod_groups)
    ind_group_pos, ind_leaf_pos = _lane_layout(ind_group_members, ind_groups)

    if not attack_stats.empty:
        attack_stats = attack_stats.copy()
        attack_stats["attack_label"] = attack_stats[attack_stats.columns[0]].astype(str)
        attack_stats["attack_group"] = attack_stats["attack_label"].map(lambda s: _path_context(s, keep=1) or "Attack Scope")
        attack_order = (
            attack_stats.groupby("attack_group", as_index=False)["mean_abs"]
            .sum()
            .sort_values("mean_abs", ascending=False)["attack_group"]
            .tolist()
        )
        attack_group_members = {
            group: attack_stats.loc[attack_stats["attack_group"] == group]
            .sort_values("mean_abs", ascending=False)["attack_label"].tolist()
            for group in attack_order
        }
        attack_group_pos, attack_leaf_pos = _lane_layout(attack_group_members, attack_order)
    else:
        attack_order = []
        attack_group_pos = {}
        attack_leaf_pos = {}

    mod_order = [label for group in mod_groups for label in mod_group_members[group]]
    ind_order = [label for group in ind_groups for label in ind_group_members[group]]
    mod_codes = {label: f"M{i+1:02d}" for i, label in enumerate(mod_order)}
    ind_codes = {label: f"O{i+1:02d}" for i, label in enumerate(ind_order)}
    attack_codes = {label: f"A{i+1:02d}" for i, label in enumerate(sum([attack_group_members[g] for g in attack_order], []))}

    max_abs = float(df["estimate"].abs().max() or 1.0)
    group_edges = (
        df.groupby(["mod_root", "ind_group"], as_index=False)
        .agg(
            mean_est=("estimate", "mean"),
            mean_abs=("estimate", lambda s: float(np.mean(np.abs(s)))),
            min_p=("p_value", "min"),
            n_paths=("estimate", "count"),
        )
    )

    def _line_rgba(est: float, p_val: float | None, strong_alpha: float = 0.95) -> str:
        if p_val is None or pd.isna(p_val):
            alpha = 0.18
        elif p_val < 0.01:
            alpha = strong_alpha
        elif p_val < 0.05:
            alpha = 0.78
        elif p_val < 0.10:
            alpha = 0.48
        else:
            alpha = 0.18
        return f"rgba(29,78,137,{alpha})" if est >= 0 else f"rgba(192,57,43,{alpha})"

    x_mod_group, x_mod_leaf, x_center, x_ind_group, x_ind_leaf = 0.0, 1.0, 2.35, 3.7, 4.7
    z_all = [pos[1] for pos in mod_group_pos.values()] + [pos[1] for pos in ind_group_pos.values()] + [pos[1] for pos in attack_group_pos.values()]
    z_min = min(z_all) - 1.2 if z_all else -2.0
    z_max = max(z_all) + 1.2 if z_all else 2.0
    y_max_candidates = [abs(pos[0]) for pos in mod_leaf_pos.values()] + [abs(pos[0]) for pos in ind_leaf_pos.values()] + [abs(pos[0]) for pos in attack_leaf_pos.values()]
    y_limit = max(y_max_candidates + [1.8]) + 0.9

    traces: List[go.BaseTraceType] = []
    group_edge_count = 0
    leaf_edge_count = 0
    leaf_edge_meta: List[Dict[str, Any]] = []

    plane_y = np.array([[-y_limit, y_limit], [-y_limit, y_limit]])
    plane_z = np.array([[z_min, z_min], [z_max, z_max]])
    plane_x = np.full((2, 2), x_center)
    traces.append(go.Surface(
        x=plane_x,
        y=plane_y,
        z=plane_z,
        showscale=False,
        opacity=0.14,
        hoverinfo="skip",
        colorscale=[[0, "rgba(240,192,64,0.28)"], [1, "rgba(231,111,81,0.18)"]],
        name="AE corridor",
    ))

    for _, row in group_edges.iterrows():
        mod_group = str(row["mod_root"])
        ind_group = str(row["ind_group"])
        if mod_group not in mod_group_pos or ind_group not in ind_group_pos:
            continue
        y0, z0 = mod_group_pos[mod_group]
        y1, z1 = ind_group_pos[ind_group]
        est = float(row["mean_est"])
        p_val = float(row["min_p"]) if pd.notna(row["min_p"]) else None
        p_text = f"{p_val:.4f}" if p_val is not None else "n/a"
        traces.append(go.Scatter3d(
            x=[x_mod_group, 1.55, x_center, 3.05, x_ind_group],
            y=[y0, y0 * 0.5, (y0 + y1) / 2, y1 * 0.5, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
            mode="lines",
            line=dict(color=_line_rgba(est, p_val, strong_alpha=0.88), width=max(5.0, row["mean_abs"] / max_abs * 12.0)),
            hovertemplate=(
                f"<b>{mod_group}</b> → <b>{ind_group}</b><br>"
                f"Mean β = {est:.3f}<br>"
                f"Mean |β| = {float(row['mean_abs']):.3f}<br>"
                f"Paths aggregated = {int(row['n_paths'])}<br>"
                f"Best p = {p_text}<extra>Group summary</extra>"
            ),
            showlegend=False,
            visible=True,
        ))
        group_edge_count += 1

    for _, row in df.iterrows():
        rhs = str(row["rhs_label"])
        lhs = str(row["lhs_label"])
        if rhs not in mod_leaf_pos or lhs not in ind_leaf_pos:
            continue
        y0, z0 = mod_leaf_pos[rhs]
        y1, z1 = ind_leaf_pos[lhs]
        est = float(row["estimate"])
        p_val = float(row["p_value"]) if pd.notna(row["p_value"]) else None
        p_text = f"{p_val:.4f}" if p_val is not None else "n/a"
        traces.append(go.Scatter3d(
            x=[x_mod_leaf, 1.75, x_center, 2.95, x_ind_leaf],
            y=[y0, y0 * 0.62, (y0 + y1) / 2, y1 * 0.62, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
            mode="lines",
            line=dict(color=_line_rgba(est, p_val), width=max(3.0, abs(est) / max_abs * 9.0)),
            hovertemplate=(
                f"<b>{mod_codes.get(rhs, rhs)}</b> {rhs}<br>"
                f"→ <b>{ind_codes.get(lhs, lhs)}</b> {lhs}<br>"
                f"β = {est:.3f} {_p_stars(p_val)}<br>"
                f"p = {p_text}<extra>Leaf path</extra>"
            ),
            showlegend=False,
            visible=False,
        ))
        leaf_edge_meta.append({"p": p_val if p_val is not None else 1.0})
        leaf_edge_count += 1

    # group nodes
    traces.extend([
        go.Scatter3d(
            x=[x_mod_group] * len(mod_groups),
            y=[mod_group_pos[g][0] for g in mod_groups],
            z=[mod_group_pos[g][1] for g in mod_groups],
            mode="markers+text",
            marker=dict(size=16, color="#dbe8fb", symbol="square", line=dict(color=PALETTE["navy"], width=2)),
            text=[_clip_label(g, 22) for g in mod_groups],
            textposition="middle left",
            textfont=dict(size=10, color=PALETTE["navy"]),
            customdata=np.array(mod_groups, dtype=object),
            hovertemplate="<b>%{customdata}</b><extra>Profile family</extra>",
            showlegend=False,
            visible=True,
        ),
        go.Scatter3d(
            x=[x_ind_group] * len(ind_groups),
            y=[ind_group_pos[g][0] for g in ind_groups],
            z=[ind_group_pos[g][1] for g in ind_groups],
            mode="markers+text",
            marker=dict(size=16, color="#d8f2ef", symbol="square", line=dict(color=PALETTE["teal"], width=2)),
            text=[_clip_label(g, 22) for g in ind_groups],
            textposition="middle right",
            textfont=dict(size=10, color=PALETTE["teal"]),
            customdata=np.array(ind_groups, dtype=object),
            hovertemplate="<b>%{customdata}</b><extra>Opinion family</extra>",
            showlegend=False,
            visible=True,
        ),
    ])

    # attack scope nodes
    if not attack_stats.empty:
        attack_hover = []
        attack_sizes = []
        attack_text = []
        attack_labels_order = sum([attack_group_members[g] for g in attack_order], [])
        mean_abs_max = float(max(attack_stats["mean_abs"].max(), 0.01))
        for attack_label in attack_labels_order:
            row = attack_stats.loc[attack_stats["attack_label"] == attack_label].iloc[0]
            attack_hover.append([
                attack_codes[attack_label],
                attack_label,
                float(row["mean_ae"]),
                float(row["mean_abs"]),
                float(row["sd_ae"]) if pd.notna(row["sd_ae"]) else 0.0,
                int(row["n_rows"]),
            ])
            attack_sizes.append(10 + float(row["mean_abs"]) / mean_abs_max * 14)
            attack_text.append(attack_codes[attack_label] if len(attack_labels_order) > 5 else _clip_label(_leaf(attack_label), 16))
        traces.append(go.Scatter3d(
            x=[x_center] * len(attack_labels_order),
            y=[attack_leaf_pos[a][0] for a in attack_labels_order],
            z=[attack_leaf_pos[a][1] for a in attack_labels_order],
            mode="markers+text",
            marker=dict(
                size=attack_sizes,
                color=[v[3] for v in attack_hover],
                colorscale="YlOrRd",
                line=dict(color="white", width=1.6),
                opacity=0.92,
            ),
            text=attack_text,
            textposition="top center",
            textfont=dict(size=9, color=PALETTE["orange"]),
            customdata=np.array(attack_hover, dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                "Mean AE = %{customdata[2]:.2f}<br>"
                "Mean |Δ| = %{customdata[3]:.2f}<br>"
                "SD AE = %{customdata[4]:.2f}<br>"
                "Rows = %{customdata[5]}<extra>Attack scope context</extra>"
            ),
            showlegend=False,
            visible=True,
        ))

    traces.extend([
        go.Scatter3d(
            x=[x_mod_leaf] * len(mod_order),
            y=[mod_leaf_pos[label][0] for label in mod_order],
            z=[mod_leaf_pos[label][1] for label in mod_order],
            mode="markers+text",
            marker=dict(size=10, color=PALETTE["navy"], symbol="circle", line=dict(color="white", width=1.6)),
            text=[mod_codes[label] for label in mod_order],
            textposition="middle left",
            textfont=dict(size=9, color=PALETTE["ink"]),
            customdata=np.array([[mod_codes[label], label, _infer_sem_moderator_groups(label)[0]] for label in mod_order], dtype=object),
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>Family: %{customdata[2]}<extra>Moderator leaf</extra>",
            showlegend=False,
            visible=False,
        ),
        go.Scatter3d(
            x=[x_ind_leaf] * len(ind_order),
            y=[ind_leaf_pos[label][0] for label in ind_order],
            z=[ind_leaf_pos[label][1] for label in ind_order],
            mode="markers+text",
            marker=dict(size=10, color=PALETTE["teal"], symbol="diamond", line=dict(color="white", width=1.6)),
            text=[ind_codes[label] for label in ind_order],
            textposition="middle right",
            textfont=dict(size=9, color=PALETTE["ink"]),
            customdata=np.array([[ind_codes[label], label, next((g for g in ind_groups if label in ind_group_members[g]), "Opinion Targets")] for label in ind_order], dtype=object),
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>Family: %{customdata[2]}<extra>Opinion leaf</extra>",
            showlegend=False,
            visible=False,
        ),
    ])

    # trace visibility masks
    base_count = 1  # center plane
    group_edge_start = 1
    leaf_edge_start = group_edge_start + group_edge_count
    node_start = leaf_edge_start + leaf_edge_count
    total_traces = len(traces)

    attack_trace_count = 1 if not attack_stats.empty else 0
    always_visible_idx = {0, node_start, node_start + 1}
    if attack_trace_count:
        always_visible_idx.add(node_start + 2)
    mod_leaf_trace_idx = node_start + 2 + attack_trace_count
    ind_leaf_trace_idx = node_start + 3 + attack_trace_count

    def _mask(view: str) -> List[bool]:
        vis = [False] * total_traces
        for idx in always_visible_idx:
            if idx < total_traces:
                vis[idx] = True
        if view == "group":
            for idx in range(group_edge_start, leaf_edge_start):
                vis[idx] = True
        elif view == "leaf":
            vis[mod_leaf_trace_idx] = True
            vis[ind_leaf_trace_idx] = True
            for idx in range(leaf_edge_start, node_start):
                vis[idx] = True
        elif view == "sig":
            vis[mod_leaf_trace_idx] = True
            vis[ind_leaf_trace_idx] = True
            for idx, meta in enumerate(leaf_edge_meta, start=leaf_edge_start):
                vis[idx] = meta["p"] < 0.05
        elif view == "hsig":
            vis[mod_leaf_trace_idx] = True
            vis[ind_leaf_trace_idx] = True
            for idx, meta in enumerate(leaf_edge_meta, start=leaf_edge_start):
                vis[idx] = meta["p"] < 0.01
        return vis

    cameras = {
        "Perspective": dict(eye=dict(x=1.75, y=1.55, z=0.95)),
        "Profile Side": dict(eye=dict(x=0.15, y=2.3, z=0.55)),
        "Opinion Side": dict(eye=dict(x=-0.15, y=-2.3, z=0.55)),
        "Top Down": dict(eye=dict(x=0.0, y=0.15, z=2.65)),
    }

    fig = go.Figure(traces)
    fig.update_layout(
        paper_bgcolor="white",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(760, 90 * max(len(mod_groups), len(ind_groups)) + 320),
        title=dict(text="Hierarchical Structural Equation Model — 3D Moderation Space", font_size=14),
        margin=dict(l=20, r=20, t=92, b=110),
        scene=dict(
            xaxis=dict(visible=False, range=[-0.35, 5.05]),
            yaxis=dict(visible=False, range=[-y_limit, y_limit]),
            zaxis=dict(visible=False, range=[z_min, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=2.4, y=1.6, z=1.1),
            bgcolor="white",
            camera=dict(eye=dict(x=1.75, y=1.55, z=0.95)),
        ),
        annotations=[
            dict(x=0.12, y=1.08, xref="paper", yref="paper", showarrow=False,
                 text="<b>PROFILE ONTOLOGY</b>", font=dict(size=11, color=PALETTE["navy"])),
            dict(x=0.50, y=1.08, xref="paper", yref="paper", showarrow=False,
                 text="<b>ATTACK-EFFECTIVITY SCOPE</b>", font=dict(size=11, color=PALETTE["orange"])),
            dict(x=0.88, y=1.08, xref="paper", yref="paper", showarrow=False,
                 text="<b>OPINION SPACE</b>", font=dict(size=11, color=PALETTE["teal"])),
            dict(x=0.50, y=-0.13, xref="paper", yref="paper", showarrow=False,
                 text=(
                     "Default view collapses to group summary for readability. Switch to leaf detail for M## / O## codes; hover nodes and paths for full labels and coefficients.<br>"
                     "Attack nodes in the middle summarize the manipulated attack space for the run; SEM paths themselves connect profile-side moderators to attack-effectivity indicators in opinion space."
                 ),
                 font=dict(size=8.6, color=PALETTE["muted"])),
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.24,
                xanchor="center",
                y=1.17,
                yanchor="top",
                buttons=[
                    dict(label="Group Summary", method="update", args=[{"visible": _mask("group")}]),
                    dict(label="Leaf Detail", method="update", args=[{"visible": _mask("leaf")}]),
                    dict(label="Significant", method="update", args=[{"visible": _mask("sig")}]),
                    dict(label="Highly Significant", method="update", args=[{"visible": _mask("hsig")}]),
                ],
                bgcolor=PALETTE["panel"],
                bordercolor=PALETTE["line"],
                font=dict(size=10),
                pad=dict(l=4, r=4, t=4, b=4),
            ),
            dict(
                type="buttons",
                direction="right",
                x=0.76,
                xanchor="center",
                y=1.17,
                yanchor="top",
                buttons=[
                    dict(label=name, method="relayout", args=[{"scene.camera": camera}])
                    for name, camera in cameras.items()
                ],
                bgcolor=PALETTE["panel"],
                bordercolor=PALETTE["line"],
                font=dict(size=10),
                pad=dict(l=4, r=4, t=4, b=4),
            ),
        ],
    )
    fig.update_traces(selector=dict(type="surface"), visible=True)
    fig.update_layout(showlegend=False)
    for idx, visible in enumerate(_mask("group")):
        fig.data[idx].visible = visible
    return fig


def _html_sem_network(
    sem_coeff_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
) -> str:
    """Custom interactive 3D hierarchical SEM network with explicit UI controls."""
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return "<p>No SEM path data available.</p>"

    opinion_lookup: Dict[str, str] = {}
    opinion_group_lookup: Dict[str, str] = {}
    attack_stats = pd.DataFrame()
    if long_df is not None and not long_df.empty:
        opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
        attack_col = next((c for c in ("attack_execute_tactic", "attack_leaf_label", "attack_leaf") if c in long_df.columns), "attack_leaf")
        if opinion_col in long_df.columns:
            opinion_values = sorted(long_df[opinion_col].dropna().unique())
            opinion_display = _unique_display_map(opinion_values)
            for opinion_value in opinion_values:
                leaf_name = _pretty_indicator(str(opinion_value))
                opinion_lookup.setdefault(leaf_name, opinion_display[opinion_value])
                opinion_group_lookup.setdefault(leaf_name, _path_context(opinion_value, keep=1) or "Opinion Targets")
        if attack_col in long_df.columns:
            attack_stats = (
                long_df.groupby(attack_col, as_index=False)
                .agg(
                    mean_ae=("adversarial_effectivity", "mean"),
                    mean_abs=("abs_delta_score", "mean"),
                    sd_ae=("adversarial_effectivity", "std"),
                    n_rows=("scenario_id", "count"),
                )
                .rename(columns={attack_col: "attack_label"})
            )

    df["rhs_label"] = df["rhs"].astype(str).map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
    df["lhs_leaf"] = df["lhs"].astype(str).map(_pretty_indicator)
    df["lhs_label"] = df["lhs_leaf"].map(lambda leaf: opinion_lookup.get(leaf, leaf))
    df["mod_root"] = df["rhs_label"].map(lambda s: _infer_sem_moderator_groups(s)[0])
    df["ind_group"] = df["lhs_leaf"].map(lambda s: opinion_group_lookup.get(s, "Opinion Targets"))
    df["mod_role"] = df["mod_root"].map(
        lambda g: "control" if g == "Model Controls" else "demographic" if g == "Demographics" else "profile"
    )

    mod_rank = (
        df.groupby(["mod_root", "mod_role", "rhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["mod_root", "abs_est", "rhs_label"], ascending=[True, False, True])
    )
    ind_rank = (
        df.groupby(["ind_group", "lhs_label"], as_index=False)
        .agg(abs_est=("estimate", lambda s: float(np.max(np.abs(s)))))
        .sort_values(["ind_group", "abs_est", "lhs_label"], ascending=[True, False, True])
    )

    mod_groups = (
        mod_rank.groupby("mod_root", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["mod_root"]
        .tolist()
    )
    ind_groups = (
        ind_rank.groupby("ind_group", as_index=False)["abs_est"]
        .sum()
        .sort_values("abs_est", ascending=False)["ind_group"]
        .tolist()
    )
    mod_group_members = {
        group: mod_rank.loc[mod_rank["mod_root"] == group, "rhs_label"].tolist()
        for group in mod_groups
    }
    ind_group_members = {
        group: ind_rank.loc[ind_rank["ind_group"] == group, "lhs_label"].tolist()
        for group in ind_groups
    }
    mod_role_lookup = {row["rhs_label"]: row["mod_role"] for _, row in mod_rank.iterrows()}

    def _lane_layout(group_members: Dict[str, List[str]], group_order: List[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        n_groups = max(len(group_order), 1)
        z_vals = np.linspace((n_groups - 1) * 2.6 / 2, -(n_groups - 1) * 2.6 / 2, n_groups) if n_groups > 1 else np.array([0.0])
        group_pos: Dict[str, Tuple[float, float]] = {}
        leaf_pos: Dict[str, Tuple[float, float]] = {}
        for z_lane, group in zip(z_vals, group_order):
            labels = group_members.get(group, [])
            if len(labels) <= 1:
                y_vals = np.array([0.0] * max(len(labels), 1))
            else:
                y_vals = np.linspace((len(labels) - 1) * 1.12 / 2, -(len(labels) - 1) * 1.12 / 2, len(labels))
            group_pos[group] = (0.0, float(z_lane))
            for y_val, label in zip(y_vals, labels):
                leaf_pos[label] = (float(y_val), float(z_lane))
        return group_pos, leaf_pos

    def _reorder_within_groups(
        frame: pd.DataFrame,
        src_col: str,
        src_groups: Dict[str, List[str]],
        tgt_col: str,
        tgt_groups: Dict[str, List[str]],
        n_iter: int = 4,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        src_members = {k: list(v) for k, v in src_groups.items()}
        tgt_members = {k: list(v) for k, v in tgt_groups.items()}
        for _ in range(n_iter):
            _, tgt_leaf = _lane_layout(tgt_members, list(tgt_members.keys()))
            src_scores = (
                frame.groupby(src_col)[[tgt_col, "estimate"]]
                .apply(lambda g: float(np.average(
                    [tgt_leaf[val][0] + tgt_leaf[val][1] * 0.15 for val in g[tgt_col]],
                    weights=np.abs(g["estimate"]).to_numpy(),
                )))
                .to_dict()
            )
            for group, labels in src_members.items():
                src_members[group] = sorted(labels, key=lambda label: (src_scores.get(label, 0.0), label))

            _, src_leaf = _lane_layout(src_members, list(src_members.keys()))
            tgt_scores = (
                frame.groupby(tgt_col)[[src_col, "estimate"]]
                .apply(lambda g: float(np.average(
                    [src_leaf[val][0] + src_leaf[val][1] * 0.15 for val in g[src_col]],
                    weights=np.abs(g["estimate"]).to_numpy(),
                )))
                .to_dict()
            )
            for group, labels in tgt_members.items():
                tgt_members[group] = sorted(labels, key=lambda label: (tgt_scores.get(label, 0.0), label))
        return src_members, tgt_members

    mod_group_members, ind_group_members = _reorder_within_groups(
        df,
        "rhs_label",
        mod_group_members,
        "lhs_label",
        ind_group_members,
    )

    mod_group_pos, mod_leaf_pos = _lane_layout(mod_group_members, mod_groups)
    ind_group_pos, ind_leaf_pos = _lane_layout(ind_group_members, ind_groups)

    attack_group_members: Dict[str, List[str]] = {}
    attack_group_pos: Dict[str, Tuple[float, float]] = {}
    attack_leaf_pos: Dict[str, Tuple[float, float]] = {}
    attack_order: List[str] = []
    if not attack_stats.empty:
        attack_stats["attack_group"] = attack_stats["attack_label"].map(lambda s: _path_context(s, keep=1) or "Attack Scope")
        attack_order = (
            attack_stats.groupby("attack_group", as_index=False)["mean_abs"]
            .sum()
            .sort_values("mean_abs", ascending=False)["attack_group"]
            .tolist()
        )
        attack_group_members = {
            group: attack_stats.loc[attack_stats["attack_group"] == group]
            .sort_values("mean_abs", ascending=False)["attack_label"].tolist()
            for group in attack_order
        }
        attack_group_pos, attack_leaf_pos = _lane_layout(attack_group_members, attack_order)

    mod_order = [label for group in mod_groups for label in mod_group_members[group]]
    ind_order = [label for group in ind_groups for label in ind_group_members[group]]
    attack_order_flat = [label for group in attack_order for label in attack_group_members[group]]

    mod_codes = {label: f"M{i+1:02d}" for i, label in enumerate(mod_order)}
    ind_codes = {label: f"O{i+1:02d}" for i, label in enumerate(ind_order)}
    attack_codes = {label: f"A{i+1:02d}" for i, label in enumerate(attack_order_flat)}

    x_mod_group, x_mod_leaf, x_attack_group, x_attack_leaf, x_ind_group, x_ind_leaf = 0.0, 1.0, 2.0, 2.45, 3.7, 4.7

    profile_group_nodes = [
        dict(
            id=f"pg::{group}",
            label=group,
            short=_clip_label(group, 24),
            x=x_mod_group,
            y=mod_group_pos[group][0],
            z=mod_group_pos[group][1],
            family=group,
            role="group",
        )
        for group in mod_groups
    ]
    profile_leaf_nodes = [
        dict(
            id=f"pl::{label}",
            code=mod_codes[label],
            label=label,
            short=_clip_label(label.replace("Big Five ", ""), 24),
            x=x_mod_leaf,
            y=mod_leaf_pos[label][0],
            z=mod_leaf_pos[label][1],
            family=next((g for g in mod_groups if label in mod_group_members[g]), "Other Moderators"),
            role=mod_role_lookup.get(label, "profile"),
        )
        for label in mod_order
    ]
    opinion_group_nodes = [
        dict(
            id=f"og::{group}",
            label=group,
            short=_clip_label(group, 24),
            x=x_ind_group,
            y=ind_group_pos[group][0],
            z=ind_group_pos[group][1],
            family=group,
            role="group",
        )
        for group in ind_groups
    ]
    opinion_leaf_nodes = [
        dict(
            id=f"ol::{label}",
            code=ind_codes[label],
            label=label,
            short=_clip_label(label, 24),
            x=x_ind_leaf,
            y=ind_leaf_pos[label][0],
            z=ind_leaf_pos[label][1],
            family=next((g for g in ind_groups if label in ind_group_members[g]), "Opinion Targets"),
            role="leaf",
        )
        for label in ind_order
    ]
    attack_group_nodes = [
        dict(
            id=f"ag::{group}",
            label=group,
            short=_clip_label(group, 20),
            x=x_attack_group,
            y=attack_group_pos[group][0],
            z=attack_group_pos[group][1],
            family=group,
            role="group",
        )
        for group in attack_order
    ]
    attack_leaf_nodes = []
    if not attack_stats.empty:
        mean_abs_max = float(max(attack_stats["mean_abs"].max(), 0.01))
        for attack_label in attack_order_flat:
            row = attack_stats.loc[attack_stats["attack_label"] == attack_label].iloc[0]
            attack_leaf_nodes.append(dict(
                id=f"al::{attack_label}",
                code=attack_codes[attack_label],
                label=attack_label,
                short=_clip_label(_leaf(attack_label), 22),
                x=x_attack_leaf,
                y=attack_leaf_pos[attack_label][0],
                z=attack_leaf_pos[attack_label][1],
                family=row["attack_group"],
                mean_ae=float(row["mean_ae"]),
                mean_abs=float(row["mean_abs"]),
                sd_ae=float(row["sd_ae"]) if pd.notna(row["sd_ae"]) else 0.0,
                n_rows=int(row["n_rows"]),
                size=float(10 + float(row["mean_abs"]) / mean_abs_max * 14),
            ))

    group_edges = []
    grouped = (
        df.groupby(["mod_root", "ind_group"], as_index=False)
        .agg(
            mean_est=("estimate", "mean"),
            mean_abs=("estimate", lambda s: float(np.mean(np.abs(s)))),
            min_p=("p_value", "min"),
            n_paths=("estimate", "count"),
        )
    )
    for _, row in grouped.iterrows():
        mod_group = str(row["mod_root"])
        ind_group = str(row["ind_group"])
        if mod_group not in mod_group_pos or ind_group not in ind_group_pos:
            continue
        y0, z0 = mod_group_pos[mod_group]
        y1, z1 = ind_group_pos[ind_group]
        group_edges.append(dict(
            id=f"ge::{mod_group}::{ind_group}",
            source_group=mod_group,
            target_group=ind_group,
            source_role="group",
            estimate=float(row["mean_est"]),
            abs_est=float(row["mean_abs"]),
            p=float(row["min_p"]) if pd.notna(row["min_p"]) else 1.0,
            n_paths=int(row["n_paths"]),
            x=[x_mod_group, 1.10, 2.15, 3.00, x_ind_group],
            y=[y0, y0 * 0.58, (y0 + y1) / 2, y1 * 0.58, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
        ))

    leaf_edges = []
    for _, row in df.iterrows():
        rhs = str(row["rhs_label"])
        lhs = str(row["lhs_label"])
        if rhs not in mod_leaf_pos or lhs not in ind_leaf_pos:
            continue
        y0, z0 = mod_leaf_pos[rhs]
        y1, z1 = ind_leaf_pos[lhs]
        src_group = next((g for g in mod_groups if rhs in mod_group_members[g]), "Other Moderators")
        tgt_group = next((g for g in ind_groups if lhs in ind_group_members[g]), "Opinion Targets")
        leaf_edges.append(dict(
            id=f"le::{rhs}::{lhs}",
            source=rhs,
            target=lhs,
            source_code=mod_codes[rhs],
            target_code=ind_codes[lhs],
            source_group=src_group,
            target_group=tgt_group,
            source_role=mod_role_lookup.get(rhs, "profile"),
            estimate=float(row["estimate"]),
            abs_est=float(abs(row["estimate"])),
            p=float(row["p_value"]) if pd.notna(row["p_value"]) else 1.0,
            x=[x_mod_leaf, 1.75, 2.35, 2.95, x_ind_leaf],
            y=[y0, y0 * 0.62, (y0 + y1) / 2, y1 * 0.62, y1],
            z=[z0, z0, (z0 + z1) / 2, z1, z1],
        ))

    payload = {
        "profile_groups": mod_groups,
        "opinion_groups": ind_groups,
        "profile_group_nodes": profile_group_nodes,
        "profile_leaf_nodes": profile_leaf_nodes,
        "attack_group_nodes": attack_group_nodes,
        "attack_leaf_nodes": attack_leaf_nodes,
        "opinion_group_nodes": opinion_group_nodes,
        "opinion_leaf_nodes": opinion_leaf_nodes,
        "group_edges": group_edges,
        "leaf_edges": leaf_edges,
        "max_abs_beta": float(max(df["estimate"].abs().max(), 0.01)),
        "z_range": [
            float(min(
                [node["z"] for node in profile_group_nodes + opinion_group_nodes + attack_group_nodes] + [-1.8]
            ) - 1.2),
            float(max(
                [node["z"] for node in profile_group_nodes + opinion_group_nodes + attack_group_nodes] + [1.8]
            ) + 1.2),
        ],
        "y_limit": float(max(
            [abs(node["y"]) for node in profile_leaf_nodes + opinion_leaf_nodes + attack_leaf_nodes] + [1.8]
        ) + 0.95),
    }

    profile_filter_html = "".join(
        f"""<label class="semn-chip"><input type="checkbox" class="semn-profile-group" value="{group}" checked> <span>{group}</span></label>"""
        for group in mod_groups
    )
    opinion_filter_html = "".join(
        f"""<label class="semn-chip"><input type="checkbox" class="semn-opinion-group" value="{group}" checked> <span>{group}</span></label>"""
        for group in ind_groups
    )

    return f"""
<div id="semn-root">
  <style>
    #semn-root .semn-shell{{display:grid;grid-template-columns:minmax(290px,330px) minmax(0,1fr);gap:16px;align-items:start}}
    #semn-root .semn-card{{background:#f7faff;border:1px solid #dbe3ef;border-radius:12px;padding:12px 13px;box-shadow:0 3px 14px rgba(20,33,61,0.05)}}
    #semn-root .semn-card + .semn-card{{margin-top:10px}}
    #semn-root .semn-title{{font-weight:800;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px}}
    #semn-root .semn-sub{{font-size:0.75rem;color:{PALETTE['muted']};line-height:1.45;margin-bottom:8px}}
    #semn-root .semn-segment{{display:flex;flex-wrap:wrap;gap:6px}}
    #semn-root .semn-btn{{padding:6px 9px;border-radius:999px;border:1px solid #c8d7ec;background:#fff;color:{PALETTE['ink']};cursor:pointer;font-size:0.75rem;font-weight:700}}
    #semn-root .semn-btn.active{{background:{PALETTE['blue']};border-color:{PALETTE['blue']};color:#fff}}
    #semn-root .semn-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
    #semn-root .semn-grid.one{{grid-template-columns:1fr}}
    #semn-root .semn-row{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
    #semn-root .semn-toggle{{display:flex;align-items:center;gap:7px;font-size:0.76rem;color:{PALETTE['ink']};font-weight:600}}
    #semn-root .semn-select{{width:100%;padding:7px 8px;border-radius:8px;border:1px solid #dbe3ef;background:#fff;color:{PALETTE['ink']};font-size:0.80rem}}
    #semn-root .semn-slider-wrap{{background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:9px 10px}}
    #semn-root .semn-slider-meta{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:0.76rem;color:{PALETTE['muted']};font-weight:700}}
    #semn-root input[type="range"]{{width:100%;accent-color:{PALETTE['blue']}}}
    #semn-root .semn-quick{{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}}
    #semn-root .semn-chip{{display:flex;align-items:center;gap:7px;padding:6px 8px;border-radius:9px;background:#fff;border:1px solid #dbe3ef;font-size:0.75rem;color:{PALETTE['ink']}}}
    #semn-root .semn-filter-list{{display:flex;flex-wrap:wrap;gap:6px;max-height:140px;overflow:auto;padding-right:3px}}
    #semn-root .semn-stage{{display:flex;flex-direction:column;gap:12px}}
    #semn-root .semn-banner{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%);border:1px solid #dbe3ef;border-radius:12px;padding:11px 13px}}
    #semn-root .semn-status{{font-size:0.79rem;color:{PALETTE['muted']};line-height:1.45}}
    #semn-root .semn-status strong{{color:{PALETTE['ink']}}}
    #semn-root .semn-legend{{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}}
    #semn-root .semn-legend-item{{font-size:0.73rem;color:{PALETTE['muted']};display:flex;align-items:center;gap:6px}}
    #semn-root .semn-swatch{{display:inline-block;width:28px;height:8px;border-radius:999px}}
    #semn-root .semn-swatch.pos{{background:linear-gradient(90deg,rgba(255,219,210,0.75),rgba(192,57,43,0.95))}}
    #semn-root .semn-swatch.neg{{background:linear-gradient(90deg,rgba(216,232,255,0.75),rgba(29,78,137,0.95))}}
    #semn-root .semn-swatch.sig{{background:linear-gradient(90deg,rgba(120,120,120,0.20),rgba(120,120,120,0.95))}}
    #semn-root #semn-plot{{background:#fff;border:1px solid #dbe3ef;border-radius:14px;min-height:660px}}
    #semn-root .semn-bottom{{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:12px}}
    #semn-root .semn-panel{{background:#fff;border:1px solid #dbe3ef;border-radius:12px;padding:12px 13px}}
    #semn-root .semn-panel h4{{margin:0 0 8px;font-size:0.82rem;color:{PALETTE['navy']}}}
    #semn-root .semn-list{{display:flex;flex-direction:column;gap:7px}}
    #semn-root .semn-item{{padding:7px 8px;border-radius:10px;background:#f8fbff;border:1px solid #e0eaf8}}
    #semn-root .semn-item-top{{display:flex;justify-content:space-between;gap:8px;align-items:center;font-size:0.76rem}}
    #semn-root .semn-item-top strong{{color:{PALETTE['ink']}}}
    #semn-root .semn-badge{{display:inline-flex;align-items:center;padding:2px 6px;border-radius:999px;font-size:0.66rem;font-weight:800;letter-spacing:0.02em}}
    #semn-root .semn-badge.hsig{{background:rgba(231,111,81,0.14);color:{PALETTE['red']}}}
    #semn-root .semn-badge.sig{{background:rgba(29,78,137,0.14);color:{PALETTE['blue']}}}
    #semn-root .semn-badge.weak{{background:rgba(20,33,61,0.08);color:{PALETTE['muted']}}}
    #semn-root .semn-item-sub{{font-size:0.72rem;color:{PALETTE['muted']};line-height:1.4;margin-top:3px}}
    #semn-root .semn-map{{display:flex;flex-wrap:wrap;gap:7px}}
    #semn-root .semn-map span{{padding:4px 6px;border-radius:8px;background:#f7faff;border:1px solid #e0eaf8;font-size:0.72rem;color:{PALETTE['ink']}}}
    @media (max-width: 1120px) {{
      #semn-root .semn-shell{{grid-template-columns:1fr}}
      #semn-root .semn-bottom{{grid-template-columns:1fr}}
    }}
  </style>
  <div class="semn-shell">
    <div>
      <div class="semn-card">
        <div class="semn-title">View Presets</div>
        <div class="semn-sub">Start with a simple baseline summary of PROFILE-side moderation families, then progressively add leaf-level SEM paths and attack-context structure.</div>
        <div class="semn-segment" id="semn-view-mode">
          <button class="semn-btn active" data-view="overview">Baseline</button>
          <button class="semn-btn" data-view="leaf">Leaf Detail</button>
          <button class="semn-btn" data-view="all">All Layers</button>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Path Threshold</div>
        <div class="semn-slider-wrap">
          <div class="semn-slider-meta"><span>Maximum p-value shown</span><span id="semn-p-display">0.10</span></div>
          <input type="range" id="semn-p-slider" min="0" max="100" value="50" step="1">
          <div class="semn-quick">
            <button class="semn-btn" data-p="0.01">0.01</button>
            <button class="semn-btn" data-p="0.05">0.05</button>
            <button class="semn-btn active" data-p="0.10">0.10</button>
            <button class="semn-btn" data-p="1.00">All</button>
          </div>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Effect Size Filter</div>
        <div class="semn-slider-wrap">
          <div class="semn-slider-meta"><span>Min |β| shown</span><span id="semn-b-display">0.00</span></div>
          <input type="range" id="semn-b-slider" min="0" max="100" value="0" step="1">
          <div class="semn-quick">
            <button class="semn-btn active" data-b="0.00">All</button>
            <button class="semn-btn" data-b="0.25">0.25</button>
            <button class="semn-btn" data-b="0.50">0.50</button>
            <button class="semn-btn" data-b="1.00">1.00</button>
          </div>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Search Moderators</div>
        <input id="semn-search" type="text" class="semn-select" placeholder="Type to filter by moderator or outcome name…" style="margin-bottom:6px">
        <div id="semn-search-results" class="semn-map" style="max-height:90px;overflow:auto"></div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Layer Controls</div>
        <div class="semn-sub">Presets set a good default; these switches let you explicitly choose which moderation layers stay visible.</div>
        <div class="semn-grid one">
          <label class="semn-toggle"><input type="checkbox" id="semn-show-group-edges" checked> Show family ribbons</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-leaf-edges"> Show leaf-level paths</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-group-nodes" checked> Show family nodes</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-leaf-nodes"> Show leaf nodes</label>
        </div>
        <div class="semn-row" style="margin-top:10px">
          <label class="semn-toggle"><input type="checkbox" id="semn-include-controls"> Include model controls</label>
          <label class="semn-toggle"><input type="checkbox" id="semn-show-attack" checked> Show attack context</label>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Path Filters</div>
        <div class="semn-grid">
          <div>
            <div class="semn-sub" style="margin-bottom:6px">Direction</div>
            <div class="semn-segment" id="semn-sign-mode">
              <button class="semn-btn active" data-sign="all">All</button>
              <button class="semn-btn" data-sign="positive">Positive</button>
              <button class="semn-btn" data-sign="negative">Negative</button>
            </div>
          </div>
          <div>
            <div class="semn-sub" style="margin-bottom:6px">Labels</div>
            <select id="semn-label-density" class="semn-select">
              <option value="minimal">Minimal</option>
              <option value="codes" selected>Codes</option>
              <option value="short">Short labels</option>
            </select>
          </div>
        </div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Hierarchy Filters</div>
        <div class="semn-sub">Choose which PROFILE-side families and OPINION-space families remain in the 3D moderation view.</div>
        <div class="semn-sub" style="font-weight:700;color:{PALETTE['ink']};margin-bottom:6px">Profile families</div>
        <div class="semn-filter-list">{profile_filter_html}</div>
        <div class="semn-sub" style="font-weight:700;color:{PALETTE['ink']};margin:10px 0 6px">Opinion families</div>
        <div class="semn-filter-list">{opinion_filter_html}</div>
      </div>

      <div class="semn-card">
        <div class="semn-title">Camera</div>
        <div class="semn-segment" id="semn-camera-mode">
          <button class="semn-btn active" data-camera="perspective">Perspective</button>
          <button class="semn-btn" data-camera="profile">Profile Side</button>
          <button class="semn-btn" data-camera="opinion">Opinion Side</button>
          <button class="semn-btn" data-camera="top">Top Down</button>
        </div>
      </div>
    </div>

    <div class="semn-stage">
      <div class="semn-banner">
        <div class="semn-status" id="semn-status"></div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <button class="semn-btn" id="semn-export-btn" style="font-size:0.76rem;padding:5px 10px" title="Download current 3D view as PNG">⬇ Export PNG</button>
          <div class="semn-legend">
            <div class="semn-legend-item"><span class="semn-swatch pos"></span><span>positive β (susceptibility↑)</span></div>
            <div class="semn-legend-item"><span class="semn-swatch neg"></span><span>negative β (resistance↑)</span></div>
            <div class="semn-legend-item"><span class="semn-swatch sig"></span><span>opacity ∝ significance</span></div>
            <div class="semn-legend-item"><span style="font-weight:800;color:{PALETTE['ink']}">|β|</span><span>width ∝ effect size</span></div>
          </div>
        </div>
      </div>
      <div id="semn-plot"></div>
      <div class="semn-bottom">
        <div class="semn-panel">
          <h4>Interpretation</h4>
          <div id="semn-logic" class="semn-sub" style="margin:0"></div>
        </div>
        <div class="semn-panel">
          <h4>Highlighted Paths</h4>
          <div id="semn-focus" class="semn-list"></div>
        </div>
        <div class="semn-panel">
          <h4>Code Legend</h4>
          <div id="semn-map" class="semn-map" style="max-height:120px;overflow:auto"></div>
          <details style="margin-top:8px">
            <summary style="font-size:0.75rem;font-weight:700;color:{PALETTE['navy']};cursor:pointer">Full code list ▸</summary>
            <div id="semn-legend-full" style="margin-top:6px;display:flex;flex-direction:column;gap:4px;max-height:200px;overflow:auto;font-size:0.72rem;line-height:1.4;color:{PALETTE['ink']}"></div>
          </details>
        </div>
      </div>
    </div>
  </div>

  <script>
  (function(){{
    const DATA = {json.dumps(payload)};
    const root = document.getElementById('semn-root');
    const plotEl = root.querySelector('#semn-plot');

    const cameras = {{
      perspective: {{eye: {{x: 1.7, y: 1.55, z: 0.95}}}},
      profile:     {{eye: {{x: 0.18, y: 2.35, z: 0.55}}}},
      opinion:     {{eye: {{x: -0.18, y: -2.35, z: 0.55}}}},
      top:         {{eye: {{x: 0.0, y: 0.14, z: 2.7}}}},
    }};

    function sliderToP(val) {{
      return Math.pow(10, -2 + (parseFloat(val) / 50));
    }}
    function pToSlider(p) {{
      const clamped = Math.max(0.01, Math.min(1, p));
      return Math.round((Math.log10(clamped) + 2) * 50);
    }}
    function sliderToBeta(val) {{
      // 0..100 → 0..maxAbsBeta
      return parseFloat(val) / 100 * DATA.max_abs_beta;
    }}
    function betaToSlider(b) {{
      return Math.round(b / Math.max(DATA.max_abs_beta, 0.01) * 100);
    }}
    function fmtBeta(b) {{ return b.toFixed(2); }}
    function fmtP(p) {{
      if (p >= 0.995) return 'All';
      if (p < 0.02) return p.toFixed(3);
      if (p < 0.1) return p.toFixed(2);
      return p.toFixed(2);
    }}
    function colorForEdge(est, p, maxAbs) {{
      const t = Math.max(0.12, Math.min(1, Math.abs(est) / Math.max(maxAbs, 0.01)));
      const pos0 = [255, 224, 216], pos1 = [192, 57, 43];
      const neg0 = [218, 232, 255], neg1 = [29, 78, 137];
      const base = est >= 0 ? pos0 : neg0;
      const end  = est >= 0 ? pos1 : neg1;
      const r = Math.round(base[0] + (end[0] - base[0]) * t);
      const g = Math.round(base[1] + (end[1] - base[1]) * t);
      const b = Math.round(base[2] + (end[2] - base[2]) * t);
      let a = 0.16;
      if (p <= 0.01) a = 0.98;
      else if (p <= 0.05) a = 0.82;
      else if (p <= 0.10) a = 0.54;
      return `rgba(${{r}},${{g}},${{b}},${{a}})`;
    }}
    function activeButtonValue(containerId, attr) {{
      const btn = root.querySelector(`#${{containerId}} .semn-btn.active`);
      return btn ? btn.dataset[attr] : null;
    }}
    function selectedValues(selector) {{
      return Array.from(root.querySelectorAll(selector)).filter(el => el.checked).map(el => el.value);
    }}
    function labelText(node, density, mode) {{
      if (!node) return '';
      if (node.role === 'group') return node.short || node.label;
      if (density === 'minimal') return mode === 'leaf' || mode === 'all' ? (node.code || '') : '';
      if (density === 'codes') return node.code || '';
      return node.code ? `${{node.code}}  ${{node.short || node.label}}` : (node.short || node.label);
    }}
    function pBadge(p) {{
      if (p <= 0.01) return ['Highly significant', 'hsig'];
      if (p <= 0.05) return ['Significant', 'sig'];
      return ['Exploratory', 'weak'];
    }}
    function applyPreset(mode) {{
      const groupEdges = root.querySelector('#semn-show-group-edges');
      const leafEdges = root.querySelector('#semn-show-leaf-edges');
      const groupNodes = root.querySelector('#semn-show-group-nodes');
      const leafNodes = root.querySelector('#semn-show-leaf-nodes');
      if (!groupEdges || !leafEdges || !groupNodes || !leafNodes) return;
      if (mode === 'overview') {{
        groupEdges.checked = true;
        leafEdges.checked = false;
        groupNodes.checked = true;
        leafNodes.checked = false;
      }} else if (mode === 'leaf') {{
        groupEdges.checked = false;
        leafEdges.checked = true;
        groupNodes.checked = false;
        leafNodes.checked = true;
      }} else {{
        groupEdges.checked = true;
        leafEdges.checked = true;
        groupNodes.checked = true;
        leafNodes.checked = true;
      }}
    }}
    function syncQuickPButtons(pMax) {{
      root.querySelectorAll('.semn-btn[data-p]').forEach(el => el.classList.remove('active'));
      const matches = [
        ['0.01', 0.01],
        ['0.05', 0.05],
        ['0.10', 0.10],
        ['1.00', 1.00],
      ];
      const found = matches.find(([, v]) => Math.abs(v - pMax) < 1e-3);
      if (!found) return;
      const btn = root.querySelector(`.semn-btn[data-p="${{found[0]}}"]`);
      if (btn) btn.classList.add('active');
    }}
    function state() {{
      const bSlider = root.querySelector('#semn-b-slider');
      return {{
        mode: activeButtonValue('semn-view-mode', 'view') || 'overview',
        sign: activeButtonValue('semn-sign-mode', 'sign') || 'all',
        pMax: sliderToP(root.querySelector('#semn-p-slider').value),
        betaMin: bSlider ? sliderToBeta(bSlider.value) : 0,
        searchTerm: (root.querySelector('#semn-search') || {{}}).value || '',
        showGroupEdges: root.querySelector('#semn-show-group-edges').checked,
        showLeafEdges: root.querySelector('#semn-show-leaf-edges').checked,
        showGroupNodes: root.querySelector('#semn-show-group-nodes').checked,
        showLeafNodes: root.querySelector('#semn-show-leaf-nodes').checked,
        includeControls: root.querySelector('#semn-include-controls').checked,
        showAttack: root.querySelector('#semn-show-attack').checked,
        labelDensity: root.querySelector('#semn-label-density').value,
        camera: activeButtonValue('semn-camera-mode', 'camera') || 'perspective',
        profileGroups: selectedValues('.semn-profile-group'),
        opinionGroups: selectedValues('.semn-opinion-group'),
      }};
    }}
    function edgePass(edge, st) {{
      if (edge.p > st.pMax) return false;
      if (edge.abs_est < st.betaMin) return false;
      if (st.sign === 'positive' && edge.estimate <= 0) return false;
      if (st.sign === 'negative' && edge.estimate >= 0) return false;
      if (!st.profileGroups.includes(edge.source_group)) return false;
      if (!st.opinionGroups.includes(edge.target_group)) return false;
      if (!st.includeControls && edge.source_role === 'control') return false;
      if (st.searchTerm) {{
        const q = st.searchTerm.toLowerCase();
        const srcMatch = (edge.source || '').toLowerCase().includes(q) ||
                         (edge.source_code || '').toLowerCase().includes(q);
        const tgtMatch = (edge.target || '').toLowerCase().includes(q) ||
                         (edge.target_code || '').toLowerCase().includes(q);
        if (!srcMatch && !tgtMatch) return false;
      }}
      return true;
    }}
    function nodePass(node, st) {{
      if (!node) return false;
      if (node.family && DATA.profile_groups.includes(node.family)) {{
        if (!st.profileGroups.includes(node.family)) return false;
        if (!st.includeControls && node.role === 'control') return false;
      }}
      if (node.family && DATA.opinion_groups.includes(node.family)) {{
        if (!st.opinionGroups.includes(node.family)) return false;
      }}
      return true;
    }}
    function makeLineTrace(edge, maxAbs, name) {{
      return {{
        type: 'scatter3d',
        mode: 'lines',
        x: edge.x,
        y: edge.y,
        z: edge.z,
        line: {{
          color: colorForEdge(edge.estimate, edge.p, maxAbs),
          width: Math.max(name === 'group' ? 5.5 : 3.0, (Math.abs(edge.estimate) / Math.max(maxAbs, 0.01)) * (name === 'group' ? 13.0 : 9.0)),
        }},
        hovertemplate: name === 'group'
          ? `<b>${{edge.source_group}}</b> → <b>${{edge.target_group}}</b><br>Mean β = ${{edge.estimate.toFixed(3)}}<br>Mean |β| = ${{edge.abs_est.toFixed(3)}}<br>Best p = ${{edge.p.toFixed(4)}}<br>Aggregated paths = ${{edge.n_paths}}<extra>Group summary</extra>`
          : `<b>${{edge.source_code}}</b> ${{edge.source}}<br>→ <b>${{edge.target_code}}</b> ${{edge.target}}<br>β = ${{edge.estimate.toFixed(3)}}<br>p = ${{edge.p.toFixed(4)}}<extra>Leaf path</extra>`,
        showlegend: false,
      }};
    }}
    function makeNodeTrace(nodes, marker, density, mode, hoverTitle) {{
      return {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        z: nodes.map(n => n.z),
        text: nodes.map(n => labelText(n, density, mode)),
        textposition: marker.textposition,
        textfont: marker.textfont,
        marker: marker.marker,
        customdata: nodes.map(n => [n.code || '', n.label, n.family || '', n.role || '']),
        hovertemplate: `<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>Family: %{{customdata[2]}}<br>Type: %{{customdata[3]}}<extra>${{hoverTitle}}</extra>`,
        showlegend: false,
      }};
    }}
    function makeAttackTrace(nodes, density, mode) {{
      return {{
        type: 'scatter3d',
        mode: 'markers+text',
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        z: nodes.map(n => n.z),
        text: nodes.map(n => density === 'minimal' ? n.code : (density === 'codes' ? n.code : `${{n.code}}  ${{n.short}}`)),
        textposition: 'top center',
        textfont: {{size: 9, color: '{PALETTE['orange']}'}},
        marker: {{
          size: nodes.map(n => n.size || 10),
          color: nodes.map(n => n.mean_ae),
          colorscale: 'RdBu_r',
          cmid: 0,
          line: {{color: 'white', width: 1.6}},
          opacity: 0.92,
        }},
        customdata: nodes.map(n => [n.code, n.label, n.mean_ae, n.mean_abs, n.sd_ae, n.n_rows]),
        hovertemplate: `<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>Mean AE = %{{customdata[2]:.2f}}<br>Mean |Δ| = %{{customdata[3]:.2f}}<br>SD AE = %{{customdata[4]:.2f}}<br>Rows = %{{customdata[5]}}<extra>Attack scope</extra>`,
        showlegend: false,
      }};
    }}
    function buildTraces(st) {{
      const traces = [];
      const visibleGroupEdges = DATA.group_edges.filter(edge => edgePass(edge, st));
      const visibleLeafEdges = DATA.leaf_edges.filter(edge => edgePass(edge, st));

      if (st.showAttack) {{
        traces.push({{
          type: 'surface',
          x: [[2.22, 2.22], [2.22, 2.22]],
          y: [[-DATA.y_limit, DATA.y_limit], [-DATA.y_limit, DATA.y_limit]],
          z: [[DATA.z_range[0], DATA.z_range[0]], [DATA.z_range[1], DATA.z_range[1]]],
          showscale: false,
          opacity: 0.12,
          hoverinfo: 'skip',
          colorscale: [[0, 'rgba(240,192,64,0.30)'], [1, 'rgba(231,111,81,0.16)']],
          name: 'Attack scope plane',
        }});
      }}

      if (st.showGroupEdges) visibleGroupEdges.forEach(edge => traces.push(makeLineTrace(edge, DATA.max_abs_beta, 'group')));
      if (st.showLeafEdges) visibleLeafEdges.forEach(edge => traces.push(makeLineTrace(edge, DATA.max_abs_beta, 'leaf')));

      const profileGroupNodes = DATA.profile_group_nodes.filter(n => nodePass(n, st));
      const opinionGroupNodes = DATA.opinion_group_nodes.filter(n => nodePass(n, st));
      const profileLeafNodes = DATA.profile_leaf_nodes.filter(n => nodePass(n, st));
      const opinionLeafNodes = DATA.opinion_leaf_nodes.filter(n => nodePass(n, st));
      const attackGroupNodes = DATA.attack_group_nodes;
      const attackLeafNodes = DATA.attack_leaf_nodes;

      if (st.showGroupNodes) {{
        traces.push(makeNodeTrace(profileGroupNodes, {{
          marker: {{size: 17, color: '#dbe8fb', symbol: 'square', line: {{color: '{PALETTE['navy']}', width: 2}}}},
          textposition: 'middle left',
          textfont: {{size: 10, color: '{PALETTE['navy']}'}},
        }}, 'short', st.mode, 'Profile family'));
        if (st.showAttack && attackGroupNodes.length) {{
          traces.push(makeNodeTrace(attackGroupNodes, {{
            marker: {{size: 15, color: '#f7e4bf', symbol: 'square', line: {{color: '{PALETTE['orange']}', width: 1.8}}}},
            textposition: 'middle center',
            textfont: {{size: 9.5, color: '{PALETTE['orange']}'}},
          }}, 'short', st.mode, 'Attack family'));
        }}
        traces.push(makeNodeTrace(opinionGroupNodes, {{
          marker: {{size: 17, color: '#d8f2ef', symbol: 'square', line: {{color: '{PALETTE['teal']}', width: 2}}}},
          textposition: 'middle right',
          textfont: {{size: 10, color: '{PALETTE['teal']}'}},
        }}, 'short', st.mode, 'Opinion family'));
      }}
      if (st.showLeafNodes) {{
        traces.push(makeNodeTrace(profileLeafNodes, {{
          marker: {{size: 10, color: '{PALETTE['navy']}', symbol: 'circle', line: {{color: 'white', width: 1.5}}}},
          textposition: 'middle left',
          textfont: {{size: 8.5, color: '{PALETTE['ink']}'}},
        }}, st.labelDensity, st.mode, 'Profile moderator'));
        if (st.showAttack && attackLeafNodes.length) traces.push(makeAttackTrace(attackLeafNodes, st.labelDensity, st.mode));
        traces.push(makeNodeTrace(opinionLeafNodes, {{
          marker: {{size: 10, color: '{PALETTE['teal']}', symbol: 'diamond', line: {{color: 'white', width: 1.5}}}},
          textposition: 'middle right',
          textfont: {{size: 8.5, color: '{PALETTE['ink']}'}},
        }}, st.labelDensity, st.mode, 'Opinion indicator'));
      }}
      return {{
        traces,
        visibleGroupEdges,
        visibleLeafEdges,
        profileGroupNodes,
        opinionGroupNodes,
        profileLeafNodes,
        opinionLeafNodes,
        attackGroupNodes,
        attackLeafNodes,
      }};
    }}
    function updateStatus(st, built) {{
      const modeText = st.mode === 'overview' ? 'group summary' : st.mode === 'leaf' ? 'leaf detail' : 'all layers';
      const pText = fmtP(st.pMax);
      const edgeCount = (st.showGroupEdges ? built.visibleGroupEdges.length : 0) + (st.showLeafEdges ? built.visibleLeafEdges.length : 0);
      const controlText = st.includeControls ? 'including model controls' : 'inter-individual differences only';
      const hsig = built.visibleLeafEdges.filter(edge => edge.p <= 0.01).length;
      const sig = built.visibleLeafEdges.filter(edge => edge.p > 0.01 && edge.p <= 0.05).length;
      const exploratory = built.visibleLeafEdges.filter(edge => edge.p > 0.05).length;
      const profileShown = st.showLeafNodes ? built.profileLeafNodes.length : built.profileGroupNodes.length;
      const opinionShown = st.showLeafNodes ? built.opinionLeafNodes.length : built.opinionGroupNodes.length;
      const attackShown = st.showAttack ? (st.showLeafNodes ? built.attackLeafNodes.length : built.attackGroupNodes.length) : 0;
      root.querySelector('#semn-status').innerHTML =
        `<strong>Preset:</strong> ${{modeText}}<br>` +
        `<strong>Visible paths:</strong> ${{edgeCount}} under p ≤ ${{pText}}, ${{st.sign}} sign filter, ${{controlText}}<br>` +
        `<strong>Coverage:</strong> ${{profileShown}} profile-side nodes, ${{opinionShown}} opinion-side nodes` +
        `${{st.showAttack ? `, ${{attackShown}} attack-context nodes` : ''}}<br>` +
        `<strong>Significance mix:</strong> ${{hsig}} highly significant, ${{sig}} significant, ${{exploratory}} exploratory leaf paths<br>` +
        `<strong>Logic:</strong> PROFILE-side moderators shape attacked opinion-shift indicators; attack nodes in the middle show manipulated context for the run, not direct SEM regressors.`;
      root.querySelector('#semn-p-display').textContent = fmtP(st.pMax);
    }}
    function updateLogic(st, built) {{
      const ribbonTxt = st.showGroupEdges ? `${{built.visibleGroupEdges.length}} family ribbons` : 'no family ribbons';
      const leafTxt = st.showLeafEdges ? `${{built.visibleLeafEdges.length}} leaf-level SEM paths` : 'no leaf-level paths';
      const txt = st.mode === 'overview'
        ? `Baseline keeps the moderation story readable first: which PROFILE-side families shape which parts of opinion space under cybermanipulation.`
        : st.mode === 'leaf'
        ? `Leaf detail exposes each SEM coefficient individually. M## codes are profile-side moderators; O## codes are opinion indicators; A## nodes summarize attack context for the run.`
        : `All layers overlays family structure, leaf-level coefficients, and attack context so you can compare the coarse moderation map against exact SEM paths.`;
      root.querySelector('#semn-logic').textContent = `${{txt}} Current selection shows ${{ribbonTxt}} and ${{leafTxt}}.`;
    }}
    function updateFocus(st, visibleGroupEdges, visibleLeafEdges) {{
      const items = []
        .concat(st.showGroupEdges ? visibleGroupEdges.map(item => Object.assign({{_kind: 'group'}}, item)) : [])
        .concat(st.showLeafEdges ? visibleLeafEdges.map(item => Object.assign({{_kind: 'leaf'}}, item)) : [])
        .sort((a,b) => (a.p - b.p) || (Math.abs(b.estimate) - Math.abs(a.estimate)))
        .slice(0, 8);
      const wrap = root.querySelector('#semn-focus');
      if (!items.length) {{
        wrap.innerHTML = `<div class="semn-sub" style="margin:0">No paths remain under the current filters.</div>`;
        return;
      }}
      wrap.innerHTML = items.map(item => {{
        const badge = pBadge(item.p);
        const title = item._kind === 'group'
          ? `${{item.source_group}} → ${{item.target_group}}`
          : `${{item.source_code}} → ${{item.target_code}}`;
        const sub = item._kind === 'group'
          ? `${{item.n_paths}} constituent paths · mean β = ${{item.estimate.toFixed(3)}} · best p = ${{item.p.toFixed(4)}}`
          : `${{item.source}} → ${{item.target}} · β = ${{item.estimate.toFixed(3)}} · p = ${{item.p.toFixed(4)}}`;
        return `
          <div class="semn-item">
            <div class="semn-item-top">
              <strong>${{title}}</strong>
              <span class="semn-badge ${{badge[1]}}">${{badge[0]}}</span>
            </div>
            <div class="semn-item-sub">${{sub}}</div>
          </div>`;
      }}).join('');
    }}
    function updateNodeMap(st, profileLeafNodes, opinionLeafNodes, attackLeafNodes) {{
      const nodes = [];
      if (profileLeafNodes.length) nodes.push('<span><strong>Profile moderators</strong></span>');
      profileLeafNodes.forEach(n => nodes.push(`<span><strong>${{n.code}}</strong> ${{n.short}}</span>`));
      if (opinionLeafNodes.length) nodes.push('<span><strong>Opinion indicators</strong></span>');
      opinionLeafNodes.forEach(n => nodes.push(`<span><strong>${{n.code}}</strong> ${{n.short}}</span>`));
      if (st.showAttack && attackLeafNodes.length) nodes.push('<span><strong>Attack context</strong></span>');
      if (st.showAttack) attackLeafNodes.forEach(n => nodes.push(`<span><strong>${{n.code}}</strong> ${{n.short}}</span>`));
      root.querySelector('#semn-map').innerHTML = nodes.join('');
    }}
    function render() {{
      const st = state();
      const built = buildTraces(st);
      const layout = {{
        paper_bgcolor: 'white',
        margin: {{l: 0, r: 0, t: 16, b: 0}},
        font: {{family: 'IBM Plex Sans, Avenir Next, Segoe UI, sans-serif'}},
        scene: {{
          xaxis: {{visible: false, range: [-0.35, 5.05]}},
          yaxis: {{visible: false, range: [-DATA.y_limit, DATA.y_limit]}},
          zaxis: {{visible: false, range: DATA.z_range}},
          aspectmode: 'manual',
          aspectratio: {{x: 2.45, y: 1.55, z: 1.1}},
          bgcolor: 'white',
          camera: cameras[st.camera],
        }},
        showlegend: false,
      }};
      Plotly.react(plotEl, built.traces, layout, {{displayModeBar: false, responsive: true}});
      syncQuickPButtons(st.pMax);
      updateStatus(st, built);
      updateLogic(st, built);
      updateFocus(st, built.visibleGroupEdges, built.visibleLeafEdges);
      updateNodeMap(st, built.profileLeafNodes, built.opinionLeafNodes, built.attackLeafNodes);
    }}

    root.querySelectorAll('.semn-btn[data-view]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('#semn-view-mode .semn-btn').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      applyPreset(btn.dataset.view);
      render();
    }}));
    root.querySelectorAll('.semn-btn[data-sign]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('#semn-sign-mode .semn-btn').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      render();
    }}));
    root.querySelectorAll('.semn-btn[data-camera]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('#semn-camera-mode .semn-btn').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      render();
    }}));
    root.querySelector('#semn-p-slider').addEventListener('input', () => {{
      const p = sliderToP(root.querySelector('#semn-p-slider').value);
      root.querySelector('#semn-p-display').textContent = fmtP(p);
      syncQuickPButtons(p);
      render();
    }});
    root.querySelectorAll('.semn-btn[data-p]').forEach(btn => btn.addEventListener('click', () => {{
      root.querySelectorAll('.semn-btn[data-p]').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      root.querySelector('#semn-p-slider').value = pToSlider(parseFloat(btn.dataset.p));
      root.querySelector('#semn-p-display').textContent = btn.dataset.p === '1.00' ? 'All' : btn.dataset.p;
      render();
    }}));

    const bSlider = root.querySelector('#semn-b-slider');
    const bDisplay = root.querySelector('#semn-b-display');
    if (bSlider) {{
      bSlider.addEventListener('input', () => {{
        bDisplay.textContent = fmtBeta(sliderToBeta(bSlider.value));
        root.querySelectorAll('.semn-btn[data-b]').forEach(el => el.classList.remove('active'));
        render();
      }});
      root.querySelectorAll('.semn-btn[data-b]').forEach(btn => btn.addEventListener('click', () => {{
        root.querySelectorAll('.semn-btn[data-b]').forEach(el => el.classList.remove('active'));
        btn.classList.add('active');
        bSlider.value = betaToSlider(parseFloat(btn.dataset.b));
        bDisplay.textContent = fmtBeta(parseFloat(btn.dataset.b));
        render();
      }}));
    }}

    const searchEl = root.querySelector('#semn-search');
    if (searchEl) searchEl.addEventListener('input', () => {{
      const q = searchEl.value.toLowerCase().trim();
      const resultsEl = root.querySelector('#semn-search-results');
      if (q.length < 2) {{ resultsEl.innerHTML = ''; render(); return; }}
      const matches = [
        ...DATA.profile_leaf_nodes.map(n => ({{code:n.code,label:n.label,type:'Moderator'}})),
        ...DATA.opinion_leaf_nodes.map(n => ({{code:n.code,label:n.label,type:'Opinion'}})),
      ].filter(n => n.label.toLowerCase().includes(q) || (n.code||'').toLowerCase().includes(q)).slice(0,8);
      resultsEl.innerHTML = matches.map(m => `<span title="${{m.label}}" style="cursor:default"><b>${{m.code}}</b> ${{m.type}}: ${{m.label.substring(0,30)}}</span>`).join('');
      render();
    }});

    const exportBtn = root.querySelector('#semn-export-btn');
    if (exportBtn) exportBtn.addEventListener('click', () => {{
      Plotly.toImage(plotEl, {{format:'png', width:1600, height:1000}}).then(url => {{
        const a = document.createElement('a');
        a.href = url; a.download = 'sem_network.png'; a.click();
      }});
    }});

    // Build full code legend
    const legendFull = root.querySelector('#semn-legend-full');
    if (legendFull) {{
      const mods = DATA.profile_leaf_nodes.map(n => `<div><b>${{n.code}}</b> = ${{n.label}}</div>`);
      const ops = DATA.opinion_leaf_nodes.map(n => `<div><b>${{n.code}}</b> = ${{n.label}}</div>`);
      const atks = DATA.attack_leaf_nodes.map(n => `<div><b>${{n.code}}</b> = ${{n.label}}</div>`);
      legendFull.innerHTML = '<b style="font-size:0.72rem;color:#14213d">Moderators (M)</b>' + mods.join('') +
        '<b style="font-size:0.72rem;color:#1f7a8c;margin-top:8px;display:block">Outcomes (O)</b>' + ops.join('') +
        '<b style="font-size:0.72rem;color:#e76f51;margin-top:8px;display:block">Attacks (A)</b>' + atks.join('');
    }}

    root.querySelectorAll('#semn-show-group-edges,#semn-show-leaf-edges,#semn-show-group-nodes,#semn-show-leaf-nodes,#semn-include-controls,#semn-show-attack,#semn-label-density,.semn-profile-group,.semn-opinion-group')
      .forEach(el => el.addEventListener('change', render));

    const parentPanel = root.closest('.tab-panel');
    if (parentPanel) {{
      const obs = new MutationObserver(() => {{
        if (parentPanel.classList.contains('active')) {{
          setTimeout(() => Plotly.Plots.resize(plotEl), 40);
        }}
      }});
      obs.observe(parentPanel, {{attributes: true, attributeFilter: ['class']}});
    }}
    window.addEventListener('resize', () => Plotly.Plots.resize(plotEl));
    applyPreset('overview');
    render();
  }})();
  </script>
</div>"""


def _fig_sem_heatmap(
    sem_coeff_df: pd.DataFrame,
    exploratory_df: pd.DataFrame,
    long_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    df = sem_coeff_df[sem_coeff_df["op"] == "~"].copy()
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["p_value"]  = pd.to_numeric(df["p_value"],  errors="coerce")
    indicators = df["lhs"].unique().tolist()
    hm = df.pivot_table(index="rhs", columns="lhs", values="estimate", aggfunc="mean")

    if not exploratory_df.empty and "normalized_weight_pct" in exploratory_df.columns:
        order = exploratory_df.sort_values("normalized_weight_pct", ascending=False)["moderator_label"].tolist()
        hm = hm.reindex([r for r in order if r in hm.index])

    hm = hm[[c for c in indicators if c in hm.columns]]

    indicator_labels: Dict[str, str] = {col: _pretty_indicator(str(col)) for col in hm.columns}
    if long_df is not None and not long_df.empty:
        opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
        if opinion_col in long_df.columns:
            opinion_values = sorted(long_df[opinion_col].dropna().unique())
            opinion_display = _unique_display_map(opinion_values)
            for value in opinion_values:
                indicator_labels[_pretty_indicator(str(value))] = opinion_display[value]
    col_labels = [_wrap_label(indicator_labels.get(_pretty_indicator(str(c)), _pretty_indicator(str(c))), 18) for c in hm.columns]

    annot, hover = [], []
    for rhs in hm.index:
        row_a, row_h = [], []
        for lhs in hm.columns:
            sub = df[(df["lhs"] == lhs) & (df["rhs"] == rhs)]
            if sub.empty:
                row_a.append("")
                row_h.append(f"{rhs} → {_leaf(lhs)}<br>No data")
            else:
                r = sub.iloc[0]
                row_a.append(f"{r['estimate']:.2f}{_p_stars(r['p_value'])}")
                row_h.append(f"<b>{rhs}</b> → <b>{_leaf(lhs)}</b><br>"
                             f"β = {r['estimate']:.3f} {_p_stars(r['p_value'])}<br>"
                             f"p = {r['p_value']:.4f}")
        annot.append(row_a)
        hover.append(row_h)

    fig = go.Figure(go.Heatmap(
        z=hm.fillna(0).values, x=col_labels, y=list(hm.index),
        colorscale="RdBu_r", zmid=0,
        text=annot, texttemplate="%{text}", textfont=dict(size=10.5),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="β", thickness=13),
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(420, 38 * len(hm) + 130),
        margin=dict(l=230, r=40, t=52, b=110),
        title=dict(text="SEM Path Coefficients — Moderators → Opinion Indicators (★ p<.05)", font_size=14),
        xaxis=dict(tickangle=-30),
    )
    return fig


def _fig_violin(long_df: pd.DataFrame) -> go.Figure:
    """Violin + strip scatter of AE and |Δ| by opinion leaf."""
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col, abs_col = "adversarial_effectivity", "abs_delta_score"
    if op_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    opinions = sorted(long_df[op_col].dropna().unique())
    opinion_labels = _unique_display_map(opinions)
    colors   = px.colors.qualitative.Bold[:max(len(opinions), 4)]

    fig = make_subplots(1, 2,
                        subplot_titles=["Adversarial Effectivity (AE)",
                                        "Absolute Opinion Shift |Δ|"],
                        horizontal_spacing=0.09)

    for i, (op, clr) in enumerate(zip(opinions, colors)):
        sub = long_df[long_df[op_col] == op]
        lbl = opinion_labels[op]

        for col_idx, ycol in enumerate([ae_col, abs_col], 1):
            if ycol not in sub.columns:
                continue
            y = sub[ycol].dropna().values
            if len(y) == 0:
                continue

            fig.add_trace(go.Violin(
                y=y, x0=lbl, name=lbl, legendgroup=lbl, showlegend=(col_idx == 1),
                fillcolor=clr, line_color=clr, opacity=0.55,
                box=dict(visible=True, width=0.18),
                meanline=dict(visible=True, color="#333"),
                points=False,
                hoverinfo="y+name",
            ), row=1, col=col_idx)

            rng = np.random.RandomState(i * 7 + col_idx)
            jit = rng.uniform(-0.14, 0.14, len(y))
            fig.add_trace(go.Scatter(
                x=[lbl] * len(y), y=y,
                mode="markers",
                marker=dict(color=clr, size=4.5, opacity=0.38,
                            line=dict(color="white", width=0.3)),
                showlegend=False, name=lbl, hoverinfo="y",
            ), row=1, col=col_idx)

    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=520, violinmode="group",
        title=dict(text="Distribution of Outcomes by Opinion Leaf", font_size=14),
        margin=dict(l=60, r=30, t=52, b=90),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(tickangle=-20, tickfont_size=9)
    return fig


def _fig_raw_attack_comparison(long_df: pd.DataFrame) -> go.Figure:
    """Box + strip: AE and |Δ| grouped by attack vector, color = opinion leaf."""
    atk_col = next((c for c in ("attack_execute_tactic", "attack_leaf_label", "attack_leaf") if c in long_df.columns), "attack_leaf")
    op_col  = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col, abs_col = "adversarial_effectivity", "abs_delta_score"
    for c in (atk_col, op_col, ae_col):
        if c not in long_df.columns:
            return go.Figure().add_annotation(text=f"Column '{c}' missing", showarrow=False)

    attacks  = sorted(long_df[atk_col].dropna().unique())
    opinions = sorted(long_df[op_col].dropna().unique())
    atk_labels = _unique_display_map(attacks)
    op_labels  = _unique_display_map(opinions)
    palette    = px.colors.qualitative.Bold

    fig = make_subplots(1, 2,
        subplot_titles=["Adversarial Effectivity by Attack Vector",
                        "Absolute Opinion Shift |Δ| by Attack Vector"],
        horizontal_spacing=0.14)

    for oi, op in enumerate(opinions):
        sub = long_df[long_df[op_col] == op]
        clr = palette[oi % len(palette)]
        lbl = op_labels[op]
        for ci, ycol in enumerate([ae_col, abs_col], 1):
            if ycol not in sub.columns:
                continue
            x_vals = [_wrap_label(atk_labels[a], 22) for a in attacks]
            y_boxes = [sub[sub[atk_col] == a][ycol].dropna().values for a in attacks]
            for xi, (xv, yv) in enumerate(zip(x_vals, y_boxes)):
                if len(yv) == 0:
                    continue
                fig.add_trace(go.Box(
                    x=[xv] * len(yv), y=yv,
                    name=lbl, legendgroup=lbl,
                    showlegend=(ci == 1 and xi == 0),
                    marker_color=clr, line_color=clr,
                    boxmean="sd", whiskerwidth=0.5,
                    width=0.14, opacity=0.78,
                    hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>{ycol}: %{{y:.1f}}<extra></extra>",
                ), row=1, col=ci)

    # mean AE per attack vector as diamond markers
    atk_means = long_df.groupby(atk_col)[ae_col].mean()
    fig.add_trace(go.Scatter(
        x=[_wrap_label(atk_labels[a], 22) for a in attacks],
        y=[atk_means.get(a, 0) for a in attacks],
        mode="markers", marker=dict(symbol="diamond", size=11,
            color=PALETTE["navy"], line=dict(color="white", width=1.2)),
        name="Overall mean AE", showlegend=True,
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1, row=1, col=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=540, boxmode="group",
        title=dict(text="Raw Outcome Distributions by Attack Vector", font_size=14),
        margin=dict(l=65, r=30, t=55, b=110),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font_size=9),
    )
    fig.update_xaxes(tickangle=-20, tickfont_size=9)
    return fig


def _fig_raw_score_scatter(long_df: pd.DataFrame) -> go.Figure:
    """Baseline vs. post scatter with regression trend and AE density marginals."""
    if not {"pre_score", "post_score"}.issubset(long_df.columns):
        # Try alternate column names
        pre_col  = next((c for c in long_df.columns if "baseline" in c.lower() or "pre_score" in c.lower()), None)
        post_col = next((c for c in long_df.columns if "post_score" in c.lower() or "attacked" in c.lower()), None)
        if not pre_col or not post_col:
            return go.Figure().add_annotation(text="Pre/post score columns unavailable", showarrow=False)
    else:
        pre_col, post_col = "pre_score", "post_score"

    ae_col = "adversarial_effectivity"
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    df = long_df[[pre_col, post_col, op_col]].dropna()
    if ae_col in long_df.columns:
        df = long_df[[pre_col, post_col, op_col, ae_col]].dropna()

    opinions = sorted(df[op_col].dropna().unique())
    op_labels = _unique_display_map(opinions)
    palette   = px.colors.qualitative.Bold

    fig = go.Figure()
    # Identity line
    rng = [float(df[pre_col].min() - 2), float(df[pre_col].max() + 2)]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines",
        line=dict(color="#aab4c8", width=1.5, dash="dot"),
        name="No change (y = x)", showlegend=True,
    ))

    for oi, op in enumerate(opinions):
        sub = df[df[op_col] == op]
        clr = palette[oi % len(palette)]
        lbl = op_labels[op]
        color_vals = sub[ae_col].values if ae_col in sub.columns else None

        scatter_kw = dict(
            x=sub[pre_col].values, y=sub[post_col].values,
            mode="markers",
            marker=dict(size=5, opacity=0.55, line=dict(color="white", width=0.3),
                        color=(color_vals if color_vals is not None else clr),
                        colorscale="RdBu_r" if color_vals is not None else None,
                        cmid=0 if color_vals is not None else None,
                        showscale=False),
            name=lbl, legendgroup=lbl, showlegend=True,
            hovertemplate=f"<b>{lbl}</b><br>Pre: %{{x:.1f}}<br>Post: %{{y:.1f}}<extra></extra>",
        )
        fig.add_trace(go.Scatter(**scatter_kw))

        # Regression line per opinion
        if len(sub) >= 4:
            x_np = sub[pre_col].values
            y_np = sub[post_col].values
            try:
                m, b_intercept = np.polyfit(x_np, y_np, 1)
                xs = np.array([x_np.min(), x_np.max()])
                fig.add_trace(go.Scatter(
                    x=xs, y=m * xs + b_intercept, mode="lines",
                    line=dict(color=clr, width=1.8, dash="solid"),
                    name=f"{lbl} trend", legendgroup=lbl, showlegend=False,
                ))
            except Exception:
                pass

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=540,
        xaxis_title="Pre-attack Opinion Score",
        yaxis_title="Post-attack Opinion Score",
        title=dict(text="Pre vs. Post Attack Scores — Colour = AE direction", font_size=14),
        margin=dict(l=70, r=30, t=55, b=65),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", font_size=9),
    )
    return fig


def _fig_susceptibility_scatter(profile_index_df: pd.DataFrame,
                                long_df: pd.DataFrame) -> go.Figure:
    if profile_index_df.empty:
        return go.Figure().add_annotation(text="Profile index unavailable", showarrow=False)

    agg = long_df.groupby("profile_id").agg(
        mean_ae=("adversarial_effectivity", "mean"),
        mean_abs=("abs_delta_score", "mean"),
    ).reset_index()
    merged = profile_index_df.merge(agg, on="profile_id", how="left").dropna(
        subset=["mean_ae", "mean_abs", "susceptibility_index_pct"])

    fig = go.Figure(go.Scatter(
        x=merged["mean_abs"], y=merged["mean_ae"],
        mode="markers",
        marker=dict(
            size=merged["susceptibility_index_pct"].fillna(50) / 6 + 7,
            color=merged["susceptibility_index_pct"],
            colorscale="RdBu_r", cmin=0, cmax=100,
            opacity=0.82, line=dict(color="white", width=0.8),
            colorbar=dict(title="Susceptibility<br>Index (%)", thickness=13),
            showscale=True,
        ),
        text=merged["profile_id"],
        customdata=np.column_stack([merged["susceptibility_index_pct"],
                                    merged["mean_ae"], merged["mean_abs"]]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Susceptibility: %{customdata[0]:.0f}th pct<br>"
            "Mean AE: %{customdata[1]:.1f}<br>"
            "Mean |Δ|: %{customdata[2]:.1f}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#666", line_width=1,
                  annotation_text="AE = 0 (no net manipulation)", annotation_font_size=9)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=520,
        title=dict(text="Profile Susceptibility Map — size & color = susceptibility index", font_size=14),
        xaxis_title="Mean Absolute Opinion Shift",
        yaxis_title="Mean Adversarial Effectivity",
        margin=dict(l=70, r=30, t=52, b=60),
    )
    return fig


def _fig_moderator_forest(exploratory_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    if exploratory_df.empty:
        return go.Figure().add_annotation(text="Moderator data unavailable", showarrow=False)

    df = exploratory_df.copy()
    for c in ["multivariate_estimate", "univariate_estimate", "multivariate_p_value",
               "multivariate_conf_low", "multivariate_conf_high",
               "univariate_conf_low", "univariate_conf_high", "multivariate_q_value",
               "elastic_net_estimate", "rf_permutation_importance", "ridge_estimate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prefer ridge (all-feature model) → fall back to elastic_net → OLS for sorting/display
    has_ridge = "ridge_estimate"       in df.columns and df["ridge_estimate"].notna().any()
    has_enet  = "elastic_net_estimate" in df.columns and df["elastic_net_estimate"].notna().any()
    has_mv    = "multivariate_estimate" in df.columns and df["multivariate_estimate"].notna().any()

    sort_col = "ridge_estimate" if has_ridge else ("elastic_net_estimate" if has_enet else "multivariate_estimate")

    # Keep top_n features by absolute sort value; sort ascending so largest is at top of plot
    df = (df.assign(_abs_sort=df[sort_col].abs())
            .sort_values("_abs_sort", ascending=False)
            .head(top_n)
            .sort_values("_abs_sort", ascending=True)   # ascending=True → largest at top in horizontal bar
            .drop(columns=["_abs_sort"])
            .reset_index(drop=True))

    has_uni   = "univariate_estimate"   in df.columns and df["univariate_estimate"].notna().any()
    has_q     = "multivariate_q_value"  in df.columns
    has_ci_mv = ("multivariate_conf_high" in df.columns and "multivariate_conf_low" in df.columns
                 and df["multivariate_conf_high"].notna().any())
    has_ci_uv = ("univariate_conf_high" in df.columns and "univariate_conf_low" in df.columns
                 and df["univariate_conf_high"].notna().any())

    label_col = "moderator_label" if "moderator_label" in df.columns else "label"
    labels = df[label_col].fillna(df.get("term", "")).tolist()

    fig = go.Figure()

    # ── Ridge (all features) — primary estimator ──────────────────────
    if has_ridge:
        ridge_colors = [PALETTE["red"] if (not pd.isna(v) and v < 0) else PALETTE["teal"]
                        for v in df["ridge_estimate"].tolist()]
        # Group colour by ontology_group for additional visual coding
        fig.add_trace(go.Scatter(
            x=df["ridge_estimate"], y=labels, mode="markers",
            name="Ridge — all features (primary)",
            marker=dict(color=ridge_colors, size=13, symbol="square",
                        line=dict(color="white", width=1)),
            hovertemplate="%{y}<br><b>Ridge coef=%{x:.4f}</b>  (all features, std-scaled)<extra></extra>",
        ))

    # ── OLS Multivariate (Big Five benchmark) ─────────────────────────
    if has_mv:
        mv_vals = df["multivariate_estimate"].tolist()
        mv_colors = [PALETTE["red"] if (not pd.isna(v) and v < 0) else PALETTE["blue"]
                     for v in mv_vals]
        hov_extra = (df[["multivariate_p_value", "multivariate_q_value"]].values
                     if has_q else df[["multivariate_p_value"]].values if "multivariate_p_value" in df.columns
                     else np.full((len(df), 1), np.nan))
        hov_tmpl  = ("%{y}<br>OLS β=%{x:.3f}<br>p=%{customdata[0]:.4f}<br>q=%{customdata[1]:.4f}<extra></extra>"
                     if has_q else "%{y}<br>OLS β=%{x:.3f}<extra></extra>")
        fig.add_trace(go.Scatter(
            x=df["multivariate_estimate"], y=labels, mode="markers",
            name="OLS multivariate (Big Five benchmark)",
            marker=dict(color=mv_colors, size=10, symbol="diamond",
                        line=dict(color="white", width=1.2)),
            error_x=dict(
                type="data",
                array=(df["multivariate_conf_high"] - df["multivariate_estimate"]).abs() if has_ci_mv else None,
                arrayminus=(df["multivariate_estimate"] - df["multivariate_conf_low"]).abs() if has_ci_mv else None,
                color="#aaa", thickness=1.5, width=5, visible=True,
            ),
            customdata=hov_extra,
            hovertemplate=hov_tmpl,
        ))

    # ── OLS Univariate (for Big Five rows) ────────────────────────────
    if has_uni:
        uv_mask = df["univariate_estimate"].notna()
        uv_df = df[uv_mask]
        uv_labs = [labels[i] for i in uv_df.index.tolist()]
        if not uv_df.empty:
            fig.add_trace(go.Scatter(
                x=uv_df["univariate_estimate"], y=uv_labs, mode="markers",
                name="OLS univariate (Big Five)",
                marker=dict(color="#9aaac8", size=8, symbol="circle-open", line=dict(width=1.5)),
                error_x=dict(
                    type="data",
                    array=(uv_df["univariate_conf_high"] - uv_df["univariate_estimate"]).abs() if has_ci_uv else None,
                    arrayminus=(uv_df["univariate_estimate"] - uv_df["univariate_conf_low"]).abs() if has_ci_uv else None,
                    color="#9aaac8", thickness=1, width=3, visible=True,
                ),
                hovertemplate="%{y}<br>OLS univariate β=%{x:.3f}<extra></extra>",
            ))

    # ── FDR significance stars ─────────────────────────────────────────
    if has_q and "multivariate_q_value" in df.columns:
        sig = df[df["multivariate_q_value"] < 0.05]
        if not sig.empty:
            sig_labs = [labels[i] for i in sig.index.tolist()]
            fig.add_trace(go.Scatter(
                x=sig["multivariate_estimate"], y=sig_labs,
                mode="markers", name="FDR q<.05",
                marker=dict(color=PALETTE["gold"], size=18, symbol="star",
                            line=dict(color="#d95d39", width=1.5)),
                hovertemplate="%{y}<br>FDR q<.05<extra></extra>",
            ))

    # ── Ontology group colour legend (text annotations at right edge) ──
    if "ontology_group" in df.columns:
        group_col = {
            "Big Five: Neuroticism": "#c0392b", "Big Five: Extraversion": "#1d4e89",
            "Big Five: Openness": "#2980b9",    "Big Five: Conscientiousness": "#27ae60",
            "Big Five: Agreeableness": "#8e44ad",
            "Dual Process": "#e67e22",           "Digital Literacy": "#16a085",
            "Political Engagement: Institutional Trust": "#922b21",
            "Political Engagement: Ideology": "#1a5276",
            "Political Engagement: Interest": "#117a65",
            "Political Engagement: Efficacy": "#7d6608",
            "Political Psychology: Institutional Trust": "#922b21",
            "Political Psychology: Ideology": "#1a5276",
            "Political Psychology: Engagement": "#117a65",
            "Socioeconomic Status: Economic": "#5d4037",
            "Socioeconomic Status: Employment": "#795548",
            "Social Context: Online Behavior": "#0277bd",
            "Social Context: Social Capital": "#00838f",
            "Demographics: Age": "#607d8b", "Demographics: Sex": "#90a4ae",
            "Demographics: Education": "#78909c", "Demographics: News Diet": "#b0bec5",
        }

    note = (
        "<b>■ Ridge (all modeled profile features, std-scaled)</b> = primary estimator — retains all survey-mappable predictors in the current run.<br>"
        "<b>◇ OLS</b> = Big Five benchmark (8 predictors). "
        "Near-zero |coef| should be read alongside the execution-integrity diagnostics and the between-profile variance partitioning."
    )

    fig.add_vline(x=0, line_dash="dot", line_color="#777", line_width=1)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(520, 26 * len(df) + 170),
        title=dict(
            text=f"Moderator Coefficients — ■ Ridge all-feature (primary) · ◇ OLS Big Five (benchmark) · top {top_n} by |coef|",
            font_size=12),
        xaxis_title="Standardised coefficient estimate",
        margin=dict(l=280, r=30, t=65, b=80),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        annotations=[dict(
            x=0, y=-0.12, xref="paper", yref="paper",
            text=note, showarrow=False,
            font=dict(size=9, color="#555"),
            align="center", xanchor="center",
        )],
    )
    return fig


def _fig_hierarchical_importance(weight_df: pd.DataFrame) -> go.Figure:
    if weight_df.empty:
        return go.Figure().add_annotation(text="Weight data unavailable", showarrow=False)

    df = weight_df.copy()
    df["normalized_weight_pct"] = pd.to_numeric(df["normalized_weight_pct"], errors="coerce").fillna(0)
    if "estimate" in df.columns:
        df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    if "mean_abs_estimate" in df.columns:
        df["mean_abs_estimate"] = pd.to_numeric(df["mean_abs_estimate"], errors="coerce")
    df_pos = df[df["normalized_weight_pct"] > 0].copy()
    if df_pos.empty:
        return go.Figure().add_annotation(text="All weights zero", showarrow=False)

    path_rows: List[Tuple[List[str], float]] = []
    for _, row in df_pos.iterrows():
        label = str(row.get("moderator_label") or row.get("term") or "Unnamed moderator")
        ontology_group = None if pd.isna(row.get("ontology_group")) else str(row.get("ontology_group"))
        path_rows.append((_moderator_hierarchy(label, ontology_group), float(row["normalized_weight_pct"])))

    ids, labels, parents, values, paths = _build_tree_nodes(path_rows)

    top_n = min(18, len(df_pos))
    df_top = df_pos.sort_values("normalized_weight_pct", ascending=False).head(top_n).iloc[::-1]
    estimate_col = "estimate" if "estimate" in df_top.columns else "weighted_mean_estimate"
    if estimate_col in df_top.columns:
        df_top[estimate_col] = pd.to_numeric(df_top[estimate_col], errors="coerce").fillna(0)
    else:
        df_top[estimate_col] = 0.0
    df_top["leaf_label"] = [
        _clip_label(_moderator_hierarchy(
            str(row.get("moderator_label") or row.get("term") or "Unnamed moderator"),
            None if pd.isna(row.get("ontology_group")) else str(row.get("ontology_group")),
        )[-1], 34)
        for _, row in df_top.iterrows()
    ]

    fig = make_subplots(
        1,
        2,
        column_widths=[0.40, 0.60],
        horizontal_spacing=0.20,
        subplot_titles=["Ontology hierarchy", f"Top leaf moderators ({top_n} of {len(df_pos)})"],
        specs=[[{"type": "treemap"}, {"type": "xy"}]],
    )

    fig.add_trace(go.Treemap(
        ids=ids,
        labels=[_wrap_label(label, 16) for label in labels],
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+value",
        marker=dict(
            colors=values,
            colorscale="Blues",
            line=dict(width=1.6, color="white"),
        ),
        tiling=dict(pad=4),
        customdata=np.array(paths, dtype=object),
        hovertemplate="%{customdata}<br>Weight: %{value:.1f}%<extra></extra>",
        root_color="#eef4ff",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_top["normalized_weight_pct"],
        y=df_top["leaf_label"],
        orientation="h",
        marker=dict(
            color=df_top[estimate_col],
            colorscale="RdBu_r",
            cmid=0,
            line=dict(color="white", width=0.6),
            colorbar=dict(title="Signed β", thickness=12),
        ),
        text=[f"{v:.1f}%" for v in df_top["normalized_weight_pct"]],
        textposition="outside",
        cliponaxis=False,
        customdata=np.column_stack([
            df_top["moderator_label"].astype(str),
            df_top[estimate_col].astype(float),
            df_top["normalized_weight_pct"].astype(float),
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Weight: %{customdata[2]:.1f}%<br>"
            "Signed β: %{customdata[1]:.3f}<extra></extra>"
        ),
    ), row=1, col=2)

    fig.update_xaxes(
        title="Normalized importance share (%)",
        row=1,
        col=2,
        automargin=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=9.2),
        row=1,
        col=2,
        automargin=True,
    )
    fig.update_annotations(font=dict(size=12.5))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(620, 30 * top_n + 220),
        showlegend=False,
        title=dict(text="Hierarchical Feature Importance — Conditional Susceptibility Model", font_size=14),
        margin=dict(l=50, r=110, t=78, b=54),
        bargap=0.22,
    )
    return fig


def _fig_profile_heatmap(long_df: pd.DataFrame, profile_index_df: pd.DataFrame) -> go.Figure:
    op_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    ae_col = "adversarial_effectivity"
    if op_col not in long_df.columns or ae_col not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    matrix = long_df.pivot_table(index="profile_id", columns=op_col,
                                  values=ae_col, aggfunc="mean")
    if not profile_index_df.empty and "susceptibility_index_pct" in profile_index_df.columns:
        order = (profile_index_df.sort_values("susceptibility_index_pct", ascending=False)
                 ["profile_id"].tolist())
        matrix = matrix.reindex([p for p in order if p in matrix.index])

    opinion_labels = _unique_display_map(list(matrix.columns))
    col_labels = [_wrap_label(opinion_labels[c], 18) for c in matrix.columns]
    n = len(matrix)
    fig = go.Figure(go.Heatmap(
        z=matrix.fillna(0).values, x=col_labels, y=matrix.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Mean AE", thickness=13),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=max(500, 9 * n + 180),
        title=dict(text="Per-profile AE Heatmap — Sorted by Susceptibility Index (↓ most susceptible)",
                   font_size=14),
        xaxis_title="Opinion leaf",
        yaxis=dict(title="Profile ID", tickfont_size=7.5),
        margin=dict(l=95, r=30, t=52, b=80),
    )
    return fig


def _fig_baseline_post(long_df: pd.DataFrame) -> go.Figure:
    if "baseline_score" not in long_df.columns:
        return go.Figure().add_annotation(text="Data unavailable", showarrow=False)

    has_ae = "adversarial_effectivity" in long_df.columns
    opinion_col = "opinion_leaf" if "opinion_leaf" in long_df.columns else "opinion_leaf_label"
    has_leaf = opinion_col in long_df.columns
    opinion_text = None
    if has_leaf:
        vals = long_df[opinion_col].astype(str).tolist()
        display_map = _unique_display_map(sorted(set(vals)))
        opinion_text = [display_map.get(v, _leaf(v)) for v in vals]

    fig = go.Figure(go.Scatter(
        x=long_df["baseline_score"],
        y=long_df["post_score"],
        mode="markers",
        marker=dict(
            size=5.5, opacity=0.55,
            color=long_df["adversarial_effectivity"].values if has_ae else PALETTE["blue"],
            colorscale="RdBu_r" if has_ae else None,
            cmid=0 if has_ae else None,
            line=dict(color="rgba(255,255,255,0.25)", width=0.3),
            colorbar=dict(title="AE", thickness=12) if has_ae else None,
            showscale=has_ae,
        ),
        text=opinion_text if has_leaf else None,
        hovertemplate="Baseline: %{x}<br>Post: %{y}<br>%{text}<extra></extra>",
    ))

    lo = float(min(long_df["baseline_score"].min(), long_df["post_score"].min())) - 30
    hi = float(max(long_df["baseline_score"].max(), long_df["post_score"].max())) + 30
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#666", dash="dash", width=1.2),
                             name="No change", showlegend=True))
    fig.add_annotation(x=lo + 60, y=hi - 30, text="Opinion moved up",
                       font=dict(size=9, color=PALETTE["teal"]), showarrow=False)
    fig.add_annotation(x=hi - 60, y=lo + 30, text="Opinion moved down",
                       font=dict(size=9, color=PALETTE["orange"]), showarrow=False)
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        height=520,
        title=dict(text="Baseline vs Post-attack Opinion — Coloured by Adversarial Effectivity", font_size=14),
        xaxis_title="Baseline opinion score",
        yaxis_title="Post-attack opinion score",
        margin=dict(l=70, r=30, t=52, b=60),
    )
    return fig


# ─── perturbation explorer (simple) ──────────────────────────────────────────

def _html_perturbation_explorer(task_coeff_df: pd.DataFrame,
                                long_df: pd.DataFrame) -> str:
    if task_coeff_df.empty:
        return "<p>No coefficient data available.</p>"

    SLIDERS = [
        ("profile_cont_big_five_agreeableness_mean_pct",         "Agreeableness",    50, 0, 100),
        ("profile_cont_big_five_conscientiousness_mean_pct",     "Conscientiousness", 50, 0, 100),
        ("profile_cont_big_five_extraversion_mean_pct",          "Extraversion",      50, 0, 100),
        ("profile_cont_big_five_neuroticism_mean_pct",           "Neuroticism",       50, 0, 100),
        ("profile_cont_big_five_openness_to_experience_mean_pct","Openness",          50, 0, 100),
        ("profile_cont_chronological_age",                       "Age",               40, 18, 80),
    ]

    attack_values = sorted(task_coeff_df["attack_leaf"].dropna().unique())
    opinion_values = sorted(task_coeff_df["opinion_leaf"].dropna().unique())
    attack_labels = _unique_display_map(attack_values)
    # The conditional-susceptibility index deliberately pools attacks per opinion
    # so each task is well populated for PROFILE-trait moderation; make that
    # intent explicit rather than showing a raw pooled slug.
    if "DISARM_attacks_pooled" in attack_labels:
        attack_labels["DISARM_attacks_pooled"] = "All DISARM attacks (pooled for profile moderation)"
    opinion_labels = _unique_display_map(opinion_values)

    tasks_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (ak, ok), grp in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        tasks_json.setdefault(str(ak), {})[str(ok)] = dict(
            zip(grp["term"].tolist(), grp["estimate"].astype(float).tolist())
        )

    slider_defs = json.dumps([{"term": t, "label": lbl, "default": d, "min": mn, "max": mx}
                               for t, lbl, d, mn, mx in SLIDERS])

    return f"""
<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;padding:4px 0">
<div style="min-width:250px;flex:0 0 265px">
  <h3 style="margin:0 0 10px;font-size:0.95rem;color:{PALETTE['ink']}">Profile Sliders</h3>
  <div id="pe-sliders" style="display:flex;flex-direction:column;gap:8px;"></div>
  <label style="display:block;margin-top:12px;font-size:0.82rem;font-weight:600;">Sex
    <select id="pe-sex" style="width:100%;margin-top:3px;padding:5px;border-radius:6px;border:1px solid #dbe3ef">
      <option>Male</option><option>Female</option><option>Other</option>
    </select></label>
  <button onclick="pe_reset()" style="margin-top:12px;padding:6px 14px;background:{PALETTE['blue']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.85rem">Reset</button>
</div>
<div style="flex:1;min-width:320px">
  <h3 style="margin:0 0 4px;font-size:0.95rem;color:{PALETTE['ink']}">Predicted AE Grid</h3>
  <div style="font-size:0.78rem;color:{PALETTE['muted']};margin-bottom:8px">Red = manipulation succeeded · Blue = resistance/backfire</div>
  <div id="pe-grid"></div>
</div>
</div>
<script>
(function(){{
const TASKS={json.dumps(tasks_json)};
const ATTACKS={json.dumps(attack_values)};
const OPINIONS={json.dumps(opinion_values)};
const ATTACK_LABELS={json.dumps(attack_labels)};
const OPINION_LABELS={json.dumps(opinion_labels)};
const SL={slider_defs};
function vals(){{
  const v={{'Intercept':1}};
  SL.forEach(s=>v[s.term]=parseFloat(document.getElementById('pe-sl-'+s.term).value));
  const sx=document.getElementById('pe-sex').value;
  v['profile_cat__profile_cat_sex_Female']=sx==='Female'?1:0;
  v['profile_cat__profile_cat_sex_Male']=sx==='Male'?1:0;
  v['profile_cat__profile_cat_sex_Other']=sx==='Other'?1:0;
  return v;
}}
function cellBg(ae){{
  const t=Math.max(-1,Math.min(1,ae/60));
  if(t>=0)return `rgb(${{Math.round(190+65*t)}},${{Math.round(60-40*t)}},${{Math.round(60-40*t)}})`;
  const u=-t;return `rgb(${{Math.round(60-40*u)}},${{Math.round(60+60*u)}},${{Math.round(190+65*u)}})`;
}}
function upd(){{
  const v=vals();
  let h='<table style="border-collapse:collapse;font-size:0.82rem">';
  h+='<tr><th style="padding:5px 8px;border-bottom:2px solid #dbe3ef;text-align:left;font-size:0.76rem">Attack \\ Opinion</th>';
  OPINIONS.forEach(o=>h+=`<th title="${{o}}" style="padding:5px 7px;border-bottom:2px solid #dbe3ef;font-size:0.76rem;min-width:80px">${{OPINION_LABELS[o]}}</th>`);
  h+='</tr>';
  ATTACKS.forEach(a=>{{
    h+=`<tr><td title="${{a}}" style="padding:5px 8px;font-weight:600;border-right:1px solid #dbe3ef;font-size:0.78rem;white-space:nowrap">${{ATTACK_LABELS[a]}}</td>`;
    OPINIONS.forEach(o=>{{
      const c=(TASKS[a]||{{}})[o]||{{}};
      let ae=0;Object.entries(c).forEach(([t,e])=>ae+=e*(v[t]||0));
      const bg=cellBg(ae),tc=Math.abs(ae)>28?'#fff':'#14213d';
      h+=`<td style="text-align:center;padding:7px;background:${{bg}};color:${{tc}};border:2px solid rgba(255,255,255,0.4);border-radius:4px;font-weight:700;font-size:0.9rem">${{ae.toFixed(1)}}</td>`;
    }});h+='</tr>';
  }});
  document.getElementById('pe-grid').innerHTML=h+'</table>';
}}
function build(){{
  const c=document.getElementById('pe-sliders');
  SL.forEach(s=>{{
    const d=document.createElement('div');
    d.innerHTML=`<label style="font-size:0.81rem;font-weight:600;color:{PALETTE['ink']};display:flex;justify-content:space-between">
      <span>${{s.label}}</span><span id="pe-v-${{s.term}}">${{s.default}}</span></label>
      <input type="range" id="pe-sl-${{s.term}}" min="${{s.min}}" max="${{s.max}}" value="${{s.default}}"
        style="width:100%;accent-color:{PALETTE['blue']}"
        oninput="document.getElementById('pe-v-${{s.term}}').textContent=this.value;peUpd()">`;
    c.appendChild(d);
  }});
  document.getElementById('pe-sex').onchange=()=>peUpd();
}}
window.peUpd=upd;window.pe_reset=function(){{SL.forEach(s=>{{document.getElementById('pe-sl-'+s.term).value=s.default;document.getElementById('pe-v-'+s.term).textContent=s.default;}});document.getElementById('pe-sex').value='Male';upd();}};
build();upd();
}})();
</script>"""


# ─── conditional susceptibility estimator ────────────────────────────────────

def _html_cs_estimator(
    task_coeff_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> str:
    """
    Full conditional susceptibility estimation tool embedded in the dashboard.
    Features:
    - Profile builder: Big Five (mean + collapsible facets), age, sex, heuristic/resilience
    - Task selector: any subset of attack × opinion combinations
    - Live AE prediction grid (attack × opinion, RdBu colour)
    - Susceptibility gauge (percentile vs re-computed distribution on selected tasks)
    - Radar chart: utils profile vs population average (Plotly)
    - Feature contribution waterfall: top positive & negative drivers
    - Load random profile · Reset · Quick presets
    """
    if task_coeff_df.empty:
        return "<p>Conditional susceptibility data unavailable.</p>"

    feat_cols = sorted([
        c for c in long_df.columns
        if (c.startswith("profile_cont_") or c.startswith("profile_cat__"))
        and c != "profile_cont_heuristic_shift_sensitivity_proxy"
    ])

    profile_feats_df = long_df.groupby("profile_id")[feat_cols].first().reset_index()
    feat_means = {c: float(profile_feats_df[c].mean()) for c in feat_cols}
    feat_sds = {c: float(profile_feats_df[c].std(ddof=0)) for c in feat_cols}

    profiles_json: Dict[str, Dict[str, float]] = {}
    for _, row in profile_feats_df.iterrows():
        profiles_json[str(row["profile_id"])] = {
            c: float(row[c]) for c in feat_cols if not pd.isna(row[c])
        }

    # observed per-profile per-task mean AE (powers the k-NN estimator) and
    # profile-free cell baselines (context-only reference estimator)
    obs_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    cell_means_json: Dict[str, Dict[str, float]] = {}
    if "adversarial_effectivity" in long_df.columns:
        obs_grp = (
            long_df.dropna(subset=["adversarial_effectivity"])
            .groupby(["profile_id", "attack_leaf", "opinion_leaf"])["adversarial_effectivity"]
            .mean()
        )
        for (pid, ak, ok), val in obs_grp.items():
            obs_json.setdefault(str(pid), {}).setdefault(str(ak), {})[str(ok)] = round(float(val), 3)
        cell_grp = (
            long_df.dropna(subset=["adversarial_effectivity"])
            .groupby(["attack_leaf", "opinion_leaf"])["adversarial_effectivity"]
            .mean()
        )
        for (ak, ok), val in cell_grp.items():
            cell_means_json.setdefault(str(ak), {})[str(ok)] = round(float(val), 3)

    all_attacks = sorted(task_coeff_df["attack_leaf"].dropna().unique())
    all_opinions = sorted(task_coeff_df["opinion_leaf"].dropna().unique())
    attack_labels = _unique_display_map(all_attacks)
    # Pooling is intentional here (profile-trait moderation per opinion); label it.
    if "DISARM_attacks_pooled" in attack_labels:
        attack_labels["DISARM_attacks_pooled"] = "All DISARM attacks (pooled for profile moderation)"
    opinion_labels = _unique_display_map(all_opinions)
    attack_context = {value: _path_context(value, keep=1) for value in all_attacks}
    opinion_context = {value: _path_context(value, keep=1) for value in all_opinions}

    tasks_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    weights_json: Dict[str, Dict[str, float]] = {}
    for (ak, ok), grp in task_coeff_df.groupby(["attack_leaf", "opinion_leaf"]):
        tasks_json.setdefault(str(ak), {})[str(ok)] = dict(
            zip(grp["term"].tolist(), grp["estimate"].astype(float).tolist())
        )
        weights_json.setdefault(str(ak), {})[str(ok)] = 1.0

    if not task_summary_df.empty:
        for _, row in task_summary_df.iterrows():
            ak = str(row["attack_leaf"])
            ok = str(row["opinion_leaf"])
            weights_json.setdefault(ak, {})[ok] = float(row.get("reliability_weight", 1.0))

    # Big Five structure for grouped sliders
    BIG5_GROUPS = [
        ("agreeableness", "Agreeableness", [
            "altruism", "compliance", "modesty", "straightforwardness",
            "tender_mindedness", "trust",
        ]),
        ("conscientiousness", "Conscientiousness", [
            "achievement_striving", "competence", "deliberation",
            "dutifulness", "order", "self_discipline",
        ]),
        ("extraversion", "Extraversion", [
            "activity_level", "assertiveness", "excitement_seeking",
            "gregariousness", "positive_emotions", "warmth",
        ]),
        ("neuroticism", "Neuroticism", [
            "anger_hostility", "anxiety", "depression",
            "impulsiveness", "self_consciousness", "stress_vulnerability",
        ]),
        ("openness_to_experience", "Openness to Experience", [
            "actions", "aesthetics", "fantasy",
            "feelings", "ideas", "values",
        ]),
    ]

    radar_labels = [g[1] for g in BIG5_GROUPS]
    radar_means  = [round(feat_means.get(
        f"profile_cont_big_five_{g[0]}_mean_pct", 50.0), 1) for g in BIG5_GROUPS]

    big5_group_blocks: List[str] = []
    for group_key, group_label, facets in BIG5_GROUPS:
        facet_blocks = "".join(
            f"""
        <div style="margin-bottom:5px">
          <div style="display:flex;justify-content:space-between;font-size:0.76rem;color:{PALETTE['muted']}">
            <span>{facet.replace("_", " ").title()}</span>
            <span id="cse-fv-{group_key}-{facet}">50</span>
          </div>
          <input type="range" id="cse-sf-{group_key}-{facet}" min="0" max="100" value="50" step="1"
            style="width:100%;accent-color:{PALETTE['sky']}"
            oninput="cse_facet_change('{group_key}','{facet}',this.value)">
        </div>"""
            for facet in facets
        )
        big5_group_blocks.append(
            f"""
    <div class="cse-group" id="cse-g-{group_key}" style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer"
           onclick="cse_toggle('{group_key}')">
        <span style="font-weight:600;font-size:0.83rem;color:{PALETTE['ink']}">{group_label}</span>
        <span style="display:flex;gap:6px;align-items:center">
          <span id="cse-mv-{group_key}"
            style="font-size:0.75rem;font-weight:700;color:{PALETTE['blue']};min-width:32px;text-align:right">50</span>
          <span id="cse-arr-{group_key}" style="font-size:0.7rem;color:{PALETTE['muted']}">▶</span>
        </span>
      </div>
      <input type="range" id="cse-sl-{group_key}-mean" min="0" max="100" value="50" step="1"
        style="width:100%;margin-top:3px;accent-color:{PALETTE['blue']}"
        oninput="cse_mean_change('{group_key}',this.value)">
      <div id="cse-facets-{group_key}" style="display:none;margin-top:6px;padding:6px 8px;background:rgba(255,255,255,0.7);border-radius:7px">
        {facet_blocks}
      </div>
    </div>"""
        )
    big5_groups_html = "".join(big5_group_blocks)

    attack_checks_html = "".join(
        f"""<label style="font-size:0.82rem;display:flex;align-items:flex-start;gap:8px;cursor:pointer;padding:6px 7px;border-radius:8px;background:rgba(255,255,255,0.55)">
        <input type="checkbox" checked id="cse-atk-{i}" onchange="cse_update()" style="accent-color:{PALETTE['blue']};margin-top:2px">
        <span style="display:flex;flex-direction:column;gap:1px">
          <span title="{atk}" style="font-weight:600;color:{PALETTE['ink']}">{attack_labels[atk]}</span>
          <span style="font-size:0.71rem;color:{PALETTE['muted']}">{attack_context[atk] or "Attack family"}</span>
        </span></label>"""
        for i, atk in enumerate(all_attacks)
    )

    opinion_checks_html = "".join(
        f"""<label style="font-size:0.82rem;display:flex;align-items:flex-start;gap:8px;cursor:pointer;padding:6px 7px;border-radius:8px;background:rgba(255,255,255,0.55)">
        <input type="checkbox" checked id="cse-op-{i}" onchange="cse_update()" style="accent-color:{PALETTE['blue']};margin-top:2px">
        <span style="display:flex;flex-direction:column;gap:1px">
          <span title="{op}" style="font-weight:600;color:{PALETTE['ink']}">{opinion_labels[op]}</span>
          <span style="font-size:0.71rem;color:{PALETTE['muted']}">{opinion_context[op] or "Opinion family"}</span>
        </span></label>"""
        for i, op in enumerate(all_opinions)
    )

    return f"""
<div id="cse-root" style="display:grid;grid-template-columns:minmax(295px,330px) minmax(0,1fr);grid-template-rows:auto;gap:16px;align-items:start">

<!-- ══ LEFT: profile builder ══ -->
<div style="display:flex;flex-direction:column;gap:10px">

  <div style="background:#f0f5ff;border-radius:10px;padding:12px 14px">
    <div style="font-weight:700;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:10px">
      👤 Profile Configuration
    </div>
    <div style="font-size:0.76rem;line-height:1.45;color:{PALETTE['muted']};margin:-2px 0 10px">
      Build a synthetic profile manually or load a random observed profile. The score updates against the currently selected conditional task scope.
    </div>

    <!-- Estimation approach -->
    <div style="margin-bottom:10px">
      <label style="font-size:0.83rem;font-weight:600;color:{PALETTE['ink']}">Estimation approach
        <select id="cse-approach" onchange="cse_update()"
          style="width:100%;margin-top:3px;padding:5px;border-radius:6px;border:1px solid #dbe3ef;font-size:0.84rem">
          <option value="ridge" selected>Ridge task models (regularized linear)</option>
          <option value="knn">k-NN observed profiles (nonparametric)</option>
          <option value="cell">Cell baseline (context only, profile-free)</option>
          <option value="ensemble">Ensemble (ridge + k-NN average)</option>
        </select>
      </label>
      <div style="font-size:0.70rem;line-height:1.4;color:{PALETTE['muted']};margin-top:4px">
        Ridge extrapolates linear task coefficients; k-NN interpolates the observed AE of the most similar simulated profiles; the cell baseline ignores the profile entirely and shows pure context expectation; the ensemble averages the two profile-aware approaches.
      </div>
    </div>

    <!-- Big Five groups -->
    {big5_groups_html}

    <!-- Age -->
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:0.83rem;font-weight:600;color:{PALETTE['ink']}">
        <span>Age</span><span id="cse-v-age">40</span>
      </div>
      <input type="range" id="cse-sl-age" min="18" max="80" value="40" step="1"
        style="width:100%;accent-color:{PALETTE['blue']}"
        oninput="document.getElementById('cse-v-age').textContent=this.value;cse_update()">
    </div>

    <!-- Sex -->
    <div style="margin-bottom:6px">
      <label style="font-size:0.83rem;font-weight:600;color:{PALETTE['ink']}">Sex
        <select id="cse-sex" onchange="cse_update()"
          style="width:100%;margin-top:3px;padding:5px;border-radius:6px;border:1px solid #dbe3ef;font-size:0.88rem">
          <option>Male</option><option>Female</option><option>Other</option>
        </select>
      </label>
    </div>

    <!-- Action buttons -->
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px">
      <button onclick="cse_reset()" style="flex:1;padding:7px;background:{PALETTE['blue']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.80rem;font-weight:600">Reset to mean</button>
      <button onclick="cse_random()" style="flex:1;padding:7px;background:{PALETTE['teal']};color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:0.80rem;font-weight:600">Load random observed</button>
    </div>
  </div>

  <!-- Task selector -->
  <div style="background:#f0f5ff;border-radius:10px;padding:12px 14px">
    <div style="font-weight:700;font-size:0.92rem;color:{PALETTE['navy']};margin-bottom:8px">
      🎯 Conditional Task Scope
    </div>
    <div style="font-size:0.76rem;line-height:1.45;color:{PALETTE['muted']};margin:-1px 0 10px">
      Only the selected attack × opinion tasks contribute to the Conditional Susceptibility Score.
    </div>
    <div id="cse-task-summary" style="display:inline-flex;align-items:center;gap:8px;padding:5px 9px;border-radius:999px;background:rgba(29,78,137,0.08);color:{PALETTE['blue']};font-size:0.74rem;font-weight:700;margin-bottom:12px"></div>

    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <div style="font-size:0.78rem;font-weight:700;color:{PALETTE['muted']}">Attack vectors <span id="cse-atk-count"></span></div>
      <div style="display:flex;gap:5px">
        <button type="button" onclick="cse_all_atk(true)" style="padding:4px 8px;border-radius:999px;border:1px solid #bdd0ea;background:#fff;color:{PALETTE['blue']};cursor:pointer;font-size:0.72rem;font-weight:700">Select all</button>
        <button type="button" onclick="cse_all_atk(false)" style="padding:4px 8px;border-radius:999px;border:1px solid #ead2cc;background:#fff;color:{PALETTE['orange']};cursor:pointer;font-size:0.72rem;font-weight:700">Clear</button>
      </div>
    </div>
    <div id="cse-atk-checks" style="display:flex;flex-direction:column;gap:5px;margin-bottom:12px;max-height:160px;overflow:auto;padding-right:3px">
      {attack_checks_html}
    </div>

    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <div style="font-size:0.78rem;font-weight:700;color:{PALETTE['muted']}">Opinion targets <span id="cse-op-count"></span></div>
      <div style="display:flex;gap:5px">
        <button type="button" onclick="cse_all_op(true)" style="padding:4px 8px;border-radius:999px;border:1px solid #bdd0ea;background:#fff;color:{PALETTE['blue']};cursor:pointer;font-size:0.72rem;font-weight:700">Select all</button>
        <button type="button" onclick="cse_all_op(false)" style="padding:4px 8px;border-radius:999px;border:1px solid #ead2cc;background:#fff;color:{PALETTE['orange']};cursor:pointer;font-size:0.72rem;font-weight:700">Clear</button>
      </div>
    </div>
    <div id="cse-op-checks" style="display:flex;flex-direction:column;gap:5px;max-height:170px;overflow:auto;padding-right:3px">
      {opinion_checks_html}
    </div>

    <div style="font-size:0.72rem;line-height:1.45;color:{PALETTE['muted']};margin-top:8px">
      This selector defines the condition under which the score is estimated. Unselected tasks are excluded rather than down-weighted.
    </div>
  </div>

</div><!-- end left panel -->

<!-- ══ RIGHT: results ══ -->
<div style="display:flex;flex-direction:column;gap:14px">

  <!-- AE Grid -->
  <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
    <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
      📊 Predicted AE per Task
      <span style="font-weight:400;font-size:0.76rem;color:{PALETTE['muted']};margin-left:8px">
        deep red = strong predicted movement toward the goal; blue = predicted resistance (AE near 0)
      </span>
    </div>
    <div id="cse-grid" style="overflow:auto"></div>
  </div>

  <!-- Gauge + Radar side by side -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">

    <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
      <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
        🎯 Conditional Susceptibility Score
      </div>
      <div id="cse-gauge-wrap" style="text-align:center;padding:6px 0"></div>
      <div id="cse-gauge-text" style="text-align:center;font-size:0.78rem;color:{PALETTE['muted']};margin-top:4px"></div>
    </div>

    <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
      <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:4px">
        🕸 Profile Radar
        <span style="font-size:0.72rem;font-weight:400;color:{PALETTE['muted']}"> vs population avg</span>
      </div>
      <div id="cse-radar" style="height:200px"></div>
    </div>

  </div>

  <!-- Feature contributions -->
  <div style="background:#fff;border:1px solid #dbe3ef;border-radius:10px;padding:14px">
    <div style="font-weight:700;font-size:0.88rem;color:{PALETTE['navy']};margin-bottom:8px">
      📈 Feature Contributions to Susceptibility
      <span style="font-size:0.72rem;font-weight:400;color:{PALETTE['muted']}">
        (vs population mean · shows marginal effect)
      </span>
    </div>
    <div id="cse-contrib" style="font-size:0.82rem"></div>
  </div>

</div><!-- end right panel -->
</div><!-- end grid -->

<script>
(function(){{
// ── embedded data ──────────────────────────────────────────────────────────
const TASKS    = {json.dumps(tasks_json)};
const WEIGHTS  = {json.dumps(weights_json)};
const PROFILES = {json.dumps(profiles_json)};
const FEAT_MEANS = {json.dumps(feat_means)};
const FEAT_SDS = {json.dumps(feat_sds)};
const OBS = {json.dumps(obs_json)};
const CELL_MEANS = {json.dumps(cell_means_json)};
const KNN_K = 5;
const RADAR_LABELS = {json.dumps(radar_labels)};
const RADAR_MEANS  = {json.dumps(radar_means)};
const ALL_ATTACKS  = {json.dumps(all_attacks)};
const ALL_OPINIONS = {json.dumps(all_opinions)};
const ATTACK_LABELS = {json.dumps(attack_labels)};
const OPINION_LABELS = {json.dumps(opinion_labels)};
const B5_GROUPS    = {json.dumps([g[0] for g in BIG5_GROUPS])};
const B5_LABELS    = {json.dumps([g[1] for g in BIG5_GROUPS])};
const B5_FACETS    = {json.dumps({g[0]: g[2] for g in BIG5_GROUPS})};
const BIG5_NAMES   = {json.dumps({g[0]: f"profile_cont_big_five_{g[0]}_mean_pct" for g in BIG5_GROUPS})};
const BIG5_FACET_NAMES = {json.dumps(
    {g[0]: {f: f"profile_cont_big_five_{g[0]}_{f}_pct" for f in g[2]} for g in BIG5_GROUPS})};

let _radar_initialized = false;

// ── helpers ────────────────────────────────────────────────────────────────
function getVals() {{
  const v = {{'Intercept': 1}};
  // Big Five means/facets
  B5_GROUPS.forEach(g => {{
    const facetDiv = document.getElementById('cse-facets-'+g);
    const facetsVisible = facetDiv && facetDiv.style.display !== 'none';
    if (facetsVisible) {{
      // use facet values, compute mean
      let fsum = 0, fn = 0;
      (B5_FACETS[g] || []).forEach(f => {{
        const fv = parseFloat(document.getElementById('cse-sf-'+g+'-'+f)?.value || 50);
        v[BIG5_FACET_NAMES[g][f]] = fv;
        fsum += fv; fn++;
      }});
      const meanV = fn > 0 ? fsum / fn : 50;
      v[BIG5_NAMES[g]] = meanV;
      document.getElementById('cse-mv-'+g).textContent = meanV.toFixed(0);
      document.getElementById('cse-sl-'+g+'-mean').value = meanV.toFixed(0);
    }} else {{
      const mv = parseFloat(document.getElementById('cse-sl-'+g+'-mean')?.value || 50);
      v[BIG5_NAMES[g]] = mv;
      // also set facets to mean value (so they're consistent)
      (B5_FACETS[g] || []).forEach(f => {{
        v[BIG5_FACET_NAMES[g][f]] = mv;
      }});
    }}
  }});
  // age
  v['profile_cont_chronological_age'] = parseFloat(document.getElementById('cse-sl-age').value || 40);
  // sex
  const sx = document.getElementById('cse-sex').value;
  v['profile_cat__profile_cat_sex_Female'] = sx==='Female'?1:0;
  v['profile_cat__profile_cat_sex_Male']   = sx==='Male'  ?1:0;
  v['profile_cat__profile_cat_sex_Other']  = sx==='Other' ?1:0;
  return v;
}}

function getSelectedTasks() {{
  const atks = ALL_ATTACKS.filter((_,i) => document.getElementById('cse-atk-'+i)?.checked);
  const ops  = ALL_OPINIONS.filter((_,i) => document.getElementById('cse-op-'+i)?.checked);
  const pairs = [];
  atks.forEach(a => {{
    ops.forEach(o => {{
      if ((TASKS[a]||{{}})[o]) pairs.push([a,o]);
    }});
  }});
  return pairs;
}}

function predictRidge(pf, attackKey, opinionKey) {{
  const c = ((TASKS[attackKey] || {{}})[opinionKey]) || {{}};
  let ae = 0;
  Object.entries(c).forEach(([t,e]) => ae += e * (pf[t] ?? 0));
  /* AE is non-negative by measurement design; linear extrapolation below
     zero is a model artifact, so predictions are floored at 0 (resisted). */
  return Math.max(0, ae);
}}

function knnNeighbors(pf) {{
  // z-scored euclidean distance to every observed profile
  const dists = Object.entries(PROFILES).map(([pid, vals]) => {{
    let d2 = 0, used = 0;
    Object.entries(FEAT_SDS).forEach(([term, sd]) => {{
      if (!sd || sd < 1e-9) return;
      const a = (pf[term] ?? FEAT_MEANS[term] ?? 0);
      const b = (vals[term] ?? FEAT_MEANS[term] ?? 0);
      const z = (a - b) / sd;
      d2 += z * z; used++;
    }});
    return {{pid, d: used > 0 ? Math.sqrt(d2 / used) : 1e9}};
  }}).sort((x, y) => x.d - y.d);
  return dists;
}}

function predictKnn(neighbors, attackKey, opinionKey, excludePid) {{
  /* observed AE values are already >= 0; floor kept for safety */
  let wsum = 0, wtot = 0, taken = 0;
  for (const nb of neighbors) {{
    if (taken >= KNN_K) break;
    if (excludePid && nb.pid === excludePid) continue;
    const v = ((OBS[nb.pid] || {{}})[attackKey] || {{}})[opinionKey];
    if (v === undefined) continue;
    const w = 1 / (0.18 + nb.d);
    wsum += v * w; wtot += w; taken++;
  }}
  return Math.max(0, wtot > 0 ? wsum / wtot : (((CELL_MEANS[attackKey] || {{}})[opinionKey]) ?? 0));
}}

function getApproach() {{
  return document.getElementById('cse-approach')?.value || 'ridge';
}}

function predictAE(pf, attackKey, opinionKey, ctx) {{
  const approach = ctx?.approach || getApproach();
  if (approach === 'cell') return Math.max(0, ((CELL_MEANS[attackKey] || {{}})[opinionKey]) ?? 0);
  if (approach === 'knn') {{
    const nbs = ctx?.neighbors || knnNeighbors(pf);
    return predictKnn(nbs, attackKey, opinionKey, ctx?.excludePid);
  }}
  if (approach === 'ensemble') {{
    const nbs = ctx?.neighbors || knnNeighbors(pf);
    return 0.5 * predictRidge(pf, attackKey, opinionKey) + 0.5 * predictKnn(nbs, attackKey, opinionKey, ctx?.excludePid);
  }}
  return predictRidge(pf, attackKey, opinionKey);
}}

function computeScore(pf, selectedPairs) {{
  if (selectedPairs.length === 0) return {{ae_map:{{}}, raw:0, pct:50, dist:[], agreement:null}};
  const approach = getApproach();
  const ctx = {{approach}};
  if (approach === 'knn' || approach === 'ensemble') ctx.neighbors = knnNeighbors(pf);
  const ae_map = {{}};
  let wsum = 0, wtot = 0;
  selectedPairs.forEach(([a,o]) => {{
    const ae = predictAE(pf, a, o, ctx);
    if (!ae_map[a]) ae_map[a] = {{}};
    ae_map[a][o] = ae;
    const w = ((WEIGHTS[a] || {{}})[o]) || 1;
    wsum += ae * w; wtot += w;
  }});
  const raw = wtot > 0 ? wsum / wtot : 0;

  // distribution: re-score all observed profiles on selected tasks using the
  // same approach (k-NN scores leave the profile itself out for honesty)
  const dist = Object.entries(PROFILES).map(([pid, pfOrig]) => {{
    const octx = {{approach}};
    if (approach === 'knn' || approach === 'ensemble') {{
      octx.neighbors = knnNeighbors(pfOrig);
      octx.excludePid = pid;
    }}
    let ws=0, wt=0;
    selectedPairs.forEach(([a,o]) => {{
      const w = ((WEIGHTS[a] || {{}})[o]) || 1;
      ws += predictAE(pfOrig, a, o, octx) * w;
      wt += w;
    }});
    return wt > 0 ? ws/wt : 0;
  }}).sort((a,b)=>a-b);

  const below = dist.filter(v => v <= raw).length;
  const pct   = Math.round(below / dist.length * 100);

  // cross-approach agreement readout (ridge vs k-NN on this profile)
  let agreement = null;
  if (approach !== 'cell') {{
    const nbs = ctx.neighbors || knnNeighbors(pf);
    let rsum = 0, ksum = 0, n = 0;
    selectedPairs.forEach(([a,o]) => {{
      rsum += predictRidge(pf, a, o);
      ksum += predictKnn(nbs, a, o);
      n++;
    }});
    if (n > 0) agreement = {{ridge: rsum / n, knn: ksum / n}};
  }}
  return {{ae_map, raw, pct, dist, agreement}};
}}

function aeColor(ae) {{
  /* 0 = resisted (cool blue), ramps through neutral to deep red at ~60+ */
  const t = Math.max(0, Math.min(1, ae / 60));
  const r = Math.round(70 + 165 * t);
  const g = Math.round(110 - 60 * t);
  const b = Math.round(190 - 140 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function renderTaskScope(selectedPairs) {{
  const selAtks = ALL_ATTACKS.filter((_,i) => document.getElementById('cse-atk-'+i)?.checked);
  const selOps  = ALL_OPINIONS.filter((_,i) => document.getElementById('cse-op-'+i)?.checked);
  document.getElementById('cse-atk-count').textContent = `(${{
    selAtks.length
  }}/${{ALL_ATTACKS.length}})`;
  document.getElementById('cse-op-count').textContent = `(${{
    selOps.length
  }}/${{ALL_OPINIONS.length}})`;
  document.getElementById('cse-task-summary').textContent =
    `Conditional score uses ${{selectedPairs.length}} / ${{Object.values(TASKS).reduce((n, obj) => n + Object.keys(obj).length, 0)}} configured tasks`;
}}

function renderGrid(ae_map) {{
  const selAtks = ALL_ATTACKS.filter((_,i) => document.getElementById('cse-atk-'+i)?.checked);
  const selOps  = ALL_OPINIONS.filter((_,i) => document.getElementById('cse-op-'+i)?.checked);
  if (!selAtks.length || !selOps.length) {{
    document.getElementById('cse-grid').innerHTML='<p style="color:#aaa;font-size:0.82rem">Select at least one attack and one opinion.</p>';
    return;
  }}
  let h='<table style="border-collapse:collapse;font-size:0.82rem;width:100%">';
  h+=`<tr><th style="padding:6px 10px;border-bottom:2px solid #dbe3ef;text-align:left;font-size:0.75rem;color:{PALETTE['muted']}">Attack \\ Opinion</th>`;
  selOps.forEach(o=>h+=`<th title="${{o}}" style="padding:6px 8px;border-bottom:2px solid #dbe3ef;font-size:0.75rem;color:{PALETTE['muted']};min-width:90px">${{OPINION_LABELS[o]}}</th>`);
  h+='</tr>';
  selAtks.forEach(a=>{{
    h+=`<tr><td title="${{a}}" style="padding:7px 10px;font-weight:600;border-right:1px solid #dbe3ef;white-space:nowrap;font-size:0.80rem;color:{PALETTE['ink']};min-width:180px">${{ATTACK_LABELS[a]}}</td>`;
    selOps.forEach(o=>{{
      const ae = ((ae_map[a] || {{}})[o]) ?? 0;
      const bg=aeColor(ae), tc=ae>22?'#fff':'#10141b';
      const lbl = ae>=15?'↑ effective':(ae>=5?'↑ slight':'≈ resisted');
      h+=`<td title="${{a}} | ${{o}}" style="text-align:center;padding:8px 6px;background:${{bg}};color:${{tc}};border:2px solid rgba(255,255,255,0.35);border-radius:5px;font-weight:700;font-size:0.92rem">
        ${{ae.toFixed(1)}}<br><span style="font-size:0.65rem;opacity:0.85">${{lbl}}</span></td>`;
    }});
    h+='</tr>';
  }});
  document.getElementById('cse-grid').innerHTML=h+'</table>';
}}

function renderGauge(pct, raw, nProfiles, agreement) {{
  const gc = pct<33?'{PALETTE['teal']}':pct<67?'{PALETTE['amber']}':'{PALETTE['red']}';
  const label = pct>=75?'High conditional susceptibility':pct>=50?'Moderately high':pct>=25?'Moderately low':'Low conditional susceptibility';
  const agr = agreement
    ? `<div style="font-size:0.70rem;color:{PALETTE['muted']};margin-top:4px">ridge ${{agreement.ridge.toFixed(1)}} vs k-NN ${{agreement.knn.toFixed(1)}} (cross-approach check)</div>`
    : '';
  document.getElementById('cse-gauge-wrap').innerHTML=`
    <div style="position:relative;display:inline-block;width:200px">
      <div style="background:#e8edf5;border-radius:20px;height:18px;overflow:hidden;width:200px">
        <div style="background:${{gc}};height:100%;width:${{pct}}%;border-radius:20px;transition:width 0.35s ease"></div>
      </div>
      <div style="margin-top:8px;font-size:1.6rem;font-weight:800;color:${{gc}}">${{pct}}th</div>
      <div style="font-size:0.78rem;color:{PALETTE['muted']}">${{label}}</div>
      <div style="font-size:0.72rem;color:{PALETTE['muted']};margin-top:2px">raw score: ${{raw.toFixed(1)}}</div>
      ${{agr}}
    </div>`;
  document.getElementById('cse-gauge-text').textContent=
    `Conditional score ranks at the ${{pct}}th percentile vs ${{nProfiles}} simulated profiles under the selected task scope and estimation approach`;
}}

function renderRadar(pf) {{
  const userVals = B5_GROUPS.map(g => pf[BIG5_NAMES[g]] ?? 50);
  const closed_u = [...userVals, userVals[0]];
  const closed_m = [...RADAR_MEANS, RADAR_MEANS[0]];
  const closed_l = [...RADAR_LABELS, RADAR_LABELS[0]];
  const data = [
    {{type:'scatterpolar',r:closed_u,theta:closed_l,fill:'toself',name:'Your profile',
      fillcolor:'rgba(29,78,137,0.18)',line:{{color:'{PALETTE['blue']}',width:2}}}},
    {{type:'scatterpolar',r:closed_m,theta:closed_l,fill:'toself',name:'Population avg',
      fillcolor:'rgba(42,157,143,0.12)',line:{{color:'{PALETTE['teal']}',width:2,dash:'dot'}}}}
  ];
  const layout = {{
    polar:{{radialaxis:{{visible:true,range:[0,100],tickfont:{{size:8}}}},
            angularaxis:{{tickfont:{{size:9}}}}}},
    showlegend:true,legend:{{x:0.5,xanchor:'center',y:-0.15,orientation:'h',font:{{size:8}}}},
    margin:{{l:30,r:30,t:10,b:30}},paper_bgcolor:'white',font_family:'IBM Plex Sans,sans-serif',
    height:200,
  }};
  if (!_radar_initialized) {{
    Plotly.newPlot('cse-radar', data, layout, {{displayModeBar:false, responsive:true}});
    _radar_initialized = true;
  }} else {{
    Plotly.react('cse-radar', data, layout);
  }}
}}

function renderContrib(pf, selectedPairs) {{
  const contribs = {{}};
  if (!selectedPairs.length) {{
    document.getElementById('cse-contrib').innerHTML =
      '<div style="font-size:0.80rem;color:{PALETTE["muted"]}">Select at least one attack and one opinion target to see feature contributions.</div>';
    return;
  }}
  selectedPairs.forEach(([a,o]) => {{
    const c = ((TASKS[a] || {{}})[o]) || {{}};
    Object.entries(c).forEach(([term, coef]) => {{
      if (term==='Intercept') return;
      const delta = (pf[term]??0) - (FEAT_MEANS[term]??0);
      const contrib = coef * delta;
      if (!contribs[term]) contribs[term] = 0;
      contribs[term] += contrib;
    }});
  }});
  const sorted = Object.entries(contribs).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,10);
  const maxAbs = Math.max(...sorted.map(([,v])=>Math.abs(v)), 0.01);

  let h='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">';
  const pos = sorted.filter(([,v])=>v>0).slice(0,5);
  const neg = sorted.filter(([,v])=>v<0).slice(0,5);

  function renderSide(items, title, color) {{
    let s=`<div><div style="font-weight:700;font-size:0.78rem;color:${{color}};margin-bottom:5px">${{title}}</div>`;
    items.forEach(([term,val])=>{{
      const lbl=term.replace('profile_cont_','').replace('profile_cat__profile_cat_','').replace(/_/g,' ').replace(' z','').replace('big five ','').trim();
      const w=Math.round(Math.abs(val)/maxAbs*100);
      s+=`<div style="margin-bottom:4px">
        <div style="display:flex;justify-content:space-between;font-size:0.76rem">
          <span style="color:{PALETTE['ink']};text-overflow:ellipsis;overflow:hidden;max-width:140px" title="${{lbl}}">${{lbl}}</span>
          <span style="font-weight:700;color:${{color}}">${{val>0?'+':''}}${{val.toFixed(1)}}</span>
        </div>
        <div style="background:#f0f0f5;border-radius:4px;height:6px;overflow:hidden">
          <div style="background:${{color}};height:100%;width:${{w}}%;border-radius:4px"></div>
        </div></div>`;
    }});
    return s+'</div>';
  }}

  h+=renderSide(pos, '↑ Increases susceptibility', '{PALETTE['red']}');
  h+=renderSide(neg, '↓ Decreases susceptibility', '{PALETTE['teal']}');
  document.getElementById('cse-contrib').innerHTML=h+'</div>';
}}

// ── main update ─────────────────────────────────────────────────────────────
function cse_update() {{
  const pf   = getVals();
  const pairs = getSelectedTasks();
  const {{ae_map, raw, pct, dist, agreement}} = computeScore(pf, pairs);
  renderTaskScope(pairs);
  renderGrid(ae_map);
  renderGauge(pct, raw, dist.length, agreement);
  renderRadar(pf);
  renderContrib(pf, pairs);
}}
window.cse_update = cse_update;

// ── interactions ────────────────────────────────────────────────────────────
window.cse_toggle = function(g) {{
  const fd = document.getElementById('cse-facets-'+g);
  const arr = document.getElementById('cse-arr-'+g);
  const ms = document.getElementById('cse-sl-'+g+'-mean');
  if (fd.style.display==='none') {{
    fd.style.display='block'; arr.textContent='▼'; ms.style.opacity='0.4';
    // sync facets to current mean
    const mv = parseFloat(ms.value);
    (B5_FACETS[g]||[]).forEach(f => {{
      const el=document.getElementById('cse-sf-'+g+'-'+f);
      if(el){{el.value=mv;document.getElementById('cse-fv-'+g+'-'+f).textContent=mv;}}
    }});
  }} else {{
    fd.style.display='none'; arr.textContent='▶'; ms.style.opacity='1';
  }}
  cse_update();
}};

window.cse_mean_change = function(g, val) {{
  document.getElementById('cse-mv-'+g).textContent = val;
  cse_update();
}};

window.cse_facet_change = function(g, f, val) {{
  document.getElementById('cse-fv-'+g+'-'+f).textContent = val;
  // recompute mean from all facets
  const facets = B5_FACETS[g]||[];
  let sum=0;
  facets.forEach(ff=>{{
    sum+=parseFloat(document.getElementById('cse-sf-'+g+'-'+ff)?.value||50);
  }});
  const mean = (sum/facets.length).toFixed(0);
  document.getElementById('cse-mv-'+g).textContent=mean;
  document.getElementById('cse-sl-'+g+'-mean').value=mean;
  cse_update();
}};

window.cse_reset = function() {{
  B5_GROUPS.forEach(g=>{{
    const el=document.getElementById('cse-sl-'+g+'-mean');
    if(el){{el.value=50;document.getElementById('cse-mv-'+g).textContent=50;}}
    document.getElementById('cse-facets-'+g).style.display='none';
    document.getElementById('cse-arr-'+g).textContent='▶';
    if(el)el.style.opacity='1';
    (B5_FACETS[g]||[]).forEach(f=>{{
      const fe=document.getElementById('cse-sf-'+g+'-'+f);
      if(fe){{fe.value=50;document.getElementById('cse-fv-'+g+'-'+f).textContent=50;}}
    }});
  }});
  document.getElementById('cse-sl-age').value=40;
  document.getElementById('cse-v-age').textContent=40;
  document.getElementById('cse-sex').value='Male';
  cse_update();
}};

window.cse_random = function() {{
  const pids=Object.keys(PROFILES);
  const pf=PROFILES[pids[Math.floor(Math.random()*pids.length)]];
  B5_GROUPS.forEach(g=>{{
    const term=BIG5_NAMES[g]; const v=Math.round(pf[term]??50);
    document.getElementById('cse-sl-'+g+'-mean').value=v;
    document.getElementById('cse-mv-'+g).textContent=v;
    (B5_FACETS[g]||[]).forEach(f=>{{
      const ft=BIG5_FACET_NAMES[g][f]; const fv=Math.round(pf[ft]??v);
      const fe=document.getElementById('cse-sf-'+g+'-'+f);
      if(fe){{fe.value=fv;document.getElementById('cse-fv-'+g+'-'+f).textContent=fv;}}
    }});
  }});
  const age=Math.round(pf['profile_cont_chronological_age']??40);
  document.getElementById('cse-sl-age').value=age;
  document.getElementById('cse-v-age').textContent=age;
  const sex=pf['profile_cat__profile_cat_sex_Female']>0.5?'Female':pf['profile_cat__profile_cat_sex_Other']>0.5?'Other':'Male';
  document.getElementById('cse-sex').value=sex;
  cse_update();
}};

window.cse_all_atk = function(sel){{ALL_ATTACKS.forEach((_,i)=>{{const el=document.getElementById('cse-atk-'+i);if(el)el.checked=sel;}});cse_update();}};
window.cse_all_op  = function(sel){{ALL_OPINIONS.forEach((_,i)=>{{const el=document.getElementById('cse-op-'+i);if(el)el.checked=sel;}});cse_update();}};

// initialise
cse_update();
}})();
</script>"""


# ─── dashboard HTML ───────────────────────────────────────────────────────────

def _fig_umap_embedding(
    embedding_dashboard_dict: Dict[str, Any],
    color_by: str = "ontology",
) -> go.Figure:
    """Interactive 2-D UMAP scatter of all ontology leaves.

    Features:
    - Convex hull outlines per semantic group (filled, semi-transparent)
    - Bold centroid annotation labels with white background boxes
    - Per-leaf hover with full path + embedding text
    - color_by='ontology' uses PROFILE/ATTACK/OPINION colours
    - color_by='cluster' uses semantic k-means cluster colours
    """
    points = embedding_dashboard_dict.get("points", [])
    if not points:
        fig = go.Figure()
        fig.add_annotation(
            text="No embedding data available — run embed_ontology() first.",
            showarrow=False, font=dict(size=14, color=PALETTE["muted"]),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return _apply_style(fig, height=600)

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    ontologies = [p["ontology"] for p in points]
    clusters = [str(p["cluster"]) for p in points]
    leaves = [p["leaf"].replace("_", " ") for p in points]
    paths = [p["path"] for p in points]
    texts = [
        (p.get("text", "")[:110] + "…") if len(p.get("text", "")) > 110 else p.get("text", "")
        for p in points
    ]

    ONT_COLORS = {
        "PROFILE": PALETTE["blue"],
        "ATTACK":  PALETTE["orange"],
        "OPINION": PALETTE["teal"],
    }
    ONT_FILL = {
        "PROFILE": "rgba(29,78,137,0.10)",
        "ATTACK":  "rgba(231,111,81,0.10)",
        "OPINION": "rgba(42,157,143,0.10)",
    }

    CLUSTER_PALETTE = [
        "#e76f51", "#2a9d8f", "#1d4e89", "#c89b3c", "#8338ec",
        "#f72585", "#3a86ff", "#06d6a0", "#ef476f", "#118ab2",
        "#073b4c", "#ffd166",
    ]

    # Primary path segment (first level below ontology root) for group labelling.
    primary_groups = []
    for p in points:
        parts = p["path"].split(" > ")
        primary_groups.append(parts[1] if len(parts) > 1 else parts[0])

    hover = [
        f"<b>{leaf}</b><br>"
        f"<span style='color:#888;font-size:11px'>{ont} · {grp.replace('_', ' ')}</span>"
        f"<br><i style='font-size:10px'>{path}</i>"
        f"<br><span style='font-size:11px'>{txt}</span>"
        for leaf, ont, path, txt, grp in zip(leaves, ontologies, paths, texts, primary_groups)
    ]

    def _convex_hull_trace(idx_list: List[int], color: str, fillcolor: str, name: str) -> Optional[go.Scatter]:
        """Return a closed convex hull Scatter trace, or None if < 3 points."""
        if len(idx_list) < 3:
            return None
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            return None
        pts = np.column_stack([[xs[i] for i in idx_list], [ys[i] for i in idx_list]])
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None
        hull_xs = pts[hull.vertices, 0].tolist() + [pts[hull.vertices[0], 0]]
        hull_ys = pts[hull.vertices, 1].tolist() + [pts[hull.vertices[0], 1]]
        return go.Scatter(
            x=hull_xs, y=hull_ys,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line=dict(width=1.5, color=color, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            name=f"hull_{name}",
        )

    def _label_annotation(cx: float, cy: float, label: str, color: str) -> Dict[str, Any]:
        return dict(
            x=cx, y=cy,
            xref="x", yref="y",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color=color, family="IBM Plex Mono, monospace"),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
            align="center",
        )

    fig = go.Figure()
    annotations: List[Dict[str, Any]] = []

    if color_by == "ontology":
        # Hull per primary group, colored by ontology
        unique_groups = sorted(set(primary_groups))
        for grp in unique_groups:
            g_idx = [i for i, g in enumerate(primary_groups) if g == grp]
            if not g_idx:
                continue
            ont = ontologies[g_idx[0]]
            col = ONT_COLORS.get(ont, "#888")
            fill = ONT_FILL.get(ont, "rgba(128,128,128,0.08)")
            hull_tr = _convex_hull_trace(g_idx, col, fill, grp)
            if hull_tr is not None:
                fig.add_trace(hull_tr)

        # Scatter per ontology with hover
        for ont in ["PROFILE", "ATTACK", "OPINION"]:
            mask = [i for i, o in enumerate(ontologies) if o == ont]
            if not mask:
                continue
            fig.add_trace(go.Scatter(
                x=xs[np.array(mask)].tolist(), y=ys[np.array(mask)].tolist(),
                mode="markers+text",
                name=ont,
                marker=dict(
                    size=9,
                    color=ONT_COLORS.get(ont, "#888"),
                    line=dict(width=0.8, color="#fff"),
                    opacity=0.82,
                ),
                text=[leaves[i] for i in mask],
                textfont=dict(size=6, color=ONT_COLORS.get(ont, "#888")),
                textposition="top center",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[hover[i] for i in mask],
            ))

        # Centroid annotations per primary group
        for grp in unique_groups:
            g_idx = [i for i, g in enumerate(primary_groups) if g == grp]
            ont = ontologies[g_idx[0]]
            col = ONT_COLORS.get(ont, "#444")
            cx = float(xs[np.array(g_idx)].mean())
            cy = float(ys[np.array(g_idx)].mean())
            label = grp.replace("_", " ")
            annotations.append(_label_annotation(cx, cy, label, col))

    else:  # color_by == "cluster"
        unique_clusters = sorted(
            set(clusters),
            key=lambda c: int(c) if c.lstrip("-").isdigit() else 0,
        )
        cluster_color_map = {
            clust: CLUSTER_PALETTE[ci % len(CLUSTER_PALETTE)]
            for ci, clust in enumerate(unique_clusters)
        }

        # Build cluster labels from dominant primary group
        cluster_labels: Dict[str, str] = {}
        for clust in unique_clusters:
            mask = [i for i, c in enumerate(clusters) if c == clust]
            grp_counts: Dict[str, int] = {}
            for i in mask:
                k = f"{ontologies[i]}: {primary_groups[i].replace('_', ' ')}"
                grp_counts[k] = grp_counts.get(k, 0) + 1
            if grp_counts:
                top = max(grp_counts, key=grp_counts.__getitem__)
                cluster_labels[clust] = top.split(": ", 1)[-1][:28]
            else:
                cluster_labels[clust] = f"Cluster {clust}"

        # Hull + scatter per cluster
        for clust in unique_clusters:
            mask = [i for i, c in enumerate(clusters) if c == clust]
            col = cluster_color_map[clust]
            fill = col.replace("#", "rgba(") if col.startswith("#") else col
            # Build fill from hex
            r = int(col[1:3], 16)
            g_val = int(col[3:5], 16)
            b = int(col[5:7], 16)
            fill = f"rgba({r},{g_val},{b},0.10)"

            hull_tr = _convex_hull_trace(mask, col, fill, clust)
            if hull_tr is not None:
                fig.add_trace(hull_tr)

            label_short = cluster_labels[clust]
            fig.add_trace(go.Scatter(
                x=xs[np.array(mask)].tolist(), y=ys[np.array(mask)].tolist(),
                mode="markers",
                name=label_short,
                marker=dict(size=9, color=col, opacity=0.85, line=dict(width=0.8, color="#fff")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[hover[i] for i in mask],
            ))

            # Centroid annotation
            cx = float(xs[np.array(mask)].mean())
            cy = float(ys[np.array(mask)].mean())
            annotations.append(_label_annotation(cx, cy, label_short, col))

    model = embedding_dashboard_dict.get("model", "unknown")
    cluster_algo = embedding_dashboard_dict.get("cluster_algo", "")
    cluster_n = embedding_dashboard_dict.get("cluster_n", "")
    dim = embedding_dashboard_dict.get("dim", "")

    fig.update_layout(
        title=dict(
            text=(
                f"Semantic Embedding Space — UMAP 2D projection<br>"
                f"<span style='font-size:11px;color:#4a5d7a'>"
                f"Model: {model} ({dim}d) &nbsp;·&nbsp; {cluster_algo} k={cluster_n} &nbsp;·&nbsp; "
                f"Convex hulls = semantic group boundaries</span>"
            ),
            font=dict(size=15),
        ),
        xaxis=dict(title="UMAP Dim 1", showgrid=True, gridcolor=PALETTE["line"], zeroline=False),
        yaxis=dict(title="UMAP Dim 2", showgrid=True, gridcolor=PALETTE["line"], zeroline=False,
                   scaleanchor="x"),
        legend=dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.9)",
            bordercolor=PALETTE["line"], borderwidth=1,
        ),
        annotations=annotations,
        margin=dict(l=50, r=170, t=90, b=50),
        height=680,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
    )
    return fig


def _html_umap_embedding_tab(
    embedding_data_path: Optional[Path] = None,
    ontology_root: Optional[Path] = None,
    production_embedding_path: Optional[Path] = None,
) -> str:
    """Semantic embedding tab: ONE ontology source shown at a time (Test or
    Production toggle), colored by the top-level family (PROFILE / ATTACK /
    OPINION) with a consistent legend; wheel-zoom, no box select."""

    def _load_points(path: Optional[Path]) -> List[Dict[str, Any]]:
        if not path or not Path(path).exists():
            return []
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            return d.get("points", []) or []
        except Exception:
            return []

    sources: Dict[str, List[Dict[str, Any]]] = {}
    test_points = _load_points(embedding_data_path)
    if test_points:
        sources["test"] = test_points
    prod_points = _load_points(production_embedding_path)
    if prod_points:
        sources["production"] = prod_points

    if not sources:
        return """
<div style="padding:32px;background:var(--c-card,#f5f7fa);border-radius:6px;border:1px solid var(--c-line,#dbe3ef);max-width:700px;margin:24px auto;">
  <h3 style="color:var(--c-ink,#1d4e89);margin:0 0 12px">Semantic Embedding: not yet computed</h3>
  <p style="color:var(--c-muted,#4a5d7a);line-height:1.7">
    Run <code>embed_ontology(...)</code> from src.backend.utils.embeddings.semantic_embedding into
    evaluation/&lt;run&gt;/embeddings, then regenerate the dashboard.
  </p>
</div>"""

    fam_colors = {"PROFILE": PALETTE["blue"], "ATTACK": PALETTE["orange"], "OPINION": PALETTE["teal"]}
    payload: Dict[str, Any] = {"sources": {}}
    for src_name, pts in sources.items():
        payload["sources"][src_name] = [
            {
                "x": float(pt.get("x", 0.0)), "y": float(pt.get("y", 0.0)),
                "f": str(pt.get("ontology", "OTHER")).upper(),
                "leaf": str(pt.get("leaf", "")),
                "path": str(pt.get("path", "")),
            }
            for pt in pts
        ]
    payload_json = json.dumps(payload, ensure_ascii=False)
    fam_json = json.dumps(fam_colors)
    default_src = "test" if "test" in sources else next(iter(sources))
    buttons = "".join(
        f"<button class='se-btn{' active' if name == default_src else ''}' data-src='{name}'>{name.capitalize()} ontology</button>"
        for name in ["test", "production"] if name in sources
    )

    return f"""
<div id="se-root">
  <style>
    #se-root .se-bar{{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}}
    #se-root .se-btn{{padding:6px 13px;border-radius:4px;border:1px solid var(--c-line2,#c8d7ec);background:var(--c-sheet,#fff);color:var(--c-ink,#27364a);cursor:pointer;font-size:0.78rem;font-weight:700}}
    #se-root .se-btn.active{{background:{PALETTE['blue']};border-color:{PALETTE['blue']};color:#fff}}
    #se-root .se-note{{font-size:0.74rem;color:var(--c-muted,#5b6675)}}
    #se-root #se-plot{{width:100%;min-height:640px}}
  </style>
  <div class="se-bar">
    {buttons}
    <span class="se-note">UMAP projection of text-embedding-3-large leaf embeddings; one ontology source at a time, colored by top-level family. Scroll to zoom, drag to pan.</span>
  </div>
  <div id="se-plot"></div>
  <script>
  (function(){{
    const DATA = {payload_json};
    const FAM = {fam_json};
    let current = '{default_src}';
    function draw() {{
      const pts = DATA.sources[current] || [];
      const fams = ['PROFILE','ATTACK','OPINION'];
      const traces = fams.map(f => {{
        const sub = pts.filter(p => p.f === f);
        return {{
          type: 'scattergl', mode: 'markers', name: f,
          x: sub.map(p => p.x), y: sub.map(p => p.y),
          text: sub.map(p => `<b>${{p.leaf}}</b><br>${{p.path}}`),
          hovertemplate: '%{{text}}<extra>' + f + '</extra>',
          marker: {{ size: current === 'production' ? 5 : 9, opacity: 0.82,
                    color: FAM[f] || '#888', line: {{color: 'white', width: 0.5}} }},
        }};
      }}).filter(t => t.x.length);
      const dark = document.documentElement.dataset.theme === 'dark';
      const layout = {{
        dragmode: 'pan',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: dark ? 'rgba(255,255,255,0.015)' : 'rgba(20,40,80,0.02)',
        font: {{ color: dark ? '#e6ebf4' : '#171c24', family: 'IBM Plex Sans, sans-serif' }},
        xaxis: {{ title: 'UMAP-1', gridcolor: 'rgba(140,160,190,0.16)', zeroline: false }},
        yaxis: {{ title: 'UMAP-2', gridcolor: 'rgba(140,160,190,0.16)', zeroline: false }},
        legend: {{ orientation: 'h', y: 1.04, bgcolor: 'rgba(0,0,0,0)' }},
        height: 660, margin: {{l: 56, r: 20, t: 36, b: 52}},
      }};
      Plotly.react('se-plot', traces, layout,
        {{scrollZoom: true, displaylogo: false, modeBarButtonsToRemove: ['select2d','lasso2d','zoom2d','autoScale2d']}});
    }}
    document.querySelectorAll('#se-root .se-btn').forEach(btn => btn.addEventListener('click', () => {{
      document.querySelectorAll('#se-root .se-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      current = btn.dataset.src;
      draw();
    }}));
    const obs = new MutationObserver(() => draw());
    obs.observe(document.documentElement, {{attributes: true, attributeFilter: ['data-theme']}});
    setTimeout(draw, 60);
  }})();
  </script>
</div>"""


def _fig_effect_ranking(effect_df: pd.DataFrame, kind: str = "attack", long_df: pd.DataFrame | None = None) -> go.Figure:
    """Distribution-first ranking: per condition a violin of raw scenario AE with
    jittered points plus the cluster-bootstrap mean CI, ordered by mean AE."""
    if effect_df.empty or "mean_ae" not in effect_df.columns:
        return go.Figure().add_annotation(text="Effect summary unavailable", showarrow=False)
    work = effect_df.sort_values("mean_ae", ascending=True).reset_index(drop=True)
    cond_col = "attack_leaf" if kind == "attack" else "opinion_leaf"
    label_map = dict(zip(work[cond_col].astype(str), work["label"].astype(str))) if cond_col in work.columns else {}
    order = work[cond_col].astype(str).tolist() if cond_col in work.columns else work["label"].astype(str).tolist()

    palette = px.colors.qualitative.Bold + px.colors.qualitative.Safe
    fig = go.Figure()
    # Match the raw scenario distribution to the dimension the effect summary is
    # keyed on. For attacks that is the DISARM Execute tactic (a handful of real,
    # human-readable vectors) rather than the near-unique per-scenario triplet id.
    if kind == "attack":
        raw_col = next(
            (c for c in ("attack_execute_tactic", "attack_leaf_label", "attack_leaf")
             if long_df is not None and not long_df.empty and c in long_df.columns),
            "attack_leaf",
        )
    else:
        raw_col = "opinion_leaf"
    have_raw = long_df is not None and not long_df.empty and raw_col in long_df.columns and "adversarial_effectivity" in long_df.columns
    for i, cond in enumerate(order):
        color = palette[i % len(palette)]
        label = label_map.get(cond, cond)
        if have_raw:
            vals = long_df.loc[long_df[raw_col].astype(str) == cond, "adversarial_effectivity"].dropna()
            if len(vals):
                fig.add_trace(go.Violin(
                    x=vals, y=[label] * len(vals), orientation="h",
                    side="positive", width=1.5, points="all", pointpos=-0.45, jitter=0.35,
                    marker=dict(size=3.2, opacity=0.45, color=color),
                    line=dict(color=color, width=1.4),
                    opacity=0.85, meanline_visible=True, name=label, showlegend=False,
                    hoverinfo="skip",
                ))
        row = work.iloc[i]
        fig.add_trace(go.Scatter(
            x=[row["mean_ae"]], y=[label], mode="markers",
            marker=dict(symbol="diamond", size=13, color=color, line=dict(color="white", width=1.6)),
            error_x=dict(type="data", symmetric=False,
                         array=[max(0.0, row["ci_high"] - row["mean_ae"])],
                         arrayminus=[max(0.0, row["mean_ae"] - row["ci_low"])],
                         color=color, thickness=2.4, width=7),
            name=label, showlegend=False,
            hovertemplate=(f"<b>{label}</b><br>mean AE %{{x:.2f}}"
                           f"<br>95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                           f"<br>n={int(row.get('n_obs', 0))}<extra></extra>"),
        ))
    unit = "attack vector" if kind == "attack" else "opinion leaf"
    fig.add_vline(x=0, line=dict(color="rgba(130,140,160,0.55)", width=1, dash="dot"))
    fig.update_layout(
        title=f"Adversarial effectivity per {unit}: raw scenario distribution (violin + points) with bootstrap mean CI (diamond)",
        xaxis_title="AE per scenario (0 = resisted)",
        violinmode="overlay",
        margin=dict(l=10, r=24, t=52, b=46),
    )
    return _apply_style(fig, height=max(420, 64 * len(order) + 150))


def _fig_model_ladder(ladder_df: pd.DataFrame, meta: Dict[str, Any]) -> go.Figure:
    """Grouped-CV model ladder: context-only vs profile-augmented predictive R2."""
    if ladder_df.empty or "cv_r2_mean" not in ladder_df.columns:
        return go.Figure().add_annotation(text="Model ladder unavailable", showarrow=False)
    name_map = {
        "M0_context": "Context only<br>(attack + opinion FE)",
        "M0b_context_baseline": "Context + baseline",
        "M1_profile_linear": "+ Profile (ridge)",
        "M2_profile_boosted": "+ Profile (gradient boosting)",
    }
    work = ladder_df.copy()
    work["display"] = work["model"].astype(str).map(lambda m: name_map.get(m, m))
    colors = [PALETTE["muted"], PALETTE["sky"], PALETTE["blue"], PALETTE["orange"]][: len(work)]
    fig = go.Figure(
        go.Bar(
            x=work["display"], y=work["cv_r2_mean"],
            error_y=dict(type="data", array=work["cv_r2_sd"].fillna(0), color="rgba(120,130,150,0.9)", thickness=1.4),
            marker=dict(color=colors),
            text=[f"{v:.3f}" for v in work["cv_r2_mean"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>CV-R2 (unseen profiles): %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(130,140,160,0.6)", width=1))
    inc = ""
    try:
        rows = {r["model"]: r for r in work.to_dict(orient="records")}
        gb_inc = rows.get("M2_profile_boosted", {}).get("delta_r2_vs_context_baseline")
        if gb_inc is not None and np.isfinite(gb_inc):
            inc = f" | profile increment (boosted) = {gb_inc:+.4f}"
    except Exception:
        pass
    fig.update_layout(
        title=f"Predictive model ladder, GroupKFold by profile (n={meta.get('n_obs', '?')} scenarios, {meta.get('n_profiles', '?')} held-out profile folds){inc}",
        yaxis_title="CV-R2 on unseen profiles",
        margin=dict(l=60, r=30, t=58, b=70),
    )
    return _apply_style(fig, height=460)


def _ontology_dendrogram_segments(paths_ordered: List[str]) -> List[Dict[str, float]]:
    """Rectangular dendrogram segments for an ordered list of ontology paths.

    Leaves sit at x = 0..n-1, y = 0. Internal nodes sit at the mean x of their
    children with height proportional to subtree depth, so the drawn tree IS
    the ontology hierarchy rather than any value-based clustering.
    """
    if not paths_ordered:
        return []
    split_paths = [[seg.strip() for seg in str(p).split(">") if seg.strip()] for p in paths_ordered]
    max_len = max(len(parts) for parts in split_paths)

    def _node_children(prefix: tuple, depth: int) -> List[tuple]:
        seen: List[tuple] = []
        for parts in split_paths:
            if len(parts) > depth and tuple(parts[:depth]) == prefix:
                child = tuple(parts[: depth + 1])
                if child not in seen:
                    seen.append(child)
        return seen

    leaf_x = {tuple(parts): float(i) for i, parts in enumerate(split_paths)}
    segments: List[Dict[str, float]] = []

    def _layout(prefix: tuple, depth: int) -> Tuple[float, float]:
        children = _node_children(prefix, depth)
        is_leaf_here = any(tuple(parts) == prefix for parts in split_paths)
        if not children:
            return leaf_x.get(prefix, 0.0), 0.0
        child_points: List[Tuple[float, float]] = []
        for child in children:
            cx, cy = _layout(child, depth + 1)
            child_points.append((cx, cy))
        if is_leaf_here:
            child_points.append((leaf_x[prefix], 0.0))
        xs = [cp[0] for cp in child_points]
        node_y = float(max_len - depth)
        node_x = float(np.mean(xs))
        for cx, cy in child_points:
            segments.append({"x0": cx, "y0": cy, "x1": cx, "y1": node_y})
        segments.append({"x0": min(xs), "y0": node_y, "x1": max(xs), "y1": node_y})
        return node_x, node_y

    roots = _node_children(tuple(), 0)
    root_points = [_layout(r, 1) for r in roots]
    if len(root_points) > 1:
        top_y = float(max_len + 1)
        xs = [rp[0] for rp in root_points]
        for rx, ry in root_points:
            segments.append({"x0": rx, "y0": ry, "x1": rx, "y1": top_y})
        segments.append({"x0": min(xs), "y0": top_y, "x1": max(xs), "y1": top_y})
    return segments


def _fig_moderation_heatmap_dendro(
    scan_df: pd.DataFrame,
    context_paths: Dict[str, str],
    context_name: str = "opinion",
    title_prefix: str = "Profile moderation heatmap",
) -> go.Figure:
    """Feature x context moderation heatmap with the REAL ontology hierarchy
    mounted as a top dendrogram and the profile-feature family hierarchy as a
    right dendrogram. Uses ALL profile features from the cluster-robust scan."""
    needed = {"feature", "context", "estimate"}
    if scan_df.empty or not needed.issubset(scan_df.columns):
        return go.Figure().add_annotation(text="Moderation scan unavailable", showarrow=False)

    work = scan_df.copy()
    work["family"] = work["ontology_group"] if "ontology_group" in work.columns else "Other"
    work["family"] = work["family"].fillna("Other").astype(str)

    ctx_with_paths = sorted(
        work["context"].astype(str).unique().tolist(),
        key=lambda c: context_paths.get(c, c),
    )
    ctx_paths_ordered = [context_paths.get(c, c) for c in ctx_with_paths]
    ctx_labels = _unique_display_map(ctx_with_paths)

    feat_rows = (
        work[["feature", "family"]].drop_duplicates()
        .sort_values(["family", "feature"]).reset_index(drop=True)
    )
    features_ordered = feat_rows["feature"].tolist()
    feature_pseudo_paths = [f"{r['family']} > {_pretty_moderator_label(r['feature'])}" for r in feat_rows.to_dict(orient="records")]

    pivot = work.pivot_table(index="feature", columns="context", values="estimate", aggfunc="mean")
    pivot = pivot.reindex(index=features_ordered, columns=ctx_with_paths)
    qpivot = (
        work.pivot_table(index="feature", columns="context", values="q_value", aggfunc="mean")
        .reindex(index=features_ordered, columns=ctx_with_paths)
        if "q_value" in work.columns else pd.DataFrame(index=pivot.index, columns=pivot.columns)
    )

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.16, 0.84], column_widths=[0.86, 0.14],
        horizontal_spacing=0.004, vertical_spacing=0.006,
        shared_xaxes=True, shared_yaxes=True,
        specs=[[{}, None], [{}, {}]],
    )

    for seg in _ontology_dendrogram_segments(ctx_paths_ordered):
        fig.add_trace(go.Scatter(
            x=[seg["x0"], seg["x1"]], y=[seg["y0"], seg["y1"]],
            mode="lines", line=dict(color="#7e8ca0", width=1.1),
            hoverinfo="skip", showlegend=False), row=1, col=1)

    for seg in _ontology_dendrogram_segments(feature_pseudo_paths):
        fig.add_trace(go.Scatter(
            x=[seg["y0"], seg["y1"]], y=[seg["x0"], seg["x1"]],
            mode="lines", line=dict(color="#7e8ca0", width=1.1),
            hoverinfo="skip", showlegend=False), row=2, col=2)

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=list(range(len(ctx_with_paths))),
        y=list(range(len(features_ordered))),
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="b (AE per SD)", thickness=13, len=0.78, y=0.40),
        hovertext=[[f"{_pretty_moderator_label(f)}<br>{ctx_labels[c]}<br>b = {pivot.loc[f, c]:.3f}"
                    + (f"<br>q = {qpivot.loc[f, c]:.3f}" if pd.notna(qpivot.loc[f, c]) else "")
                    for c in ctx_with_paths] for f in features_ordered],
        hoverinfo="text",
    ), row=2, col=1)

    annotations = []
    for yi, f in enumerate(features_ordered):
        for xi, c in enumerate(ctx_with_paths):
            q = qpivot.loc[f, c]
            if pd.notna(q) and q < 0.05:
                annotations.append(dict(x=xi, y=yi, text="*", showarrow=False,
                                        xref="x3", yref="y3", font=dict(size=12, color="#101010")))
    fig.update_layout(annotations=annotations)

    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)
    fig.update_xaxes(
        tickmode="array", tickvals=list(range(len(ctx_with_paths))),
        ticktext=[ctx_labels[c] for c in ctx_with_paths],
        tickangle=34, tickfont=dict(size=9.5), automargin=True, row=2, col=1)
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(features_ordered))),
        ticktext=[_pretty_moderator_label(f) for f in features_ordered],
        tickfont=dict(size=8.6), automargin=True, autorange="reversed", row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_layout(
        title=f"{title_prefix}: ALL profile features x {context_name} (cluster-robust slopes; * q<0.05). Top dendrogram = {context_name} ontology hierarchy; right = profile family hierarchy.",
        margin=dict(l=10, r=8, t=58, b=120),
        showlegend=False,
    )
    return _apply_style(fig, height=max(640, 17 * len(features_ordered) + 280))


def _fig_pooled_moderation_forest(scan_pooled_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """Forest of pooled cluster-robust moderation slopes (AE per SD of feature),
    with 95% CIs and FDR annotation."""
    needed = {"feature", "estimate", "std_error"}
    if scan_pooled_df.empty or not needed.issubset(scan_pooled_df.columns):
        return go.Figure().add_annotation(text="Pooled moderation scan unavailable", showarrow=False)
    work = scan_pooled_df.copy()
    work["abs_est"] = work["estimate"].abs()
    work = work.sort_values("abs_est", ascending=False).head(top_n)
    work = work.sort_values("estimate").reset_index(drop=True)
    labels = [
        (str(r.get("label")) if pd.notna(r.get("label")) else _pretty_moderator_label(str(r["feature"])))
        for r in work.to_dict(orient="records")
    ]
    colors = [PALETTE["red"] if v >= 0 else PALETTE["teal"] for v in work["estimate"]]
    sig = ["*" if (pd.notna(q) and q < 0.05) else ("+" if (pd.notna(q) and q < 0.10) else "")
           for q in work.get("q_value", pd.Series([np.nan] * len(work)))]
    fig = go.Figure(go.Scatter(
        x=work["estimate"],
        y=[f"{l} {s}".strip() for l, s in zip(labels, sig)],
        mode="markers",
        marker=dict(size=9, color=colors, line=dict(color="white", width=1)),
        error_x=dict(type="data", array=(1.96 * work["std_error"]).tolist(),
                     color="rgba(120,130,150,0.85)", thickness=1.4, width=4),
        customdata=np.stack([
            work["p_value"].to_numpy() if "p_value" in work.columns else np.full(len(work), np.nan),
            work["q_value"].to_numpy() if "q_value" in work.columns else np.full(len(work), np.nan),
        ], axis=1),
        hovertemplate="<b>%{y}</b><br>b = %{x:.3f} AE per SD<br>p = %{customdata[0]:.4f}, q = %{customdata[1]:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="rgba(130,140,160,0.6)", width=1))
    fig.update_layout(
        title="Pooled profile moderation slopes on AE (cluster-robust SEs; * q<0.05, + q<0.10; warm = amplifies)",
        xaxis_title="b (AE change per SD of feature)",
        margin=dict(l=10, r=26, t=52, b=46),
        yaxis=dict(tickfont=dict(size=9.5)),
    )
    return _apply_style(fig, height=max(460, 22 * len(work) + 170))


def _html_key_findings(
    attack_effects: pd.DataFrame,
    opinion_effects: pd.DataFrame,
    pooled_scan: pd.DataFrame,
    profile_effects: pd.DataFrame,
    ladder_df: pd.DataFrame,
    ladder_meta: Dict[str, Any],
    long_df: pd.DataFrame,
) -> str:
    """Answer-first overview: which attacks, which opinions, which profile features."""

    def _fig_html(fig: go.Figure) -> str:
        return fig.to_html(include_plotlyjs=False, full_html=False, config=PLOTLY_CONFIG)

    cards: List[Tuple[str, str, str]] = []
    if not attack_effects.empty:
        top_atk = attack_effects.iloc[0]
        cards.append(("Most effective attack", str(top_atk.get("label", "n/a")).replace("_", " "),
                      f"mean AE {top_atk['mean_ae']:+.1f} [{top_atk['ci_low']:.1f}, {top_atk['ci_high']:.1f}]"))
        weakest = attack_effects.iloc[-1]
        cards.append(("Least effective attack", str(weakest.get("label", "n/a")).replace("_", " "),
                      f"mean AE {weakest['mean_ae']:+.1f} [{weakest['ci_low']:.1f}, {weakest['ci_high']:.1f}]"))
    if not opinion_effects.empty:
        top_op = opinion_effects.iloc[0]
        cards.append(("Most movable opinion", str(top_op.get("label", "n/a")).replace("_", " "),
                      f"mean AE {top_op['mean_ae']:+.1f} [{top_op['ci_low']:.1f}, {top_op['ci_high']:.1f}]"))
    if not pooled_scan.empty:
        sig = pooled_scan.sort_values("p_value")
        amp = sig[sig["estimate"] > 0].head(1)
        prot = sig[sig["estimate"] < 0].head(1)
        if not amp.empty:
            row = amp.iloc[0]
            cards.append(("Top amplifying trait", _pretty_moderator_label(str(row["feature"])),
                          f"b={row['estimate']:+.2f} per SD, q={row['q_value']:.3f}"))
        if not prot.empty:
            row = prot.iloc[0]
            cards.append(("Top protective trait", _pretty_moderator_label(str(row["feature"])),
                          f"b={row['estimate']:+.2f} per SD, q={row['q_value']:.3f}"))
    if not ladder_df.empty:
        try:
            rows = {r["model"]: r for r in ladder_df.to_dict(orient="records")}
            gb = rows.get("M2_profile_boosted", {})
            cards.append(("Profile predictive increment",
                          f"{gb.get('delta_r2_vs_context_baseline', float('nan')):+.4f} CV-R2",
                          "boosted model vs context+baseline, unseen profiles"))
        except Exception:
            pass
    if "adversarial_effectivity" in long_df.columns:
        ae = long_df["adversarial_effectivity"].dropna()
        if len(ae):
            cards.append(("Overall manipulation yield", f"{(ae > 0).mean() * 100:.1f}% scenarios moved",
                          f"grand mean AE {ae.mean():+.2f} across {len(ae)} scenarios"))

    cards_html = "".join(
        f"""<div class="kf-card"><div class="kf-k">{k}</div><div class="kf-v">{v}</div><div class="kf-s">{s}</div></div>"""
        for k, v, s in cards
    )

    figs_html = ""
    if not attack_effects.empty:
        figs_html += f"<div class='kf-figwrap'>{_fig_html(_fig_effect_ranking(attack_effects, 'attack', long_df))}</div>"
    if not opinion_effects.empty:
        figs_html += f"<div class='kf-figwrap'>{_fig_html(_fig_effect_ranking(opinion_effects, 'opinion', long_df))}</div>"

    profile_rows_html = ""
    if not profile_effects.empty:
        head = profile_effects.head(6).to_dict(orient="records")
        tail = profile_effects.tail(6).to_dict(orient="records")
        def _rows(rows):
            return "".join(
                f"<tr><td>{r['profile_id']}</td><td class='num'>{r['eb_mean_ae']:+.2f}</td>"
                f"<td class='num'>{r['raw_mean_ae']:+.2f}</td><td class='num'>{r['eb_percentile']:.0f}</td>"
                f"<td class='num'>{int(r['n_obs'])}</td></tr>"
                for r in rows
            )
        profile_rows_html = f"""
        <div class="kf-tables">
          <div>
            <div class="kf-table-title">Most susceptible profiles (EB-shrunken mean AE)</div>
            <table class="kf-table"><tr><th>Profile</th><th>EB mean AE</th><th>Raw</th><th>Pct</th><th>n</th></tr>{_rows(head)}</table>
          </div>
          <div>
            <div class="kf-table-title">Most resistant profiles</div>
            <table class="kf-table"><tr><th>Profile</th><th>EB mean AE</th><th>Raw</th><th>Pct</th><th>n</th></tr>{_rows(tail)}</table>
          </div>
        </div>"""

    return f"""
<div id="kf-root">
  <style>
    #kf-root .kf-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(225px,1fr));gap:10px;margin-bottom:14px}}
    #kf-root .kf-card{{background:linear-gradient(160deg,rgba(29,78,137,0.06),rgba(42,157,143,0.05));border:1px solid #d8dfe9;border-radius:6px;padding:12px 13px;min-width:0;overflow:hidden}}
    #kf-root .kf-k{{font:700 9px "IBM Plex Mono",monospace;letter-spacing:0.10em;text-transform:uppercase;color:#5b6675;overflow-wrap:anywhere}}
    #kf-root .kf-v{{font-weight:800;font-size:clamp(0.78rem,1.1vw,0.98rem);margin-top:5px;line-height:1.32;overflow-wrap:anywhere;word-break:break-word;hyphens:auto}}
    #kf-root .kf-s{{font-size:0.72rem;color:#5b6675;margin-top:4px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}}
    #kf-root .kf-figwrap{{margin-bottom:12px}}
    #kf-root .kf-tables{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;margin-top:8px}}
    #kf-root .kf-table-title{{font-weight:700;font-size:0.82rem;margin-bottom:6px}}
    #kf-root .kf-table{{width:100%;border-collapse:collapse;font-size:0.78rem}}
    #kf-root .kf-table th{{text-align:left;font:700 9.5px "IBM Plex Mono",monospace;letter-spacing:0.08em;text-transform:uppercase;color:#5b6675;border-bottom:1.5px solid #c9d2df;padding:5px 8px}}
    #kf-root .kf-table td{{padding:5px 8px;border-bottom:1px solid #e3e8f0}}
    #kf-root .kf-table td.num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
  </style>
  <div class="kf-grid">{cards_html}</div>
  {figs_html}
  {profile_rows_html}
</div>"""


def _html_moderation_paths(
    scan_by_attack: pd.DataFrame,
    cell_effects: pd.DataFrame,
    pooled_scan: pd.DataFrame,
) -> str:
    """Tripartite 2D moderation-path diagram: profile features -> attacks -> opinions."""
    if scan_by_attack.empty or cell_effects.empty:
        return "<p>Moderation path data unavailable for this run.</p>"

    work = scan_by_attack.copy()
    work["abs_est"] = work["estimate"].abs()
    top_features = (
        work.groupby("feature")["abs_est"].max().sort_values(ascending=False).head(18).index.tolist()
    )
    work = work[work["feature"].isin(top_features)]

    attacks = sorted(set(work["context"].astype(str)) | set(cell_effects["attack_leaf"].astype(str)))
    opinions = sorted(set(cell_effects["opinion_leaf"].astype(str)))
    atk_labels = _unique_display_map(attacks)
    op_labels = _unique_display_map(opinions)

    group_lookup: Dict[str, str] = {}
    if "ontology_group" in work.columns:
        group_lookup = (
            work[["feature", "ontology_group"]]
            .dropna().drop_duplicates()
            .set_index("feature")["ontology_group"].astype(str).to_dict()
        )
    feature_nodes = [
        {"id": f, "label": _pretty_moderator_label(f), "family": group_lookup.get(f, "Other")}
        for f in top_features
    ]
    mod_edges = [
        {
            "f": str(r["feature"]), "a": str(r["context"]),
            "est": float(r["estimate"]), "p": float(r.get("p_value", np.nan)),
            "q": float(r.get("q_value", np.nan)),
        }
        for r in work.to_dict(orient="records")
        if np.isfinite(r["estimate"])
    ]
    ae_edges = [
        {
            "a": str(r["attack_leaf"]), "o": str(r["opinion_leaf"]),
            "ae": float(r["mean_ae"]),
            "lo": float(r.get("ci_low", np.nan)), "hi": float(r.get("ci_high", np.nan)),
        }
        for r in cell_effects.to_dict(orient="records")
        if np.isfinite(r["mean_ae"])
    ]
    pooled = {
        str(r["feature"]): {"est": float(r["estimate"]), "q": float(r.get("q_value", np.nan))}
        for r in pooled_scan.to_dict(orient="records")
    } if not pooled_scan.empty else {}

    payload = json.dumps({
        "features": feature_nodes,
        "attacks": [{"id": a, "label": atk_labels[a]} for a in attacks],
        "opinions": [{"id": o, "label": op_labels[o]} for o in opinions],
        "modEdges": mod_edges,
        "aeEdges": ae_edges,
        "pooled": pooled,
    }, ensure_ascii=False)

    return f"""
<div id="mp-root">
  <style>
    #mp-root .mp-shell{{display:grid;grid-template-columns:minmax(250px,290px) minmax(0,1fr);gap:14px;align-items:start}}
    #mp-root .mp-card{{background:#f7faff;border:1px solid #dbe3ef;border-radius:6px;padding:12px 13px;margin-bottom:10px}}
    #mp-root .mp-title{{font-weight:800;font-size:0.88rem;margin-bottom:7px}}
    #mp-root .mp-sub{{font-size:0.75rem;line-height:1.5;color:#5b6675;margin-bottom:8px}}
    #mp-root .mp-slider-meta{{display:flex;justify-content:space-between;font-size:0.75rem;font-weight:700;color:#5b6675;margin-bottom:4px}}
    #mp-root input[type=range]{{width:100%;accent-color:{PALETTE['blue']}}}
    #mp-root .mp-seg{{display:flex;gap:5px;flex-wrap:wrap}}
    #mp-root .mp-btn{{padding:5px 10px;border-radius:4px;border:1px solid #c8d7ec;background:#fff;cursor:pointer;font-size:0.74rem;font-weight:700;color:#27364a}}
    #mp-root .mp-btn.active{{background:{PALETTE['blue']};border-color:{PALETTE['blue']};color:#fff}}
    #mp-root .mp-search{{width:100%;padding:7px 9px;border-radius:5px;border:1px solid #dbe3ef;font-size:0.8rem}}
    #mp-root .mp-canvas{{background:linear-gradient(180deg,#fcfdff,#f6f9ff);border:1px solid #dbe3ef;border-radius:6px;overflow:auto;min-height:560px}}
    #mp-root .mp-readout{{font-size:0.76rem;line-height:1.55;color:#41506a;background:#fff;border:1px solid #dbe3ef;border-radius:6px;padding:9px 11px;margin-top:10px}}
    #mp-root svg text{{font-family:"IBM Plex Sans",sans-serif}}
  </style>
  <div class="mp-shell">
    <div>
      <div class="mp-card">
        <div class="mp-title">Moderation Paths</div>
        <div class="mp-sub">Left: profile features (top by conditional effect). Middle: attack vectors. Right: opinion leaves. Feature-to-attack curves show how strongly the feature amplifies (warm) or dampens (cool) that attack's effectivity. Attack-to-opinion curves show where each attack lands. Hover any label for the full name.</div>
      </div>
      <div class="mp-card">
        <div class="mp-title">Significance filter</div>
        <div class="mp-slider-meta"><span>max q (FDR)</span><span id="mp-q-val">0.25</span></div>
        <input type="range" id="mp-q" min="0.01" max="1" step="0.01" value="0.25">
        <div class="mp-slider-meta" style="margin-top:9px"><span>min |slope|</span><span id="mp-b-val">0.0</span></div>
        <input type="range" id="mp-b" min="0" max="30" step="0.5" value="0">
      </div>
      <div class="mp-card">
        <div class="mp-title">Direction</div>
        <div class="mp-seg" id="mp-sign">
          <button class="mp-btn active" data-sign="all">All</button>
          <button class="mp-btn" data-sign="pos">Amplifying</button>
          <button class="mp-btn" data-sign="neg">Protective</button>
        </div>
      </div>
      <div class="mp-card">
        <div class="mp-title">AE layer</div>
        <div class="mp-seg" id="mp-ae">
          <button class="mp-btn active" data-ae="on">Show attack to opinion</button>
          <button class="mp-btn" data-ae="off">Hide</button>
        </div>
        <div class="mp-sub" style="margin-top:8px">Click any node to isolate its paths. Click the background to reset.</div>
      </div>
      <div class="mp-card">
        <div class="mp-title">Search feature</div>
        <input class="mp-search" id="mp-search" placeholder="e.g. neuroticism">
      </div>
      <div class="mp-readout" id="mp-readout"></div>
    </div>
    <div class="mp-canvas" id="mp-canvas"></div>
  </div>
  <script>
  (function(){{
    const D = {payload};
    const state = {{q: 0.25, b: 0, sign: 'all', ae: true, focus: null, search: ''}};
    const canvas = document.getElementById('mp-canvas');
    const COL_W = 320, ROW_F = 30, ROW_A = 44, ROW_O = 30, PAD = 56;
    const TRUNC = t => t.length > 30 ? t.slice(0, 28) + '..' : t;
    const labelW = arr => Math.min(330, Math.max(...arr.map(t => TRUNC(t).length), 8) * 6.4 + 26);

    function visModEdges() {{
      return D.modEdges.filter(e => {{
        if (!(isFinite(e.q) ? e.q <= state.q : state.q >= 1)) return false;
        if (Math.abs(e.est) < state.b) return false;
        if (state.sign === 'pos' && e.est < 0) return false;
        if (state.sign === 'neg' && e.est > 0) return false;
        if (state.search && !e.f.toLowerCase().includes(state.search)) return false;
        if (state.focus) {{
          if (state.focus.kind === 'f' && e.f !== state.focus.id) return false;
          if (state.focus.kind === 'a' && e.a !== state.focus.id) return false;
          if (state.focus.kind === 'o') return false;
        }}
        return true;
      }});
    }}
    function visAeEdges() {{
      if (!state.ae) return [];
      return D.aeEdges.filter(e => {{
        if (state.focus) {{
          if (state.focus.kind === 'a' && e.a !== state.focus.id) return false;
          if (state.focus.kind === 'o' && e.o !== state.focus.id) return false;
          if (state.focus.kind === 'f') {{
            const feats = new Set(visModEdges().map(m => m.a));
            if (!feats.has(e.a)) return false;
          }}
        }}
        return true;
      }});
    }}
    function curve(x0, y0, x1, y1) {{
      const mx = (x0 + x1) / 2;
      return `M ${{x0}} ${{y0}} C ${{mx}} ${{y0}}, ${{mx}} ${{y1}}, ${{x1}} ${{y1}}`;
    }}
    function render() {{
      const mods = visModEdges();
      const aes = visAeEdges();
      const usedF = state.search
        ? D.features.filter(f => f.id.toLowerCase().includes(state.search))
        : D.features;
      const H = Math.max(usedF.length * ROW_F, D.attacks.length * ROW_A, D.opinions.length * ROW_O) + PAD * 2;
      const leftW = labelW(usedF.map(f => f.label));
      const rightW = labelW(D.opinions.map(o => o.label));
      const W = leftW + COL_W * 2 + rightW + 60;
      const fy = {{}}, ay = {{}}, oy = {{}};
      usedF.forEach((f, i) => fy[f.id] = PAD + i * ROW_F + (H - 2 * PAD - usedF.length * ROW_F) / 2 + ROW_F / 2);
      D.attacks.forEach((a, i) => ay[a.id] = PAD + i * ROW_A + (H - 2 * PAD - D.attacks.length * ROW_A) / 2 + ROW_A / 2);
      D.opinions.forEach((o, i) => oy[o.id] = PAD + i * ROW_O + (H - 2 * PAD - D.opinions.length * ROW_O) / 2 + ROW_O / 2);
      const xF = leftW, xA = xF + COL_W, xO = xA + COL_W;
      const maxB = Math.max(...D.modEdges.map(e => Math.abs(e.est)), 1);
      const maxAE = Math.max(...D.aeEdges.map(e => Math.abs(e.ae)), 1);
      let s = `<svg width="${{W}}" height="${{H}}" viewBox="0 0 ${{W}} ${{H}}" style="display:block">`;
      s += `<text x="${{xF - 8}}" y="${{PAD - 24}}" text-anchor="end" style="font:700 10px 'IBM Plex Mono',monospace;letter-spacing:1.4px;fill:#5b6675">PROFILE FEATURES</text>`;
      s += `<text x="${{xA}}" y="${{PAD - 24}}" text-anchor="middle" style="font:700 10px 'IBM Plex Mono',monospace;letter-spacing:1.4px;fill:#5b6675">ATTACK VECTORS</text>`;
      s += `<text x="${{xO + 8}}" y="${{PAD - 24}}" text-anchor="start" style="font:700 10px 'IBM Plex Mono',monospace;letter-spacing:1.4px;fill:#5b6675">OPINION LEAVES</text>`;
      aes.forEach(e => {{
        if (ay[e.a] === undefined || oy[e.o] === undefined) return;
        const w = 0.8 + 4.4 * Math.abs(e.ae) / maxAE;
        const col = e.ae >= 0 ? 'rgba(231,111,81,0.50)' : 'rgba(42,157,143,0.50)';
        s += `<path d="${{curve(xA + 12, ay[e.a], xO - 12, oy[e.o])}}" fill="none" stroke="${{col}}" stroke-width="${{w}}" data-tip="${{e.a}} to ${{e.o}}: mean AE ${{e.ae.toFixed(2)}}"/>`;
      }});
      mods.forEach(e => {{
        if (fy[e.f] === undefined || ay[e.a] === undefined) return;
        const sig = isFinite(e.q) && e.q < 0.05;
        const w = 0.8 + 4.4 * Math.abs(e.est) / maxB;
        const op = sig ? 0.85 : 0.38;
        const col = e.est >= 0 ? `rgba(214,69,61,${{op}})` : `rgba(15,157,99,${{op}})`;
        s += `<path d="${{curve(xF + 12, fy[e.f], xA - 12, ay[e.a])}}" fill="none" stroke="${{col}}" stroke-width="${{w}}" ${{sig ? '' : 'stroke-dasharray="5 4"'}} data-tip="${{e.f}} x ${{e.a}}: b=${{e.est.toFixed(2)}}, q=${{isFinite(e.q) ? e.q.toFixed(3) : 'n/a'}}"/>`;
      }});
      usedF.forEach(f => {{
        const sel = state.focus && state.focus.kind === 'f' && state.focus.id === f.id;
        const pooled = D.pooled[f.id] || {{}};
        const pcol = (pooled.est || 0) >= 0 ? '{PALETTE['red']}' : '{PALETTE['teal']}';
        s += `<g class="mp-node" data-kind="f" data-id="${{f.id}}" style="cursor:pointer">`;
        s += `<circle cx="${{xF}}" cy="${{fy[f.id]}}" r="${{sel ? 7 : 5.4}}" fill="${{pcol}}" stroke="${{sel ? '{PALETTE['gold']}' : 'white'}}" stroke-width="${{sel ? 2.4 : 1.2}}"/>`;
        s += `<text x="${{xF - 11}}" y="${{fy[f.id] + 3.5}}" text-anchor="end" style="font-size:10.4px;fill:#27364a;font-weight:${{sel ? 800 : 600}}">${{TRUNC(f.label)}}<title>${{f.label}}</title></text></g>`;
      }});
      D.attacks.forEach(a => {{
        const sel = state.focus && state.focus.kind === 'a' && state.focus.id === a.id;
        s += `<g class="mp-node" data-kind="a" data-id="${{a.id}}" style="cursor:pointer">`;
        s += `<rect x="${{xA - 7}}" y="${{ay[a.id] - 7}}" width="14" height="14" rx="2.5" fill="{PALETTE['blue']}" stroke="${{sel ? '{PALETTE['gold']}' : 'white'}}" stroke-width="${{sel ? 2.4 : 1.2}}"/>`;
        s += `<text x="${{xA}}" y="${{ay[a.id] - 13}}" text-anchor="middle" style="font-size:10.4px;fill:#27364a;font-weight:${{sel ? 800 : 700}}">${{TRUNC(a.label)}}<title>${{a.label}}</title></text></g>`;
      }});
      D.opinions.forEach(o => {{
        const sel = state.focus && state.focus.kind === 'o' && state.focus.id === o.id;
        s += `<g class="mp-node" data-kind="o" data-id="${{o.id}}" style="cursor:pointer">`;
        s += `<circle cx="${{xO}}" cy="${{oy[o.id]}}" r="${{sel ? 7 : 5.4}}" fill="{PALETTE['teal']}" stroke="${{sel ? '{PALETTE['gold']}' : 'white'}}" stroke-width="${{sel ? 2.4 : 1.2}}"/>`;
        s += `<text x="${{xO + 11}}" y="${{oy[o.id] + 3.5}}" text-anchor="start" style="font-size:10.4px;fill:#27364a;font-weight:${{sel ? 800 : 600}}">${{TRUNC(o.label)}}<title>${{o.label}}</title></text></g>`;
      }});
      s += '</svg>';
      canvas.innerHTML = s;
      canvas.querySelectorAll('.mp-node').forEach(el => el.addEventListener('click', ev => {{
        ev.stopPropagation();
        const kind = el.dataset.kind, id = el.dataset.id;
        state.focus = (state.focus && state.focus.kind === kind && state.focus.id === id) ? null : {{kind, id}};
        render();
      }}));
      canvas.querySelectorAll('path[data-tip]').forEach(el => {{
        el.addEventListener('mouseenter', () => {{
          document.getElementById('mp-readout').textContent = el.dataset.tip;
        }});
      }});
      const sigCount = mods.filter(e => isFinite(e.q) && e.q < 0.05).length;
      document.getElementById('mp-readout').innerHTML =
        `<b>${{mods.length}}</b> moderation paths visible under q &le; ${{state.q.toFixed(2)}} and |b| &ge; ${{state.b}}; ` +
        `<b>${{sigCount}}</b> survive FDR at q&lt;0.05 (solid). Dashed = exploratory. ` +
        `Warm = amplifies attack effectivity, cool = protective. Hover a path for exact values.`;
    }}
    canvas.addEventListener('click', () => {{ state.focus = null; render(); }});
    document.getElementById('mp-q').addEventListener('input', e => {{
      state.q = parseFloat(e.target.value);
      document.getElementById('mp-q-val').textContent = state.q.toFixed(2);
      render();
    }});
    document.getElementById('mp-b').addEventListener('input', e => {{
      state.b = parseFloat(e.target.value);
      document.getElementById('mp-b-val').textContent = state.b.toFixed(1);
      render();
    }});
    document.querySelectorAll('#mp-sign .mp-btn').forEach(btn => btn.addEventListener('click', () => {{
      document.querySelectorAll('#mp-sign .mp-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.sign = btn.dataset.sign;
      render();
    }}));
    document.querySelectorAll('#mp-ae .mp-btn').forEach(btn => btn.addEventListener('click', () => {{
      document.querySelectorAll('#mp-ae .mp-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.ae = btn.dataset.ae === 'on';
      render();
    }}));
    document.getElementById('mp-search').addEventListener('input', e => {{
      state.search = e.target.value.trim().toLowerCase();
      render();
    }});
    render();
  }})();
  </script>
</div>"""


def _html_supplementary_analyses(
    long_df: pd.DataFrame,
    mantel_pairs_df: pd.DataFrame,
    mantel_meta: Dict[str, Any],
) -> str:
    """Supplementary battery: (S1) profile-space distance vs moderation-pattern
    distance with Mantel permutation test; (S2) baseline extremity vs movement;
    (S3) AE by attack complexity tier; (S4) confidence calibration."""
    figs: List[str] = []

    def _fig_html(fig: go.Figure) -> str:
        return fig.to_html(include_plotlyjs=False, full_html=False, config=PLOTLY_CONFIG)

    if not mantel_pairs_df.empty:
        x = mantel_pairs_df["feature_distance"].to_numpy(dtype=float)
        y = mantel_pairs_df["moderation_distance"].to_numpy(dtype=float)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 50)
        r = mantel_meta.get("mantel_r", float("nan"))
        pval = mantel_meta.get("mantel_p", float("nan"))
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram2dContour(
            x=x, y=y, ncontours=7, colorscale="Blues", showscale=False,
            opacity=0.5, hoverinfo="skip"))
        fig1.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, color=PALETTE["blue"], opacity=0.5, line=dict(color="white", width=0.4)),
            customdata=mantel_pairs_df[["profile_a", "profile_b"]].to_numpy(),
            hovertemplate="%{customdata[0]} vs %{customdata[1]}<br>feature distance %{x:.2f}<br>moderation distance %{y:.2f}<extra></extra>",
            name="profile pairs", showlegend=False))
        fig1.add_trace(go.Scatter(
            x=xs, y=np.polyval(coef, xs), mode="lines",
            line=dict(color=PALETTE["orange"], width=2.6), name="OLS fit", showlegend=False))
        fig1.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False, align="left",
            text=(f"Mantel r = {r:.3f}<br>permutation p = {pval:.4f}"
                  f"<br>{mantel_meta.get('n_pairs', '?')} pairs, {mantel_meta.get('n_permutations', '?')} permutations"),
            font=dict(size=12), bgcolor="rgba(255,255,255,0.75)", bordercolor="#c9d2df", borderwidth=1)
        fig1.update_layout(
            title="S1. Profile-configuration distance vs susceptibility-pattern distance (Mantel permutation test)",
            xaxis_title="Pairwise distance in z-scored profile feature space",
            yaxis_title="Pairwise distance between per-task AE patterns")
        figs.append(_fig_html(_apply_style(fig1, height=560)))

    work = long_df.dropna(subset=["adversarial_effectivity"]).copy() if "adversarial_effectivity" in long_df.columns else pd.DataFrame()

    if not work.empty and "baseline_score" in work.columns:
        bx = work["baseline_score"].abs().to_numpy(dtype=float)
        by = work["adversarial_effectivity"].to_numpy(dtype=float)
        q = np.polyfit(bx, by, 2)
        xs = np.linspace(0, float(bx.max()), 60)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=bx, y=by, mode="markers",
            marker=dict(size=4, color=PALETTE["teal"], opacity=0.35),
            hoverinfo="skip", showlegend=False))
        fig2.add_trace(go.Scatter(
            x=xs, y=np.polyval(q, xs), mode="lines",
            line=dict(color=PALETTE["red"], width=2.6), name="quadratic fit", showlegend=False))
        try:
            from scipy import stats as sps
            rr, pp = sps.pearsonr(bx, by)
            fig2.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", xanchor="right", showarrow=False,
                                text=f"Pearson r = {rr:.3f}, p = {pp:.2e}",
                                font=dict(size=12), bgcolor="rgba(255,255,255,0.75)",
                                bordercolor="#c9d2df", borderwidth=1)
        except Exception:
            pass
        fig2.update_layout(
            title="S2. Prior conviction vs achieved movement (ceiling and resistance check)",
            xaxis_title="|baseline score| (prior conviction strength)",
            yaxis_title="AE (movement toward goal)")
        figs.append(_fig_html(_apply_style(fig2, height=480)))

    if not work.empty and "attack_complexity_tier" in work.columns and work["attack_complexity_tier"].astype(str).str.len().gt(0).any():
        tiers = sorted(t for t in work["attack_complexity_tier"].dropna().astype(str).unique() if t)
        fig3 = go.Figure()
        palette = px.colors.qualitative.Bold
        for i, t in enumerate(tiers):
            vals = work.loc[work["attack_complexity_tier"].astype(str) == t, "adversarial_effectivity"]
            fig3.add_trace(go.Violin(
                y=vals, x=[t] * len(vals), points="all", jitter=0.32, pointpos=-0.4,
                marker=dict(size=3, opacity=0.4, color=palette[i % len(palette)]),
                line=dict(color=palette[i % len(palette)]),
                meanline_visible=True, showlegend=False, name=t))
        fig3.update_layout(
            title="S3. AE by attack complexity tier (structural mechanism validation)",
            yaxis_title="AE per scenario", xaxis_title="Complexity tier")
        figs.append(_fig_html(_apply_style(fig3, height=470)))

    if not work.empty and "post_confidence" in work.columns and work["post_confidence"].notna().any():
        fig4 = go.Figure(go.Scatter(
            x=work["post_confidence"], y=work["adversarial_effectivity"], mode="markers",
            marker=dict(size=4, color=PALETTE["amber"], opacity=0.4), showlegend=False))
        fig4.update_layout(
            title="S4. Post-elicitation confidence vs movement (calibration)",
            xaxis_title="agent confidence", yaxis_title="AE")
        figs.append(_fig_html(_apply_style(fig4, height=420)))

    intro = """
  <div style="font-size:0.82rem;line-height:1.6;color:var(--c-muted,#5b6675);max-width:980px;margin-bottom:12px">
    Reviewer-oriented robustness battery. S1 tests whether the geometry of the profile configuration space
    is informative for susceptibility: if profiles that are close in feature space show similar
    per-task AE patterns, the moderation surface is smooth and interpolation (k-NN estimation,
    embedding-based generalization) is justified. S2 checks the expected resistance gradient from prior
    conviction. S3 validates the structural complexity-tier ordering. Distances and tests are computed
    at the profile level; the Mantel permutation respects the non-independence of pairwise distances.
  </div>"""
    if not figs:
        return "<p>Supplementary inputs unavailable for this run.</p>"
    return f"<div id='supp-root'>{intro}{''.join(f'<div style=margin-bottom:14px>{f}</div>' for f in figs)}</div>"


def _strip_dashes(text: str) -> str:
    """Remove em and en dashes from utils-visible markup (style guide: none anywhere)."""
    return (
        str(text)
        .replace(" — ", ", ")
        .replace("—", "-")
        .replace(" – ", ", ")
        .replace("–", "-")
    )


# Light-chrome hexes used by the embedded tab fragments, mapped to themeable
# CSS variables (with the original light value as fallback). The swap is only
# applied OUTSIDE <script> blocks so Plotly JSON and JS template literals are
# never touched; JS-injected DOM is themed by the dark-override stylesheet in
# the shell instead.
_THEME_HEX_MAP: List[Tuple[str, str]] = [
    ("#f7faff", "var(--c-card,#f7faff)"),
    ("#f0f5ff", "var(--c-card,#f0f5ff)"),
    ("#f5f8ff", "var(--c-card,#f5f8ff)"),
    ("#f8fbff", "var(--c-soft,#f8fbff)"),
    ("#fbfcfe", "var(--c-panel,#fbfcfe)"),
    ("#fcfdff", "var(--c-panel,#fcfdff)"),
    ("#f6f9ff", "var(--c-soft,#f6f9ff)"),
    ("#f9fbff", "var(--c-soft,#f9fbff)"),
    ("#eef5ff", "var(--c-soft,#eef5ff)"),
    ("#f4f7ff", "var(--c-soft,#f4f7ff)"),
    ("#e8edf5", "var(--c-track,#e8edf5)"),
    ("#e2eaf5", "var(--c-line,#e2eaf5)"),
    ("#e3e8f0", "var(--c-line,#e3e8f0)"),
    ("#dbe3ef", "var(--c-line,#dbe3ef)"),
    ("#c8d7ec", "var(--c-line2,#c8d7ec)"),
    ("#bdd0ea", "var(--c-line2,#bdd0ea)"),
    ("#c9d2df", "var(--c-line2,#c9d2df)"),
    ("#d8dfe9", "var(--c-line,#d8dfe9)"),
    ("#ead2cc", "var(--c-line2,#ead2cc)"),
    ("#f0f0f5", "var(--c-track,#f0f0f5)"),
    ("background:#fff;", "background:var(--c-sheet,#fff);"),
    ("background:#fff}", "background:var(--c-sheet,#fff)}"),
    ("background:#fff\"", "background:var(--c-sheet,#fff)\""),
    ("background:#ffffff", "background:var(--c-sheet,#ffffff)"),
    ("background:linear-gradient(180deg,#ffffff 0%,#fbfdff 100%)", "background:var(--c-panel,#fbfdff)"),
    ("background:linear-gradient(135deg,#f8fbff 0%,#eef5ff 100%)", "background:var(--c-soft,#f0f5ff)"),
    ("background:rgba(255,255,255,0.55)", "background:var(--c-soft,rgba(255,255,255,0.55))"),
    ("background:rgba(255,255,255,0.7)", "background:var(--c-soft,rgba(255,255,255,0.7))"),
    ("#14213d", "var(--c-ink,#14213d)"),
    ("#0f2240", "var(--c-ink,#0f2240)"),
    ("#27364a", "var(--c-ink,#27364a)"),
    ("#41506a", "var(--c-muted,#41506a)"),
    ("#4a5d7a", "var(--c-muted,#4a5d7a)"),
    ("#5b6675", "var(--c-muted,#5b6675)"),
]

_SCRIPT_SPLIT_RE = re.compile(r"(<script\b.*?</script>)", re.DOTALL | re.IGNORECASE)


def _themeify_fragment(html: str) -> str:
    """Swap light-chrome colors for CSS variables outside script blocks."""
    parts = _SCRIPT_SPLIT_RE.split(html)
    out: List[str] = []
    for part in parts:
        if part.lower().startswith("<script"):
            out.append(part)
            continue
        for old, new in _THEME_HEX_MAP:
            part = part.replace(old, new)
        out.append(part)
    return "".join(out)


def _render_dashboard_html(
    run_id: str,
    summary_cards: Dict[str, Any],
    figure_divs: List[Tuple[str, str]],
    notes: List[str],
) -> str:
    plotly_js = get_plotlyjs()
    summary_cards = {k: _strip_dashes(v) for k, v in summary_cards.items()}
    figure_divs = [(_strip_dashes(t), _themeify_fragment(_strip_dashes(h))) for t, h in figure_divs]
    notes = [_strip_dashes(n) for n in notes]
    design_badge = f"{summary_cards.get('Attack Vectors', 'n/a')} attacks x {summary_cards.get('Opinion Leaves', 'n/a')} opinions x {summary_cards.get('Profiles', 'n/a')} profiles"

    cards_html = "".join(
        f"<div class='kpi'><div class='kpi-k'>{k}</div><div class='kpi-v'>{v}</div></div>"
        for k, v in summary_cards.items()
    )

    CATEGORIES = [
        ("Overview",          ["Key Findings"]),
        ("Ontologies",        ["Ontology Explorer", "Semantic Embedding Space"]),
        ("Factorial Space",   ["Factorial 3D Surface", "Factorial Heat + Contour"]),
        ("Moderation",        ["Moderation Paths", "Moderation Scan Heatmap", "SEM Heatmap", "Moderator Forest", "Hierarchical Importance", "Permutation Importance (FDR)", "BCa Coefficient CIs"]),
        ("Estimation",        ["Conditional Susceptibility Estimator", "Model Ladder", "Task Reliability Surface", "Bootstrap Rank Stability", "Bayesian Rank CIs"]),
        ("Profiles",          ["Susceptibility Map", "Profile Heatmap", "Profile Feature Network", "Profile Network Explorer"]),
        ("Variance",          ["Multilevel ICC Decomposition"]),
        ("Raw Data",          ["Distribution by Opinion Leaf", "Distribution by Attack Vector", "Score Trajectory"]),
        ("Supplementary",     ["Supplementary Analyses"]),
        ("Diagnostics",       ["Audit & Robustness"]),
    ]

    tab_index = {title: idx for idx, (title, _) in enumerate(figure_divs)}
    categorised: set = {n for _, ns in CATEGORIES for n in ns}

    nav_groups = []
    for cat_label, tab_names in CATEGORIES:
        btns = [f"<button class='nav-item' data-tab='tab-{tab_index[n]}'><span class='nav-tick'></span>{n}</button>"
                for n in tab_names if n in tab_index]
        if btns:
            nav_groups.append(
                f"<div class='nav-group'><div class='nav-label'>{cat_label}</div>"
                f"{''.join(btns)}</div>")

    extra = [f"<button class='nav-item' data-tab='tab-{tab_index[t]}'><span class='nav-tick'></span>{t}</button>"
             for t in tab_index if t not in categorised]
    if extra:
        nav_groups.append(
            f"<div class='nav-group'><div class='nav-label'>Other</div>{''.join(extra)}</div>")

    panels = "".join(
        f"<section id='tab-{i}' class='tab-panel{' active' if i == 0 else ''}' data-title='{t}'>"
        f"<div class='panel-crumb'><span>{run_id.upper()}</span><span class='crumb-sep'>/</span><span id='crumb-{i}'>{t}</span></div>{h}</section>"
        for i, (t, h) in enumerate(figure_divs)
    )
    notes_html = "".join(f"<li>{n}</li>" for n in notes)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{run_id} | Cognitive Susceptibility Console</title>
<style>
:root{{
  --bg:#f2f4f8;--bg2:#e9edf3;--panel:#ffffff;--panel2:#f7f9fc;--line:#d8dfe9;--line2:#c9d2df;
  --ink:#171c24;--muted:#5b6675;--accent:#2563eb;--accent-soft:rgba(37,99,235,0.10);
  --good:#0f9d63;--bad:#d6453d;--amber:#b97f10;
  --sidebar-bg:#fbfcfe;--topbar-bg:rgba(255,255,255,0.92);
  --shadow:0 1px 2px rgba(15,23,42,0.05),0 8px 24px rgba(15,23,42,0.06);
  --mono:"IBM Plex Mono","SF Mono",Menlo,monospace;
}}
html[data-theme="dark"]{{
  --bg:#0b0e14;--bg2:#0e1219;--panel:#11151d;--panel2:#161b25;--line:#222a37;--line2:#2c3645;
  --ink:#e6ebf4;--muted:#8a96a8;--accent:#4c9aff;--accent-soft:rgba(76,154,255,0.13);
  --good:#3dd68c;--bad:#f0574f;--amber:#f2b350;
  --sidebar-bg:#0d1117;--topbar-bg:rgba(13,17,23,0.92);
  --shadow:0 1px 2px rgba(0,0,0,0.4),0 10px 30px rgba(0,0,0,0.35);
  --c-card:#141a24;--c-panel:#11151d;--c-sheet:#0f141c;--c-soft:#1a2230;
  --c-line:#232c3b;--c-line2:#2e3a4d;--c-track:#1d2532;
  --c-ink:#e6ebf4;--c-muted:#8c99ab;
}}
*{{box-sizing:border-box}}
html,body{{margin:0;min-height:100vh}}
body{{background:var(--bg);color:var(--ink);font-family:Inter,"IBM Plex Sans","Segoe UI",sans-serif;
  font-size:14px;transition:background 0.25s ease,color 0.25s ease}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;opacity:0.5;
  background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);
  background-size:42px 42px;mask-image:radial-gradient(ellipse at 30% 0%,rgba(0,0,0,0.35),transparent 60%)}}

/* layout */
.app{{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:100vh}}
.app.nav-collapsed{{grid-template-columns:0 minmax(0,1fr)}}
.app.nav-collapsed .sidebar{{width:0;min-width:0;padding:0;border-right:none;overflow:hidden}}
.sidebar{{background:var(--sidebar-bg);border-right:1px solid var(--line);position:sticky;top:0;
  height:100vh;overflow-y:auto;overflow-x:hidden;padding:14px 10px 20px;z-index:30;transition:width .2s}}
.brand{{padding:6px 8px 14px;border-bottom:1px solid var(--line);margin-bottom:10px}}
.brand .b-run{{font:700 11px var(--mono);letter-spacing:0.14em;color:var(--accent);text-transform:uppercase}}
.brand .b-name{{font-weight:800;font-size:0.95rem;margin-top:3px;letter-spacing:0.01em}}
.brand .b-sub{{font-size:0.70rem;color:var(--muted);margin-top:3px;line-height:1.45}}
.nav-group{{margin-bottom:12px}}
.nav-label{{font:700 9.5px var(--mono);letter-spacing:0.16em;text-transform:uppercase;color:var(--muted);
  padding:4px 8px 5px}}
.nav-item{{display:flex;align-items:center;gap:8px;width:100%;text-align:left;background:none;border:none;
  border-left:2px solid transparent;color:var(--muted);padding:6px 8px;font-size:0.80rem;font-weight:600;
  cursor:pointer;border-radius:0 4px 4px 0;font-family:inherit}}
.nav-item:hover{{color:var(--ink);background:var(--accent-soft)}}
.nav-item.active{{color:var(--accent);border-left-color:var(--accent);background:var(--accent-soft)}}
.nav-tick{{width:5px;height:5px;border-radius:1px;background:currentColor;opacity:0.55;flex:0 0 auto}}

/* main column */
.main{{min-width:0;display:flex;flex-direction:column}}
.topbar{{position:sticky;top:0;z-index:25;display:flex;align-items:center;gap:12px;
  padding:10px 18px;background:var(--topbar-bg);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}}
.tb-burger{{background:var(--panel2);border:1px solid var(--line);color:var(--muted);width:30px;height:30px;
  border-radius:4px;cursor:pointer;font-size:0.95rem;line-height:1}}
.tb-burger:hover{{color:var(--ink);border-color:var(--line2)}}
.tb-title{{font-weight:800;font-size:0.95rem;letter-spacing:0.01em;white-space:nowrap}}
.tb-badge{{font:600 10.5px var(--mono);color:var(--muted);border:1px solid var(--line);border-radius:3px;
  padding:3px 8px;letter-spacing:0.05em;white-space:nowrap}}
.tb-spacer{{flex:1}}
.tb-btn{{display:inline-flex;align-items:center;gap:6px;background:var(--panel2);border:1px solid var(--line);
  color:var(--muted);border-radius:4px;padding:5px 11px;font-size:0.76rem;font-weight:700;cursor:pointer;font-family:inherit}}
.tb-btn:hover{{color:var(--ink);border-color:var(--line2)}}
.tb-btn.on{{color:var(--accent);border-color:var(--accent);background:var(--accent-soft)}}

/* KPI drawer */
.kpis{{display:none;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:8px;
  padding:12px 18px;border-bottom:1px solid var(--line);background:var(--bg2)}}
.kpis.open{{display:grid}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:9px 11px}}
.kpi-k{{font:700 9px var(--mono);letter-spacing:0.12em;text-transform:uppercase;color:var(--muted)}}
.kpi-v{{font:700 1.04rem var(--mono);margin-top:3px;font-variant-numeric:tabular-nums}}

/* content panels */
.content{{padding:16px 18px 40px;min-width:0}}
.tab-panel{{display:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:14px 16px 18px;box-shadow:var(--shadow);animation:fadeUp .18s ease}}
.tab-panel.active{{display:block}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:none}}}}
.panel-crumb{{display:flex;align-items:center;gap:7px;font:600 10.5px var(--mono);letter-spacing:0.10em;
  text-transform:uppercase;color:var(--muted);margin:0 0 12px;padding-bottom:9px;border-bottom:1px solid var(--line)}}
.panel-crumb .crumb-sep{{opacity:0.45}}
.panel-crumb span:last-child{{color:var(--accent)}}
.tab-panel .js-plotly-plot,.tab-panel .plotly-graph-div{{width:100%!important}}

/* notes slide-over */
.notes-pane{{position:fixed;top:0;right:-430px;width:412px;max-width:92vw;height:100vh;z-index:60;
  background:var(--panel);border-left:1px solid var(--line);box-shadow:-18px 0 50px rgba(0,0,0,0.28);
  transition:right .22s ease;display:flex;flex-direction:column}}
.notes-pane.open{{right:0}}
.notes-head{{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--line)}}
.notes-head h3{{margin:0;font-size:0.88rem;letter-spacing:0.02em}}
.notes-body{{padding:12px 18px 30px;overflow-y:auto}}
.notes-body li{{margin:0 0 10px;color:var(--muted);font-size:0.79rem;line-height:1.55}}
.notes-body li b{{color:var(--ink)}}
.notes-close{{background:none;border:1px solid var(--line);color:var(--muted);border-radius:4px;
  width:26px;height:26px;cursor:pointer;font-size:0.85rem}}
.scrim{{position:fixed;inset:0;background:rgba(0,0,0,0.35);z-index:55;opacity:0;pointer-events:none;transition:opacity .2s}}
.scrim.on{{opacity:1;pointer-events:auto}}

@media(max-width:980px){{
  .app{{grid-template-columns:0 1fr}}
  .sidebar{{position:fixed;left:-260px;width:248px;transition:left .2s}}
  .app.nav-open .sidebar{{left:0;display:block}}
}}

/* ── dark-theme propagation into JS-injected tab content ─────────────────── */
html[data-theme="dark"] .tab-panel{{color:var(--ink)}}
html[data-theme="dark"] :is(#ontx-inspector,#ontx-results,#ontx-status,#ontx-compare,#ontx-highlights,#ontx-chipline,#ontx-focus) :is(div,span,strong){{color:var(--ink)!important}}
html[data-theme="dark"] :is(#ontx-inspector,#ontx-results,#ontx-compare,#ontx-highlights,#ontx-focus) :is(div,a)[style*="background"]{{background:var(--c-card)!important;border-color:var(--c-line)!important}}
html[data-theme="dark"] #ontx-root .ontx-sub,html[data-theme="dark"] #ontx-root .ontx-note,html[data-theme="dark"] #ontx-root .ontx-status{{color:var(--c-muted)!important}}
html[data-theme="dark"] #ontx-canvas-wrap{{background:linear-gradient(180deg,#0e131b 0%,#10161f 100%)!important}}
html[data-theme="dark"] #ontx-root :is(.ontx-compare-card,.ontx-metric,.ontx-banner,.ontx-canvas-head,.ontx-card,.ontx-panel,.ontx-canvas-card){{background:var(--c-card)!important;border-color:var(--c-line)!important}}
html[data-theme="dark"] #ontx-root .ontx-metric .k{{color:var(--c-muted)!important}}
html[data-theme="dark"] #ontx-root .ontx-metric .v{{color:var(--ink)!important}}
html[data-theme="dark"] #ontx-root .ontx-canvas-head strong,html[data-theme="dark"] #ontx-root .ontx-canvas-head span{{color:var(--ink)!important}}
html[data-theme="dark"] #ontx-svg text{{fill:#dbe5f2!important;stroke:#0e131b!important}}
html[data-theme="dark"] #mp-canvas{{background:linear-gradient(180deg,#0e131b 0%,#10161f 100%)!important}}
html[data-theme="dark"] #mp-canvas svg text{{fill:#dbe5f2!important}}
html[data-theme="dark"] #cse-root :is(#cse-grid,#cse-contrib,#cse-gauge-wrap,#cse-gauge-text,#cse-task-summary) :is(td,th,span,div){{color:var(--ink)}}
html[data-theme="dark"] #cse-root #cse-grid th,html[data-theme="dark"] #cse-root #cse-grid td:first-child{{color:var(--ink)!important}}
html[data-theme="dark"] #cse-root select,html[data-theme="dark"] #ontx-root select,html[data-theme="dark"] #ontx-root input[type="text"]{{background:var(--c-card)!important;color:var(--ink)!important;border-color:var(--c-line)!important}}
html[data-theme="dark"] .tab-panel table{{color:var(--ink)}}
html[data-theme="dark"] #kf-root .kf-card{{background:linear-gradient(160deg,rgba(76,154,255,0.08),rgba(61,214,140,0.05))}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="app" id="app">
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="b-run">{run_id}</div>
      <div class="b-name">Susceptibility Console</div>
      <div class="b-sub">Ontology-constrained cognitive-manipulation simulation, conditional moderation analytics</div>
    </div>
    {''.join(nav_groups)}
  </aside>
  <div class="main">
    <header class="topbar">
      <button class="tb-burger" id="tb-burger" title="Toggle navigation">&#9776;</button>
      <div class="tb-title">Cognitive Susceptibility Console</div>
      <div class="tb-badge">{design_badge}</div>
      <div class="tb-spacer"></div>
      <button class="tb-btn" id="tb-kpis">Metrics</button>
      <button class="tb-btn" id="tb-notes">Notes</button>
      <button class="tb-btn" id="tb-theme" title="Toggle light / dark"><span id="tb-theme-icon">🌑</span></button>
    </header>
    <div class="kpis" id="kpis">{cards_html}</div>
    <main class="content">
      {panels}
    </main>
  </div>
</div>
<div class="scrim" id="scrim"></div>
<aside class="notes-pane" id="notes-pane">
  <div class="notes-head"><h3>Methodological Notes</h3><button class="notes-close" id="notes-close">&times;</button></div>
  <div class="notes-body"><ul style="padding-left:18px;margin:0">{notes_html}</ul></div>
</aside>
<script>
(function(){{
const btns=Array.from(document.querySelectorAll('.nav-item'));
const pans=Array.from(document.querySelectorAll('.tab-panel'));
const themedAt={{}};

function plotlyPatch(){{
  const dark=document.documentElement.dataset.theme==='dark';
  const ink=dark?'#e6ebf4':'#171c24';
  const grid=dark?'rgba(148,166,196,0.16)':'rgba(70,90,120,0.16)';
  const plotBg=dark?'rgba(255,255,255,0.015)':'rgba(20,40,80,0.02)';
  return {{
    'paper_bgcolor':'rgba(0,0,0,0)','plot_bgcolor':plotBg,'font.color':ink,
    'xaxis.gridcolor':grid,'yaxis.gridcolor':grid,
    'xaxis.zerolinecolor':grid,'yaxis.zerolinecolor':grid,
    'xaxis.linecolor':grid,'yaxis.linecolor':grid,
    'legend.bgcolor':'rgba(0,0,0,0)',
    'polar.bgcolor':'rgba(0,0,0,0)',
    'scene.bgcolor':'rgba(0,0,0,0)',
    'scene.xaxis.gridcolor':grid,'scene.yaxis.gridcolor':grid,'scene.zaxis.gridcolor':grid,
    'scene.xaxis.color':ink,'scene.yaxis.color':ink,'scene.zaxis.color':ink
  }};
}}
function themePanelPlots(panel){{
  const theme=document.documentElement.dataset.theme;
  panel.querySelectorAll('.js-plotly-plot').forEach(gd=>{{
    try{{
      Plotly.Plots.resize(gd);
      if(themedAt[gd.id]!==theme){{
        Plotly.relayout(gd,plotlyPatch());
        themedAt[gd.id]=theme;
      }}
    }}catch(e){{}}
  }});
}}
function activate(id){{
  btns.forEach(b=>b.classList.toggle('active',b.dataset.tab===id));
  pans.forEach(p=>p.classList.toggle('active',p.id===id));
  const panel=document.getElementById(id);
  if(panel){{requestAnimationFrame(()=>requestAnimationFrame(()=>themePanelPlots(panel)));}}
  try{{localStorage.setItem('dash-tab',id);}}catch(e){{}}
}}
btns.forEach(b=>b.addEventListener('click',()=>activate(b.dataset.tab)));

/* theme */
const themeBtn=document.getElementById('tb-theme');
function themeIcon(){{
  const el=document.getElementById('tb-theme-icon');
  if(el){{el.textContent=document.documentElement.dataset.theme==='dark'?'🌑':'🌕';}}
}}
function setTheme(t){{
  document.documentElement.dataset.theme=t;
  try{{localStorage.setItem('dash-theme',t);}}catch(e){{}}
  themeIcon();
  const active=document.querySelector('.tab-panel.active');
  if(active){{requestAnimationFrame(()=>themePanelPlots(active));}}
}}
themeBtn.addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
let savedTheme=null;try{{savedTheme=localStorage.getItem('dash-theme');}}catch(e){{}}
if(savedTheme==='light'||savedTheme==='dark')document.documentElement.dataset.theme=savedTheme;
themeIcon();

/* KPI drawer */
const kpiBtn=document.getElementById('tb-kpis');
const kpis=document.getElementById('kpis');
kpiBtn.addEventListener('click',()=>{{
  kpis.classList.toggle('open');
  kpiBtn.classList.toggle('on',kpis.classList.contains('open'));
}});

/* notes slide-over */
const notesPane=document.getElementById('notes-pane');
const scrim=document.getElementById('scrim');
const notesBtn=document.getElementById('tb-notes');
function notesOpen(on){{
  notesPane.classList.toggle('open',on);
  scrim.classList.toggle('on',on);
  notesBtn.classList.toggle('on',on);
}}
notesBtn.addEventListener('click',()=>notesOpen(!notesPane.classList.contains('open')));
document.getElementById('notes-close').addEventListener('click',()=>notesOpen(false));
scrim.addEventListener('click',()=>notesOpen(false));

/* sidebar toggle: a single hidden-state class drives both breakpoints */
const app=document.getElementById('app');
document.getElementById('tb-burger').addEventListener('click',()=>{{
  const mobile=window.matchMedia('(max-width:980px)').matches;
  if(mobile){{
    app.classList.toggle('nav-open');
  }}else{{
    app.classList.toggle('nav-collapsed');
  }}
  const active=document.querySelector('.tab-panel.active');
  if(active){{setTimeout(()=>themePanelPlots(active),230);}}
}});

window.addEventListener('resize',()=>{{
  const active=document.querySelector('.tab-panel.active');
  if(active)themePanelPlots(active);
}});

let savedTab=null;try{{savedTab=localStorage.getItem('dash-tab');}}catch(e){{}}
if(savedTab&&document.getElementById(savedTab)){{activate(savedTab);}}
else if(btns.length){{activate(btns[0].dataset.tab);}}
}})();
</script>
</body>
</html>""".strip()


# ─── profile feature network ──────────────────────────────────────────────────

_NETWORK_METRIC_LABELS = {
    "eigenvector_centrality": "Eigenvector Centrality",
    "betweenness_centrality": "Betweenness Centrality",
    "degree_centrality": "Degree Centrality",
    "closeness_centrality": "Closeness Centrality",
    "pagerank": "PageRank",
    "clustering_coefficient": "Clustering Coefficient",
    "strength": "Weighted Degree",
    "participation_coefficient": "Participation Coefficient",
    "bridge_ratio": "Bridge Ratio",
    "within_module_zscore": "Within-Module Z",
    "positive_strength": "Positive-Correlation Strength",
    "negative_strength": "Negative-Correlation Strength",
    "same_family_strength_share": "Within-Family Strength Share",
    "k_core": "K-Core Index",
    "community_size": "Community Size",
}

_ONTOLOGY_GROUP_COLORS = [
    "#1d4e89", "#2a9d8f", "#e76f51", "#c89b3c", "#9b59b6",
    "#e74c3c", "#27ae60", "#2980b9", "#f39c12", "#16a085",
    "#8e44ad", "#d35400", "#2c3e50", "#c0392b", "#1abc9c",
    "#7f8c8d", "#6c5ce7", "#fd79a8", "#00b894", "#a29bfe",
]


def _fig_profile_network(
    centrality_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    global_metrics: Dict[str, Any],
    default_metric: str = "eigenvector_centrality",
) -> go.Figure:
    """Interactive Plotly network of profile feature correlations.

    Nodes = profile features, sized/annotated by centrality metric,
    colored by ontology group. Edges = Spearman correlations (|rho| >= threshold).
    Dropdown buttons switch node sizing between 6 centrality metrics.
    """
    if centrality_df.empty or layout_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No network data available (run stage 06 first)",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color=PALETTE["muted"]),
        )
        fig.update_layout(
            title="Profile Feature Correlation Network",
            plot_bgcolor=PALETTE["panel"], paper_bgcolor=PALETTE["panel"],
        )
        return fig

    merged = centrality_df.merge(layout_df, on="term", how="left").dropna(subset=["x", "y"])
    if merged.empty:
        merged = centrality_df.copy()
        rng = np.random.default_rng(42)
        merged["x"] = rng.uniform(-1, 1, len(merged))
        merged["y"] = rng.uniform(-1, 1, len(merged))

    # Normalise each metric to [8, 40] for node size
    def _norm_size(series: pd.Series, lo: float = 8.0, hi: float = 40.0) -> np.ndarray:
        vals = series.fillna(0.0).values.astype(float)
        mn, mx = vals.min(), vals.max()
        if mx - mn < 1e-10:
            return np.full(len(vals), (lo + hi) / 2)
        return lo + (vals - mn) / (mx - mn) * (hi - lo)

    metric_cols = list(_NETWORK_METRIC_LABELS.keys())
    size_arrays: Dict[str, np.ndarray] = {}
    for m in metric_cols:
        if m in merged.columns:
            size_arrays[m] = _norm_size(merged[m])
        else:
            size_arrays[m] = np.full(len(merged), 14.0)

    # Ontology group color mapping
    groups = sorted(merged["ontology_group"].unique())
    group_color_map: Dict[str, str] = {
        g: _ONTOLOGY_GROUP_COLORS[i % len(_ONTOLOGY_GROUP_COLORS)]
        for i, g in enumerate(groups)
    }
    node_colors = merged["ontology_group"].map(group_color_map).fillna("#7f8c8d").tolist()

    # Build edge trace (all edges in one scatter using None separators)
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_hover: List[str] = []
    edge_widths: List[float] = []

    pos_map = {row["term"]: (row["x"], row["y"]) for _, row in merged.iterrows()}

    if not edge_df.empty:
        for _, erow in edge_df.iterrows():
            src, tgt = erow.get("source"), erow.get("target")
            if src not in pos_map or tgt not in pos_map:
                continue
            x0, y0 = pos_map[src]
            x1, y1 = pos_map[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            rho = float(erow.get("rho", 0.0))
            edge_widths.append(abs(rho) * 4.0)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="rgba(100,130,160,0.30)"),
        hoverinfo="skip",
        showlegend=False,
        name="correlations",
    )

    # Build node traces: one per ontology group for legend
    default_sizes = size_arrays[default_metric]
    node_traces: List[go.Scatter] = []
    for grp in groups:
        mask = merged["ontology_group"] == grp
        gdf = merged[mask]
        gsizes = default_sizes[mask.values]
        gcolor = group_color_map[grp]

        hover_texts = [
            (
                f"<b>{row['label']}</b><br>"
                f"Group: {row['ontology_group']}<br>"
                f"Eigenvector: {row.get('eigenvector_centrality', 0):.4f}<br>"
                f"Betweenness: {row.get('betweenness_centrality', 0):.4f}<br>"
                f"Degree: {row.get('degree_centrality', 0):.4f}<br>"
                f"Strength: {row.get('strength', 0):.4f}<br>"
                f"Participation: {row.get('participation_coefficient', 0):.4f}<br>"
                f"Bridge Ratio: {row.get('bridge_ratio', 0):.4f}<br>"
                f"PageRank: {row.get('pagerank', 0):.4f}<br>"
                f"Clustering: {row.get('clustering_coefficient', 0):.4f}<br>"
                f"Community: {int(row.get('community', -1))}"
            )
            for _, row in gdf.iterrows()
        ]

        node_traces.append(
            go.Scatter(
                x=gdf["x"].tolist(),
                y=gdf["y"].tolist(),
                mode="markers+text",
                marker=dict(
                    size=gsizes.tolist(),
                    color=gcolor,
                    opacity=0.85,
                    line=dict(width=1.0, color="white"),
                ),
                text=[_wrap_label(r["label"], width=12) for _, r in gdf.iterrows()],
                textposition="top center",
                textfont=dict(size=7, color="#2c3e50"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
                name=grp,
                legendgroup=grp,
                showlegend=True,
            )
        )

    # updatemenus: buttons to switch sizing metric
    # Each button updates marker.size for all node traces
    n_node_traces = len(node_traces)
    # Traces: [edge_trace] + node_traces (indices 1..n_node_traces)
    button_list: List[Dict[str, Any]] = []
    for metric in metric_cols:
        if metric not in _NETWORK_METRIC_LABELS:
            continue
        sizes_for_metric = size_arrays[metric]
        # Split sizes by group (must match node_traces order)
        new_sizes_per_trace: List[List[float]] = []
        for grp in groups:
            mask = (merged["ontology_group"] == grp).values
            new_sizes_per_trace.append(sizes_for_metric[mask].tolist())

        # build args: first element updates all node traces
        args_update: Dict[str, Any] = {}
        for trace_i in range(n_node_traces):
            args_update[f"marker.size[{trace_i + 1}]"] = new_sizes_per_trace[trace_i]
        # Plotly buttons use list-form: [{trace_property: new_values}, ...]
        # Simpler: pass a list of dicts, one per trace
        sizes_list = [{}] + [{"marker.size": s} for s in new_sizes_per_trace]

        button_list.append(
            dict(
                label=_NETWORK_METRIC_LABELS[metric],
                method="update",
                args=[
                    sizes_list,
                    {"title": f"Profile Feature Correlation Network — node size: {_NETWORK_METRIC_LABELS[metric]}"},
                ],
            )
        )

    updatemenus = [
        dict(
            buttons=button_list,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.02, yanchor="bottom",
            bgcolor=PALETTE["panel"],
            bordercolor=PALETTE["line"],
            font=dict(size=11),
        )
    ]

    # Global metrics annotation
    gm = global_metrics or {}
    gm_lines = [
        f"<b>Network Metrics</b>",
        f"Nodes: {gm.get('n_nodes', '—')}",
        f"Edges: {gm.get('n_edges', '—')} (|ρ|≥{gm.get('corr_threshold', 0.15)})",
        f"Density: {gm.get('density', 0):.3f}",
        f"Avg Clustering: {gm.get('avg_clustering', 0):.3f}",
        f"Transitivity: {gm.get('transitivity', 0):.3f}",
        f"Communities: {gm.get('n_communities', '—')}",
        f"Modularity: {gm.get('modularity_score', 0):.3f}",
    ]

    fig = go.Figure(data=[edge_trace, *node_traces])
    fig.update_layout(
        title=dict(
            text=f"Profile Feature Correlation Network — node size: {_NETWORK_METRIC_LABELS.get(default_metric, default_metric)}",
            font=dict(size=15, color=PALETTE["ink"]),
        ),
        showlegend=True,
        legend=dict(
            title="Ontology Group",
            font=dict(size=10),
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=PALETTE["line"],
            borderwidth=1,
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x"),
        plot_bgcolor=PALETTE["panel"],
        paper_bgcolor=PALETTE["panel"],
        margin=dict(l=10, r=200, t=60, b=30),
        updatemenus=updatemenus,
        annotations=[
            dict(
                text="<br>".join(gm_lines),
                xref="paper", yref="paper",
                x=1.18, y=0.5,
                xanchor="left", yanchor="middle",
                showarrow=False,
                font=dict(size=10, color=PALETTE["ink"]),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=PALETTE["line"],
                borderwidth=1,
                borderpad=8,
                align="left",
            )
        ],
        height=700,
    )

    return fig


def _fig_task_reliability(task_summary_df: pd.DataFrame) -> go.Figure:
    if task_summary_df.empty:
        return go.Figure().add_annotation(text="Task-level reliability data unavailable", showarrow=False)

    work = task_summary_df.copy()
    work["attack_label"] = work["attack_leaf"].apply(lambda x: _unique_display_map(work["attack_leaf"].astype(str).tolist()).get(str(x), _leaf(x)))
    work["opinion_label"] = work["opinion_leaf"].apply(lambda x: _unique_display_map(work["opinion_leaf"].astype(str).tolist()).get(str(x), _leaf(x)))
    reliability = work.pivot_table(index="attack_label", columns="opinion_label", values="reliability_weight", aggfunc="mean")
    cv_mse = work.pivot_table(index="attack_label", columns="opinion_label", values="cv_mse", aggfunc="mean")

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.12,
        subplot_titles=["Reliability weight", "Cross-validated MSE"],
    )
    fig.add_trace(
        go.Heatmap(
            z=reliability.values,
            x=[_wrap_label(c, 18) for c in reliability.columns],
            y=[_wrap_label(i, 18) for i in reliability.index],
            colorscale="Blues",
            colorbar=dict(title="Weight", x=0.44),
            hovertemplate="Attack: %{y}<br>Opinion: %{x}<br>Weight: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=cv_mse.values,
            x=[_wrap_label(c, 18) for c in cv_mse.columns],
            y=[_wrap_label(i, 18) for i in cv_mse.index],
            colorscale="YlOrRd",
            colorbar=dict(title="CV-MSE", x=1.02),
            hovertemplate="Attack: %{y}<br>Opinion: %{x}<br>CV-MSE: %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Task Reliability Surface",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        margin=dict(l=20, r=30, t=70, b=30),
        height=620,
    )
    return fig


def _fig_multilevel_icc_sunburst(icc: Dict[str, Any]) -> go.Figure:
    """Three-level ICC decomposition: profile / attack / opinion / residual."""
    if not icc:
        return go.Figure().add_annotation(text="Multilevel ICC unavailable", showarrow=False)
    parts = [
        ("Profile (between)", float(icc.get("icc_profile") or 0.0), "#1d4e89"),
        ("Attack (between)", float(icc.get("icc_attack") or 0.0), "#e76f51"),
        ("Opinion (between)", float(icc.get("icc_opinion") or 0.0), "#2a9d8f"),
        ("Within-residual", float(icc.get("icc_residual") or 0.0), "#94a3b8"),
    ]
    labels = ["Total"] + [p[0] for p in parts]
    parents = [""] + ["Total"] * len(parts)
    values = [sum(max(0.0, p[1]) for p in parts)] + [max(0.0, p[1]) for p in parts]
    colors = ["#0f1d3a"] + [p[2] for p in parts]
    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            insidetextorientation="radial",
            marker=dict(colors=colors),
            hovertemplate="%{label}<br>Variance share: %{value:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Multilevel ICC Decomposition (n_obs={int(icc.get('n_obs', 0))}, method={icc.get('method', '—')})",
        paper_bgcolor=PALETTE["panel"],
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    return fig


def _fig_permutation_forest(perm_df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """FDR-controlled permutation feature importance forest."""
    if perm_df.empty or not {"term", "observed_importance", "p_value", "q_value"}.issubset(perm_df.columns):
        return go.Figure().add_annotation(text="Permutation importance unavailable", showarrow=False)
    work = perm_df.copy().sort_values("observed_importance", ascending=False).head(top_n)
    work["color"] = np.where(work["q_value"] <= 0.05, "#2a9d8f",
                             np.where(work["q_value"] <= 0.10, "#e9c46a", "#cbd5e1"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=work["observed_importance"],
        y=work["term"].apply(lambda t: _pretty(t)[:55]),
        orientation="h",
        marker_color=work["color"],
        text=[f"q={q:.3f}" for q in work["q_value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>|corr|: %{x:.3f}<br>q-value: %{customdata:.4f}<extra></extra>",
        customdata=work["q_value"],
    ))
    fig.update_layout(
        title="Permutation Feature Importance (BH-FDR)",
        xaxis_title="|cluster-mean correlation|",
        yaxis_autorange="reversed",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        margin=dict(l=240, r=80, t=60, b=40),
        height=max(520, 22 * len(work) + 120),
    )
    return fig


def _fig_bca_coefficients(bca_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """BCa cluster-bootstrap coefficient CIs (forest)."""
    if bca_df.empty or not {"term", "estimate", "ci_low", "ci_high"}.issubset(bca_df.columns):
        return go.Figure().add_annotation(text="BCa coefficients unavailable", showarrow=False)
    work = bca_df.copy().reindex(
        bca_df["estimate"].abs().sort_values(ascending=False).index
    ).head(top_n)
    work["significant"] = (work["ci_low"] * work["ci_high"]) > 0
    colors = np.where(work["significant"], "#1d4e89", "#94a3b8")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=work["estimate"],
        y=work["term"].apply(lambda t: _pretty(t)[:55]),
        mode="markers",
        marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
        error_x=dict(
            type="data", symmetric=False,
            array=(work["ci_high"] - work["estimate"]).clip(lower=0),
            arrayminus=(work["estimate"] - work["ci_low"]).clip(lower=0),
            color="#4a5d7a", thickness=1.4, width=0,
        ),
        hovertemplate="<b>%{y}</b><br>β=%{x:.3f}<br>BCa 95%% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
        customdata=np.stack([work["ci_low"].to_numpy(), work["ci_high"].to_numpy()], axis=1),
    ))
    fig.add_vline(x=0, line=dict(color="#94a3b8", dash="dot"))
    fig.update_layout(
        title="BCa Cluster-Bootstrap Ridge Coefficients (95% CI)",
        xaxis_title="Standardised ridge coefficient",
        yaxis_autorange="reversed",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        margin=dict(l=240, r=40, t=60, b=40),
        height=max(520, 22 * len(work) + 120),
    )
    return fig


def _fig_rank_credible_intervals(rank_df: pd.DataFrame) -> go.Figure:
    if rank_df.empty or not {"profile_id", "mean_rank", "rank_low_95", "rank_high_95"}.issubset(rank_df.columns):
        return go.Figure().add_annotation(text="Bayesian rank intervals unavailable", showarrow=False)
    work = rank_df.sort_values("mean_rank").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=work["mean_rank"],
        y=work["profile_id"],
        mode="markers",
        marker=dict(size=9, color=work["mean_rank"], colorscale="Tealgrn", line=dict(color="white", width=1)),
        error_x=dict(
            type="data", symmetric=False,
            array=(work["rank_high_95"] - work["mean_rank"]).clip(lower=0),
            arrayminus=(work["mean_rank"] - work["rank_low_95"]).clip(lower=0),
            color="#4a5d7a", thickness=1.1, width=0,
        ),
        customdata=np.stack(
            [work["rank_low_95"].to_numpy(), work["rank_high_95"].to_numpy(),
             work.get("rank_sd", pd.Series([0]*len(work))).to_numpy(),
             work.get("top_decile_share", pd.Series([0]*len(work))).to_numpy()], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>Mean rank: %{x:.1f}<br>"
            "95% CI: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<br>"
            "SD: %{customdata[2]:.2f}<br>Top-decile share: %{customdata[3]:.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Bayesian Rank Stability (95% credible interval)",
        xaxis_title="Profile rank (1 = highest susceptibility)",
        yaxis_title="Profile",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        margin=dict(l=140, r=30, t=60, b=40),
        height=max(520, 18 * len(work) + 150),
    )
    return fig


def _fig_bootstrap_rank_stability(profile_index_df: pd.DataFrame) -> go.Figure:
    """Profile rank-stability dot plot with bootstrap CIs.

    Robust to partially missing columns: falls back from explicit rank CI
    columns to merged *_boot duplicates, then to a rank_sd-derived band, and
    finally to a CI-free dot ranking so the tab never renders empty.
    """
    if profile_index_df.empty or "profile_id" not in profile_index_df.columns:
        return go.Figure().add_annotation(text="Bootstrap rank intervals unavailable", showarrow=False)

    work = profile_index_df.copy()
    if "susceptibility_index_pct" not in work.columns:
        candidates = [c for c in ["eb_percentile", "observed_effectivity_pct"] if c in work.columns]
        if not candidates:
            return go.Figure().add_annotation(text="No susceptibility index column available", showarrow=False)
        work["susceptibility_index_pct"] = pd.to_numeric(work[candidates[0]], errors="coerce")

    for col in ["rank_ci_low", "rank_ci_high", "rank_sd"]:
        if col not in work.columns and f"{col}_boot" in work.columns:
            work[col] = work[f"{col}_boot"]
    work["rank_sd"] = pd.to_numeric(work.get("rank_sd"), errors="coerce")
    if "rank_ci_low" not in work.columns or work["rank_ci_low"].isna().all():
        sd = work["rank_sd"].fillna(work["rank_sd"].median() if work["rank_sd"].notna().any() else 0.0)
        n = max(len(work), 1)
        half_band = (sd / n * 100.0 * 1.96).clip(lower=0.0)
        work["rank_ci_low"] = (work["susceptibility_index_pct"] - half_band).clip(lower=0)
        work["rank_ci_high"] = (work["susceptibility_index_pct"] + half_band).clip(upper=100)
    work["rank_ci_low"] = pd.to_numeric(work["rank_ci_low"], errors="coerce").fillna(work["susceptibility_index_pct"])
    work["rank_ci_high"] = pd.to_numeric(work["rank_ci_high"], errors="coerce").fillna(work["susceptibility_index_pct"])

    work = work.sort_values("susceptibility_index_pct", ascending=False).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["susceptibility_index_pct"],
            y=work["profile_id"],
            mode="markers",
            marker=dict(
                size=np.clip(work["rank_sd"].fillna(3.0).to_numpy() * 2.0 + 9.0, 9.0, 21.0),
                color=work["susceptibility_index_pct"],
                colorscale="Tealgrn",
                line=dict(color="white", width=1),
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(work["rank_ci_high"] - work["susceptibility_index_pct"]).clip(lower=0),
                arrayminus=(work["susceptibility_index_pct"] - work["rank_ci_low"]).clip(lower=0),
                color="#4a5d7a",
                thickness=1.1,
                width=0,
            ),
            customdata=np.stack(
                [
                    work["rank_ci_low"].to_numpy(),
                    work["rank_ci_high"].to_numpy(),
                    work["rank_sd"].fillna(np.nan).to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Rank: %{x:.1f}<br>"
                "90% CI: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<br>"
                "SD: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Bootstrap Rank Stability",
        xaxis_title="Conditional susceptibility percentile",
        yaxis_title="Profile",
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor="#f4f7ff",
        font_family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        margin=dict(l=140, r=30, t=60, b=40),
        height=max(520, 18 * len(work) + 150),
    )
    return fig


def _html_quality_robustness(
    quality_diagnostics: Dict[str, Any],
    icc_data: Dict[str, Any],
    ridge_summary: Dict[str, Any],
    rf_summary: Dict[str, Any],
    enet_summary: Dict[str, Any],
    ladder_meta: Dict[str, Any] | None = None,
) -> str:
    def _pct(value: Any) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except Exception:
            return "n/a"

    def _num(value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "n/a"

    abs_icc = icc_data.get("abs_delta_score", {}).get("icc1") if isinstance(icc_data, dict) else None
    cards = [
        ("Baseline fallback", _pct(quality_diagnostics.get("baseline_fallback_used_rate"))),
        ("Post fallback", _pct(quality_diagnostics.get("post_fallback_used_rate"))),
        ("Attack heuristic pass", _pct(quality_diagnostics.get("attack_heuristic_pass_rate"))),
        ("Post heuristic pass", _pct(quality_diagnostics.get("post_heuristic_pass_rate"))),
        ("Mean attack realism", _num(quality_diagnostics.get("mean_attack_realism_score"))),
        ("Mean post plausibility", _num(quality_diagnostics.get("mean_post_plausibility_score"))),
        ("ICC(1) |delta|", _num(abs_icc)),
        ("Ridge CV-R2 (panel)", _num(ridge_summary.get("cv_r2"))),
        ("RF OOB R2 (panel)", _num(rf_summary.get("oob_r2"))),
        ("EN selected", str(enet_summary.get("n_features_selected", "n/a"))),
    ]
    card_html = "".join(
        f"<div class='diag-card'><div class='diag-k'>{k}</div><div class='diag-v'>{v}</div></div>"
        for k, v in cards
    )

    ladder_html = ""
    models = (ladder_meta or {}).get("models") or []
    if models:
        rows = "".join(
            f"<tr><td>{m.get('model')}</td>"
            f"<td class='num'>{float(m.get('cv_r2_mean', float('nan'))):.4f}</td>"
            f"<td class='num'>{float(m.get('cv_r2_sd', 0.0) or 0.0):.4f}</td>"
            f"<td class='num'>{float(m.get('delta_r2_vs_context_baseline', float('nan'))):+.4f}</td></tr>"
            for m in models
        )
        ladder_html = f"""
  <div class="diag-panel">
    <h3>Scenario-Level Model Ladder (GroupKFold by profile)</h3>
    <p style="margin-bottom:8px">Outcome: {(ladder_meta or {}).get('outcome', 'adversarial_effectivity')} on {(ladder_meta or {}).get('n_obs', '?')} scenarios from {(ladder_meta or {}).get('n_profiles', '?')} profiles. CV-R2 measures prediction on profiles the model never saw. The increment column is the profile contribution beyond attack and opinion context plus the baseline position.</p>
    <table class="diag-table">
      <tr><th>Model</th><th>CV-R2</th><th>SD</th><th>Increment vs context+baseline</th></tr>
      {rows}
    </table>
  </div>"""

    return f"""
<div id="diag-root">
  <style>
    #diag-root {{display:grid;gap:14px}}
    #diag-root .diag-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
    #diag-root .diag-card {{background:#f5f8ff;border:1px solid #dbe3ef;border-radius:6px;padding:12px 13px}}
    #diag-root .diag-k {{font-size:0.70rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:{PALETTE['muted']};margin-bottom:4px}}
    #diag-root .diag-v {{font-size:1.14rem;font-weight:800;color:{PALETTE['ink']};font-family:"IBM Plex Mono",monospace}}
    #diag-root .diag-panel {{background:#fbfcfe;border:1px solid #dbe3ef;border-radius:6px;padding:14px 16px;line-height:1.65;color:{PALETTE['ink']}}}
    #diag-root .diag-panel h3 {{margin:0 0 8px;color:{PALETTE['blue']};font-size:0.95rem}}
    #diag-root .diag-panel p {{margin:0}}
    #diag-root .diag-table {{width:100%;border-collapse:collapse;font-size:0.80rem}}
    #diag-root .diag-table th {{text-align:left;font:700 10px "IBM Plex Mono",monospace;letter-spacing:0.08em;text-transform:uppercase;color:{PALETTE['muted']};border-bottom:1.5px solid #c9d2df;padding:5px 8px}}
    #diag-root .diag-table td {{padding:5px 8px;border-bottom:1px solid #e3e8f0}}
    #diag-root .diag-table td.num {{font-family:"IBM Plex Mono",monospace}}
  </style>
  <div class="diag-grid">{card_html}</div>
  {ladder_html}
  <div class="diag-panel">
    <h3>Execution Integrity</h3>
    <p>Fallback rates near zero and high review scores indicate the elicitation stages executed under normal API-backed conditions. High fallback rates or zero-valued review scores would mean downstream coefficients should be treated as pipeline diagnostics only.</p>
  </div>
  <div class="diag-panel">
    <h3>How To Read The Signal</h3>
    <p>The profile-aggregate panel metrics (ridge CV-R2, RF OOB R2, elastic-net selection) and the scenario-level GroupKFold ladder answer the same question at two resolutions: do profile features carry predictive signal for adversarial effectivity beyond attack and opinion context? Convergent positive increments across both levels support profile-level claims; divergence or near-zero increments mean context dominates and profile claims should stay descriptive.</p>
  </div>
</div>"""


def _html_profile_network_explorer(
    centrality_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    global_metrics: Dict[str, Any],
) -> str:
    if centrality_df.empty or layout_df.empty:
        return "<p>Profile network explorer unavailable.</p>"

    merged = centrality_df.merge(layout_df, on="term", how="left").dropna(subset=["x", "y"]).copy()
    if merged.empty:
        return "<p>Profile network explorer unavailable.</p>"

    merged["ontology_group"] = merged["ontology_group"].fillna("Other")
    if "ontology_family" not in merged.columns:
        merged["ontology_family"] = merged["ontology_group"].apply(_network_ontology_family)
    else:
        merged["ontology_family"] = merged["ontology_family"].fillna(merged["ontology_group"].apply(_network_ontology_family))
    if "feature_type" not in merged.columns:
        merged["feature_type"] = merged["term"].apply(_network_feature_type)
    else:
        merged["feature_type"] = merged["feature_type"].fillna(merged["term"].apply(_network_feature_type))

    metric_labels = dict(_NETWORK_METRIC_LABELS)
    groups = sorted(merged["ontology_group"].dropna().astype(str).unique().tolist())
    families = sorted(merged["ontology_family"].dropna().astype(str).unique().tolist())
    feature_types = sorted(merged["feature_type"].dropna().astype(str).unique().tolist())
    communities = sorted(int(v) for v in merged["community"].dropna().unique().tolist())
    group_color_map = {g: _ONTOLOGY_GROUP_COLORS[i % len(_ONTOLOGY_GROUP_COLORS)] for i, g in enumerate(groups)}

    payload = {
        "nodes": merged.to_dict(orient="records"),
        "edges": edge_df.to_dict(orient="records"),
        "global_metrics": global_metrics or {},
        "metric_labels": metric_labels,
        "group_colors": group_color_map,
        "groups": groups,
        "families": families,
        "feature_types": feature_types,
        "communities": communities,
    }
    payload_json = json.dumps(payload)
    html = """
<div id="netx-root">
  <style>
    #netx-root { display:grid; gap:16px; }
    #netx-root * { box-sizing:border-box; }
    #netx-root .netx-shell {
      display:grid;
      grid-template-columns: minmax(270px,300px) minmax(0,1fr) minmax(270px,300px);
      gap:14px;
      align-items:start;
    }
    #netx-root .netx-panel {
      background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
      border:1px solid #dbe3ef; border-radius:18px;
      box-shadow:0 10px 28px rgba(15,34,64,0.08);
      padding:14px 15px;
    }
    #netx-root .netx-title {
      margin:0 0 5px; color:__BLUE__;
      font-size:0.93rem; font-weight:800; letter-spacing:0.01em;
    }
    #netx-root .netx-sub { color:__MUTED__; font-size:0.72rem; line-height:1.55; margin:0 0 9px; }
    #netx-root .netx-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
    #netx-root .netx-field label {
      display:block; margin-bottom:3px; font-size:0.64rem; font-weight:800;
      letter-spacing:0.05em; text-transform:uppercase; color:__MUTED__;
    }
    #netx-root select,
    #netx-root input[type="search"],
    #netx-root input[type="range"] {
      width:100%; border:1px solid #cfd9ea; border-radius:10px;
      background:#fff; padding:7px 9px; font:inherit; color:__INK__;
    }
    #netx-root input[type="range"] { padding:0; }
    #netx-root .netx-readout { color:__MUTED__; font-size:0.68rem; line-height:1.5; margin-top:4px; }
    #netx-root .netx-presets { display:flex; flex-wrap:wrap; gap:6px; margin-top:11px; }
    #netx-root .netx-preset, #netx-root .netx-toolbtn, #netx-root .netx-simtoggle {
      border:none; border-radius:999px; padding:6px 11px; cursor:pointer;
      font-size:0.70rem; font-weight:800; letter-spacing:0.01em;
      transition:transform 0.12s, box-shadow 0.12s, background 0.12s;
    }
    #netx-root .netx-preset {
      background:#edf4ff; color:__BLUE__;
      box-shadow:inset 0 0 0 1px rgba(29,78,137,0.10);
    }
    #netx-root .netx-toolbtn {
      background:#f4f7ff; color:__INK__;
      box-shadow:inset 0 0 0 1px rgba(15,34,64,0.08);
    }
    #netx-root .netx-simtoggle {
      background:#fff7ee; color:#c45c1a;
      box-shadow:inset 0 0 0 1px rgba(196,92,26,0.15);
    }
    #netx-root .netx-simtoggle.running {
      background:#fff0e0; color:#c45c1a;
      animation: netx-pulse 1.2s ease-in-out infinite;
    }
    @keyframes netx-pulse { 0%,100%{opacity:1} 50%{opacity:0.65} }
    #netx-root .netx-preset:hover, #netx-root .netx-toolbtn:hover, #netx-root .netx-simtoggle:hover {
      transform:translateY(-1px); box-shadow:0 6px 16px rgba(15,34,64,0.10);
    }
    #netx-root .netx-global {
      display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:11px;
    }
    #netx-root .netx-metric {
      background:#f5f8ff; border:1px solid #dbe3ef; border-radius:13px; padding:9px 10px;
    }
    #netx-root .netx-metric .k {
      color:__MUTED__; font-size:0.63rem; font-weight:800;
      letter-spacing:0.04em; text-transform:uppercase;
    }
    #netx-root .netx-metric .v { color:__INK__; font-size:0.97rem; font-weight:800; margin-top:2px; }
    #netx-root .netx-stagebar {
      display:flex; justify-content:space-between; gap:10px;
      align-items:flex-start; margin-bottom:8px;
    }
    #netx-root .netx-actions { display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end; }
    #netx-root .netx-summary { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:9px; }
    #netx-root .netx-pill {
      padding:4px 9px; border-radius:999px;
      background:#eff5ff; color:__BLUE__;
      font-size:0.67rem; font-weight:800; letter-spacing:0.02em;
    }
    #netx-root .netx-pill.warn { background:#fff3e0; color:#c45c1a; }
    #netx-root .netx-canvas-wrap {
      position:relative; min-height:780px;
      border:1px solid #dbe3ef; border-radius:18px; overflow:hidden;
      background:
        radial-gradient(circle at 16% 18%,rgba(42,157,143,0.08),transparent 24%),
        radial-gradient(circle at 82% 14%,rgba(29,78,137,0.09),transparent 26%),
        linear-gradient(180deg,#fcfdff 0%,#f5f8ff 100%);
    }
    #netx-root .netx-canvas-wrap::before {
      content:""; position:absolute; inset:0; pointer-events:none;
      background-image:
        linear-gradient(rgba(15,34,64,0.035) 1px,transparent 1px),
        linear-gradient(90deg,rgba(15,34,64,0.035) 1px,transparent 1px);
      background-size:32px 32px;
    }
    #netx-root svg {
      width:100%; height:780px; display:block; position:relative; z-index:1;
      cursor:grab; utils-select:none; touch-action:none;
    }
    #netx-root svg.is-panning { cursor:grabbing; }
    #netx-root svg.lasso-mode { cursor:crosshair; }
    #netx-root .netx-node { cursor:pointer; }
    #netx-root .netx-label {
      font-size:10.5px; font-weight:700; fill:#23364f; pointer-events:none;
      paint-order:stroke; stroke:#ffffff; stroke-width:3px; stroke-linejoin:round;
    }
    #netx-root .netx-label.selected-label { font-size:11.5px; fill:__INK__; }
    #netx-root .netx-foot {
      margin-top:9px; color:__MUTED__; font-size:0.70rem; line-height:1.55;
    }
    #netx-root .netx-foot kbd {
      font:inherit; font-size:0.63rem; background:#edf4ff; color:__BLUE__;
      border-radius:4px; padding:1px 5px; font-weight:700;
    }
    /* Mini-map */
    #netx-minimap {
      position:absolute; bottom:12px; right:12px;
      width:160px; height:108px; border-radius:10px;
      border:1.5px solid #cfd9ea;
      box-shadow:0 4px 12px rgba(15,34,64,0.12);
      background:rgba(245,248,255,0.94);
      z-index:5; pointer-events:none; display:block;
    }
    /* Tooltip */
    #netx-tooltip {
      display:none; position:absolute; z-index:20;
      background:#ffffff; border:1.5px solid #dbe3ef; border-radius:12px;
      padding:10px 12px; box-shadow:0 8px 24px rgba(15,34,64,0.14);
      max-width:220px; pointer-events:none; font-size:0.75rem;
    }
    #netx-tooltip strong { display:block; font-size:0.82rem; color:__INK__; margin-bottom:3px; }
    #netx-tooltip .tt-group { color:__BLUE__; font-size:0.68rem; font-weight:700; margin-bottom:1px; }
    #netx-tooltip .tt-type { color:__MUTED__; font-size:0.65rem; margin-bottom:5px; }
    #netx-tooltip .tt-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:4px; }
    #netx-tooltip .tt-kv .tt-k { color:__MUTED__; font-size:0.60rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; }
    #netx-tooltip .tt-kv .tt-v { color:__INK__; font-size:0.78rem; font-weight:800; }
    /* Path highlight */
    .netx-path-node { filter:drop-shadow(0 0 5px rgba(200,155,60,0.9)); }
    /* List items */
    #netx-root .netx-list { display:grid; gap:7px; }
    #netx-root .netx-item {
      padding:9px 10px; border-radius:12px;
      background:#f8fbff; border:1px solid #dbe3ef;
    }
    #netx-root .netx-item strong { display:block; color:__INK__; font-size:0.77rem; line-height:1.4; }
    #netx-root .netx-item span { display:block; color:__MUTED__; font-size:0.68rem; line-height:1.45; margin-top:2px; }
    #netx-root .netx-item.path-item { border-color:#c89b3c; background:#fffbf0; }
    #netx-root .netx-local-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    /* Sim controls row */
    #netx-root .netx-sim-row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-top:10px; }
    /* Section separators in right panel */
    #netx-root .netx-section { margin-top:12px; padding-top:12px; border-top:1px solid #eaf0f8; }
    @media (max-width:1360px) {
      #netx-root .netx-shell { grid-template-columns:1fr; }
      #netx-root .netx-global { grid-template-columns:repeat(3,minmax(0,1fr)); }
    }
    @media (max-width:760px) {
      #netx-root .netx-grid, #netx-root .netx-global, #netx-root .netx-local-grid { grid-template-columns:1fr; }
      #netx-root .netx-stagebar { flex-direction:column; }
      #netx-root .netx-actions { justify-content:flex-start; }
    }
  </style>

  <div class="netx-shell">

    <!-- ═══ LEFT PANEL: Controls ════════════════════════════════════════════ -->
    <aside class="netx-panel">
      <div class="netx-title">Network Controls</div>
      <p class="netx-sub">
        Correlation overlay on the mixed-type hierarchical profile panel.
        <b>Circles</b> = continuous features; <b>diamonds</b> = categorical/dummy.
        Colors encode ontology group; community hulls (toggleable) mark detected clusters.
      </p>
      <div class="netx-grid">
        <div class="netx-field">
          <label>Node Metric</label>
          <select id="netx-metric"></select>
        </div>
        <div class="netx-field">
          <label>Ontology Family</label>
          <select id="netx-family"></select>
        </div>
        <div class="netx-field">
          <label>Ontology Group</label>
          <select id="netx-group"></select>
        </div>
        <div class="netx-field">
          <label>Feature Type</label>
          <select id="netx-feature-type"></select>
        </div>
        <div class="netx-field">
          <label>Community</label>
          <select id="netx-community"></select>
        </div>
        <div class="netx-field">
          <label>Edge Sign</label>
          <select id="netx-sign">
            <option value="all">All edges</option>
            <option value="positive">Positive only</option>
            <option value="negative">Negative only</option>
          </select>
        </div>
        <div class="netx-field">
          <label>Focus Mode</label>
          <select id="netx-focus">
            <option value="all">Whole subgraph</option>
            <option value="ego1">Ego (1-hop)</option>
            <option value="ego2">Ego (2-hop)</option>
            <option value="community">Community</option>
          </select>
        </div>
        <div class="netx-field">
          <label>Label Density</label>
          <select id="netx-labels">
            <option value="hubs">Hubs + selection</option>
            <option value="all">All labels</option>
            <option value="none">No labels</option>
          </select>
        </div>
        <div class="netx-field" style="grid-column:1/-1">
          <label>Search Node</label>
          <input id="netx-search" type="search" placeholder="Feature label or term…"/>
        </div>
      </div>

      <div class="netx-field" style="margin-top:11px">
        <label>Min |ρ| Threshold</label>
        <input id="netx-threshold" type="range" min="0.15" max="0.80" step="0.01" value="__THRESHOLD__"/>
        <div class="netx-readout" id="netx-threshold-readout"></div>
      </div>

      <!-- Force simulation & view options -->
      <div class="netx-sim-row">
        <button class="netx-simtoggle" id="netx-force-btn" title="Physics-based force relaxation (Fruchterman-Reingold). Nodes repel; edges attract proportional to |ρ|.">⚡ Relax Layout</button>
        <button class="netx-toolbtn" id="netx-reset-sim-btn" title="Reset node positions to original layout">↺ Reset Positions</button>
      </div>
      <div class="netx-sim-row" style="margin-top:6px">
        <button class="netx-toolbtn" id="netx-hull-btn" title="Toggle community convex hulls">⬡ Hulls</button>
        <button class="netx-toolbtn" id="netx-path-btn" title="Path analysis: click to set source, then click target to trace shortest path">🔗 Path Mode</button>
        <button class="netx-toolbtn" id="netx-export-btn" title="Download current view as SVG">↓ SVG</button>
      </div>

      <div class="netx-presets" style="margin-top:10px">
        <button class="netx-preset" data-preset="all">Overview</button>
        <button class="netx-preset" data-preset="bridges">Bridge Lens</button>
        <button class="netx-preset" data-preset="hubs">Hub Lens</button>
        <button class="netx-preset" data-preset="bigfive">Big Five</button>
        <button class="netx-preset" data-preset="social">Social Context</button>
        <button class="netx-preset" data-preset="politics">Political Psych</button>
        <button class="netx-preset" data-preset="dummies">Dummy Features</button>
        <button class="netx-preset" data-preset="kcore">K-Core Shell</button>
      </div>

      <div class="netx-global" id="netx-global"></div>
    </aside>

    <!-- ═══ CENTER PANEL: Canvas ════════════════════════════════════════════ -->
    <section class="netx-panel" style="position:relative">
      <div class="netx-stagebar">
        <div>
          <div class="netx-title">Profile Network Explorer</div>
          <p class="netx-sub">
            <b>Shift+drag</b> for lasso multi-select · <b>Drag node</b> to pin position · <b>Dbl-click node</b> to unpin ·
            Path mode: set source then click target · <kbd>R</kbd> reset view · <kbd>H</kbd> hulls · <kbd>Esc</kbd> deselect
          </p>
        </div>
        <div class="netx-actions">
          <button class="netx-toolbtn" id="netx-zoom-in">＋</button>
          <button class="netx-toolbtn" id="netx-zoom-out">－</button>
          <button class="netx-toolbtn" id="netx-fit-btn">Fit</button>
          <button class="netx-toolbtn" id="netx-center-selected">Center</button>
          <button class="netx-toolbtn" id="netx-reset-view">Reset</button>
        </div>
      </div>
      <div class="netx-summary" id="netx-subgraph-summary"></div>
      <div class="netx-canvas-wrap" id="netx-canvas-wrap">
        <svg id="netx-svg" viewBox="0 0 1000 780" aria-label="Interactive profile feature network">
          <g id="netx-viewport">
            <g id="netx-hulls"></g>
            <g id="netx-edges"></g>
            <g id="netx-path-edges"></g>
            <g id="netx-nodes"></g>
            <g id="netx-labels"></g>
          </g>
          <rect id="netx-lasso-rect" x="0" y="0" width="0" height="0"
            fill="rgba(29,78,137,0.06)" stroke="#1d4e89" stroke-width="1.2"
            stroke-dasharray="5 3" rx="3" visibility="hidden"/>
        </svg>
        <canvas id="netx-minimap" width="160" height="108"></canvas>
        <div id="netx-tooltip"></div>
      </div>
      <div class="netx-foot">
        Drag background to pan · Scroll to zoom (centered on cursor) · Drag nodes to reposition (they stay pinned) ·
        Dbl-click to unpin · <kbd>Shift</kbd>+drag for lasso multi-select
      </div>
    </section>

    <!-- ═══ RIGHT PANEL: Inspection ════════════════════════════════════════ -->
    <aside class="netx-panel" style="display:grid;gap:12px">

      <!-- Selected node -->
      <div>
        <div class="netx-title">Selected Node</div>
        <div class="netx-list" id="netx-selected"></div>
      </div>

      <!-- Local metrics grid -->
      <div class="netx-section">
        <div class="netx-title">Local Metrics</div>
        <div class="netx-local-grid" id="netx-local-cards"></div>
      </div>

      <!-- Multi-selection analysis -->
      <div class="netx-section" id="netx-multisel-section" style="display:none">
        <div class="netx-title">Selection Analysis</div>
        <div id="netx-multisel-stats"></div>
      </div>

      <!-- Path analysis -->
      <div class="netx-section">
        <div class="netx-title">Path Analysis</div>
        <div id="netx-path-info" class="netx-readout"></div>
        <div class="netx-list" id="netx-path-list" style="margin-top:6px"></div>
      </div>

      <!-- Top nodes in view -->
      <div class="netx-section">
        <div class="netx-title">Top Nodes In View</div>
        <div class="netx-list" id="netx-ranking"></div>
      </div>

    </aside>
  </div><!-- end shell -->

  <script>
  (function() {
    /* ── DATA ─────────────────────────────────────────────────────────────── */
    const DATA = __PAYLOAD__;
    const WORLD_W = 1000;
    const WORLD_H = 780;
    const PADDING = 80;

    /* ── DOM REFS ────────────────────────────────────────────────────────── */
    const svg         = document.getElementById('netx-svg');
    const viewport    = document.getElementById('netx-viewport');
    const hullsLayer  = document.getElementById('netx-hulls');
    const edgesLayer  = document.getElementById('netx-edges');
    const pathEdgesLayer = document.getElementById('netx-path-edges');
    const nodesLayer  = document.getElementById('netx-nodes');
    const labelsLayer = document.getElementById('netx-labels');
    const lassoRect   = document.getElementById('netx-lasso-rect');
    const minimap     = document.getElementById('netx-minimap');
    const tooltip     = document.getElementById('netx-tooltip');
    const canvasWrap  = document.getElementById('netx-canvas-wrap');

    const metricSel      = document.getElementById('netx-metric');
    const familySel      = document.getElementById('netx-family');
    const groupSel       = document.getElementById('netx-group');
    const featureTypeSel = document.getElementById('netx-feature-type');
    const communitySel   = document.getElementById('netx-community');
    const signSel        = document.getElementById('netx-sign');
    const focusSel       = document.getElementById('netx-focus');
    const labelsSel      = document.getElementById('netx-labels');
    const searchInput    = document.getElementById('netx-search');
    const thresholdInput = document.getElementById('netx-threshold');
    const thresholdReadout = document.getElementById('netx-threshold-readout');
    const globalRoot     = document.getElementById('netx-global');
    const summaryRoot    = document.getElementById('netx-subgraph-summary');
    const selectedRoot   = document.getElementById('netx-selected');
    const localCardsRoot = document.getElementById('netx-local-cards');
    const multiSelSection = document.getElementById('netx-multisel-section');
    const multiSelStats  = document.getElementById('netx-multisel-stats');
    const pathInfoEl     = document.getElementById('netx-path-info');
    const pathListEl     = document.getElementById('netx-path-list');
    const rankingRoot    = document.getElementById('netx-ranking');
    const forceBtn       = document.getElementById('netx-force-btn');
    const hullBtn        = document.getElementById('netx-hull-btn');
    const pathBtn        = document.getElementById('netx-path-btn');

    /* ── HELPERS ─────────────────────────────────────────────────────────── */
    const num = (v, fallback=0) => { const p=Number(v); return Number.isFinite(p)?p:fallback; };
    const clamp = (v,lo,hi) => Math.min(hi, Math.max(lo, v));
    const safeText = v => String(v??'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
    const fmt4 = v => num(v).toFixed(4);
    const fmt3 = v => num(v).toFixed(3);
    const fmt2 = v => num(v).toFixed(2);

    /* ── DATA INIT ───────────────────────────────────────────────────────── */
    DATA.nodes = (DATA.nodes||[]).map(n => ({
      ...n,
      x: num(n.x), y: num(n.y),
      label: String(n.label||n.term||'Unknown'),
      ontology_group: String(n.ontology_group||'Other'),
      ontology_family: String(n.ontology_family||(String(n.ontology_group||'').split(':')[0]||'Other')),
      feature_type: String(n.feature_type||(String(n.term||'').startsWith('profile_cat__')?'Categorical dummy':'Continuous subscale')),
      community: Number.isFinite(Number(n.community)) ? Number(n.community) : -1,
    }));
    DATA.edges = (DATA.edges||[]).map(e => ({
      ...e, rho: num(e.rho), abs_rho: num(e.abs_rho, Math.abs(num(e.rho))),
    }));

    const nodeMap  = new Map(DATA.nodes.map(n => [n.term, n]));
    const adjacency = new Map(DATA.nodes.map(n => [n.term, []]));
    DATA.edges.forEach(e => {
      if (!adjacency.has(e.source)||!adjacency.has(e.target)) return;
      adjacency.get(e.source).push(e);
      adjacency.get(e.target).push({...e, source:e.target, target:e.source});
    });

    /* Store original positions for reset */
    DATA.nodes.forEach(n => { n._ox = n.x; n._oy = n.y; });

    /* ── COORDINATE NORMALISATION ────────────────────────────────────────── */
    function normaliseCoords() {
      const xs = DATA.nodes.map(n => n.x), ys = DATA.nodes.map(n => n.y);
      const minX=Math.min(...xs), maxX=Math.max(...xs);
      const minY=Math.min(...ys), maxY=Math.max(...ys);
      const spanX=Math.max(1e-6, maxX-minX), spanY=Math.max(1e-6, maxY-minY);
      DATA.nodes.forEach(n => {
        n.sx = PADDING + ((n.x-minX)/spanX)*(WORLD_W-2*PADDING);
        n.sy = PADDING + ((n.y-minY)/spanY)*(WORLD_H-2*PADDING);
      });
    }
    function resetOriginalPositions() {
      DATA.nodes.forEach(n => { n.x=n._ox; n.y=n._oy; });
      normaliseCoords();
    }

    /* ── K-CORE COMPUTATION ──────────────────────────────────────────────── */
    function computeCoreNumbers() {
      const active=new Map(DATA.nodes.map(n=>[n.term,true]));
      const degree=new Map(DATA.nodes.map(n=>[n.term,(adjacency.get(n.term)||[]).length]));
      const core=new Map(DATA.nodes.map(n=>[n.term,0]));
      let remaining=DATA.nodes.length, k=0;
      while(remaining>0){
        let progressed=false, changed=true;
        while(changed){
          changed=false;
          DATA.nodes.forEach(n=>{
            if(!active.get(n.term)) return;
            if(num(degree.get(n.term))<=k){
              active.set(n.term,false); remaining--; core.set(n.term,k);
              progressed=true; changed=true;
              (adjacency.get(n.term)||[]).forEach(e=>{
                if(active.get(e.target)) degree.set(e.target, num(degree.get(e.target))-1);
              });
            }
          });
        }
        if(!progressed) k++;
      }
      return core;
    }

    /* ── NETWORK METRIC DERIVATION ───────────────────────────────────────── */
    function deriveNetworkMetrics() {
      const commCounts=new Map();
      DATA.nodes.forEach(n => commCounts.set(n.community,(commCounts.get(n.community)||0)+1));
      const coreNums = computeCoreNumbers();
      const withinByComm = new Map();

      DATA.nodes.forEach(n => {
        const incident = adjacency.get(n.term)||[];
        let totalStr=0,posStr=0,negStr=0,posDeg=0,negDeg=0,sameFamStr=0;
        const commStr=new Map();
        incident.forEach(e=>{
          const w=num(e.abs_rho, Math.abs(num(e.rho)));
          const rho=num(e.rho);
          const nb=nodeMap.get(e.target);
          totalStr+=w;
          if(rho>=0){posDeg++;posStr+=w;}else{negDeg++;negStr+=w;}
          const nbComm=nb?nb.community:-1;
          commStr.set(nbComm,(commStr.get(nbComm)||0)+w);
          if(nb&&nb.ontology_family===n.ontology_family) sameFamStr+=w;
        });
        const withinStr=commStr.get(n.community)||0;
        withinByComm.set(n.community,[...( withinByComm.get(n.community)||[]),withinStr]);
        n.degree = Number.isFinite(Number(n.degree))?Number(n.degree):incident.length;
        n.community_size = Number.isFinite(Number(n.community_size))?Number(n.community_size):(commCounts.get(n.community)||0);
        n.strength = Number.isFinite(Number(n.strength))?Number(n.strength):totalStr;
        n.positive_degree = Number.isFinite(Number(n.positive_degree))?Number(n.positive_degree):posDeg;
        n.negative_degree = Number.isFinite(Number(n.negative_degree))?Number(n.negative_degree):negDeg;
        n.positive_strength = Number.isFinite(Number(n.positive_strength))?Number(n.positive_strength):posStr;
        n.negative_strength = Number.isFinite(Number(n.negative_strength))?Number(n.negative_strength):negStr;
        n.within_community_strength = Number.isFinite(Number(n.within_community_strength))?Number(n.within_community_strength):withinStr;
        const betw = Math.max(0,totalStr-withinStr);
        n.between_community_strength = Number.isFinite(Number(n.between_community_strength))?Number(n.between_community_strength):betw;
        n.participation_coefficient = Number.isFinite(Number(n.participation_coefficient))?Number(n.participation_coefficient):
          (totalStr>1e-12 ? 1-[...commStr.values()].reduce((a,v)=>a+(v/totalStr)**2,0) : 0);
        n.bridge_ratio = Number.isFinite(Number(n.bridge_ratio))?Number(n.bridge_ratio):(totalStr>1e-12?betw/totalStr:0);
        n.same_family_strength_share = Number.isFinite(Number(n.same_family_strength_share))?Number(n.same_family_strength_share):(totalStr>1e-12?sameFamStr/totalStr:0);
        n.signed_balance = Number.isFinite(Number(n.signed_balance))?Number(n.signed_balance):(totalStr>1e-12?(posStr-negStr)/totalStr:0);
        n.k_core = Number.isFinite(Number(n.k_core))?Number(n.k_core):num(coreNums.get(n.term));
      });

      DATA.nodes.forEach(n=>{
        const vals=withinByComm.get(n.community)||[];
        const mean=vals.length?vals.reduce((a,v)=>a+v,0)/vals.length:0;
        const sd=Math.sqrt(vals.length?vals.reduce((a,v)=>a+(v-mean)**2,0)/vals.length:0);
        n.within_module_zscore = Number.isFinite(Number(n.within_module_zscore))?Number(n.within_module_zscore):(sd>1e-12?(num(n.within_community_strength)-mean)/sd:0);
      });
    }

    /* ── CONTROLS POPULATION ─────────────────────────────────────────────── */
    function optMarkup(vals, fmt) {
      return vals.map(v=>`<option value="${safeText(v)}">${safeText(fmt(v))}</option>`).join('');
    }
    function populateControls() {
      metricSel.innerHTML = optMarkup(Object.keys(DATA.metric_labels||{}), k=>DATA.metric_labels[k]||k);
      familySel.innerHTML = `<option value="all">All families</option>${optMarkup(DATA.families||[],v=>v)}`;
      groupSel.innerHTML  = `<option value="all">All groups</option>${optMarkup(DATA.groups||[],v=>v)}`;
      featureTypeSel.innerHTML = `<option value="all">All types</option>${optMarkup(DATA.feature_types||[],v=>v)}`;
      communitySel.innerHTML = `<option value="all">All communities</option>${optMarkup((DATA.communities||[]).map(String),v=>'Community '+v)}`;
    }

    /* ── FORCE SIMULATION ────────────────────────────────────────────────── */
    const SIM = {
      running: false, alpha: 1.0, alphaDecay: 0.014,
      velocities: new Map(), pinned: new Set(), raf: null,
    };

    function simInit() {
      SIM.velocities = new Map(DATA.nodes.map(n=>[n.term,{vx:0,vy:0}]));
      SIM.alpha = 1.0;
    }

    function simStep() {
      const REPULSION = 2800;
      const ATTRACTION = 0.04;
      const DAMPING = 0.82;
      const thr = state.threshold;
      const activeEdges = DATA.edges.filter(e=>Math.abs(e.rho)>=thr);
      const nodes = DATA.nodes;

      for (let i=0; i<nodes.length; i++) {
        const ni = nodes[i];
        let fx=0, fy=0;
        for (let j=0; j<nodes.length; j++) {
          if (i===j) continue;
          const nj=nodes[j];
          const dx=ni.sx-nj.sx, dy=ni.sy-nj.sy;
          const dist2=dx*dx+dy*dy+1;
          const dist=Math.sqrt(dist2);
          const force=REPULSION/dist2*SIM.alpha;
          fx+=force*dx/dist; fy+=force*dy/dist;
        }
        const vel=SIM.velocities.get(ni.term);
        if(vel){vel.vx+=fx; vel.vy+=fy;}
      }

      activeEdges.forEach(e=>{
        const src=nodeMap.get(e.source), tgt=nodeMap.get(e.target);
        if(!src||!tgt) return;
        const dx=tgt.sx-src.sx, dy=tgt.sy-src.sy;
        const dist=Math.sqrt(dx*dx+dy*dy)+0.1;
        const force=ATTRACTION*Math.abs(e.rho)*dist*SIM.alpha;
        const sv=SIM.velocities.get(src.term), tv=SIM.velocities.get(tgt.term);
        if(sv){sv.vx+=force*dx/dist; sv.vy+=force*dy/dist;}
        if(tv){tv.vx-=force*dx/dist; tv.vy-=force*dy/dist;}
      });

      nodes.forEach(n=>{
        if(SIM.pinned.has(n.term)) return;
        const vel=SIM.velocities.get(n.term);
        if(!vel) return;
        vel.vx*=DAMPING; vel.vy*=DAMPING;
        n.sx=clamp(n.sx+vel.vx,30,WORLD_W-30);
        n.sy=clamp(n.sy+vel.vy,30,WORLD_H-30);
      });

      SIM.alpha=Math.max(0,SIM.alpha-SIM.alphaDecay);
      if(SIM.alpha<=0) stopSim();
    }

    function startSim() {
      if(!SIM.velocities.size) simInit();
      SIM.running=true;
      forceBtn.textContent='⚡ Stop Relax';
      forceBtn.classList.add('running');
      function tick(){
        if(!SIM.running) return;
        for(let i=0;i<4;i++) simStep();
        render();
        SIM.raf=requestAnimationFrame(tick);
      }
      SIM.raf=requestAnimationFrame(tick);
    }

    function stopSim() {
      SIM.running=false;
      if(SIM.raf) cancelAnimationFrame(SIM.raf);
      forceBtn.textContent='⚡ Relax Layout';
      forceBtn.classList.remove('running');
    }

    /* ── CONVEX HULL ─────────────────────────────────────────────────────── */
    function convexHull(pts) {
      if(pts.length<2) return pts;
      if(pts.length===2) return pts;
      const P=[...pts].sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
      const cross=(O,A,B)=>(A[0]-O[0])*(B[1]-O[1])-(A[1]-O[1])*(B[0]-O[0]);
      const lower=[], upper=[];
      for(const p of P){while(lower.length>=2&&cross(lower[lower.length-2],lower[lower.length-1],p)<=0)lower.pop();lower.push(p);}
      for(const p of [...P].reverse()){while(upper.length>=2&&cross(upper[upper.length-2],upper[upper.length-1],p)<=0)upper.pop();upper.push(p);}
      return lower.slice(0,-1).concat(upper.slice(0,-1));
    }

    /* ── PATH ANALYSIS (BFS) ─────────────────────────────────────────────── */
    const PATH = { mode: false, source: null, target: null, terms: [], edgeKeys: new Set() };

    function bfsPath(srcTerm, tgtTerm, nodeSet, thr) {
      const adj=new Map();
      DATA.edges.filter(e=>Math.abs(e.rho)>=thr&&nodeSet.has(e.source)&&nodeSet.has(e.target))
        .forEach(e=>{
          if(!adj.has(e.source)) adj.set(e.source,[]);
          if(!adj.has(e.target)) adj.set(e.target,[]);
          adj.get(e.source).push(e.target);
          adj.get(e.target).push(e.source);
        });
      const dist=new Map([[srcTerm,0]]);
      const prev=new Map([[srcTerm,null]]);
      const queue=[srcTerm];
      while(queue.length){
        const cur=queue.shift();
        if(cur===tgtTerm) break;
        for(const nb of (adj.get(cur)||[])){
          if(!dist.has(nb)){dist.set(nb,dist.get(cur)+1);prev.set(nb,cur);queue.push(nb);}
        }
      }
      if(!dist.has(tgtTerm)) return [];
      const path=[];
      let cur=tgtTerm;
      while(cur!==null){path.unshift(cur);cur=prev.get(cur);}
      return path;
    }

    function computePathEdgeKeys(pathTerms) {
      const keys=new Set();
      for(let i=0;i<pathTerms.length-1;i++){
        keys.add(pathTerms[i]+'||'+pathTerms[i+1]);
        keys.add(pathTerms[i+1]+'||'+pathTerms[i]);
      }
      return keys;
    }

    /* ── LASSO SELECTION ─────────────────────────────────────────────────── */
    const LASSO = { active:false, svgX0:0, svgY0:0, svgX1:0, svgY1:0 };

    function lassoStart(evt) {
      if(!evt.shiftKey) return;
      evt.preventDefault(); evt.stopPropagation();
      LASSO.active=true;
      const p=svgPoint(evt);
      LASSO.svgX0=LASSO.svgX1=p.x; LASSO.svgY0=LASSO.svgY1=p.y;
      updateLassoRect();
    }
    function lassoMove(evt) {
      if(!LASSO.active) return;
      const p=svgPoint(evt);
      LASSO.svgX1=p.x; LASSO.svgY1=p.y;
      updateLassoRect();
    }
    function lassoEnd() {
      if(!LASSO.active) return;
      LASSO.active=false;
      lassoRect.setAttribute('visibility','hidden');
      /* Convert SVG-space rect → world space; nodes in world space (sx,sy) */
      const wx0=(Math.min(LASSO.svgX0,LASSO.svgX1)-state.panX)/state.scale;
      const wy0=(Math.min(LASSO.svgY0,LASSO.svgY1)-state.panY)/state.scale;
      const wx1=(Math.max(LASSO.svgX0,LASSO.svgX1)-state.panX)/state.scale;
      const wy1=(Math.max(LASSO.svgY0,LASSO.svgY1)-state.panY)/state.scale;
      if(Math.abs(wx1-wx0)<5&&Math.abs(wy1-wy0)<5) return; /* too small, ignore */
      const enclosed=visibleNodes().filter(n=>n.sx>=wx0&&n.sx<=wx1&&n.sy>=wy0&&n.sy<=wy1);
      if(enclosed.length>0){
        state.selectedTerms=new Set(enclosed.map(n=>n.term));
        state.selectedTerm=enclosed[0].term;
      }
      render();
    }
    function updateLassoRect() {
      const x=Math.min(LASSO.svgX0,LASSO.svgX1);
      const y=Math.min(LASSO.svgY0,LASSO.svgY1);
      const w=Math.abs(LASSO.svgX1-LASSO.svgX0);
      const h=Math.abs(LASSO.svgY1-LASSO.svgY0);
      lassoRect.setAttribute('x',x); lassoRect.setAttribute('y',y);
      lassoRect.setAttribute('width',w); lassoRect.setAttribute('height',h);
      lassoRect.setAttribute('visibility',LASSO.active?'visible':'hidden');
    }

    /* ── TOOLTIP ─────────────────────────────────────────────────────────── */
    function showTooltip(node, evt) {
      const metric=state.metric;
      const mLabel=DATA.metric_labels[metric]||metric;
      tooltip.innerHTML=`
        <strong>${safeText(node.label)}</strong>
        <div class="tt-group">${safeText(node.ontology_group)}</div>
        <div class="tt-type">${safeText(node.feature_type)} · community ${node.community} · k-core ${num(node.k_core).toFixed(0)}</div>
        <div class="tt-grid">
          <div class="tt-kv"><div class="tt-k">${safeText(mLabel)}</div><div class="tt-v">${fmt4(node[metric])}</div></div>
          <div class="tt-kv"><div class="tt-k">Degree</div><div class="tt-v">${num(node.degree).toFixed(0)}</div></div>
          <div class="tt-kv"><div class="tt-k">Participation</div><div class="tt-v">${fmt3(node.participation_coefficient)}</div></div>
          <div class="tt-kv"><div class="tt-k">Bridge Ratio</div><div class="tt-v">${fmt3(node.bridge_ratio)}</div></div>
          <div class="tt-kv"><div class="tt-k">Within-mod Z</div><div class="tt-v">${fmt3(node.within_module_zscore)}</div></div>
          <div class="tt-kv"><div class="tt-k">Signed Bal.</div><div class="tt-v">${fmt3(node.signed_balance)}</div></div>
        </div>`;
      const rect=canvasWrap.getBoundingClientRect();
      let tx=evt.clientX-rect.left+14, ty=evt.clientY-rect.top+12;
      if(tx+230>rect.width) tx=evt.clientX-rect.left-230;
      if(ty+180>rect.height) ty=evt.clientY-rect.top-180;
      tooltip.style.cssText=`display:block;position:absolute;left:${tx}px;top:${ty}px`;
    }
    function hideTooltip(){ tooltip.style.display='none'; }

    /* ── MINI-MAP ────────────────────────────────────────────────────────── */
    function renderMinimap(nodes, edges) {
      const ctx=minimap.getContext('2d');
      const W=minimap.width, H=minimap.height;
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle='rgba(245,248,255,0.94)'; ctx.fillRect(0,0,W,H);
      ctx.strokeStyle='rgba(200,210,230,0.7)'; ctx.lineWidth=0.5;
      ctx.strokeRect(0,0,W,H);
      const sx=W/WORLD_W, sy=H/WORLD_H;
      /* edges */
      ctx.lineWidth=0.3; ctx.strokeStyle='rgba(29,78,137,0.12)';
      edges.forEach(e=>{
        const src=nodeMap.get(e.source), tgt=nodeMap.get(e.target);
        if(!src||!tgt) return;
        ctx.beginPath(); ctx.moveTo(src.sx*sx,src.sy*sy); ctx.lineTo(tgt.sx*sx,tgt.sy*sy); ctx.stroke();
      });
      /* nodes */
      nodes.forEach(n=>{
        const color=DATA.group_colors[n.ontology_group]||'#7f8c8d';
        ctx.fillStyle=color; ctx.beginPath(); ctx.arc(n.sx*sx,n.sy*sy,2,0,Math.PI*2); ctx.fill();
      });
      /* viewport rect */
      const vx=(-state.panX/state.scale)*sx;
      const vy=(-state.panY/state.scale)*sy;
      const vw=(WORLD_W/state.scale)*sx;
      const vh=(WORLD_H/state.scale)*sy;
      ctx.strokeStyle='rgba(231,111,81,0.85)'; ctx.lineWidth=1.2;
      ctx.strokeRect(vx,vy,vw,vh);
    }

    /* ── EXPORT SVG ──────────────────────────────────────────────────────── */
    function exportSVG() {
      const serializer=new XMLSerializer();
      const svgStr=serializer.serializeToString(svg);
      const blob=new Blob([svgStr],{type:'image/svg+xml;charset=utf-8'});
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a');
      a.href=url; a.download='profile_network.svg';
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
    }

    /* ── STATE ───────────────────────────────────────────────────────────── */
    const defaultThreshold=num(DATA.global_metrics&&DATA.global_metrics.corr_threshold,num(thresholdInput.value,0.15));
    const defaultMetric=Object.keys(DATA.metric_labels||{}).includes('strength')?'strength':
      (Object.keys(DATA.metric_labels||{})[0]||'eigenvector_centrality');

    const state = {
      metric: defaultMetric,
      family: 'all', group: 'all', featureType: 'all', community: 'all',
      sign: 'all', focus: 'all', labels: 'hubs',
      threshold: defaultThreshold,
      search: '',
      selectedTerm: null,
      selectedTerms: new Set(),
      scale: 1, panX: 0, panY: 0,
      draggingTerm: null, panning: false,
      showHulls: true, pathMode: false,
    };

    metricSel.value=state.metric;
    thresholdInput.value=state.threshold.toFixed(2);

    /* ── SVG COORDINATE UTILITIES ────────────────────────────────────────── */
    function svgPoint(evt) {
      const rect=svg.getBoundingClientRect();
      return {
        x: ((evt.clientX-rect.left)/rect.width)*WORLD_W,
        y: ((evt.clientY-rect.top)/rect.height)*WORLD_H,
      };
    }
    function worldPoint(evt) {
      const p=svgPoint(evt);
      return { x:(p.x-state.panX)/state.scale, y:(p.y-state.panY)/state.scale };
    }
    function applyTransform() {
      viewport.setAttribute('transform',`matrix(${state.scale} 0 0 ${state.scale} ${state.panX} ${state.panY})`);
    }

    /* ── VISIBLE NODE / EDGE COMPUTATION ─────────────────────────────────── */
    function visibleNodesBase() {
      const q=state.search.trim().toLowerCase();
      return DATA.nodes.filter(n=>{
        if(state.family!=='all'&&n.ontology_family!==state.family) return false;
        if(state.group!=='all'&&n.ontology_group!==state.group) return false;
        if(state.featureType!=='all'&&n.feature_type!==state.featureType) return false;
        if(state.community!=='all'&&String(n.community)!==state.community) return false;
        if(q&&!n.label.toLowerCase().includes(q)&&!String(n.term||'').toLowerCase().includes(q)) return false;
        return true;
      });
    }
    function egoTerms(depth) {
      if(!state.selectedTerm||!nodeMap.has(state.selectedTerm)) return null;
      if(state.focus==='community'){
        const sel=nodeMap.get(state.selectedTerm);
        return new Set(DATA.nodes.filter(n=>n.community===sel.community).map(n=>n.term));
      }
      const visited=new Set([state.selectedTerm]);
      let frontier=[state.selectedTerm];
      for(let s=0;s<depth;s++){
        const next=[];
        frontier.forEach(t=>{
          (adjacency.get(t)||[]).forEach(e=>{
            if(!visited.has(e.target)){visited.add(e.target);next.push(e.target);}
          });
        });
        frontier=next;
      }
      return visited;
    }
    function visibleNodes() {
      const base=visibleNodesBase();
      if(state.focus==='all') return base;
      const scope=egoTerms(state.focus==='ego2'?2:1);
      if(!scope) return base;
      return base.filter(n=>scope.has(n.term));
    }
    function visibleEdges(nodeSet) {
      return DATA.edges.filter(e=>{
        if(!nodeSet.has(e.source)||!nodeSet.has(e.target)) return false;
        if(num(e.abs_rho)<state.threshold) return false;
        if(state.sign==='positive'&&num(e.rho)<=0) return false;
        if(state.sign==='negative'&&num(e.rho)>=0) return false;
        return true;
      });
    }

    /* ── SIZE SCALE ──────────────────────────────────────────────────────── */
    function sizeScale(nodes, metric) {
      const vals=nodes.map(n=>metric==='within_module_zscore'?Math.max(0,num(n[metric])):num(n[metric]));
      const lo=Math.min(...vals), hi=Math.max(...vals);
      if(!Number.isFinite(lo)||!Number.isFinite(hi)||Math.abs(hi-lo)<1e-10)
        return new Map(nodes.map(n=>[n.term,13]));
      return new Map(nodes.map((n,i)=>[n.term, 8+(vals[i]-lo)/(hi-lo)*21]));
    }

    /* ── CENTER ON SELECTED ──────────────────────────────────────────────── */
    function centerOnSelected() {
      if(!state.selectedTerm||!nodeMap.has(state.selectedTerm)) return;
      const n=nodeMap.get(state.selectedTerm);
      state.panX=WORLD_W/2-n.sx*state.scale;
      state.panY=WORLD_H/2-n.sy*state.scale;
      applyTransform();
    }

    function fitToView(nodes) {
      if(!nodes||!nodes.length) return;
      const xs=nodes.map(n=>n.sx), ys=nodes.map(n=>n.sy);
      const minX=Math.min(...xs)-40, maxX=Math.max(...xs)+40;
      const minY=Math.min(...ys)-40, maxY=Math.max(...ys)+40;
      const scaleX=WORLD_W/(maxX-minX), scaleY=WORLD_H/(maxY-minY);
      state.scale=clamp(Math.min(scaleX,scaleY)*0.92,0.3,3.0);
      state.panX=WORLD_W/2-((minX+maxX)/2)*state.scale;
      state.panY=WORLD_H/2-((minY+maxY)/2)*state.scale;
      applyTransform();
    }

    /* ── RENDER HULLS ────────────────────────────────────────────────────── */
    function renderHulls(nodes) {
      if(!state.showHulls){ hullsLayer.innerHTML=''; return; }
      const byComm=new Map();
      nodes.forEach(n=>{
        if(!byComm.has(n.community)) byComm.set(n.community,[]);
        byComm.get(n.community).push([n.sx,n.sy]);
      });
      const commArr=[...byComm.keys()].sort((a,b)=>a-b);
      const commColors={};
      commArr.forEach((c,i)=>{ commColors[c]=_ONTOLOGY_GROUP_COLORS[i%_ONTOLOGY_GROUP_COLORS.length]; });
      /* Expand hulls by 18px for visual breathing room */
      const EXPAND=18;
      let svgStr='';
      byComm.forEach((pts,comm)=>{
        if(pts.length<1) return;
        const hull=convexHull(pts);
        const cx=hull.reduce((s,p)=>s+p[0],0)/hull.length;
        const cy=hull.reduce((s,p)=>s+p[1],0)/hull.length;
        const expanded=hull.map(p=>{
          const dx=p[0]-cx, dy=p[1]-cy;
          const d=Math.sqrt(dx*dx+dy*dy)||1;
          return [cx+(dx+EXPAND*dx/d), cy+(dy+EXPAND*dy/d)];
        });
        const color=commColors[comm]||'#888';
        const d=expanded.map((p,i)=>`${i===0?'M':'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')+'Z';
        svgStr+=`<path d="${d}" fill="${color}" fill-opacity="0.06" stroke="${color}" stroke-width="1.2" stroke-opacity="0.40" stroke-dasharray="5 4"/>`;
        const textX=cx.toFixed(1), textY=(cy-EXPAND-4).toFixed(1);
        svgStr+=`<text x="${textX}" y="${textY}" text-anchor="middle" font-size="11" fill="${color}" fill-opacity="0.60" font-weight="800">C${comm}</text>`;
      });
      hullsLayer.innerHTML=svgStr;
    }
    /* expose color map for JS access */
    const _ONTOLOGY_GROUP_COLORS=["#1d4e89","#2a9d8f","#e76f51","#c89b3c","#9b59b6","#e74c3c","#27ae60","#2980b9","#f39c12","#16a085","#8e44ad","#d35400","#2c3e50","#c0392b","#1abc9c","#7f8c8d","#6c5ce7","#fd79a8","#00b894","#a29bfe"];

    /* ── RENDER PATH HIGHLIGHT ───────────────────────────────────────────── */
    function renderPathEdges() {
      if(!PATH.terms.length){ pathEdgesLayer.innerHTML=''; return; }
      let svgStr='';
      for(let i=0;i<PATH.terms.length-1;i++){
        const src=nodeMap.get(PATH.terms[i]);
        const tgt=nodeMap.get(PATH.terms[i+1]);
        if(!src||!tgt) continue;
        svgStr+=`<line x1="${src.sx.toFixed(1)}" y1="${src.sy.toFixed(1)}" x2="${tgt.sx.toFixed(1)}" y2="${tgt.sy.toFixed(1)}" stroke="__GOLD__" stroke-width="3.5" stroke-linecap="round" opacity="0.90"/>`;
      }
      pathEdgesLayer.innerHTML=svgStr;
    }

    /* ── RENDER GLOBAL METRICS ───────────────────────────────────────────── */
    function renderGlobal(nodes, edges) {
      const nE=edges.length, nN=nodes.length;
      const density=nN>1?(2*nE)/(nN*(nN-1)):0;
      const pos=edges.filter(e=>num(e.rho)>0).length;
      const neg=edges.filter(e=>num(e.rho)<0).length;
      const crossComm=edges.filter(e=>nodeMap.get(e.source)?.community!==nodeMap.get(e.target)?.community).length;
      const crossFam=edges.filter(e=>nodeMap.get(e.source)?.ontology_family!==nodeMap.get(e.target)?.ontology_family).length;
      const meanPart=nN?nodes.reduce((a,n)=>a+num(n.participation_coefficient),0)/nN:0;
      const meanBridge=nN?nodes.reduce((a,n)=>a+num(n.bridge_ratio),0)/nN:0;
      const meanAbs=nE?edges.reduce((a,e)=>a+num(e.abs_rho),0)/nE:0;
      const cards=[
        ['Visible nodes',nN],['Visible edges',nE],
        ['Subgraph density',density.toFixed(3)],['Mean |ρ|',meanAbs.toFixed(3)],
        ['Positive share',nE?`${(pos/nE*100).toFixed(1)}%`:'0%'],
        ['Negative share',nE?`${(neg/nE*100).toFixed(1)}%`:'0%'],
        ['Cross-community',nE?`${(crossComm/nE*100).toFixed(1)}%`:'0%'],
        ['Cross-family',nE?`${(crossFam/nE*100).toFixed(1)}%`:'0%'],
        ['Mean participation',meanPart.toFixed(3)],['Mean bridge ratio',meanBridge.toFixed(3)],
        ['Modularity',num(DATA.global_metrics&&DATA.global_metrics.modularity_score).toFixed(3)],
        ['Family assortativity',num(DATA.global_metrics&&DATA.global_metrics.ontology_family_assortativity).toFixed(3)],
      ];
      globalRoot.innerHTML=cards.map(([k,v])=>`<div class="netx-metric"><div class="k">${safeText(k)}</div><div class="v">${safeText(v)}</div></div>`).join('');
    }

    /* ── RENDER SUMMARY PILLS ────────────────────────────────────────────── */
    function renderSummary(nodes, edges) {
      const comms=new Set(nodes.map(n=>n.community));
      const fams=new Set(nodes.map(n=>n.ontology_family));
      const types=new Set(nodes.map(n=>n.feature_type));
      const nDummy=nodes.filter(n=>n.feature_type==='Categorical dummy').length;
      const nCont=nodes.length-nDummy;
      const pathBadge=PATH.mode?`<span class="netx-pill warn">Path mode${PATH.source?' · source set':''}</span>`:'';
      const lassoBadge=state.selectedTerms.size>1?`<span class="netx-pill warn">${state.selectedTerms.size} selected</span>`:'';
      summaryRoot.innerHTML=[
        `${nodes.length} nodes`,`${edges.length} edges`,
        `${comms.size} communities`,`${fams.size} families`,
        `${nCont} continuous / ${nDummy} dummy`,
        `${safeText(DATA.metric_labels[state.metric]||state.metric)} sizing`,
        pathBadge, lassoBadge,
      ].filter(Boolean).map(t=>`<span class="netx-pill">${t}</span>`).join('');
    }

    /* ── RENDER RANKING ──────────────────────────────────────────────────── */
    function renderRanking(nodes) {
      const metric=state.metric;
      const sorted=[...nodes].sort((a,b)=>num(b[metric])-num(a[metric])).slice(0,8);
      rankingRoot.innerHTML=sorted.length
        ? sorted.map((n,i)=>`<div class="netx-item"><strong>${i+1}. ${safeText(n.label)}</strong><span>${safeText(n.ontology_group)} · ${safeText(n.feature_type)} · ${safeText(DATA.metric_labels[metric]||metric)}=${num(n[metric]).toFixed(4)}</span></div>`).join('')
        : '<div class="netx-item"><span>No nodes pass current filters.</span></div>';
    }

    /* ── RENDER SELECTED NODE PANEL ──────────────────────────────────────── */
    function renderSelected(edges) {
      const node=state.selectedTerm?nodeMap.get(state.selectedTerm):null;
      if(!node){
        selectedRoot.innerHTML='<div class="netx-item"><span>Click a node to inspect its topology, bridge metrics, and strongest correlations. Shift+drag for multi-select.</span></div>';
        localCardsRoot.innerHTML='';
        return;
      }
      const nbrs=(adjacency.get(node.term)||[])
        .filter(e=>num(e.abs_rho)>=state.threshold)
        .filter(e=>state.sign==='all'||(state.sign==='positive'?num(e.rho)>0:num(e.rho)<0))
        .map(e=>{
          const nb=nodeMap.get(e.target);
          return {label:nb?nb.label:e.target, group:nb?nb.ontology_group:'Other', ftype:nb?nb.feature_type:'', rho:num(e.rho), abs_rho:num(e.abs_rho)};
        })
        .sort((a,b)=>b.abs_rho-a.abs_rho).slice(0,8);
      const pinned=SIM.pinned.has(node.term)?'📌 pinned':'';
      selectedRoot.innerHTML=`<div class="netx-item"><strong>${safeText(node.label)}</strong>
        <span>${safeText(node.ontology_group)} · ${safeText(node.feature_type)} ${pinned}</span>
        <span>Community ${node.community} · Degree ${num(node.degree).toFixed(0)} · k-core ${num(node.k_core).toFixed(0)}</span>
      </div>`+nbrs.map(nb=>`<div class="netx-item"><strong>${safeText(nb.label)}</strong><span>${safeText(nb.group)} · ρ=${nb.rho.toFixed(3)} · |ρ|=${nb.abs_rho.toFixed(3)}</span></div>`).join('');

      const cards=[
        ['Participation',fmt3(node.participation_coefficient)],
        ['Bridge Ratio',fmt3(node.bridge_ratio)],
        ['Within-mod Z',fmt3(node.within_module_zscore)],
        ['Pos. Strength',fmt2(node.positive_strength)],
        ['Neg. Strength',fmt2(node.negative_strength)],
        ['Signed Balance',fmt3(node.signed_balance)],
        ['Within-comm.',fmt2(node.within_community_strength)],
        ['Family share',`${(num(node.same_family_strength_share)*100).toFixed(1)}%`],
        ['Eigenvector',fmt4(node.eigenvector_centrality)],
        ['PageRank',fmt4(node.pagerank)],
        ['Closeness',fmt4(node.closeness_centrality)],
        ['Betweenness',fmt4(node.betweenness_centrality)],
      ];
      localCardsRoot.innerHTML=cards.map(([k,v])=>`<div class="netx-metric"><div class="k">${safeText(k)}</div><div class="v">${safeText(v)}</div></div>`).join('');
    }

    /* ── RENDER MULTI-SELECT ANALYSIS ────────────────────────────────────── */
    function renderMultiSelect(nodes) {
      if(state.selectedTerms.size<2){ multiSelSection.style.display='none'; return; }
      multiSelSection.style.display='';
      const sel=nodes.filter(n=>state.selectedTerms.has(n.term));
      if(!sel.length){ multiSelSection.style.display='none'; return; }
      const metric=state.metric;
      const mean=sel.reduce((a,n)=>a+num(n[metric]),0)/sel.length;
      const fams=new Set(sel.map(n=>n.ontology_family));
      const comms=new Set(sel.map(n=>n.community));
      const types=new Set(sel.map(n=>n.feature_type));
      const meanPart=sel.reduce((a,n)=>a+num(n.participation_coefficient),0)/sel.length;
      const meanBridge=sel.reduce((a,n)=>a+num(n.bridge_ratio),0)/sel.length;
      multiSelStats.innerHTML=`<div class="netx-list">
        <div class="netx-item"><strong>${sel.length} nodes selected</strong>
          <span>${comms.size} communities · ${fams.size} families · ${types.size} feature types</span>
          <span>Mean ${safeText(DATA.metric_labels[metric]||metric)} = ${mean.toFixed(4)}</span>
          <span>Mean participation = ${meanPart.toFixed(3)} · Mean bridge ratio = ${meanBridge.toFixed(3)}</span>
        </div>
        ${sel.slice(0,5).map(n=>`<div class="netx-item"><strong>${safeText(n.label)}</strong><span>${safeText(n.ontology_group)} · ${safeText(DATA.metric_labels[metric]||metric)}=${num(n[metric]).toFixed(3)}</span></div>`).join('')}
        ${sel.length>5?`<div class="netx-item"><span>…and ${sel.length-5} more</span></div>`:''}
      </div>`;
    }

    /* ── RENDER PATH INFO ────────────────────────────────────────────────── */
    function renderPathInfo() {
      if(!PATH.mode){ pathInfoEl.textContent='Path analysis inactive. Click "Path Mode" to enable.'; pathListEl.innerHTML=''; return; }
      if(!PATH.source){ pathInfoEl.textContent='Path mode: click a node to set source.'; pathListEl.innerHTML=''; return; }
      const srcNode=nodeMap.get(PATH.source);
      if(!PATH.terms.length){
        pathInfoEl.textContent=`Source: ${srcNode?srcNode.label:PATH.source}. Now click target node to trace shortest path.`;
        pathListEl.innerHTML='';
        return;
      }
      pathInfoEl.textContent=`Shortest path: ${PATH.terms.length} nodes (${PATH.terms.length-1} hops)`;
      pathListEl.innerHTML=PATH.terms.map((t,i)=>{
        const n=nodeMap.get(t);
        return `<div class="netx-item path-item"><strong>${i===0?'⬤':i===PATH.terms.length-1?'⬤':'○'} ${safeText(n?n.label:t)}</strong><span>${n?safeText(n.ontology_group)+' · '+safeText(n.feature_type):''}</span></div>`;
      }).join('');
    }

    /* ── RENDER SCENE ────────────────────────────────────────────────────── */
    function renderScene(nodes, edges) {
      const metric=state.metric;
      const sizeMap=sizeScale(nodes,metric);
      const ordered=[...nodes].sort((a,b)=>num(b[metric])-num(a[metric]));
      const hubTerms=new Set(ordered.slice(0,12).map(n=>n.term));
      const neighborTerms=new Set((adjacency.get(state.selectedTerm)||[]).map(e=>e.target));
      const pathTermSet=new Set(PATH.terms);
      const labelTerms=new Set();
      if(state.labels==='all') nodes.forEach(n=>labelTerms.add(n.term));
      else if(state.labels==='hubs'){
        hubTerms.forEach(t=>labelTerms.add(t));
        if(state.selectedTerm) labelTerms.add(state.selectedTerm);
        neighborTerms.forEach(t=>labelTerms.add(t));
        PATH.terms.forEach(t=>labelTerms.add(t));
      }
      const q=state.search.trim().toLowerCase();

      /* Render hulls */
      renderHulls(nodes);
      renderPathEdges();

      /* Render edges */
      edgesLayer.innerHTML=edges.map(e=>{
        const src=nodeMap.get(e.source), tgt=nodeMap.get(e.target);
        if(!src||!tgt) return '';
        const onPath=PATH.edgeKeys.has(e.source+'||'+e.target);
        const selected=state.selectedTerm&&(e.source===state.selectedTerm||e.target===state.selectedTerm);
        const inMultiSel=state.selectedTerms.size>1&&(state.selectedTerms.has(e.source)&&state.selectedTerms.has(e.target));
        const rho=num(e.rho);
        let stroke, opacity;
        if(rho>=0){
          opacity=selected||inMultiSel?0.72:0.22;
          stroke=`rgba(29,78,137,${opacity})`;
        } else {
          opacity=selected||inMultiSel?0.70:0.20;
          stroke=`rgba(231,111,81,${opacity})`;
        }
        const width=(selected||inMultiSel?1.4:0.35)+num(e.abs_rho)*3.2;
        return `<line x1="${src.sx.toFixed(1)}" y1="${src.sy.toFixed(1)}" x2="${tgt.sx.toFixed(1)}" y2="${tgt.sy.toFixed(1)}" stroke="${stroke}" stroke-width="${width.toFixed(2)}" stroke-linecap="round"><title>${safeText(src.label)} ↔ ${safeText(tgt.label)} | ρ=${rho.toFixed(3)}</title></line>`;
      }).join('');

      /* Render nodes — circle for continuous, diamond for categorical/dummy */
      nodesLayer.innerHTML=nodes.map(n=>{
        const r=sizeMap.get(n.term)||12;
        const color=DATA.group_colors[n.ontology_group]||'#7f8c8d';
        const selected=state.selectedTerm===n.term;
        const inMultiSel=state.selectedTerms.has(n.term);
        const onPath=pathTermSet.has(n.term);
        const isSource=PATH.source===n.term;
        const neighbor=!selected&&state.selectedTerm&&neighborTerms.has(n.term);
        const stroke=selected?'__INK__':(isSource?'#c89b3c':(onPath?'__GOLD__':(inMultiSel?'#2a9d8f':(neighbor?'__GOLD__':'#ffffff'))));
        const strokeW=selected||isSource?3.2:(onPath||inMultiSel||neighbor?2.2:1.0);
        const dimmed=q&&!n.label.toLowerCase().includes(q)&&!String(n.term||'').toLowerCase().includes(q);
        const opac=dimmed?0.35:0.92;
        const isCat=n.feature_type==='Categorical dummy';
        let shape;
        if(isCat){
          /* Diamond */
          const d=r*1.12;
          shape=`<polygon points="${n.sx.toFixed(1)},${(n.sy-d).toFixed(1)} ${(n.sx+d).toFixed(1)},${n.sy.toFixed(1)} ${n.sx.toFixed(1)},${(n.sy+d).toFixed(1)} ${(n.sx-d).toFixed(1)},${n.sy.toFixed(1)}" fill="${color}" fill-opacity="${opac}" stroke="${stroke}" stroke-width="${strokeW}"/>`;
        } else {
          /* Circle */
          shape=`<circle cx="${n.sx.toFixed(1)}" cy="${n.sy.toFixed(1)}" r="${r.toFixed(1)}" fill="${color}" fill-opacity="${opac}" stroke="${stroke}" stroke-width="${strokeW}"/>`;
        }
        return `<g class="netx-node${onPath?' netx-path-node':''}" data-term="${safeText(n.term)}" role="button" aria-label="${safeText(n.label)}">${shape}<title>${safeText(n.label)} | ${safeText(n.ontology_group)} | ${safeText(DATA.metric_labels[metric]||metric)}=${num(n[metric]).toFixed(4)}</title></g>`;
      }).join('');

      /* Render labels */
      labelsLayer.innerHTML=state.labels==='none'?'':
        nodes.filter(n=>labelTerms.has(n.term)).map(n=>{
          const r=sizeMap.get(n.term)||12;
          const sel=state.selectedTerm===n.term;
          return `<text class="netx-label${sel?' selected-label':''}" x="${(n.sx+r+4).toFixed(1)}" y="${(n.sy+4).toFixed(1)}">${safeText(n.label)}</text>`;
        }).join('');

      /* Wire up node events */
      nodesLayer.querySelectorAll('.netx-node').forEach(el=>{
        const term=el.getAttribute('data-term');
        el.addEventListener('mouseenter',evt=>{
          const n=nodeMap.get(term);
          if(n) showTooltip(n,evt);
        });
        el.addEventListener('mouseleave',hideTooltip);
        el.addEventListener('click',evt=>{
          evt.stopPropagation();
          if(PATH.mode){
            if(!PATH.source){ PATH.source=term; }
            else if(PATH.source===term){ PATH.source=null; PATH.terms=[]; PATH.edgeKeys=new Set(); }
            else {
              const allNodes=visibleNodes();
              const nodeSet=new Set(allNodes.map(n=>n.term));
              PATH.terms=bfsPath(PATH.source,term,nodeSet,state.threshold);
              PATH.edgeKeys=computePathEdgeKeys(PATH.terms);
              PATH.target=term;
            }
          } else {
            state.selectedTerm=term;
            state.selectedTerms=new Set([term]);
          }
          render();
        });
        el.addEventListener('dblclick',evt=>{
          evt.stopPropagation();
          if(SIM.pinned.has(term)){ SIM.pinned.delete(term); } else { SIM.pinned.add(term); }
          render();
        });
        el.addEventListener('mousedown',evt=>{ evt.stopPropagation(); state.draggingTerm=term; });
      });
    }

    /* ── MAIN RENDER ─────────────────────────────────────────────────────── */
    function render() {
      const nodes=visibleNodes();
      const nodeSet=new Set(nodes.map(n=>n.term));
      if(state.selectedTerm&&!nodeSet.has(state.selectedTerm)&&state.focus==='all') state.selectedTerm=null;
      const edges=visibleEdges(nodeSet);
      renderScene(nodes,edges);
      renderGlobal(nodes,edges);
      renderSummary(nodes,edges);
      renderRanking(nodes);
      renderSelected(edges);
      renderMultiSelect(nodes);
      renderPathInfo();
      thresholdReadout.textContent=`|ρ| ≥ ${state.threshold.toFixed(2)} · ${edges.length} edges survive`;
      applyTransform();
      renderMinimap(nodes,edges);
    }

    /* ── PRESET SYSTEM ───────────────────────────────────────────────────── */
    function setPreset(p) {
      state.family='all'; state.group='all'; state.featureType='all';
      state.community='all'; state.sign='all'; state.focus='all';
      state.labels='hubs'; state.threshold=defaultThreshold;
      state.metric=defaultMetric;
      const hasMet=(m)=>Object.keys(DATA.metric_labels).includes(m);
      if(p==='bridges'){
        state.metric=hasMet('participation_coefficient')?'participation_coefficient':'betweenness_centrality';
        state.threshold=0.24; state.labels='hubs';
      } else if(p==='hubs'){
        state.metric=hasMet('strength')?'strength':'eigenvector_centrality';
        state.threshold=0.18;
      } else if(p==='bigfive'){
        state.family='Big Five';
        state.metric=hasMet('within_module_zscore')?'within_module_zscore':'eigenvector_centrality';
        state.labels='all'; state.threshold=0.16;
      } else if(p==='social'){
        state.family='Social Context';
        state.metric=hasMet('bridge_ratio')?'bridge_ratio':'strength';
        state.labels='all'; state.threshold=0.18;
      } else if(p==='politics'){
        state.family='Political Psychology';
        state.metric=hasMet('eigenvector_centrality')?'eigenvector_centrality':defaultMetric;
        state.labels='all'; state.threshold=0.18;
      } else if(p==='dummies'){
        state.featureType='Categorical dummy';
        state.metric=hasMet('bridge_ratio')?'bridge_ratio':defaultMetric;
        state.threshold=0.20;
      } else if(p==='kcore'){
        state.metric=hasMet('k_core')?'k_core':'eigenvector_centrality';
        state.threshold=0.30; state.labels='hubs';
      }
      metricSel.value=state.metric; familySel.value=state.family; groupSel.value=state.group;
      featureTypeSel.value=state.featureType; communitySel.value=state.community;
      signSel.value=state.sign; focusSel.value=state.focus; labelsSel.value=state.labels;
      thresholdInput.value=state.threshold.toFixed(2);
      state.selectedTerms=new Set(); state.selectedTerm=null;
      render();
    }

    /* ── EVENTS ──────────────────────────────────────────────────────────── */
    /* Background click → deselect */
    svg.addEventListener('click',()=>{
      if(PATH.mode) return;
      state.selectedTerm=null; state.selectedTerms=new Set(); render();
    });

    /* Panning / dragging */
    svg.addEventListener('mousedown',evt=>{
      if(evt.shiftKey){ lassoStart(evt); return; }
      state.panning=true; svg.classList.add('is-panning');
    });
    window.addEventListener('mouseup',()=>{
      if(LASSO.active) lassoEnd();
      state.draggingTerm=null; state.panning=false; svg.classList.remove('is-panning');
    });
    window.addEventListener('mousemove',evt=>{
      if(LASSO.active){ lassoMove(evt); return; }
      if(state.draggingTerm&&nodeMap.has(state.draggingTerm)){
        evt.preventDefault();
        const p=worldPoint(evt);
        const n=nodeMap.get(state.draggingTerm);
        n.sx=clamp(p.x,24,WORLD_W-24); n.sy=clamp(p.y,24,WORLD_H-24);
        SIM.pinned.add(state.draggingTerm);
        render(); return;
      }
      if(!state.panning) return;
      const mX=(evt.movementX/svg.getBoundingClientRect().width)*WORLD_W;
      const mY=(evt.movementY/svg.getBoundingClientRect().height)*WORLD_H;
      state.panX+=mX; state.panY+=mY; applyTransform();
      renderMinimap(visibleNodes(),visibleEdges(new Set(visibleNodes().map(n=>n.term))));
    });

    /* Wheel zoom centered on cursor */
    svg.addEventListener('wheel',evt=>{
      evt.preventDefault();
      const mouse=svgPoint(evt);
      const world=worldPoint(evt);
      const factor=evt.deltaY<0?1.12:0.89;
      const nextScale=clamp(state.scale*factor,0.25,6.0);
      state.panX=mouse.x-world.x*nextScale;
      state.panY=mouse.y-world.y*nextScale;
      state.scale=nextScale; applyTransform();
      renderMinimap(visibleNodes(),visibleEdges(new Set(visibleNodes().map(n=>n.term))));
    },{passive:false});

    /* Control changes */
    metricSel.addEventListener('change',e=>{state.metric=e.target.value;render();});
    familySel.addEventListener('change',e=>{state.family=e.target.value;render();});
    groupSel.addEventListener('change',e=>{state.group=e.target.value;render();});
    featureTypeSel.addEventListener('change',e=>{state.featureType=e.target.value;render();});
    communitySel.addEventListener('change',e=>{state.community=e.target.value;render();});
    signSel.addEventListener('change',e=>{state.sign=e.target.value;render();});
    focusSel.addEventListener('change',e=>{state.focus=e.target.value;render();});
    labelsSel.addEventListener('change',e=>{state.labels=e.target.value;render();});
    thresholdInput.addEventListener('input',e=>{state.threshold=num(e.target.value,defaultThreshold);render();});
    searchInput.addEventListener('input',e=>{
      state.search=e.target.value;
      const q=state.search.trim().toLowerCase();
      if(q){const m=DATA.nodes.find(n=>n.label.toLowerCase().includes(q)||String(n.term||'').toLowerCase().includes(q));if(m)state.selectedTerm=m.term;}
      render();
    });

    /* Preset buttons */
    document.querySelectorAll('#netx-root .netx-preset').forEach(b=>b.addEventListener('click',()=>setPreset(b.dataset.preset)));

    /* Toolbar buttons */
    document.getElementById('netx-zoom-in').addEventListener('click',()=>{state.scale=clamp(state.scale*1.18,0.25,6.0);applyTransform();renderMinimap(visibleNodes(),visibleEdges(new Set(visibleNodes().map(n=>n.term))));});
    document.getElementById('netx-zoom-out').addEventListener('click',()=>{state.scale=clamp(state.scale*0.85,0.25,6.0);applyTransform();renderMinimap(visibleNodes(),visibleEdges(new Set(visibleNodes().map(n=>n.term))));});
    document.getElementById('netx-reset-view').addEventListener('click',()=>{state.scale=1;state.panX=0;state.panY=0;applyTransform();renderMinimap(visibleNodes(),visibleEdges(new Set(visibleNodes().map(n=>n.term))));});
    document.getElementById('netx-center-selected').addEventListener('click',()=>centerOnSelected());
    document.getElementById('netx-fit-btn').addEventListener('click',()=>fitToView(visibleNodes()));

    forceBtn.addEventListener('click',()=>{ if(SIM.running) stopSim(); else startSim(); });
    document.getElementById('netx-reset-sim-btn').addEventListener('click',()=>{ stopSim(); SIM.pinned.clear(); resetOriginalPositions(); render(); });
    hullBtn.addEventListener('click',()=>{ state.showHulls=!state.showHulls; hullBtn.style.background=state.showHulls?'#edf4ff':'#f4f7ff'; render(); });
    pathBtn.addEventListener('click',()=>{
      PATH.mode=!PATH.mode;
      if(!PATH.mode){ PATH.source=null; PATH.target=null; PATH.terms=[]; PATH.edgeKeys=new Set(); }
      pathBtn.style.background=PATH.mode?'#fff7ee':'#f4f7ff';
      pathBtn.style.color=PATH.mode?'#c45c1a':'__INK__';
      svg.classList.toggle('lasso-mode',false);
      render();
    });
    document.getElementById('netx-export-btn').addEventListener('click',exportSVG);

    /* Keyboard shortcuts */
    document.addEventListener('keydown',evt=>{
      if(evt.target.tagName==='INPUT'||evt.target.tagName==='SELECT') return;
      if(evt.key==='Escape'){state.selectedTerm=null;state.selectedTerms=new Set();PATH.mode=false;PATH.source=null;PATH.terms=[];PATH.edgeKeys=new Set();pathBtn.style.background='#f4f7ff';pathBtn.style.color='__INK__';render();}
      if(evt.key==='r'||evt.key==='R'){state.scale=1;state.panX=0;state.panY=0;applyTransform();}
      if(evt.key==='h'||evt.key==='H'){state.showHulls=!state.showHulls;hullBtn.style.background=state.showHulls?'#edf4ff':'#f4f7ff';render();}
      if(evt.key==='f'||evt.key==='F'){fitToView(visibleNodes());}
      if(evt.key==='p'||evt.key==='P'){if(SIM.running)stopSim();else startSim();}
    });

    /* ── INIT ────────────────────────────────────────────────────────────── */
    normaliseCoords();
    deriveNetworkMetrics();
    populateControls();
    metricSel.value=state.metric;
    thresholdInput.value=state.threshold.toFixed(2);
    hullBtn.style.background='#edf4ff'; /* hulls on by default */
    render();

  })();
  </script>
</div>
"""
    replacements = {
        "__PAYLOAD__": payload_json,
        "__THRESHOLD__": f"{float(global_metrics.get('corr_threshold', 0.15) or 0.15):.2f}",
        "__BLUE__": PALETTE["blue"],
        "__MUTED__": PALETTE["muted"],
        "__INK__": PALETTE["ink"],
        "__GOLD__": PALETTE["gold"],
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html




# ─── main entry point ─────────────────────────────────────────────────────────

def generate_research_visuals(
    sem_long_csv_path: str | Path,
    sem_result_json_path: str | Path,
    ols_params_csv_path: str | Path,
    output_dir: str | Path,
    run_id: str,
) -> Dict[str, Any]:
    output_root  = Path(output_dir)
    figures_dir  = output_root / "figures"
    snap_dir     = output_root / "data_snapshots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    long_df    = pd.read_csv(sem_long_csv_path)
    sem_result = json.loads(Path(sem_result_json_path).read_text(encoding="utf-8"))
    ols_params = pd.read_csv(ols_params_csv_path)

    s05 = Path(sem_long_csv_path).resolve().parent
    s06 = Path(sem_result_json_path).resolve().parent
    stage_outputs_root = s05.parent

    def _load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    profile_df       = _load(s05 / "profile_level_effectivity.csv")
    profile_index_df = _load(s06 / "profile_susceptibility_index.csv")
    exploratory_df        = _load(s06 / "exploratory_moderator_comparison.csv")
    expanded_moderator_df = _load(s06 / "expanded_moderator_comparison.csv")
    weight_df             = _load(s06 / "moderator_weight_table.csv")
    task_coeff_df    = _load(s06 / "conditional_susceptibility_task_coefficients.csv")
    task_summary_df  = _load(s06 / "conditional_susceptibility_task_summary.csv")
    bootstrap_rank_df = _load(s06 / "conditional_susceptibility_bootstrap_ranks.csv")
    network_centrality_df = _load(s06 / "profile_network_centrality.csv")
    network_edge_df       = _load(s06 / "profile_network_edges.csv")
    network_layout_df     = _load(s06 / "profile_network_layout.csv")
    network_global_path   = s06 / "profile_network_global_metrics.json"
    network_global_metrics: Dict[str, Any] = {}
    if network_global_path.exists():
        try:
            network_global_metrics = json.loads(network_global_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    ontology_catalog_path = stage_outputs_root / "01_create_scenarios" / "ontology_leaf_catalog.json"
    ontology_catalog = (
        json.loads(ontology_catalog_path.read_text(encoding="utf-8"))
        if ontology_catalog_path.exists() else {}
    )
    ontology_payload = _load_dashboard_ontology_payload(ontology_catalog)
    quality_diagnostics_path = s06 / "analysis_quality_diagnostics.json"
    quality_diagnostics: Dict[str, Any] = {}
    if quality_diagnostics_path.exists():
        try:
            quality_diagnostics = json.loads(quality_diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    ridge_summary_path = s06 / "ridge_full_summary.json"
    ridge_summary: Dict[str, Any] = {}
    if ridge_summary_path.exists():
        try:
            ridge_summary = json.loads(ridge_summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    rf_summary_path = s06 / "rf_summary.json"
    rf_summary: Dict[str, Any] = {}
    if rf_summary_path.exists():
        try:
            rf_summary = json.loads(rf_summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # current design: scenario-level ML artifacts (model ladder, scans, effect summaries)
    effect_attacks_df = _load(s06 / "effect_summary_attacks.csv")
    effect_opinions_df = _load(s06 / "effect_summary_opinions.csv")
    effect_cells_df = _load(s06 / "effect_summary_cells.csv")
    effect_profiles_df = _load(s06 / "effect_summary_profiles.csv")
    scan_attack_df = _load(s06 / "moderation_scan_by_attack.csv")
    scan_opinion_df = _load(s06 / "moderation_scan_by_opinion.csv")
    scan_pooled_df = _load(s06 / "moderation_scan_pooled.csv")
    ladder_df = _load(s06 / "scenario_model_ladder.csv")
    ladder_meta: Dict[str, Any] = {}
    ladder_meta_path = s06 / "scenario_ml_summary.json"
    if ladder_meta_path.exists():
        try:
            ladder_meta = json.loads(ladder_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    mantel_pairs_df = _load(s06 / "supplementary_profile_distance_pairs.csv")
    mantel_meta: Dict[str, Any] = {}
    mantel_meta_path = s06 / "supplementary_profile_distance_mantel.json"
    if mantel_meta_path.exists():
        try:
            mantel_meta = json.loads(mantel_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    enet_summary_path = s06 / "elastic_net_summary.json"
    enet_summary: Dict[str, Any] = {}
    if enet_summary_path.exists():
        try:
            enet_summary = json.loads(enet_summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Try to find embedding_dashboard.json (generated by semantic_embedding.py).
    # Walk up from the stage 07 output dir to the run root so the artifact in
    # evaluation/<run_id>/embeddings is found regardless of nesting depth.
    embedding_data_path: Optional[Path] = None
    production_embedding_path: Optional[Path] = None
    # Check both the flat run-root location (embeddings/) and the grouped layout
    # (visuals/embeddings/), so the dashboard renders the semantic explorer whether
    # the run is finalized into grouped folders or left flat.
    for base in [output_root, output_root.parent, output_root.parent.parent, output_root.parent.parent.parent]:
        for sub in ("embeddings", "visuals/embeddings"):
            if embedding_data_path is None and (base / sub / "embedding_dashboard.json").exists():
                embedding_data_path = base / sub / "embedding_dashboard.json"
        for sub in ("embeddings_production", "visuals/embeddings_production"):
            if production_embedding_path is None and (base / sub / "embedding_dashboard.json").exists():
                production_embedding_path = base / sub / "embedding_dashboard.json"
        if embedding_data_path is not None and production_embedding_path is not None:
            break

    sem_coeff_df = pd.DataFrame(sem_result.get("coefficients", []))
    fit          = sem_result.get("fit_indices", {})
    icc_data     = {}
    icc_path     = s06 / "intraclass_correlation.json"
    if icc_path.exists():
        try:
            icc_data = json.loads(icc_path.read_text())
        except Exception:
            pass

    abs_icc = icc_data.get("abs_delta_score", {}).get("icc1") if isinstance(icc_data, dict) else None
    icc_str = f"{float(abs_icc):.3f}" if abs_icc is not None else "n/a"
    if not bootstrap_rank_df.empty and "profile_id" in bootstrap_rank_df.columns:
        profile_index_df = profile_index_df.merge(
            bootstrap_rank_df,
            on="profile_id",
            how="left",
            suffixes=("", "_boot"),
        )

    n_profiles = int(long_df["profile_id"].nunique()) if "profile_id" in long_df.columns else "n/a"
    n_attacks = int(long_df["attack_leaf"].nunique()) if "attack_leaf" in long_df.columns else "n/a"
    n_opinions = int(long_df["opinion_leaf"].nunique()) if "opinion_leaf" in long_df.columns else "n/a"
    pct_pos    = (
        f"{(long_df['adversarial_effectivity'] > 0).mean() * 100:.1f}%"
        if "adversarial_effectivity" in long_df.columns else "n/a"
    )
    summary_cards: Dict[str, Any] = {
        "Profiles":        n_profiles,
        "Scenarios":       len(long_df),
        "Attack Vectors":  n_attacks,
        "Opinion Leaves":  n_opinions,
        "Mean |Δ|":        f"{long_df['abs_delta_score'].mean():.1f}" if "abs_delta_score" in long_df.columns else "n/a",
        "Mean AE":         f"{long_df['adversarial_effectivity'].mean():.1f}" if "adversarial_effectivity" in long_df.columns else "n/a",
        "% AE > 0":        pct_pos,
        "Baseline Fallback": (
            f"{float(quality_diagnostics.get('baseline_fallback_used_rate', 0.0)) * 100:.1f}%"
            if quality_diagnostics.get("baseline_fallback_used_rate") is not None
            else "n/a"
        ),
        "Post Fallback": (
            f"{float(quality_diagnostics.get('post_fallback_used_rate', 0.0)) * 100:.1f}%"
            if quality_diagnostics.get("post_fallback_used_rate") is not None
            else "n/a"
        ),
        "ICC(1) |Δ|":      icc_str,
        "Ridge CV-R²":     f"{float(ridge_summary.get('cv_r2')):.3f}" if ridge_summary.get("cv_r2") is not None else "n/a",
        "RF OOB R²":       f"{float(rf_summary.get('oob_r2')):.3f}" if rf_summary.get("oob_r2") is not None else "n/a",
        "CFI":             f"{float(fit['CFI']):.3f}" if fit.get("CFI") is not None else "n/a",
        "RMSEA":           f"{float(fit['RMSEA']):.3f}" if fit.get("RMSEA") is not None else "n/a",
    }

    figure_divs: List[Tuple[str, str]] = []
    visual_files: List[str] = []

    def _add_fig(title: str, fig: go.Figure, fname: str) -> None:
        visual_files.append(_save_figure_html(fig, figures_dir / fname))
        figure_divs.append((title, fig.to_html(include_plotlyjs=False, full_html=False, config=PLOTLY_CONFIG)))

    def _add_html(title: str, html: str, fname: Optional[str] = None) -> None:
        if fname:
            visual_files.append(_save_html_block(html, figures_dir / fname, title))
        figure_divs.append((title, html))

    _add_html(
        "Key Findings",
        _html_key_findings(
            attack_effects=effect_attacks_df,
            opinion_effects=effect_opinions_df,
            pooled_scan=scan_pooled_df,
            profile_effects=effect_profiles_df,
            ladder_df=ladder_df,
            ladder_meta=ladder_meta,
            long_df=long_df,
        ),
        "key_findings.html",
    )
    _add_html("Ontology Explorer", _html_ontology_explorer(ontology_payload), "ontology_explorer.html")
    _add_fig("Factorial 3D Surface",    _fig_factorial_3d(long_df),          "factorial_3d.html")
    _add_fig("Factorial Heat + Contour", _fig_factorial_2d(long_df),          "factorial_2d.html")

    attack_path_map = (
        long_df[["attack_leaf"]].dropna().drop_duplicates().assign(p=lambda d: d["attack_leaf"].astype(str))
        .set_index("attack_leaf")["p"].to_dict()
        if "attack_leaf" in long_df.columns else {}
    )
    opinion_path_map = (
        long_df[["opinion_leaf"]].dropna().drop_duplicates().assign(p=lambda d: d["opinion_leaf"].astype(str))
        .set_index("opinion_leaf")["p"].to_dict()
        if "opinion_leaf" in long_df.columns else {}
    )
    if not scan_attack_df.empty and not effect_cells_df.empty:
        _add_html(
            "Moderation Paths",
            _html_moderation_paths(scan_attack_df, effect_cells_df, scan_pooled_df),
            "moderation_paths.html",
        )
        _add_fig(
            "Moderation Scan Heatmap",
            _fig_moderation_heatmap_dendro(scan_attack_df, attack_path_map, "attack", "Conditional moderation heatmap"),
            "moderation_scan_heatmap.html",
        )
    elif not sem_coeff_df.empty:
        _add_html("Moderation Paths", _html_sem_network(sem_coeff_df, long_df), "sem_network.html")

    if not ladder_df.empty:
        _add_fig("Model Ladder", _fig_model_ladder(ladder_df, ladder_meta), "model_ladder.html")

    if not scan_opinion_df.empty:
        _add_fig(
            "SEM Heatmap",
            _fig_moderation_heatmap_dendro(scan_opinion_df, opinion_path_map, "opinion", "Profile moderation heatmap"),
            "sem_heatmap.html",
        )
    elif not sem_coeff_df.empty:
        _add_fig("SEM Heatmap",  _fig_sem_heatmap(sem_coeff_df, exploratory_df, long_df), "sem_heatmap.html")

    if not task_coeff_df.empty:
        _add_html("Conditional Susceptibility Estimator",
                  _html_cs_estimator(task_coeff_df, task_summary_df, long_df),
                  "conditional_susceptibility_estimator.html")
    if not task_summary_df.empty:
        _add_fig("Task Reliability Surface", _fig_task_reliability(task_summary_df), "task_reliability.html")

    _add_fig("Distribution by Opinion Leaf",   _fig_violin(long_df),               "violin.html")
    _add_fig("Distribution by Attack Vector",  _fig_raw_attack_comparison(long_df), "attack_comparison.html")

    if not profile_index_df.empty:
        _add_fig("Susceptibility Map", _fig_susceptibility_scatter(profile_index_df, long_df),
                 "susceptibility_map.html")
        _add_fig("Bootstrap Rank Stability", _fig_bootstrap_rank_stability(profile_index_df), "bootstrap_rank_stability.html")

    # ── current design: advanced inferential outputs ─────────────────────────────────
    advanced_icc_path = s06 / "advanced_multilevel_icc.json"
    advanced_icc: Dict[str, Any] = {}
    if advanced_icc_path.exists():
        try:
            advanced_icc = json.loads(advanced_icc_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if advanced_icc:
        _add_fig("Multilevel ICC Decomposition", _fig_multilevel_icc_sunburst(advanced_icc), "multilevel_icc.html")

    advanced_perm_df = _load(s06 / "advanced_permutation_importance.csv")
    if not advanced_perm_df.empty:
        _add_fig("Permutation Importance (FDR)", _fig_permutation_forest(advanced_perm_df), "permutation_importance.html")

    advanced_bca_df = _load(s06 / "advanced_bca_bootstrap_ridge.csv")
    if not advanced_bca_df.empty:
        _add_fig("BCa Coefficient CIs", _fig_bca_coefficients(advanced_bca_df), "bca_coefficients.html")

    advanced_rank_df = _load(s06 / "advanced_rank_stability.csv")
    if not advanced_rank_df.empty:
        _add_fig("Bayesian Rank CIs", _fig_rank_credible_intervals(advanced_rank_df), "bayesian_rank_intervals.html")

    # Use expanded table (all ~100 features with ridge estimates) when available;
    # fall back to OLS-only exploratory table.
    if not scan_pooled_df.empty:
        _add_fig("Moderator Forest", _fig_pooled_moderation_forest(scan_pooled_df), "moderator_forest.html")
    else:
        forest_df = expanded_moderator_df if not expanded_moderator_df.empty else exploratory_df
        if not forest_df.empty:
            _add_fig("Moderator Forest", _fig_moderator_forest(forest_df), "moderator_forest.html")
    if not weight_df.empty:
        _add_fig("Hierarchical Importance", _fig_hierarchical_importance(weight_df),        "hierarchical_importance.html")

    _add_fig("Profile Heatmap",  _fig_profile_heatmap(long_df, profile_index_df), "profile_heatmap.html")
    _add_fig("Score Trajectory", _fig_baseline_post(long_df),                     "baseline_post.html")

    _add_html(
        "Supplementary Analyses",
        _html_supplementary_analyses(long_df, mantel_pairs_df, mantel_meta),
        "supplementary_analyses.html",
    )

    # ── Profile Feature Correlation Network ─────────────────────────────────
    _add_fig(
        "Profile Feature Network",
        _fig_profile_network(
            centrality_df=network_centrality_df,
            edge_df=network_edge_df,
            layout_df=network_layout_df,
            global_metrics=network_global_metrics,
        ),
        "profile_network.html",
    )
    _add_html(
        "Profile Network Explorer",
        _html_profile_network_explorer(
            centrality_df=network_centrality_df,
            edge_df=network_edge_df,
            layout_df=network_layout_df,
            global_metrics=network_global_metrics,
        ),
        "profile_network_explorer.html",
    )

    # ── Semantic Embedding UMAP tab ─────────────────────────────────────────
    _add_html(
        "Semantic Embedding Space",
        _html_umap_embedding_tab(
            embedding_data_path=embedding_data_path,
            production_embedding_path=production_embedding_path,
        ),
        "umap_embedding.html",
    )
    _add_html(
        "Audit & Robustness",
        _html_quality_robustness(
            quality_diagnostics=quality_diagnostics,
            icc_data=icc_data,
            ridge_summary=ridge_summary,
            rf_summary=rf_summary,
            enet_summary=enet_summary,
            ladder_meta=ladder_meta,
        ),
        "audit_robustness.html",
    )

    # snapshots
    long_df.to_csv(snap_dir / "sem_long_encoded_snapshot.csv", index=False)
    if not profile_df.empty:
        profile_df.to_csv(snap_dir / "profile_level_effectivity_snapshot.csv", index=False)
    if not profile_index_df.empty:
        profile_index_df.to_csv(snap_dir / "profile_susceptibility_snapshot.csv", index=False)
    if not exploratory_df.empty:
        exploratory_df.to_csv(snap_dir / "moderator_coefficients_snapshot.csv", index=False)

    n_total = (
        n_profiles * n_attacks * n_opinions
        if isinstance(n_profiles, int) and isinstance(n_attacks, int) and isinstance(n_opinions, int)
        else "n/a"
    )
    notes = [
        "All profiles are attacked; the dashboard visualizes heterogeneity of manipulation outcomes, not a treatment-vs-control contrast.",
        f"This study uses a {n_profiles} x {n_attacks} x {n_opinions} ontology-driven factorial design ({len(long_df)} admissible scenarios out of {n_total} candidate tuples after structural compatibility filtering).",
        (
            f"Execution integrity: baseline fallback {float(quality_diagnostics.get('baseline_fallback_used_rate', 0.0)) * 100:.1f}%, "
            f"post fallback {float(quality_diagnostics.get('post_fallback_used_rate', 0.0)) * 100:.1f}%. Low fallback rates indicate clean API-backed elicitation."
            if quality_diagnostics
            else "Execution diagnostics were not available."
        ),
        "Adversarial effectivity (AE = delta x d_k) is non-negative by design in this run: the post-exposure score is constrained to the segment between the baseline and the adversarial goal, so <b>AE measures how far the opinion moved toward the goal</b> and AE near zero means the profile resisted.",
        "Key Findings answers the primary questions directly: which attacks are most effective, which opinions are most movable, which profile features amplify or protect, and how much predictive signal profiles add beyond context.",
        "Model Ladder: CV-R2 is computed with GroupKFold over profiles, so it measures generalization to unseen profiles; the profile increment is the honest effect-size statement for inter-individual differences.",
        "Moderation Paths and the Moderation Scan Heatmap show conditional moderation: within-attack cluster-robust slopes of AE on each z-scored profile feature, BH-FDR corrected across the full scan.",
        f"Conditional Susceptibility Estimator: configure any profile, choose any attack x opinion scope, then compare estimation approaches (ridge task models, k-NN over the {n_profiles} simulated profiles, context-only cell baseline, ensemble).",
        "Profile Network Explorer: circles = continuous features; diamonds = categorical/dummy. Shift+drag for lasso multi-select. Hulls outline detected communities. Path mode traces shortest network paths.",
        f"ICC(1) for absolute delta is {icc_str}; the Multilevel ICC tab decomposes variance into profile, attack, and opinion components.",
        f"SEM fit (CFI={float(fit['CFI']):.3f}, RMSEA={float(fit['RMSEA']):.3f}) is retained as a supplementary small-panel diagnostic, not as the primary moderation estimator." if fit.get("CFI") is not None and fit.get("RMSEA") is not None else "SEM fit indices were unavailable.",
    ]

    dashboard_path = output_root / "dashboard_results.html"
    dashboard_path.write_text(
        _render_dashboard_html(run_id, summary_cards, figure_divs, notes),
        encoding="utf-8",
    )
    visual_files.append(str(dashboard_path))

    return {
        "dashboard_path": str(dashboard_path),
        "visual_files":   visual_files,
        "summary_cards":  summary_cards,
    }
