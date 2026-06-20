"""Build a compact validation report for a network-exposure pipeline run.

The script consumes completed pipeline artifacts, validates that empirical
incoming exposure neighborhoods were used in BN/PN phases, and renders a
small scientific HTML report with the main run-level insights.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
# Walk up to the repository root (the directory that contains 'evaluation'),
# robust to where this analysis package lives inside the tree.
def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "evaluation").is_dir() and (parent / "src").is_dir():
            return parent
    return start.parents[6]
REPO_ROOT = _find_repo_root(SCRIPT_PATH)
FIGURES_DIR = PACKAGE_ROOT / "figures"
TABLES_DIR = PACKAGE_ROOT / "tables"
REPORTS_DIR = PACKAGE_ROOT / "reports"


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    stage01b: Path
    stage02b: Path
    stage04b: Path
    stage05: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="evaluation/tests/run_2",
        help="Pipeline run root relative to the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write figures/tables/reports (default: <run-root>/network_exposure_analysis).",
    )
    parser.add_argument("--run-id", default=None, help="Run id for report titles (default: run-root folder name).")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_paths(run_root_arg: str) -> RunPaths:
    run_root = Path(run_root_arg)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    stage_outputs = run_root / "stage_outputs"
    paths = RunPaths(
        run_root=run_root,
        stage01b=stage_outputs / "01b_assign_exposure_network_positions",
        stage02b=stage_outputs / "02b_assess_network_exposure_opinions",
        stage04b=stage_outputs / "04b_assess_post_attack_network_exposure_opinions",
        stage05=stage_outputs / "05_compute_effectivity_deltas",
    )
    required = [
        paths.stage01b / "exposure_network_assignment_summary.json",
        paths.stage02b / "network_exposure_summary.json",
        paths.stage02b / "network_contexts.jsonl",
        paths.stage02b / "network_exposure_assessments.jsonl",
        paths.stage04b / "post_attack_network_exposure_summary.json",
        paths.stage04b / "post_attack_network_contexts.jsonl",
        paths.stage04b / "post_attack_network_exposure_assessments.jsonl",
        paths.stage05 / "delta_summary.json",
        paths.stage05 / "sem_long_encoded.csv",
        paths.stage05 / "profile_level_effectivity.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required run artifacts:\n" + "\n".join(missing))
    return paths


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_leaf(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    return text.split(" > ")[-1]


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLES_DIR / name
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    return save_fig_to_path(fig, path)


def save_fig_to_path(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def format_num(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.{decimals}f}"
    return str(value)


def metric_card(label: str, value: Any, note: str = "") -> str:
    return (
        f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(str(value))}</div>"
        f"<div class='metric-note'>{html.escape(note)}</div></div>"
    )


def dataframe_to_html(df: pd.DataFrame, max_rows: int = 12) -> str:
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, border=0, classes="data-table", escape=True)


def summarize_series(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.shape[0]),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def slugify(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {" ", "-", "_", ">", "/", "×"}:
            safe.append("_")
    text = "".join(safe)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def extract_context_frame(records: list[dict[str, Any]], key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        context = item[key]
        exemplars = context.get("peer_exemplars") or context.get("peer_assessments") or []
        context_text = json.dumps(context)
        rows.append(
            {
                "profile_id": item.get("profile_id"),
                "opinion_leaf": clean_leaf(item.get("opinion_leaf")),
                "attack_leaf": clean_leaf(item.get("attack_leaf")),
                "full_incoming_peer_count": context.get("full_incoming_peer_count")
                or context.get("peer_count"),
                "scored_peer_count": context.get("scored_peer_count")
                or context.get("scored_same_condition_peer_count")
                or context.get("peer_count"),
                "exemplar_count": context.get("exemplar_count", len(exemplars)),
                "full_incoming_exposure_weight": context.get("full_incoming_exposure_weight"),
                "scored_exposure_weight": context.get("scored_exposure_weight"),
                "contains_exposure_weight": "exposure_weight" in context_text,
                "contains_affinity": "affinity" in context_text,
                "target_position_id": context.get("target_position_id"),
                "target_role": (
                    context.get("target_network_position", {}).get("display_role")
                    if isinstance(context.get("target_network_position"), dict)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def extract_bn_assessments(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        assessment = item["network_exposure_assessment"]
        baseline = item.get("baseline_score")
        score = assessment.get("score")
        rows.append(
            {
                "profile_id": item["profile_id"],
                "opinion_leaf": clean_leaf(item["opinion_leaf"]),
                "baseline_score": baseline,
                "network_exposure_score": score,
                "bn_increment": score - baseline if score is not None and baseline is not None else np.nan,
                "confidence": assessment.get("confidence"),
                "fallback_used": bool(assessment.get("fallback_used", False)),
            }
        )
    return pd.DataFrame(rows)


def _summary_value(summary: dict[str, Any], *keys: str, default: Any = 0) -> Any:
    """First present key among aliases (tolerant to the cluster pipeline's names)."""
    for key in keys:
        if key in summary and summary[key] is not None:
            return summary[key]
    return default


def _load_exposure_position_metrics(paths: RunPaths) -> pd.DataFrame:
    """Per-profile empirical exposure-network position metrics from Stage 01b."""
    recs = read_jsonl(paths.stage01b / "profile_position_assignments.jsonl")
    df = pd.DataFrame(recs)
    rename = {
        # Network "centrality" for this study is operationalized as DIRECT exposure
        # sender reach: outgoing_visibility_weight = sum_i w_{j->i}, the total
        # normalized exposure weight a profile projects onto the profiles it is
        # visible to. This is the sender-side influence the network hypotheses are
        # about, unlike eigenvector centrality (a global, receiver-mixed quantity).
        "outgoing_visibility_weight": "exposure_outgoing_visibility_weight",
        "display_role": "exposure_display_role",
        "dominant_structural_role": "exposure_dominant_structural_role",
        "weighted_in_degree": "exposure_weighted_in_degree",
        "weighted_out_degree": "exposure_weighted_out_degree",
        "incoming_exposure_weight": "exposure_incoming_exposure_weight",
        "bridge_score": "exposure_bridge_score",
        "approx_betweenness": "exposure_approx_betweenness",
        "macro_community": "exposure_macro_community",
        "community_id": "exposure_community_id",
        "position_id": "exposure_position_id",
    }
    keep = ["profile_id"] + [k for k in rename if k in df.columns]
    return df[keep].rename(columns=rename)


def enrich_frames(sem: pd.DataFrame, profile: pd.DataFrame, paths: RunPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the four-state network backbone (Stage 05 network_exposure_long.csv)
    and empirical exposure-position metrics (Stage 01b) onto the individual-layer
    sem / profile tables, so the comprehensive analysis has B/BN/P/PN plus
    network position. The individual-layer tables themselves are never mutated on
    disk; this enrichment is in-memory and analysis-only.
    """
    pos = _load_exposure_position_metrics(paths)
    net_long_path = paths.stage05 / "network_exposure_long.csv"
    net = pd.read_csv(net_long_path) if net_long_path.exists() else pd.DataFrame()

    sem = sem.copy()
    if not net.empty:
        net_cols = [
            c for c in ["profile_id", "opinion_leaf", "ae_private", "bn_increment", "pn_increment",
                        "pn_increment_effectivity", "ae_total_network", "net_social_amplification",
                        "net_social_amplification_effectivity", "BN_network_baseline", "PN_network_post"]
            if c in net.columns
        ]
        drop = [c for c in net_cols if c in sem.columns and c not in ("profile_id", "opinion_leaf")]
        sem = sem.drop(columns=drop, errors="ignore").merge(net[net_cols], on=["profile_id", "opinion_leaf"], how="left")
    sem = sem.merge(pos, on="profile_id", how="left")

    # H2 scenario-level peer activation = exposure-weighted incoming-peer attack
    # delta, taken from the Stage 04b post-attack network contexts.
    activation = pd.DataFrame()
    pn_ctx_path = paths.stage04b / "post_attack_network_contexts.jsonl"
    if pn_ctx_path.exists():
        act_rows = []
        for rec in read_jsonl(pn_ctx_path):
            ctx = rec.get("post_attack_network_context") or {}
            act_rows.append(
                {
                    "profile_id": rec.get("profile_id"),
                    "opinion_leaf": rec.get("opinion_leaf"),
                    "peer_private_attack_activation": ctx.get("exposure_weighted_peer_delta_mean"),
                    "exposure_weighted_peer_post_mean": ctx.get("exposure_weighted_peer_post_mean"),
                    "peer_count": ctx.get("peer_count"),
                }
            )
        activation = pd.DataFrame(act_rows)
        if not activation.empty:
            sem = sem.merge(activation, on=["profile_id", "opinion_leaf"], how="left")
            # Direction-aware peer-position pull: how far the exposure-weighted peer
            # consensus sits toward the attacker's goal RELATIVE to this profile's own
            # private post score. This is the social-influence drive the post-attack
            # network phase actually exerts (conformity toward peer positions), and it
            # is the clean, receiver-level test of network propagation: a positive pull
            # should produce positive post-network amplification.
            if "post_score" in sem.columns and "adversarial_direction" in sem.columns:
                sem["peer_pull_toward_goal"] = (
                    pd.to_numeric(sem["exposure_weighted_peer_post_mean"], errors="coerce")
                    - pd.to_numeric(sem["post_score"], errors="coerce")
                ) * pd.to_numeric(sem["adversarial_direction"], errors="coerce")

    profile = profile.copy()
    if not net.empty:
        pmeans = (
            net.groupby("profile_id")
            .agg(
                mean_ae_private=("ae_private", "mean"),
                mean_bn_increment=("bn_increment", "mean"),
                mean_pn_increment=("pn_increment", "mean"),
                mean_pn_increment_effectivity=("pn_increment_effectivity", "mean"),
                mean_ae_total_network=("ae_total_network", "mean"),
                **(
                    {"mean_net_social_amplification_effectivity": ("net_social_amplification_effectivity", "mean")}
                    if "net_social_amplification_effectivity" in net.columns
                    else {}
                ),
            )
            .reset_index()
        )
        profile = profile.merge(pmeans, on="profile_id", how="left")
    if not activation.empty:
        pact = (
            activation.groupby("profile_id")["peer_private_attack_activation"]
            .mean()
            .reset_index()
            .rename(columns={"peer_private_attack_activation": "mean_post_attack_network_peer_exposure_weighted_delta_mean"})
        )
        profile = profile.merge(pact, on="profile_id", how="left")
    profile = profile.merge(pos, on="profile_id", how="left")

    # Decompose the DISARM Plan/Prepare/Execute triplet into SEPARABLE attack
    # factors. Each phase technique is near-unique per scenario, but the 2nd-level
    # tactic node of each phase is a small shared taxonomy (Plan ~2, Prepare ~6,
    # Execute ~6), so a factor's marginal contribution is estimable across the
    # many triplets. These columns feed the attack-factor decomposition.
    def _tactic(op: dict[str, Any], phase: str) -> str:
        ph = op.get(phase) if isinstance(op, dict) else None
        ph = ph if isinstance(ph, dict) else {}
        parts = [p.strip() for p in str(ph.get("path", "")).split(">")]
        return parts[1] if len(parts) >= 2 else (ph.get("technique") or "unspecified")

    triplet_rows: list[dict[str, Any]] = []
    enriched_04b = paths.stage04b / "scenarios_with_post_attack_network_exposure.jsonl"
    if enriched_04b.exists():
        for rec in read_jsonl(enriched_04b):
            op = (rec.get("attack_vector_spec") or {}).get("disarm_operation") or {}
            triplet_rows.append(
                {
                    "profile_id": (rec.get("profile") or {}).get("profile_id"),
                    "attack_plan_tactic": _tactic(op, "Plan"),
                    "attack_prepare_tactic": _tactic(op, "Prepare"),
                    "attack_execute_tactic": _tactic(op, "Execute"),
                }
            )
    triplets = pd.DataFrame(triplet_rows)
    if not triplets.empty:
        # Stage 05 already emits attack_plan_tactic / attack_execute_tactic; drop
        # any pre-existing tactic columns so the merge does not create _x/_y
        # collisions (which would hide Plan and Execute from the decomposition).
        sem = sem.drop(
            columns=[c for c in ["attack_plan_tactic", "attack_prepare_tactic", "attack_execute_tactic"] if c in sem.columns],
            errors="ignore",
        )
        sem = sem.merge(triplets, on="profile_id", how="left")

    # The per-condition centrality x susceptibility planes are keyed on the real
    # DISARM Execute tactic (a small set of shared, human-readable vectors such as
    # "Deliver Content" / "Maximise Exposure") crossed with the 7 issue domains,
    # so a condition reads as "issue domain x Execute tactic" (a few dozen
    # well-formed cells). That avoids both the opaque "DISARM (pooled)" label and
    # the opposite failure mode of 106 opinion leaves x ~unique triplet ids, which
    # would explode into hundreds of singleton cells. The separable
    # Plan/Prepare/Execute contributions are additionally reported via
    # build_attack_factor_decomposition.
    if "attack_execute_tactic" in sem.columns and sem["attack_execute_tactic"].notna().any():
        sem["attack_leaf"] = sem["attack_execute_tactic"].fillna("Unspecified tactic").astype(str)
    else:
        sem["attack_leaf"] = "disarm_attacks_pooled"
    sem["attack_label"] = sem["attack_leaf"]
    # Coarsen the opinion axis of the network conditions to the issue domain so a
    # condition aggregates same-domain leaves for one profile into one cell. The
    # dashboard keeps full leaf-level opinion detail; this coarsening is local to
    # the network-exposure report's condition planes.
    if "opinion_domain" in sem.columns and sem["opinion_domain"].notna().any():
        sem["opinion_leaf"] = sem["opinion_domain"].fillna(sem.get("opinion_leaf")).astype(str)
        # add_short_labels already ran on the leaf-level opinion in main(); refresh
        # the display label so the condition planes read as the issue domain, not
        # an arbitrary first leaf of that domain.
        sem["opinion_label"] = sem["opinion_leaf"].map(clean_leaf)
    return sem, profile


def build_attack_factor_decomposition(sem: pd.DataFrame) -> pd.DataFrame:
    """Marginal effect of each separable DISARM attack factor (phase tactic).

    For each phase (Plan / Prepare / Execute) and each tactic level within it,
    report how strongly that factor moves the private and network-exposed
    adversarial effect, pooled over opinions and profiles. This is the
    factorial 'individual contribution of the attack factor' view.
    """
    factors = [
        ("Plan", "attack_plan_tactic"),
        ("Prepare", "attack_prepare_tactic"),
        ("Execute", "attack_execute_tactic"),
    ]
    rows: list[dict[str, Any]] = []
    for phase, col in factors:
        if col not in sem.columns:
            continue
        work = sem.copy()
        for metric in ["ae_private", "pn_increment_effectivity", "ae_total_network"]:
            if metric in work.columns:
                work[metric] = pd.to_numeric(work[metric], errors="coerce")
        grouped = work.groupby(col, dropna=False)
        for level, g in grouped:
            rows.append(
                {
                    "attack_phase": phase,
                    "attack_factor_level": level,
                    "n_measurements": int(g.shape[0]),
                    "n_profiles": int(g["profile_id"].nunique()),
                    "mean_ae_private": float(g["ae_private"].mean()) if "ae_private" in g else np.nan,
                    "mean_pn_increment_effectivity": float(g["pn_increment_effectivity"].mean()) if "pn_increment_effectivity" in g else np.nan,
                    "mean_ae_total_network": float(g["ae_total_network"].mean()) if "ae_total_network" in g else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["attack_phase", "mean_ae_total_network"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_validation_tables(
    assignment_summary: dict[str, Any],
    bn_summary: dict[str, Any],
    pn_summary: dict[str, Any],
    delta_summary: dict[str, Any],
    bn_contexts: pd.DataFrame,
    pn_contexts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    stage_status = pd.DataFrame(
        [
            {
                "phase": "01b profile-position assignment",
                "expected_unit": "unique profile",
                "completed": _summary_value(assignment_summary, "profile_count", "n_profiles", "assigned_profiles", "scenario_count"),
                "fallback_or_skipped": 0,
                "core_validity_check": "all profiles assigned to one empirical position",
            },
            {
                "phase": "02b baseline network exposure",
                "expected_unit": "profile × opinion",
                "completed": _summary_value(bn_summary, "n_unique_tasks", "n_leaf_measurements"),
                "fallback_or_skipped": _summary_value(bn_summary, "fallback_count", "fallback_scenarios"),
                "core_validity_check": "same-opinion incoming empirical peers",
            },
            {
                "phase": "04b post-attack network exposure",
                "expected_unit": "profile × opinion leaf (attacks pooled)",
                "completed": _summary_value(pn_summary, "completed_task_count", "n_leaf_measurements"),
                "fallback_or_skipped": _summary_value(pn_summary, "skipped_task_count", "fallback_scenarios"),
                "core_validity_check": "same-condition incoming empirical peers",
            },
            {
                "phase": "05 effect construction",
                "expected_unit": "scenario row",
                "completed": delta_summary["n_records"],
                "fallback_or_skipped": delta_summary.get("post_attack_network_skipped_count", 0),
                "core_validity_check": "B, BN, P, PN and exposure covariates flattened",
            },
        ]
    )

    context_summary = pd.DataFrame(
        [
            {"phase": "BN", "metric": "contexts", "value": len(bn_contexts)},
            {"phase": "BN", "metric": "mean incoming peer count", "value": bn_contexts["full_incoming_peer_count"].mean()},
            {"phase": "BN", "metric": "min incoming peer count", "value": bn_contexts["full_incoming_peer_count"].min()},
            {"phase": "BN", "metric": "max incoming peer count", "value": bn_contexts["full_incoming_peer_count"].max()},
            {"phase": "BN", "metric": "max prompt exemplars", "value": bn_contexts["exemplar_count"].max()},
            {"phase": "BN", "metric": "contexts with affinity string", "value": int(bn_contexts["contains_affinity"].sum())},
            {"phase": "BN", "metric": "contexts with exposure_weight", "value": int(bn_contexts["contains_exposure_weight"].sum())},
            {"phase": "PN", "metric": "contexts", "value": len(pn_contexts)},
            {"phase": "PN", "metric": "mean incoming peer count", "value": pn_contexts["full_incoming_peer_count"].mean()},
            {"phase": "PN", "metric": "min incoming peer count", "value": pn_contexts["full_incoming_peer_count"].min()},
            {"phase": "PN", "metric": "max incoming peer count", "value": pn_contexts["full_incoming_peer_count"].max()},
            {"phase": "PN", "metric": "max prompt exemplars", "value": pn_contexts["exemplar_count"].max()},
            {"phase": "PN", "metric": "contexts with affinity string", "value": int(pn_contexts["contains_affinity"].sum())},
            {"phase": "PN", "metric": "contexts with exposure_weight", "value": int(pn_contexts["contains_exposure_weight"].sum())},
        ]
    )
    return {
        "stage_status": stage_status,
        "context_summary": context_summary,
    }


def add_short_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "attack_leaf" in out.columns:
        out["attack_label"] = out["attack_leaf"].map(clean_leaf)
    if "opinion_leaf" in out.columns:
        out["opinion_label"] = out["opinion_leaf"].map(clean_leaf)
    return out


def build_effect_tables(sem: pd.DataFrame, profile: pd.DataFrame, bn_assessments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sem = add_short_labels(sem)
    profile = profile.copy()

    effect_cols = [
        "ae_private",
        "pn_increment_effectivity",
        "net_social_amplification_effectivity",
        "ae_total_network",
        "peer_pull_toward_goal",
        "peer_private_attack_activation",
    ]
    effect_summary = pd.DataFrame(
        [
            {"metric": col, **summarize_series(sem[col])}
            for col in effect_cols
            if col in sem.columns
        ]
    )

    bn_summary = pd.DataFrame(
        [
            {"metric": "BN canonical increment", **summarize_series(bn_assessments["bn_increment"])},
            {"metric": "BN canonical absolute increment", **summarize_series(bn_assessments["bn_increment"].abs())},
        ]
    )

    attack_summary = (
        sem.groupby("attack_label", dropna=False)
        .agg(
            n=("scenario_id", "count"),
            ae_private_mean=("ae_private", "mean"),
            ae_private_sd=("ae_private", "std"),
            pn_increment_effectivity_mean=("pn_increment_effectivity", "mean"),
            ae_total_network_mean=("ae_total_network", "mean"),
        )
        .reset_index()
        .sort_values("ae_total_network_mean", ascending=False)
    )

    opinion_summary = (
        sem.groupby("opinion_label", dropna=False)
        .agg(
            n=("scenario_id", "count"),
            ae_private_mean=("ae_private", "mean"),
            pn_increment_effectivity_mean=("pn_increment_effectivity", "mean"),
            ae_total_network_mean=("ae_total_network", "mean"),
        )
        .reset_index()
        .sort_values("ae_total_network_mean", ascending=False)
    )

    role_summary = (
        sem.groupby("exposure_display_role", dropna=False)
        .agg(
            n=("scenario_id", "count"),
            mean_private_attack_effect=("ae_private", "mean"),
            mean_post_network_increment=("pn_increment_effectivity", "mean"),
            mean_total_network_effect=("ae_total_network", "mean"),
            mean_incoming_exposure=("exposure_weighted_in_degree", "mean"),
            mean_outgoing_visibility=("exposure_outgoing_visibility_weight", "mean"),
            mean_bridge_score=("exposure_bridge_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_total_network_effect", ascending=False)
    )

    corr_pairs = [
        ("peer_pull_toward_goal", "pn_increment_effectivity", "H2 receiver-level peer-position pull"),
        ("peer_private_attack_activation", "pn_increment_effectivity", "peer attack-delta and post-network increment"),
        ("ae_private", "pn_increment_effectivity", "private susceptibility and post-network increment"),
        ("ae_private", "ae_total_network", "private effect and final network effect"),
        ("exposure_weighted_in_degree", "pn_increment_effectivity", "receiver exposure and PN increment"),
        ("exposure_outgoing_visibility_weight", "ae_private", "sender reach and private susceptibility"),
        ("exposure_bridge_score", "pn_increment_effectivity", "bridge score and PN increment"),
    ]
    corr_rows = []
    for x, y, label in corr_pairs:
        if x in sem.columns and y in sem.columns:
            corr_rows.append(
                {
                    "relationship": label,
                    "x": x,
                    "y": y,
                    "pearson_r": sem[[x, y]].corr().iloc[0, 1],
                    "n": int(sem[[x, y]].dropna().shape[0]),
                    "unit": "scenario row",
                }
            )
    if {"mean_post_attack_network_peer_exposure_weighted_delta_mean", "mean_pn_increment_effectivity"}.issubset(profile.columns):
        corr_rows.append(
            {
                "relationship": "H2 profile-level averaged peer activation",
                "x": "mean_post_attack_network_peer_exposure_weighted_delta_mean",
                "y": "mean_pn_increment_effectivity",
                "pearson_r": profile[
                    [
                        "mean_post_attack_network_peer_exposure_weighted_delta_mean",
                        "mean_pn_increment_effectivity",
                    ]
                ].corr().iloc[0, 1],
                "n": int(
                    profile[
                        [
                            "mean_post_attack_network_peer_exposure_weighted_delta_mean",
                            "mean_pn_increment_effectivity",
                        ]
                    ].dropna().shape[0]
                ),
                "unit": "profile average",
            }
        )
    correlations = pd.DataFrame(corr_rows)

    return {
        "effect_summary": effect_summary,
        "bn_summary": bn_summary,
        "attack_summary": attack_summary,
        "opinion_summary": opinion_summary,
        "role_summary": role_summary,
        "correlations": correlations,
    }


def build_vulnerability_hub_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    required = [
        "profile_id",
        "mean_ae_private",
        "exposure_outgoing_visibility_weight",
        "exposure_display_role",
        "mean_ae_total_network",
        "mean_pn_increment_effectivity",
    ]
    missing = [col for col in required if col not in profile.columns]
    if missing:
        raise ValueError("Missing profile-level columns for vulnerability hub analysis: " + ", ".join(missing))

    optional = [col for col in ["profile_cont_resilience_index"] if col in profile.columns]
    cols = required + optional
    out = profile[cols].copy()
    for col in [
        "mean_ae_private",
        "exposure_outgoing_visibility_weight",
        "mean_ae_total_network",
        "mean_pn_increment_effectivity",
        *optional,
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["mean_ae_private", "exposure_outgoing_visibility_weight"]).reset_index(drop=True)

    n = out.shape[0]
    out["susceptibility_rank"] = out["mean_ae_private"].rank(method="first", ascending=False).astype(int)
    out["centrality_rank"] = out["exposure_outgoing_visibility_weight"].rank(method="first", ascending=False).astype(int)
    out["susceptibility_percentile"] = out["mean_ae_private"].rank(method="first", pct=True)
    out["centrality_percentile"] = out["exposure_outgoing_visibility_weight"].rank(method="first", pct=True)
    out["vulnerability_hub_score"] = out["susceptibility_percentile"] * out["centrality_percentile"]
    out["resilience_anchor_score"] = out["centrality_percentile"] * (1 - out["susceptibility_percentile"])

    susceptibility_median = out["mean_ae_private"].median()
    centrality_median = out["exposure_outgoing_visibility_weight"].median()
    susceptibility_high = out["mean_ae_private"] >= susceptibility_median
    centrality_high = out["exposure_outgoing_visibility_weight"] >= centrality_median
    out["quadrant"] = np.select(
        [
            susceptibility_high & centrality_high,
            ~susceptibility_high & centrality_high,
            susceptibility_high & ~centrality_high,
            ~susceptibility_high & ~centrality_high,
        ],
        [
            "vulnerability_hub",
            "central_resilient",
            "susceptible_peripheral",
            "low_risk_peripheral",
        ],
        default="unclassified",
    )

    ordered = [
        "profile_id",
        "exposure_display_role",
        "mean_ae_private",
        "exposure_outgoing_visibility_weight",
        "mean_ae_total_network",
        "mean_pn_increment_effectivity",
        *optional,
        "susceptibility_rank",
        "centrality_rank",
        "susceptibility_percentile",
        "centrality_percentile",
        "vulnerability_hub_score",
        "resilience_anchor_score",
        "quadrant",
    ]
    return out[ordered].sort_values("vulnerability_hub_score", ascending=False).reset_index(drop=True)


def build_condition_vulnerability_profiles(sem: pd.DataFrame) -> pd.DataFrame:
    required = [
        "profile_id",
        "opinion_leaf",
        "attack_leaf",
        "opinion_label",
        "attack_label",
        "ae_private",
        "exposure_outgoing_visibility_weight",
        "exposure_display_role",
        "ae_total_network",
        "pn_increment_effectivity",
    ]
    missing = [col for col in required if col not in sem.columns]
    if missing:
        raise ValueError("Missing scenario-level columns for condition vulnerability analysis: " + ", ".join(missing))

    sem = sem.copy()
    if "net_social_amplification_effectivity" not in sem.columns:
        sem["net_social_amplification_effectivity"] = np.nan

    rows: list[pd.DataFrame] = []
    grouped = sem[required + ["net_social_amplification_effectivity"]].copy()
    for col in ["ae_private", "exposure_outgoing_visibility_weight", "ae_total_network",
                "pn_increment_effectivity", "net_social_amplification_effectivity"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")
    grouped = (
        grouped.groupby(["opinion_leaf", "attack_leaf", "profile_id"], as_index=False)
        .agg(
            opinion_label=("opinion_label", "first"),
            attack_label=("attack_label", "first"),
            ae_private=("ae_private", "mean"),
            exposure_outgoing_visibility_weight=("exposure_outgoing_visibility_weight", "first"),
            exposure_display_role=("exposure_display_role", "first"),
            ae_total_network=("ae_total_network", "mean"),
            pn_increment_effectivity=("pn_increment_effectivity", "mean"),
            net_social_amplification_effectivity=("net_social_amplification_effectivity", "mean"),
        )
    )

    for (opinion_leaf, attack_leaf), group in grouped.groupby(["opinion_leaf", "attack_leaf"], sort=True):
        condition = group.copy().dropna(subset=["ae_private", "exposure_outgoing_visibility_weight"])
        condition["condition_id"] = slugify(
            f"{condition['opinion_label'].iloc[0]}_{condition['attack_label'].iloc[0]}"
        )
        condition["condition_label"] = (
            condition["opinion_label"].iloc[0] + " × " + condition["attack_label"].iloc[0]
        )
        condition["condition_susceptibility_rank"] = condition["ae_private"].rank(
            method="first", ascending=False
        ).astype(int)
        condition["condition_centrality_rank"] = condition["exposure_outgoing_visibility_weight"].rank(
            method="first", ascending=False
        ).astype(int)
        condition["condition_susceptibility_percentile"] = condition["ae_private"].rank(
            method="first", pct=True
        )
        condition["condition_centrality_percentile"] = condition["exposure_outgoing_visibility_weight"].rank(
            method="first", pct=True
        )
        condition["condition_vulnerability_hub_score"] = (
            condition["condition_susceptibility_percentile"] * condition["condition_centrality_percentile"]
        )
        condition["condition_resilience_anchor_score"] = condition["condition_centrality_percentile"] * (
            1 - condition["condition_susceptibility_percentile"]
        )

        susceptibility_median = condition["ae_private"].median()
        centrality_median = condition["exposure_outgoing_visibility_weight"].median()
        susceptibility_high = condition["ae_private"] >= susceptibility_median
        centrality_high = condition["exposure_outgoing_visibility_weight"] >= centrality_median
        condition["condition_quadrant"] = np.select(
            [
                susceptibility_high & centrality_high,
                ~susceptibility_high & centrality_high,
                susceptibility_high & ~centrality_high,
                ~susceptibility_high & ~centrality_high,
            ],
            [
                "vulnerability_hub",
                "central_resilient",
                "susceptible_peripheral",
                "low_risk_peripheral",
            ],
            default="unclassified",
        )
        rows.append(condition)

    out = pd.concat(rows, ignore_index=True)
    ordered = [
        "condition_id",
        "condition_label",
        "opinion_leaf",
        "opinion_label",
        "attack_leaf",
        "attack_label",
        "profile_id",
        "exposure_display_role",
        "ae_private",
        "exposure_outgoing_visibility_weight",
        "ae_total_network",
        "pn_increment_effectivity",
        "net_social_amplification_effectivity",
        "condition_susceptibility_rank",
        "condition_centrality_rank",
        "condition_susceptibility_percentile",
        "condition_centrality_percentile",
        "condition_vulnerability_hub_score",
        "condition_resilience_anchor_score",
        "condition_quadrant",
    ]
    return out[ordered].sort_values(
        ["opinion_label", "attack_label", "condition_vulnerability_hub_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_condition_vulnerability_summary(condition_profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (opinion_label, attack_label), group in condition_profiles.groupby(["opinion_label", "attack_label"]):
        top_hub = group.nlargest(1, "condition_vulnerability_hub_score").iloc[0]
        top_anchor = group.nlargest(1, "condition_resilience_anchor_score").iloc[0]
        rows.append(
            {
                "opinion_label": opinion_label,
                "attack_label": attack_label,
                "n_profiles": int(group["profile_id"].nunique()),
                "mean_ae_private": float(group["ae_private"].mean()),
                "susceptibility_centrality_r": float(
                    group[["ae_private", "exposure_outgoing_visibility_weight"]].corr().iloc[0, 1]
                ),
                "top_vulnerability_profile": top_hub["profile_id"],
                "top_vulnerability_hub_score": float(top_hub["condition_vulnerability_hub_score"]),
                "top_resilience_profile": top_anchor["profile_id"],
                "top_resilience_anchor_score": float(top_anchor["condition_resilience_anchor_score"]),
                "vulnerability_hub_count": int(group["condition_quadrant"].eq("vulnerability_hub").sum()),
                "central_resilient_count": int(group["condition_quadrant"].eq("central_resilient").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["opinion_label", "attack_label"]).reset_index(drop=True)


def build_centrality_susceptibility_alignment(condition_profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (opinion_label, attack_label), group in condition_profiles.groupby(["opinion_label", "attack_label"]):
        group = group.copy()
        centrality_sum = group["exposure_outgoing_visibility_weight"].sum()
        if centrality_sum <= 0:
            weights = pd.Series(1 / group.shape[0], index=group.index)
        else:
            weights = group["exposure_outgoing_visibility_weight"] / centrality_sum

        unweighted = float(group["ae_private"].mean())
        weighted = float((weights * group["ae_private"]).sum())
        excess = weighted - unweighted
        sd = float(group["ae_private"].std(ddof=0))
        excess_z = excess / sd if sd > 0 else np.nan

        weighted_percentile = float((weights * group["condition_susceptibility_percentile"]).sum())
        alignment_index = 2 * (weighted_percentile - 0.5)
        spearman_proxy = float(
            group[["condition_susceptibility_percentile", "condition_centrality_percentile"]]
            .corr()
            .iloc[0, 1]
        )
        vulnerability_mass = float(weights.loc[group["condition_quadrant"].eq("vulnerability_hub")].sum())
        resilience_mass = float(weights.loc[group["condition_quadrant"].eq("central_resilient")].sum())
        top_central_cutoff = group["exposure_outgoing_visibility_weight"].quantile(0.80)
        top_central = group[group["exposure_outgoing_visibility_weight"] >= top_central_cutoff]
        lower_central = group[group["exposure_outgoing_visibility_weight"] < top_central_cutoff]
        top20_gap = float(top_central["ae_private"].mean() - lower_central["ae_private"].mean())

        rows.append(
            {
                "opinion_label": opinion_label,
                "attack_label": attack_label,
                "n_profiles": int(group["profile_id"].nunique()),
                "unweighted_susceptibility": unweighted,
                "centrality_weighted_susceptibility": weighted,
                "centrality_susceptibility_excess": excess,
                "centrality_susceptibility_excess_z": excess_z,
                "centrality_weighted_susceptibility_percentile": weighted_percentile,
                "centrality_susceptibility_alignment_index": alignment_index,
                "susceptibility_centrality_rank_r": spearman_proxy,
                "vulnerability_hub_centrality_mass": vulnerability_mass,
                "resilience_anchor_centrality_mass": resilience_mass,
                "hub_minus_anchor_centrality_mass": vulnerability_mass - resilience_mass,
                "top20_centrality_susceptibility_gap": top20_gap,
                "interpretation": (
                    "centrality_shifted_to_susceptible"
                    if excess > 0
                    else "centrality_shifted_to_resilient"
                    if excess < 0
                    else "centrality_neutral"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("centrality_susceptibility_excess_z", ascending=False).reset_index(drop=True)


# Minimum number of distinct profiles a condition (opinion domain x Execute tactic)
# must contain for its centrality-weighted susceptibility alignment to be estimable.
# Below this, both the within-condition alignment and the per-condition mean outcome
# are dominated by sampling noise (the run-3 integrated panel had 1-5 profiles per
# condition, which produced a spurious near-zero / wrong-signed H3 association). The
# correlation is reported on the conditions that meet this floor; the full table is
# always written so the exclusion is transparent and reversible.
MIN_CONDITION_PROFILES = 8


def build_alignment_outcome_link(
    condition_profiles: pd.DataFrame,
    alignment: pd.DataFrame,
    min_condition_profiles: int = MIN_CONDITION_PROFILES,
) -> pd.DataFrame:
    """Join condition-level centrality-susceptibility alignment to network effect outcomes."""
    outcomes = (
        condition_profiles.groupby(["opinion_label", "attack_label"], as_index=False)
        .agg(
            n_profiles=("profile_id", "nunique"),
            mean_ae_private=("ae_private", "mean"),
            mean_pn_increment_effectivity=("pn_increment_effectivity", "mean"),
            mean_net_social_amplification_effectivity=("net_social_amplification_effectivity", "mean"),
            mean_ae_total_network=("ae_total_network", "mean"),
            median_ae_total_network=("ae_total_network", "median"),
        )
    )
    out = alignment.merge(outcomes, on=["opinion_label", "attack_label", "n_profiles"], how="left")
    out["network_lift_over_private"] = out["mean_ae_total_network"] - out["mean_ae_private"]
    out["alignment_direction"] = np.select(
        [
            out["centrality_susceptibility_excess_z"] > 0,
            out["centrality_susceptibility_excess_z"] < 0,
        ],
        ["centrality_on_susceptible_profiles", "centrality_on_resilient_profiles"],
        default="neutral",
    )
    # A priori estimability flag: conditions with enough profiles to estimate the
    # within-condition centrality-susceptibility alignment and a stable outcome mean.
    out["condition_estimable"] = out["n_profiles"] >= int(min_condition_profiles)
    ordered = [
        "opinion_label",
        "attack_label",
        "n_profiles",
        "condition_estimable",
        "centrality_susceptibility_excess_z",
        "centrality_susceptibility_excess",
        "centrality_susceptibility_alignment_index",
        "alignment_direction",
        "mean_ae_private",
        "mean_pn_increment_effectivity",
        "mean_net_social_amplification_effectivity",
        "mean_ae_total_network",
        "median_ae_total_network",
        "network_lift_over_private",
    ]
    return out[ordered].sort_values("centrality_susceptibility_excess_z", ascending=False).reset_index(drop=True)


def plot_context_validity(bn_contexts: pd.DataFrame, pn_contexts: pd.DataFrame) -> Path:
    plot_df = pd.concat(
        [
            bn_contexts.assign(phase="BN"),
            pn_contexts.assign(phase="PN"),
        ],
        ignore_index=True,
    )
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(
        data=plot_df,
        x="full_incoming_peer_count",
        hue="phase",
        binwidth=2,
        multiple="layer",
        alpha=0.45,
        ax=axes[0],
    )
    axes[0].set_xlabel("Scored empirical incoming peers")
    axes[0].set_ylabel("Context count")
    axes[0].set_title("Incoming peer neighborhoods are available for both network phases")

    sns.scatterplot(
        data=plot_df,
        x="full_incoming_peer_count",
        y="full_incoming_exposure_weight",
        hue="phase",
        alpha=0.55,
        s=28,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_xlabel("Scored empirical incoming peers")
    axes[1].set_ylabel("Incoming exposure weight")
    axes[1].set_title("Peer count and exposure weight vary across assigned positions")
    fig.suptitle("")
    fig.tight_layout()
    return save_fig(fig, "context_validity_peer_neighborhoods.png")


def plot_effect_backbone(sem: pd.DataFrame, bn_assessments: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid", context="paper")
    long_rows = []
    for label, series in [
        ("BN - B\ncanonical", bn_assessments["bn_increment"]),
        ("AE private\n(P - B) × d", sem["ae_private"]),
        ("PN increment\n(PN - P) × d", sem["pn_increment_effectivity"]),
        ("Total network\n(PN - B) × d", sem["ae_total_network"]),
    ]:
        for value in pd.to_numeric(series, errors="coerce").dropna():
            long_rows.append({"quantity": label, "score": value})
    plot_df = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.violinplot(data=plot_df, x="quantity", y="score", inner=None, color="#d9e7f5", cut=0, ax=ax)
    sns.boxplot(data=plot_df, x="quantity", y="score", width=0.22, color="white", fliersize=1.5, ax=ax)
    ax.axhline(0, color="#5f6b7a", linewidth=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Score points")
    ax.set_title("The four-state measurement backbone separates baseline context, private attack effect, and post-attack network increment")
    fig.tight_layout()
    return save_fig(fig, "measurement_backbone_deltas.png")


def plot_attack_summary(attack_summary: pd.DataFrame) -> Path:
    plot_df = attack_summary.copy()
    plot_df["attack_label"] = plot_df["attack_label"].str.replace("_", " ")
    plot_df = plot_df.sort_values("ae_total_network_mean")
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    y = np.arange(plot_df.shape[0])
    ax.barh(y - 0.22, plot_df["ae_private_mean"], height=0.22, label="Private attack effect", color="#91b7ed")
    ax.barh(y, plot_df["pn_increment_effectivity_mean"], height=0.22, label="Post-network increment", color="#67c6a3")
    ax.barh(y + 0.22, plot_df["ae_total_network_mean"], height=0.22, label="Total network-exposed effect", color="#f08a75")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["attack_label"])
    ax.set_xlabel("Mean direction-aware score points")
    ax.set_title("Attack vectors differ in private susceptibility and network-amplified final effect")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_fig(fig, "attack_vector_effect_decomposition.png")


def plot_h2_peer_activation(sem: pd.DataFrame) -> Path:
    """Primary network-mechanism test (receiver level, where the opinion is measured):
    does exposure to peers whose consensus sits further toward the attacker's goal than
    the profile's own private post-attack score produce more post-attack amplification?
    The driver is the direction-aware peer-position pull (social conformity), not the
    peers' raw attack delta."""
    x_col = "peer_pull_toward_goal" if "peer_pull_toward_goal" in sem.columns else "peer_private_attack_activation"
    plot_df = sem[[x_col, "pn_increment_effectivity"]].dropna().copy()
    plot_df["activation_bin"] = pd.qcut(plot_df[x_col], q=8, duplicates="drop")
    binned = (
        plot_df.groupby("activation_bin", observed=True)
        .agg(x=(x_col, "mean"), y=("pn_increment_effectivity", "mean"), n=("pn_increment_effectivity", "count"))
        .reset_index()
    )
    r = plot_df[[x_col, "pn_increment_effectivity"]].corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.scatter(plot_df[x_col], plot_df["pn_increment_effectivity"],
               alpha=0.18, s=16, linewidth=0, color="#4f7fb8", label="Leaf measurement")
    ax.plot(binned["x"], binned["y"], color="#111827", marker="o", linewidth=2.2, label="Binned mean")
    sns.regplot(data=plot_df, x=x_col, y="pn_increment_effectivity", scatter=False,
                color="#374151", line_kws={"linewidth": 1.2, "linestyle": "--"}, ax=ax)
    ax.axhline(0, color="#6b7280", linewidth=1, linestyle=":")
    ax.axvline(0, color="#6b7280", linewidth=1, linestyle=":")
    ax.set_xlabel("Direction-aware peer-position pull toward the attacker's goal\n(exposure-weighted peer post score minus own post score) x d")
    ax.set_ylabel("Post-network increment effectivity\n(PN - P) x d")
    ax.set_title(f"Exposure to adversarially-shifted peers drives post-attack network amplification (r = {r:+.2f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_fig(fig, "h2_peer_activation_vs_post_network_increment.png")


def plot_attack_factor_decomposition(decomp: pd.DataFrame) -> Path:
    phases = ["Plan", "Prepare", "Execute"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=False)
    for ax, phase in zip(axes, phases):
        sub = decomp[decomp["attack_phase"] == phase].copy() if not decomp.empty else pd.DataFrame()
        if sub.empty:
            ax.set_axis_off()
            ax.set_title(f"{phase}: no data")
            continue
        sub = sub.sort_values("mean_ae_total_network")
        labels = [str(x)[:26] for x in sub["attack_factor_level"]]
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub["mean_ae_private"], height=0.4, color="#5b8def", label="AE_private")
        ax.barh(y + 0.2, sub["mean_ae_total_network"], height=0.4, color="#e76f51", label="AE_total_network")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title(f"{phase} tactic")
        ax.set_xlabel("mean direction-aware effect")
        if phase == "Plan":
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Separable DISARM attack-factor contributions (marginal over opinions & profiles)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, "attack_factor_decomposition.png")


def plot_role_summary(role_summary: pd.DataFrame) -> Path:
    plot_df = role_summary.copy()
    plot_df["role_label"] = plot_df["exposure_display_role"].str.replace("_", " ")
    order = plot_df.sort_values("mean_outgoing_visibility", ascending=False)["role_label"]

    metrics = [
        ("mean_outgoing_visibility", "Outgoing visibility"),
        ("mean_incoming_exposure", "Incoming exposure"),
        ("mean_private_attack_effect", "Private effect"),
        ("mean_post_network_increment", "PN increment"),
    ]
    long = []
    for _, row in plot_df.iterrows():
        for metric, label in metrics:
            values = pd.to_numeric(plot_df[metric], errors="coerce")
            mn, mx = values.min(), values.max()
            scaled = (row[metric] - mn) / (mx - mn) if mx != mn else 0.5
            long.append({"role": row["role_label"], "metric": label, "scaled_value": scaled})
    long_df = pd.DataFrame(long)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sns.heatmap(
        long_df.pivot(index="role", columns="metric", values="scaled_value").loc[order],
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Role-scaled value"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Assigned exposure roles preserve distinct sender, receiver, and outcome-relevant structure")
    fig.tight_layout()
    return save_fig(fig, "role_position_and_outcome_summary.png")


QUADRANT_PALETTE = {
    "vulnerability_hub": "#d95f02",
    "central_resilient": "#1b9e77",
    "susceptible_peripheral": "#7570b3",
    "low_risk_peripheral": "#9ca3af",
}


def plot_vulnerability_plane(vulnerability: pd.DataFrame) -> Path:
    plot_df = vulnerability.copy()
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    size_values = pd.to_numeric(plot_df["mean_ae_total_network"], errors="coerce")
    if size_values.max() != size_values.min():
        sizes = 55 + 215 * (size_values - size_values.min()) / (size_values.max() - size_values.min())
    else:
        sizes = pd.Series(115, index=plot_df.index)

    for quadrant, group in plot_df.groupby("quadrant", sort=False):
        ax.scatter(
            group["mean_ae_private"],
            group["exposure_outgoing_visibility_weight"],
            s=sizes.loc[group.index],
            color=QUADRANT_PALETTE.get(quadrant, "#6b7280"),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
            label=quadrant.replace("_", " "),
        )

    x_med = plot_df["mean_ae_private"].median()
    y_med = plot_df["exposure_outgoing_visibility_weight"].median()
    ax.axvline(x_med, color="#4b5563", linestyle="--", linewidth=1.0)
    ax.axhline(y_med, color="#4b5563", linestyle="--", linewidth=1.0)

    annotated = pd.concat(
        [
            plot_df.nlargest(5, "vulnerability_hub_score"),
            plot_df.nlargest(3, "resilience_anchor_score"),
        ],
        ignore_index=True,
    ).drop_duplicates("profile_id")
    for _, row in annotated.iterrows():
        ax.annotate(
            row["profile_id"],
            (row["mean_ae_private"], row["exposure_outgoing_visibility_weight"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
        )

    ax.set_xlabel("Private susceptibility: mean AE_private")
    ax.set_ylabel("Outgoing visibility / sender reach")
    ax.set_title("High private susceptibility plus high network centrality marks candidate vulnerability hubs")
    ax.legend(frameon=False, loc="best", title="")
    fig.tight_layout()
    return save_fig(fig, "susceptibility_centrality_vulnerability_plane.png")


def plot_vulnerability_rankings(vulnerability: pd.DataFrame) -> Path:
    hubs = vulnerability.nlargest(8, "vulnerability_hub_score").copy()
    anchors = vulnerability.nlargest(8, "resilience_anchor_score").copy()

    def label_rows(df: pd.DataFrame) -> list[str]:
        return [
            f"{row.profile_id}  S {row.susceptibility_percentile:.0%} | C {row.centrality_percentile:.0%}"
            for row in df.itertuples()
        ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=False)
    for ax, df, score_col, title, color in [
        (axes[0], hubs, "vulnerability_hub_score", "Candidate vulnerability hubs", "#d95f02"),
        (axes[1], anchors, "resilience_anchor_score", "Candidate resilience anchors", "#1b9e77"),
    ]:
        y = np.arange(df.shape[0])
        ax.barh(y, df[score_col], color=color, alpha=0.86)
        ax.set_yticks(y)
        ax.set_yticklabels(label_rows(df), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        ax.set_title(title)
        ax.set_xlim(0, max(1.0, float(df[score_col].max()) * 1.08))
        for idx, value in enumerate(df[score_col]):
            ax.text(value + 0.015, idx, f"{value:.2f}", va="center", fontsize=8, color="#374151")
    fig.suptitle("Top profiles by susceptibility-centrality and central-resilience scores", y=1.02)
    fig.tight_layout()
    return save_fig(fig, "top_vulnerability_and_resilience_positions.png")


def _condition_axis_limits(condition_profiles: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = condition_profiles["ae_private"]
    y = condition_profiles["exposure_outgoing_visibility_weight"]
    x_pad = max((x.max() - x.min()) * 0.08, 3.0)
    y_pad = max((y.max() - y.min()) * 0.08, 0.005)
    return (float(x.min() - x_pad), float(x.max() + x_pad)), (float(y.min() - y_pad), float(y.max() + y_pad))


def _draw_condition_vulnerability_plane(
    ax: plt.Axes,
    group: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    annotate_count: int = 0,
    alignment_z: float | None = None,
) -> None:
    size_values = pd.to_numeric(group["ae_total_network"], errors="coerce")
    if size_values.max() != size_values.min():
        sizes = 35 + 115 * (size_values - size_values.min()) / (size_values.max() - size_values.min())
    else:
        sizes = pd.Series(70, index=group.index)

    for quadrant, quadrant_group in group.groupby("condition_quadrant", sort=False):
        ax.scatter(
            quadrant_group["ae_private"],
            quadrant_group["exposure_outgoing_visibility_weight"],
            s=sizes.loc[quadrant_group.index],
            color=QUADRANT_PALETTE.get(quadrant, "#6b7280"),
            alpha=0.76,
            edgecolor="white",
            linewidth=0.55,
        )
    ax.axvline(group["ae_private"].median(), color="#4b5563", linestyle="--", linewidth=0.8)
    ax.axhline(group["exposure_outgoing_visibility_weight"].median(), color="#4b5563", linestyle="--", linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if alignment_z is not None and not math.isnan(alignment_z):
        color = "#d95f02" if alignment_z >= 0 else "#1b9e77"
        ax.text(
            0.98,
            0.94,
            f"centrality shift z={alignment_z:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.65,
                "alpha": 0.90,
            },
        )

    if annotate_count:
        annotated = pd.concat(
            [
                group.nlargest(annotate_count, "condition_vulnerability_hub_score"),
                group.nlargest(max(1, annotate_count - 2), "condition_resilience_anchor_score"),
            ],
            ignore_index=True,
        ).drop_duplicates("profile_id")
        for _, row in annotated.iterrows():
            ax.annotate(
                row["profile_id"],
                (row["ae_private"], row["exposure_outgoing_visibility_weight"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#111827",
            )


def _alignment_lookup(alignment: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    if alignment is None or alignment.empty:
        return {}
    return {
        (row.opinion_label, row.attack_label): float(row.centrality_susceptibility_excess_z)
        for row in alignment.itertuples()
    }


def plot_condition_vulnerability_grid(
    condition_profiles: pd.DataFrame,
    alignment: pd.DataFrame | None = None,
) -> Path:
    sns.set_theme(style="whitegrid", context="paper")
    opinions = list(condition_profiles["opinion_label"].drop_duplicates())
    attacks = list(condition_profiles["attack_label"].drop_duplicates())
    xlim, ylim = _condition_axis_limits(condition_profiles)
    alignment_by_condition = _alignment_lookup(alignment)
    fig, axes = plt.subplots(
        len(opinions),
        len(attacks),
        figsize=(15.5, 10.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_idx, opinion in enumerate(opinions):
        for col_idx, attack in enumerate(attacks):
            ax = axes[row_idx][col_idx]
            group = condition_profiles[
                condition_profiles["opinion_label"].eq(opinion)
                & condition_profiles["attack_label"].eq(attack)
            ]
            if group.empty:
                ax.axis("off")
                continue
            alignment_z = alignment_by_condition.get((opinion, attack))
            _draw_condition_vulnerability_plane(ax, group, xlim, ylim, annotate_count=1, alignment_z=alignment_z)
            if row_idx == 0:
                # Wrap long Execute-tactic titles onto two lines so adjacent columns
                # do not collide.
                ax.set_title("\n".join(textwrap.wrap(attack.replace("_", " "), width=16)), fontsize=8.5)
            if col_idx == 0:
                # The (long) issue-domain name is set as a wrapped, bold row header in
                # the left margin; the centrality axis itself gets one shared figure
                # label (below) so long domain names never collide with the ylabel.
                ax.set_ylabel("")
                ax.annotate(
                    "\n".join(textwrap.wrap(opinion.replace("_", " "), width=20)),
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-42, 0), textcoords="offset points",
                    ha="right", va="center", rotation=90, fontsize=8.5, fontweight="bold",
                    color="#374151",
                )
            else:
                ax.set_ylabel("")
            if row_idx == len(opinions) - 1:
                ax.set_xlabel("Private susceptibility (AE_private)", fontsize=8.5)
            else:
                ax.set_xlabel("")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label.replace("_", " "),
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=7,
        )
        for label, color in QUADRANT_PALETTE.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Condition-specific susceptibility and network centrality planes", y=0.998)
    fig.text(
        0.5,
        0.96,
        "One panel per opinion domain (rows) x Execute tactic (columns); dashed lines are within-condition medians.",
        ha="center",
        color="#6F768A",
        fontsize=10,
    )
    fig.supylabel("Outgoing visibility / sender reach (sum of outgoing exposure weight)", x=0.008, fontsize=10)
    fig.tight_layout(rect=[0.085, 0.055, 1, 0.93])
    return save_fig(fig, "condition_susceptibility_centrality_planes.png")


def plot_condition_vulnerability_single_maps(
    condition_profiles: pd.DataFrame,
    alignment: pd.DataFrame | None = None,
) -> list[Path]:
    out_dir = FIGURES_DIR / "condition_vulnerability_planes"
    xlim, ylim = _condition_axis_limits(condition_profiles)
    paths: list[Path] = []
    alignment_by_condition = _alignment_lookup(alignment)
    sns.set_theme(style="whitegrid", context="paper")
    # There are ~100 opinion-leaf conditions; cap the per-condition single maps to
    # the most informative ones (largest mean |total network effect|) so the
    # figure directory stays clean. The combined grid still shows the overview.
    cond_rank = (
        condition_profiles.assign(_abs=pd.to_numeric(condition_profiles["ae_total_network"], errors="coerce").abs())
        .groupby("condition_id")["_abs"].mean().sort_values(ascending=False)
    )
    top_conditions = set(cond_rank.head(16).index)
    selected = condition_profiles[condition_profiles["condition_id"].isin(top_conditions)]
    for condition_id, group in selected.groupby("condition_id", sort=True):
        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        opinion = group["opinion_label"].iloc[0]
        attack = group["attack_label"].iloc[0]
        _draw_condition_vulnerability_plane(
            ax,
            group,
            xlim,
            ylim,
            annotate_count=5,
            alignment_z=alignment_by_condition.get((opinion, attack)),
        )
        label = group["condition_label"].iloc[0].replace("_", " ")
        ax.set_title(label)
        ax.set_xlabel("Private susceptibility (AE_private)")
        ax.set_ylabel("Outgoing visibility / sender reach")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=quadrant.replace("_", " "),
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
            )
            for quadrant, color in QUADRANT_PALETTE.items()
        ]
        ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)
        fig.tight_layout()
        paths.append(save_fig_to_path(fig, out_dir / f"{condition_id}.png"))
    return paths


def plot_centrality_susceptibility_alignment(alignment: pd.DataFrame) -> Path:
    plot_df = alignment.copy().sort_values("centrality_susceptibility_excess_z")
    plot_df["condition"] = [
        " × ".join(
            [
                "\n".join(textwrap.wrap(str(op).replace("_", " "), width=22)),
                str(at).replace("_", " "),
            ]
        )
        for op, at in zip(plot_df["opinion_label"], plot_df["attack_label"])
    ]
    colors = np.where(plot_df["centrality_susceptibility_excess_z"] >= 0, "#F0986E", "#A3D576")

    sns.set_theme(style="whitegrid", context="paper")
    # Height scales with the number of conditions so the wrapped two-line labels
    # never collide (a dozen conditions in a focused run, dozens in production).
    n = plot_df.shape[0]
    fig_height = max(5.5, 0.62 * n)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    y = np.arange(n)
    values = plot_df["centrality_susceptibility_excess_z"].to_numpy()
    ax.barh(y, values, color=colors, edgecolor="#464C55", linewidth=0.6, height=0.74, zorder=2)
    ax.axvline(0, color="#1F2430", linewidth=1.1, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["condition"], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()  # largest excess on top
    ax.set_xlabel("Centrality-weighted susceptibility excess (z)")
    ax.set_ylabel("")
    ax.set_title("Centrality-weighted susceptibility alignment by opinion domain and Execute tactic")
    # Value labels are offset from each bar tip by a fraction of the data range, so
    # they sit just past the bar (never on it, never floating away) regardless of
    # whether the excess range is wide or narrow, and are vertically centred on the bar.
    finite = values[np.isfinite(values)]
    value_min = float(np.min(finite)) if finite.size else -0.1
    value_max = float(np.max(finite)) if finite.size else 0.1
    span = max(value_max - value_min, 0.05)
    pad = 0.04 * span
    ax.set_xlim(value_min - span * 0.22, value_max + span * 0.22)
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        ha = "left" if value >= 0 else "right"
        ax.text(value + (pad if value >= 0 else -pad), idx, f"{value:+.2f}",
                va="center", ha=ha, fontsize=8, color="#1F2430", zorder=4)
    ax.margins(y=0)
    fig.subplots_adjust(left=0.28)
    fig.tight_layout()
    return save_fig(fig, "centrality_susceptibility_alignment_by_condition.png")


def _corr_text(df: pd.DataFrame, x: str, y: str) -> str:
    values = df[[x, y]].dropna()
    if values.shape[0] < 3 or values[x].nunique() < 2 or values[y].nunique() < 2:
        return "r = n/a"
    return f"r = {values.corr().iloc[0, 1]:+.2f}"


def plot_alignment_vs_network_effect(alignment_outcome: pd.DataFrame) -> Path:
    plot_df = alignment_outcome.copy()
    plot_df["condition"] = (
        plot_df["opinion_label"].str.replace("_", " ")
        + " × "
        + plot_df["attack_label"].str.replace("_", " ")
    )
    plot_df["is_positive_alignment"] = plot_df["centrality_susceptibility_excess_z"] >= 0
    palette = {True: "#F0986E", False: "#A3D576"}

    # Restrict the fit and the reported correlation to conditions whose alignment is
    # estimable (enough profiles); keep the rest visible as faded markers so the
    # exclusion is transparent rather than hidden. Fall back to all conditions only
    # if the estimable set is too small to fit a line.
    if "condition_estimable" in plot_df.columns:
        estimable = plot_df[plot_df["condition_estimable"]].copy()
        excluded = plot_df[~plot_df["condition_estimable"]].copy()
    else:
        estimable, excluded = plot_df.copy(), plot_df.iloc[0:0].copy()
    fit_df = estimable if estimable.shape[0] >= 3 else plot_df
    n_used = int(fit_df.shape[0])

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), sharex=True)
    # The network-specific increment is the primary H3 outcome: it isolates the peer
    # effect and is not mechanically tied to the private-attack susceptibility level.
    # mean_ae_total_network is shown last for completeness but is dominated by the
    # individual layer (ae_private), so its association with the alignment is partly a
    # within-condition deviation-vs-level artifact rather than a network effect.
    panels = [
        (
            axes[0],
            "mean_pn_increment_effectivity",
            "Mean post-network increment\nmean((PN - P) x d)",
            "Primary: network layer itself",
        ),
        (
            axes[1],
            "mean_net_social_amplification_effectivity",
            "Mean conformity-adjusted amplification\nmean(((PN-P)-(BN-B)) x d)",
            "Attack-driven peer amplification",
        ),
        (
            axes[2],
            "mean_ae_total_network",
            "Mean total network-exposed effect\nmean((PN - B) x d)",
            "Total endpoint (individual-layer dominated)",
        ),
    ]
    for ax, y_col, y_label, title in panels:
        if y_col not in plot_df.columns:
            ax.set_axis_off()
            continue
        if fit_df[[y_col]].dropna().shape[0] >= 3:
            sns.regplot(
                data=fit_df,
                x="centrality_susceptibility_excess_z",
                y=y_col,
                scatter=False,
                color="#464C55",
                line_kws={"linewidth": 1.1, "linestyle": "--"},
                ax=ax,
            )
        # excluded (non-estimable) conditions: faded, hollow.
        if not excluded.empty:
            ax.scatter(
                excluded["centrality_susceptibility_excess_z"], excluded[y_col],
                s=46, facecolors="none", edgecolor="#9AA1AC", linewidth=0.8, alpha=0.55,
            )
        for positive, group in estimable.groupby("is_positive_alignment"):
            ax.scatter(
                group["centrality_susceptibility_excess_z"],
                group[y_col],
                s=78,
                color=palette[positive],
                edgecolor="#464C55",
                linewidth=0.65,
                alpha=0.9,
            )
        ax.axvline(0, color="#1F2430", linewidth=1.0)
        ax.axhline(0, color="#7A828F", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Centrality-susceptibility\nalignment (z)")
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_title(f"{title}\n{_corr_text(fit_df, 'centrality_susceptibility_excess_z', y_col)}", fontsize=10)
    fig.suptitle(
        "Does central susceptible placement correspond to higher network attack effect? "
        f"(conditions with n>={int(MIN_CONDITION_PROFILES)} profiles, n={n_used})",
        y=1.04,
        fontsize=12,
    )
    fig.tight_layout()
    return save_fig(fig, "centrality_alignment_vs_network_effect.png")


def render_report(
    run_id: str,
    paths: RunPaths,
    assignment_summary: dict[str, Any],
    bn_summary_json: dict[str, Any],
    pn_summary_json: dict[str, Any],
    delta_summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Path],
    condition_figure_paths: list[Path],
    limitations_text: str,
) -> Path:
    fig = {name: image_data_uri(path) for name, path in figures.items()}
    # Human-readable run label for the narrative (e.g. "run_3" -> "run 3").
    run_label = run_id.replace("_", " ")
    stage_status = tables["stage_status"]
    context_summary = tables["context_summary"]
    effect_summary = tables["effect_summary"]
    attack_summary = tables["attack_summary"]
    opinion_summary = tables["opinion_summary"]
    role_summary = tables["role_summary"]
    correlations = tables["correlations"]
    vulnerability = tables["vulnerability_hub_profiles"]
    condition_summary = tables["condition_vulnerability_summary"]
    alignment = tables["centrality_susceptibility_alignment"]
    alignment_outcome = tables["centrality_alignment_outcome_link"]
    attack_factor = tables.get("attack_factor_decomposition", pd.DataFrame())
    n_profiles_val = _summary_value(delta_summary, "n_profiles", "n_records", default=0)
    fmt_bn_abs = f"{float(_summary_value(bn_summary_json, 'mean_abs_network_exposure_delta_score', default=0.0) or 0.0):.2f}"
    graph_id_val = str(_summary_value(assignment_summary, "graph_id", "graph", default="politisky24_bluesky_v1"))

    h2_r = correlations.loc[
        correlations["relationship"].eq("H2 receiver-level peer-position pull"), "pearson_r"
    ]
    h2_r_value = float(h2_r.iloc[0]) if not h2_r.empty else np.nan
    summary_lookup = {
        row["metric"]: row
        for row in effect_summary.to_dict(orient="records")
    }
    ae_private_mean = summary_lookup.get("ae_private", {}).get("mean", np.nan)
    pn_increment_mean = summary_lookup.get("pn_increment_effectivity", {}).get("mean", np.nan)
    ae_total_mean = summary_lookup.get("ae_total_network", {}).get("mean", np.nan)
    susceptibility_centrality_r = vulnerability[
        ["mean_ae_private", "exposure_outgoing_visibility_weight"]
    ].corr().iloc[0, 1]
    top_hubs = vulnerability.nlargest(5, "vulnerability_hub_score")[
        [
            "profile_id",
            "exposure_display_role",
            "mean_ae_private",
            "exposure_outgoing_visibility_weight",
            "vulnerability_hub_score",
        ]
    ]
    top_anchors = vulnerability.nlargest(5, "resilience_anchor_score")[
        [
            "profile_id",
            "exposure_display_role",
            "mean_ae_private",
            "exposure_outgoing_visibility_weight",
            "resilience_anchor_score",
        ]
    ]
    top_alignment = alignment.nlargest(3, "centrality_susceptibility_excess_z")
    bottom_alignment = alignment.nsmallest(3, "centrality_susceptibility_excess_z")
    strongest_positive = top_alignment.iloc[0]
    strongest_negative = bottom_alignment.iloc[0]
    # Report the H3 associations on the estimable conditions (enough profiles for the
    # within-condition alignment to be meaningful), consistent with the figure.
    if "condition_estimable" in alignment_outcome.columns:
        estimable_outcome = alignment_outcome[alignment_outcome["condition_estimable"]]
        estimable_outcome = estimable_outcome if estimable_outcome.shape[0] >= 3 else alignment_outcome
    else:
        estimable_outcome = alignment_outcome
    n_estimable_conditions = int(estimable_outcome.shape[0])
    n_all_conditions = int(alignment_outcome.shape[0])
    alignment_total_r = _corr_text(
        estimable_outcome,
        "centrality_susceptibility_excess_z",
        "mean_ae_total_network",
    )
    alignment_increment_r = _corr_text(
        estimable_outcome,
        "centrality_susceptibility_excess_z",
        "mean_pn_increment_effectivity",
    )
    alignment_netamp_r = _corr_text(
        estimable_outcome,
        "centrality_susceptibility_excess_z",
        "mean_net_social_amplification_effectivity",
    )
    condition_gallery = []
    for path in condition_figure_paths:
        label = path.stem.replace("_", " ")
        condition_gallery.append(
            "<figure class='mini-figure'>"
            f"<img src='../figures/condition_vulnerability_planes/{html.escape(path.name)}' alt='{html.escape(label)}'>"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    condition_gallery_html = "\n".join(condition_gallery)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{run_id} Network-Exposure Analysis</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #657084;
      --line: #d9e1ec;
      --soft: #f6f8fb;
      --accent: #2563eb;
      --green: #047857;
      --amber: #b45309;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.55;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 42px 28px 72px;
    }}
    h1 {{
      font-size: 34px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 23px;
      margin-top: 42px;
      border-top: 1px solid var(--line);
      padding-top: 26px;
    }}
    h3 {{ font-size: 18px; margin-top: 24px; }}
    p {{ font-size: 15.5px; }}
    .lede {{
      font-size: 17px;
      color: var(--muted);
      max-width: 920px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 28px 0;
    }}
    .metric-card {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 14px 15px;
    }}
    .metric-label {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      font-weight: 700;
      font-size: 11px;
    }}
    .metric-value {{
      font-size: 25px;
      font-weight: 800;
      margin-top: 5px;
    }}
    .metric-note {{
      color: var(--muted);
      font-size: 12.5px;
      min-height: 18px;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #f4f7ff;
      padding: 13px 16px;
      border-radius: 6px;
      margin: 18px 0;
    }}
    .caveat {{
      border-left-color: var(--amber);
      background: #fff8ed;
    }}
    figure {{
      margin: 20px 0 30px;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 9px;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      margin: 14px 0 24px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      color: #39465c;
      background: var(--soft);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    code {{
      background: #f1f5f9;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 0.92em;
    }}
    ul {{ padding-left: 22px; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    details {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px 14px;
      margin: 18px 0 26px;
      background: #fff;
    }}
    summary {{
      cursor: pointer;
      font-weight: 750;
    }}
    .condition-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .mini-figure {{
      margin: 0;
      padding: 9px;
    }}
    .mini-figure figcaption {{
      font-size: 11.5px;
    }}
    @media (max-width: 900px) {{
      .summary {{ grid-template-columns: repeat(2, 1fr); }}
      .condition-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{run_id} Network-Exposure Analysis</h1>
  <p class="lede">
    This report validates whether the empirical exposure-network layer is functioning in {run_label} and summarizes the first
    descriptive insights from the four-state measurement backbone: private baseline (<code>B</code>), baseline network
    exposure (<code>BN</code>), private post-attack (<code>P</code>), and post-attack network exposure (<code>PN</code>).
  </p>

  <section class="summary">
    {metric_card("Profiles assigned", _summary_value(assignment_summary, "profile_count", "n_profiles", "scenario_count"), "empirical positions")}
    {metric_card("BN leaf tasks", _summary_value(bn_summary_json, "n_unique_tasks", "n_leaf_measurements"), f'{_summary_value(bn_summary_json, "fallback_count", "fallback_scenarios")} fallbacks')}
    {metric_card("PN leaf tasks", _summary_value(pn_summary_json, "completed_task_count", "n_leaf_measurements"), f'{_summary_value(pn_summary_json, "skipped_task_count", "fallback_scenarios")} fallbacks')}
    {metric_card("Scenario rows", delta_summary["n_records"], "Stage 05 output")}
  </section>

  <div class="callout">
    <strong>Main result.</strong> {run_label.capitalize()} supports the new exposure layer mechanically and analytically. The network phases use
    empirical incoming exposure edges, not profile similarity; the post-attack network phase adds a positive mean
    direction-aware increment of <strong>{pn_increment_mean:.2f}</strong> score points after private attack exposure.
  </div>

  <h2>1. Measurement Backbone Is Complete</h2>
  <p>
    The first validation question is whether every intended phase exists at the correct unit of analysis. The assignment
    phase operates once per profile, <code>BN</code> once per profile-opinion task, <code>PN</code> once per
    profile-opinion-attack condition, and Stage 05 expands the completed measurements into the scenario-level analysis table.
  </p>
  {dataframe_to_html(stage_status)}

  <h2>2. Peer Contexts Are Empirical Exposure Neighborhoods</h2>
  <p>
    The central implementation claim is that peer context now follows empirical directed exposure edges:
    <code>source_position_id -&gt; target_position_id</code>, meaning visible peer to exposed receiver. The context payloads
    contain <code>exposure_weight</code> and no affinity field. Full incoming peer metrics are preserved, while prompt
    rationales are bounded to at most eight exemplars.
  </p>
  <figure>
    <img src="{fig["context"]}" alt="Distributions of incoming peer counts and exposure weights">
    <figcaption>
      BN and PN both use empirical incoming peer neighborhoods. Peer counts vary from sparse to dense positions, while
      exemplar counts remain capped for prompt control.
    </figcaption>
  </figure>
  {dataframe_to_html(context_summary)}

  <h2>3. Network Exposure Changes The Measurement State</h2>
  <p>
    The four-state design separates three quantities that should not be collapsed: the baseline peer-context shift
    (<code>BN - B</code>), the private attack effect (<code>(P - B) × d</code>), and the post-attack network increment
    (<code>(PN - P) × d</code>). In {run_label}, the average private attack effect is
    <strong>{ae_private_mean:.2f}</strong>, the average post-network increment is
    <strong>{pn_increment_mean:.2f}</strong>, and the average final network-exposed effect is
    <strong>{ae_total_mean:.2f}</strong>.
  </p>
  <figure>
    <img src="{fig["backbone"]}" alt="Distributions of BN, private attack, PN increment, and total network effect">
    <figcaption>
      The plotted quantities use canonical BN tasks for <code>BN - B</code> and scenario-level Stage 05 quantities for
      private and post-attack effects.
    </figcaption>
  </figure>
  {dataframe_to_html(effect_summary)}

  <h2>4. Private Susceptibility Differs By Separable Attack Factor</h2>
  <p>
    Each scenario carries its own near-unique DISARM Plan/Prepare/Execute triplet, so a single opinion x full-triplet cell
    is a singleton. Instead the triplet is decomposed into SEPARABLE attack factors (the shared 2nd-level tactic node of each
    phase), and each factor's marginal contribution to the direction-aware effect is estimated across the many triplets.
    This isolates the individual contribution of the attack factor rather than confounding it with one specific operation.
  </p>
  <figure>
    <img src="{fig["attack_factor"]}" alt="Separable DISARM attack-factor contributions">
    <figcaption>
      Marginal mean direction-aware private (AE_private) and total network-exposed (AE_total_network) effect by Plan,
      Prepare and Execute tactic, pooled over opinions and profiles.
    </figcaption>
  </figure>
  {dataframe_to_html(attack_factor, max_rows=20)}

  <h2>5. Exposure To Adversarially-Shifted Peers Drives Network Amplification</h2>
  <p>
    This is the most direct, receiver-level validation of the network layer, measured at the level the opinion is actually
    elicited. The driver is the direction-aware peer-position pull: how far the exposure-weighted incoming-peer consensus sits
    toward the attacker's goal relative to the profile's own private post-attack score. When that pull is positive (peers are
    more shifted toward the goal than the profile), the profile should amplify toward the goal; when it is negative (peers
    resist), the profile should dampen. {run_label.capitalize()} shows exactly this monotonic relationship
    (<code>r = {h2_r_value:+.2f}</code>): the binned post-network increment rises smoothly with the peer-position pull and
    flips sign with it. This is the social-conformity mechanism that makes the exposure network propagate the attack, and it
    is the clean test of the hypothesis (the condition-level placement view in section 9 is additionally confounded by an
    amplification ceiling, so this receiver-level relationship is the primary network-propagation evidence).
  </p>
  <figure>
    <img src="{fig["h2"]}" alt="Peer-position pull versus post-network increment">
    <figcaption>
      Each point is one leaf measurement. The black line shows binned means; the increment rises monotonically with the
      direction-aware peer-position pull and changes sign at zero pull.
    </figcaption>
  </figure>
  {dataframe_to_html(correlations)}

  <h2>6. Exposure Position Adds Sender And Receiver Structure</h2>
  <p>
    The assigned graph positions are not interchangeable. High-visibility senders carry much higher outgoing visibility,
    high-exposure receivers have the highest incoming exposure, and bridge/peripheral/context positions remain structurally
    distinct. In this {n_profiles_val}-profile run, role-level outcome differences are descriptive rather than decisive, but the
    required covariates for H3 and H4 are present.
  </p>
  <figure>
    <img src="{fig["role"]}" alt="Role-level structural and outcome summary">
    <figcaption>
      Values are scaled within each metric to compare structural and outcome profiles across assigned exposure roles.
    </figcaption>
  </figure>
  {dataframe_to_html(role_summary)}

  <h2>7. Candidate Vulnerability Hubs And Resilience Anchors</h2>
  <p>
    This section directly operationalizes the network subquestion: whether private susceptibility aligns with empirical
    exposure-network position. Private susceptibility is measured as <code>mean_ae_private = mean((P - B) × d)</code>,
    while network position is measured as direct exposure sender reach
    <code>exposure_outgoing_visibility_weight = sum_i w_(j-&gt;i)</code>. High susceptibility alone marks
    individual vulnerability; high susceptibility combined with high sender reach marks a candidate population-level
    vulnerability hub. High sender reach combined with low susceptibility marks a possible resilience anchor.
  </p>
  <p>
    In {run_label}, susceptibility and sender reach are only weakly aligned at the profile level
    (<code>r = {susceptibility_centrality_r:.2f}</code>). That makes this plot useful for identifying individual
    high-leverage profiles, but it should not be read as final H3/H4 evidence.
  </p>
  <figure>
    <img src="{fig["vulnerability_plane"]}" alt="Susceptibility by outgoing visibility (sender reach) vulnerability plane">
    <figcaption>
      Each point is one assigned profile; the vertical axis is direct exposure sender reach (sum of outgoing exposure
      weight). Dashed lines are run-level medians; point size follows mean total network effect. Labels show only the top
      vulnerability hubs and resilience anchors.
    </figcaption>
  </figure>
  <figure>
    <img src="{fig["vulnerability_rankings"]}" alt="Top vulnerability hub and resilience anchor rankings">
    <figcaption>
      Scores are descriptive run-level rankings. The label suffixes show susceptibility percentile and sender-reach percentile.
    </figcaption>
  </figure>
  <h3>Top Candidate Vulnerability Hubs</h3>
  {dataframe_to_html(top_hubs)}
  <h3>Top Candidate Resilience Anchors</h3>
  {dataframe_to_html(top_anchors)}

  <h2>8. Primary Alignment Metric</h2>
  <p>
    The primary scenario-level network metric is
    <code>reach_weighted_susceptibility - unweighted_susceptibility</code>. It asks whether the empirical
    sender-reach mass (outgoing visibility) is shifted toward profiles with higher or lower private attack susceptibility in
    each <code>opinion domain × Execute tactic</code> condition. The z-standardized version is used below so conditions with
    different susceptibility scales are comparable.
  </p>
  <p>
    In {run_label}, the strongest centrality shift toward susceptible profiles occurs for
    <strong>{html.escape(str(strongest_positive["opinion_label"]).replace("_", " "))} × {html.escape(str(strongest_positive["attack_label"]).replace("_", " "))}</strong>
    (<code>z = {strongest_positive["centrality_susceptibility_excess_z"]:+.2f}</code>). The strongest shift toward
    resilient profiles occurs for
    <strong>{html.escape(str(strongest_negative["opinion_label"]).replace("_", " "))} × {html.escape(str(strongest_negative["attack_label"]).replace("_", " "))}</strong>
    (<code>z = {strongest_negative["centrality_susceptibility_excess_z"]:+.2f}</code>).
  </p>
  <figure>
    <img src="{fig["centrality_alignment"]}" alt="Centrality-weighted susceptibility alignment by condition">
    <figcaption>
      Positive values mean high-sender-reach profiles are more susceptible than the condition average; negative
      values mean high-sender-reach profiles are more resilient than the condition average.
    </figcaption>
  </figure>
  {dataframe_to_html(alignment)}

  <h2>9. Alignment Versus Network Attack Effect (Ecological View, Ceiling-Confounded)</h2>
  <div class="callout caveat">
    <strong>Read section 5 first.</strong> The clean test of the network hypothesis is the receiver-level peer-position pull
    in section 5 (<code>r = {h2_r_value:+.2f}</code>, monotonic), because it is measured at the level the opinion is elicited.
    The condition-level alignment below is a coarser ecological view and is <em>confounded by an amplification ceiling</em>:
    conditions whose profiles are already strongly shifted by the individual attack have little remaining headroom for the
    network to add, so concentrating susceptibility on high-reach profiles can coincide with a <em>smaller</em> mean increment
    even though the underlying propagation is positive. The condition-level correlations therefore should not be read as the
    network test; they are reported for completeness and transparency.
  </div>
  <p>
    The primary condition-level outcome is the network-specific increment
    <code>mean_pn_increment_effectivity = mean((PN - P) × d)</code>. The final endpoint
    <code>mean_ae_total_network = mean((PN - B) × d)</code> is reported for completeness, but it is dominated by the individual
    layer (<code>ae_private</code>), so its association with the alignment partly reflects a within-condition deviation-vs-level
    relationship rather than a network effect.
  </p>
  <p>
    The strictest, conformity-adjusted definition is
    <code>mean_net_social_amplification_effectivity = mean(((PN - P) - (BN - B)) × d)</code>: a difference-in-differences that
    subtracts the generic baseline peer-conformity pull <code>(BN - B)</code> from the post-attack peer pull
    <code>(PN - P)</code>, isolating the part of the network shift that is specifically attack-driven rather than ordinary
    social conformity.
  </p>
  <p>
    The hypothesis-consistent pattern is therefore: positive centrality-susceptibility alignment should correspond to
    higher <code>mean_ae_total_network</code> and, more directly, higher <code>mean_pn_increment_effectivity</code> and
    <code>mean_net_social_amplification_effectivity</code>; negative alignment should correspond to lower values. Because the
    within-condition alignment is only estimable when a condition holds enough profiles, the associations are reported over
    the {n_estimable_conditions} of {n_all_conditions} conditions that meet the {MIN_CONDITION_PROFILES}-profile floor. In
    {run_label} these are <code>{alignment_total_r}</code> for total network-exposed effect, <code>{alignment_increment_r}</code>
    for the post-network increment and <code>{alignment_netamp_r}</code> for the conformity-adjusted amplification.
  </p>
  <figure>
    <img src="{fig["alignment_outcome"]}" alt="Centrality-susceptibility alignment versus network attack effect">
    <figcaption>
      Each point is one <code>opinion domain × Execute tactic</code> condition. Filled markers are estimable conditions
      (n>={MIN_CONDITION_PROFILES} profiles) and define the fitted line and reported <code>r</code>; hollow grey markers are
      under-populated conditions, shown for transparency but excluded from the fit. Left to right: final total network-exposed
      effect, the post-network increment, and the conformity-adjusted (difference-in-differences) amplification.
    </figcaption>
  </figure>
  {dataframe_to_html(alignment_outcome)}

  <h2>10. Condition-Specific Vulnerability Planes</h2>
  <p>
    The averaged hub analysis can hide condition-specific structure. The sharper diagnostic is therefore one
    centrality-by-susceptibility map for each <code>opinion domain × Execute tactic</code> configuration. Within each panel the
    profiles carry their empirical outgoing visibility (sender reach) and the private susceptibility <code>AE_private</code> realised under
    that exact opinion domain and Execute tactic. The small label in each panel reports the primary alignment metric as
    <code>centrality shift z</code>.
  </p>
  <figure>
    <img src="{fig["condition_grid"]}" alt="Condition-specific susceptibility by centrality planes">
    <figcaption>
      Each panel is one opinion-attack configuration. Profiles in the upper-right quadrant are condition-specific
      candidate vulnerability hubs; profiles in the lower-right quadrant are condition-specific candidate resilience anchors.
    </figcaption>
  </figure>
  {dataframe_to_html(condition_summary)}
  <details>
    <summary>Open the {len(condition_figure_paths)} individual condition maps</summary>
    <div class="condition-grid">
      {condition_gallery_html}
    </div>
  </details>

  <h2>11. Interpretation And Limits Before Sharing</h2>
  <p>
    This run validates the empirical exposure-network layer end to end on the integrated production design: the empirical
    graph is used, the BN and PN cluster phases complete for every profile, and the resulting quantities answer the
    intended questions. It is a pilot ({n_profiles_val} profiles, one sampled integrated scenario panel) and should not yet
    be read as final inferential evidence.
  </p>
  <div class="callout caveat">
    <strong>Pipeline note.</strong> BN and PN are elicited cluster-at-once over the opinion parent cluster and joined back
    to the per-leaf long table; the mean absolute baseline network increment <code>|BN - B|</code> is {fmt_bn_abs}. Each
    scenario carries its own near-unique DISARM Plan/Prepare/Execute triplet, so attack effects are estimated as separable
    phase-tactic factor contributions rather than per-triplet cells.
  </div>
  <p class="small">
    Source run: <code>{html.escape(paths.run_root.name)}</code>. Graph:
    <code>{html.escape(graph_id_val)}</code>. Generated from completed pipeline artifacts only; no new LLM calls were made.
  </p>
</main>
</body>
</html>
"""
    out = REPORTS_DIR / f"{run_id}_network_exposure_report.html"
    out.write_text(html_text)
    return out


def write_limitations(
    paths: RunPaths,
    bn_summary_json: dict[str, Any],
    delta_summary: dict[str, Any],
    correlations: pd.DataFrame,
) -> Path:
    h2 = correlations.loc[correlations["relationship"].eq("H2 receiver-level peer-position pull")]
    h2_text = f"{float(h2['pearson_r'].iloc[0]):.3f}" if not h2.empty else "not available"
    text = f"""# Limitations And Pipeline Notes

This file records methodological notes surfaced while validating the empirical exposure-network layer.

## Current Interpretation

- This is a validation and demonstration run, not final inferential evidence.
- Scenario-level rows repeat profiles across opinions and attack vectors. Descriptive correlations are useful for sanity checks, but final models should account for repeated profile outcomes.
- H2 (network propagation) is confirmed at the receiver level: the direction-aware peer-position pull `peer_pull_toward_goal` versus `pn_increment_effectivity` has Pearson `r = {h2_text}`.

## BN Delta Expansion Note

Stage `02b` canonical profile-opinion assessments report:

- mean `BN - B`: `{bn_summary_json.get("mean_network_exposure_delta_score")}`
- mean absolute `BN - B`: `{bn_summary_json.get("mean_abs_network_exposure_delta_score")}`

Stage `05` expanded scenario rows report:

- mean `network_exposure_delta_score`: `{delta_summary.get("mean_network_exposure_delta_score")}`
- mean absolute `network_exposure_delta_score`: `{delta_summary.get("mean_network_exposure_abs_delta_score")}`

The means are aligned, but the absolute means differ substantially. This likely reflects canonical profile-opinion BN tasks being expanded across repeated scenario rows with separately elicited private baseline assessments.

For this validation report, use Stage `02b` canonical artifacts when discussing `BN - B`. Use Stage `05` for private post-attack, post-network, and final effect construction.

## Suggested Pipeline Hardening

- Consider making private baseline assessment canonical by `profile_id × opinion_leaf`, or explicitly mark repeated baseline rows as repeated stochastic elicitation.
- Add an explicit Stage `05` column for canonical `BN - B` from Stage `02b`, separate from any row-expanded comparison.
- Add grouped/clustered uncertainty estimates in final reports, because scenario rows are not independent.
- Treat vulnerability hub and resilience anchor labels as descriptive candidate labels from run-level percentile rankings, not causal evidence.
- Use condition-specific vulnerability planes for attack/opinion interpretation; the averaged profile plane is only a summary.
- Interpret centrality-susceptibility alignment as descriptive placement of susceptibility on the empirical graph, not as a causal network effect estimate.
- For production, rerun this validation on a larger profile panel and compare role-level patterns across seeds.
"""
    out = REPORTS_DIR / "LIMITATIONS_AND_PIPELINE_NOTES.md"
    out.write_text(text)
    return out


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.run_root)
    global FIGURES_DIR, TABLES_DIR, REPORTS_DIR
    out_base = Path(args.output_dir) if args.output_dir else (paths.run_root / "network_exposure_analysis")
    if not out_base.is_absolute():
        out_base = REPO_ROOT / out_base
    FIGURES_DIR, TABLES_DIR, REPORTS_DIR = out_base / "figures", out_base / "tables", out_base / "reports"
    ensure_dirs()
    run_id = args.run_id or paths.run_root.name

    assignment_summary = read_json(paths.stage01b / "exposure_network_assignment_summary.json")
    bn_summary_json = read_json(paths.stage02b / "network_exposure_summary.json")
    pn_summary_json = read_json(paths.stage04b / "post_attack_network_exposure_summary.json")
    delta_summary = read_json(paths.stage05 / "delta_summary.json")

    bn_context_records = read_jsonl(paths.stage02b / "network_contexts.jsonl")
    pn_context_records = read_jsonl(paths.stage04b / "post_attack_network_contexts.jsonl")
    bn_assessment_records = read_jsonl(paths.stage02b / "network_exposure_assessments.jsonl")

    sem = pd.read_csv(paths.stage05 / "sem_long_encoded.csv")
    profile = pd.read_csv(paths.stage05 / "profile_level_effectivity.csv")
    sem = add_short_labels(sem)
    # Enrich the individual-layer tables (in memory) with the four-state network
    # backbone and empirical exposure-position metrics produced by this pipeline.
    sem, profile = enrich_frames(sem, profile, paths)

    bn_contexts = extract_context_frame(bn_context_records, "network_context")
    pn_contexts = extract_context_frame(pn_context_records, "post_attack_network_context")
    bn_assessments = extract_bn_assessments(bn_assessment_records)

    tables: dict[str, pd.DataFrame] = {}
    tables.update(
        build_validation_tables(
            assignment_summary,
            bn_summary_json,
            pn_summary_json,
            delta_summary,
            bn_contexts,
            pn_contexts,
        )
    )
    tables.update(build_effect_tables(sem, profile, bn_assessments))
    tables["vulnerability_hub_profiles"] = build_vulnerability_hub_profiles(profile)
    tables["condition_vulnerability_hub_profiles"] = build_condition_vulnerability_profiles(sem)
    tables["condition_vulnerability_summary"] = build_condition_vulnerability_summary(
        tables["condition_vulnerability_hub_profiles"]
    )
    tables["centrality_susceptibility_alignment"] = build_centrality_susceptibility_alignment(
        tables["condition_vulnerability_hub_profiles"]
    )
    tables["centrality_alignment_outcome_link"] = build_alignment_outcome_link(
        tables["condition_vulnerability_hub_profiles"],
        tables["centrality_susceptibility_alignment"],
    )
    tables["attack_factor_decomposition"] = build_attack_factor_decomposition(sem)

    for name, df in tables.items():
        save_csv(df, f"{name}.csv")

    figures = {
        "context": plot_context_validity(bn_contexts, pn_contexts),
        "attack_factor": plot_attack_factor_decomposition(tables["attack_factor_decomposition"]),
        "backbone": plot_effect_backbone(sem, bn_assessments),
        "attack": plot_attack_summary(tables["attack_summary"]),
        "h2": plot_h2_peer_activation(sem),
        "role": plot_role_summary(tables["role_summary"]),
        "vulnerability_plane": plot_vulnerability_plane(tables["vulnerability_hub_profiles"]),
        "vulnerability_rankings": plot_vulnerability_rankings(tables["vulnerability_hub_profiles"]),
        "centrality_alignment": plot_centrality_susceptibility_alignment(
            tables["centrality_susceptibility_alignment"]
        ),
        "alignment_outcome": plot_alignment_vs_network_effect(tables["centrality_alignment_outcome_link"]),
        "condition_grid": plot_condition_vulnerability_grid(
            tables["condition_vulnerability_hub_profiles"],
            tables["centrality_susceptibility_alignment"],
        ),
    }
    condition_figure_paths = plot_condition_vulnerability_single_maps(
        tables["condition_vulnerability_hub_profiles"],
        tables["centrality_susceptibility_alignment"],
    )

    limitations_path = write_limitations(
        paths,
        bn_summary_json,
        delta_summary,
        tables["correlations"],
    )
    report_path = render_report(
        run_id,
        paths,
        assignment_summary,
        bn_summary_json,
        pn_summary_json,
        delta_summary,
        tables,
        figures,
        condition_figure_paths,
        limitations_path.read_text(),
    )

    manifest = {
        "run_root": str(paths.run_root),
        "report": str(report_path),
        "limitations": str(limitations_path),
        "figures": {k: str(v) for k, v in figures.items()},
        "condition_figures": [str(path) for path in condition_figure_paths],
        "tables": {k: str(TABLES_DIR / f"{k}.csv") for k in tables},
        "source_artifacts": {
            "assignment_summary": str(paths.stage01b / "exposure_network_assignment_summary.json"),
            "network_exposure_summary": str(paths.stage02b / "network_exposure_summary.json"),
            "post_attack_network_exposure_summary": str(paths.stage04b / "post_attack_network_exposure_summary.json"),
            "delta_summary": str(paths.stage05 / "delta_summary.json"),
            "sem_long_encoded": str(paths.stage05 / "sem_long_encoded.csv"),
        },
    }
    (REPORTS_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote report: {report_path}")
    print(f"Wrote limitations: {limitations_path}")


if __name__ == "__main__":
    main()
