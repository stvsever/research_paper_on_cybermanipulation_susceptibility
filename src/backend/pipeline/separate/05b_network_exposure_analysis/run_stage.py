from __future__ import annotations

"""
Stage 05b - Empirical exposure-network analysis (additive, gated).

Runs only when the exposure-network side branches (02b / 04b) produced output,
i.e. when the run was launched with --with-network-exposure. It reads:
  * Stage 05 `network_exposure_long.csv` (per profile x opinion leaf: the four
    measurement states B / BN / P / PN and the hypothesis deltas), and
  * Stage 01b `profile_position_assignments.jsonl` (per-profile empirical
    network-position metrics: centrality, reach, bridge, role, community).

It joins outcome deltas onto network position, tests the four exposure-network
hypotheses (H1-H4), and writes concise machine-readable complements:
  * per_profile_network_outcomes.csv
  * network_position_outcome_correlations.csv
  * network_hypotheses.json

It then invokes the comprehensive exposure-network report builder (the sibling
report_builder/ package) adapted to this pipeline's cluster / DISARM-triplet /
production-ontology outputs, which emits
~13 figures, the separable attack-factor decomposition, the validation/effect/
vulnerability tables and a rich self-contained HTML report under figures/,
tables/ and reports/.

This keeps the network analysis in the pipeline and its results in the run's
evaluation folder, integrated with the individual-layer analysis but never
altering it. If no network data is present the stage is a no-op.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.io import abs_path, read_jsonl, stage_manifest_path, write_json
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.schemas import StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)

# Per-profile empirical network-position metrics worth correlating with outcomes.
POSITION_METRICS = [
    "weighted_in_degree", "weighted_out_degree", "eigenvector_centrality",
    "approx_betweenness", "local_clustering", "bridge_score",
    "incoming_peer_count", "outgoing_receiver_count", "incoming_exposure_weight",
    "outgoing_visibility_weight", "cascade_reach_potential",
    "h2_neighborhood_activation_readiness", "h3_central_susceptible_sender_readiness",
    "h4_central_resilient_sender_dampening_capacity",
]


class Stage05bConfig(StageConfig):
    exposure_network_root: str | None = None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3 or x[mask].std(ddof=0) == 0 or y[mask].std(ddof=0) == 0:
        return float("nan"), n
    return float(np.corrcoef(x[mask], y[mask])[0, 1]), n


def _locate_inputs(input_path: str) -> tuple[Path, Path]:
    stage05_dir = Path(input_path).resolve().parent
    stage_outputs_root = stage05_dir.parent
    long_csv = stage05_dir / "network_exposure_long.csv"
    assignments = stage_outputs_root / "01b_assign_exposure_network_positions" / "profile_position_assignments.jsonl"
    return long_csv, assignments


def _domain_of(leaf: Any) -> str:
    parts = str(leaf).split(" > ")
    return parts[1] if len(parts) > 1 else "other"


def _write_interactive_html(path: Path, long_df: pd.DataFrame, per_profile: pd.DataFrame, run_id: str) -> Path | None:
    """Interactive hover-able exposure-network map (production ontology is too
    rich for static condition subplots). Returns None if plotly is unavailable."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:  # pragma: no cover
        return None

    pp = per_profile.copy()
    # Network "centrality" axis is direct exposure sender reach (outgoing_visibility_weight).
    x = pd.to_numeric(pp.get("outgoing_visibility_weight"), errors="coerce")
    y = pd.to_numeric(pp.get("mean_ae_private"), errors="coerce")
    color = pd.to_numeric(pp.get("mean_ae_total_network"), errors="coerce")
    role = pp.get("dominant_structural_role", pd.Series([""] * len(pp)))
    p_hover = [
        f"{pid}<br>role: {r}<br>AE_private: {ap:.1f}<br>AE_total_net: {at:.1f}"
        for pid, r, ap, at in zip(pp.get("profile_id", []), role, y.fillna(0), color.fillna(0))
    ]
    ld = long_df.copy()
    bx = pd.to_numeric(ld.get("bn_increment"), errors="coerce")
    by = pd.to_numeric(ld.get("pn_increment_effectivity"), errors="coerce")
    dom = ld.get("opinion_leaf", pd.Series([""] * len(ld))).map(_domain_of)
    leaf = ld.get("opinion_leaf", pd.Series([""] * len(ld))).map(lambda s: str(s).split(" > ")[-1])
    l_hover = [f"{lf}<br>domain: {d}<br>BN-B: {b:.0f}<br>(PN-P)*d: {p:.0f}"
               for lf, d, b, p in zip(leaf, dom, bx.fillna(0), by.fillna(0))]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Profile: outgoing visibility (sender reach) x private susceptibility (color = total network effect)",
            "Leaf: baseline network increment (BN-B) vs post-network effectivity ((PN-P)*d)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=9, color=color, colorscale="RdBu", reversescale=True,
                        colorbar=dict(title="AE_total<br>network", x=0.46), line=dict(width=0.5, color="#333")),
            text=p_hover, hoverinfo="text",
        ),
        row=1, col=1,
    )
    for domain in sorted(set(dom)):
        mask = dom == domain
        fig.add_trace(
            go.Scatter(
                x=bx[mask], y=by[mask], mode="markers", name=str(domain)[:22],
                marker=dict(size=6, opacity=0.6),
                text=[h for h, m in zip(l_hover, mask) if m], hoverinfo="text",
            ),
            row=1, col=2,
        )
    fig.update_layout(template="plotly_white", title=f"{run_id}: interactive empirical exposure-network map",
                      height=560, legend=dict(font=dict(size=9), title="opinion domain"))
    fig.update_xaxes(title_text="outgoing visibility (sender reach)", row=1, col=1)
    fig.update_yaxes(title_text="mean AE_private", row=1, col=1)
    fig.update_xaxes(title_text="BN - B", row=1, col=2)
    fig.update_yaxes(title_text="(PN - P) * d", row=1, col=2)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return path


def run_stage(input_path: str, output_dir: str, config: Stage05bConfig) -> StageArtifactManifest:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    long_csv, assignments_path = _locate_inputs(input_path)

    if not long_csv.exists() or not assignments_path.exists():
        LOGGER.info("Stage 05b: no network-exposure data found; writing empty no-op manifest.")
        manifest = StageArtifactManifest(
            stage_id="05b", stage_name="network_exposure_analysis",
            input_path=abs_path(input_path), primary_output_path=abs_path(Path(output_dir)),
            output_files=[], record_count=0, metadata={"status": "skipped_no_network_data"},
        )
        write_json(stage_manifest_path(output_dir), manifest.model_dump())
        return manifest

    long_df = pd.read_csv(long_csv)
    positions = pd.DataFrame([dict(r) for r in read_jsonl(str(assignments_path))])

    # Per-profile outcome aggregation (mean over the profile's opinion leaves).
    agg = (
        long_df.groupby("profile_id")
        .agg(
            n_leaves=("opinion_leaf", "count"),
            mean_ae_private=("ae_private", "mean"),
            mean_bn_increment=("bn_increment", "mean"),
            mean_pn_increment=("pn_increment", "mean"),
            mean_pn_increment_effectivity=("pn_increment_effectivity", "mean"),
            mean_ae_total_network=("ae_total_network", "mean"),
            **(
                {"mean_net_social_amplification_effectivity": ("net_social_amplification_effectivity", "mean")}
                if "net_social_amplification_effectivity" in long_df.columns
                else {}
            ),
        )
        .reset_index()
    )
    pos_cols = ["profile_id", "position_id", "dominant_structural_role", "macro_community"] + [
        m for m in POSITION_METRICS if m in positions.columns
    ]
    pos_cols = [c for c in pos_cols if c in positions.columns]
    per_profile = agg.merge(positions[pos_cols], on="profile_id", how="left")
    for m in POSITION_METRICS:
        if m in per_profile.columns:
            per_profile[m] = per_profile[m].map(_to_float)

    # Correlations of empirical network position with the network-exposed effect.
    corr_rows: List[Dict[str, Any]] = []
    for m in [c for c in POSITION_METRICS if c in per_profile.columns]:
        for outcome in ["mean_ae_total_network", "mean_pn_increment_effectivity", "mean_ae_private"]:
            r, n = _pearson(per_profile[m], per_profile[outcome])
            corr_rows.append({"position_metric": m, "outcome": outcome, "pearson_r": round(r, 4) if r == r else None, "n": n})
    corr_df = pd.DataFrame(corr_rows)

    # Hypothesis tests (network-exposure layer design H1-H4).
    ae_private = long_df["ae_private"].dropna()
    pn_eff = long_df["pn_increment_effectivity"].dropna()
    net_amp = (
        long_df["net_social_amplification_effectivity"].dropna()
        if "net_social_amplification_effectivity" in long_df.columns
        else pd.Series(dtype=float)
    )
    reach_col = "outgoing_visibility_weight" if "outgoing_visibility_weight" in per_profile.columns else None
    if reach_col is None and "weighted_out_degree" in per_profile.columns:
        reach_col = "weighted_out_degree"
    h3_r, h3_n = (_pearson(per_profile[reach_col], per_profile["mean_ae_total_network"]) if reach_col else (float("nan"), 0))
    centrality_activation = (
        float((per_profile[reach_col].fillna(0) * per_profile["mean_ae_private"].fillna(0)).sum())
        if reach_col else None
    )
    hypotheses = {
        "H1_private_susceptibility_heterogeneity": {
            "claim": "Profiles differ in direction-aware private attack susceptibility (AE_private).",
            "mean_ae_private": round(float(ae_private.mean()), 3) if len(ae_private) else None,
            "sd_ae_private": round(float(ae_private.std(ddof=0)), 3) if len(ae_private) else None,
            "between_profile_sd": round(float(agg["mean_ae_private"].std(ddof=0)), 3) if len(agg) else None,
            "pct_toward_goal": round(float((ae_private > 0).mean() * 100), 1) if len(ae_private) else None,
        },
        "H2_network_amplification_or_attenuation": {
            "claim": "Post-attack peer context shifts the private attack effect (PN_increment_effectivity).",
            "mean_pn_increment_effectivity": round(float(pn_eff.mean()), 3) if len(pn_eff) else None,
            "sd_pn_increment_effectivity": round(float(pn_eff.std(ddof=0)), 3) if len(pn_eff) else None,
            "pct_amplifying": round(float((pn_eff > 0).mean() * 100), 1) if len(pn_eff) else None,
            "pct_dampening": round(float((pn_eff < 0).mean() * 100), 1) if len(pn_eff) else None,
            # Difference-in-differences: post-attack peer pull net of baseline conformity.
            "mean_net_social_amplification_effectivity": round(float(net_amp.mean()), 3) if len(net_amp) else None,
            "pct_amplifying_conformity_adjusted": round(float((net_amp > 0).mean() * 100), 1) if len(net_amp) else None,
        },
        "H3_central_susceptible_sender_amplification": {
            "claim": "High outgoing-reach susceptible profiles raise population-level final effect.",
            "reach_metric": reach_col,
            "corr_reach_vs_ae_total_network": round(h3_r, 4) if h3_r == h3_r else None,
            "n": h3_n,
            "centrality_weighted_private_activation": round(centrality_activation, 2) if centrality_activation is not None else None,
        },
        "H4_central_resilient_sender_attenuation": {
            "claim": "Central profiles with resistant private effects reduce final network effectiveness.",
            "note": "Inspect high-reach profiles with mean_ae_private <= 0 in per_profile_network_outcomes.csv.",
            "n_high_reach_resistant": int(
                ((per_profile[reach_col] > per_profile[reach_col].median()) & (per_profile["mean_ae_private"] <= 0)).sum()
            ) if reach_col else None,
        },
        "measurement_backbone": {
            "n_profiles": int(per_profile["profile_id"].nunique()),
            "n_leaf_measurements": int(len(long_df)),
            "mean_bn_increment": round(float(long_df["bn_increment"].dropna().mean()), 3) if long_df["bn_increment"].notna().any() else None,
            "mean_pn_increment": round(float(long_df["pn_increment"].dropna().mean()), 3) if long_df["pn_increment"].notna().any() else None,
            "network_active_share": round(float((long_df["bn_increment"].fillna(0) != 0).mean() * 100), 1),
        },
    }

    # Concise machine-readable complements (H1-H4 summary + per-profile table).
    per_profile_csv = Path(output_dir) / "per_profile_network_outcomes.csv"
    corr_csv = Path(output_dir) / "network_position_outcome_correlations.csv"
    hyp_json = Path(output_dir) / "network_hypotheses.json"
    per_profile.to_csv(per_profile_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)
    write_json(hyp_json, hypotheses)

    # Interactive map (production ontology is too rich for static condition
    # subplots): hover-able profile centrality x susceptibility and per-leaf
    # network-mechanism scatter.
    interactive_html = _write_interactive_html(
        Path(output_dir) / "network_exposure_interactive.html", long_df, per_profile, config.run_id
    )

    # Comprehensive visual analysis (Thomas's full report, adapted to this
    # pipeline): ~13 figures + decomposition tables + rich HTML into figures/,
    # tables/, reports/ under the same output dir. Guarded so the concise
    # complements above are always produced even if the rich report fails.
    run_root = Path(input_path).resolve().parent.parent  # stage05 dir -> stage_outputs -> run_root
    run_root = run_root.parent
    report_script = Path(__file__).resolve().parent / "report_builder" / "build_run_network_exposure_report.py"
    rich_report: Path | None = None
    if report_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(report_script), "--run-root", str(run_root),
                 "--output-dir", str(Path(output_dir).resolve()), "--run-id", config.run_id],
                check=True, capture_output=True, text=True,
            )
            candidates = list((Path(output_dir) / "reports").glob("*_network_exposure_report.html"))
            rich_report = candidates[0] if candidates else None
            LOGGER.info("Stage 05b: comprehensive exposure-network report generated.")
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
            LOGGER.warning("Stage 05b comprehensive report failed: %s", (exc.stderr or "")[-800:])

    output_files = [abs_path(per_profile_csv), abs_path(corr_csv), abs_path(hyp_json)]
    if interactive_html is not None:
        output_files.append(abs_path(interactive_html))
    if rich_report is not None:
        output_files.append(abs_path(rich_report))
    manifest = StageArtifactManifest(
        stage_id="05b", stage_name="network_exposure_analysis",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(rich_report if rich_report is not None else hyp_json),
        output_files=output_files,
        record_count=int(len(per_profile)),
        metadata={"status": "completed", "comprehensive_report": rich_report is not None,
                  **{k: hypotheses[k] for k in ("measurement_backbone",)}},
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    LOGGER.info("Stage 05b completed: %s profiles analysed", len(per_profile))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 05b - Empirical exposure-network analysis")
    parser.add_argument("--input-path", required=True, help="Stage 05 primary output (sem_long_encoded.csv)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exposure-network-root", default=None)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    config = Stage05bConfig(
        stage_name="network_exposure_analysis", run_id=args.run_id, seed=args.seed,
        exposure_network_root=args.exposure_network_root,
    )
    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 05b done: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
