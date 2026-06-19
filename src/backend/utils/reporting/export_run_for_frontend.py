from __future__ import annotations

"""
Copy a completed run's artefacts into the Next.js frontend's public directory.

The frontend reads everything it needs from /public/runs/<run_id>/... so it
remains a static site (no API server required). This script consolidates:

- datasets/sem_long_encoded.csv             (long-form scenarios + outcomes)
- datasets/profile_level_effectivity.csv    (profile-level summary)
- sem/profile_susceptibility_index.csv      (susceptibility ranking)
- sem/expanded_moderator_comparison.csv     (ridge + OLS forest)
- sem/conditional_susceptibility_task_*.csv
- sem/profile_network_centrality.csv        (network nodes)
- sem/profile_network_edges.csv             (network edges)
- sem/profile_network_global_metrics.json   (network globals)
- sem/intraclass_correlation.json
- sem/advanced_*.{csv,json}                 (current design advanced layer)
- sem/analysis_quality_diagnostics.json
- report/assumption_register.json + peer_review_critiques.json
- ontologies/{profile,attack,opinion}.json  (resolved from --ontology-root)
- scenario_compatibility_audit.json         (current design)
- manifest.json                             (synthesised here)

Usage:
    python3 -m src.backend.utils.reporting.export_run_for_frontend \\
        --run-output evaluation/tests/run_1 \\
        --run-id run_1 \\
        --frontend src/frontend \\
        --ontology-root src/backend/ontology/01_separated/test \\
        --paper-title "Pilot test run 1"
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stat_csv_summary(path: Path, outcome: str) -> Dict[str, Optional[float]]:
    """Compute mean and pct-positive of an outcome column without pandas dep."""
    if not path.exists():
        return {"mean": None, "pct_positive": None, "n": 0}
    import csv
    n = 0
    n_pos = 0
    total = 0.0
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                v = float(row.get(outcome) or "")
            except (TypeError, ValueError):
                continue
            n += 1
            total += v
            if v > 0:
                n_pos += 1
    return {
        "mean": total / n if n else None,
        "pct_positive": (n_pos / n) if n else None,
        "n": n,
    }


def _unique_count(path: Path, column: str) -> int:
    if not path.exists():
        return 0
    import csv
    seen = set()
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            v = row.get(column)
            if v:
                seen.add(v)
    return len(seen)


def export_run(
    run_output: Path,
    run_id: str,
    frontend_dir: Path,
    ontology_root: Optional[Path],
    paper_title: Optional[str],
) -> Dict[str, Any]:
    target = frontend_dir / "public" / "runs" / run_id
    target.mkdir(parents=True, exist_ok=True)

    artefacts: Dict[str, str] = {}

    # ── datasets ────────────────────────────────────────────────────────────
    datasets_src = run_output / "datasets"
    datasets_dst = target / "datasets"
    for fname in [
        "sem_long_encoded.csv",
        "sem_long_raw.csv",
        "profile_level_effectivity.csv",
        "profile_sem_wide.csv",
        "delta_summary.json",
    ]:
        if _copy(datasets_src / fname, datasets_dst / fname):
            artefacts[f"datasets/{fname}"] = str(datasets_dst / fname)

    # ── sem outputs ─────────────────────────────────────────────────────────
    sem_src = run_output / "sem"
    sem_dst = target / "sem"
    for fname in [
        "sem_result.json",
        "sem_coefficients.csv",
        "sem_fit_indices.json",
        "ols_robust_params.csv",
        "exploratory_moderator_comparison.csv",
        "expanded_moderator_comparison.csv",
        "moderator_weight_table.csv",
        "profile_susceptibility_index.csv",
        "profile_susceptibility_breakdown.csv",
        "conditional_susceptibility_task_coefficients.csv",
        "conditional_susceptibility_task_summary.csv",
        "conditional_susceptibility_artifact.json",
        "conditional_susceptibility_bootstrap_ranks.csv",
        "conditional_susceptibility_group_contributions.csv",
        "intraclass_correlation.json",
        "ridge_full_summary.json",
        "ridge_full_coefficients.csv",
        "elastic_net_summary.json",
        "elastic_net_coefficients.csv",
        "elastic_net_selected.csv",
        "rf_summary.json",
        "rf_feature_importance.csv",
        "analysis_quality_diagnostics.json",
        "profile_network_centrality.csv",
        "profile_network_edges.csv",
        "profile_network_layout.csv",
        "profile_network_global_metrics.json",
        # current design advanced
        "advanced_multilevel_icc.json",
        "advanced_mixed_effects_coefficients.csv",
        "advanced_permutation_importance.csv",
        "advanced_bca_bootstrap_ridge.csv",
        "advanced_rank_stability.csv",
        "advanced_network_diagnostics.json",
    ]:
        if _copy(sem_src / fname, sem_dst / fname):
            artefacts[f"sem/{fname}"] = str(sem_dst / fname)

    # ── report ──────────────────────────────────────────────────────────────
    report_src = run_output / "report"
    report_dst = target / "report"
    for fname in [
        "moderation_report.txt",
        "methodology_audit.txt",
        "assumption_register.json",
        "peer_review_critiques.json",
        "report_summary.json",
    ]:
        if _copy(report_src / fname, report_dst / fname):
            artefacts[f"report/{fname}"] = str(report_dst / fname)

    # ── compatibility audit (current design) ──────────────────────────────────────
    audit_src = run_output / "stage_outputs" / "01_create_scenarios" / "scenario_compatibility_audit.json"
    if not audit_src.exists():
        # Fallback to older layout
        audit_src = run_output / "scenario_compatibility_audit.json"
    if _copy(audit_src, target / "scenario_compatibility_audit.json"):
        artefacts["scenario_compatibility_audit.json"] = str(target / "scenario_compatibility_audit.json")

    # ── ontology snapshot ─────────────────────────────────────────────────
    if ontology_root is not None:
        ont_dst = target / "ontologies"
        for sub in ["PROFILE/profile.json", "ATTACK/attack.json", "OPINION/opinion.json"]:
            tgt_name = sub.split("/")[-1]
            if _copy(ontology_root / sub, ont_dst / tgt_name):
                artefacts[f"ontologies/{tgt_name}"] = str(ont_dst / tgt_name)

    # ── derive summary metrics for the manifest ───────────────────────────
    long_path = datasets_dst / "sem_long_encoded.csv"
    ae_summary = _stat_csv_summary(long_path, "adversarial_effectivity")
    n_profiles = _unique_count(long_path, "profile_id")
    n_attacks = _unique_count(long_path, "attack_leaf")
    n_opinions = _unique_count(long_path, "opinion_leaf")

    sem_fit = _read_json(sem_dst / "sem_fit_indices.json") or {}
    icc = _read_json(sem_dst / "intraclass_correlation.json") or {}
    ridge_full = _read_json(sem_dst / "ridge_full_summary.json") or {}
    rf_summary = _read_json(sem_dst / "rf_summary.json") or {}
    advanced_icc = _read_json(sem_dst / "advanced_multilevel_icc.json") or {}
    audit = _read_json(target / "scenario_compatibility_audit.json") or {}

    summary = {
        "n_profiles": n_profiles,
        "n_attacks": n_attacks,
        "n_opinions": n_opinions,
        "n_scenarios": ae_summary.get("n", 0),
        "attack_ratio": (audit.get("n_attack_scenarios") or 0) / max(1, ae_summary.get("n", 1)),
        "paper_title": paper_title,
        "adversarial_goal": (audit or {}).get("adversarial_goal"),
        "mean_ae": ae_summary.get("mean"),
        "pct_ae_positive": ae_summary.get("pct_positive"),
        "cfi": (sem_fit or {}).get("CFI"),
        "rmsea": (sem_fit or {}).get("RMSEA"),
        "ridge_cv_r2": (ridge_full or {}).get("cv_r2"),
        "rf_oob_r2": (rf_summary or {}).get("oob_r2"),
        "icc_profile": (advanced_icc or {}).get("icc_profile"),
        "icc_attack": (advanced_icc or {}).get("icc_attack"),
        "icc_opinion": (advanced_icc or {}).get("icc_opinion"),
        "abs_icc1": ((icc or {}).get("abs_delta_score") or {}).get("icc1") if isinstance(icc, dict) else None,
    }

    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "artefacts": artefacts,
        "notes": [
            "All AE figures use signed adversarial_effectivity = (post − baseline) × adversarial_direction.",
            "Direction-neutral opinion leaves (d_k = 0) are excluded from primary AE scoring but retained for diversity.",
            "Compatibility-rule exclusions are listed in scenario_compatibility_audit.json.",
        ],
    }

    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Update /public/runs/index.json with the discovered runs
    runs_root = frontend_dir / "public" / "runs"
    discovered = sorted(p.name for p in runs_root.iterdir() if (p / "manifest.json").exists())
    default_run = run_id if run_id in discovered else (discovered[-1] if discovered else run_id)
    (runs_root / "index.json").write_text(
        json.dumps({"default_run": default_run, "runs": discovered}, indent=2),
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a completed run for the Next.js frontend.")
    parser.add_argument("--run-output", required=True, help="Path to evaluation/<run_id>")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--frontend", required=True, help="Path to src/frontend")
    parser.add_argument("--ontology-root", default=None, help="Root of PROFILE / ATTACK / OPINION used by the run")
    parser.add_argument("--paper-title", default=None)
    args = parser.parse_args()

    manifest = export_run(
        run_output=Path(args.run_output),
        run_id=args.run_id,
        frontend_dir=Path(args.frontend),
        ontology_root=Path(args.ontology_root) if args.ontology_root else None,
        paper_title=args.paper_title,
    )
    print(json.dumps({"run_id": args.run_id, "summary": manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
