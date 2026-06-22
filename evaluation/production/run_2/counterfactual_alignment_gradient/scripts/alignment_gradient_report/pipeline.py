from __future__ import annotations

"""Top-level report orchestration and manifest writing.

`build_report` is the single programmatic entrypoint used by the CLI wrapper and branch launcher.
"""

from pathlib import Path
from typing import Any

from .data import (
    _condition_table,
    _condition_vulnerability_plane_summary,
    _design_balance,
    _level_summary,
    _load_sem,
    _quality_table,
    _read_design,
    _write_tables,
)
from .html_report import _render_html
from .models import (
    _figure4_model_summary,
    _fit_models,
    _fit_robustness_models,
    _original_vs_branch_comparison,
)
from .paths import abs_path, ensure_dir, write_json
from .plots_diagnostics import (
    _plot_alignment_outcomes,
    _plot_branch_alignment_vs_network_effect,
    _plot_condition_vulnerability_planes,
    _plot_level_means,
    _plot_quality_gates,
    _plot_representative_overlays,
    _plot_robustness_results,
    _plot_target_vs_achieved,
)
from .plots_network import _plot_full_condition_network_overlays
from .plots_publication import (
    _plot_publication_network_mechanism_composite,
    _plot_publication_network_mechanism_composite_abc,
)


def build_report(branch_root: str | Path, run_id: str, report_mode: str = "full") -> dict[str, Any]:
    """Build the requested branch report artifacts and return the analysis manifest."""
    branch_root = Path(branch_root).resolve()
    design = _read_design(branch_root)
    sem = _load_sem(branch_root)
    condition = _condition_table(sem, design)
    level = _level_summary(condition)
    models = _fit_models(condition)
    robustness = _fit_robustness_models(sem, condition, design)
    design_balance = _design_balance(condition)
    original_comparison = _original_vs_branch_comparison(branch_root, condition)
    vulnerability_plane_summary = _condition_vulnerability_plane_summary(sem, condition)
    figure4_summary = _figure4_model_summary(condition, robustness)
    quality = _quality_table(branch_root, condition)

    output_root = ensure_dir(branch_root / "network_exposure_analysis")
    figures_dir = ensure_dir(output_root / "figures")
    reports_dir = ensure_dir(output_root / "reports")
    tables = _write_tables(
        branch_root,
        condition,
        level,
        models,
        robustness,
        design_balance,
        original_comparison,
        vulnerability_plane_summary,
        figure4_summary,
        quality,
    )
    figures: dict[str, Path] = {}
    publication_figures: dict[str, Path] = {}
    if report_mode in {"h3h4", "full"}:
        figures.update(
            {
                "target_vs_achieved": _plot_target_vs_achieved(condition, figures_dir),
                "alignment_outcomes": _plot_alignment_outcomes(condition, figure4_summary, figures_dir),
                "condition_vulnerability_planes": _plot_condition_vulnerability_planes(sem, condition, figures_dir),
                "level_means": _plot_level_means(level, figures_dir),
                "representative_overlays": _plot_representative_overlays(sem, condition, figures_dir),
                "robustness_coefficients": _plot_robustness_results(robustness, figures_dir),
                "quality_gates": _plot_quality_gates(quality, figures_dir),
            }
    )
    if report_mode in {"branch_network", "full"}:
        figures["branch_alignment_outcome"] = _plot_branch_alignment_vs_network_effect(condition, figures_dir)
        full_overlay, overlay_nodes, overlay_edges = _plot_full_condition_network_overlays(
            sem, condition, branch_root, figures_dir
        )
        figures["full_network_overlay"] = full_overlay
        tables["branch_full_network_overlay_nodes"] = overlay_nodes
        tables["branch_full_network_overlay_edges"] = overlay_edges
    if report_mode == "full":
        publication_figures = _plot_publication_network_mechanism_composite(sem, condition, branch_root)
        publication_figures.update(
            {
                f"abc_{key}": path
                for key, path in _plot_publication_network_mechanism_composite_abc(
                    sem, condition, figure4_summary, branch_root
                ).items()
            }
        )
    report = _render_html(
        run_id=run_id,
        branch_root=branch_root,
        report_mode=report_mode,
        condition=condition,
        level=level,
        models=models,
        robustness=robustness,
        design_balance=design_balance,
        original_comparison=original_comparison,
        quality=quality,
        figures=figures,
        reports_dir=reports_dir,
    )
    manifest = {
        "run_id": run_id,
        "report_mode": report_mode,
        "branch_root": abs_path(branch_root),
        "report": abs_path(report),
        "figures": {key: abs_path(path) for key, path in figures.items()},
        "publication_figures": {key: abs_path(path) for key, path in publication_figures.items()},
        "tables": {key: abs_path(path) for key, path in tables.items()},
        "source_artifacts": {
            "sem_long_encoded": abs_path(
                branch_root
                / "merged_outputs"
                / "stage_outputs"
                / "05_compute_effectivity_deltas"
                / "sem_long_encoded.csv"
            ),
            "alignment_design_manifest": abs_path(branch_root / "design" / "alignment_design_manifest.json"),
            "stage04_private_fallback_sensitivity": abs_path(
                branch_root / "design" / "stage04_private_fallback_sensitivity.csv"
            ),
            "original_network_exposure_analysis": abs_path(branch_root.parent / "network_exposure_analysis"),
        },
    }
    manifest_path = reports_dir / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    manifest["analysis_manifest"] = abs_path(manifest_path)
    return manifest
