from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.io import abs_path, ensure_dir, write_json
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.ontology_utils import default_ontology_root


LOGGER = logging.getLogger(__name__)


DEFAULT_PAPER_TITLE = (
    "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions: "
    "An Ontology-Constrained Multi-Agent Simulation Approach"
)


@dataclass
class StageSpec:
    stage_id: str
    stage_name: str
    script_path: Path


def _stage_specs(project_root: Path, with_network_exposure: bool = False) -> List[StageSpec]:
    base = project_root / "src" / "backend" / "pipeline" / "separate"
    # The empirical exposure-network stages are additive. With network exposure
    # OFF the stage list is identical to the core individual-layer pipeline.
    specs: List[StageSpec] = [
        StageSpec("01", "create_scenarios", base / "01_create_scenarios" / "run_stage.py"),
    ]
    if with_network_exposure:
        specs.append(StageSpec("01b", "assign_exposure_network_positions", base / "01b_assign_exposure_network_positions" / "run_stage.py"))
    specs.append(StageSpec("02", "assess_baseline_opinions", base / "02_assess_baseline_opinions" / "run_stage.py"))
    if with_network_exposure:
        specs.append(StageSpec("02b", "assess_network_exposure_opinions", base / "02b_assess_network_exposure_opinions" / "run_stage.py"))
    specs.append(StageSpec("03", "run_opinion_attacks", base / "03_run_opinion_attacks" / "run_stage.py"))
    specs.append(StageSpec("04", "assess_post_attack_opinions", base / "04_assess_post_attack_opinions" / "run_stage.py"))
    if with_network_exposure:
        specs.append(StageSpec("04b", "assess_post_attack_network_exposure_opinions", base / "04b_assess_post_attack_network_exposure_opinions" / "run_stage.py"))
    specs.append(StageSpec("05", "compute_effectivity_deltas", base / "05_compute_effectivity_deltas" / "run_stage.py"))
    if with_network_exposure:
        specs.append(StageSpec("05b", "network_exposure_analysis", base / "05b_network_exposure_analysis" / "run_stage.py"))
    specs.append(StageSpec("06", "construct_structural_equation_model", base / "06_construct_structural_equation_model" / "run_stage.py"))
    specs.append(StageSpec("07", "generate_research_visuals", base / "07_generate_research_visuals" / "run_stage.py"))
    specs.append(StageSpec("08", "generate_publication_assets", base / "08_generate_publication_assets" / "run_stage.py"))
    specs.append(StageSpec("09", "build_research_report", base / "09_build_research_report" / "run_stage.py"))
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ontology-driven attack-opinion simulation pipeline")

    parser.add_argument("--output-root", default="evaluation/tests/run_1")
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--paper-title", default=DEFAULT_PAPER_TITLE)
    parser.add_argument("--report-root", default="research_report/report")
    parser.add_argument("--report-assets-root", default="research_report/assets")

    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--n-profiles", type=int, default=None)
    parser.add_argument(
        "--integrated-scenarios-path",
        default=None,
        help="Path to a pre-built integrated scenarios .jsonl. When set, stage 01 selects rows "
        "from it (no ontology sampling) and downstream stages run unchanged.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-ratio", type=float, default=0.5)
    parser.add_argument(
        "--attack-leaf",
        default="ATTACK_VECTORS > Social_Media_Misinformation > Misleading_Narrative_Framing",
    )
    parser.add_argument(
        "--attack-leaves",
        default=None,
        help="Comma-01_separated attack leaves; takes precedence over --attack-leaf",
    )
    parser.add_argument("--focus-opinion-domain", default=None)
    parser.add_argument(
        "--focus-opinion-domains",
        default=None,
        help="Comma-separated opinion parent clusters to concentrate the integrated sample into "
        "(densifies the exposure network for the network-position correlations).",
    )
    parser.add_argument("--opinion-leaves", default=None, help="Comma-01_separated explicit opinion leaf selection")
    parser.add_argument("--max-opinion-leaves", type=int, default=None)
    parser.add_argument("--profile-candidate-multiplier", type=int, default=2)
    parser.add_argument("--primary-moderator", default="posthoc_profile_susceptibility_index")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--enforce-compatibility-rules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-direction-neutral-opinions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--realism-weight-temperature", type=float, default=1.5)

    parser.add_argument("--use-test-ontology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ontology-root", default=None)

    parser.add_argument("--openrouter-model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=2)
    parser.add_argument(
        "--profile-generation-mode",
        choices=["deterministic", "llm", "hybrid"],
        default="deterministic",
    )

    parser.add_argument("--self-supervise-attack-realism", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realism-threshold", type=float, default=0.72)
    parser.add_argument("--self-supervise-opinion-coherence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coherence-threshold", type=float, default=0.72)
    parser.add_argument("--generate-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-static-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-report", action=argparse.BooleanOptionalAction, default=True)

    # Additive empirical exposure-network layer (stages 01b / 02b / 04b). OFF by
    # default so the core individual-layer run is unaffected.
    parser.add_argument(
        "--with-network-exposure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the additive empirical exposure-network stages (01b/02b/04b) alongside the core pipeline.",
    )
    parser.add_argument(
        "--exposure-network-root",
        default=None,
        help="Path to the exposure-network substrate. Defaults to src/data/exposure_networks/politisky24_bluesky_v1.",
    )
    parser.add_argument(
        "--network-exposure-top-k",
        type=int,
        default=8,
        help="Bounded number of incoming peer exemplars sent to the network-exposure prompts.",
    )
    parser.add_argument(
        "--network-scenario-cap",
        type=int,
        default=500,
        help="Hard cap on integrated scenarios when the exposure-network layer is on. Above it the "
        "media-keyword heuristic selects the network-congruent subset. Only enforced with --with-network-exposure.",
    )

    stage_choices = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    parser.add_argument("--resume-from-stage", default="01", choices=stage_choices)
    parser.add_argument("--stop-after-stage", default="09", choices=stage_choices)

    parser.add_argument("--save-raw-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print a live, professional per-stage progress monitor (call counts, rate, ETA). Default off.",
    )

    parser.add_argument("--run-stage-checks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage-check-scenarios", type=int, default=3)

    return parser.parse_args()


def _build_common_stage_args(args: argparse.Namespace, stage_log_file: Path) -> List[str]:
    return [
        "--run-id",
        args.run_id,
        "--seed",
        str(args.seed),
        "--log-file",
        abs_path(stage_log_file),
        "--log-level",
        args.log_level,
    ]


def _format_duration(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN / negative guard
        return "?"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _raw_call_count(raw_llm_dir: Optional[Path]) -> int:
    if raw_llm_dir is None:
        return 0
    try:
        return sum(1 for _ in raw_llm_dir.iterdir())
    except OSError:
        return 0


def _progress_loop(
    stage: StageSpec,
    raw_llm_dir: Optional[Path],
    expected_calls: Optional[int],
    start_count: int,
    start_time: float,
    stop_event: threading.Event,
) -> None:
    """Daemon loop printing a professional live progress line for a stage.

    Progress is inferred from the raw-LLM provenance directory (one file per
    completed call), so it needs no instrumentation inside the stage code.
    """
    while not stop_event.wait(4.0):
        done = max(0, _raw_call_count(raw_llm_dir) - start_count)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        if expected_calls and expected_calls > 0:
            remaining = max(0, expected_calls - done)
            eta = remaining / rate if rate > 0 else float("nan")
            pct = min(100.0, 100.0 * done / expected_calls)
            bar_len = 24
            filled = int(bar_len * min(1.0, done / expected_calls))
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"      [{bar}] {done}/{expected_calls} ({pct:0.0f}%)"
                f"  | ~{remaining} left  | {rate:0.2f} calls/s"
                f"  | elapsed {_format_duration(elapsed)}  | ETA {_format_duration(eta)}",
                flush=True,
            )
        elif raw_llm_dir is not None:
            print(
                f"      ... {stage.stage_name}: {done} calls"
                f"  | {rate:0.2f} calls/s  | elapsed {_format_duration(elapsed)}",
                flush=True,
            )
        else:
            print(f"      ... {stage.stage_name}: running  | elapsed {_format_duration(elapsed)}", flush=True)


def _call_stage(
    stage: StageSpec,
    cmd: List[str],
    cwd: Path,
    *,
    verbose: bool = False,
    raw_llm_dir: Optional[Path] = None,
    expected_calls: Optional[int] = None,
) -> None:
    LOGGER.info("Running stage %s (%s)", stage.stage_id, stage.stage_name)
    LOGGER.debug("Command: %s", " ".join(cmd))
    start_time = time.time()
    start_count = _raw_call_count(raw_llm_dir)

    monitor: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    if verbose:
        rule = "-" * 70
        calls_label = f"~{expected_calls} LLM calls" if expected_calls else "no/elastic LLM calls"
        print(f"\n{rule}\n  STAGE {stage.stage_id}  |  {stage.stage_name}  |  {calls_label}\n{rule}", flush=True)
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_progress_loop,
            args=(stage, raw_llm_dir, expected_calls, start_count, start_time, stop_event),
            daemon=True,
        )
        monitor.start()

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if stop_event is not None:
        stop_event.set()
    if monitor is not None:
        monitor.join(timeout=3)
    duration = time.time() - start_time

    if result.returncode != 0:
        LOGGER.error("Stage %s failed with code %s", stage.stage_id, result.returncode)
        LOGGER.error("STDOUT:\n%s", result.stdout)
        LOGGER.error("STDERR:\n%s", result.stderr)
        if verbose:
            print(f"  [FAILED] stage {stage.stage_id} after {_format_duration(duration)}", flush=True)
        raise RuntimeError(f"Stage {stage.stage_id} failed")
    if result.stdout.strip():
        LOGGER.info("Stage %s stdout:\n%s", stage.stage_id, result.stdout.strip())
    if result.stderr.strip():
        LOGGER.warning("Stage %s stderr:\n%s", stage.stage_id, result.stderr.strip())

    if verbose:
        done = max(0, _raw_call_count(raw_llm_dir) - start_count)
        calls_done = f"  |  {done} LLM calls" if (raw_llm_dir is not None and (expected_calls or done)) else ""
        print(f"  [done] stage {stage.stage_id} in {_format_duration(duration)}{calls_done}", flush=True)


def _pick_output_file(manifest: Dict[str, object], needle: str) -> str:
    for path in manifest.get("output_files", []):
        if isinstance(path, str) and needle in path:
            return path
    raise RuntimeError(f"Could not find output containing '{needle}' in manifest")


def _copy_outputs(stage_outputs_root: Path, output_root: Path) -> None:
    def _copy_dir_contents(source_dir: Path, target_dir: Path) -> None:
        if not source_dir.exists():
            return
        ensure_dir(target_dir)
        for item in source_dir.iterdir():
            target = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    datasets_dir = ensure_dir(output_root / "datasets")
    sem_dir = ensure_dir(output_root / "sem")
    report_dir = ensure_dir(output_root / "report")
    visuals_dir = ensure_dir(output_root / "visuals")
    publication_assets_dir = ensure_dir(output_root / "publication_assets")
    paper_dir = ensure_dir(output_root / "paper")
    paper_publication_assets_dir = ensure_dir(paper_dir / "publication_assets")

    stage05 = stage_outputs_root / "05_compute_effectivity_deltas"
    stage06 = stage_outputs_root / "06_construct_structural_equation_model"
    stage07 = stage_outputs_root / "07_generate_research_visuals"
    stage08 = stage_outputs_root / "08_generate_publication_assets"
    stage09 = stage_outputs_root / "09_build_research_report"

    for filename in [
        "effectivity_deltas.jsonl",
        "sem_long_raw.csv",
        "sem_long_encoded.csv",
        "sem_long_encoded.jsonl",
        "profile_level_effectivity.csv",
        "profile_sem_wide.csv",
        "delta_summary.json",
        "sem_long_rows.jsonl",
    ]:
        src = stage05 / filename
        if src.exists():
            shutil.copy2(src, datasets_dir / filename)

    for filename in [
        "sem_model_spec.txt",
        "sem_result.json",
        "sem_coefficients.csv",
        "sem_fit_indices.json",
        "ols_robust_summary.txt",
        "ols_robust_params.csv",
        "bootstrap_primary_params.csv",
        "exploratory_moderator_comparison.csv",
        "moderator_weight_table.csv",
        "profile_multivariate_model_spec.txt",
        "profile_susceptibility_index.csv",
        "profile_susceptibility_breakdown.csv",
        "latent_attack_effectivity_scores.csv",
        "profile_level_effectivity.csv",
        "profile_sem_wide.csv",
    ]:
        src = stage06 / filename
        if src.exists():
            shutil.copy2(src, sem_dir / filename)

    for filename in [
        "moderation_report.txt",
        "methodology_audit.txt",
        "assumption_register.json",
        "peer_review_critiques.json",
    ]:
        src = stage06 / filename
        if src.exists():
            shutil.copy2(src, report_dir / filename)

    if stage07.exists():
        _copy_dir_contents(stage07, visuals_dir)
        _copy_dir_contents(stage07, paper_publication_assets_dir)

    if stage08.exists():
        _copy_dir_contents(stage08, publication_assets_dir)
        _copy_dir_contents(stage08, paper_publication_assets_dir)

    # Additive empirical exposure-network analysis (present only on network runs).
    stage05b = stage_outputs_root / "05b_network_exposure_analysis"
    if stage05b.exists() and any(stage05b.iterdir()):
        _copy_dir_contents(stage05b, ensure_dir(output_root / "network_exposure_analysis"))

    if stage09.exists():
        for filename in ["report_summary.json"]:
            for source_root in [PROJECT_ROOT / "research_report" / "report", stage09]:
                src = source_root / filename
                if src.exists():
                    shutil.copy2(src, paper_dir / filename)
                    break
        for filename in ["main.tex", "main.pdf", "references.bib"]:
            for source_root in [PROJECT_ROOT / "research_report" / "report", stage09]:
                src = source_root / filename
                if src.exists():
                    shutil.copy2(src, paper_dir / filename)
                    break


def _run_stage_checks(
    project_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    ontology_root: Path,
) -> None:
    stage_checks_root = ensure_dir(output_root / "stage_checks")
    stage_check_assets = ensure_dir(stage_checks_root / "report_assets")
    stage_check_report = ensure_dir(stage_checks_root / "paper")

    cmd = [
        sys.executable,
        abs_path(project_root / "src" / "backend" / "pipeline" / "full" / "run_full_pipeline.py"),
        "--output-root",
        abs_path(stage_checks_root),
        "--run-id",
        f"{args.run_id}_stage_checks",
        "--paper-title",
        args.paper_title,
        "--report-root",
        abs_path(stage_check_report),
        "--report-assets-root",
        abs_path(stage_check_assets),
        "--n-scenarios",
        str(args.stage_check_scenarios),
        "--seed",
        str(args.seed + 1000),
        "--attack-ratio",
        str(args.attack_ratio),
        "--attack-leaf",
        args.attack_leaf,
        *(["--attack-leaves", args.attack_leaves] if args.attack_leaves else []),
        "--focus-opinion-domain",
        args.focus_opinion_domain if args.focus_opinion_domain is not None else "",
        "--primary-moderator",
        args.primary_moderator,
        "--bootstrap-samples",
        str(min(args.bootstrap_samples, 100)),
        "--openrouter-model",
        args.openrouter_model,
        "--temperature",
        str(args.temperature),
        "--max-repair-iter",
        str(args.max_repair_iter),
        "--profile-generation-mode",
        args.profile_generation_mode,
        "--resume-from-stage",
        "01",
        "--stop-after-stage",
        "08",
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-concurrency",
        str(args.max_concurrency),
        "--log-level",
        args.log_level,
        "--no-run-stage-checks",
        "--no-build-report",
    ]

    if args.n_profiles is not None:
        cmd.extend(["--n-profiles", str(min(args.stage_check_scenarios, args.n_profiles))])

    cmd.append("--use-test-ontology" if args.use_test_ontology else "--no-use-test-ontology")
    cmd.append("--save-raw-llm" if args.save_raw_llm else "--no-save-raw-llm")
    cmd.append("--generate-visuals" if args.generate_visuals else "--no-generate-visuals")
    cmd.append("--export-static-figures" if args.export_static_figures else "--no-export-static-figures")
    cmd.append("--self-supervise-attack-realism" if args.self_supervise_attack_realism else "--no-self-supervise-attack-realism")
    cmd.append("--self-supervise-opinion-coherence" if args.self_supervise_opinion_coherence else "--no-self-supervise-opinion-coherence")
    cmd.extend(["--realism-threshold", str(args.realism_threshold), "--coherence-threshold", str(args.coherence_threshold)])

    if args.ontology_root:
        cmd.extend(["--ontology-root", abs_path(ontology_root)])
    if args.max_opinion_leaves is not None:
        cmd.extend(["--max-opinion-leaves", str(args.max_opinion_leaves)])
    cmd.extend(["--profile-candidate-multiplier", str(args.profile_candidate_multiplier)])

    LOGGER.info("Running stage checks in %s", stage_checks_root)
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.error("Stage checks failed. STDOUT:\n%s", result.stdout)
        LOGGER.error("Stage checks failed. STDERR:\n%s", result.stderr)
        raise RuntimeError("Stage checks failed")
    LOGGER.info("Stage checks completed in %s", stage_checks_root)


def _load_existing_manifests(stage_specs: List[StageSpec], stage_outputs_root: Path) -> Dict[str, Dict[str, object]]:
    manifests: Dict[str, Dict[str, object]] = {}
    for stage in stage_specs:
        manifest_path = stage_outputs_root / f"{stage.stage_id}_{stage.stage_name}" / "manifest.json"
        if manifest_path.exists():
            manifests[stage.stage_id] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifests


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[4]
    load_dotenv(project_root / ".env")

    resolved_output_root = ensure_dir(project_root / args.output_root)
    config_dir = ensure_dir(resolved_output_root / "config")
    logs_dir = ensure_dir(resolved_output_root / "logs")
    stage_outputs_root = ensure_dir(resolved_output_root / "stage_outputs")
    provenance_dir = ensure_dir(resolved_output_root / "provenance")
    raw_llm_dir = ensure_dir(provenance_dir / "raw_llm") if args.save_raw_llm else None

    report_root = ensure_dir(project_root / args.report_root)
    report_assets_root = ensure_dir(project_root / args.report_assets_root)

    setup_logging(logs_dir / "pipeline.log", args.log_level)

    ontology_root = (
        Path(args.ontology_root)
        if args.ontology_root
        else default_ontology_root(project_root, use_test_ontology=args.use_test_ontology)
    )

    config_payload = {
        "run_id": args.run_id,
        "paper_title": args.paper_title,
        "output_root": abs_path(resolved_output_root),
        "report_root": abs_path(report_root),
        "report_assets_root": abs_path(report_assets_root),
        "n_scenarios": args.n_scenarios,
        "n_profiles": args.n_profiles,
        "integrated_scenarios_path": args.integrated_scenarios_path,
        "seed": args.seed,
        "attack_ratio": args.attack_ratio,
        "attack_leaf": args.attack_leaf,
        "attack_leaves": args.attack_leaves,
        "opinion_leaves": args.opinion_leaves,
        "focus_opinion_domain": args.focus_opinion_domain,
        "focus_opinion_domains": args.focus_opinion_domains,
        "network_scenario_cap": args.network_scenario_cap if args.with_network_exposure else None,
        "max_opinion_leaves": args.max_opinion_leaves,
        "profile_candidate_multiplier": args.profile_candidate_multiplier,
        "primary_moderator": args.primary_moderator,
        "bootstrap_samples": args.bootstrap_samples,
        "use_test_ontology": args.use_test_ontology,
        "ontology_root": abs_path(ontology_root),
        "openrouter_model": args.openrouter_model,
        "temperature": args.temperature,
        "max_repair_iter": args.max_repair_iter,
        "profile_generation_mode": args.profile_generation_mode,
        "self_supervise_attack_realism": args.self_supervise_attack_realism,
        "realism_threshold": args.realism_threshold,
        "self_supervise_opinion_coherence": args.self_supervise_opinion_coherence,
        "coherence_threshold": args.coherence_threshold,
        "generate_visuals": args.generate_visuals,
        "export_static_figures": args.export_static_figures,
        "build_report": args.build_report,
        "with_network_exposure": args.with_network_exposure,
        "exposure_network_root": args.exposure_network_root,
        "network_exposure_top_k": args.network_exposure_top_k,
        "resume_from_stage": args.resume_from_stage,
        "stop_after_stage": args.stop_after_stage,
        "save_raw_llm": args.save_raw_llm,
        "timeout_sec": args.timeout_sec,
        "max_concurrency": args.max_concurrency,
        "run_stage_checks": args.run_stage_checks,
        "stage_check_scenarios": args.stage_check_scenarios,
        "verbose": args.verbose,
    }
    write_json(config_dir / "pipeline_config.json", config_payload)

    stage_specs = _stage_specs(project_root, with_network_exposure=args.with_network_exposure)
    stage_ids = [stage.stage_id for stage in stage_specs]
    start_idx = stage_ids.index(args.resume_from_stage)
    stop_idx = stage_ids.index(args.stop_after_stage)
    if start_idx > stop_idx:
        raise RuntimeError("resume-from-stage must be <= stop-after-stage")

    if args.build_report and not args.export_static_figures:
        raise RuntimeError("Report build requires --export-static-figures")

    previous_output: Optional[Path] = None
    existing_manifests = _load_existing_manifests(stage_specs, stage_outputs_root)
    stage_manifests: Dict[str, Dict[str, object]] = {}

    for stage in stage_specs:
        stage_idx = stage_ids.index(stage.stage_id)
        if stage_idx < start_idx:
            existing_manifest = existing_manifests.get(stage.stage_id)
            if existing_manifest:
                stage_manifests[stage.stage_id] = existing_manifest
                # Mirror the post-execution rule below: 02b/04b/05b are additive
                # side branches and must not advance the main-chain previous_output.
                # Otherwise resuming into stage 06 would feed it the last side
                # branch's file instead of Stage 05's sem_long.
                if stage.stage_id not in {"02b", "04b", "05b"}:
                    previous_output = Path(existing_manifest["primary_output_path"])
            continue
        if stage_idx > stop_idx:
            continue

        if stage.stage_id == "07" and not args.generate_visuals:
            LOGGER.info("Skipping stage 07 because --no-generate-visuals is set.")
            continue
        if stage.stage_id == "08" and not args.export_static_figures:
            LOGGER.info("Skipping stage 08 because --no-export-static-figures is set.")
            continue
        if stage.stage_id == "09" and not args.build_report:
            LOGGER.info("Skipping stage 09 because --no-build-report is set.")
            continue

        stage_output_dir = ensure_dir(stage_outputs_root / f"{stage.stage_id}_{stage.stage_name}")
        stage_log_file = logs_dir / f"stage_{stage.stage_id}_{stage.stage_name}.log"

        if stage.stage_id == "01":
            input_path = ""
        elif stage.stage_id in {"07", "08", "09"}:
            if "05" not in stage_manifests:
                raise RuntimeError(f"Stage {stage.stage_id} requires outputs from stage 05")
            input_path = str(stage_manifests["05"]["primary_output_path"])
        elif stage.stage_id == "02b":
            # Side branch: measure network exposure on the private baseline rows
            # without diverting the main 02 -> 03 chain.
            input_path = str(stage_manifests["02"]["primary_output_path"])
        elif stage.stage_id == "04b":
            # Side branch: measure post-attack network exposure on the private
            # post-attack rows without diverting the main 04 -> 05 chain.
            input_path = str(stage_manifests["04"]["primary_output_path"])
        elif stage.stage_id == "05b":
            # Side branch: empirical exposure-network analysis over Stage 05 output.
            input_path = str(stage_manifests["05"]["primary_output_path"])
        else:
            if previous_output is None:
                raise RuntimeError(f"Missing input for stage {stage.stage_id}")
            input_path = abs_path(previous_output)

        cmd = [
            sys.executable,
            abs_path(stage.script_path),
            "--output-dir",
            abs_path(stage_output_dir),
            "--input-path",
            input_path,
        ]
        cmd.extend(_build_common_stage_args(args, stage_log_file))

        if stage.stage_id == "01":
            cmd.extend(
                [
                    "--n-scenarios",
                    str(args.n_scenarios),
                    "--attack-ratio",
                    str(args.attack_ratio),
                    "--attack-leaf",
                    args.attack_leaf,
                    *(["--integrated-scenarios-path", abs_path(Path(args.integrated_scenarios_path))]
                      if args.integrated_scenarios_path else []),
                    *(["--attack-leaves", args.attack_leaves] if args.attack_leaves else []),
                    *(["--opinion-leaves", args.opinion_leaves] if args.opinion_leaves else []),
                    *(["--focus-opinion-domains", args.focus_opinion_domains] if args.focus_opinion_domains else []),
                    # The exposure-network layer caps the simulated scenario space; over the cap
                    # stage 01 engages the media-keyword heuristic. Individual-only runs are uncapped.
                    *(["--network-scenario-cap", str(args.network_scenario_cap)] if args.with_network_exposure else []),
                    "--focus-opinion-domain",
                    args.focus_opinion_domain if args.focus_opinion_domain is not None else "",
                    "--profile-generation-mode",
                    args.profile_generation_mode,
                    "--profile-candidate-multiplier",
                    str(args.profile_candidate_multiplier),
                    "--ontology-root",
                    abs_path(ontology_root),
                    "--openrouter-model",
                    args.openrouter_model,
                    "--temperature",
                    str(args.temperature),
                    "--max-repair-iter",
                    str(args.max_repair_iter),
                    "--timeout-sec",
                    str(args.timeout_sec),
                ]
            )
            if args.n_profiles is not None:
                cmd.extend(["--n-profiles", str(args.n_profiles)])
            if args.max_opinion_leaves is not None:
                cmd.extend(["--max-opinion-leaves", str(args.max_opinion_leaves)])
            if args.use_test_ontology:
                cmd.append("--use-test-ontology")
            if args.save_raw_llm and raw_llm_dir is not None:
                cmd.extend(["--save-raw-llm", "--raw-llm-dir", abs_path(raw_llm_dir)])
            cmd.append("--enforce-compatibility-rules" if args.enforce_compatibility_rules else "--no-enforce-compatibility-rules")
            cmd.append("--drop-direction-neutral-opinions" if args.drop_direction_neutral_opinions else "--no-drop-direction-neutral-opinions")
            cmd.extend(["--realism-weight-temperature", str(args.realism_weight_temperature)])

        if stage.stage_id in {"02", "03", "04"}:
            cmd.extend(
                [
                    "--openrouter-model",
                    args.openrouter_model,
                    "--temperature",
                    str(args.temperature),
                    "--max-repair-iter",
                    str(args.max_repair_iter),
                    "--timeout-sec",
                    str(args.timeout_sec),
                    "--max-concurrency",
                    str(args.max_concurrency),
                ]
            )
            if args.save_raw_llm and raw_llm_dir is not None:
                cmd.extend(["--save-raw-llm", "--raw-llm-dir", abs_path(raw_llm_dir)])

        if stage.stage_id in {"02", "04"}:
            cmd.append("--self-supervise-opinion-coherence" if args.self_supervise_opinion_coherence else "--no-self-supervise-opinion-coherence")
            cmd.extend(["--coherence-threshold", str(args.coherence_threshold)])

        # --- Additive empirical exposure-network stages (only present when on) ---
        if stage.stage_id == "01b":
            if args.exposure_network_root:
                cmd.extend(["--exposure-network-root", abs_path(Path(args.exposure_network_root))])

        if stage.stage_id in {"02b", "04b"}:
            cmd.extend(
                [
                    "--openrouter-model",
                    args.openrouter_model,
                    "--temperature",
                    str(args.temperature),
                    "--max-repair-iter",
                    str(args.max_repair_iter),
                    "--timeout-sec",
                    str(args.timeout_sec),
                    "--max-concurrency",
                    str(args.max_concurrency),
                    "--network-exposure-top-k",
                    str(args.network_exposure_top_k),
                ]
            )
            cmd.append("--self-supervise-opinion-coherence" if args.self_supervise_opinion_coherence else "--no-self-supervise-opinion-coherence")
            cmd.extend(["--coherence-threshold", str(args.coherence_threshold)])
            if args.exposure_network_root:
                cmd.extend(["--exposure-network-root", abs_path(Path(args.exposure_network_root))])
            if args.save_raw_llm and raw_llm_dir is not None:
                cmd.extend(["--save-raw-llm", "--raw-llm-dir", abs_path(raw_llm_dir)])

        if stage.stage_id == "05b":
            if args.exposure_network_root:
                cmd.extend(["--exposure-network-root", abs_path(Path(args.exposure_network_root))])

        if stage.stage_id == "03":
            cmd.append("--self-supervise-attack-realism" if args.self_supervise_attack_realism else "--no-self-supervise-attack-realism")
            cmd.extend(["--realism-threshold", str(args.realism_threshold)])
            cmd.extend(["--ontology-root", abs_path(ontology_root)])

        if stage.stage_id == "05":
            cmd.extend(["--primary-moderator", args.primary_moderator])
            cmd.extend(["--ontology-root", abs_path(ontology_root)])

        if stage.stage_id == "06":
            cmd.extend(
                [
                    "--primary-moderator",
                    args.primary_moderator,
                    "--bootstrap-samples",
                    str(args.bootstrap_samples),
                ]
            )

        if stage.stage_id == "07":
            sem_result_path = _pick_output_file(stage_manifests["06"], "sem_result.json")
            ols_params_path = _pick_output_file(stage_manifests["06"], "ols_robust_params.csv")
            cmd.extend(["--sem-result-path", sem_result_path, "--ols-params-path", ols_params_path])

        if stage.stage_id == "08":
            cmd.extend(
                [
                    "--paper-title",
                    args.paper_title,
                    "--report-assets-root",
                    abs_path(report_assets_root),
                    "--sem-result-path",
                    _pick_output_file(stage_manifests["06"], "sem_result.json"),
                    "--ols-params-path",
                    _pick_output_file(stage_manifests["06"], "ols_robust_params.csv"),
                    "--bootstrap-params-path",
                    _pick_output_file(stage_manifests["06"], "bootstrap_primary_params.csv"),
                    "--exploratory-comparison-path",
                    _pick_output_file(stage_manifests["06"], "exploratory_moderator_comparison.csv"),
                    "--config-path",
                    abs_path(config_dir / "pipeline_config.json"),
                    "--ontology-catalog-path",
                    _pick_output_file(stage_manifests["01"], "ontology_leaf_catalog.json"),
                    "--assumptions-path",
                    _pick_output_file(stage_manifests["06"], "assumption_register.json"),
                    "--critiques-path",
                    _pick_output_file(stage_manifests["06"], "peer_review_critiques.json"),
                ]
            )

        if stage.stage_id == "09":
            cmd.extend(
                [
                    "--paper-title",
                    args.paper_title,
                    "--report-root",
                    abs_path(report_root),
                    "--report-assets-root",
                    abs_path(report_assets_root),
                    "--sem-result-path",
                    _pick_output_file(stage_manifests["06"], "sem_result.json"),
                    "--ols-params-path",
                    _pick_output_file(stage_manifests["06"], "ols_robust_params.csv"),
                    "--bootstrap-params-path",
                    _pick_output_file(stage_manifests["06"], "bootstrap_primary_params.csv"),
                    "--exploratory-comparison-path",
                    _pick_output_file(stage_manifests["06"], "exploratory_moderator_comparison.csv"),
                    "--config-path",
                    abs_path(config_dir / "pipeline_config.json"),
                ]
            )

        expected_calls = args.n_scenarios if stage.stage_id in {"02", "04", "02b", "04b"} else None
        _call_stage(
            stage,
            cmd,
            project_root,
            verbose=args.verbose,
            raw_llm_dir=raw_llm_dir,
            expected_calls=expected_calls,
        )

        manifest_path = stage_output_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing manifest for stage {stage.stage_id}: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_manifests[stage.stage_id] = manifest
        # 02b / 04b / 05b are additive side branches: keep them out of the
        # main-chain pointer so 03 still consumes 02's output, 05 consumes 04's,
        # and 06 consumes 05's.
        if stage.stage_id not in {"02b", "04b", "05b"}:
            previous_output = Path(manifest["primary_output_path"])
        LOGGER.info("Stage %s complete. Primary output: %s", stage.stage_id, Path(manifest["primary_output_path"]))

    _copy_outputs(stage_outputs_root, resolved_output_root)

    write_json(
        provenance_dir / "run_manifest.json",
        {
            "pipeline_config": config_payload,
            "stage_outputs_root": abs_path(stage_outputs_root),
            "logs_dir": abs_path(logs_dir),
            "datasets_dir": abs_path(resolved_output_root / "datasets"),
            "sem_dir": abs_path(resolved_output_root / "sem"),
            "report_dir": abs_path(resolved_output_root / "report"),
            "visuals_dir": abs_path(resolved_output_root / "visuals"),
            "publication_assets_dir": abs_path(resolved_output_root / "publication_assets"),
            "paper_dir": abs_path(resolved_output_root / "paper"),
        },
    )

    if args.run_stage_checks:
        _run_stage_checks(
            project_root=project_root,
            output_root=resolved_output_root,
            args=args,
            ontology_root=ontology_root,
        )

    LOGGER.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
