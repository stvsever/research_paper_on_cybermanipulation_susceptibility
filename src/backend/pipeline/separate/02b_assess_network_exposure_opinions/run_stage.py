from __future__ import annotations

"""
Technical overview
------------------
Stage 02b measures the pre-attack network-exposure baseline. It starts from
private baseline opinions produced by Stage 02 and re-elicits each profile's
opinion after adding context from empirically assigned incoming exposure peers.

The peer set is not selected by profile similarity. It is resolved through the
Stage 01b PolitiSky24 position assignment and the directed exposure graph:

    visible peer profile -> exposed target profile

For each profile x opinion task, the stage summarizes all scored incoming peers
for analysis while sending only a bounded set of peer exemplars to the LLM
prompt. The original private baseline assessment is preserved unchanged, so
network exposure remains an additional measurement phase rather than a
replacement baseline.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import statistics
import sys
import threading
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.agentic_framework.agents import OpinionCoherenceReviewResponse
from src.backend.agentic_framework.factory import AgentFactory
from src.backend.utils.io import (
    abs_path,
    clean_filename,
    env_get_required,
    read_jsonl,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.live_artifacts import (
    append_live_error,
    append_live_result,
    init_live_stage,
    update_live_status,
)
from src.backend.utils.network_exposure import (
    build_baseline_network_context,
    exposure_assignments_from_rows,
    load_edge_index,
    load_exposure_network_package,
)
from src.backend.utils.scenario_realism import assess_baseline_opinion_heuristics, profile_context_snapshot
from src.backend.utils.schemas import (
    OpinionAssessment,
    ProfileConfiguration,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)

LOGGER = logging.getLogger(__name__)


class Stage02bConfig(StageConfig):
    self_supervise_opinion_coherence: bool = True
    coherence_threshold: float = 0.72
    network_exposure_top_k: int = 8
    exposure_network_root: str | None = None


@dataclass(frozen=True)
class CanonicalBaselineTask:
    profile_id: str
    opinion_leaf: str
    scenario_id: str
    profile: ProfileConfiguration
    baseline_score: int
    baseline_confidence: float
    baseline_reasoning: str
    baseline_replicate_count: int
    baseline_score_sd: float


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _fallback_score(task: CanonicalBaselineTask) -> int:
    seed_text = f"{task.scenario_id}:{task.opinion_leaf}:{task.profile_id}:network_exposure"
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    shift = int(digest[:8], 16) % 81 - 40
    return max(-1000, min(1000, int(task.baseline_score + shift)))


def _default_review() -> OpinionCoherenceReviewResponse:
    return OpinionCoherenceReviewResponse(
        plausibility_score=0.0,
        consistency_score=0.0,
        rewrite_required=False,
        rewrite_feedback="",
        notes="review_unavailable",
    )


def _measurement_id(profile_id: str, opinion_leaf: str) -> str:
    return f"{profile_id}_{clean_filename(opinion_leaf)}_network_exposure"


def _build_canonical_tasks(
    scenario_rows: list[dict[str, Any]],
) -> tuple[list[CanonicalBaselineTask], list[tuple[dict[str, Any], ScenarioRecord, OpinionAssessment]]]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], ScenarioRecord, OpinionAssessment]]] = {}
    parsed_rows: list[tuple[dict[str, Any], ScenarioRecord, OpinionAssessment]] = []

    for row in scenario_rows:
        if "baseline_assessment" not in row:
            raise RuntimeError("Stage 02b requires Stage 02 rows with baseline_assessment.")
        scenario = ScenarioRecord.model_validate(row)
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])
        if baseline.phase != "baseline":
            raise RuntimeError(f"Expected baseline phase for {scenario.scenario_id}; got {baseline.phase}")
        parsed = (row, scenario, baseline)
        parsed_rows.append(parsed)
        grouped.setdefault((scenario.profile.profile_id, scenario.opinion_leaf), []).append(parsed)

    tasks: list[CanonicalBaselineTask] = []
    for (profile_id, opinion_leaf), items in sorted(grouped.items(), key=lambda item: item[0]):
        items_sorted = sorted(items, key=lambda item: item[1].scenario_id)
        scores = [float(item[2].score) for item in items_sorted]
        confidences = [float(item[2].confidence) for item in items_sorted]
        _, first_scenario, first_baseline = items_sorted[0]
        tasks.append(
            CanonicalBaselineTask(
                profile_id=profile_id,
                opinion_leaf=opinion_leaf,
                scenario_id=_measurement_id(profile_id, opinion_leaf),
                profile=first_scenario.profile,
                baseline_score=int(round(statistics.mean(scores))),
                baseline_confidence=float(statistics.mean(confidences)) if confidences else 0.0,
                baseline_reasoning=str(first_baseline.reasoning),
                baseline_replicate_count=len(items_sorted),
                baseline_score_sd=float(_safe_pstdev(scores) or 0.0),
            )
        )
    return tasks, parsed_rows


def _build_network_contexts(
    tasks: list[CanonicalBaselineTask],
    max_exemplars: int,
    assignments_by_profile: dict[str, Any],
    package: Any,
    edge_index: Any,
) -> dict[tuple[str, str], dict[str, Any]]:
    tasks_by_opinion: dict[str, list[CanonicalBaselineTask]] = {}
    for task in tasks:
        tasks_by_opinion.setdefault(task.opinion_leaf, []).append(task)
    contexts: dict[tuple[str, str], dict[str, Any]] = {}

    for task in tasks:
        peer_payloads = {
            peer.profile_id: {
                "score": int(peer.baseline_score),
                "confidence": round(float(peer.baseline_confidence), 4),
                "reasoning": peer.baseline_reasoning,
            }
            for peer in tasks_by_opinion.get(task.opinion_leaf, [])
            if peer.profile_id != task.profile_id
        }
        context = build_baseline_network_context(
            target_profile_id=task.profile_id,
            target_baseline_assessment={
                "score": int(task.baseline_score),
                "confidence": float(task.baseline_confidence),
                "reasoning": task.baseline_reasoning,
            },
            peer_baseline_assessments_by_profile=peer_payloads,
            assignments_by_profile=assignments_by_profile,
            package=package,
            max_exemplars=max_exemplars,
            edge_index=edge_index,
        )
        context.update(
            {
                "opinion_leaf": task.opinion_leaf,
                "target_baseline_replicate_count": int(task.baseline_replicate_count),
                "target_baseline_score_sd": round(float(task.baseline_score_sd), 4),
                "max_exemplars": int(max_exemplars),
                "top_k_legacy_arg": int(max_exemplars),
            }
        )
        contexts[(task.profile_id, task.opinion_leaf)] = context
    return contexts


def _assignment_graph_id(assignments_by_profile: dict[str, Any]) -> str:
    graph_ids = {str(assignment.graph_id) for assignment in assignments_by_profile.values()}
    if len(graph_ids) != 1:
        raise RuntimeError(f"Expected one exposure graph id in assignments, found {sorted(graph_ids)}")
    return next(iter(graph_ids))


def _exposure_network_provenance(package: Any) -> dict[str, Any]:
    return {
        "graph_id": package.graph_id,
        "graph_root": str(package.root),
        "edge_direction": package.manifest.get("edge_direction"),
        "edge_meaning": package.manifest.get("edge_meaning"),
        "interaction_weight_formula": package.manifest.get("interaction_weight_formula"),
    }


def _context_mean(contexts: list[dict[str, Any]], key: str) -> float | None:
    values = [float(context.get(key) or 0.0) for context in contexts]
    return float(statistics.mean(values)) if values else None


def _enrich_row_with_result(
    source_row: dict[str, Any],
    baseline: OpinionAssessment,
    result: Dict[str, object],
) -> dict[str, Any]:
    row = dict(source_row)
    assessment = result["assessment"]
    assert isinstance(assessment, OpinionAssessment)
    row["network_exposure_assessment"] = assessment.model_dump()
    row["network_exposure_context"] = result["network_context"]
    row["network_exposure_coherence_review"] = result["review"].model_dump()
    row["network_exposure_heuristic_checks"] = result["heuristics"]
    row["network_exposure_delta_score"] = int(assessment.score - baseline.score)
    return row


def run_stage(input_path: str, output_dir: str, config: Stage02bConfig) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 02b requires --openrouter-model")

    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"

    scenario_rows = [dict(row) for row in read_jsonl(input_path)]
    tasks, parsed_rows = _build_canonical_tasks(scenario_rows)
    max_exemplars = max(0, int(config.network_exposure_top_k))
    assignments_by_profile = exposure_assignments_from_rows(scenario_rows)
    graph_id = _assignment_graph_id(assignments_by_profile)
    package = load_exposure_network_package(
        graph_root=config.exposure_network_root,
        graph_id=graph_id,
        validate=True,
    )
    assigned_positions = {assignment.position_id for assignment in assignments_by_profile.values()}
    edge_index = load_edge_index(
        package,
        target_positions=assigned_positions,
        source_positions=assigned_positions,
    )
    network_contexts = _build_network_contexts(
        tasks,
        max_exemplars=max_exemplars,
        assignments_by_profile=assignments_by_profile,
        package=package,
        edge_index=edge_index,
    )

    thread_local = threading.local()

    def _agents_for_thread() -> tuple:
        if not hasattr(thread_local, "bundle"):
            factory = AgentFactory(
                prompts_dir=prompts_dir,
                openrouter_api_key=env_get_required("OPENROUTER_API_KEY"),
                openrouter_model=config.openrouter_model,
                max_repair_iter=config.max_repair_iter,
                temperature=config.temperature,
                timeout_sec=config.timeout_sec,
                save_raw_dir=raw_dir,
            )
            thread_local.bundle = (
                factory.network_exposure_opinion_agent(),
                factory.opinion_coherence_reviewer_agent(),
            )
        return thread_local.bundle

    def _process_task(task: CanonicalBaselineTask) -> Dict[str, object]:
        network_context = network_contexts[(task.profile_id, task.opinion_leaf)]
        local_review_rewrite_count = 0
        local_heuristic_fail_count = 0
        agent, reviewer_agent = _agents_for_thread()

        try:
            assessment = agent.assess(
                run_id=config.run_id,
                call_id=f"{task.scenario_id}_network_exposure",
                scenario_id=task.scenario_id,
                opinion_leaf=task.opinion_leaf,
                profile=task.profile,
                baseline_score=task.baseline_score,
                network_context=network_context,
            )
        except Exception as exc:
            LOGGER.warning(
                "Network-exposure agent failed for %s, using deterministic fallback: %s",
                task.scenario_id,
                exc,
            )
            assessment = OpinionAssessment(
                scenario_id=task.scenario_id,
                phase="network_exposure_baseline",
                opinion_leaf=task.opinion_leaf,
                score=_fallback_score(task),
                confidence=0.3,
                reasoning="Deterministic fallback due to agent failure.",
                model_name="fallback_deterministic",
            )

        heuristics = assess_baseline_opinion_heuristics(
            score=assessment.score,
            confidence=assessment.confidence,
        )
        review = _default_review()

        if (
            config.self_supervise_opinion_coherence
            and assessment.model_name != "fallback_deterministic"
        ):
            try:
                review = reviewer_agent.review(
                    run_id=config.run_id,
                    call_id=f"{task.scenario_id}_network_exposure_review_1",
                    phase="network_exposure_baseline",
                    scenario_id=task.scenario_id,
                    opinion_leaf=task.opinion_leaf,
                    profile_snapshot=profile_context_snapshot(task.profile),
                    generated_assessment=assessment,
                    attack_present=False,
                    baseline_score=task.baseline_score,
                    attack_vector_spec={"network_context": network_context},
                    heuristic_checks=heuristics,
                )
            except Exception as exc:
                LOGGER.warning("Network-exposure coherence reviewer failed for %s: %s", task.scenario_id, exc)
                review = _default_review()

            needs_rewrite = (
                review.rewrite_required
                or review.plausibility_score < config.coherence_threshold
                or review.consistency_score < config.coherence_threshold
                or not bool(heuristics["checks"].get("overall_pass", False))
            )
            if needs_rewrite:
                local_review_rewrite_count += 1
                feedback_parts = []
                if review.rewrite_feedback:
                    feedback_parts.append(review.rewrite_feedback)
                if not bool(heuristics["checks"].get("overall_pass", False)):
                    feedback_parts.append("Use a high-resolution non-coarse score within the allowed range.")
                try:
                    assessment = agent.assess(
                        run_id=config.run_id,
                        call_id=f"{task.scenario_id}_network_exposure_rewrite",
                        scenario_id=task.scenario_id,
                        opinion_leaf=task.opinion_leaf,
                        profile=task.profile,
                        baseline_score=task.baseline_score,
                        network_context=network_context,
                        review_feedback=" ".join(feedback_parts).strip(),
                    )
                    heuristics = assess_baseline_opinion_heuristics(
                        score=assessment.score,
                        confidence=assessment.confidence,
                    )
                    try:
                        review = reviewer_agent.review(
                            run_id=config.run_id,
                            call_id=f"{task.scenario_id}_network_exposure_review_2",
                            phase="network_exposure_baseline",
                            scenario_id=task.scenario_id,
                            opinion_leaf=task.opinion_leaf,
                            profile_snapshot=profile_context_snapshot(task.profile),
                            generated_assessment=assessment,
                            attack_present=False,
                            baseline_score=task.baseline_score,
                            attack_vector_spec={"network_context": network_context},
                            heuristic_checks=heuristics,
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    LOGGER.warning("Network-exposure rewrite failed for %s: %s", task.scenario_id, exc)

        if not bool(heuristics["checks"].get("overall_pass", False)):
            local_heuristic_fail_count += 1

        return {
            "task": task,
            "assessment": assessment,
            "network_context": network_context,
            "review": review,
            "heuristics": heuristics,
            "plausibility_score": float(review.plausibility_score),
            "consistency_score": float(review.consistency_score),
            "review_rewrite_count": local_review_rewrite_count,
            "heuristic_fail_count": local_heuristic_fail_count,
        }

    parsed_by_key: dict[tuple[str, str], tuple[dict[str, Any], ScenarioRecord, OpinionAssessment]] = {}
    for source_row, scenario, baseline in parsed_rows:
        parsed_by_key.setdefault((scenario.profile.profile_id, scenario.opinion_leaf), (source_row, scenario, baseline))

    def _enriched_row_from_result(result: Dict[str, object]) -> dict[str, Any]:
        task = result["task"]
        assert isinstance(task, CanonicalBaselineTask)
        source_row, _, baseline = parsed_by_key[(task.profile_id, task.opinion_leaf)]
        return _enrich_row_with_result(source_row, baseline, result)

    init_live_stage(
        output_dir,
        run_id=config.run_id,
        stage_id="02b",
        stage_name="assess_network_exposure_opinions",
        phase="network_exposure_baseline",
        total_count=len(tasks),
    )
    max_workers = max(1, int(config.max_concurrency or 1))
    results: list[Dict[str, object] | None] = [None] * len(tasks)
    completed_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(_process_task, task): index for index, task in enumerate(tasks)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            task = tasks[index]
            try:
                result = future.result()
            except Exception as exc:
                failed_count += 1
                append_live_error(
                    output_dir,
                    {
                        "scenario_id": task.scenario_id,
                        "profile_id": task.profile_id,
                        "opinion_leaf": task.opinion_leaf,
                        "message": str(exc),
                    },
                )
                update_live_status(output_dir, completed_count=completed_count, failed_count=failed_count, status="failed")
                raise
            results[index] = result
            completed_count += 1
            append_live_result(output_dir, _enriched_row_from_result(result))
            update_live_status(output_dir, completed_count=completed_count, failed_count=failed_count, status="running")

    results = [result for result in results if result is not None]

    result_by_key = {
        (result["task"].profile_id, result["task"].opinion_leaf): result
        for result in results
    }

    enriched_rows: list[dict[str, Any]] = []
    for source_row, scenario, baseline in parsed_rows:
        result = result_by_key[(scenario.profile.profile_id, scenario.opinion_leaf)]
        enriched_rows.append(_enrich_row_with_result(source_row, baseline, result))

    assessments_jsonl = Path(output_dir) / "network_exposure_assessments.jsonl"
    contexts_jsonl = Path(output_dir) / "network_contexts.jsonl"
    enriched_jsonl = Path(output_dir) / "scenarios_with_network_exposure.jsonl"
    summary_json = Path(output_dir) / "network_exposure_summary.json"

    write_jsonl(
        assessments_jsonl,
        (
            {
                "profile_id": result["task"].profile_id,
                "opinion_leaf": result["task"].opinion_leaf,
                "baseline_score": result["task"].baseline_score,
                "network_exposure_assessment": result["assessment"].model_dump(),
                "network_exposure_coherence_review": result["review"].model_dump(),
                "network_exposure_heuristic_checks": result["heuristics"],
            }
            for result in results
        ),
    )
    write_jsonl(
        contexts_jsonl,
        (
            {
                "profile_id": result["task"].profile_id,
                "scenario_id": result["task"].scenario_id,
                "opinion_leaf": result["task"].opinion_leaf,
                "network_context": result["network_context"],
            }
            for result in results
        ),
    )
    write_jsonl(enriched_jsonl, enriched_rows)

    fallback_count = sum(
        1 for result in results if result["assessment"].model_name == "fallback_deterministic"
    )
    review_rewrite_count = int(sum(int(result["review_rewrite_count"]) for result in results))
    heuristic_fail_count = int(sum(int(result["heuristic_fail_count"]) for result in results))
    contexts = [result["network_context"] for result in results]
    peer_counts = [int(context.get("peer_count") or 0) for context in contexts]
    full_incoming_counts = [int(context.get("full_incoming_peer_count") or 0) for context in contexts]
    exemplar_counts = [int(context.get("exemplar_count") or 0) for context in contexts]
    scores = [int(result["assessment"].score) for result in results]
    deltas = [
        int(result["assessment"].score - result["task"].baseline_score)
        for result in results
    ]

    write_json(
        summary_json,
        {
            "n_records": len(enriched_rows),
            "n_unique_tasks": len(results),
            "max_exemplars": max_exemplars,
            "top_k_legacy_arg": max_exemplars,
            "fallback_count": fallback_count,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "mean_network_exposure_delta_score": float(statistics.mean(deltas)) if deltas else None,
            "mean_abs_network_exposure_delta_score": (
                float(statistics.mean(abs(value) for value in deltas)) if deltas else None
            ),
            "mean_full_incoming_peer_count": float(statistics.mean(full_incoming_counts)) if full_incoming_counts else None,
            "mean_scored_peer_count": float(statistics.mean(peer_counts)) if peer_counts else None,
            "mean_exemplar_count": float(statistics.mean(exemplar_counts)) if exemplar_counts else None,
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
            "exposure_network": _exposure_network_provenance(package),
        },
    )

    manifest = StageArtifactManifest(
        stage_id="02b",
        stage_name="assess_network_exposure_opinions",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[
            abs_path(assessments_jsonl),
            abs_path(contexts_jsonl),
            abs_path(enriched_jsonl),
            abs_path(summary_json),
        ],
        record_count=len(enriched_rows),
        metadata={
            "n_unique_tasks": len(results),
            "max_exemplars": max_exemplars,
            "top_k_legacy_arg": max_exemplars,
            "fallback_count": fallback_count,
            "openrouter_model": config.openrouter_model,
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
            "mean_full_incoming_peer_count": _context_mean(contexts, "full_incoming_peer_count"),
            "mean_scored_peer_count": _context_mean(contexts, "peer_count"),
            "mean_exemplar_count": _context_mean(contexts, "exemplar_count"),
            "exposure_network": _exposure_network_provenance(package),
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    update_live_status(output_dir, completed_count=completed_count, failed_count=failed_count, status="completed")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02b - Network-exposure baseline opinions")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openrouter-model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=2)
    parser.add_argument(
        "--self-supervise-opinion-coherence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--coherence-threshold", type=float, default=0.72)
    parser.add_argument("--network-exposure-top-k", type=int, default=8)
    parser.add_argument("--exposure-network-root", default=None)
    parser.add_argument("--save-raw-llm", action="store_true", default=False)
    parser.add_argument("--raw-llm-dir", default=None)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")

    config = Stage02bConfig(
        stage_name="assess_network_exposure_opinions",
        run_id=args.run_id,
        seed=args.seed,
        openrouter_model=args.openrouter_model,
        temperature=args.temperature,
        max_repair_iter=args.max_repair_iter,
        self_supervise_opinion_coherence=args.self_supervise_opinion_coherence,
        coherence_threshold=args.coherence_threshold,
        network_exposure_top_k=args.network_exposure_top_k,
        exposure_network_root=args.exposure_network_root,
        save_raw_llm=args.save_raw_llm,
        raw_llm_dir=args.raw_llm_dir,
        timeout_sec=args.timeout_sec,
        max_concurrency=args.max_concurrency,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 02b completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
