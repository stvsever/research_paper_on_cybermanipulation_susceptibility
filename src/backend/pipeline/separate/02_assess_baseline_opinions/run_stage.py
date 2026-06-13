from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import sys
import threading
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.agentic_framework.agents import OpinionCoherenceReviewResponse
from src.backend.agentic_framework.factory import AgentFactory
from src.backend.utils.io import (
    abs_path,
    env_get_required,
    read_jsonl,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.scenario_realism import assess_baseline_opinion_heuristics, profile_context_snapshot
from src.backend.utils.schemas import OpinionAssessment, ScenarioRecord, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


class Stage02Config(StageConfig):
    self_supervise_opinion_coherence: bool = True
    coherence_threshold: float = 0.72


def _fallback_score(scenario: ScenarioRecord) -> int:
    seed_text = f"{scenario.scenario_id}:{scenario.opinion_leaf}:{scenario.profile.profile_id}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return int((value % 2001) - 1000)


def _default_review() -> OpinionCoherenceReviewResponse:
    return OpinionCoherenceReviewResponse(
        plausibility_score=0.0,
        consistency_score=0.0,
        rewrite_required=False,
        rewrite_feedback="",
        notes="review_unavailable",
    )


def run_stage(input_path: str, output_dir: str, config: Stage02Config) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 02 requires --openrouter-model")

    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"

    scenario_rows = read_jsonl(input_path)
    scenarios = [ScenarioRecord.model_validate(row) for row in scenario_rows]

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
                factory.baseline_opinion_agent(),
                factory.opinion_coherence_reviewer_agent(),
            )
        return thread_local.bundle

    def _process_scenario(scenario: ScenarioRecord) -> Dict[str, object]:
        agent, reviewer_agent = _agents_for_thread()
        local_review_rewrite_count = 0
        local_heuristic_fail_count = 0

        try:
            assessment = agent.assess(
                run_id=config.run_id,
                call_id=f"{scenario.scenario_id}_baseline",
                scenario_id=scenario.scenario_id,
                opinion_leaf=scenario.opinion_leaf,
                profile=scenario.profile,
            )
        except Exception as exc:
            LOGGER.warning(
                "Baseline agent failed for %s, using deterministic fallback: %s",
                scenario.scenario_id,
                exc,
            )
            assessment = OpinionAssessment(
                scenario_id=scenario.scenario_id,
                phase="baseline",
                opinion_leaf=scenario.opinion_leaf,
                score=_fallback_score(scenario),
                confidence=0.3,
                reasoning="Deterministic fallback due to agent failure.",
                model_name="fallback_deterministic",
            )

        heuristics = assess_baseline_opinion_heuristics(
            score=assessment.score,
            confidence=assessment.confidence,
        )
        review = _default_review()

        if config.self_supervise_opinion_coherence and assessment.model_name != "fallback_deterministic":
            try:
                review = reviewer_agent.review(
                    run_id=config.run_id,
                    call_id=f"{scenario.scenario_id}_baseline_review_1",
                    phase="baseline",
                    scenario_id=scenario.scenario_id,
                    opinion_leaf=scenario.opinion_leaf,
                    profile_snapshot=profile_context_snapshot(scenario.profile),
                    generated_assessment=assessment,
                    attack_present=False,
                    heuristic_checks=heuristics,
                )
            except Exception as exc:
                LOGGER.warning("Baseline coherence reviewer failed for %s: %s", scenario.scenario_id, exc)
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
                    feedback_parts.append(
                        "Use a high-resolution non-coarse score within the allowed range."
                    )
                try:
                    assessment = agent.assess(
                        run_id=config.run_id,
                        call_id=f"{scenario.scenario_id}_baseline_rewrite",
                        scenario_id=scenario.scenario_id,
                        opinion_leaf=scenario.opinion_leaf,
                        profile=scenario.profile,
                        review_feedback=" ".join(feedback_parts).strip(),
                    )
                    heuristics = assess_baseline_opinion_heuristics(
                        score=assessment.score,
                        confidence=assessment.confidence,
                    )
                    try:
                        review = reviewer_agent.review(
                            run_id=config.run_id,
                            call_id=f"{scenario.scenario_id}_baseline_review_2",
                            phase="baseline",
                            scenario_id=scenario.scenario_id,
                            opinion_leaf=scenario.opinion_leaf,
                            profile_snapshot=profile_context_snapshot(scenario.profile),
                            generated_assessment=assessment,
                            attack_present=False,
                            heuristic_checks=heuristics,
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    LOGGER.warning("Baseline rewrite failed for %s: %s", scenario.scenario_id, exc)

        if not bool(heuristics["checks"].get("overall_pass", False)):
            local_heuristic_fail_count += 1
        row = scenario.model_dump()
        row["baseline_assessment"] = assessment.model_dump()
        row["baseline_coherence_review"] = review.model_dump()
        row["baseline_heuristic_checks"] = heuristics
        return {
            "assessment": assessment,
            "enriched_row": row,
            "plausibility_score": float(review.plausibility_score),
            "consistency_score": float(review.consistency_score),
            "review_rewrite_count": local_review_rewrite_count,
            "heuristic_fail_count": local_heuristic_fail_count,
        }

    max_workers = max(1, int(config.max_concurrency or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_scenario, scenarios))

    assessments = [result["assessment"] for result in results]
    enriched_rows = [result["enriched_row"] for result in results]
    plausibility_scores = [float(result["plausibility_score"]) for result in results]
    consistency_scores = [float(result["consistency_score"]) for result in results]
    review_rewrite_count = int(sum(int(result["review_rewrite_count"]) for result in results))
    heuristic_fail_count = int(sum(int(result["heuristic_fail_count"]) for result in results))

    baseline_jsonl = Path(output_dir) / "baseline_assessments.jsonl"
    enriched_jsonl = Path(output_dir) / "scenarios_with_baseline.jsonl"
    summary_json = Path(output_dir) / "baseline_summary.json"

    write_jsonl(baseline_jsonl, (a.model_dump() for a in assessments))
    write_jsonl(enriched_jsonl, enriched_rows)

    fallback_count = sum(1 for a in assessments if a.model_name == "fallback_deterministic")
    write_json(
        summary_json,
        {
            "n_records": len(assessments),
            "fallback_count": fallback_count,
            "score_min": min(a.score for a in assessments),
            "score_max": max(a.score for a in assessments),
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
            "mean_plausibility_score": (sum(plausibility_scores) / len(plausibility_scores)) if plausibility_scores else None,
            "mean_consistency_score": (sum(consistency_scores) / len(consistency_scores)) if consistency_scores else None,
        },
    )

    manifest = StageArtifactManifest(
        stage_id="02",
        stage_name="assess_baseline_opinions",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(baseline_jsonl), abs_path(enriched_jsonl), abs_path(summary_json)],
        record_count=len(assessments),
        metadata={
            "fallback_count": fallback_count,
            "openrouter_model": config.openrouter_model,
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02 - Baseline opinions")
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

    config = Stage02Config(
        stage_name="assess_baseline_opinions",
        run_id=args.run_id,
        seed=args.seed,
        openrouter_model=args.openrouter_model,
        temperature=args.temperature,
        max_repair_iter=args.max_repair_iter,
        self_supervise_opinion_coherence=args.self_supervise_opinion_coherence,
        coherence_threshold=args.coherence_threshold,
        save_raw_llm=args.save_raw_llm,
        raw_llm_dir=args.raw_llm_dir,
        timeout_sec=args.timeout_sec,
        max_concurrency=args.max_concurrency,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 02 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
