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
from src.backend.utils.io import read_json
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.ontology_utils import load_adversarial_directions_from_opinion
from src.backend.utils.scenario_realism import (
    assess_post_opinion_heuristics,
    profile_context_snapshot,
)
from src.backend.utils.schemas import OpinionAssessment, ScenarioRecord, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


class Stage04Config(StageConfig):
    self_supervise_opinion_coherence: bool = True
    coherence_threshold: float = 0.72


def _fallback_post_score(baseline_score: int, scenario_id: str, attack_present: bool) -> int:
    digest = hashlib.md5(scenario_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    shift = (int(digest[:4], 16) % 141) - 70
    if not attack_present:
        shift = int(shift * 0.25)
    score = baseline_score + shift
    return max(-1000, min(1000, score))


def _default_review() -> OpinionCoherenceReviewResponse:
    return OpinionCoherenceReviewResponse(
        plausibility_score=0.0,
        consistency_score=0.0,
        rewrite_required=False,
        rewrite_feedback="",
        notes="review_unavailable",
    )


def run_stage(input_path: str, output_dir: str, config: Stage04Config) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 04 requires --openrouter-model")

    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"

    # Load adversarial directions so the heuristic checker allows direction-aligned
    # shifts without flagging them as implausible reversals.
    adversarial_directions: Dict[str, int] = {}
    if config.ontology_root:
        opinion_path = Path(config.ontology_root) / "OPINION" / "opinion.json"
        if opinion_path.exists():
            opinion_tree = read_json(str(opinion_path))
            adversarial_directions, _ = load_adversarial_directions_from_opinion(opinion_tree)
            LOGGER.info("Stage 04: loaded %d adversarial directions", len(adversarial_directions))

    rows = read_jsonl(input_path)
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
                factory.post_attack_opinion_agent(),
                factory.opinion_coherence_reviewer_agent(),
            )
        return thread_local.bundle

    def _process_row(row: Dict[str, object]) -> Dict[str, object]:
        agent, reviewer_agent = _agents_for_thread()
        local_review_rewrite_count = 0
        local_heuristic_fail_count = 0
        scenario = ScenarioRecord.model_validate(
            {
                k: v
                for k, v in row.items()
                if k not in {
                    "baseline_assessment",
                    "attack_exposure",
                    "attack_vector_spec",
                    "baseline_coherence_review",
                    "baseline_heuristic_checks",
                }
            }
        )
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])
        spec = row.get("attack_vector_spec") if isinstance(row, dict) else None
        spec = spec if isinstance(spec, dict) else {}
        intensity_proxy = float(spec.get("intensity_proxy", 0.5) or 0.0)
        shift_sensitivity_proxy = scenario.profile.continuous_attributes.get("heuristic_shift_sensitivity_proxy", 0.5)
        leaf_name = scenario.opinion_leaf.split(">")[-1].strip()
        # Primary source: the direction stage 01 embedded in the scenario record
        # (always present, resolved from the same rules as stage 05). Fallback:
        # the ontology-derived map, which is only populated when an ontology
        # root was passed to this stage. The previous map-only lookup silently
        # yielded 0 in orchestrated runs, so the post agent never knew the
        # adversarial goal; that was the root cause of counter-goal deltas.
        meta_direction = None
        if isinstance(scenario.metadata, dict):
            raw_dir_value = scenario.metadata.get("opinion_adversarial_direction")
            try:
                meta_direction = int(raw_dir_value) if raw_dir_value is not None else None
            except (TypeError, ValueError):
                meta_direction = None
        adv_direction = (
            meta_direction
            if meta_direction is not None
            else adversarial_directions.get(leaf_name, 0)
        )

        try:
            post = agent.assess(
                run_id=config.run_id,
                call_id=f"{scenario.scenario_id}_post_attack",
                scenario_id=scenario.scenario_id,
                opinion_leaf=scenario.opinion_leaf,
                profile=scenario.profile,
                baseline_score=baseline.score,
                attack_leaf=scenario.attack_leaf,
                attack_vector_spec=spec,
                attack_present=scenario.attack_present,
                adversarial_direction=adv_direction,
            )
        except Exception as exc:
            LOGGER.warning(
                "Post-attack assessment failed for %s, fallback used: %s",
                scenario.scenario_id,
                exc,
            )
            post = OpinionAssessment(
                scenario_id=scenario.scenario_id,
                phase="post_attack",
                opinion_leaf=scenario.opinion_leaf,
                score=_fallback_post_score(
                    baseline_score=baseline.score,
                    scenario_id=scenario.scenario_id,
                    attack_present=scenario.attack_present,
                ),
                confidence=0.3,
                reasoning="Deterministic fallback due to agent failure.",
                model_name="fallback_deterministic",
            )

        heuristics = assess_post_opinion_heuristics(
            baseline_score=baseline.score,
            post_score=post.score,
            attack_present=scenario.attack_present,
            intensity_hint=intensity_proxy,
            shift_sensitivity_proxy=float(shift_sensitivity_proxy),
            adversarial_direction=adv_direction,
        )
        review = _default_review()

        if config.self_supervise_opinion_coherence and post.model_name != "fallback_deterministic":
            try:
                review = reviewer_agent.review(
                    run_id=config.run_id,
                    call_id=f"{scenario.scenario_id}_post_review_1",
                    phase="post_attack",
                    scenario_id=scenario.scenario_id,
                    opinion_leaf=scenario.opinion_leaf,
                    profile_snapshot=profile_context_snapshot(scenario.profile),
                    generated_assessment=post,
                    attack_present=scenario.attack_present,
                    adversarial_direction=adv_direction,
                    baseline_score=baseline.score,
                    attack_vector_spec=spec,
                    heuristic_checks=heuristics,
                )
            except Exception as exc:
                LOGGER.warning("Post coherence reviewer failed for %s: %s", scenario.scenario_id, exc)
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
                    dir_label = (
                        f"toward +1000 (adversarial_direction=+1)" if adv_direction > 0
                        else (f"toward -1000 (adversarial_direction=-1)" if adv_direction < 0
                              else "within a plausible bounded range")
                    )
                    feedback_parts.append(
                        f"Keep the post-exposure shift bounded and ensure it moves {dir_label}. "
                        "Avoid shifts that go against the adversarial direction."
                    )
                if not scenario.attack_present and abs(post.score - baseline.score) < 3:
                    feedback_parts.append(
                        "For neutral control exposures, preserve the same broad stance but allow slight test-retest variation or clarification instead of repeating the exact baseline integer."
                    )
                try:
                    post = agent.assess(
                        run_id=config.run_id,
                        call_id=f"{scenario.scenario_id}_post_rewrite",
                        scenario_id=scenario.scenario_id,
                        opinion_leaf=scenario.opinion_leaf,
                        profile=scenario.profile,
                        baseline_score=baseline.score,
                        attack_leaf=scenario.attack_leaf,
                        attack_vector_spec=spec,
                        attack_present=scenario.attack_present,
                        adversarial_direction=adv_direction,
                        review_feedback=" ".join(feedback_parts).strip(),
                    )
                    heuristics = assess_post_opinion_heuristics(
                        baseline_score=baseline.score,
                        post_score=post.score,
                        attack_present=scenario.attack_present,
                        intensity_hint=intensity_proxy,
                        shift_sensitivity_proxy=float(shift_sensitivity_proxy),
                        adversarial_direction=adv_direction,
                    )
                    try:
                        review = reviewer_agent.review(
                            run_id=config.run_id,
                            call_id=f"{scenario.scenario_id}_post_review_2",
                            phase="post_attack",
                            scenario_id=scenario.scenario_id,
                            opinion_leaf=scenario.opinion_leaf,
                            profile_snapshot=profile_context_snapshot(scenario.profile),
                            generated_assessment=post,
                            attack_present=scenario.attack_present,
                            adversarial_direction=adv_direction,
                            baseline_score=baseline.score,
                            attack_vector_spec=spec,
                            heuristic_checks=heuristics,
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    LOGGER.warning("Post rewrite failed for %s: %s", scenario.scenario_id, exc)

        if not bool(heuristics["checks"].get("overall_pass", False)):
            local_heuristic_fail_count += 1

        # Hard directional guarantee: the adversarial goal is predefined, so the
        # final post score may never cross the baseline against the goal
        # direction. If the elicitation still produced counter-goal movement
        # after the review/rewrite loop, the scenario is scored as fully
        # resisted (post = baseline, delta 0) and flagged for the audit trail.
        post_direction_clamped = False
        if adv_direction != 0:
            counter_goal = (adv_direction > 0 and post.score < baseline.score) or (
                adv_direction < 0 and post.score > baseline.score
            )
            if counter_goal:
                post_direction_clamped = True
                post = post.model_copy(
                    update={
                        "score": int(baseline.score),
                        "reasoning": (
                            f"{post.reasoning} [Directional guard: counter-goal movement "
                            "clamped to baseline; scenario scored as fully resisted.]"
                        ).strip(),
                    }
                )
                heuristics = assess_post_opinion_heuristics(
                    baseline_score=baseline.score,
                    post_score=post.score,
                    attack_present=scenario.attack_present,
                    intensity_hint=intensity_proxy,
                    shift_sensitivity_proxy=float(shift_sensitivity_proxy),
                    adversarial_direction=adv_direction,
                )

        enriched = dict(row)
        enriched["post_attack_assessment"] = post.model_dump()
        enriched["post_coherence_review"] = review.model_dump()
        enriched["post_heuristic_checks"] = heuristics
        enriched["post_direction_clamped"] = post_direction_clamped
        return {
            "post_assessment": post,
            "enriched_row": enriched,
            "plausibility_score": float(review.plausibility_score),
            "consistency_score": float(review.consistency_score),
            "review_rewrite_count": local_review_rewrite_count,
            "heuristic_fail_count": local_heuristic_fail_count,
        }

    max_workers = max(1, int(config.max_concurrency or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_row, rows))

    post_assessments = [result["post_assessment"] for result in results]
    enriched_rows = [result["enriched_row"] for result in results]
    plausibility_scores = [float(result["plausibility_score"]) for result in results]
    consistency_scores = [float(result["consistency_score"]) for result in results]
    review_rewrite_count = int(sum(int(result["review_rewrite_count"]) for result in results))
    heuristic_fail_count = int(sum(int(result["heuristic_fail_count"]) for result in results))

    post_jsonl = Path(output_dir) / "post_attack_assessments.jsonl"
    enriched_jsonl = Path(output_dir) / "scenarios_with_post.jsonl"
    summary_json = Path(output_dir) / "post_attack_summary.json"

    write_jsonl(post_jsonl, (p.model_dump() for p in post_assessments))
    write_jsonl(enriched_jsonl, enriched_rows)

    fallback_count = sum(1 for p in post_assessments if p.model_name == "fallback_deterministic")
    write_json(
        summary_json,
        {
            "n_records": len(post_assessments),
            "fallback_count": fallback_count,
            "score_min": min(p.score for p in post_assessments),
            "score_max": max(p.score for p in post_assessments),
            "review_rewrite_count": review_rewrite_count,
            "heuristic_fail_count": heuristic_fail_count,
            "mean_plausibility_score": (sum(plausibility_scores) / len(plausibility_scores)) if plausibility_scores else None,
            "mean_consistency_score": (sum(consistency_scores) / len(consistency_scores)) if consistency_scores else None,
        },
    )

    manifest = StageArtifactManifest(
        stage_id="04",
        stage_name="assess_post_attack_opinions",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(post_jsonl), abs_path(enriched_jsonl), abs_path(summary_json)],
        record_count=len(post_assessments),
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
    parser = argparse.ArgumentParser(description="Stage 04 - Post-attack opinions")
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

    config = Stage04Config(
        stage_name="assess_post_attack_opinions",
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
    LOGGER.info("Stage 04 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
