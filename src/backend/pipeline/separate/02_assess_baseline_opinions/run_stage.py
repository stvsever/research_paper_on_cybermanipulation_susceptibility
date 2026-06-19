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
from src.backend.utils.scenario.scenario_realism import assess_baseline_opinion_heuristics, profile_context_snapshot
from src.backend.utils.schemas import (
    OpinionAssessment,
    OpinionClusterAssessment,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)

LOGGER = logging.getLogger(__name__)


class Stage02Config(StageConfig):
    self_supervise_opinion_coherence: bool = True
    coherence_threshold: float = 0.72


def _cluster_baseline_heuristics(assessment: OpinionClusterAssessment) -> Dict[str, object]:
    """Aggregate per-leaf baseline heuristic checks for a cluster assessment."""
    per_leaf = [
        assess_baseline_opinion_heuristics(score=ls.score, confidence=ls.confidence)
        for ls in assessment.leaf_scores
    ]
    n = len(per_leaf)
    n_pass = sum(1 for h in per_leaf if bool(h["checks"].get("overall_pass", False)))
    distinct = len({ls.score for ls in assessment.leaf_scores})
    return {
        "n_leaves": n,
        "n_pass": n_pass,
        "pass_rate": (n_pass / n) if n else 0.0,
        "distinct_scores": distinct,
        "overall_pass": bool(n and n_pass >= max(1, int(0.7 * n)) and distinct > 1),
    }


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


def _run_cluster_baseline(
    scenarios: List[ScenarioRecord],
    input_path: str,
    output_dir: str,
    config: Stage02Config,
    prompts_dir: Path,
    raw_dir,
) -> StageArtifactManifest:
    """Cluster-batched baseline: one agent call per scenario returns a score for
    every leaf of the issue-domain cluster (integrated-scenario design)."""
    thread_local = threading.local()

    def _agent():
        if not hasattr(thread_local, "agent"):
            factory = AgentFactory(
                prompts_dir=prompts_dir,
                openrouter_api_key=env_get_required("OPENROUTER_API_KEY"),
                openrouter_model=config.openrouter_model,
                max_repair_iter=config.max_repair_iter,
                temperature=config.temperature,
                timeout_sec=config.timeout_sec,
                save_raw_dir=raw_dir,
            )
            thread_local.agent = factory.cluster_baseline_opinion_agent()
        return thread_local.agent

    def _process(scenario: ScenarioRecord) -> Dict[str, object]:
        agent = _agent()
        cluster = scenario.opinion_cluster
        leaves = [{"leaf": lf.leaf, "path": lf.path} for lf in cluster.leaves]
        rewrite_count = 0
        try:
            assessment = agent.assess(
                run_id=config.run_id,
                call_id=f"{scenario.scenario_id}_baseline_cluster",
                scenario_id=scenario.scenario_id,
                cluster_key=cluster.key,
                cluster_parent=cluster.parent_name,
                leaves=leaves,
                profile=scenario.profile,
            )
        except Exception as exc:
            LOGGER.warning("Cluster baseline failed for %s, deterministic fallback: %s", scenario.scenario_id, exc)
            assessment = _fallback_cluster_assessment(scenario, phase="baseline")

        heur = _cluster_baseline_heuristics(assessment)
        # One rewrite pass if the cluster degenerated (coarse / collapsed scores).
        # Gated on the coherence-control switch so it can be turned off to keep the
        # run at exactly one baseline call per scenario.
        if (
            config.self_supervise_opinion_coherence
            and assessment.model_name != "fallback_deterministic"
            and not heur["overall_pass"]
        ):
            rewrite_count = 1
            try:
                assessment = agent.assess(
                    run_id=config.run_id,
                    call_id=f"{scenario.scenario_id}_baseline_cluster_rewrite",
                    scenario_id=scenario.scenario_id,
                    cluster_key=cluster.key,
                    cluster_parent=cluster.parent_name,
                    leaves=leaves,
                    profile=scenario.profile,
                    review_feedback=(
                        "Use high-resolution non-coarse integer scores (avoid multiples of 50) and ensure the "
                        "leaves do not all share one value; reflect item-by-item differences for this person."
                    ),
                )
                heur = _cluster_baseline_heuristics(assessment)
            except Exception as exc:
                LOGGER.warning("Cluster baseline rewrite failed for %s: %s", scenario.scenario_id, exc)

        # Validate coverage: every requested leaf must be scored; fill any gaps deterministically.
        assessment = _ensure_cluster_coverage(scenario, assessment, phase="baseline")

        row = scenario.model_dump()
        row["baseline_cluster_assessment"] = assessment.model_dump()
        row["baseline_cluster_heuristics"] = heur
        return {
            "assessment": assessment,
            "enriched_row": row,
            "rewrite_count": rewrite_count,
            "fallback": assessment.model_name == "fallback_deterministic",
            "n_leaf_scores": len(assessment.leaf_scores),
        }

    max_workers = max(1, int(config.max_concurrency or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process, scenarios))

    assessments = [r["assessment"] for r in results]
    enriched_rows = [r["enriched_row"] for r in results]
    all_scores = [ls.score for a in assessments for ls in a.leaf_scores]

    baseline_jsonl = Path(output_dir) / "baseline_assessments.jsonl"
    enriched_jsonl = Path(output_dir) / "scenarios_with_baseline.jsonl"
    summary_json = Path(output_dir) / "baseline_summary.json"

    write_jsonl(baseline_jsonl, (a.model_dump() for a in assessments))
    write_jsonl(enriched_jsonl, enriched_rows)

    fallback_count = sum(1 for r in results if r["fallback"])
    write_json(
        summary_json,
        {
            "assessment_mode": "cluster_batched",
            "n_scenarios": len(assessments),
            "n_leaf_scores": len(all_scores),
            "fallback_count": fallback_count,
            "score_min": min(all_scores) if all_scores else None,
            "score_max": max(all_scores) if all_scores else None,
            "review_rewrite_count": int(sum(int(r["rewrite_count"]) for r in results)),
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
            "assessment_mode": "cluster_batched",
            "fallback_count": fallback_count,
            "openrouter_model": config.openrouter_model,
            "n_leaf_scores": len(all_scores),
        },
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def _fallback_cluster_assessment(scenario: ScenarioRecord, phase: str) -> OpinionClusterAssessment:
    from src.backend.utils.schemas import ClusterLeafScore

    leaf_scores = []
    for lf in scenario.opinion_cluster.leaves:
        seed_text = f"{scenario.scenario_id}:{lf.leaf}:{scenario.profile.profile_id}"
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        score = int((int(digest[:8], 16) % 2001) - 1000)
        leaf_scores.append(
            ClusterLeafScore(leaf=lf.leaf, score=score, confidence=0.3, reasoning="Deterministic fallback.")
        )
    return OpinionClusterAssessment(
        scenario_id=scenario.scenario_id,
        phase=phase,
        cluster_key=scenario.opinion_cluster.key,
        leaf_scores=leaf_scores,
        model_name="fallback_deterministic",
    )


def _ensure_cluster_coverage(
    scenario: ScenarioRecord, assessment: OpinionClusterAssessment, phase: str
) -> OpinionClusterAssessment:
    """Guarantee every cluster leaf has a score; fill omissions deterministically."""
    from src.backend.utils.schemas import ClusterLeafScore

    by_leaf = {ls.leaf: ls for ls in assessment.leaf_scores}
    filled = list(assessment.leaf_scores)
    fallback = _fallback_cluster_assessment(scenario, phase=phase)
    fb_by_leaf = {ls.leaf: ls for ls in fallback.leaf_scores}
    present = set(by_leaf)
    for lf in scenario.opinion_cluster.leaves:
        if lf.leaf not in present:
            filled.append(fb_by_leaf[lf.leaf])
    if len(filled) != len(assessment.leaf_scores):
        return assessment.model_copy(update={"leaf_scores": filled})
    return assessment


def run_stage(input_path: str, output_dir: str, config: Stage02Config) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 02 requires --openrouter-model")

    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"

    scenario_rows = read_jsonl(input_path)
    scenarios = [ScenarioRecord.model_validate(row) for row in scenario_rows]

    if not any(s.opinion_cluster is not None for s in scenarios):
        raise RuntimeError(
            "Stage 02 requires integrated opinion-cluster scenarios; the legacy "
            "per-leaf baseline path was retired."
        )
    LOGGER.info("Stage 02: cluster-batched baseline for %d scenarios", len(scenarios))
    return _run_cluster_baseline(scenarios, input_path, output_dir, config, prompts_dir, raw_dir)


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
