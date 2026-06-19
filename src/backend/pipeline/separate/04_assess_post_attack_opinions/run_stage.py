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
from src.backend.utils.scenario.scenario_realism import (
    assess_post_opinion_heuristics,
    profile_context_snapshot,
)
from src.backend.utils.schemas import (
    ClusterLeafScore,
    OpinionAssessment,
    OpinionClusterAssessment,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)

LOGGER = logging.getLogger(__name__)


class Stage04Config(StageConfig):
    self_supervise_opinion_coherence: bool = True
    coherence_threshold: float = 0.72


def _clamp_leaf_to_goal(post_score: int, baseline_score: int, adversarial_direction: int) -> int:
    """Hard directional guarantee per leaf: the post score may never cross the
    baseline against the adversary's predefined goal direction. Counter-goal
    movement is scored as fully resisted (post = baseline)."""
    if adversarial_direction > 0 and post_score < baseline_score:
        return baseline_score
    if adversarial_direction < 0 and post_score > baseline_score:
        return baseline_score
    return post_score


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


def _run_cluster_post(
    rows: List[Dict[str, object]],
    input_path: str,
    output_dir: str,
    config: Stage04Config,
    prompts_dir: Path,
    raw_dir,
) -> StageArtifactManifest:
    """Cluster-batched post-attack assessment: one agent call per scenario
    estimates the post-exposure score for every leaf of the issue-domain cluster
    inside each leaf's admissible (baseline -> goal) interval, then applies a
    hard per-leaf directional clamp (integrated-scenario design)."""
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
            thread_local.agent = factory.cluster_post_attack_opinion_agent()
        return thread_local.agent

    def _process(row: Dict[str, object]) -> Dict[str, object]:
        scenario = ScenarioRecord.model_validate(
            {
                k: v
                for k, v in row.items()
                if k not in {
                    "baseline_assessment", "attack_exposure", "attack_vector_spec",
                    "baseline_coherence_review", "baseline_heuristic_checks",
                    "baseline_cluster_assessment", "baseline_cluster_heuristics",
                }
            }
        )
        cluster = scenario.opinion_cluster
        baseline_cluster = OpinionClusterAssessment.model_validate(row["baseline_cluster_assessment"])
        baseline_by_leaf = {ls.leaf: int(ls.score) for ls in baseline_cluster.leaf_scores}
        spec = row.get("attack_vector_spec") if isinstance(row, dict) else None
        spec = spec if isinstance(spec, dict) else {}
        intensity_proxy = float(spec.get("intensity_proxy", 0.5) or 0.0)
        shift_proxy = float(scenario.profile.continuous_attributes.get("heuristic_shift_sensitivity_proxy", 0.5))

        direction_by_leaf = {lf.leaf: int(lf.adversarial_direction) for lf in cluster.leaves}
        leaf_inputs = [
            {
                "leaf": lf.leaf,
                "path": lf.path,
                "baseline_score": baseline_by_leaf.get(lf.leaf, 0),
                "adversarial_direction": int(lf.adversarial_direction),
            }
            for lf in cluster.leaves
        ]

        agent = _agent()
        rewrite_count = 0
        try:
            post = agent.assess(
                run_id=config.run_id,
                call_id=f"{scenario.scenario_id}_post_cluster",
                scenario_id=scenario.scenario_id,
                cluster_key=cluster.key,
                cluster_parent=cluster.parent_name,
                leaves=leaf_inputs,
                profile=scenario.profile,
                attack_present=scenario.attack_present,
                attack_leaf=scenario.attack_leaf,
                attack_vector_spec=spec,
            )
        except Exception as exc:
            LOGGER.warning("Cluster post failed for %s, deterministic fallback: %s", scenario.scenario_id, exc)
            post = _fallback_cluster_post(scenario, baseline_by_leaf, direction_by_leaf)

        def _heur(p: OpinionClusterAssessment):
            checks = []
            for ls in p.leaf_scores:
                b = baseline_by_leaf.get(ls.leaf, 0)
                checks.append(
                    assess_post_opinion_heuristics(
                        baseline_score=b,
                        post_score=int(ls.score),
                        attack_present=scenario.attack_present,
                        intensity_hint=intensity_proxy,
                        shift_sensitivity_proxy=shift_proxy,
                        adversarial_direction=direction_by_leaf.get(ls.leaf, 0),
                    )
                )
            n = len(checks)
            n_pass = sum(1 for c in checks if bool(c["checks"].get("overall_pass", False)))
            return {"n_leaves": n, "n_pass": n_pass, "pass_rate": (n_pass / n) if n else 0.0,
                    "overall_pass": bool(n and n_pass >= max(1, int(0.7 * n)))}

        heur = _heur(post)
        # Gated on the coherence-control switch so it can be turned off to keep the
        # run at exactly one post-attack call per scenario.
        if (
            config.self_supervise_opinion_coherence
            and post.model_name != "fallback_deterministic"
            and not heur["overall_pass"]
        ):
            rewrite_count = 1
            try:
                post = agent.assess(
                    run_id=config.run_id,
                    call_id=f"{scenario.scenario_id}_post_cluster_rewrite",
                    scenario_id=scenario.scenario_id,
                    cluster_key=cluster.key,
                    cluster_parent=cluster.parent_name,
                    leaves=leaf_inputs,
                    profile=scenario.profile,
                    attack_present=scenario.attack_present,
                    attack_leaf=scenario.attack_leaf,
                    attack_vector_spec=spec,
                    review_feedback=(
                        "Keep every leaf inside its admissible [baseline -> goal] interval (never cross the "
                        "baseline against the adversarial_direction), use high-resolution non-coarse integers, "
                        "and let movement vary across leaves (full resistance = post equal to baseline)."
                    ),
                )
                heur = _heur(post)
            except Exception as exc:
                LOGGER.warning("Cluster post rewrite failed for %s: %s", scenario.scenario_id, exc)

        post = _ensure_cluster_post_coverage(scenario, post, baseline_by_leaf, direction_by_leaf)

        # Hard per-leaf directional clamp: counter-goal movement -> fully resisted.
        clamped = 0
        new_scores = []
        for ls in post.leaf_scores:
            b = baseline_by_leaf.get(ls.leaf, 0)
            d = direction_by_leaf.get(ls.leaf, 0)
            clamped_score = _clamp_leaf_to_goal(int(ls.score), b, d)
            if clamped_score != int(ls.score):
                clamped += 1
            new_scores.append(ls.model_copy(update={"score": clamped_score, "adversarial_direction": d}))
        post = post.model_copy(update={"leaf_scores": new_scores})

        enriched = dict(row)
        enriched["post_cluster_assessment"] = post.model_dump()
        enriched["post_cluster_heuristics"] = heur
        enriched["post_cluster_clamped_leaves"] = clamped
        return {
            "post": post,
            "enriched_row": enriched,
            "rewrite_count": rewrite_count,
            "fallback": post.model_name == "fallback_deterministic",
            "clamped": clamped,
            "n_leaf_scores": len(post.leaf_scores),
        }

    max_workers = max(1, int(config.max_concurrency or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process, rows))

    posts = [r["post"] for r in results]
    enriched_rows = [r["enriched_row"] for r in results]
    all_scores = [ls.score for p in posts for ls in p.leaf_scores]

    post_jsonl = Path(output_dir) / "post_attack_assessments.jsonl"
    enriched_jsonl = Path(output_dir) / "scenarios_with_post.jsonl"
    summary_json = Path(output_dir) / "post_attack_summary.json"

    write_jsonl(post_jsonl, (p.model_dump() for p in posts))
    write_jsonl(enriched_jsonl, enriched_rows)

    fallback_count = sum(1 for r in results if r["fallback"])
    write_json(
        summary_json,
        {
            "assessment_mode": "cluster_batched",
            "n_scenarios": len(posts),
            "n_leaf_scores": len(all_scores),
            "fallback_count": fallback_count,
            "score_min": min(all_scores) if all_scores else None,
            "score_max": max(all_scores) if all_scores else None,
            "review_rewrite_count": int(sum(int(r["rewrite_count"]) for r in results)),
            "direction_clamped_leaves": int(sum(int(r["clamped"]) for r in results)),
        },
    )

    manifest = StageArtifactManifest(
        stage_id="04",
        stage_name="assess_post_attack_opinions",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(post_jsonl), abs_path(enriched_jsonl), abs_path(summary_json)],
        record_count=len(posts),
        metadata={
            "assessment_mode": "cluster_batched",
            "fallback_count": fallback_count,
            "openrouter_model": config.openrouter_model,
            "n_leaf_scores": len(all_scores),
            "direction_clamped_leaves": int(sum(int(r["clamped"]) for r in results)),
        },
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def _fallback_cluster_post(scenario, baseline_by_leaf, direction_by_leaf) -> OpinionClusterAssessment:
    leaf_scores = []
    for lf in scenario.opinion_cluster.leaves:
        b = baseline_by_leaf.get(lf.leaf, 0)
        digest = hashlib.md5(f"{scenario.scenario_id}:{lf.leaf}".encode("utf-8"), usedforsecurity=False).hexdigest()
        magnitude = int(digest[:4], 16) % 60  # 0..59 movement toward goal
        d = direction_by_leaf.get(lf.leaf, 0)
        score = max(-1000, min(1000, b + d * magnitude))
        leaf_scores.append(ClusterLeafScore(leaf=lf.leaf, score=score, confidence=0.3, reasoning="Deterministic fallback.", adversarial_direction=d))
    return OpinionClusterAssessment(
        scenario_id=scenario.scenario_id, phase="post_attack",
        cluster_key=scenario.opinion_cluster.key, leaf_scores=leaf_scores,
        model_name="fallback_deterministic",
    )


def _ensure_cluster_post_coverage(scenario, post: OpinionClusterAssessment, baseline_by_leaf, direction_by_leaf) -> OpinionClusterAssessment:
    present = {ls.leaf for ls in post.leaf_scores}
    if all(lf.leaf in present for lf in scenario.opinion_cluster.leaves):
        return post
    fb = _fallback_cluster_post(scenario, baseline_by_leaf, direction_by_leaf)
    fb_by_leaf = {ls.leaf: ls for ls in fb.leaf_scores}
    filled = list(post.leaf_scores) + [fb_by_leaf[lf.leaf] for lf in scenario.opinion_cluster.leaves if lf.leaf not in present]
    return post.model_copy(update={"leaf_scores": filled})


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

    if not any(isinstance(r, dict) and "baseline_cluster_assessment" in r for r in rows):
        raise RuntimeError(
            "Stage 04 requires integrated cluster scenarios with baseline_cluster_assessment; "
            "the legacy per-leaf post-attack path was retired."
        )
    LOGGER.info("Stage 04: cluster-batched post-attack for %d scenarios", len(rows))
    return _run_cluster_post(rows, input_path, output_dir, config, prompts_dir, raw_dir)


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
