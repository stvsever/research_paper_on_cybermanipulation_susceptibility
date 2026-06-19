from __future__ import annotations

"""
Stage 02b - Baseline network-exposure opinions (BN), CLUSTER form.

Starts from the private baseline opinions produced by Stage 02 (one
`baseline_cluster_assessment` per scenario covering every leaf of the opinion
parent cluster) and re-elicits, in ONE agent call per scenario, the same
profile's opinion on every leaf after it sees incoming empirical exposure-peer
baseline context for that leaf.

Peer context is not selected by profile similarity. It is resolved through the
Stage 01b PolitiSky24 position assignment and the directed exposure graph:

    visible peer profile -> exposed target profile

For each (profile, opinion leaf) the stage finds the incoming empirical peers
who also produced a private baseline on the same leaf, computes peer counts and
exposure-weighted summaries over the scored incoming neighborhood, and sends a
bounded set of peer exemplars to the prompt. The private baseline cluster
assessment is preserved unchanged; network exposure is an additive phase.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.backend.utils.network_exposure import (
    build_baseline_network_context,
    exposure_assignments_from_rows,
    load_edge_index,
    load_exposure_network_package,
)
from src.backend.utils.schemas import (
    OpinionClusterAssessment,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)

LOGGER = logging.getLogger(__name__)


class Stage02bConfig(StageConfig):
    self_supervise_opinion_coherence: bool = False
    coherence_threshold: float = 0.72
    network_exposure_top_k: int = 8
    exposure_network_root: str | None = None


def _leaf_path_index(scenario: ScenarioRecord) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if scenario.opinion_cluster is not None:
        for leaf in scenario.opinion_cluster.leaves:
            out[str(leaf.leaf)] = str(leaf.path or leaf.leaf)
    return out


def _exposure_network_provenance(package: Any) -> Dict[str, Any]:
    return {
        "graph_id": package.graph_id,
        "graph_root": str(package.root),
        "edge_direction": package.manifest.get("edge_direction"),
        "edge_meaning": package.manifest.get("edge_meaning"),
        "interaction_weight_formula": package.manifest.get("interaction_weight_formula"),
    }


def run_stage(input_path: str, output_dir: str, config: Stage02bConfig) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 02b requires --openrouter-model")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"
    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    max_exemplars = max(0, int(config.network_exposure_top_k))

    scenario_rows = [dict(row) for row in read_jsonl(input_path)]
    if not scenario_rows:
        raise RuntimeError(f"Stage 02b received no rows: {input_path}")

    # Empirical exposure substrate + Stage 01b position assignments.
    assignments_by_profile = exposure_assignments_from_rows(scenario_rows)
    graph_ids = {str(a.graph_id) for a in assignments_by_profile.values()}
    if len(graph_ids) != 1:
        raise RuntimeError(f"Expected one exposure graph id, found {sorted(graph_ids)}")
    package = load_exposure_network_package(
        graph_root=config.exposure_network_root, graph_id=next(iter(graph_ids)), validate=True
    )
    assigned_positions = {a.position_id for a in assignments_by_profile.values()}
    edge_index = load_edge_index(package, target_positions=assigned_positions, source_positions=assigned_positions)

    # Parse rows; index each profile's private baseline score per leaf PATH.
    parsed: List[Dict[str, Any]] = []
    baseline_by_leaf: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in scenario_rows:
        cluster = row.get("baseline_cluster_assessment")
        if not isinstance(cluster, dict):
            raise RuntimeError("Stage 02b requires Stage 02 rows with baseline_cluster_assessment.")
        scenario = ScenarioRecord.model_validate(
            {k: v for k, v in row.items() if k not in {"baseline_cluster_assessment", "baseline_cluster_heuristics"}}
        )
        profile_id = scenario.profile.profile_id
        short_to_path = _leaf_path_index(scenario)
        leaf_scores = {str(ls.get("leaf")): ls for ls in cluster.get("leaf_scores", [])}
        leaves: List[Dict[str, Any]] = []
        for short, ls in leaf_scores.items():
            path = short_to_path.get(short, short)
            entry = {
                "score": int(ls.get("score", 0)),
                "confidence": float(ls.get("confidence", 0.5) or 0.5),
                "reasoning": str(ls.get("reasoning", "")),
            }
            baseline_by_leaf.setdefault(path, {})[profile_id] = entry
            leaves.append({"leaf": short, "path": path, "baseline": entry})
        parsed.append({"row": row, "scenario": scenario, "profile_id": profile_id, "leaves": leaves})

    def _build_leaf_payloads(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        profile_id = item["profile_id"]
        payloads: List[Dict[str, Any]] = []
        for leaf in item["leaves"]:
            path = leaf["path"]
            peers = {
                pid: payload
                for pid, payload in baseline_by_leaf.get(path, {}).items()
                if pid != profile_id
            }
            context = build_baseline_network_context(
                target_profile_id=profile_id,
                target_baseline_assessment=leaf["baseline"],
                peer_baseline_assessments_by_profile=peers,
                assignments_by_profile=assignments_by_profile,
                package=package,
                max_exemplars=max_exemplars,
                edge_index=edge_index,
            )
            context["opinion_leaf"] = path
            payloads.append(
                {
                    "leaf": leaf["leaf"],
                    "path": path,
                    "baseline_score": int(leaf["baseline"]["score"]),
                    "network_context": context,
                }
            )
        return payloads

    import threading

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
            thread_local.agent = factory.cluster_network_exposure_opinion_agent()
        return thread_local.agent

    def _process(item: Dict[str, Any]) -> Dict[str, Any]:
        scenario = item["scenario"]
        leaf_payloads = _build_leaf_payloads(item)
        cluster = scenario.opinion_cluster
        cluster_key = cluster.key if cluster is not None else scenario.opinion_leaf
        cluster_parent = cluster.parent_name if cluster is not None else ""
        try:
            assessment = _agent().assess(
                run_id=config.run_id,
                call_id=f"{scenario.scenario_id}_network_exposure_cluster",
                scenario_id=scenario.scenario_id,
                cluster_key=cluster_key,
                cluster_parent=cluster_parent,
                leaves=leaf_payloads,
                profile=scenario.profile,
            )
            bn_by_short = {ls.leaf: ls for ls in assessment.leaf_scores}
            model_name = assessment.model_name
        except Exception as exc:  # deterministic fallback: BN = private baseline (no network effect)
            LOGGER.warning("Stage 02b agent failed for %s; falling back to baseline: %s", scenario.scenario_id, exc)
            bn_by_short = {}
            model_name = "fallback_deterministic"
        return {"item": item, "leaf_payloads": leaf_payloads, "bn_by_short": bn_by_short, "model_name": model_name}

    workers = max(1, int(config.max_concurrency or 1))
    results: List[Dict[str, Any]] = [None] * len(parsed)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(_process, item): i for i, item in enumerate(parsed)}
        for f in as_completed(fut):
            results[fut[f]] = f.result()

    # Assemble outputs.
    enriched_rows: List[Dict[str, Any]] = []
    assessment_records: List[Dict[str, Any]] = []
    context_records: List[Dict[str, Any]] = []
    deltas: List[int] = []
    peer_counts: List[int] = []

    for res in results:
        item = res["item"]
        scenario = item["scenario"]
        profile_id = item["profile_id"]
        bn_by_short = res["bn_by_short"]
        model_name = res["model_name"]
        cluster_leaf_scores = []
        for lp in res["leaf_payloads"]:
            short, path = lp["leaf"], lp["path"]
            b = int(lp["baseline_score"])
            bn_ls = bn_by_short.get(short)
            bn_score = int(bn_ls.score) if bn_ls is not None else b
            bn_conf = float(bn_ls.confidence) if bn_ls is not None else 0.3
            bn_reason = str(bn_ls.reasoning) if bn_ls is not None else "fallback: held at private baseline"
            cluster_leaf_scores.append(
                {"leaf": short, "score": bn_score, "confidence": bn_conf, "reasoning": bn_reason}
            )
            deltas.append(bn_score - b)
            peer_counts.append(int(lp["network_context"].get("peer_count") or 0))
            assessment_records.append(
                {
                    "profile_id": profile_id,
                    "opinion_leaf": path,
                    "baseline_score": b,
                    "network_exposure_assessment": {
                        "scenario_id": scenario.scenario_id,
                        "phase": "network_exposure_baseline",
                        "opinion_leaf": path,
                        "score": bn_score,
                        "confidence": bn_conf,
                        "reasoning": bn_reason,
                        "model_name": model_name,
                    },
                }
            )
            context_records.append(
                {"profile_id": profile_id, "scenario_id": scenario.scenario_id,
                 "opinion_leaf": path, "network_context": lp["network_context"]}
            )
        network_cluster = OpinionClusterAssessment(
            scenario_id=scenario.scenario_id,
            phase="network_exposure_baseline",
            cluster_key=(scenario.opinion_cluster.key if scenario.opinion_cluster else scenario.opinion_leaf),
            leaf_scores=[
                {"leaf": ls["leaf"], "score": ls["score"], "confidence": ls["confidence"], "reasoning": ls["reasoning"]}
                for ls in cluster_leaf_scores
            ],
            model_name=model_name,
        ).model_dump()
        out_row = dict(item["row"])
        out_row["network_exposure_cluster_assessment"] = network_cluster
        enriched_rows.append(out_row)

    enriched_jsonl = Path(output_dir) / "scenarios_with_network_exposure.jsonl"
    assessments_jsonl = Path(output_dir) / "network_exposure_assessments.jsonl"
    contexts_jsonl = Path(output_dir) / "network_contexts.jsonl"
    summary_json = Path(output_dir) / "network_exposure_summary.json"

    write_jsonl(enriched_jsonl, enriched_rows)
    write_jsonl(assessments_jsonl, assessment_records)
    write_jsonl(contexts_jsonl, context_records)

    fallback = sum(1 for r in results if r["model_name"] == "fallback_deterministic")
    write_json(
        summary_json,
        {
            "n_scenarios": len(enriched_rows),
            "n_leaf_measurements": len(assessment_records),
            "calls_made": len(results),
            "fallback_scenarios": fallback,
            "max_exemplars": max_exemplars,
            "mean_network_exposure_delta_score": float(statistics.mean(deltas)) if deltas else None,
            "mean_abs_network_exposure_delta_score": float(statistics.mean(abs(d) for d in deltas)) if deltas else None,
            "mean_scored_peer_count": float(statistics.mean(peer_counts)) if peer_counts else None,
            "exposure_network": _exposure_network_provenance(package),
        },
    )

    manifest = StageArtifactManifest(
        stage_id="02b",
        stage_name="assess_network_exposure_opinions",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(enriched_jsonl), abs_path(assessments_jsonl), abs_path(contexts_jsonl), abs_path(summary_json)],
        record_count=len(enriched_rows),
        metadata={
            "n_leaf_measurements": len(assessment_records),
            "calls_made": len(results),
            "fallback_scenarios": fallback,
            "openrouter_model": config.openrouter_model,
            "exposure_network": _exposure_network_provenance(package),
        },
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02b - Cluster network-exposure baseline opinions")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openrouter-model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=1)
    parser.add_argument("--self-supervise-opinion-coherence", action=argparse.BooleanOptionalAction, default=False)
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
    LOGGER.info("Stage 02b completed: %s scenarios", manifest.record_count)


if __name__ == "__main__":
    main()
