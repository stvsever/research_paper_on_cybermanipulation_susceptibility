from __future__ import annotations

"""
Stage 04b - Post-attack network-exposure opinions (PN), CLUSTER form.

Starts from the private post-attack opinions produced by Stage 04 (one
`post_cluster_assessment` per scenario) and re-elicits, in ONE agent call per
scenario, the profile's post-attack opinion on every leaf after it sees the
incoming empirical exposure peers' post-attack reactions to the same leaf.

Peer context is resolved through the Stage 01b PolitiSky24 position assignment
and the directed exposure graph (visible peer -> exposed receiver). Because each
integrated scenario carries its own near-unique DISARM Plan/Prepare/Execute
attack triplet, peers are pooled over attacks: the incoming neighbors who
produced a private post-attack score on the same opinion leaf. The full DISARM
triplet for the target is passed to the agent so it reasons about the operation
as a campaign, not a single attack label. The private post-attack cluster
assessment is preserved unchanged; this is an additive measurement.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import statistics
import sys
import threading
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
    build_post_attack_network_context,
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


class Stage04bConfig(StageConfig):
    self_supervise_opinion_coherence: bool = False
    coherence_threshold: float = 0.72
    network_exposure_top_k: int = 8
    exposure_network_root: str | None = None


def _leaf_meta_index(scenario: ScenarioRecord) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if scenario.opinion_cluster is not None:
        for leaf in scenario.opinion_cluster.leaves:
            out[str(leaf.leaf)] = {"path": str(leaf.path or leaf.leaf), "adversarial_direction": int(leaf.adversarial_direction)}
    return out


def _exposure_network_provenance(package: Any) -> Dict[str, Any]:
    return {
        "graph_id": package.graph_id,
        "graph_root": str(package.root),
        "edge_direction": package.manifest.get("edge_direction"),
        "edge_meaning": package.manifest.get("edge_meaning"),
        "interaction_weight_formula": package.manifest.get("interaction_weight_formula"),
    }


def run_stage(input_path: str, output_dir: str, config: Stage04bConfig) -> StageArtifactManifest:
    if config.openrouter_model is None:
        raise RuntimeError("Stage 04b requires --openrouter-model")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[5]
    prompts_dir = project_root / "src" / "backend" / "agentic_framework" / "prompts"
    raw_dir = config.raw_llm_dir if config.save_raw_llm else None
    max_exemplars = max(0, int(config.network_exposure_top_k))

    scenario_rows = [dict(row) for row in read_jsonl(input_path)]
    if not scenario_rows:
        raise RuntimeError(f"Stage 04b received no rows: {input_path}")

    assignments_by_profile = exposure_assignments_from_rows(scenario_rows)
    graph_ids = {str(a.graph_id) for a in assignments_by_profile.values()}
    if len(graph_ids) != 1:
        raise RuntimeError(f"Expected one exposure graph id, found {sorted(graph_ids)}")
    package = load_exposure_network_package(
        graph_root=config.exposure_network_root, graph_id=next(iter(graph_ids)), validate=True
    )
    assigned_positions = {a.position_id for a in assignments_by_profile.values()}
    edge_index = load_edge_index(package, target_positions=assigned_positions, source_positions=assigned_positions)

    # Parse rows; index each profile's private baseline + post score per leaf PATH.
    parsed: List[Dict[str, Any]] = []
    post_by_leaf: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in scenario_rows:
        base_cluster = row.get("baseline_cluster_assessment")
        post_cluster = row.get("post_cluster_assessment")
        if not isinstance(base_cluster, dict) or not isinstance(post_cluster, dict):
            raise RuntimeError("Stage 04b requires rows with baseline_cluster_assessment and post_cluster_assessment.")
        scenario = ScenarioRecord.model_validate(
            {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "baseline_cluster_assessment", "baseline_cluster_heuristics",
                    "post_cluster_assessment", "post_cluster_heuristics", "post_cluster_clamped_leaves",
                    "attack_vector_spec",
                }
            }
        )
        profile_id = scenario.profile.profile_id
        leaf_meta = _leaf_meta_index(scenario)
        base_scores = {str(ls.get("leaf")): ls for ls in base_cluster.get("leaf_scores", [])}
        post_scores = {str(ls.get("leaf")): ls for ls in post_cluster.get("leaf_scores", [])}
        spec = row.get("attack_vector_spec") if isinstance(row.get("attack_vector_spec"), dict) else {}
        leaves: List[Dict[str, Any]] = []
        for short, post_ls in post_scores.items():
            base_ls = base_scores.get(short, {})
            meta = leaf_meta.get(short, {"path": short, "adversarial_direction": 0})
            path = meta["path"]
            entry = {
                "baseline_score": int(base_ls.get("score", 0)),
                "score": int(post_ls.get("score", 0)),  # post score (key name 'score' for context builder)
                "confidence": float(post_ls.get("confidence", 0.5) or 0.5),
                "reasoning": str(post_ls.get("reasoning", "")),
            }
            post_by_leaf.setdefault(path, {})[profile_id] = entry
            leaves.append(
                {"leaf": short, "path": path, "adversarial_direction": int(meta["adversarial_direction"]), "post": entry}
            )
        parsed.append(
            {"row": row, "scenario": scenario, "profile_id": profile_id, "leaves": leaves, "attack_vector_spec": spec}
        )

    def _build_leaf_payloads(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        profile_id = item["profile_id"]
        payloads: List[Dict[str, Any]] = []
        for leaf in item["leaves"]:
            path = leaf["path"]
            peers = {pid: p for pid, p in post_by_leaf.get(path, {}).items() if pid != profile_id}
            context = build_post_attack_network_context(
                target_profile_id=profile_id,
                target_private_post_assessment=leaf["post"],
                same_condition_peer_post_assessments_by_profile=peers,
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
                    "baseline_score": int(leaf["post"]["baseline_score"]),
                    "private_post_score": int(leaf["post"]["score"]),
                    "adversarial_direction": int(leaf["adversarial_direction"]),
                    "network_context": context,
                }
            )
        return payloads

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
            thread_local.agent = factory.cluster_post_attack_network_exposure_opinion_agent()
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
                call_id=f"{scenario.scenario_id}_post_attack_network_exposure_cluster",
                scenario_id=scenario.scenario_id,
                cluster_key=cluster_key,
                cluster_parent=cluster_parent,
                leaves=leaf_payloads,
                profile=scenario.profile,
                attack_present=bool(scenario.attack_present),
                attack_leaf=scenario.attack_leaf,
                attack_vector_spec=item["attack_vector_spec"],
            )
            pn_by_short = {ls.leaf: ls for ls in assessment.leaf_scores}
            model_name = assessment.model_name
        except Exception as exc:  # fallback: PN = private post (no network effect)
            LOGGER.warning("Stage 04b agent failed for %s; falling back to private post: %s", scenario.scenario_id, exc)
            pn_by_short = {}
            model_name = "fallback_deterministic"
        return {"item": item, "leaf_payloads": leaf_payloads, "pn_by_short": pn_by_short, "model_name": model_name}

    workers = max(1, int(config.max_concurrency or 1))
    results: List[Dict[str, Any]] = [None] * len(parsed)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(_process, item): i for i, item in enumerate(parsed)}
        for f in as_completed(fut):
            results[fut[f]] = f.result()

    enriched_rows: List[Dict[str, Any]] = []
    assessment_records: List[Dict[str, Any]] = []
    context_records: List[Dict[str, Any]] = []
    pn_increments: List[int] = []
    effectivity: List[int] = []

    for res in results:
        item = res["item"]
        scenario = item["scenario"]
        profile_id = item["profile_id"]
        pn_by_short = res["pn_by_short"]
        model_name = res["model_name"]
        cluster_leaf_scores = []
        for lp in res["leaf_payloads"]:
            short, path = lp["leaf"], lp["path"]
            p = int(lp["private_post_score"])
            d = int(lp["adversarial_direction"])
            pn_ls = pn_by_short.get(short)
            pn_score = int(pn_ls.score) if pn_ls is not None else p
            pn_conf = float(pn_ls.confidence) if pn_ls is not None else 0.3
            pn_reason = str(pn_ls.reasoning) if pn_ls is not None else "fallback: held at private post-attack"
            cluster_leaf_scores.append({"leaf": short, "score": pn_score, "confidence": pn_conf, "reasoning": pn_reason})
            pn_increments.append(pn_score - p)
            if d:
                effectivity.append((pn_score - p) * d)
            assessment_records.append(
                {
                    "profile_id": profile_id,
                    "opinion_leaf": path,
                    "baseline_score": int(lp["baseline_score"]),
                    "private_post_score": p,
                    "adversarial_direction": d,
                    "post_attack_network_exposure_assessment": {
                        "scenario_id": scenario.scenario_id,
                        "phase": "post_attack_network_exposure",
                        "opinion_leaf": path,
                        "score": pn_score,
                        "confidence": pn_conf,
                        "reasoning": pn_reason,
                        "model_name": model_name,
                    },
                }
            )
            context_records.append(
                {"profile_id": profile_id, "scenario_id": scenario.scenario_id,
                 "opinion_leaf": path, "post_attack_network_context": lp["network_context"]}
            )
        network_cluster = OpinionClusterAssessment(
            scenario_id=scenario.scenario_id,
            phase="post_attack_network_exposure",
            cluster_key=(scenario.opinion_cluster.key if scenario.opinion_cluster else scenario.opinion_leaf),
            leaf_scores=[
                {"leaf": ls["leaf"], "score": ls["score"], "confidence": ls["confidence"], "reasoning": ls["reasoning"]}
                for ls in cluster_leaf_scores
            ],
            model_name=model_name,
        ).model_dump()
        out_row = dict(item["row"])
        out_row["post_attack_network_exposure_cluster_assessment"] = network_cluster
        enriched_rows.append(out_row)

    enriched_jsonl = Path(output_dir) / "scenarios_with_post_attack_network_exposure.jsonl"
    assessments_jsonl = Path(output_dir) / "post_attack_network_exposure_assessments.jsonl"
    contexts_jsonl = Path(output_dir) / "post_attack_network_contexts.jsonl"
    summary_json = Path(output_dir) / "post_attack_network_exposure_summary.json"

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
            "mean_pn_increment": float(statistics.mean(pn_increments)) if pn_increments else None,
            "mean_pn_increment_effectivity": float(statistics.mean(effectivity)) if effectivity else None,
            "exposure_network": _exposure_network_provenance(package),
        },
    )

    manifest = StageArtifactManifest(
        stage_id="04b",
        stage_name="assess_post_attack_network_exposure_opinions",
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
    parser = argparse.ArgumentParser(description="Stage 04b - Cluster post-attack network-exposure opinions")
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
    config = Stage04bConfig(
        stage_name="assess_post_attack_network_exposure_opinions",
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
    LOGGER.info("Stage 04b completed: %s scenarios", manifest.record_count)


if __name__ == "__main__":
    main()
