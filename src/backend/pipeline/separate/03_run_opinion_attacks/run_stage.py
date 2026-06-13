from __future__ import annotations

"""
Stage 03 - Specify attack vectors (deterministic, zero LLM calls).

Methodological position (run_1 final design)
---------------------------------------------
Earlier pipeline generations materialised one synthetic exposure artifact per
scenario (a single post, DM, or transcript) and then showed that artifact to
the post-exposure agent. That concretisation step is methodologically wrong
for most of the ATTACK ontology: campaign-style vectors (astroturf waves,
multi-persona operations, repost-bot amplification, pervasive doubt
engineering) are not reducible to one message a respondent reads once, so a
single generated artifact systematically under-represents the vector and adds
an uncontrolled generation-quality confound between the attack taxonomy and
the outcome.

Stage 03 therefore no longer generates content. It deterministically compiles
an ATTACK VECTOR SPECIFICATION per scenario from the ontology and the
scenario record:

  - the attack leaf and its structural metadata (mechanism, primary system,
    platform hint, complexity tier, temporal horizon, epistemic target,
    personalization / orchestration requirements),
  - the direction-aware persuasion frame (goal, baseline-vs-goal alignment,
    motivational lever family, emotional register, issue frame) built by
    build_attack_context without any profile-conditioned amplification
    signal,
  - a complexity-tier-derived intensity proxy used only as the structural
    envelope for the post-stage heuristic shift band.

Stage 04 then asks the simulation agent to estimate the effectiveness of the
SPECIFIED VECTOR on the given profile configuration for the given opinion
item, integrating over the realistic contact surface of such an operation,
instead of reacting to one hand-written artifact.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.io import abs_path, ensure_dir, read_json, read_jsonl, stage_manifest_path, write_json, write_jsonl
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.ontology_utils import load_adversarial_directions_from_opinion
from src.backend.utils.compatibility_rules import load_attack_metadata_index
from src.backend.utils.scenario_realism import build_attack_context
from src.backend.utils.schemas import (
    OpinionAssessment,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)

LOGGER = logging.getLogger(__name__)


class Stage03Config(StageConfig):
    self_supervise_attack_realism: bool = True
    realism_threshold: float = 0.72


# Keys copied from build_attack_context output into the downstream spec.
# Profile-snapshot keys and the profile-conditioned recommended_shift_band are
# deliberately excluded: per-profile susceptibility is the estimand and must
# not be pre-encoded into the vector specification shown to the agent.
_SPEC_CONTEXT_KEYS = (
    "opinion_domain",
    "opinion_leaf_label",
    "attack_leaf_label",
    "adversarial_direction",
    "adversarial_direction_label",
    "baseline_vs_goal",
    "persuasion_goal",
    "motivational_lever",
    "emotional_register",
    "issue_frame",
    "attack_mechanism",
    "attack_primary_system",
    "attack_platform_hint",
    "attack_complexity_tier",
    "attack_temporal_horizon",
    "attack_epistemic_target",
    "attack_requires_personalization",
    "attack_agent_orchestration_required",
)


def _tier_intensity_proxy(complexity_tier: str) -> float:
    """Structural envelope proxy per complexity tier (NOT profile-conditioned).

    Higher tiers describe operations with a larger realistic contact surface
    (campaigns, synthetic artifacts, orchestrated multi-agent operations), so
    they support a wider plausible movement envelope in the heuristic check.
    """
    tier = (complexity_tier or "").upper()
    if tier.startswith("T4"):
        return 0.70
    if tier.startswith("T3"):
        return 0.60
    if tier.startswith("T2"):
        return 0.50
    if tier.startswith("T1"):
        return 0.35
    return 0.50


def run_stage(input_path: str, output_dir: str, config: Stage03Config) -> StageArtifactManifest:
    ensure_dir(output_dir)

    adversarial_directions: Dict[str, int] = {}
    attack_metadata_index: Dict[str, object] = {}
    if config.ontology_root:
        opinion_path = Path(config.ontology_root) / "OPINION" / "opinion.json"
        attack_path = Path(config.ontology_root) / "ATTACK" / "attack.json"
        if opinion_path.exists():
            opinion_tree = read_json(str(opinion_path))
            adversarial_directions, _ = load_adversarial_directions_from_opinion(opinion_tree)
            LOGGER.info("Stage 03: loaded %d adversarial directions", len(adversarial_directions))
        if attack_path.exists():
            attack_tree = read_json(str(attack_path))
            attack_metadata_index = load_attack_metadata_index(attack_tree)
            LOGGER.info("Stage 03: loaded %d attack metadata entries", len(attack_metadata_index))
    if not attack_metadata_index:
        LOGGER.warning(
            "Stage 03: no attack metadata index available (ontology_root=%s); "
            "specs will carry empty structural metadata",
            config.ontology_root,
        )

    rows = read_jsonl(input_path)
    enriched_rows: List[Dict[str, Any]] = []
    spec_records: List[Dict[str, Any]] = []
    n_attack = 0
    n_direction_from_metadata = 0

    for row in rows:
        scenario = ScenarioRecord.model_validate(
            {k: v for k, v in row.items() if k not in {
                "baseline_assessment", "baseline_coherence_review", "baseline_heuristic_checks",
            }}
        )
        baseline = OpinionAssessment.model_validate(row["baseline_assessment"])

        # Direction: scenario metadata is authoritative (written by stage 01
        # from the same direction rules stage 05 scores against); the
        # ontology-derived map is the fallback.
        meta_direction: Optional[int] = None
        if isinstance(scenario.metadata, dict):
            raw_dir = scenario.metadata.get("opinion_adversarial_direction")
            try:
                meta_direction = int(raw_dir) if raw_dir is not None else None
            except (TypeError, ValueError):
                meta_direction = None
        if meta_direction is not None:
            adv_direction = meta_direction
            n_direction_from_metadata += 1
        else:
            leaf_name = scenario.opinion_leaf.split(">")[-1].strip()
            adv_direction = adversarial_directions.get(leaf_name, 0)

        attack_leaf = scenario.attack_leaf or ""
        attack_meta_obj = attack_metadata_index.get(attack_leaf)
        attack_meta_dict: Dict[str, Any] = {}
        if attack_meta_obj is not None:
            attack_meta_dict = {
                "mechanism": getattr(attack_meta_obj, "mechanism", ""),
                "primary_system": getattr(attack_meta_obj, "primary_system", ""),
                "platform_hint": getattr(attack_meta_obj, "platform_hint", ""),
                "complexity_tier": getattr(attack_meta_obj, "complexity_tier", ""),
                "temporal_horizon": getattr(attack_meta_obj, "temporal_horizon", ""),
                "epistemic_target": getattr(attack_meta_obj, "epistemic_target", ""),
                "requires_personalization": getattr(attack_meta_obj, "requires_personalization", False),
                "agent_orchestration_required": getattr(attack_meta_obj, "agent_orchestration_required", False),
            }

        if scenario.attack_present and attack_leaf:
            n_attack += 1
            full_context = build_attack_context(
                opinion_leaf=scenario.opinion_leaf,
                attack_leaf=attack_leaf,
                profile=scenario.profile,
                baseline_score=baseline.score,
                adversarial_direction=adv_direction,
                attack_metadata=attack_meta_dict,
            )
            attack_context = {k: full_context.get(k) for k in _SPEC_CONTEXT_KEYS}
            spec: Dict[str, Any] = {
                "attack_present": True,
                "attack_leaf": attack_leaf,
                "attack_leaf_label": attack_context.get("attack_leaf_label", ""),
                "attack_context": attack_context,
                "intensity_proxy": _tier_intensity_proxy(str(attack_context.get("attack_complexity_tier", ""))),
                "spec_source": "deterministic_ontology_v1",
            }
        else:
            spec = {
                "attack_present": False,
                "attack_leaf": None,
                "attack_leaf_label": "CONTROL_NONE",
                "attack_context": {},
                "intensity_proxy": 0.0,
                "spec_source": "deterministic_ontology_v1",
            }

        enriched = dict(row)
        enriched["attack_vector_spec"] = spec
        enriched_rows.append(enriched)
        spec_records.append({"scenario_id": scenario.scenario_id, **spec})

    output_root = Path(output_dir)
    enriched_jsonl = output_root / "scenarios_with_attack_spec.jsonl"
    specs_jsonl = output_root / "attack_vector_specs.jsonl"
    summary_json = output_root / "attack_spec_summary.json"

    write_jsonl(enriched_jsonl, enriched_rows)
    write_jsonl(specs_jsonl, spec_records)
    write_json(
        summary_json,
        {
            "n_rows": len(enriched_rows),
            "n_attack": n_attack,
            "n_control": len(enriched_rows) - n_attack,
            "n_direction_from_metadata": n_direction_from_metadata,
            "n_attack_metadata_entries": len(attack_metadata_index),
            "llm_calls": 0,
            "note": (
                "Stage 03 compiles deterministic attack-vector specifications from the "
                "ATTACK ontology; no synthetic exposure artifacts are generated. The "
                "post-exposure agent in stage 04 estimates the effectiveness of the "
                "specified vector directly."
            ),
        },
    )

    manifest = StageArtifactManifest(
        stage_id="03",
        stage_name="run_opinion_attacks",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(enriched_jsonl), abs_path(specs_jsonl), abs_path(summary_json)],
        record_count=len(enriched_rows),
        metadata={
            "n_attack": n_attack,
            "deterministic": True,
            "llm_calls": 0,
            "n_attack_metadata_entries": len(attack_metadata_index),
        },
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03 - deterministic attack vector specification")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ontology-root", default=None)
    # Accepted for orchestrator compatibility; unused because this stage makes
    # no LLM calls.
    parser.add_argument("--openrouter-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=2)
    parser.add_argument("--self-supervise-attack-realism", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realism-threshold", type=float, default=0.72)
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

    config = Stage03Config(
        stage_name="run_opinion_attacks",
        run_id=args.run_id,
        seed=args.seed,
        ontology_root=args.ontology_root,
        openrouter_model=args.openrouter_model,
        temperature=args.temperature,
        max_repair_iter=args.max_repair_iter,
        save_raw_llm=args.save_raw_llm,
        raw_llm_dir=args.raw_llm_dir,
        timeout_sec=args.timeout_sec,
        max_concurrency=args.max_concurrency,
        self_supervise_attack_realism=args.self_supervise_attack_realism,
        realism_threshold=args.realism_threshold,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 03 completed: %s records (deterministic, no LLM calls)", manifest.record_count)


if __name__ == "__main__":
    main()
