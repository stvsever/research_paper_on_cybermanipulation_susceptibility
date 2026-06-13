from __future__ import annotations

"""
Technical overview
------------------
Stage 01 constructs the scenario panel that drives the rest of the simulation.
It is more than a simple sampler: it decides which pseudoprofiles, attack
leaves, and opinion leaves will be combined into the run-specific design.

Current design logic:
- ontologies are loaded hierarchically but only leaf nodes are used for
  estimation
- pseudoprofiles are generated first, either deterministically or with LLM
  adjustments layered on top of deterministic seeds
- one or more opinion leaves are selected per profile, optionally restricted to
  a focused opinion domain
- attack selection is resolved here as well, so downstream stages receive a
  fully explicit scenario manifest

For attacked-only profile-panel runs, this stage effectively defines the panel
shape:

    profile x opinion leaf x attack leaf

This is why one profile can yield multiple attacked rows later in the pipeline.
The stage also tries to spread profiles over the sampled design space rather
than drawing a purely naive random set, which makes small testing runs more
informative and less redundant.
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.compatibility_rules import (
    ScenarioAdmissibility,
    evaluate_scenario_admissibility,
    load_attack_metadata_index,
    load_opinion_metadata_index,
)
from src.backend.utils.io import (
    abs_path,
    ensure_dir,
    env_get_required,
    stage_manifest_path,
    write_json,
    write_jsonl,
)
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.ontology_utils import (
    default_ontology_root,
    find_primary_node,
    flatten_leaf_paths,
    load_adversarial_directions_from_opinion,
    load_ontology_triplet,
)
from src.backend.utils.profile_sampling import sample_profile
from src.backend.utils.scenario_realism import extract_opinion_domain, extract_leaf_label
from src.backend.utils.schemas import ProfileConfiguration, ScenarioRecord, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)

ATTACK_PRIMARY_AXIS_MARKER = "Primary_Axis > Attack_Family"


class Stage01Config(StageConfig):
    n_scenarios: int = 10
    n_profiles: Optional[int] = None
    attack_ratio: float = 0.5
    attack_leaf: Optional[str] = None
    attack_leaves: Optional[str] = None  # comma-separated; takes precedence over attack_leaf
    opinion_leaves: Optional[str] = None  # comma-separated explicit opinion leaf selection
    profile_generation_mode: str = "deterministic"
    focus_opinion_domain: Optional[str] = None
    max_opinion_leaves: Optional[int] = None
    profile_candidate_multiplier: int = 2
    # current design: meta-node compatibility-aware scenario filtering
    enforce_compatibility_rules: bool = True
    realism_weight_temperature: float = 1.5
    drop_direction_neutral_opinions: bool = False


def _resolve_attack_leaves(
    available_leaves: List[str],
    configured_leaves_csv: Optional[str],
    configured_leaf_single: Optional[str],
) -> List[str]:
    """Resolve attack leaves for this run.

    Priority: comma-separated --attack-leaves > single --attack-leaf.
    Each entry is matched case-insensitively against available ontology leaves.
    The sentinel value ALL selects every available primary attack leaf.
    """
    if configured_leaves_csv and configured_leaves_csv.strip().upper() == "ALL":
        return list(available_leaves)
    if configured_leaves_csv:
        requested = [t.strip() for t in configured_leaves_csv.split(",") if t.strip()]
        resolved: List[str] = []
        for req in requested:
            req_lower = req.lower()
            matched = [al for al in available_leaves if req_lower in al.lower()]
            if not matched:
                raise ValueError(f"Attack leaf not found in ontology: {req}")
            resolved.append(matched[0])
        if not resolved:
            raise ValueError("No attack leaves resolved from --attack-leaves")
        return resolved
    if configured_leaf_single:
        if configured_leaf_single not in available_leaves:
            raise ValueError(
                f"Configured attack leaf not found in ontology: {configured_leaf_single}"
            )
        return [configured_leaf_single]
    for leaf in available_leaves:
        if "misleading_narrative_framing" in leaf.lower():
            return [leaf]
    return [available_leaves[0]]


def _scenario_attack_leaves(attack_tree: Dict[str, Any]) -> List[str]:
    """Return only primary ATTACK leaves that represent scenario techniques."""
    return [
        leaf
        for leaf in flatten_leaf_paths(attack_tree)
        if ATTACK_PRIMARY_AXIS_MARKER in leaf
    ]


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _spread_positions(length: int, k: int) -> List[int]:
    if k <= 0 or length <= 0:
        return []
    if k >= length:
        return list(range(length))
    if k == 1:
        return [length // 2]
    raw_positions = [round(i * (length - 1) / (k - 1)) for i in range(k)]
    deduped: List[int] = []
    for idx in raw_positions:
        if idx not in deduped:
            deduped.append(idx)
    candidate = 0
    while len(deduped) < k and candidate < length:
        if candidate not in deduped:
            deduped.append(candidate)
        candidate += 1
    return sorted(deduped[:k])


def _select_opinion_leaves(
    opinion_leaves: List[str],
    focus_opinion_domain: Optional[str],
    max_opinion_leaves: Optional[int],
) -> List[str]:
    candidate_leaves = opinion_leaves
    if focus_opinion_domain:
        normalized_domain = _slugify(focus_opinion_domain)
        candidate_leaves = [
            leaf
            for leaf in opinion_leaves
            if _slugify(extract_opinion_domain(leaf)) == normalized_domain
        ]
        if not candidate_leaves:
            raise RuntimeError(
                f"No opinion leaves found for focus domain '{focus_opinion_domain}'"
            )

    candidate_leaves = sorted(candidate_leaves)
    if not max_opinion_leaves or max_opinion_leaves >= len(candidate_leaves):
        return candidate_leaves

    selected_positions = _spread_positions(len(candidate_leaves), max_opinion_leaves)
    return [candidate_leaves[idx] for idx in selected_positions]


def _target_profile_count(
    config: Stage01Config,
    selected_opinion_leaves: List[str],
    n_attack_leaves: int = 1,
) -> int:
    if config.n_profiles is not None and config.n_profiles > 0:
        return config.n_profiles
    n_leaves = max(1, len(selected_opinion_leaves))
    n_attacks = max(1, n_attack_leaves)
    return max(1, int(math.ceil(config.n_scenarios / (n_leaves * n_attacks))))


def _profile_feature_vector(profile: ProfileConfiguration) -> np.ndarray:
    """Build a numeric vector from all profile attributes for diversity selection.

    Continuous values are roughly scaled to [0, 1].  Categorical values are
    hashed to a stable float.  All profiles from the same ontology produce
    vectors of identical length because they share the same attribute keys.
    """
    parts: List[float] = []
    for key in sorted(profile.continuous_attributes):
        val = float(profile.continuous_attributes[key])
        if "_pct" in key or "mean_pct" in key:
            parts.append(max(0.0, min(1.0, val / 100.0)))
        elif "age" in key:
            parts.append(max(0.0, min(1.0, val / 100.0)))
        elif "income" in key:
            parts.append(max(0.0, min(1.0, val / 200_000.0)))
        else:
            parts.append(max(0.0, min(1.0, val / 100.0)))
    for key in sorted(profile.categorical_attributes):
        parts.append((hash(profile.categorical_attributes[key]) % 10000) / 10000.0)
    return np.array(parts, dtype=float) if parts else np.array([0.0])


def _select_diverse_candidates(candidates: List[Dict[str, object]], target_count: int) -> List[Dict[str, object]]:
    if target_count >= len(candidates):
        return list(candidates)

    vectors = np.vstack([item["feature_vector"] for item in candidates])
    mean_vector = vectors.mean(axis=0)
    selected_indices: List[int] = []

    # Stratify by first categorical dimension to ensure representation
    cat_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(candidates):
        cat_attrs = item["profile_result"].profile.categorical_attributes
        strat_key = next(iter(sorted(cat_attrs.values())), "all") if cat_attrs else "all"
        cat_groups[strat_key].append(idx)

    for group_key in sorted(cat_groups):
        if len(selected_indices) >= target_count:
            break
        options = cat_groups[group_key]
        if not options:
            continue
        chosen = max(options, key=lambda i: float(np.linalg.norm(vectors[i] - mean_vector)))
        if chosen not in selected_indices:
            selected_indices.append(chosen)

    if not selected_indices:
        start_idx = int(np.argmax(np.linalg.norm(vectors - mean_vector, axis=1)))
        selected_indices.append(start_idx)

    remaining = [idx for idx in range(len(candidates)) if idx not in selected_indices]
    while len(selected_indices) < target_count and remaining:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            distances = [float(np.linalg.norm(vectors[idx] - vectors[j])) for j in selected_indices]
            score = min(distances) if distances else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    selected = [candidates[idx] for idx in selected_indices[:target_count]]
    selected.sort(key=lambda item: item["profile_result"].profile.profile_id)
    return selected


def _allocate_profiles(
    profile_tree: Dict[str, dict],
    config: Stage01Config,
    llm_generator,
    target_profiles: int,
) -> List[Dict[str, object]]:
    candidate_count = max(
        target_profiles,
        int(math.ceil(target_profiles * max(1, config.profile_candidate_multiplier))),
    )
    sampled_profiles: List[Dict[str, object]] = []

    for idx in range(candidate_count):
        scenario_seed = config.seed + idx * 9973
        profile_id = f"profile_{idx + 1:04d}"
        profile_result = sample_profile(
            profile_tree=profile_tree,
            profile_id=profile_id,
            seed=scenario_seed,
            generation_mode=config.profile_generation_mode,
            llm_generator=llm_generator,
        )
        sampled_profiles.append(
            {
                "candidate_index": idx,
                "candidate_seed": scenario_seed,
                "profile_result": profile_result,
                "feature_vector": _profile_feature_vector(profile_result.profile),
            }
        )

    return _select_diverse_candidates(sampled_profiles, target_profiles)


def _build_attack_assignments(total_scenarios: int, attack_ratio: float) -> List[bool]:
    n_attack = max(0, min(total_scenarios, int(round(total_scenarios * attack_ratio))))
    assignments = [False] * total_scenarios
    chosen_positions = _spread_positions(total_scenarios, n_attack)
    for idx in chosen_positions:
        assignments[idx] = True
    return assignments


def run_stage(input_path: str, output_dir: str, config: Stage01Config) -> StageArtifactManifest:
    del input_path
    output_root = ensure_dir(output_dir)

    project_root = Path(__file__).resolve().parents[5]
    ontology_root = Path(config.ontology_root) if config.ontology_root else default_ontology_root(
        project_root, config.use_test_ontology
    )

    ontologies = load_ontology_triplet(ontology_root)
    opinion_leaves = flatten_leaf_paths(ontologies["OPINION"])
    attack_leaves = _scenario_attack_leaves(ontologies["ATTACK"])
    profile_tree = ontologies["PROFILE"]

    if not opinion_leaves:
        raise RuntimeError("No OPINION leaf nodes found")
    if not attack_leaves:
        raise RuntimeError("No ATTACK leaf nodes found")

    run_attack_leaves = _resolve_attack_leaves(attack_leaves, config.attack_leaves, config.attack_leaf)

    # Filter direction-neutral leaves BEFORE the spread selection so every
    # sampled slot is occupied by a leaf with a defined adversarial goal.
    candidate_opinion_leaves = opinion_leaves
    if config.drop_direction_neutral_opinions:
        early_meta_index = load_opinion_metadata_index(ontologies["OPINION"])
        directional_candidates = [
            leaf for leaf in opinion_leaves
            if leaf in early_meta_index and early_meta_index[leaf].adversarial_direction != 0
        ]
        if directional_candidates:
            candidate_opinion_leaves = directional_candidates

    if config.opinion_leaves:
        requested = [t.strip() for t in config.opinion_leaves.split(",") if t.strip()]
        selected_opinion_leaves = []
        for req in requested:
            req_lower = req.lower()
            matched = [leaf for leaf in candidate_opinion_leaves if req_lower in leaf.lower()]
            if not matched:
                raise ValueError(f"Opinion leaf not found among directional candidates: {req}")
            selected_opinion_leaves.append(matched[0])
    else:
        selected_opinion_leaves = _select_opinion_leaves(
            opinion_leaves=candidate_opinion_leaves,
            focus_opinion_domain=config.focus_opinion_domain,
            max_opinion_leaves=config.max_opinion_leaves,
        )
    if not selected_opinion_leaves:
        raise RuntimeError("No opinion leaves selected for scenario generation")

    target_profiles = _target_profile_count(config, selected_opinion_leaves, len(run_attack_leaves))

    if config.profile_generation_mode.lower() != "deterministic":
        LOGGER.warning(
            "profile_generation_mode '%s' is no longer supported; profiles are "
            "sampled programmatically from the ontology sampling rules "
            "(coherent, non-contradictory state-space selection). Proceeding "
            "deterministically.",
            config.profile_generation_mode,
        )

    selected_profiles = _allocate_profiles(
        profile_tree=profile_tree,
        config=config,
        llm_generator=lambda *args, **kwargs: None,
        target_profiles=target_profiles,
    )

    # ── Compatibility metadata indices ───────────────────────────────────────
    attack_meta_index = load_attack_metadata_index(ontologies["ATTACK"])
    opinion_meta_index = load_opinion_metadata_index(ontologies["OPINION"])
    adversarial_directions, adversarial_goal = load_adversarial_directions_from_opinion(ontologies["OPINION"])
    # The simulator declares which capabilities are available. By default the
    # full set is available — the attack-side agent always sees the target
    # profile (passed via attack_context), the simulator can run multi-turn
    # exchanges, can produce generative-AI artefacts (the LLM does), can
    # acquire OSINT-style data (we hand it the profile), can reach search-
    # index surfaces (the prompt simulates them), and can stage disruption
    # narratives. Capability flags can be flipped per-run if the user wants
    # to ablate a specific capability.
    available_capabilities = (
        "agent_orchestration",
        "profile_personalisation",
        "data_acquisition",
        "generative_ai_models",
        "search_index_targetable",
        "intrusion_or_disruption",
    )

    if config.drop_direction_neutral_opinions:
        directional_opinion_leaves = [
            leaf for leaf in selected_opinion_leaves
            if leaf in opinion_meta_index and opinion_meta_index[leaf].adversarial_direction != 0
        ]
        if directional_opinion_leaves:
            selected_opinion_leaves = directional_opinion_leaves

    total_scenarios = target_profiles * len(run_attack_leaves) * len(selected_opinion_leaves)
    attack_assignments = _build_attack_assignments(total_scenarios, config.attack_ratio)
    n_attack = int(sum(1 for item in attack_assignments if item))
    n_control = total_scenarios - n_attack

    scenarios: List[ScenarioRecord] = []
    excluded_count = 0
    excluded_log: List[Dict[str, object]] = []
    realism_weights: List[float] = []
    scenario_cursor = 0
    for profile_idx, profile_bundle in enumerate(selected_profiles, start=1):
        profile_result = profile_bundle["profile_result"]
        profile_seed = int(profile_bundle["candidate_seed"])
        diversity_vector = profile_bundle["feature_vector"].tolist()

        for attack_idx, attack_leaf_path in enumerate(run_attack_leaves, start=1):
            for leaf_idx, opinion_leaf in enumerate(selected_opinion_leaves, start=1):
                attack_present = attack_assignments[scenario_cursor]
                current_attack = attack_leaf_path if attack_present else None
                scenario_seed = profile_seed + attack_idx * 1009 + leaf_idx * 101

                # ── Compatibility evaluation (only meaningful for attacked rows) ──
                admissibility: Optional[ScenarioAdmissibility] = None
                if attack_present and attack_leaf_path in attack_meta_index and opinion_leaf in opinion_meta_index:
                    admissibility = evaluate_scenario_admissibility(
                        profile=profile_result.profile,
                        attack_meta=attack_meta_index[attack_leaf_path],
                        opinion_meta=opinion_meta_index[opinion_leaf],
                        available_capabilities=available_capabilities,
                    )
                    if config.enforce_compatibility_rules and not admissibility.admissible:
                        excluded_count += 1
                        excluded_log.append({
                            "scenario_index": scenario_cursor,
                            "profile_id": profile_result.profile.profile_id,
                            "attack_leaf": attack_leaf_path,
                            "opinion_leaf": opinion_leaf,
                            "reasons": admissibility.excluded_reasons,
                        })
                        scenario_cursor += 1
                        continue

                rw = float(admissibility.realism_weight) if admissibility is not None else 1.0
                realism_weights.append(rw)

                scenarios.append(
                    ScenarioRecord(
                        scenario_id=f"scenario_{scenario_cursor + 1:05d}",
                        scenario_index=scenario_cursor,
                        random_seed=scenario_seed,
                        profile=profile_result.profile,
                        opinion_leaf=opinion_leaf,
                        attack_present=attack_present,
                        attack_leaf=current_attack,
                        attack_primary_node=find_primary_node(attack_leaf_path) if attack_present else None,
                        metadata={
                            "profile_sampling_mode": profile_result.sampling_mode_used,
                            "run_attack_leaves": [str(al) for al in run_attack_leaves],
                            "assigned_attack_leaf": attack_leaf_path,
                            "opinion_domain": extract_opinion_domain(opinion_leaf),
                            "opinion_leaf_label": extract_leaf_label(opinion_leaf),
                            "scenario_locale": "Belgium_Flanders",
                            "scenario_year": 2026,
                            "scenario_design": "profile_panel_factorial",
                            "sampling_strategy": "diverse_profile_panel_crossed_with_attacks_and_opinions",
                            "profile_panel_index": profile_idx,
                            "attack_index_within_profile": attack_idx,
                            "leaf_repeat_index_within_profile": leaf_idx,
                            "selected_profile_count": target_profiles,
                            "selected_opinion_leaf_count": len(selected_opinion_leaves),
                            "selected_attack_leaf_count": len(run_attack_leaves),
                            "focus_opinion_domain": config.focus_opinion_domain,
                            "profile_feature_vector": diversity_vector,
                            # run_1: STRUCTURAL compatibility metadata only
                            "compatibility_realism_weight": rw,
                            "compatibility_notes": (
                                list(admissibility.notes) if admissibility else []
                            ),
                            "opinion_adversarial_direction": int(
                                opinion_meta_index[opinion_leaf].adversarial_direction
                            ) if opinion_leaf in opinion_meta_index else 0,
                        },
                    )
                )
                scenario_cursor += 1

    scenarios_jsonl = output_root / "scenarios.jsonl"
    scenarios_json = output_root / "scenarios.json"
    ontology_catalog = output_root / "ontology_leaf_catalog.json"

    write_jsonl(scenarios_jsonl, (s.model_dump() for s in scenarios))
    write_json(scenarios_json, [s.model_dump() for s in scenarios])

    # ── Compatibility audit artefact (current design) ───────────────────────────────
    audit_path = output_root / "scenario_compatibility_audit.json"
    write_json(
        audit_path,
        {
            "enforced": bool(config.enforce_compatibility_rules),
            "n_scenarios_after_filter": len(scenarios),
            "n_scenarios_excluded": excluded_count,
            "n_attack_scenarios": int(sum(1 for s in scenarios if s.attack_present)),
            "exclusion_log": excluded_log[:200],  # truncate for readability
            "exclusion_log_truncated": len(excluded_log) > 200,
            "realism_weight_stats": {
                "mean": float(sum(realism_weights) / max(1, len(realism_weights))),
                "min": float(min(realism_weights, default=1.0)),
                "max": float(max(realism_weights, default=1.0)),
                "n": len(realism_weights),
            },
            "adversarial_goal": adversarial_goal,
            "adversarial_directions_count": len(adversarial_directions),
        },
    )

    write_json(
        ontology_catalog,
        {
            "ontology_root": abs_path(ontology_root),
            "opinion_leaf_count": len(opinion_leaves),
            "attack_leaf_count": len(attack_leaves),
            "selected_attack_leaves": [str(al) for al in run_attack_leaves],
            "selected_attack_leaf": run_attack_leaves[0],
            "selected_opinion_leaves": selected_opinion_leaves,
            "opinion_leaves": opinion_leaves,
            "attack_leaves": attack_leaves,
            "selected_profile_count": target_profiles,
            "scenarios_per_profile": len(selected_opinion_leaves) * len(run_attack_leaves),
            "scenario_design": "profile_panel_factorial",
            "compatibility_enforced": bool(config.enforce_compatibility_rules),
            "n_compat_excluded": excluded_count,
            "adversarial_goal": adversarial_goal,
        },
    )

    manifest = StageArtifactManifest(
        stage_id="01",
        stage_name="create_scenarios",
        primary_output_path=abs_path(scenarios_jsonl),
        output_files=[
            abs_path(scenarios_jsonl),
            abs_path(scenarios_json),
            abs_path(ontology_catalog),
            abs_path(audit_path),
        ],
        record_count=len(scenarios),
        metadata={
            "n_attack": int(sum(1 for s in scenarios if s.attack_present)),
            "n_control": int(sum(1 for s in scenarios if not s.attack_present)),
            "n_excluded_by_compatibility": excluded_count,
            "attack_ratio": config.attack_ratio,
            "selected_attack_leaves": [str(al) for al in run_attack_leaves],
            "selected_attack_leaf_count": len(run_attack_leaves),
            "profile_generation_mode": config.profile_generation_mode,
            "use_test_ontology": config.use_test_ontology,
            "ontology_root": abs_path(ontology_root),
            "focus_opinion_domain": config.focus_opinion_domain,
            "selected_opinion_leaf_count": len(selected_opinion_leaves),
            "selected_profile_count": target_profiles,
            "scenarios_per_profile": len(selected_opinion_leaves) * len(run_attack_leaves),
            "profile_candidate_multiplier": config.profile_candidate_multiplier,
            "scenario_design": "profile_panel_factorial",
            "sampling_strategy": "diverse_profile_panel_crossed_with_attacks_and_opinions",
            "compatibility_enforced": bool(config.enforce_compatibility_rules),
            "drop_direction_neutral_opinions": bool(config.drop_direction_neutral_opinions),
            "adversarial_goal": adversarial_goal,
            "n_directional_opinion_leaves": int(sum(
                1 for leaf in selected_opinion_leaves
                if leaf in opinion_meta_index and opinion_meta_index[leaf].adversarial_direction != 0
            )),
        },
    )

    write_json(stage_manifest_path(output_root), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01 - Create scenarios")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--n-profiles", type=int, default=None)
    parser.add_argument("--attack-ratio", type=float, default=0.5)
    parser.add_argument("--attack-leaf", default=None)
    parser.add_argument("--attack-leaves", default=None, help="Comma-separated attack leaves; takes precedence over --attack-leaf")
    parser.add_argument("--profile-generation-mode", default="deterministic", choices=["deterministic", "llm", "hybrid"])
    parser.add_argument("--focus-opinion-domain", default=None)
    parser.add_argument("--opinion-leaves", default=None, help="Comma-separated explicit opinion leaves; takes precedence over spread sampling")
    parser.add_argument("--max-opinion-leaves", type=int, default=None)
    parser.add_argument("--profile-candidate-multiplier", type=int, default=2)
    parser.add_argument("--enforce-compatibility-rules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-direction-neutral-opinions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--realism-weight-temperature", type=float, default=1.5)
    parser.add_argument("--use-test-ontology", action="store_true", default=False)
    parser.add_argument("--ontology-root", default=None)
    parser.add_argument("--openrouter-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-repair-iter", type=int, default=2)
    parser.add_argument("--save-raw-llm", action="store_true", default=False)
    parser.add_argument("--raw-llm-dir", default=None)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    load_dotenv(Path(__file__).resolve().parents[5] / ".env")

    config = Stage01Config(
        stage_name="create_scenarios",
        run_id=args.run_id,
        seed=args.seed,
        n_scenarios=args.n_scenarios,
        n_profiles=args.n_profiles,
        attack_ratio=args.attack_ratio,
        attack_leaf=args.attack_leaf,
        attack_leaves=args.attack_leaves,
        opinion_leaves=args.opinion_leaves,
        profile_generation_mode=args.profile_generation_mode,
        focus_opinion_domain=args.focus_opinion_domain,
        max_opinion_leaves=args.max_opinion_leaves,
        profile_candidate_multiplier=args.profile_candidate_multiplier,
        enforce_compatibility_rules=args.enforce_compatibility_rules,
        drop_direction_neutral_opinions=args.drop_direction_neutral_opinions,
        realism_weight_temperature=args.realism_weight_temperature,
        use_test_ontology=args.use_test_ontology,
        ontology_root=args.ontology_root,
        openrouter_model=args.openrouter_model,
        temperature=args.temperature,
        max_repair_iter=args.max_repair_iter,
        save_raw_llm=args.save_raw_llm,
        raw_llm_dir=args.raw_llm_dir,
        timeout_sec=args.timeout_sec,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 01 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
