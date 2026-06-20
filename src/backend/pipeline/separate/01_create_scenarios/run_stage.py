#IMPORTANT NOTE!: This script is partially deprecated; needs to be integrated with already sampled content: '/Users/stijnvanseveren/PythonProjects/Project_CSitAoED/src/backend/pipeline/separate/01_create_scenarios/samples/02_integrated/integrated_scenarios_10000.jsonl'

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
import json
import logging
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.scenario.compatibility_rules import (
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
from src.backend.utils.scenario.profile_sampling import sample_profile
from src.backend.utils.scenario.scenario_realism import (
    compute_resilience_index,
    compute_shift_sensitivity_proxy,
    extract_opinion_domain,
    extract_leaf_label,
)
from src.backend.utils.schemas import (
    OpinionCluster,
    OpinionClusterLeaf,
    ProfileConfiguration,
    ScenarioRecord,
    StageArtifactManifest,
    StageConfig,
)

LOGGER = logging.getLogger(__name__)

ATTACK_PRIMARY_AXIS_MARKER = "Primary_Axis > Attack_Family"

# Integrated-scenario design (run_2+): every scenario already pairs a profile,
# a DISARM-red Plan/Prepare/Execute attack triplet, and an opinion parent
# cluster. Each scenario carries its OWN real attack triplet identity (the three
# phase paths are what the simulation agent sees and reasons about). The attacks
# are ~unique per scenario, so for this panel the delta cannot be attributed to
# a specific attack component; that is expected and fine. The conditional-
# susceptibility estimator pools over attacks (it estimates PROFILE moderation),
# while the full triplet and its raw attributes are preserved for provenance and
# for the larger production runs that accumulate attack-type signal over many
# configurations.


class Stage01Config(StageConfig):
    n_scenarios: int = 10
    n_profiles: Optional[int] = None
    attack_ratio: float = 0.5
    attack_leaf: Optional[str] = None
    attack_leaves: Optional[str] = None  # comma-01_separated; takes precedence over attack_leaf
    opinion_leaves: Optional[str] = None  # comma-01_separated explicit opinion leaf selection
    profile_generation_mode: str = "deterministic"
    focus_opinion_domain: Optional[str] = None
    # Integrated-design concentration: restrict the integrated candidate pool to a
    # comma-separated subset of opinion parent clusters (issue domains) and spread
    # n_scenarios across only those. Concentrating the same scenario budget into a
    # few domains is what makes the empirical exposure network dense enough for the
    # network-position correlations (each opinion leaf is then scored by many more
    # profiles, so same-leaf incoming-peer neighborhoods are large rather than ~3).
    focus_opinion_domains: Optional[str] = None
    # Network-exposure scenario governor. The exposure-network layer is capped at
    # network_scenario_cap scenarios; when the requested budget exceeds it (e.g. the
    # production 10K set) the integrated pool is first reduced with a media-keyword
    # heuristic so the retained scenarios are the ones most congruent with the
    # social-media (Bluesky) exposure substrate, then capped. Off for plain runs.
    media_filter: bool = False
    network_scenario_cap: Optional[int] = None
    # Comma-separated profile subtree substrings to drop from the integrated profile
    # before it reaches the agent and the analyses. None -> the curated default set
    # (DEFAULT_PROFILE_SKIP_SUBTREES); empty string -> keep the full profile.
    profile_skip_subtrees: Optional[str] = None
    max_opinion_leaves: Optional[int] = None
    profile_candidate_multiplier: int = 2
    # current design: meta-node compatibility-aware scenario filtering
    enforce_compatibility_rules: bool = True
    realism_weight_temperature: float = 1.5
    drop_direction_neutral_opinions: bool = False
    # Integrated-scenario design (run_2+): when set, stage 01 stops sampling
    # from the ontology and instead selects rows from this pre-built integrated
    # scenario .jsonl, mapping each into a cluster-level ScenarioRecord.
    integrated_scenarios_path: Optional[str] = None


def _resolve_attack_leaves(
    available_leaves: List[str],
    configured_leaves_csv: Optional[str],
    configured_leaf_single: Optional[str],
) -> List[str]:
    """Resolve attack leaves for this run.

    Priority: comma-01_separated --attack-leaves > single --attack-leaf.
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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


_MEDIA_WORD_RE = re.compile(r"\bmedia\b", re.IGNORECASE)


def _attack_mentions_media(attack: Dict[str, Any]) -> bool:
    """Whole-word, case-insensitive 'media' match in any DISARM triplet phase.

    Checks the leaf label and the full taxonomic path of each of the three
    Plan / Prepare / Execute phases. This is the heuristic used to keep the
    exposure-network layer congruent with the social-media (Bluesky) exposure
    substrate when the requested scenario budget exceeds the network cap: media
    operations are the ones whose propagation the empirical graph actually
    describes, so selecting them beats a uniform random draw over all topics.
    """
    triplet = (attack or {}).get("triplet") or {}
    for phase in ("Plan", "Prepare", "Execute"):
        node = triplet.get(phase) or {}
        label = str(node.get("label", ""))
        path = node.get("path") or []
        path_str = " > ".join(str(part) for part in path) if isinstance(path, list) else str(path)
        if _MEDIA_WORD_RE.search(label) or _MEDIA_WORD_RE.search(path_str):
            return True
    return False


def _normalize_domain_set(focus_domains: Optional[List[str]]) -> Optional[set]:
    if not focus_domains:
        return None
    return {_slugify(d) for d in focus_domains if d and d.strip()}


def _stratified_domain_sample(
    path: Path,
    n_target: int,
    seed: int,
    focus_domains: Optional[List[str]] = None,
    media_filter: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Stratified-random selection of n_target rows balanced across issue domains,
    in a single streaming pass over the (large) integrated .jsonl.

    Per-domain reservoir sampling (Algorithm R) keeps a bounded random subset of
    each domain's rows; the final allocation then spreads n_target as evenly as
    possible across the domains that exist. Balancing by domain guarantees that
    every retained issue domain (and therefore every directional leaf, since a
    scenario targets a whole domain cluster) appears, and that each leaf receives
    roughly n_target / n_domains observations, which is what the per-(attack,
    leaf) conditional-susceptibility task models need for stable estimation.

    Two optional concentrators sharpen this for the exposure-network layer:
      * ``focus_domains`` restricts the candidate pool to the named parent
        clusters and spreads the whole budget across just those, multiplying the
        per-leaf observation count (and therefore the same-leaf incoming-peer
        density) by 7 / len(focus_domains).
      * ``media_filter`` keeps only scenarios whose DISARM triplet mentions
        'media', used when the requested budget exceeds the network cap so the
        retained subset matches the social-media exposure substrate.
    """
    rng = random.Random(seed)
    cap = max(40, n_target)
    focus_set = _normalize_domain_set(focus_domains)
    reservoirs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Dict[str, int] = defaultdict(int)

    for row in _iter_jsonl(path):
        domain = str(((row.get("opinion_cluster") or {}).get("parent_name")) or "UNKNOWN")
        if focus_set is not None and _slugify(domain) not in focus_set:
            continue
        if media_filter and not _attack_mentions_media(row.get("attack") or {}):
            continue
        seen[domain] += 1
        reservoir = reservoirs[domain]
        if len(reservoir) < cap:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen[domain] - 1)
            if j < cap:
                reservoir[j] = row

    domains = sorted(reservoirs.keys())
    if not domains:
        raise RuntimeError(
            f"No scenarios found in integrated file: {path}"
            + (f" matching focus domains {sorted(focus_set)}" if focus_set else "")
            + (" after the media-keyword filter" if media_filter else "")
        )
    if focus_set is not None:
        missing = sorted(focus_set - {_slugify(d) for d in domains})
        if missing:
            raise RuntimeError(
                f"Requested focus opinion domains not present in integrated set: {missing}. "
                f"Available (slugified): {sorted({_slugify(d) for d in domains})}"
            )

    base = n_target // len(domains)
    remainder = n_target - base * len(domains)
    allocation = {domain: base for domain in domains}
    for domain in domains[:remainder]:
        allocation[domain] += 1

    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()
    for domain in domains:
        pool = list(reservoirs[domain])
        rng.shuffle(pool)
        take = min(allocation[domain], len(pool))
        for row in pool[:take]:
            selected.append(row)
            selected_ids.add(id(row))

    # Top up if any domain under-filled its allocation.
    if len(selected) < n_target:
        leftovers = [
            row
            for domain in domains
            for row in reservoirs[domain]
            if id(row) not in selected_ids
        ]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n_target - len(selected)])

    rng.shuffle(selected)
    return selected[:n_target], dict(seen)


# Default profile subtrees dropped from the high-resolution profile before it is
# shown to the agent and modelled. The pre-built 10K profiles carry ~520 traits
# across many overlapping taxonomies; that breadth both dilutes what the agent can
# actually condition on and leaves the moderator models hopelessly under-determined
# (p >> n). These substrings remove the redundant personality taxonomies (HEXACO and
# Eysenck duplicate Big Five; the Hexad "user types" are a gamification model with no
# political-susceptibility meaning) and the low-relevance life-admin subtrees (goals,
# values, perceived safety/legal, criminal record, administrative metadata, moral
# alignment, reproductive status), cutting roughly 40% of features while keeping the
# research-relevant core: comprehensive demographics and socioeconomics, the Big Five,
# the full political-psychology battery (ideology, GAL/TAN, libertarian/authoritarian,
# nationalism, populism, system justification, moral foundations), religion, and
# digital/media literacy. Applied to the already-sampled scenarios, so it filters
# rather than re-samples. Override with --profile-skip-subtrees (comma-separated).
DEFAULT_PROFILE_SKIP_SUBTREES: List[str] = [
    "hexad",
    "hexaco",
    "eysenck",
    "goals_",
    "goal_orientation",
    "goal_metadata",
    "safety_and_legal",
    "criminal_record",
    "administrative_and_data",
    "values_value",
    "value_orientation",
    "core_value",
    "alignment_",
    "reproductive_and_family_planning",
]


def _skip_profile_key(key: str, skip_subtrees: List[str]) -> bool:
    kl = str(key).lower()
    return any(sub in kl for sub in skip_subtrees)


def _map_integrated_profile(
    profile_dict: Dict[str, Any], skip_subtrees: Optional[List[str]] = None
) -> ProfileConfiguration:
    """Map a full integrated profile into the pipeline's ProfileConfiguration.

    Substantive attributes are carried through so the downstream agent sees a
    high-resolution profile, EXCEPT the subtrees named in ``skip_subtrees`` (see
    DEFAULT_PROFILE_SKIP_SUBTREES), which are dropped here so they are excluded
    from both the agent prompt and every downstream analysis in one place. The
    demographics block and Big Five (pct -> continuous, level -> categorical) are
    always kept; Big Five mean-pct keys and age_years use the exact names the
    realism layer and Stage 06 moderators expect.
    """
    skip = skip_subtrees if skip_subtrees is not None else DEFAULT_PROFILE_SKIP_SUBTREES
    profile_id = str(profile_dict.get("profile_id", "profile_unknown"))
    demographics = profile_dict.get("demographics", {}) or {}
    categorical: Dict[str, str] = {}
    continuous: Dict[str, float] = {}

    # sex key consumed by profile_context_snapshot / Stage 06 SEX_COLUMNS
    categorical["sex"] = str(demographics.get("sex_assigned_at_birth", "Unknown"))
    for key in (
        "sex_assigned_at_birth",
        "gender_identity",
        "gender_modality",
        "relationship_status",
        "citizenship_status",
        "gender_type",
    ):
        if demographics.get(key) is not None:
            categorical[f"demographic_{key}"] = str(demographics[key])

    big_five = demographics.get("big_five", {}) or {}
    for trait, payload in big_five.items():
        if isinstance(payload, dict):
            if payload.get("pct") is not None:
                continuous[f"big_five_{trait}_mean_pct"] = _to_float(payload["pct"], 50.0)
            if payload.get("level") is not None:
                categorical[f"big_five_{trait}_level"] = str(payload["level"])

    age_years = demographics.get("age_years")
    if age_years is not None:
        continuous["age_years"] = _to_float(age_years, 42.0)

    for key, value in (profile_dict.get("categorical_attributes", {}) or {}).items():
        if _skip_profile_key(key, skip):
            continue
        categorical[str(key)] = str(value)
    for key, value in (profile_dict.get("numeric_attributes", {}) or {}).items():
        if _skip_profile_key(key, skip):
            continue
        continuous[str(key)] = _to_float(value, 0.0)

    profile = ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=categorical,
        continuous_attributes=continuous,
        selected_leaf_nodes=[],
        metadata={"source": "integrated_scenarios_10000", "source_profile_id": profile_id},
    )
    # Structural realism proxies (Stage 04 reads heuristic_shift_sensitivity_proxy;
    # Stage 06 explicitly excludes both proxies from the moderator feature set).
    profile.continuous_attributes["heuristic_shift_sensitivity_proxy"] = compute_shift_sensitivity_proxy(profile)
    profile.continuous_attributes["resilience_index"] = compute_resilience_index(profile)
    return profile


def _derive_attack_struct(attack: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a DISARM-red triplet into structural attack metadata + a
    human-readable triplet for the post-exposure prompt.

    The complexity tier is derived from the operation's signal strength rather
    than from this repository's ATTACK ontology, because the integrated attacks
    are sampled from the external DISARM-red ontology and do not join to it.
    """
    triplet = attack.get("triplet", {}) or {}

    def _phase(name: str) -> Dict[str, Any]:
        phase = triplet.get(name, {}) or {}
        path = phase.get("path") or []
        return {
            "phase": name,
            "leaf_id": phase.get("leaf_id"),
            "tactic": str(phase.get("secondary", "")),
            "label": str(phase.get("label", "")),
            "path": " > ".join(str(part) for part in path),
            "signal_score": _to_float(phase.get("signal_score"), 0.0),
        }

    plan, prepare, execute = _phase("Plan"), _phase("Prepare"), _phase("Execute")
    signal_total = _to_float(attack.get("signal_total"), 0.0)
    if signal_total >= 15:
        tier = "T4_orchestrated"
    elif signal_total >= 11:
        tier = "T3_synthetic"
    elif signal_total >= 8:
        tier = "T2_campaign"
    else:
        tier = "T1_atomic"
    criteria = [str(item) for item in (attack.get("criteria") or [])]
    mechanism = "; ".join(criteria) if criteria else str(attack.get("inclusion_route", ""))

    return {
        "disarm_triplet": {"Plan": plan, "Prepare": prepare, "Execute": execute},
        "config_id": str(attack.get("config_id", "")),
        "source_config_id": str(attack.get("source_config_id", "")),
        "inclusion_route": str(attack.get("inclusion_route", "")),
        "signal_total": signal_total,
        "criteria": criteria,
        "complexity_tier": tier,
        "mechanism": mechanism,
        "primary_system": plan["tactic"] or "audience_targeting",
        "platform_hint": prepare["label"] or "online platforms / social media",
        "epistemic_target": execute["tactic"] or "salience",
        "temporal_horizon": "sustained" if signal_total >= 11 else "burst",
        "plan_tactic": plan["tactic"],
        "prepare_tactic": prepare["tactic"],
        "execute_tactic": execute["tactic"],
    }


def run_stage_integrated(output_dir: str, config: Stage01Config) -> StageArtifactManifest:
    """Stage 01 in integrated mode: select rows from the pre-built integrated
    scenario set (no ontology sampling) and emit cluster-level ScenarioRecords.

    Each emitted record carries the full profile, the full DISARM attack triplet
    (in metadata), and the opinion parent cluster with all its directional
    leaves. Opinions are assessed at the cluster level downstream (one call per
    scenario) and expanded back to per-leaf rows in Stage 05, so the analysis /
    visualisation structure is unchanged.
    """
    output_root = ensure_dir(output_dir)
    integrated_path = Path(config.integrated_scenarios_path)
    if not integrated_path.exists():
        raise RuntimeError(f"Integrated scenarios file not found: {integrated_path}")

    requested = int(config.n_scenarios or 100)
    focus_domains = (
        [d.strip() for d in config.focus_opinion_domains.split(",") if d.strip()]
        if config.focus_opinion_domains
        else None
    )

    # Network-exposure scenario governor. When a cap is set (the exposure-network
    # layer always passes 500) and the requested budget exceeds it, fall back to
    # the media-keyword heuristic and clamp to the cap so the simulated space stays
    # bounded and congruent with the empirical exposure substrate.
    media_filter = bool(config.media_filter)
    cap = int(config.network_scenario_cap) if config.network_scenario_cap else None
    n_target = requested
    if cap is not None and requested > cap:
        n_target = cap
        media_filter = True
        LOGGER.info(
            "Stage 01: requested %d exceeds the network-exposure cap of %d; "
            "applying the media-keyword filter and clamping to %d scenarios.",
            requested, cap, cap,
        )

    # Resolve the profile subtree skip list once: None -> curated default, empty
    # string -> keep the full profile, otherwise a comma-separated override.
    if config.profile_skip_subtrees is None:
        skip_subtrees = DEFAULT_PROFILE_SKIP_SUBTREES
    else:
        skip_subtrees = [s.strip().lower() for s in config.profile_skip_subtrees.split(",") if s.strip()]
    LOGGER.info(
        "Stage 01 integrated mode: selecting %d scenarios from %s (focus_domains=%s, media_filter=%s, profile_skip_subtrees=%d)",
        n_target, integrated_path, focus_domains or "all-7", media_filter, len(skip_subtrees),
    )
    selected, domain_counts = _stratified_domain_sample(
        integrated_path, n_target, config.seed, focus_domains=focus_domains, media_filter=media_filter,
    )
    LOGGER.info(
        "Selected %d scenarios across %d issue domains (source totals: %s)",
        len(selected),
        len({(r.get('opinion_cluster') or {}).get('parent_name') for r in selected}),
        domain_counts,
    )

    scenarios: List[ScenarioRecord] = []
    distinct_leaves: set = set()
    distinct_attacks: set = set()
    domain_selected: Dict[str, int] = defaultdict(int)
    direction_tally: Dict[int, int] = defaultdict(int)

    for idx, row in enumerate(selected):
        scenario_id = str(row.get("scenario_id") or f"scenario_{idx + 1:05d}")
        profile = _map_integrated_profile(row.get("profile", {}) or {}, skip_subtrees=skip_subtrees)
        oc = row.get("opinion_cluster", {}) or {}
        leaves_raw = oc.get("leaves", []) or []
        cluster_leaves: List[OpinionClusterLeaf] = []
        for leaf in leaves_raw:
            path = str(leaf.get("path", "")) or f"{oc.get('key', '')} > {leaf.get('leaf', '')}"
            direction = int(_to_float(leaf.get("adversarial_direction"), 0.0))
            cluster_leaves.append(
                OpinionClusterLeaf(leaf=str(leaf.get("leaf", "")), path=path, adversarial_direction=direction)
            )
            distinct_leaves.add(path)
            direction_tally[direction] += 1

        direction_summary = {str(k): int(_to_float(v, 0.0)) for k, v in (oc.get("direction_summary") or {}).items()}
        cluster = OpinionCluster(
            key=str(oc.get("key", "")),
            family=str(oc.get("family", "")),
            parent_name=str(oc.get("parent_name", "")),
            n_leaves=int(_to_float(oc.get("n_leaves"), len(cluster_leaves))),
            direction_summary=direction_summary,
            leaves=cluster_leaves,
        )
        domain_selected[cluster.parent_name] += 1

        attack_struct = _derive_attack_struct(row.get("attack", {}) or {})
        amplify = direction_summary.get("amplify_+1", direction_summary.get("amplify", 0))
        erode = direction_summary.get("erode_-1", direction_summary.get("erode", 0))
        representative_direction = 1 if amplify > erode else (-1 if erode > amplify else 0)

        # Real per-scenario attack identity (the actual triplet, not an archetype).
        attack_identity = (
            f"DISARM_op_{attack_struct['config_id']}"
            if attack_struct.get("config_id")
            else "DISARM_op_" + hashlib.sha1(json.dumps(attack_struct["disarm_triplet"], sort_keys=True).encode()).hexdigest()[:10]
        )
        distinct_attacks.add(attack_identity)

        scenarios.append(
            ScenarioRecord(
                scenario_id=scenario_id,
                scenario_index=idx,
                random_seed=config.seed + idx,
                profile=profile,
                opinion_leaf=cluster.key,
                opinion_cluster=cluster,
                attack_present=True,
                attack_leaf=attack_identity,
                attack_primary_node=attack_struct["inclusion_route"] or "broad_opinion_manipulation_pathway",
                metadata={
                    "source": "integrated_scenarios_10000",
                    "source_scenario_id": scenario_id,
                    "opinion_domain": cluster.parent_name,
                    "opinion_cluster_key": cluster.key,
                    "opinion_family": cluster.family,
                    "opinion_direction_summary": direction_summary,
                    "opinion_representative_direction": representative_direction,
                    # Scenario-level direction is the cluster's dominant direction;
                    # the authoritative per-leaf directions live in opinion_cluster.leaves
                    # and are what Stage 04/05 score against.
                    "opinion_adversarial_direction": representative_direction,
                    "n_opinion_leaves": cluster.n_leaves,
                    # Full DISARM-red triplet (paths + labels) preserved for provenance.
                    # The simulation agent is shown only the three phase paths and must
                    # reason how they combine into one operation.
                    "disarm_attack": attack_struct,
                    "attack_config_id": attack_struct["config_id"],
                    "attack_signal_total": attack_struct["signal_total"],
                    "attack_inclusion_route": attack_struct["inclusion_route"],
                    "attack_complexity_tier": attack_struct["complexity_tier"],
                    "attack_plan_tactic": attack_struct["plan_tactic"],
                    "attack_prepare_tactic": attack_struct["prepare_tactic"],
                    "attack_execute_tactic": attack_struct["execute_tactic"],
                    "scenario_locale": "Global",
                    "scenario_year": 2026,
                    "scenario_design": "integrated_cluster_panel",
                    "sampling_strategy": "stratified_by_issue_domain_from_prebuilt_integrated_set",
                    "profile_panel_index": idx + 1,
                },
            )
        )

    scenarios_jsonl = output_root / "scenarios.jsonl"
    scenarios_json = output_root / "scenarios.json"
    ontology_catalog = output_root / "ontology_leaf_catalog.json"
    audit_path = output_root / "scenario_compatibility_audit.json"

    write_jsonl(scenarios_jsonl, (s.model_dump() for s in scenarios))
    write_json(scenarios_json, [s.model_dump() for s in scenarios])

    selected_opinion_leaves = sorted(distinct_leaves)
    total_leaf_rows = sum(s.opinion_cluster.n_leaves for s in scenarios if s.opinion_cluster)

    write_json(
        audit_path,
        {
            "enforced": False,
            "scenario_source": "integrated_scenarios_10000",
            "n_scenarios": len(scenarios),
            "n_scenarios_excluded": 0,
            "n_attack_scenarios": len(scenarios),
            "n_expanded_leaf_rows_expected": total_leaf_rows,
            "issue_domains_selected": dict(domain_selected),
            "opinion_direction_tally": {str(k): v for k, v in direction_tally.items()},
            "sampling_provenance": {
                "n_scenarios_requested": requested,
                "n_scenarios_selected": len(scenarios),
                "focus_opinion_domains": focus_domains,
                "media_keyword_filter_applied": media_filter,
                "network_scenario_cap": cap,
                "cap_triggered": bool(cap is not None and requested > cap),
                "seed": config.seed,
                "profile_skip_subtrees": skip_subtrees,
                "n_profile_features_kept": (
                    len(scenarios[0].profile.categorical_attributes) + len(scenarios[0].profile.continuous_attributes)
                    if scenarios else 0
                ),
            },
            "note": (
                "Integrated cluster panel: each scenario targets one issue domain and all its "
                "directional leaves; opinions are assessed cluster-at-once and expanded to "
                "per-leaf rows in Stage 05."
            ),
        },
    )

    sorted_attacks = sorted(distinct_attacks)
    write_json(
        ontology_catalog,
        {
            "ontology_root": "external_integrated_scenarios",
            "scenario_source": abs_path(integrated_path),
            "opinion_leaf_count": len(selected_opinion_leaves),
            "attack_leaf_count": len(sorted_attacks),
            "selected_attack_leaves": sorted_attacks,
            "selected_attack_leaf": sorted_attacks[0] if sorted_attacks else "",
            "selected_opinion_leaves": selected_opinion_leaves,
            "opinion_leaves": selected_opinion_leaves,
            "attack_leaves": sorted_attacks,
            "attack_source": "external_DISARM_red_ontology",
            "attack_grouping_note": (
                "Each scenario carries its own DISARM-red triplet identity; attacks are ~unique "
                "per scenario and are pooled by the conditional-susceptibility estimator."
            ),
            "selected_profile_count": len(scenarios),
            "scenarios_per_profile": 1,
            "scenario_design": "integrated_cluster_panel",
            "compatibility_enforced": False,
            "n_compat_excluded": 0,
            "issue_domains_selected": dict(domain_selected),
            "adversarial_goal": (
                "Erode/shift support across targeted issue-position clusters per the baked "
                "per-leaf adversarial directions."
            ),
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
            "scenario_source": "integrated_scenarios_10000",
            "scenario_design": "integrated_cluster_panel",
            "sampling_strategy": "stratified_by_issue_domain_from_prebuilt_integrated_set",
            "n_attack": len(scenarios),
            "n_control": 0,
            "n_scenarios": len(scenarios),
            "n_expanded_leaf_rows_expected": total_leaf_rows,
            "n_distinct_opinion_leaves": len(selected_opinion_leaves),
            "issue_domains_selected": dict(domain_selected),
            "n_distinct_attacks": len(distinct_attacks),
            "attack_source": "external_DISARM_red_ontology",
            "attack_grouping": "per_scenario_disarm_triplet_identity",
            "selected_profile_count": len(scenarios),
            "scenarios_per_profile": 1,
            "compatibility_enforced": False,
            "opinion_assessment_mode": "cluster_batched",
        },
    )

    write_json(stage_manifest_path(output_root), manifest.model_dump())
    LOGGER.info(
        "Stage 01 integrated: %d scenarios, %d distinct leaves, ~%d expected leaf rows",
        len(scenarios),
        len(selected_opinion_leaves),
        total_leaf_rows,
    )
    return manifest


def run_stage(input_path: str, output_dir: str, config: Stage01Config) -> StageArtifactManifest:
    del input_path
    if config.integrated_scenarios_path:
        return run_stage_integrated(output_dir, config)

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
    # narratives. Capability flags can be flipped per-run if the utils wants
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
    parser.add_argument("--attack-leaves", default=None, help="Comma-01_separated attack leaves; takes precedence over --attack-leaf")
    parser.add_argument("--profile-generation-mode", default="deterministic", choices=["deterministic", "llm", "hybrid"])
    parser.add_argument("--focus-opinion-domain", default=None)
    parser.add_argument(
        "--focus-opinion-domains",
        default=None,
        help="Comma-separated opinion parent clusters (issue domains) to concentrate the "
        "integrated sample into. Densifies the exposure network for the position correlations.",
    )
    parser.add_argument(
        "--media-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict the integrated candidate pool to scenarios whose DISARM triplet mentions 'media'.",
    )
    parser.add_argument(
        "--network-scenario-cap",
        type=int,
        default=None,
        help="Hard cap on integrated scenarios for the exposure-network layer; over it the media filter engages.",
    )
    parser.add_argument(
        "--profile-skip-subtrees",
        default=None,
        help="Comma-separated profile subtree substrings to drop from the integrated profile (agent + analyses). "
        "Omit for the curated default set; pass an empty string to keep the full profile.",
    )
    parser.add_argument("--opinion-leaves", default=None, help="Comma-01_separated explicit opinion leaves; takes precedence over spread sampling")
    parser.add_argument("--max-opinion-leaves", type=int, default=None)
    parser.add_argument("--profile-candidate-multiplier", type=int, default=2)
    parser.add_argument("--enforce-compatibility-rules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-direction-neutral-opinions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--realism-weight-temperature", type=float, default=1.5)
    parser.add_argument(
        "--integrated-scenarios-path",
        default=None,
        help="Path to a pre-built integrated scenarios .jsonl. When set, stage 01 selects rows "
        "from it (stratified by issue domain) instead of sampling from the ontology.",
    )
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
        focus_opinion_domains=args.focus_opinion_domains,
        media_filter=args.media_filter,
        network_scenario_cap=args.network_scenario_cap,
        profile_skip_subtrees=args.profile_skip_subtrees,
        max_opinion_leaves=args.max_opinion_leaves,
        profile_candidate_multiplier=args.profile_candidate_multiplier,
        enforce_compatibility_rules=args.enforce_compatibility_rules,
        drop_direction_neutral_opinions=args.drop_direction_neutral_opinions,
        realism_weight_temperature=args.realism_weight_temperature,
        integrated_scenarios_path=args.integrated_scenarios_path,
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
