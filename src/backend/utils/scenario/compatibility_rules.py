from __future__ import annotations

"""
Cross-ontology STRUCTURAL compatibility engine (current design, v4 schema).

The engine resolves admissibility, capability prerequisites, complexity
tier, epistemic target, and natural-companion suggestions for any
(profile × attack × opinion) tuple by combining:

  1. Per-leaf inline metadata (highest precedence) — used only when a leaf
     has a unique structural constraint not derivable from its parent path.
  2. Top-level ``_compatibility_rules`` rules in the ATTACK ontology — apply
     attributes (``compatible_opinion_domains``, ``incompatible_opinion_domains``,
     ``requires_capability``, ``complexity_tier``, ``epistemic_target``,
     ``natural_companions``, ``natural_predecessors``) to whole subtrees via
     path-glob patterns.
  3. Top-level ``_direction_rules`` rules in the OPINION ontology — apply a
     subtree-wide adversarial-direction default to direction-neutral leaves.

Path-glob patterns
------------------
Patterns use `**` as a wildcard between path segments. A pattern matches a
leaf when every literal segment in the pattern appears in the same order
in the leaf's `>`-joined path. Example:

    pattern: "**Foundation_Model_Video_Generation**"
    matches: "Political_Opinion_Cybermanipulation_Ontology > Primary_Axis >
              Attack_Family > Ai_Generated_Synthetic_Media_And_Content >
              Foundation_Model_Video_Generation > Lip_Sync_Deepfake_Generation >
              Politician_Speech_Lip_Sync"

Rule evaluation
---------------
For attribute attributes: ``ordered_merge_last_wins_per_attribute`` —
later matching rules override earlier matching rules for the same attribute.
For list attributes (``requires_capability``, ``natural_companions``,
``natural_predecessors``): rules are UNION-merged across all matches.
Per-leaf inline metadata always overrides rule-derived attributes.

Public API
----------
- ``load_attack_metadata_index(tree)`` -> {leaf_path: AttackLeafMetadata}
- ``load_opinion_metadata_index(tree)`` -> {leaf_path: OpinionLeafMetadata}
- ``evaluate_scenario_admissibility(...)`` -> ScenarioAdmissibility
- ``filter_admissible_candidates(candidates)``
- ``realism_weighted_sample_indices(weights, target_count, seed)``

The engine is deliberately conservative: when in doubt, scenarios are
admitted rather than excluded.  Hard exclusion fires only on explicit
incompatible-domain whitelist mismatches and on explicit capability
prerequisites that the simulator declares unavailable.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.backend.utils.ontology_utils import (
    OntologyTree,
    flatten_leaf_paths,
    get_leaf_metadata,
)
from src.backend.utils.schemas import ProfileConfiguration


# ──────────────────────────────────────────────────────────────────────────────
# Path-pattern matching
# ──────────────────────────────────────────────────────────────────────────────


def _split_path(path: str, sep: str = " > ") -> List[str]:
    return [p.strip() for p in path.split(sep) if p.strip()]


def _split_pattern(pattern: str) -> List[str]:
    """Pattern segments. Empty strings (between `**`) preserved as wildcards."""
    return [p.strip() for p in pattern.replace("**", " ** ").split(" ** ")]


def path_pattern_matches(pattern: str, path_segments: List[str]) -> bool:
    """Return True if `pattern` matches the path's `>`-joined segments.

    Patterns are split on the literal `**` wildcard. Every non-empty pattern
    segment must appear as an exact match against one of the path_segments,
    in the order they appear in the pattern. Wildcards (empty segments) match
    any number of intervening segments.
    """
    parts = [p for p in _split_pattern(pattern) if p]
    if not parts:
        return True
    cursor = 0
    for needle in parts:
        # Each pattern part may itself contain `>`-joined sub-segments because
        # the JSON often writes patterns like "**Issue_Position_Taxonomy >
        # Defense_And_National_Security**". Split and match the sub-segments
        # contiguously.
        sub_needles = _split_path(needle)
        if not sub_needles:
            continue
        found = False
        while cursor + len(sub_needles) <= len(path_segments):
            if path_segments[cursor : cursor + len(sub_needles)] == sub_needles:
                cursor += len(sub_needles)
                found = True
                break
            cursor += 1
        if not found:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Metadata indices
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AttackLeafMetadata:
    """Resolved STRUCTURAL attack-leaf metadata from leaf + subtree rules."""
    leaf_path: str
    leaf_label: str
    family: str
    complexity_tier: str
    temporal_horizon: str
    epistemic_target: str
    requires_capability: Tuple[str, ...]
    compatible_opinion_domains: Tuple[str, ...]
    incompatible_opinion_domains: Tuple[str, ...]
    natural_companions: Tuple[str, ...]
    natural_predecessors: Tuple[str, ...]
    mutually_exclusive_with_paths: Tuple[str, ...]
    scenario_role: str
    is_classification_axis: bool


@dataclass
class OpinionLeafMetadata:
    leaf_path: str
    leaf_label: str
    domain: str
    adversarial_direction: int
    direction_rationale: str
    direction_source: str  # "leaf" | "rule" | "default"


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _safe_tuple_str(value: Any) -> Tuple[str, ...]:
    if not value:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _walk_leaf(tree: OntologyTree, path: str) -> Dict[str, Any]:
    parts = _split_path(path)
    node: Any = tree
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    return node if isinstance(node, dict) else {}


# ──────────────────────────────────────────────────────────────────────────────
# Attack metadata loader (per-leaf + subtree rules)
# ──────────────────────────────────────────────────────────────────────────────


_ATTACK_FAMILY_FALLBACK_NAMES = {
    "Political_Opinion_Cybermanipulation_Ontology",
    "Primary_Axis",
    "Secondary_Axes",
    "Attack_Family",
    "ATTACK_VECTORS",
}


def _resolve_attack_family(path_segments: List[str]) -> str:
    for seg in path_segments:
        if seg not in _ATTACK_FAMILY_FALLBACK_NAMES:
            return seg
    return ""


def _apply_attack_rules(
    rules: Sequence[Dict[str, Any]],
    leaf_path: str,
) -> Dict[str, Any]:
    """Walk every rule and accumulate matched attributes.

    Scalar attributes use last-match-wins. List attributes are UNION-merged.
    """
    path_segments = _split_path(leaf_path)
    scalar_attrs: Dict[str, Any] = {}
    list_attrs: Dict[str, List[str]] = {
        "requires_capability": [],
        "natural_companions": [],
        "natural_predecessors": [],
        "mutually_exclusive_with_paths": [],
    }
    for rule in rules:
        patterns = rule.get("applies_to_attack_paths", [])
        if not isinstance(patterns, list):
            continue
        if not any(path_pattern_matches(str(p), path_segments) for p in patterns):
            continue
        for key in ("complexity_tier", "temporal_horizon", "epistemic_target"):
            if key in rule:
                scalar_attrs[key] = rule[key]
        for key in ("scenario_role", "is_classification_axis"):
            if key in rule:
                scalar_attrs[key] = rule[key]
        for key in ("compatible_opinion_domains", "incompatible_opinion_domains"):
            if key in rule:
                # last-match-wins for whitelist / blacklist
                scalar_attrs[key] = list(rule[key]) if rule[key] else []
        for key in (
            "requires_capability",
            "natural_companions",
            "natural_predecessors",
            "mutually_exclusive_with_paths",
        ):
            if key in rule:
                values = rule[key] if isinstance(rule[key], list) else [rule[key]]
                for v in values:
                    if v not in list_attrs[key]:
                        list_attrs[key].append(v)
    return {**scalar_attrs, **list_attrs}


def load_attack_metadata_index(
    attack_tree: OntologyTree,
) -> Dict[str, AttackLeafMetadata]:
    rules = (
        attack_tree.get("_compatibility_rules", {}).get("rules", [])
        if isinstance(attack_tree.get("_compatibility_rules"), dict)
        else []
    )
    index: Dict[str, AttackLeafMetadata] = {}
    for leaf_path in flatten_leaf_paths(attack_tree):
        body = _walk_leaf(attack_tree, leaf_path)
        path_segments = _split_path(leaf_path)
        family = _resolve_attack_family(path_segments)
        leaf_label = path_segments[-1] if path_segments else leaf_path
        rule_attrs = _apply_attack_rules(rules, leaf_path)

        # Per-leaf inline metadata overrides rules
        complexity_tier = _safe_str(
            body.get("complexity_tier") or rule_attrs.get("complexity_tier") or "T2_campaign"
        )
        temporal_horizon = _safe_str(
            body.get("temporal_horizon") or rule_attrs.get("temporal_horizon") or "days"
        )
        epistemic_target = _safe_str(
            body.get("epistemic_target") or rule_attrs.get("epistemic_target") or "evaluative_attitude"
        )
        compatible_domains: Tuple[str, ...] = (
            _safe_tuple_str(body.get("compatible_opinion_domains"))
            if "compatible_opinion_domains" in body
            else _safe_tuple_str(rule_attrs.get("compatible_opinion_domains"))
        )
        incompatible_domains: Tuple[str, ...] = (
            _safe_tuple_str(body.get("incompatible_opinion_domains"))
            if "incompatible_opinion_domains" in body
            else _safe_tuple_str(rule_attrs.get("incompatible_opinion_domains"))
        )
        requires_capability = _safe_tuple_str(
            body.get("requires_capability") or rule_attrs.get("requires_capability")
        )
        companions = _safe_tuple_str(
            body.get("natural_companions") or rule_attrs.get("natural_companions")
        )
        predecessors = _safe_tuple_str(
            body.get("natural_predecessors") or rule_attrs.get("natural_predecessors")
        )
        mutually_exclusive = _safe_tuple_str(
            body.get("mutually_exclusive_with_paths") or rule_attrs.get("mutually_exclusive_with_paths")
        )
        scenario_role = _safe_str(
            body.get("scenario_role") or rule_attrs.get("scenario_role") or "target_exposure"
        )
        is_classification_axis = bool(
            body.get("is_classification_axis") or rule_attrs.get("is_classification_axis") or False
        )

        index[leaf_path] = AttackLeafMetadata(
            leaf_path=leaf_path,
            leaf_label=leaf_label,
            family=family,
            complexity_tier=complexity_tier,
            temporal_horizon=temporal_horizon,
            epistemic_target=epistemic_target,
            requires_capability=requires_capability,
            compatible_opinion_domains=compatible_domains,
            incompatible_opinion_domains=incompatible_domains,
            natural_companions=companions,
            natural_predecessors=predecessors,
            mutually_exclusive_with_paths=mutually_exclusive,
            scenario_role=scenario_role,
            is_classification_axis=is_classification_axis,
        )
    return index


# ──────────────────────────────────────────────────────────────────────────────
# Opinion metadata loader (per-leaf + direction rules)
# ──────────────────────────────────────────────────────────────────────────────


def _apply_opinion_direction_rules(
    rules: Sequence[Dict[str, Any]],
    leaf_path: str,
) -> Tuple[Optional[int], str, str]:
    """Return (direction_or_None, rationale, source) for the FIRST matching rule.

    First-match-wins is the documented behaviour for direction rules.
    """
    path_segments = _split_path(leaf_path)
    for rule in rules:
        patterns = rule.get("applies_to_opinion_paths", [])
        if not isinstance(patterns, list):
            continue
        if not any(path_pattern_matches(str(p), path_segments) for p in patterns):
            continue
        d = rule.get("default_direction")
        if d is None:
            continue
        try:
            return int(d), _safe_str(rule.get("rationale")), "rule"
        except (TypeError, ValueError):
            continue
    return None, "", "default"


def load_opinion_metadata_index(
    opinion_tree: OntologyTree,
) -> Dict[str, OpinionLeafMetadata]:
    rules = (
        opinion_tree.get("_direction_rules", {}).get("rules", [])
        if isinstance(opinion_tree.get("_direction_rules"), dict)
        else []
    )
    index: Dict[str, OpinionLeafMetadata] = {}
    for leaf_path in flatten_leaf_paths(opinion_tree):
        meta = get_leaf_metadata(opinion_tree, leaf_path)
        path_segments = _split_path(leaf_path)
        if path_segments and path_segments[0] == "Issue_Position_Taxonomy" and len(path_segments) >= 2:
            domain = path_segments[1]
        else:
            domain = path_segments[0] if path_segments else ""
        leaf_label = path_segments[-1] if path_segments else leaf_path

        # Per-leaf metadata
        direction = None
        rationale = ""
        source = "default"
        if "adversarial_direction" in meta:
            try:
                direction = int(meta.get("adversarial_direction") or 0)
                rationale = _safe_str(meta.get("direction_rationale"))
                source = "leaf"
            except (TypeError, ValueError):
                direction = 0
                source = "default"

        # If leaf direction is 0 OR missing, consult subtree rules
        if direction in (None, 0):
            rule_dir, rule_rationale, rule_source = _apply_opinion_direction_rules(rules, leaf_path)
            if rule_dir is not None:
                # If leaf explicitly says 0, that overrides rules
                if direction is None:
                    direction = rule_dir
                    rationale = rule_rationale
                    source = rule_source

        if direction is None:
            direction = 0
            source = "default"

        index[leaf_path] = OpinionLeafMetadata(
            leaf_path=leaf_path,
            leaf_label=leaf_label,
            domain=domain,
            adversarial_direction=int(direction),
            direction_rationale=rationale,
            direction_source=source,
        )
    return index


# ──────────────────────────────────────────────────────────────────────────────
# Admissibility evaluation (structural only)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScenarioAdmissibility:
    admissible: bool
    realism_weight: float
    excluded_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def evaluate_scenario_admissibility(
    profile: ProfileConfiguration,
    attack_meta: AttackLeafMetadata,
    opinion_meta: OpinionLeafMetadata,
    *,
    available_capabilities: Sequence[str] = (
        "agent_orchestration",
        "profile_personalisation",
        "data_acquisition",
        "generative_ai_models",
        "search_index_targetable",
        "intrusion_or_disruption",
        "legal_process_or_platform_policy_access",
        "human_source_recruitment_access",
        "interactive_media_or_game_surface",
    ),
) -> ScenarioAdmissibility:
    """Decide admissibility based on structural prerequisites only.

    Hard exclusions:
    - Opinion domain matches an explicit ``incompatible_opinion_domains`` entry.
    - Attack declares a non-empty ``compatible_opinion_domains`` whitelist and
      the opinion domain is not in that whitelist.
    - Attack declares a ``requires_capability`` not in ``available_capabilities``.

    Soft realism weight is derived from whitelist breadth + capability fit.
    """
    excluded: List[str] = []
    notes: List[str] = []

    if attack_meta.incompatible_opinion_domains and opinion_meta.domain in attack_meta.incompatible_opinion_domains:
        excluded.append(
            f"Domain '{opinion_meta.domain}' is in incompatible_opinion_domains for attack '{attack_meta.leaf_label}'."
        )
    if attack_meta.compatible_opinion_domains and opinion_meta.domain not in attack_meta.compatible_opinion_domains:
        excluded.append(
            f"Domain '{opinion_meta.domain}' is not in compatible_opinion_domains for attack '{attack_meta.leaf_label}'."
        )

    available_set = set(available_capabilities)
    missing_caps = [cap for cap in attack_meta.requires_capability if cap not in available_set]
    if missing_caps:
        excluded.append(
            f"Attack '{attack_meta.leaf_label}' requires capabilities {missing_caps} not available in this run."
        )

    # Soft realism weight
    if attack_meta.compatible_opinion_domains:
        domain_breadth_factor = 0.85 + 0.15 * (1.0 / max(1, len(attack_meta.compatible_opinion_domains)))
    else:
        domain_breadth_factor = 1.0
    capability_factor = 1.0 if not attack_meta.requires_capability or not missing_caps else 0.0
    realism_weight = max(0.4, min(1.0, 0.7 * domain_breadth_factor + 0.3 * capability_factor))

    if attack_meta.complexity_tier in {"T3_synthetic", "T4_orchestrated", "T5_sustained"}:
        notes.append(f"Attack tier '{attack_meta.complexity_tier}' is high-complexity; treat as advanced-actor scenario.")
    if opinion_meta.adversarial_direction == 0:
        notes.append("Opinion leaf is direction-neutral (0); included for diversity, excluded from primary effectivity scoring.")

    return ScenarioAdmissibility(
        admissible=(len(excluded) == 0),
        realism_weight=realism_weight,
        excluded_reasons=excluded,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch APIs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScenarioCandidate:
    profile: ProfileConfiguration
    attack_leaf_path: str
    opinion_leaf_path: str
    admissibility: ScenarioAdmissibility


def filter_admissible_candidates(
    candidates: Sequence[ScenarioCandidate],
) -> List[ScenarioCandidate]:
    return [c for c in candidates if c.admissibility.admissible]


def realism_weighted_sample_indices(
    weights: Sequence[float],
    target_count: int,
    *,
    seed: int = 0,
) -> List[int]:
    import numpy as np

    n = len(weights)
    if n == 0 or target_count <= 0:
        return []
    if target_count >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    arr = np.asarray(weights, dtype=float).clip(min=1e-6)
    probs = arr / arr.sum()
    try:
        chosen = rng.choice(n, size=target_count, replace=False, p=probs)
    except ValueError:
        chosen = np.argsort(-arr)[:target_count]
    return sorted(int(i) for i in chosen)
