#!/usr/bin/env python3
"""
Builder for the production PROFILE ontology (current design).

Design principles
-----------------
The production PROFILE ontology is a FULL hierarchical state space of
person attributes (administrative metadata, demographics, identity,
political profile, personality, etc.) used as the P-axis of the
(P x A x O) factorial simulation design.

Per-leaf STRUCTURAL sampling metadata
-------------------------------------
Every leaf carries an inline metadata block describing only STRUCTURAL
sampling constraints — what kind of value the leaf represents, which
sibling leaves are mutually exclusive options of the same construct,
how prevalent it is in the population, and what construct type it
belongs to. The fields are:

- ``modality_type``      : categorical | binary | ordinal | continuous | range | identifier | composite
- ``value_format``       : string | int | float | iso_date | range_str | enum
- ``sampling_role``      : option_value | trait_state | unknown_marker | identifier |
                            scale_dimension | scale_facet | container_metadata
- ``exclusivity_group``  : parent group key whose direct children form a
                            mutually-exclusive option set; empty if not applicable
- ``prevalence_weight``  : relative population frequency for sampling (default 1.0)
- ``is_unknown_marker``  : True for ``Unknown``, ``Prefer_Not_To_Say``, ``Other``
- ``construct_type``     : administrative | demographic | political | personality |
                            psychological | clinical | contextual | behavioural |
                            identity | linguistic | cognitive | preference
- ``description``        : one-line auto-derived from the leaf label

The ontology does NOT encode psychological-amplification hypotheses
about which traits make profiles susceptible to which attacks — those
are exactly what the inferential layer estimates.

Subtree-wide sampling rules live in the top-level ``_sampling_rules``
metanode (path-glob patterns); per-leaf metadata always overrides.

This script reads the existing PROFILE JSON, INFERS per-leaf metadata
from path patterns and leaf labels, writes the enriched file back, and
prints stats.

Run:

    python3 -m src.backend.ontology.builders.build_production_profile \
        src/backend/ontology/01_separated/production/PROFILE/profile.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Local leaf walker (bypasses backend deps so the builder runs in a thin env).
# ---------------------------------------------------------------------------
_LOCAL_METADATA_KEYS = frozenset(
    {
        "adversarial_direction",
        "description",
        "notes",
        "examples",
    }
)


def _is_local_metadata_key(key: str) -> bool:
    return key.startswith("_") or key in _LOCAL_METADATA_KEYS or not key[0].isupper()


def _is_local_leaf(child: Any) -> bool:
    if not isinstance(child, dict):
        return True
    if not child:
        return True
    return all(_is_local_metadata_key(k) for k in child)


def _local_leaf_paths(tree: Dict[str, Any], prefix: List[str] | None = None) -> List[List[str]]:
    prefix = prefix or []
    leaves: List[List[str]] = []
    for key, child in tree.items():
        if _is_local_metadata_key(key):
            continue
        path = [*prefix, key]
        if _is_local_leaf(child):
            leaves.append(path)
        else:
            leaves.extend(_local_leaf_paths(child, path))
    return leaves


# ---------------------------------------------------------------------------
# Path-glob matcher (same semantics as ATTACK / OPINION engines).
# ---------------------------------------------------------------------------


def _split_pattern(pattern: str) -> List[str]:
    return [p.strip() for p in pattern.replace("**", " ** ").split(" ** ")]


def _split_path(path: str) -> List[str]:
    return [p.strip() for p in path.split(" > ") if p.strip()]


def _path_pattern_matches(pattern: str, segments: List[str]) -> bool:
    parts = [p for p in _split_pattern(pattern) if p]
    if not parts:
        return True
    cursor = 0
    for needle in parts:
        sub = _split_path(needle)
        if not sub:
            continue
        found = False
        while cursor + len(sub) <= len(segments):
            if segments[cursor : cursor + len(sub)] == sub:
                cursor += len(sub)
                found = True
                break
            cursor += 1
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Subtree-wide sampling rules.
# Each rule applies attributes to every leaf whose path matches one of
# ``applies_to_profile_paths``. Per-leaf inferences (label-based) refine
# rule-derived defaults; per-leaf inline metadata always wins.
# ---------------------------------------------------------------------------


SAMPLING_RULES_BLOCK: Dict[str, Any] = {
    "schema_version": "v4-test-run-1-meta",
    "rule_evaluation": "ordered_merge_last_wins_per_attribute",
    "pattern_wildcard": "**",
    "pattern_separator": " > ",
    "_documentation": (
        "Path-glob rules apply STRUCTURAL sampling attributes to subtrees of "
        "the PROFILE ontology. A leaf path matches a pattern when every "
        "literal segment in the pattern appears in the same order in the "
        "leaf's `>`-joined path; segments 01_separated by `**` match any number "
        "of intervening segments. Rules are evaluated in order; LAST "
        "matching rule wins per scalar attribute. Per-leaf inline metadata "
        "always takes precedence over rule-derived attributes."
    ),
    "rules": [
        # --- Administrative ----------------------------------------------------
        {
            "rule_id": "administrative_construct_type",
            "applies_to_profile_paths": ["**Administrative_and_Data_Context**"],
            "construct_type": "administrative",
        },
        {
            "rule_id": "identifier_subtree",
            "applies_to_profile_paths": [
                "**Identifiers**",
                "**Person_Record_ID**",
                "**Study_Participant_ID**",
                "**Pseudonymous_ID**",
                "**Record_Linkage_Key**",
                "**External_System_ID**",
                "**Family_Unit_ID**",
                "**Household_ID**",
            ],
            "modality_type": "identifier",
            "value_format": "string",
            "sampling_role": "identifier",
        },
        # --- Demographics ------------------------------------------------------
        {
            "rule_id": "demographics_construct_type",
            "applies_to_profile_paths": ["**Demographics_and_Identity**"],
            "construct_type": "demographic",
        },
        {
            "rule_id": "vital_status_options",
            "applies_to_profile_paths": ["**Vital_Status > Status**"],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "exclusivity_group": "Vital_Status",
        },
        {
            "rule_id": "age_continuous_fields",
            "applies_to_profile_paths": [
                "**Chronological_Age > Age_Years**",
                "**Chronological_Age > Age_Months**",
                "**Chronological_Age > Age_Days**",
                "**Perinatal_and_Corrected_Age > Gestational_Age_At_Birth**",
                "**Perinatal_and_Corrected_Age > Corrected_Age_For_Prematurity**",
                "**Perinatal_and_Corrected_Age > Postmenstrual_Age**",
            ],
            "modality_type": "continuous",
            "value_format": "int",
            "sampling_role": "trait_state",
        },
        {
            "rule_id": "age_band_categorical",
            "applies_to_profile_paths": ["**Age_Bands**"],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
        },
        {
            "rule_id": "developmental_stage_categorical",
            "applies_to_profile_paths": ["**Developmental_Stage**"],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "construct_type": "psychological",
        },
        {
            "rule_id": "date_of_birth_iso",
            "applies_to_profile_paths": [
                "**Date_Of_Birth**",
                "**Year_Of_Birth**",
                "**Date_Of_Death**",
                "**Year_Of_Death**",
                "**Data_Collection_Timestamp**",
            ],
            "modality_type": "continuous",
            "value_format": "iso_date",
            "sampling_role": "trait_state",
        },
        # --- Political profile -------------------------------------------------
        {
            "rule_id": "political_profile_construct_type",
            "applies_to_profile_paths": ["**Political_Profile**"],
            "construct_type": "political",
            "modality_type": "continuous",
            "value_format": "float",
            "sampling_role": "scale_dimension",
        },
        {
            "rule_id": "issue_position_taxonomy_role",
            "applies_to_profile_paths": [
                "**Political_Profile > Issue_Position_Taxonomy**",
            ],
            "construct_type": "political",
            "modality_type": "ordinal",
            "value_format": "float",
            "sampling_role": "scale_dimension",
        },
        {
            "rule_id": "political_participation_role",
            "applies_to_profile_paths": [
                "**Political_Profile > Political_Participation_Taxonomy**",
            ],
            "construct_type": "behavioural",
            "modality_type": "ordinal",
            "value_format": "float",
            "sampling_role": "scale_dimension",
        },
        # --- Personality -------------------------------------------------------
        {
            "rule_id": "personality_construct_type",
            "applies_to_profile_paths": ["**Personality**"],
            "construct_type": "personality",
            "modality_type": "continuous",
            "value_format": "float",
            "sampling_role": "scale_facet",
        },
        {
            "rule_id": "big_five_facets",
            "applies_to_profile_paths": [
                "**Personality > Big_Five**",
                "**Personality > HEXACO**",
                "**Personality > Eysenck_PEN_Model**",
                "**Personality > Hexad_User_Types_Model**",
            ],
            "construct_type": "personality",
            "modality_type": "continuous",
            "value_format": "float",
            "sampling_role": "scale_facet",
        },
        # --- Identity & cognition ---------------------------------------------
        {
            "rule_id": "cognition_construct_type",
            "applies_to_profile_paths": [
                "**Cognition_and_Reasoning**",
                "**Cognitive_Style**",
                "**Need_for_Cognition**",
            ],
            "construct_type": "cognitive",
            "modality_type": "continuous",
            "value_format": "float",
            "sampling_role": "scale_dimension",
        },
        # --- Communication & language -----------------------------------------
        {
            "rule_id": "language_categorical",
            "applies_to_profile_paths": [
                "**Preferred_Language**",
                "**Native_Language**",
                "**Spoken_Languages**",
            ],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "construct_type": "linguistic",
            "exclusivity_group": "Language_Choice",
        },
        # --- Preferences -------------------------------------------------------
        {
            "rule_id": "preferences_construct_type",
            "applies_to_profile_paths": [
                "**Preferences**",
                "**Communication_Preferences**",
                "**Preferred_Channel**",
            ],
            "construct_type": "preference",
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
        },
        # --- Generic categorical leaves ---------------------------------------
        {
            "rule_id": "consent_options",
            "applies_to_profile_paths": ["**Consent**"],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "construct_type": "administrative",
        },
        {
            "rule_id": "data_quality_options",
            "applies_to_profile_paths": ["**Data_Quality**"],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "construct_type": "administrative",
        },
        {
            "rule_id": "respondent_options",
            "applies_to_profile_paths": ["**Respondent**"],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "construct_type": "administrative",
            "exclusivity_group": "Respondent",
        },
        {
            "rule_id": "data_source_options",
            "applies_to_profile_paths": [
                "**Data_Source**",
                "**Collection_Mode**",
            ],
            "modality_type": "categorical",
            "value_format": "enum",
            "sampling_role": "option_value",
            "construct_type": "administrative",
        },
    ],
    "modality_type_definitions": {
        "categorical": "Discrete unordered set of mutually exclusive options.",
        "binary": "Two-option categorical (yes/no, present/absent).",
        "ordinal": "Discrete ordered categories (e.g. Likert 1..5).",
        "continuous": "Continuous numeric value (real-valued).",
        "range": "Numeric range encoded as a string or pair.",
        "identifier": "Free-form ID string (no statistical interpretation).",
        "composite": "Container whose direct children jointly describe the construct.",
    },
    "sampling_role_definitions": {
        "option_value": "A single mutually-exclusive option under an exclusivity group.",
        "trait_state": "A state value (continuous or ordinal) for a trait/dimension.",
        "unknown_marker": "Sentinel option representing missing / withheld information.",
        "identifier": "Identifier string with no statistical role.",
        "scale_dimension": "Top-level dimension of a multi-item psychometric scale.",
        "scale_facet": "Facet / sub-dimension of a multi-item psychometric scale.",
        "container_metadata": "Container leaf whose value is a metadata sub-block.",
    },
    "construct_type_definitions": {
        "administrative": "Process / record-keeping metadata.",
        "demographic": "Population-descriptor variables (age, sex, location).",
        "political": "Political ideology, issue positions, partisan attachments.",
        "personality": "Trait and facet dimensions of personality.",
        "psychological": "Cognitive, motivational, affective constructs.",
        "clinical": "Health / clinical status indicators.",
        "contextual": "Environment, situation, life-event variables.",
        "behavioural": "Action / participation variables.",
        "identity": "Self-categorisation and group attachments.",
        "linguistic": "Language / communication variables.",
        "cognitive": "Cognitive style / reasoning variables.",
        "preference": "Preference / choice variables.",
    },
}


# ---------------------------------------------------------------------------
# Label-based heuristics
# ---------------------------------------------------------------------------


_UNKNOWN_LABELS = frozenset(
    {
        "Unknown",
        "Prefer_Not_To_Say",
        "Other",
        "Not_Recorded",
        "Missing",
        "Refused",
        "Declined",
    }
)


_CONTINUOUS_NUMERIC_LABELS = frozenset(
    {
        "Age_Years",
        "Age_Months",
        "Age_Days",
        "Gestational_Age_At_Birth",
        "Corrected_Age_For_Prematurity",
        "Postmenstrual_Age",
    }
)


_DATE_LABELS = frozenset(
    {
        "Date_Of_Birth",
        "Year_Of_Birth",
        "Date_Of_Death",
        "Year_Of_Death",
        "Date_Of_Event",
        "Year_Of_Event",
        "Data_Collection_Timestamp",
    }
)


_HUMAN_PHRASE_OVERRIDES = {
    "Ai": "AI",
    "Llm": "LLM",
    "Id": "ID",
    "Pii": "PII",
    "Pen": "PEN",
    "Hexaco": "HEXACO",
    "Lgbtq": "LGBTQ",
    "Tan": "TAN",
    "Gal": "GAL",
}


def _humanise(label: str) -> str:
    parts = [p for p in label.replace("__", "_").split("_") if p]
    out: List[str] = []
    for p in parts:
        if p in _HUMAN_PHRASE_OVERRIDES:
            out.append(_HUMAN_PHRASE_OVERRIDES[p])
        else:
            out.append(p.lower())
    return " ".join(out)


def _label_description(segments: List[str]) -> str:
    if not segments:
        return ""
    leaf = _humanise(segments[-1])
    parent = _humanise(segments[-2]) if len(segments) >= 2 else ""
    if parent:
        return f"{leaf} (within {parent})"
    return leaf


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def _apply_rules_to_segments(
    rules: List[Dict[str, Any]],
    segments: List[str],
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for rule in rules:
        patterns = rule.get("applies_to_profile_paths", [])
        if not isinstance(patterns, list):
            continue
        if not any(_path_pattern_matches(str(p), segments) for p in patterns):
            continue
        for k in (
            "modality_type",
            "value_format",
            "sampling_role",
            "exclusivity_group",
            "construct_type",
        ):
            if k in rule:
                resolved[k] = rule[k]
        if "prevalence_weight" in rule:
            try:
                resolved["prevalence_weight"] = float(rule["prevalence_weight"])
            except (TypeError, ValueError):
                pass
    return resolved


def _infer_exclusivity_group(segments: List[str]) -> str:
    """If the leaf is one of several siblings under a non-fallback parent, the
    parent name is the exclusivity group. Otherwise empty string.
    """
    if len(segments) < 2:
        return ""
    parent = segments[-2]
    if parent in {"PERSON", "Demographics_and_Identity", "Administrative_and_Data_Context"}:
        return ""
    return parent


def _bake_leaf_metadata(
    rules: List[Dict[str, Any]],
    segments: List[str],
) -> Dict[str, Any]:
    leaf_label = segments[-1] if segments else ""
    rule_attrs = _apply_rules_to_segments(rules, segments)

    # Defaults
    modality_type = rule_attrs.get("modality_type") or "categorical"
    value_format = rule_attrs.get("value_format") or "enum"
    sampling_role = rule_attrs.get("sampling_role") or "option_value"
    construct_type = rule_attrs.get("construct_type") or "demographic"
    exclusivity_group = rule_attrs.get("exclusivity_group") or _infer_exclusivity_group(segments)
    prevalence_weight = float(rule_attrs.get("prevalence_weight", 1.0))

    is_unknown_marker = leaf_label in _UNKNOWN_LABELS

    # Label-based refinement
    if leaf_label in _CONTINUOUS_NUMERIC_LABELS:
        modality_type = "continuous"
        value_format = "int"
        sampling_role = "trait_state"
    elif leaf_label in _DATE_LABELS:
        modality_type = "continuous"
        value_format = "iso_date"
        sampling_role = "trait_state"

    if is_unknown_marker:
        sampling_role = "unknown_marker"
        # Unknown markers are rarer in real samples than substantive options
        if "prevalence_weight" not in rule_attrs:
            prevalence_weight = 0.5

    return {
        "modality_type": modality_type,
        "value_format": value_format,
        "sampling_role": sampling_role,
        "exclusivity_group": exclusivity_group,
        "prevalence_weight": prevalence_weight,
        "is_unknown_marker": is_unknown_marker,
        "construct_type": construct_type,
        "description": _label_description(segments),
    }


# ---------------------------------------------------------------------------
# Top-level baker
# ---------------------------------------------------------------------------


def _walk_to_leaf(tree: Dict[str, Any], segments: List[str]) -> Optional[Dict[str, Any]]:
    node: Any = tree
    for seg in segments:
        if isinstance(node, dict) and seg in node:
            node = node[seg]
        else:
            return None
    return node if isinstance(node, dict) else None


def bake_profile_ontology(tree: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Walk the profile tree, bake per-leaf metadata, return (tree, count)."""
    # Install or refresh the sampling-rules metanode at the top.
    tree["_sampling_rules"] = SAMPLING_RULES_BLOCK
    rules = SAMPLING_RULES_BLOCK["rules"]

    leaf_segments_list = _local_leaf_paths(tree)
    enriched = 0
    construct_type_counts: Dict[str, int] = {}
    modality_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    for segments in leaf_segments_list:
        node = _walk_to_leaf(tree, segments)
        if node is None:
            continue
        meta = _bake_leaf_metadata(rules, segments)
        # Strip any prior metadata that we control; keep custom fields
        for key in (
            "modality_type",
            "value_format",
            "sampling_role",
            "exclusivity_group",
            "prevalence_weight",
            "is_unknown_marker",
            "construct_type",
            "description",
        ):
            node.pop(key, None)
        node.update(meta)
        enriched += 1
        construct_type_counts[meta["construct_type"]] = (
            construct_type_counts.get(meta["construct_type"], 0) + 1
        )
        modality_counts[meta["modality_type"]] = (
            modality_counts.get(meta["modality_type"], 0) + 1
        )
        role_counts[meta["sampling_role"]] = role_counts.get(meta["sampling_role"], 0) + 1

    metadata = tree.get("_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("schema_version", "v4-test-run-1-production")
    metadata.setdefault("ontology_role", "deployment")
    metadata.setdefault("title", "Person / Profile Ontology — Deployment State Space")
    metadata["per_leaf_metadata_schema"] = {
        "schema_version": "v4-test-run-1-baked",
        "documentation": (
            "Every PROFILE leaf carries STRUCTURAL combinatorial sampling "
            "metadata (modality_type, value_format, sampling_role, "
            "exclusivity_group, prevalence_weight, is_unknown_marker, "
            "construct_type, description). The ontology does NOT encode "
            "psychological-amplification hypotheses. Subtree-wide rules "
            "live in `_sampling_rules`; per-leaf inline metadata wins."
        ),
        "fields": [
            "modality_type",
            "value_format",
            "sampling_role",
            "exclusivity_group",
            "prevalence_weight",
            "is_unknown_marker",
            "construct_type",
            "description",
        ],
    }
    metadata["stats"] = {
        "leaf_count_total": enriched,
        "construct_type_counts": construct_type_counts,
        "modality_type_counts": modality_counts,
        "sampling_role_counts": role_counts,
        "sampling_rule_count": len(rules),
    }
    tree["_metadata"] = metadata
    return tree, enriched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: build_production_profile.py <profile_json_path>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"profile json not found: {path}", file=sys.stderr)
        sys.exit(2)
    tree = json.loads(path.read_text(encoding="utf-8"))
    tree, enriched = bake_profile_ontology(tree)
    path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    stats = tree.get("_metadata", {}).get("stats", {})
    print(
        json.dumps(
            {
                "out": str(path),
                "enriched_leaves": enriched,
                "stats": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
