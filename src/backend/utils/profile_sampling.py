from __future__ import annotations

"""
Technical overview
------------------
Builds pseudoprofiles from the hierarchical PROFILE ontology. Handles mixed
variable types:
  - Continuous trait+facet groups (personality dimensions with sub-facets)
  - Absolute numeric scalars (age, etc.)
  - Categorical single-selection (sex, education level, …)

Generalisation principle
------------------------
NO inventory names (Big Five, HEXACO, Dark Triad, …) are hardcoded here.
Structure is discovered entirely from the profile ontology leaf paths:

  Categorical dimension  — a parent node that appears as parts[-2] with 2–10
                           distinct leaf children AND whose name contains a
                           known demographic/categorical marker keyword.
  Continuous trait group — a parent node with ≥2 leaf children that is NOT a
                           categorical dimension; the inventory is inferred as
                           the grandparent node (parts[-3]).
  Absolute scalar        — a lone leaf whose first token matches a known
                           numeric-singleton list (age, income, …).

Column naming convention (mirrors the broader pipeline prefix convention):
  continuous trait facet : {inventory_key}_{trait_key}_{facet_key}_pct
  continuous trait mean  : {inventory_key}_{trait_key}_mean_pct
  absolute numeric       : {field_key}_{unit}   (e.g. age_years)
  categorical (stored)   : top-level key in categorical_attributes dict

Semantic labels
---------------
After sampling, a `semantic_description` metadata key is added to the returned
ProfileConfiguration containing a human/LLM-readable summary of all features,
generated via semantic_scale.fmt_profile().

Legacy helpers
--------------
heuristic_shift_sensitivity_proxy and resilience_index are still computed and
stored as continuous attributes but are excluded from the CSI ridge regression
(see LEGACY_EXCLUDED_FEATURE_COLUMNS in conditional_susceptibility.py).
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.backend.utils.ontology_utils import flatten_leaf_paths
from src.backend.utils.scenario_realism import (
    compute_resilience_index,
    compute_shift_sensitivity_proxy,
)
from src.backend.utils.schemas import ProfileConfiguration
from src.backend.utils.semantic_scale import fmt_profile


# ─────────────────────────────────────────────────────────────────────────────
# Categorical marker keywords
# ─────────────────────────────────────────────────────────────────────────────

# A parent node is treated as a categorical single-select dimension if its
# normalised name contains any of these tokens.  Extend freely — no other code
# changes needed to handle new categorical variables.
_CATEGORICAL_MARKERS: Set[str] = {
    "sex", "gender", "race", "ethnicity", "education",
    "religion", "nationality", "party", "marital",
    "category", "type", "group", "orientation",
}

# Absolute-numeric singletons: leaf tokens that signal a plain numeric measurement
# rather than a trait percentile.
_ABSOLUTE_SINGLETONS: Set[str] = {
    "age", "years", "income", "bmi", "weight", "height", "education_years",
}


# ─────────────────────────────────────────────────────────────────────────────
# Token helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_token(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _is_categorical_parent(name: str) -> bool:
    n = _normalize_token(name)
    return any(m in n for m in _CATEGORICAL_MARKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Ontology discovery
# ─────────────────────────────────────────────────────────────────────────────

def _extract_categorical_dimensions(
    profile_leaf_paths: List[str],
) -> Dict[str, List[str]]:
    """
    Discover all categorical single-select dimensions from the profile ontology.

    Returns {dimension_label: [level1, level2, …]} for every parent node that:
      - contains a categorical marker keyword (sex, gender, race, …), AND
      - has between 2 and 10 distinct leaf children.

    Example:
      "PERSON > … > Sex > Male"  +  "… > Sex > Female"  +  "… > Sex > Other"
      → {"Sex": ["Female", "Male", "Other"]}
    """
    parent_children: Dict[str, Set[str]] = defaultdict(set)
    for path in profile_leaf_paths:
        parts = [p.strip() for p in path.split(">")]
        if len(parts) >= 2:
            parent_children[parts[-2]].add(parts[-1])

    result: Dict[str, List[str]] = {}
    for parent, children in parent_children.items():
        if _is_categorical_parent(parent) and 2 <= len(children) <= 10:
            result[parent] = sorted(children)
    return result


def _extract_continuous_trait_structure(
    profile_leaf_paths: List[str],
) -> Dict[Tuple[str, str], List[str]]:
    """
    Discover all continuous trait+facet groups from the profile ontology.

    Returns {(inventory_key, trait_key): [facet_key, …]} for every parent node that:
      - is NOT a categorical dimension, AND
      - has ≥2 distinct leaf children (facets).

    The inventory_key is inferred as the normalised grandparent node (parts[-3]).
    If the grandparent is ambiguous / absent, falls back to "other".

    Example:
      "PERSON > … > Personality > Big_Five > Neuroticism > Anxiety"
      → {("big_five", "neuroticism"): ["anger_hostility", "anxiety", "depression", …]}
    """
    parent_children: Dict[str, Set[str]] = defaultdict(set)
    grandparent_for: Dict[str, str] = {}  # parent → first grandparent seen

    for path in profile_leaf_paths:
        parts = [p.strip() for p in path.split(">")]
        if len(parts) >= 2:
            parent = parts[-2]
            parent_children[parent].add(parts[-1])
            if parent not in grandparent_for and len(parts) >= 3:
                grandparent_for[parent] = parts[-3]

    result: Dict[Tuple[str, str], List[str]] = {}
    for parent, children in parent_children.items():
        if _is_categorical_parent(parent):
            continue
        if len(children) < 2:
            continue
        # Skip if all children look like absolute-numeric singletons
        if all(
            any(tok in _normalize_token(c) for tok in _ABSOLUTE_SINGLETONS)
            for c in children
        ):
            continue
        inv = _normalize_token(grandparent_for.get(parent, "other"))
        trait = _normalize_token(parent)
        facets = [_normalize_token(f) for f in sorted(children)]
        result[(inv, trait)] = facets

    return result


def _extract_absolute_scalars(
    profile_leaf_paths: List[str],
    already_assigned: Set[str],
) -> Dict[str, str]:
    """
    Discover absolute-numeric scalar fields (age_years, income, …).
    Returns {column_key: leaf_label}, e.g. {"age_years": "Age_Years"}.
    Skips leaf paths whose last part's normalised tokens overlap a
    trait-facet or categorical group already detected.
    """
    result: Dict[str, str] = {}
    for path in profile_leaf_paths:
        parts = [p.strip() for p in path.split(">")]
        leaf = parts[-1]
        leaf_norm = _normalize_token(leaf)
        if leaf_norm in already_assigned:
            continue
        if any(tok in leaf_norm for tok in _ABSOLUTE_SINGLETONS):
            result[leaf_norm] = leaf
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic profile generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfileSamplingResult:
    profile: ProfileConfiguration
    sampling_mode_used: str


def deterministic_profile(
    profile_leaf_paths: List[str],
    profile_id: str,
    seed: int,
) -> ProfileConfiguration:
    """
    Build a coherent pseudoprofile from the PROFILE ontology leaf paths.

    Variable types are discovered structurally from the ontology — no inventory
    names (Big Five, HEXACO, …) or demographic field names (Sex, Age) are
    hardcoded here.  Adding new personality inventories or demographic variables
    to the ontology JSON automatically propagates to profile sampling.

    Returns a ProfileConfiguration with:
      - categorical_attributes : {dim_label_lower: level_value}
      - continuous_attributes  : {column_key: float_value}
      - metadata["semantic_description"] : LLM-readable profile summary
    """
    rng = random.Random(seed)

    # ── Discover variable types from ontology ────────────────────────────────
    cat_dims = _extract_categorical_dimensions(profile_leaf_paths)
    trait_structure = _extract_continuous_trait_structure(profile_leaf_paths)

    # Build set of leaf norms already assigned to trait/cat groups
    assigned: Set[str] = set()
    for parent, children in cat_dims.items():
        assigned.update(_normalize_token(c) for c in children)
    for (inv, trait), facets in trait_structure.items():
        assigned.update(facets)

    scalar_fields = _extract_absolute_scalars(profile_leaf_paths, assigned)

    # ── Sample categorical attributes ────────────────────────────────────────
    categorical_attributes: Dict[str, str] = {}
    for dim_label, options in sorted(cat_dims.items()):
        key = _normalize_token(dim_label)
        categorical_attributes[key] = rng.choice(options)

    # ── Sample continuous attributes ─────────────────────────────────────────
    continuous_attributes: Dict[str, float] = {}

    # Absolute-numeric scalars — custom sampling per known field type
    for field_key, _leaf_label in sorted(scalar_fields.items()):
        if "age" in field_key:
            val = float(max(18, min(85, int(rng.gauss(42, 14)))))
        elif "income" in field_key:
            val = float(max(0, int(rng.gauss(55_000, 25_000))))
        elif "bmi" in field_key:
            val = round(max(15.0, min(45.0, rng.gauss(24.5, 4.0))), 1)
        else:
            val = float(max(0, rng.gauss(50, 20)))
        continuous_attributes[field_key] = val

    # Trait+facet groups — betavariate anchor + Gaussian facet perturbation
    for (inv_key, trait_key), facets in sorted(trait_structure.items()):
        trait_anchor = max(0.0, min(100.0, rng.betavariate(2.1, 2.1) * 100.0))
        facet_values: List[float] = []
        for facet_key in facets:
            fval = max(0.0, min(100.0, trait_anchor + rng.gauss(0.0, 8.5)))
            facet_values.append(fval)
            col = f"{inv_key}_{trait_key}_{facet_key}_pct"
            continuous_attributes[col] = round(fval, 3)
        if facet_values:
            mean_col = f"{inv_key}_{trait_key}_mean_pct"
            continuous_attributes[mean_col] = round(
                sum(facet_values) / len(facet_values), 3
            )

    # ── Build ProfileConfiguration (needed for legacy realism helpers) ───────
    tmp_profile = ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=categorical_attributes,
        continuous_attributes=continuous_attributes,
        selected_leaf_nodes=[],
    )
    continuous_attributes["heuristic_shift_sensitivity_proxy"] = (
        compute_shift_sensitivity_proxy(tmp_profile)
    )
    continuous_attributes["resilience_index"] = compute_resilience_index(tmp_profile)

    # ── Selected leaf nodes — all paths that contributed a variable ──────────
    # Detect by checking whether the leaf or parent appears in our discovered vars
    cat_parents_lower = {_normalize_token(p) for p in cat_dims}
    trait_parents_lower = {_normalize_token(trait) for (_, trait) in trait_structure}
    scalar_keys_lower = set(scalar_fields.keys())

    selected_leaf_nodes: List[str] = []
    for path in profile_leaf_paths:
        parts = [p.strip() for p in path.split(">")]
        parent_norm = _normalize_token(parts[-2]) if len(parts) >= 2 else ""
        leaf_norm = _normalize_token(parts[-1])
        if (
            parent_norm in cat_parents_lower
            or parent_norm in trait_parents_lower
            or leaf_norm in scalar_keys_lower
        ):
            selected_leaf_nodes.append(path)

    # ── Semantic description for LLM prompt context ──────────────────────────
    semantic_desc = fmt_profile(
        continuous=continuous_attributes,
        categorical=categorical_attributes,
        include_facets=False,
        verbose=True,
    )

    return ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=categorical_attributes,
        continuous_attributes=continuous_attributes,
        selected_leaf_nodes=selected_leaf_nodes,
        metadata={
            "generation": "deterministic",
            "semantic_description": semantic_desc,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def sample_profile(
    profile_tree: Dict[str, dict],
    profile_id: str,
    seed: int,
    generation_mode: str,
    llm_generator: Optional[
        Callable[
            [str, int, List[str], ProfileConfiguration],
            Optional[ProfileConfiguration],
        ]
    ] = None,
) -> ProfileSamplingResult:
    profile_leaf_paths = flatten_leaf_paths(profile_tree)

    deterministic = deterministic_profile(
        profile_leaf_paths=profile_leaf_paths,
        profile_id=profile_id,
        seed=seed,
    )

    mode = generation_mode.lower()

    if mode == "deterministic":
        return ProfileSamplingResult(
            profile=deterministic,
            sampling_mode_used="deterministic",
        )

    if llm_generator is None:
        return ProfileSamplingResult(
            profile=deterministic,
            sampling_mode_used="deterministic_fallback_no_llm_generator",
        )

    llm_result = llm_generator(profile_id, seed, profile_leaf_paths, deterministic)
    if llm_result is None:
        return ProfileSamplingResult(
            profile=deterministic,
            sampling_mode_used=f"{mode}_fallback_deterministic",
        )

    merged_continuous = dict(deterministic.continuous_attributes)
    merged_continuous.update(llm_result.continuous_attributes)

    merged_categorical = dict(deterministic.categorical_attributes)
    merged_categorical.update(llm_result.categorical_attributes)

    # Recompute legacy realism helpers on the merged profile
    merged_tmp = ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=merged_categorical,
        continuous_attributes=merged_continuous,
        selected_leaf_nodes=deterministic.selected_leaf_nodes,
    )
    merged_continuous["heuristic_shift_sensitivity_proxy"] = (
        compute_shift_sensitivity_proxy(merged_tmp)
    )
    merged_continuous["resilience_index"] = compute_resilience_index(merged_tmp)

    # Recompute semantic description with merged values
    semantic_desc = fmt_profile(
        continuous=merged_continuous,
        categorical=merged_categorical,
        include_facets=False,
        verbose=True,
    )

    merged = ProfileConfiguration(
        profile_id=profile_id,
        categorical_attributes=merged_categorical,
        continuous_attributes=merged_continuous,
        selected_leaf_nodes=deterministic.selected_leaf_nodes,
        metadata={
            "generation": mode,
            "deterministic_seed": seed,
            "llm_profile_adjusted": True,
            "semantic_description": semantic_desc,
        },
    )
    return ProfileSamplingResult(profile=merged, sampling_mode_used=mode)
