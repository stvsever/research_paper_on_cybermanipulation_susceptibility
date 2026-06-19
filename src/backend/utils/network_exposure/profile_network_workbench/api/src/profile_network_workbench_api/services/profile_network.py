from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

from profile_network_workbench_api.backend_adapter import (
    WorkbenchNotFoundError,
    mode_from_config,
    ontology_root,
    opinion_options,
    pipeline_config_path_for_run,
    reconstruct_profile_bundles,
)
from profile_network_workbench_api.schemas import (
    AffinityComponents,
    AffinityFormulaComponent,
    AffinityFormulaMetadata,
    AffinityFormulaWeights,
    ProfileLayoutAffinity,
    ProfileNetworkDiagnostics,
    ProfileNetworkEdge,
    ProfileNetworkNode,
    ProfileNetworkProvenance,
    ProfileNetworkResponse,
)


DEFAULT_AFFINITY_WEIGHTS = AffinityFormulaWeights(
    personality_similarity=0.60,
    ontology_leaf_overlap=0.20,
    age_context_similarity=0.12,
    categorical_similarity=0.08,
)
CATEGORICAL_WEIGHT_WARNING = (
    "Categorical demographics are deliberately low-weighted to avoid demographic-only clustering."
)


@dataclass(frozen=True)
class _ProfileItem:
    profile_id: str
    categorical: dict[str, str]
    continuous: dict[str, float]
    leaves: set[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PairAffinity:
    source: str
    target: str
    affinity: float
    components: AffinityComponents


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    return _clamp01((value - low) / (high - low))


def _categorical_similarity(left: _ProfileItem, right: _ProfileItem) -> float:
    keys = sorted(set(left.categorical) | set(right.categorical))
    if not keys:
        return 0.0
    matches = 0
    comparable = 0
    for key in keys:
        if key in left.categorical and key in right.categorical:
            comparable += 1
            if left.categorical[key] == right.categorical[key]:
                matches += 1
    return _clamp01(matches / comparable) if comparable else 0.0


def _continuous_ranges(profiles: list[_ProfileItem]) -> dict[str, tuple[float, float]]:
    keys = sorted({key for profile in profiles for key in profile.continuous})
    ranges: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = [float(profile.continuous[key]) for profile in profiles if key in profile.continuous]
        ranges[key] = (min(values), max(values)) if values else (0.0, 0.0)
    return ranges


def _personality_key(key: str) -> bool:
    return key.lower().startswith("big_five_")


def _continuous_similarity(
    left: _ProfileItem,
    right: _ProfileItem,
    ranges: dict[str, tuple[float, float]],
    *,
    personality: bool,
) -> float:
    keys = sorted(
        key
        for key in set(left.continuous) & set(right.continuous)
        if _personality_key(key) is personality
    )
    if not keys:
        return 0.0
    squared: list[float] = []
    for key in keys:
        lo, hi = ranges.get(key, (0.0, 0.0))
        span = hi - lo
        if span <= 0:
            squared.append(0.0)
            continue
        diff = (float(left.continuous[key]) - float(right.continuous[key])) / span
        squared.append(diff * diff)
    distance = math.sqrt(sum(squared) / len(squared))
    return _clamp01(1.0 - distance)


def _leaf_overlap(left: _ProfileItem, right: _ProfileItem) -> float:
    union = left.leaves | right.leaves
    if not union:
        return 0.0
    return _clamp01(len(left.leaves & right.leaves) / len(union))


def _profile_items(profile_bundles: list[dict[str, Any]]) -> list[_ProfileItem]:
    items: list[_ProfileItem] = []
    for bundle in profile_bundles:
        profile = bundle["profile_result"].profile
        items.append(
            _ProfileItem(
                profile_id=str(profile.profile_id),
                categorical={str(key): str(value) for key, value in profile.categorical_attributes.items()},
                continuous={str(key): float(value) for key, value in profile.continuous_attributes.items()},
                leaves={str(item) for item in profile.selected_leaf_nodes},
                metadata={
                    **dict(profile.metadata or {}),
                    "candidate_index": int(bundle.get("candidate_index", 0)),
                    "candidate_seed": int(bundle.get("candidate_seed", 0)),
                },
            )
        )
    return sorted(items, key=lambda item: item.profile_id)


def _pair_affinities(profiles: list[_ProfileItem]) -> list[_PairAffinity]:
    ranges = _continuous_ranges(profiles)
    pairs: list[_PairAffinity] = []
    for left_index, left in enumerate(profiles):
        for right in profiles[left_index + 1 :]:
            components = AffinityComponents(
                categorical_similarity=_categorical_similarity(left, right),
                personality_similarity=_continuous_similarity(left, right, ranges, personality=True),
                age_context_similarity=_continuous_similarity(left, right, ranges, personality=False),
                ontology_leaf_overlap=_leaf_overlap(left, right),
            )
            affinity = _clamp01(
                components.personality_similarity * DEFAULT_AFFINITY_WEIGHTS.personality_similarity
                + components.ontology_leaf_overlap * DEFAULT_AFFINITY_WEIGHTS.ontology_leaf_overlap
                + components.age_context_similarity * DEFAULT_AFFINITY_WEIGHTS.age_context_similarity
                + components.categorical_similarity * DEFAULT_AFFINITY_WEIGHTS.categorical_similarity
            )
            pairs.append(
                _PairAffinity(
                    source=left.profile_id,
                    target=right.profile_id,
                    affinity=affinity,
                    components=components,
                )
            )
    return pairs


def _display_edges(pairs: list[_PairAffinity], edge_limit_per_node: int) -> list[_PairAffinity]:
    by_node: dict[str, list[_PairAffinity]] = {}
    for pair in pairs:
        by_node.setdefault(pair.source, []).append(pair)
        by_node.setdefault(pair.target, []).append(pair)
    selected: dict[tuple[str, str], _PairAffinity] = {}
    for node_pairs in by_node.values():
        for pair in sorted(node_pairs, key=lambda item: item.affinity, reverse=True)[:edge_limit_per_node]:
            key = tuple(sorted((pair.source, pair.target)))
            selected[key] = pair
    return sorted(selected.values(), key=lambda item: (item.source, item.target))


def _formula_metadata() -> AffinityFormulaMetadata:
    return AffinityFormulaMetadata(
        label=(
            "affinity = 0.60 personality traits + 0.20 ontology overlap "
            "+ 0.12 age/context + 0.08 categorical demographics"
        ),
        default_weights=DEFAULT_AFFINITY_WEIGHTS,
        components=[
            AffinityFormulaComponent(
                key="personality_similarity",
                label="Personality traits",
                description="Big Five continuous profile traits; dominant because susceptibility-relevant similarity should mostly reflect trait structure.",
            ),
            AffinityFormulaComponent(
                key="ontology_leaf_overlap",
                label="Ontology overlap",
                description="Shared selected ontology leaves from the reconstructed profile state.",
            ),
            AffinityFormulaComponent(
                key="age_context_similarity",
                label="Age/context",
                description="Chronological age and future non-personality continuous context variables.",
            ),
            AffinityFormulaComponent(
                key="categorical_similarity",
                label="Categorical demographics",
                description="Categorical profile matches such as sex; deliberately low by default to avoid demographic-only clusters.",
            ),
        ],
        warning=CATEGORICAL_WEIGHT_WARNING,
        note="Map updates use all pairwise profile affinities; visible edges are only the readable backbone.",
    )


def _centrality(profiles: list[_ProfileItem], pairs: list[_PairAffinity]) -> dict[str, float]:
    raw: dict[str, list[float]] = {profile.profile_id: [] for profile in profiles}
    for pair in pairs:
        raw[pair.source].append(pair.affinity)
        raw[pair.target].append(pair.affinity)
    means = {profile_id: (sum(values) / len(values) if values else 0.0) for profile_id, values in raw.items()}
    if not means:
        return {}
    lo = min(means.values())
    hi = max(means.values())
    return {profile_id: _normalize(value, lo, hi) for profile_id, value in means.items()}


def _cluster_ids(profiles: list[_ProfileItem], pairs: list[_PairAffinity]) -> dict[str, str]:
    """High-affinity connected components used only for visual grouping."""
    profile_ids = [profile.profile_id for profile in profiles]
    parent = {profile_id: profile_id for profile_id in profile_ids}
    affinities = [pair.affinity for pair in pairs]
    threshold = _percentile(affinities, 0.90) if affinities else 1.0

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair in pairs:
        if pair.affinity >= threshold:
            union(pair.source, pair.target)

    components: dict[str, list[str]] = {}
    for profile_id in profile_ids:
        components.setdefault(find(profile_id), []).append(profile_id)
    ordered = sorted((sorted(values) for values in components.values()), key=lambda values: (-len(values), values[0]))
    cluster_by_profile: dict[str, str] = {}
    for index, values in enumerate(ordered, start=1):
        cluster_id = f"cluster_{index:02d}"
        for profile_id in values:
            cluster_by_profile[profile_id] = cluster_id
    return cluster_by_profile


def get_profile_network(run_id: str = "run_1", edge_limit_per_node: int = 6) -> ProfileNetworkResponse:
    edge_limit = max(1, min(20, int(edge_limit_per_node)))
    payload, config, _, profile_bundles, opinions = reconstruct_profile_bundles(run_id)
    mode = mode_from_config(payload)
    profiles = _profile_items(profile_bundles)
    if not profiles:
        raise WorkbenchNotFoundError(f"No profiles reconstructed for {run_id}")

    pairs = _pair_affinities(profiles)
    displayed = _display_edges(pairs, edge_limit)
    affinities = [pair.affinity for pair in pairs]
    p10 = _percentile(affinities, 0.10)
    p95 = _percentile(affinities, 0.95)
    centrality = _centrality(profiles, pairs)
    clusters = _cluster_ids(profiles, pairs)

    nodes = [
        ProfileNetworkNode(
            id=profile.profile_id,
            label=profile.profile_id.replace("_", " "),
            cluster_id=clusters.get(profile.profile_id, "cluster_00"),
            categorical_attributes=profile.categorical,
            continuous_attributes=profile.continuous,
            selected_leaf_nodes=sorted(profile.leaves),
            metadata=profile.metadata,
            centrality=round(centrality.get(profile.profile_id, 0.0), 6),
        )
        for profile in profiles
    ]
    edges = [
        ProfileNetworkEdge(
            source=pair.source,
            target=pair.target,
            affinity=round(pair.affinity, 6),
            normalized_affinity=round(_normalize(pair.affinity, p10, p95), 6),
            components=pair.components,
        )
        for pair in displayed
    ]
    layout_affinities = [
        ProfileLayoutAffinity(
            source=pair.source,
            target=pair.target,
            affinity=round(pair.affinity, 6),
            components=pair.components,
        )
        for pair in pairs
    ]

    return ProfileNetworkResponse(
        run_id=run_id,
        mode=mode,
        nodes=nodes,
        edges=edges,
        layout_affinities=layout_affinities,
        affinity_formula=_formula_metadata(),
        opinion_leaves=opinion_options(opinions),
        diagnostics=ProfileNetworkDiagnostics(
            profile_count=len(nodes),
            full_pair_count=len(pairs),
            displayed_edge_count=len(edges),
            edge_limit_per_node=edge_limit,
            affinity_min=round(min(affinities), 6) if affinities else None,
            affinity_max=round(max(affinities), 6) if affinities else None,
            affinity_mean=round(mean(affinities), 6) if affinities else None,
        ),
        provenance=ProfileNetworkProvenance(
            run_id=run_id,
            mode=mode,
            config_path=str(pipeline_config_path_for_run(run_id)),
            ontology_root=str(ontology_root(mode)),
            seed=int(config.seed),
            model_name=str(payload.get("openrouter_model")) if payload.get("openrouter_model") else None,
        ),
        warnings=[CATEGORICAL_WEIGHT_WARNING],
    )
