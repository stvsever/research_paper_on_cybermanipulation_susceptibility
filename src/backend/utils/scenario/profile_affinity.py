from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, List

from src.backend.utils.schemas import ProfileConfiguration


@dataclass(frozen=True)
class AffinityWeights:
    personality_similarity: float = 0.60
    ontology_leaf_overlap: float = 0.20
    age_context_similarity: float = 0.12
    categorical_similarity: float = 0.08

    def normalized(self) -> "AffinityWeights":
        total = (
            max(0.0, self.personality_similarity)
            + max(0.0, self.ontology_leaf_overlap)
            + max(0.0, self.age_context_similarity)
            + max(0.0, self.categorical_similarity)
        )
        if total <= 0.0:
            return DEFAULT_AFFINITY_WEIGHTS
        return AffinityWeights(
            personality_similarity=max(0.0, self.personality_similarity) / total,
            ontology_leaf_overlap=max(0.0, self.ontology_leaf_overlap) / total,
            age_context_similarity=max(0.0, self.age_context_similarity) / total,
            categorical_similarity=max(0.0, self.categorical_similarity) / total,
        )


DEFAULT_AFFINITY_WEIGHTS = AffinityWeights()


@dataclass(frozen=True)
class AffinityComponents:
    personality_similarity: float
    ontology_leaf_overlap: float
    age_context_similarity: float
    categorical_similarity: float

    def model_dump(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileAffinity:
    source: str
    target: str
    affinity: float
    components: AffinityComponents

    def other(self, profile_id: str) -> str:
        if profile_id == self.source:
            return self.target
        if profile_id == self.target:
            return self.source
        raise ValueError(f"Profile {profile_id} is not part of affinity pair {self.source}/{self.target}")

    def model_dump(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "affinity": self.affinity,
            "components": self.components.model_dump(),
        }


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _personality_key(key: str) -> bool:
    return key.lower().startswith("big_five_")


def _categorical_similarity(left: ProfileConfiguration, right: ProfileConfiguration) -> float:
    keys = sorted(set(left.categorical_attributes) | set(right.categorical_attributes))
    if not keys:
        return 0.0
    matches = 0
    comparable = 0
    for key in keys:
        if key in left.categorical_attributes and key in right.categorical_attributes:
            comparable += 1
            if str(left.categorical_attributes[key]) == str(right.categorical_attributes[key]):
                matches += 1
    return _clamp01(matches / comparable) if comparable else 0.0


def _continuous_ranges(profiles: list[ProfileConfiguration]) -> dict[str, tuple[float, float]]:
    keys = sorted({key for profile in profiles for key in profile.continuous_attributes})
    ranges: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = [
            float(profile.continuous_attributes[key])
            for profile in profiles
            if key in profile.continuous_attributes
        ]
        ranges[key] = (min(values), max(values)) if values else (0.0, 0.0)
    return ranges


def _continuous_similarity(
    left: ProfileConfiguration,
    right: ProfileConfiguration,
    ranges: dict[str, tuple[float, float]],
    *,
    personality: bool,
) -> float:
    keys = sorted(
        key
        for key in set(left.continuous_attributes) & set(right.continuous_attributes)
        if _personality_key(key) is personality
    )
    if not keys:
        return 0.0
    squared: list[float] = []
    for key in keys:
        low, high = ranges.get(key, (0.0, 0.0))
        span = high - low
        if span <= 0.0:
            squared.append(0.0)
            continue
        diff = (float(left.continuous_attributes[key]) - float(right.continuous_attributes[key])) / span
        squared.append(diff * diff)
    distance = math.sqrt(sum(squared) / len(squared))
    return _clamp01(1.0 - distance)


def _leaf_overlap(left: ProfileConfiguration, right: ProfileConfiguration) -> float:
    left_leaves = {str(value) for value in left.selected_leaf_nodes}
    right_leaves = {str(value) for value in right.selected_leaf_nodes}
    union = left_leaves | right_leaves
    if not union:
        return 0.0
    return _clamp01(len(left_leaves & right_leaves) / len(union))


def compute_profile_affinities(
    profiles: Iterable[ProfileConfiguration],
    weights: AffinityWeights = DEFAULT_AFFINITY_WEIGHTS,
) -> list[ProfileAffinity]:
    sorted_profiles = sorted(profiles, key=lambda profile: profile.profile_id)
    normalized_weights = weights.normalized()
    ranges = _continuous_ranges(sorted_profiles)
    affinities: list[ProfileAffinity] = []
    for left_index, left in enumerate(sorted_profiles):
        for right in sorted_profiles[left_index + 1 :]:
            components = AffinityComponents(
                personality_similarity=_continuous_similarity(left, right, ranges, personality=True),
                ontology_leaf_overlap=_leaf_overlap(left, right),
                age_context_similarity=_continuous_similarity(left, right, ranges, personality=False),
                categorical_similarity=_categorical_similarity(left, right),
            )
            affinity = _clamp01(
                components.personality_similarity * normalized_weights.personality_similarity
                + components.ontology_leaf_overlap * normalized_weights.ontology_leaf_overlap
                + components.age_context_similarity * normalized_weights.age_context_similarity
                + components.categorical_similarity * normalized_weights.categorical_similarity
            )
            affinities.append(
                ProfileAffinity(
                    source=left.profile_id,
                    target=right.profile_id,
                    affinity=affinity,
                    components=components,
                )
            )
    return affinities


def rank_profile_neighbors(
    profile_id: str,
    affinities: Iterable[ProfileAffinity],
    limit: int,
) -> list[ProfileAffinity]:
    pairs = [
        affinity
        for affinity in affinities
        if affinity.source == profile_id or affinity.target == profile_id
    ]
    pairs.sort(key=lambda item: (-item.affinity, item.source, item.target))
    return pairs[: max(0, limit)]


def affinity_weights_metadata(weights: AffinityWeights = DEFAULT_AFFINITY_WEIGHTS) -> dict[str, object]:
    normalized = weights.normalized()
    return {
        "formula": (
            "affinity = 0.60 personality + 0.20 ontology_overlap "
            "+ 0.12 age_context + 0.08 categorical"
        ),
        "weights": asdict(normalized),
        "notes": (
            "Categorical demographics are deliberately low-weighted to avoid demographic-only clustering."
        ),
    }
