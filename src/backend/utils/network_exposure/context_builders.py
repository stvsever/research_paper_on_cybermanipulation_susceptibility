from __future__ import annotations

import statistics
from typing import Any

from .defaults import DEFAULT_MAX_EXEMPLARS, EDGE_FILE_FULL
from .loader import load_edge_index
from .neighborhoods import build_incoming_exposure_neighborhood
from .schemas import (
    ExposureEdgeIndex,
    ExposureNetworkPackage,
    NetworkContextConfig,
    ProfilePositionAssignment,
)


def _assessment_field(assessment: object, field: str, default: Any = None) -> Any:
    if isinstance(assessment, dict):
        return assessment.get(field, default)
    return getattr(assessment, field, default)


def _score(assessment: object) -> int:
    return int(_assessment_field(assessment, "score", 0))


def _confidence(assessment: object) -> float:
    return float(_assessment_field(assessment, "confidence", 0.0) or 0.0)


def _reasoning(assessment: object) -> str:
    return str(_assessment_field(assessment, "reasoning", "") or "")


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    total = sum(weights)
    if not values or total <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _base_context(
    *,
    target_profile_id: str,
    assignments_by_profile: dict[str, ProfilePositionAssignment],
    package: ExposureNetworkPackage,
    peer_payloads_by_profile: dict[str, dict[str, object]],
    config: NetworkContextConfig,
    edge_index: ExposureEdgeIndex | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target_assignment = assignments_by_profile[target_profile_id]
    assigned_positions = {assignment.position_id for assignment in assignments_by_profile.values()}
    index = edge_index or load_edge_index(
        package,
        edge_file=EDGE_FILE_FULL,
        target_positions={target_assignment.position_id},
        source_positions=assigned_positions,
    )
    neighborhood = build_incoming_exposure_neighborhood(
        target_profile_id,
        assignments_by_profile,
        package,
        edge_index=index,
        peer_payloads_by_profile=peer_payloads_by_profile,
    )
    scored_neighbor_profiles = set(peer_payloads_by_profile)
    full_peers = [peer.to_context() for peer in neighborhood.incoming_peers]
    peers = [
        peer.to_context()
        for peer in neighborhood.incoming_peers
        if peer.profile_id in scored_neighbor_profiles
    ]
    exemplars = peers[: max(0, int(config.max_exemplars))]
    scored_weight = sum(float(peer["exposure_weight"]) for peer in peers)
    context = {
        "network_basis": "empirical_politisky24_bluesky_exposure",
        "graph_id": package.graph_id,
        "edge_direction": package.manifest.get("edge_direction"),
        "edge_meaning": package.manifest.get("edge_meaning"),
        "peer_selection": "all_assigned_incoming_empirical_exposure_edges",
        "exemplar_selection": "highest_ranked_incoming_exposure_edges",
        "target_profile_id": target_profile_id,
        "target_position_id": target_assignment.position_id,
        "target_network_position": target_assignment.to_context(),
        "full_incoming_peer_count": len(full_peers),
        "full_incoming_exposure_weight": _round(neighborhood.total_exposure_weight),
        "peer_count": len(peers),
        "exemplar_count": len(exemplars),
        "max_exemplars": int(config.max_exemplars),
        "total_exposure_weight": _round(scored_weight),
    }
    return context, peers, exemplars


def build_baseline_network_context(
    target_profile_id: str,
    target_baseline_assessment: object,
    peer_baseline_assessments_by_profile: dict[str, object],
    assignments_by_profile: dict[str, ProfilePositionAssignment],
    package: ExposureNetworkPackage,
    max_exemplars: int = DEFAULT_MAX_EXEMPLARS,
    edge_index: ExposureEdgeIndex | None = None,
) -> dict[str, Any]:
    peer_payloads = {
        profile_id: {
            "baseline_score": _score(assessment),
            "confidence": _confidence(assessment),
            "reasoning": _reasoning(assessment),
        }
        for profile_id, assessment in peer_baseline_assessments_by_profile.items()
    }
    context, peers, exemplars = _base_context(
        target_profile_id=target_profile_id,
        assignments_by_profile=assignments_by_profile,
        package=package,
        peer_payloads_by_profile=peer_payloads,
        config=NetworkContextConfig(max_exemplars=max_exemplars),
        edge_index=edge_index,
    )
    scores = [float(peer["baseline_score"]) for peer in peers if "baseline_score" in peer]
    weights = [float(peer["exposure_weight"]) for peer in peers if "baseline_score" in peer]
    context.update(
        {
            "target_baseline_score": _score(target_baseline_assessment),
            "target_baseline_confidence": _confidence(target_baseline_assessment),
            "peer_score_mean": _round(_safe_mean(scores)),
            "peer_score_sd": _round(_safe_pstdev(scores)),
            "exposure_weighted_peer_mean": _round(_weighted_mean(scores, weights)),
            "peer_exemplars": exemplars,
            "peer_assessments": exemplars,
        }
    )
    return context


def build_post_attack_network_context(
    target_profile_id: str,
    target_private_post_assessment: object,
    same_condition_peer_post_assessments_by_profile: dict[str, object],
    assignments_by_profile: dict[str, ProfilePositionAssignment],
    package: ExposureNetworkPackage,
    max_exemplars: int = DEFAULT_MAX_EXEMPLARS,
    edge_index: ExposureEdgeIndex | None = None,
) -> dict[str, Any]:
    target_baseline_score = int(_assessment_field(target_private_post_assessment, "baseline_score", 0) or 0)
    target_post_score = _score(target_private_post_assessment)
    peer_payloads = {}
    for profile_id, assessment in same_condition_peer_post_assessments_by_profile.items():
        baseline_score = int(_assessment_field(assessment, "baseline_score", 0) or 0)
        post_score = _score(assessment)
        peer_payloads[profile_id] = {
            "baseline_score": baseline_score,
            "post_score": post_score,
            "attack_delta": post_score - baseline_score,
            "confidence": _confidence(assessment),
            "reasoning": _reasoning(assessment),
        }
    context, peers, exemplars = _base_context(
        target_profile_id=target_profile_id,
        assignments_by_profile=assignments_by_profile,
        package=package,
        peer_payloads_by_profile=peer_payloads,
        config=NetworkContextConfig(max_exemplars=max_exemplars),
        edge_index=edge_index,
    )
    post_scores = [float(peer["post_score"]) for peer in peers if "post_score" in peer]
    deltas = [float(peer["attack_delta"]) for peer in peers if "attack_delta" in peer]
    weights = [float(peer["exposure_weight"]) for peer in peers if "post_score" in peer]
    context.update(
        {
            "target_baseline_score": target_baseline_score,
            "target_private_post_score": target_post_score,
            "target_private_attack_delta": target_post_score - target_baseline_score,
            "peer_post_mean": _round(_safe_mean(post_scores)),
            "peer_delta_mean": _round(_safe_mean(deltas)),
            "peer_post_sd": _round(_safe_pstdev(post_scores)),
            "peer_delta_sd": _round(_safe_pstdev(deltas)),
            "exposure_weighted_peer_post_mean": _round(_weighted_mean(post_scores, weights)),
            "exposure_weighted_peer_delta_mean": _round(_weighted_mean(deltas, weights)),
            "peer_exemplars": exemplars,
            "peer_assessments": exemplars,
        }
    )
    return context
