from __future__ import annotations

from .defaults import EDGE_FILE_FULL
from .loader import load_edge_index
from .schemas import (
    ExposureEdge,
    ExposureEdgeIndex,
    ExposureNeighborhood,
    ExposureNetworkPackage,
    ExposurePeerContext,
    ProfilePositionAssignment,
)


def _position_to_profile(assignments_by_profile: dict[str, ProfilePositionAssignment]) -> dict[str, str]:
    return {assignment.position_id: profile_id for profile_id, assignment in assignments_by_profile.items()}


def build_incoming_exposure_neighborhood(
    target_profile_id: str,
    assignments_by_profile: dict[str, ProfilePositionAssignment],
    package: ExposureNetworkPackage,
    *,
    edge_index: ExposureEdgeIndex | None = None,
    edge_file: str = EDGE_FILE_FULL,
    peer_payloads_by_profile: dict[str, dict[str, object]] | None = None,
) -> ExposureNeighborhood:
    if target_profile_id not in assignments_by_profile:
        raise KeyError(f"Missing exposure-network assignment for profile: {target_profile_id}")
    target_assignment = assignments_by_profile[target_profile_id]
    assigned_positions = {assignment.position_id for assignment in assignments_by_profile.values()}
    index = edge_index or load_edge_index(
        package,
        edge_file=edge_file,
        target_positions=assigned_positions,
        source_positions=assigned_positions,
    )
    position_to_profile = _position_to_profile(assignments_by_profile)
    payloads = peer_payloads_by_profile or {}
    incoming_peers: list[ExposurePeerContext] = []
    for edge in index.incoming_by_target.get(target_assignment.position_id, ()):
        peer_profile_id = position_to_profile.get(edge.source_position_id)
        if peer_profile_id is None or peer_profile_id == target_profile_id:
            continue
        incoming_peers.append(
            ExposurePeerContext(
                profile_id=peer_profile_id,
                assignment=assignments_by_profile[peer_profile_id],
                edge=edge,
                assessment=dict(payloads.get(peer_profile_id) or {}),
            )
        )
    outgoing_receivers: list[ExposurePeerContext] = []
    for edge in index.outgoing_by_source.get(target_assignment.position_id, ()):
        receiver_profile_id = position_to_profile.get(edge.target_position_id)
        if receiver_profile_id is None or receiver_profile_id == target_profile_id:
            continue
        outgoing_receivers.append(
            ExposurePeerContext(
                profile_id=receiver_profile_id,
                assignment=assignments_by_profile[receiver_profile_id],
                edge=_reverse_receiver_edge(edge),
                assessment=dict(payloads.get(receiver_profile_id) or {}),
            )
        )
    incoming_peers.sort(key=lambda peer: (peer.edge.rank_for_receiver or 10**9, -peer.edge.exposure_weight, peer.profile_id))
    outgoing_receivers.sort(key=lambda peer: (-peer.edge.exposure_weight, peer.profile_id))
    return ExposureNeighborhood(
        target_profile_id=target_profile_id,
        target_assignment=target_assignment,
        incoming_peers=tuple(incoming_peers),
        outgoing_receivers=tuple(outgoing_receivers),
    )


def _reverse_receiver_edge(edge: ExposureEdge) -> ExposureEdge:
    return ExposureEdge(
        source_position_id=edge.source_position_id,
        target_position_id=edge.target_position_id,
        exposure_weight=edge.exposure_weight,
        raw_weight=edge.raw_weight,
        total_events=edge.total_events,
        interaction_types=edge.interaction_types,
        rank_for_receiver=edge.rank_for_receiver,
    )
