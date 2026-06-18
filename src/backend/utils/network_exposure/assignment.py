from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

from .defaults import DEFAULT_ASSIGNMENT_MODE
from .schemas import ExposureNetworkPackage, ProfilePositionAssignment, clean_mapping


def _profile_id(profile: object) -> str:
    if isinstance(profile, str):
        return profile
    if isinstance(profile, dict) and profile.get("profile_id"):
        return str(profile["profile_id"])
    value = getattr(profile, "profile_id", None)
    if value:
        return str(value)
    raise ValueError(f"Cannot extract profile_id from profile object: {profile!r}")


def _position_metrics(package: ExposureNetworkPackage, position_row: dict[str, str]) -> dict[str, object]:
    position_id = str(position_row["position_id"])
    metrics = package.metrics_for_position(position_id).values
    metrics.update(clean_mapping(position_row))
    metrics["network_basis"] = "empirical_politisky24_bluesky_exposure"
    metrics["edge_direction"] = package.manifest.get("edge_direction")
    metrics["edge_meaning"] = package.manifest.get("edge_meaning")
    return metrics


def assign_profiles_to_positions(
    profiles: Iterable[object],
    package: ExposureNetworkPackage,
    *,
    seed: int = 42,
    assignment_mode: str = DEFAULT_ASSIGNMENT_MODE,
) -> dict[str, ProfilePositionAssignment]:
    if assignment_mode != DEFAULT_ASSIGNMENT_MODE:
        raise RuntimeError(f"Unsupported exposure assignment mode: {assignment_mode}")
    profile_ids = sorted({_profile_id(profile) for profile in profiles})
    candidates = sorted(package.assignment_candidates(), key=lambda row: int(float(row["assignment_rank"])))
    if len(profile_ids) > len(candidates):
        raise RuntimeError(
            f"Requested {len(profile_ids)} profiles but graph has {len(candidates)} assignment positions"
        )
    selected_positions = candidates[: len(profile_ids)]
    shuffled_profiles = list(profile_ids)
    random.Random(seed).shuffle(shuffled_profiles)
    assignments: dict[str, ProfilePositionAssignment] = {}
    for profile_id, position_row in zip(shuffled_profiles, selected_positions):
        position_id = str(position_row["position_id"])
        assignments[profile_id] = ProfilePositionAssignment(
            profile_id=profile_id,
            position_id=position_id,
            assignment_rank=int(float(position_row["assignment_rank"])),
            graph_id=package.graph_id,
            metrics=_position_metrics(package, position_row),
        )
    return dict(sorted(assignments.items()))


_ASSIGNMENT_CORE_KEYS = {"profile_id", "position_id", "assignment_rank", "graph_id"}


def profile_position_assignment_from_context(
    payload: Mapping[str, object],
) -> ProfilePositionAssignment:
    missing = [key for key in _ASSIGNMENT_CORE_KEYS if key not in payload]
    if missing:
        raise RuntimeError(
            "Exposure-network assignment is missing required fields: "
            + ", ".join(sorted(missing))
        )
    metrics = {
        str(key): value
        for key, value in payload.items()
        if str(key) not in _ASSIGNMENT_CORE_KEYS
    }
    return ProfilePositionAssignment(
        profile_id=str(payload["profile_id"]),
        position_id=str(payload["position_id"]),
        assignment_rank=int(float(payload["assignment_rank"])),
        graph_id=str(payload["graph_id"]),
        metrics=metrics,
    )


def exposure_assignment_from_row(row: Mapping[str, object]) -> ProfilePositionAssignment:
    metadata = row.get("metadata")
    assignment = (
        metadata.get("exposure_network_assignment")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(assignment, Mapping):
        profile = row.get("profile")
        profile_metadata = profile.get("metadata") if isinstance(profile, Mapping) else None
        assignment = (
            profile_metadata.get("exposure_network_assignment")
            if isinstance(profile_metadata, Mapping)
            else None
        )
    if not isinstance(assignment, Mapping):
        scenario_id = row.get("scenario_id", "<unknown>")
        raise RuntimeError(
            f"Missing exposure_network_assignment for scenario {scenario_id}. "
            "Run Stage 01b before network-exposure stages."
        )
    return profile_position_assignment_from_context(assignment)


def exposure_assignments_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, ProfilePositionAssignment]:
    assignments: dict[str, ProfilePositionAssignment] = {}
    for row in rows:
        assignment = exposure_assignment_from_row(row)
        existing = assignments.get(assignment.profile_id)
        if existing is not None:
            if (
                existing.position_id != assignment.position_id
                or existing.assignment_rank != assignment.assignment_rank
                or existing.graph_id != assignment.graph_id
            ):
                raise RuntimeError(
                    "Inconsistent exposure-network assignment for profile "
                    f"{assignment.profile_id}: {existing.to_context()} != {assignment.to_context()}"
                )
            continue
        assignments[assignment.profile_id] = assignment
    if not assignments:
        raise RuntimeError("No exposure-network assignments found in input rows.")
    return dict(sorted(assignments.items()))
