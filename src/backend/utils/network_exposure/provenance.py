from __future__ import annotations

from collections import Counter
from typing import Any

from .defaults import DEFAULT_ASSIGNMENT_MODE
from .schemas import ExposureNetworkPackage, ProfilePositionAssignment


def build_assignment_summary(
    assignments_by_profile: dict[str, ProfilePositionAssignment],
    package: ExposureNetworkPackage,
    *,
    seed: int,
    assignment_mode: str = DEFAULT_ASSIGNMENT_MODE,
) -> dict[str, Any]:
    roles = Counter(
        str(assignment.metrics.get("display_role") or assignment.metrics.get("dominant_structural_role") or "unknown")
        for assignment in assignments_by_profile.values()
    )
    communities = Counter(str(assignment.metrics.get("community_id") or "unknown") for assignment in assignments_by_profile.values())
    prompt_ready_count = sum(1 for assignment in assignments_by_profile.values() if bool(assignment.metrics.get("prompt_ready")))
    return {
        "graph_id": package.graph_id,
        "graph_root": str(package.root),
        "edge_direction": package.manifest.get("edge_direction"),
        "edge_meaning": package.manifest.get("edge_meaning"),
        "interaction_weight_formula": package.manifest.get("interaction_weight_formula"),
        "assignment_mode": assignment_mode,
        "assignment_seed": int(seed),
        "profile_count": len(assignments_by_profile),
        "prompt_ready_assignment_count": prompt_ready_count,
        "role_counts": dict(sorted(roles.items())),
        "community_counts": dict(sorted(communities.items())),
        "file_hashes": {
            filename: metadata.get("sha256")
            for filename, metadata in dict(package.manifest.get("files") or {}).items()
            if isinstance(metadata, dict)
        },
    }

