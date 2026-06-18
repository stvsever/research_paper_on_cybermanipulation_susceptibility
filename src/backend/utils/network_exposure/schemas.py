from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            return None
        lower = value.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            number = float(value)
        except ValueError:
            return value
        if number.is_integer() and "." not in value and "e" not in lower:
            return int(number)
        return number
    return value


def clean_mapping(row: dict[str, Any]) -> dict[str, Any]:
    return {key: clean_scalar(value) for key, value in row.items() if value not in ("", None)}


@dataclass(frozen=True)
class ExposureEdge:
    source_position_id: str
    target_position_id: str
    exposure_weight: float
    raw_weight: float | None = None
    total_events: int | None = None
    interaction_types: str = ""
    rank_for_receiver: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "ExposureEdge":
        rank_value = row.get("rank_for_receiver")
        total_events = row.get("total_events")
        raw_weight = row.get("raw_weight")
        return cls(
            source_position_id=str(row["source_position_id"]),
            target_position_id=str(row["target_position_id"]),
            exposure_weight=as_float(row.get("exposure_weight")),
            raw_weight=as_float(raw_weight) if raw_weight not in (None, "") else None,
            total_events=as_int(total_events) if total_events not in (None, "") else None,
            interaction_types=str(row.get("interaction_types") or ""),
            rank_for_receiver=as_int(rank_value) if rank_value not in (None, "") else None,
        )

    def to_context(self) -> dict[str, Any]:
        return {
            "source_position_id": self.source_position_id,
            "target_position_id": self.target_position_id,
            "exposure_weight": round(float(self.exposure_weight), 6),
            "raw_weight": self.raw_weight,
            "total_events": self.total_events,
            "interaction_types": self.interaction_types,
            "exposure_rank_for_receiver": self.rank_for_receiver,
        }


@dataclass(frozen=True)
class ExposureNodeMetrics:
    position_id: str
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, position_id: str, *rows: dict[str, str] | None) -> "ExposureNodeMetrics":
        merged: dict[str, Any] = {"position_id": position_id}
        for row in rows:
            if row:
                merged.update(clean_mapping(row))
        return cls(position_id=position_id, values=merged)


@dataclass(frozen=True)
class ProfilePositionAssignment:
    profile_id: str
    position_id: str
    assignment_rank: int
    graph_id: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> dict[str, Any]:
        payload = dict(self.metrics)
        payload.update(
            {
                "profile_id": self.profile_id,
                "position_id": self.position_id,
                "assignment_rank": self.assignment_rank,
                "graph_id": self.graph_id,
            }
        )
        return payload


@dataclass(frozen=True)
class ExposurePeerContext:
    profile_id: str
    assignment: ProfilePositionAssignment
    edge: ExposureEdge
    assessment: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> dict[str, Any]:
        payload = {
            "profile_id": self.profile_id,
            "position_id": self.assignment.position_id,
            **self.edge.to_context(),
        }
        payload.update(self.assessment)
        return payload


@dataclass(frozen=True)
class ExposureNeighborhood:
    target_profile_id: str
    target_assignment: ProfilePositionAssignment
    incoming_peers: tuple[ExposurePeerContext, ...]
    outgoing_receivers: tuple[ExposurePeerContext, ...] = ()

    @property
    def peer_count(self) -> int:
        return len(self.incoming_peers)

    @property
    def total_exposure_weight(self) -> float:
        return sum(peer.edge.exposure_weight for peer in self.incoming_peers)

    def top_incoming(self, limit: int) -> tuple[ExposurePeerContext, ...]:
        return tuple(self.incoming_peers[: max(0, int(limit))])


@dataclass(frozen=True)
class NetworkContextConfig:
    max_exemplars: int = 8


@dataclass(frozen=True)
class ExposureEdgeIndex:
    edge_file: str
    incoming_by_target: dict[str, tuple[ExposureEdge, ...]]
    outgoing_by_source: dict[str, tuple[ExposureEdge, ...]]

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.incoming_by_target.values())


@dataclass(frozen=True)
class ExposureNetworkPackage:
    root: Path
    graph_id: str
    manifest: dict[str, Any]
    data_dictionary: dict[str, Any]
    nodes: dict[str, dict[str, str]]
    node_metrics: dict[str, dict[str, str]]
    neighborhood_metrics: dict[str, dict[str, str]]
    propagation_metrics: dict[str, dict[str, str]]
    assignment_positions: tuple[dict[str, str], ...]

    def path(self, filename: str) -> Path:
        return self.root / filename

    @property
    def position_ids(self) -> set[str]:
        return set(self.nodes)

    def metrics_for_position(self, position_id: str) -> ExposureNodeMetrics:
        return ExposureNodeMetrics.from_rows(
            position_id,
            self.nodes.get(position_id),
            self.node_metrics.get(position_id),
            self.neighborhood_metrics.get(position_id),
            self.propagation_metrics.get(position_id),
        )

    def assignment_candidates(self) -> Iterable[dict[str, str]]:
        return self.assignment_positions

