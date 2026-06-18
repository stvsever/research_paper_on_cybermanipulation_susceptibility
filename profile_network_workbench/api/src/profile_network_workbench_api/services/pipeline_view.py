from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from profile_network_workbench_api.backend_adapter import (
    WorkbenchNotFoundError,
    mode_from_config,
    ontology_root,
    opinion_options,
    pipeline_config_path_for_run,
    read_pipeline_config,
    safe_run_id,
)
from profile_network_workbench_api.schemas import (
    AffinityComponents,
    AffinityFormulaComponent,
    AffinityFormulaMetadata,
    AffinityFormulaWeights,
    AttackOption,
    BaselineResult,
    BaselineRunResponse,
    NetworkExposureResult,
    NetworkExposureRunResponse,
    PipelineViewResponse,
    PipelineViewStageStatus,
    PostAttackNetworkExposureResult,
    PostAttackNetworkExposureRunResponse,
    PostExposureResult,
    PostExposureRunResponse,
    ProfileLayoutAffinity,
    ProfileNetworkDiagnostics,
    ProfileNetworkEdge,
    ProfileNetworkNode,
    ProfileNetworkProvenance,
    ProfileNetworkResponse,
)
from profile_network_workbench_api.settings import load_settings


STAGES: tuple[tuple[str, str], ...] = (
    ("01", "create_scenarios"),
    ("01b", "assign_exposure_network_positions"),
    ("02", "assess_baseline_opinions"),
    ("02b", "assess_network_exposure_opinions"),
    ("03", "run_opinion_attacks"),
    ("04", "assess_post_attack_opinions"),
    ("04b", "assess_post_attack_network_exposure_opinions"),
    ("05", "compute_effectivity_deltas"),
)

LIVE_RESULT_FILENAME = "live_results.jsonl"
LIVE_STATUS_FILENAME = "live_status.json"
LIVE_STAGE_IDS = {"02", "02b", "04", "04b"}


@dataclass(frozen=True)
class _PipelineArtifacts:
    run_id: str
    run_root: Path
    stage_outputs_root: Path
    statuses: dict[str, PipelineViewStageStatus]
    warnings: list[str]


def _backend_imports() -> dict[str, Any]:
    from src.backend.utils.network_exposure import (
        exposure_assignments_from_rows,
        load_edge_index,
        load_exposure_network_package,
    )
    from src.backend.utils.schemas import OpinionAssessment, ProfileConfiguration

    return {
        "exposure_assignments_from_rows": exposure_assignments_from_rows,
        "load_edge_index": load_edge_index,
        "load_exposure_network_package": load_exposure_network_package,
        "OpinionAssessment": OpinionAssessment,
        "ProfileConfiguration": ProfileConfiguration,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload or {})


def _read_jsonl(path: Path, *, ignore_invalid: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(dict(json.loads(line)))
            except json.JSONDecodeError:
                if ignore_invalid:
                    continue
                raise
    return rows


def _live_stage_dir(stage_outputs_root: Path, stage_id: str, stage_name: str) -> Path:
    return stage_outputs_root / f"{stage_id}_{stage_name}"


def _read_live_status(stage_dir: Path) -> dict[str, Any] | None:
    path = stage_dir / LIVE_STATUS_FILENAME
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _live_rows(stage_dir: Path) -> list[dict[str, Any]]:
    path = stage_dir / LIVE_RESULT_FILENAME
    if not path.exists():
        return []
    return _read_jsonl(path, ignore_invalid=True)


def _apply_live_statuses(artifacts: _PipelineArtifacts) -> _PipelineArtifacts:
    statuses = dict(artifacts.statuses)
    for stage_id, stage_name in STAGES:
        current = statuses.get(stage_id)
        if current is None:
            continue
        source = "canonical" if current.available else "missing"
        live_available = False
        live_status_value: str | None = None
        live_result_count: int | None = None
        if stage_id in LIVE_STAGE_IDS:
            stage_dir = _live_stage_dir(artifacts.stage_outputs_root, stage_id, stage_name)
            live_status = _read_live_status(stage_dir)
            live_rows = _live_rows(stage_dir)
            live_available = bool(live_status or live_rows)
            live_status_value = str(live_status.get("status")) if live_status and live_status.get("status") else None
            live_result_count = len(live_rows) if live_available else None
            if not current.available and live_available:
                source = "live_sidecar"
        statuses[stage_id] = current.model_copy(
            update={
                "live_available": live_available,
                "live_status": live_status_value,
                "live_result_count": live_result_count,
                "source": source,
            }
        )
    return _PipelineArtifacts(
        run_id=artifacts.run_id,
        run_root=artifacts.run_root,
        stage_outputs_root=artifacts.stage_outputs_root,
        statuses=statuses,
        warnings=artifacts.warnings,
    )


def _safe_artifact_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return path.resolve()


def _pipeline_artifacts(run_id: str) -> _PipelineArtifacts:
    safe_id = safe_run_id(run_id)
    settings = load_settings()
    run_root = (settings.evaluation_path / safe_id).resolve()
    if not run_root.exists():
        raise WorkbenchNotFoundError(f"Pipeline run root not found: {run_root}")

    warnings: list[str] = []
    run_manifest_path = run_root / "provenance" / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else {}
    if not run_manifest:
        warnings.append(f"Run manifest not found: {run_manifest_path}")

    stage_outputs_root = _safe_artifact_path(str(run_manifest.get("stage_outputs_root", "")))
    if stage_outputs_root is None:
        stage_outputs_root = run_root / "stage_outputs"
    stage_outputs_root = stage_outputs_root.resolve()

    statuses: dict[str, PipelineViewStageStatus] = {}
    for stage_id, stage_name in STAGES:
        manifest_path = stage_outputs_root / f"{stage_id}_{stage_name}" / "manifest.json"
        if not manifest_path.exists():
            statuses[stage_id] = PipelineViewStageStatus(
                stage_id=stage_id,
                stage_name=stage_name,
                available=False,
                manifest_path=str(manifest_path),
            )
            continue
        manifest = _read_json(manifest_path)
        primary = str(manifest.get("primary_output_path") or "")
        statuses[stage_id] = PipelineViewStageStatus(
            stage_id=stage_id,
            stage_name=stage_name,
            available=bool(primary and Path(primary).exists()),
            manifest_path=str(manifest_path),
            primary_output_path=primary or None,
            record_count=int(manifest.get("record_count") or 0),
            created_at_utc=str(manifest.get("created_at_utc") or "") or None,
        )
    return _apply_live_statuses(
        _PipelineArtifacts(
            run_id=safe_id,
            run_root=run_root,
            stage_outputs_root=stage_outputs_root,
            statuses=statuses,
            warnings=warnings,
        )
    )


def _pipeline_artifacts_from_db_data(data: Any) -> _PipelineArtifacts:
    statuses = {
        str(item["stage_id"]): PipelineViewStageStatus.model_validate(item)
        for item in data.statuses
    }
    return _apply_live_statuses(
        _PipelineArtifacts(
            run_id=str(data.run_id),
            run_root=Path(data.run_root),
            stage_outputs_root=Path(data.stage_outputs_root),
            statuses=statuses,
            warnings=list(data.warnings),
        )
    )


def _load_db_pipeline_view_data(run_id: str) -> Any | None:
    from src.backend.persistence.queries import load_pipeline_view_data

    return load_pipeline_view_data(run_id)


def _stage_rows(artifacts: _PipelineArtifacts, stage_id: str) -> list[dict[str, Any]]:
    status = artifacts.statuses.get(stage_id)
    if not status:
        return []
    if status.available and status.primary_output_path:
        path = Path(status.primary_output_path)
        if path.exists():
            return _read_jsonl(path)
    if status.source == "live_sidecar":
        stage_name = status.stage_name
        return _live_rows(_live_stage_dir(artifacts.stage_outputs_root, stage_id, stage_name))
    return []


def _stage_timestamp(status: PipelineViewStageStatus | None) -> str:
    return status.created_at_utc if status and status.created_at_utc else datetime.now(timezone.utc).isoformat()


def _artifact_dir(status: PipelineViewStageStatus | None) -> str:
    if status and status.manifest_path:
        return str(Path(status.manifest_path).parent)
    return ""


def _mapped_run_status(status: PipelineViewStageStatus | None) -> str:
    if status and status.source == "live_sidecar":
        value = (status.live_status or "running").strip()
        if value in {"queued", "running", "completed", "failed", "completed_with_errors"}:
            return value
        return "running"
    return "completed"


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
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _formula_metadata() -> AffinityFormulaMetadata:
    return AffinityFormulaMetadata(
        label="affinity = 0.60 personality + 0.20 ontology + 0.12 age/context + 0.08 categorical",
        default_weights=AffinityFormulaWeights(
            personality_similarity=0.60,
            ontology_leaf_overlap=0.20,
            age_context_similarity=0.12,
            categorical_similarity=0.08,
        ),
        components=[
            AffinityFormulaComponent(key="personality_similarity", label="Personality traits", description="Big Five profile traits."),
            AffinityFormulaComponent(key="ontology_leaf_overlap", label="Ontology overlap", description="Shared profile ontology leaves."),
            AffinityFormulaComponent(key="age_context_similarity", label="Age/context", description="Non-personality continuous profile context."),
            AffinityFormulaComponent(key="categorical_similarity", label="Categorical demographics", description="Low-weighted demographic overlap."),
        ],
        warning="Categorical demographics are deliberately low-weighted to avoid demographic-only clustering.",
        note="Pipeline view uses Stage 01b empirical exposure-network assignments; affinity metadata is retained only for live-mode compatibility.",
    )


def _display_edges(pairs: list[Any], edge_limit_per_node: int) -> list[Any]:
    by_node: dict[str, list[Any]] = {}
    for pair in pairs:
        by_node.setdefault(pair.source, []).append(pair)
        by_node.setdefault(pair.target, []).append(pair)
    selected: dict[tuple[str, str], Any] = {}
    for node_pairs in by_node.values():
        for pair in sorted(node_pairs, key=lambda item: item.affinity, reverse=True)[:edge_limit_per_node]:
            selected[tuple(sorted((pair.source, pair.target)))] = pair
    return sorted(selected.values(), key=lambda item: (item.source, item.target))


def _centrality(profile_ids: list[str], pairs: list[Any]) -> dict[str, float]:
    raw: dict[str, list[float]] = {profile_id: [] for profile_id in profile_ids}
    for pair in pairs:
        raw[pair.source].append(float(pair.affinity))
        raw[pair.target].append(float(pair.affinity))
    means = {profile_id: (sum(values) / len(values) if values else 0.0) for profile_id, values in raw.items()}
    if not means:
        return {}
    low = min(means.values())
    high = max(means.values())
    return {profile_id: _normalize(value, low, high) for profile_id, value in means.items()}


def _cluster_ids(profile_ids: list[str], pairs: list[Any]) -> dict[str, str]:
    parent = {profile_id: profile_id for profile_id in profile_ids}
    threshold = _percentile([float(pair.affinity) for pair in pairs], 0.90) if pairs else 1.0

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
        if float(pair.affinity) >= threshold:
            union(pair.source, pair.target)
    components: dict[str, list[str]] = {}
    for profile_id in profile_ids:
        components.setdefault(find(profile_id), []).append(profile_id)
    ordered = sorted((sorted(values) for values in components.values()), key=lambda values: (-len(values), values[0]))
    return {profile_id: f"cluster_{index:02d}" for index, values in enumerate(ordered, start=1) for profile_id in values}


def _schema_components(components: Any) -> AffinityComponents:
    return AffinityComponents(
        categorical_similarity=round(float(components.categorical_similarity), 6),
        personality_similarity=round(float(components.personality_similarity), 6),
        age_context_similarity=round(float(components.age_context_similarity), 6),
        ontology_leaf_overlap=round(float(components.ontology_leaf_overlap), 6),
    )


def _empty_components() -> AffinityComponents:
    return AffinityComponents(
        categorical_similarity=0.0,
        personality_similarity=0.0,
        age_context_similarity=0.0,
        ontology_leaf_overlap=0.0,
    )


def _float_metric(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _build_network(
    run_id: str,
    artifacts: _PipelineArtifacts,
    stage01b_rows: list[dict[str, Any]],
    edge_limit_per_node: int,
) -> tuple[ProfileNetworkResponse, list[str], list[str]]:
    backend = _backend_imports()
    ProfileConfiguration = backend["ProfileConfiguration"]
    exposure_assignments_from_rows = backend["exposure_assignments_from_rows"]
    load_edge_index = backend["load_edge_index"]
    load_exposure_network_package = backend["load_exposure_network_package"]

    profile_by_id: dict[str, Any] = {}
    scenario_index_by_profile: dict[str, int] = {}
    opinion_leaves: list[str] = []
    warnings: list[str] = []
    seed = 0
    assignments_by_profile = exposure_assignments_from_rows(stage01b_rows)
    for row in stage01b_rows:
        profile_payload = row.get("profile")
        if not isinstance(profile_payload, dict) or not profile_payload.get("profile_id"):
            continue
        seed = int(row.get("random_seed") or seed)
        opinion_leaf = str(row.get("opinion_leaf") or "")
        if opinion_leaf and opinion_leaf not in opinion_leaves:
            opinion_leaves.append(opinion_leaf)
        profile = ProfileConfiguration.model_validate(profile_payload)
        profile_id = str(profile.profile_id)
        current_index = scenario_index_by_profile.get(profile_id)
        scenario_index = int(row.get("scenario_index") or 0)
        if current_index is None or scenario_index < current_index:
            profile_by_id[profile_id] = profile
            scenario_index_by_profile[profile_id] = scenario_index

    profiles = [profile_by_id[profile_id] for profile_id in sorted(profile_by_id)]
    if not profiles:
        raise WorkbenchNotFoundError("Stage 01b output contains no profiles.")

    profile_ids = [str(profile.profile_id) for profile in profiles]

    config_path = pipeline_config_path_for_run(run_id)
    try:
        config_payload = read_pipeline_config(run_id)
        mode = mode_from_config(config_payload)
    except WorkbenchNotFoundError:
        config_payload = {}
        stage01_status = artifacts.statuses.get("01")
        mode = "test"
        warnings.append("Pipeline config not found; ontology mode defaulted to test.")
        if stage01_status and stage01_status.manifest_path:
            config_path = Path(stage01_status.manifest_path)

    graph_ids = {assignment.graph_id for assignment in assignments_by_profile.values()}
    if len(graph_ids) != 1:
        raise WorkbenchNotFoundError(f"Pipeline view requires one exposure graph id; found {sorted(graph_ids)}")
    graph_id = sorted(graph_ids)[0]
    graph_root = config_payload.get("exposure_network_root") or None
    package = load_exposure_network_package(graph_root=graph_root, graph_id=graph_id, validate=True)
    assigned_positions = {assignment.position_id for assignment in assignments_by_profile.values()}
    edge_index = load_edge_index(package, target_positions=assigned_positions, source_positions=assigned_positions)
    position_to_profile = {
        assignment.position_id: profile_id
        for profile_id, assignment in assignments_by_profile.items()
    }

    weights: list[float] = []
    edge_payloads: list[dict[str, Any]] = []
    for target_position, incoming_edges in edge_index.incoming_by_target.items():
        target_profile = position_to_profile.get(target_position)
        if not target_profile:
            continue
        for exposure_edge in incoming_edges:
            source_profile = position_to_profile.get(exposure_edge.source_position_id)
            if not source_profile or source_profile == target_profile:
                continue
            weight = max(0.0, min(1.0, float(exposure_edge.exposure_weight)))
            weights.append(weight)
            edge_payloads.append(
                {
                    "source": source_profile,
                    "target": target_profile,
                    "source_position_id": exposure_edge.source_position_id,
                    "target_position_id": exposure_edge.target_position_id,
                    "weight": weight,
                    "interaction_types": exposure_edge.interaction_types,
                    "rank_for_receiver": exposure_edge.rank_for_receiver,
                }
            )
    low = _percentile(weights, 0.10)
    high = _percentile(weights, 0.95)
    edge_payloads.sort(key=lambda item: (str(item["source"]), str(item["target"])))
    components = _empty_components()

    reach_values = {
        profile_id: _float_metric(assignments_by_profile[profile_id].metrics, "outgoing_visibility_weight")
        for profile_id in profile_ids
        if profile_id in assignments_by_profile
    }
    reach_low = min(reach_values.values()) if reach_values else 0.0
    reach_high = max(reach_values.values()) if reach_values else 1.0
    centrality = {
        profile_id: _normalize(value, reach_low, reach_high)
        for profile_id, value in reach_values.items()
    }

    nodes = [
        _pipeline_node(profile, assignments_by_profile, centrality)
        for profile in profiles
    ]
    communities = {
        str(node.metadata.get("exposure_network_assignment", {}).get("community_id") or "")
        for node in nodes
        if isinstance(node.metadata.get("exposure_network_assignment"), dict)
    }
    communities.discard("")
    prompt_ready_count = sum(
        1
        for node in nodes
        if isinstance(node.metadata.get("exposure_network_assignment"), dict)
        and bool(node.metadata["exposure_network_assignment"].get("prompt_ready"))
    )

    network = ProfileNetworkResponse(
        run_id=run_id,
        mode=mode,
        nodes=nodes,
        edges=[
            ProfileNetworkEdge(
                source=str(edge["source"]),
                target=str(edge["target"]),
                affinity=round(float(edge["weight"]), 6),
                normalized_affinity=round(_normalize(float(edge["weight"]), low, high), 6),
                components=components,
                weight=round(float(edge["weight"]), 6),
                normalized_weight=round(_normalize(float(edge["weight"]), low, high), 6),
                edge_kind="empirical_exposure",
                directed=True,
                source_position_id=str(edge["source_position_id"]),
                target_position_id=str(edge["target_position_id"]),
                exposure_weight=round(float(edge["weight"]), 6),
                interaction_types=str(edge.get("interaction_types") or ""),
                rank_for_receiver=edge.get("rank_for_receiver"),
            )
            for edge in edge_payloads
        ],
        layout_affinities=[
            ProfileLayoutAffinity(
                source=str(edge["source"]),
                target=str(edge["target"]),
                affinity=round(float(edge["weight"]), 6),
                components=components,
                weight=round(float(edge["weight"]), 6),
                edge_kind="empirical_exposure",
                directed=True,
                source_position_id=str(edge["source_position_id"]),
                target_position_id=str(edge["target_position_id"]),
                exposure_weight=round(float(edge["weight"]), 6),
                interaction_types=str(edge.get("interaction_types") or ""),
                rank_for_receiver=edge.get("rank_for_receiver"),
            )
            for edge in edge_payloads
        ],
        affinity_formula=_formula_metadata(),
        opinion_leaves=opinion_options(opinion_leaves),
        diagnostics=ProfileNetworkDiagnostics(
            profile_count=len(nodes),
            full_pair_count=len(edge_payloads),
            displayed_edge_count=len(edge_payloads),
            edge_limit_per_node=max(1, min(20, int(edge_limit_per_node))),
            affinity_min=round(min(weights), 6) if weights else None,
            affinity_max=round(max(weights), 6) if weights else None,
            affinity_mean=round(mean(weights), 6) if weights else None,
            edge_semantics="visible_peer_to_exposed_receiver",
            empirical_edge_count=len(edge_payloads),
            assigned_profile_count=len(assignments_by_profile),
            community_count=len(communities),
            prompt_ready_count=prompt_ready_count,
        ),
        provenance=ProfileNetworkProvenance(
            run_id=run_id,
            mode=mode,
            config_path=str(config_path),
            ontology_root=str(ontology_root(mode)),
            source="stage01b_exposure_network_assignments",
            seed=seed,
            model_name=str(config_payload.get("openrouter_model")) if config_payload.get("openrouter_model") else None,
            graph_id=package.graph_id,
            graph_root=str(package.root),
            network_basis="politisky24_bluesky_v1_empirical_exposure",
        ),
        warnings=warnings,
    )
    return network, opinion_leaves, warnings


def _pipeline_node(
    profile: Any,
    assignments_by_profile: dict[str, Any],
    centrality: dict[str, float],
) -> ProfileNetworkNode:
    profile_id = str(profile.profile_id)
    assignment = assignments_by_profile[profile_id]
    assignment_payload = assignment.to_context()
    metadata = dict(profile.metadata or {})
    metadata["exposure_network_assignment"] = assignment_payload
    return ProfileNetworkNode(
        id=profile_id,
        label=profile_id.replace("_", " "),
        cluster_id=str(assignment_payload.get("community_id") or "community_unknown"),
        categorical_attributes={str(key): str(value) for key, value in profile.categorical_attributes.items()},
        continuous_attributes={str(key): float(value) for key, value in profile.continuous_attributes.items()},
        selected_leaf_nodes=[str(item) for item in profile.selected_leaf_nodes],
        metadata=metadata,
        centrality=round(centrality.get(profile_id, 0.0), 6),
    )


def _assessment(row: dict[str, Any], key: str) -> Any | None:
    value = row.get(key)
    if not isinstance(value, dict):
        return None
    return _backend_imports()["OpinionAssessment"].model_validate(value)


def _canonical_rows(
    rows: list[dict[str, Any]],
    *,
    opinion_leaf: str,
    assessment_key: str,
    attack_leaf: str | None = None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    selected_scores: dict[str, int] = {}
    for row in rows:
        if row.get("opinion_leaf") != opinion_leaf or not isinstance(row.get(assessment_key), dict):
            continue
        if attack_leaf is not None and row.get("attack_leaf") != attack_leaf:
            continue
        profile = row.get("profile")
        if not isinstance(profile, dict) or not profile.get("profile_id"):
            continue
        profile_id = str(profile["profile_id"])
        assessment = _assessment(row, assessment_key)
        if assessment is None:
            continue
        score = int(assessment.score)
        if profile_id in selected_scores and selected_scores[profile_id] != score:
            warnings.append(
                f"Conflicting {assessment_key} scores for {profile_id}; lowest scenario_index row is displayed."
            )
        current = selected.get(profile_id)
        if current is None or int(row.get("scenario_index", 0)) < int(current.get("scenario_index", 0)):
            selected[profile_id] = row
            selected_scores[profile_id] = score
    return [selected[key] for key in sorted(selected)]


def _first_attack_leaf(rows: list[dict[str, Any]], opinion_leaf: str) -> str | None:
    for row in rows:
        if row.get("opinion_leaf") == opinion_leaf and row.get("attack_present") and row.get("attack_leaf"):
            return str(row["attack_leaf"])
    return None


def _has_attack_leaf(rows: list[dict[str, Any]], opinion_leaf: str, attack_leaf: str) -> bool:
    return any(
        row.get("opinion_leaf") == opinion_leaf
        and row.get("attack_present")
        and row.get("attack_leaf") == attack_leaf
        for row in rows
    )


def _attack_label(path: str) -> str:
    return path.split(">")[-1].strip() if ">" in path else path


def _attack_options_from_rows(rows: list[dict[str, Any]], opinion_leaf: str) -> list[AttackOption]:
    leaves: list[str] = []
    for row in rows:
        leaf = row.get("attack_leaf")
        if row.get("opinion_leaf") == opinion_leaf and row.get("attack_present") and leaf and str(leaf) not in leaves:
            leaves.append(str(leaf))
    return [
        AttackOption(
            path=leaf,
            label=_attack_label(leaf),
            family="pipeline_artifact",
            complexity_tier="pipeline_artifact",
            temporal_horizon="pipeline_artifact",
            epistemic_target="pipeline_artifact",
            compatible=True,
            notes=["Read from canonical pipeline Stage 01/04 artifacts."],
        )
        for leaf in leaves
    ]


def _baseline_run(
    run_id: str,
    rows: list[dict[str, Any]],
    opinion_leaf: str,
    profile_count: int,
    status: PipelineViewStageStatus | None,
    warnings: list[str],
) -> BaselineRunResponse | None:
    selected = _canonical_rows(rows, opinion_leaf=opinion_leaf, assessment_key="baseline_assessment", warnings=warnings)
    if not selected:
        return None
    timestamp = _stage_timestamp(status)
    results: list[BaselineResult] = []
    for row in selected:
        assessment = _assessment(row, "baseline_assessment")
        if assessment is None:
            continue
        profile_id = str(row["profile"]["profile_id"])
        results.append(
            BaselineResult(
                profile_id=profile_id,
                scenario_id=str(assessment.scenario_id),
                opinion_leaf=opinion_leaf,
                score=int(assessment.score),
                confidence=float(assessment.confidence),
                reasoning=str(assessment.reasoning),
                model_name=str(assessment.model_name),
                call_id=f"pipeline:{assessment.scenario_id}:baseline",
                timestamp=timestamp,
            )
        )
    model_name = results[0].model_name if results else "pipeline_artifact"
    return BaselineRunResponse(
        baseline_run_id=f"pipeline_{run_id}_02",
        status=_mapped_run_status(status),
        run_id=run_id,
        opinion_leaf=opinion_leaf,
        model_name=model_name,
        profile_count=profile_count,
        completed_count=len(results),
        failed_count=0,
        results=results,
        errors=[],
        artifact_dir=_artifact_dir(status),
        raw_llm_dir="",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _network_run(
    run_id: str,
    rows: list[dict[str, Any]],
    opinion_leaf: str,
    profile_count: int,
    baseline_run_id: str,
    status: PipelineViewStageStatus | None,
    warnings: list[str],
) -> NetworkExposureRunResponse | None:
    selected = _canonical_rows(rows, opinion_leaf=opinion_leaf, assessment_key="network_exposure_assessment", warnings=warnings)
    if not selected:
        return None
    timestamp = _stage_timestamp(status)
    results: list[NetworkExposureResult] = []
    for row in selected:
        assessment = _assessment(row, "network_exposure_assessment")
        baseline = _assessment(row, "baseline_assessment")
        if assessment is None:
            continue
        context = row.get("network_exposure_context") if isinstance(row.get("network_exposure_context"), dict) else {}
        baseline_score = int(baseline.score) if baseline is not None else int(context.get("target_baseline_score", 0) or 0)
        profile_id = str(row["profile"]["profile_id"])
        results.append(
            NetworkExposureResult(
                profile_id=profile_id,
                scenario_id=str(assessment.scenario_id),
                opinion_leaf=opinion_leaf,
                baseline_score=baseline_score,
                network_score=int(assessment.score),
                delta_score=int(assessment.score) - baseline_score,
                confidence=float(assessment.confidence),
                reasoning=str(assessment.reasoning),
                model_name=str(assessment.model_name),
                call_id=f"pipeline:{assessment.scenario_id}:network_exposure",
                timestamp=timestamp,
                network_context=dict(context),
            )
        )
    top_k = int((results[0].network_context or {}).get("top_k", 8) or 8) if results else 8
    return NetworkExposureRunResponse(
        network_run_id=f"pipeline_{run_id}_02b",
        status=_mapped_run_status(status),
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        opinion_leaf=opinion_leaf,
        model_name=results[0].model_name if results else "pipeline_artifact",
        profile_count=profile_count,
        completed_count=len(results),
        failed_count=0,
        top_k=top_k,
        results=results,
        errors=[],
        artifact_dir=_artifact_dir(status),
        raw_llm_dir="",
        network_contexts_path=str(Path(_artifact_dir(status)) / "network_contexts.jsonl") if status else "",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _post_run(
    run_id: str,
    rows: list[dict[str, Any]],
    opinion_leaf: str,
    attack_leaf: str,
    profile_count: int,
    baseline_run_id: str,
    status: PipelineViewStageStatus | None,
    warnings: list[str],
) -> PostExposureRunResponse | None:
    selected = _canonical_rows(
        rows,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        assessment_key="post_attack_assessment",
        warnings=warnings,
    )
    if not selected:
        return None
    timestamp = _stage_timestamp(status)
    results: list[PostExposureResult] = []
    for row in selected:
        assessment = _assessment(row, "post_attack_assessment")
        baseline = _assessment(row, "baseline_assessment")
        if assessment is None or baseline is None:
            continue
        spec = row.get("attack_vector_spec") if isinstance(row.get("attack_vector_spec"), dict) else {}
        adv = int(spec.get("adversarial_direction", row.get("adversarial_direction", 0)) or 0)
        profile_id = str(row["profile"]["profile_id"])
        results.append(
            PostExposureResult(
                profile_id=profile_id,
                scenario_id=str(assessment.scenario_id),
                opinion_leaf=opinion_leaf,
                attack_leaf=attack_leaf,
                baseline_score=int(baseline.score),
                post_score=int(assessment.score),
                delta_score=int(assessment.score) - int(baseline.score),
                adversarial_direction=adv,
                confidence=float(assessment.confidence),
                reasoning=str(assessment.reasoning),
                model_name=str(assessment.model_name),
                call_id=f"pipeline:{assessment.scenario_id}:post_attack",
                timestamp=timestamp,
                heuristic_checks=dict(row.get("post_heuristic_checks") or {}),
                post_direction_clamped=bool(row.get("post_direction_clamped", False)),
            )
        )
    return PostExposureRunResponse(
        post_run_id=f"pipeline_{run_id}_04",
        status=_mapped_run_status(status),
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        model_name=results[0].model_name if results else "pipeline_artifact",
        profile_count=profile_count,
        completed_count=len(results),
        failed_count=0,
        results=results,
        errors=[],
        artifact_dir=_artifact_dir(status),
        raw_llm_dir="",
        attack_specs_path=str(Path(_artifact_dir(status)) / "attack_vector_specs.jsonl") if status else "",
        created_at=timestamp,
        updated_at=timestamp,
    )


def _post_network_run(
    run_id: str,
    rows: list[dict[str, Any]],
    opinion_leaf: str,
    attack_leaf: str,
    profile_count: int,
    baseline_run_id: str,
    post_run_id: str,
    status: PipelineViewStageStatus | None,
    warnings: list[str],
) -> PostAttackNetworkExposureRunResponse | None:
    selected = _canonical_rows(
        rows,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        assessment_key="post_attack_network_exposure_assessment",
        warnings=warnings,
    )
    if not selected:
        return None
    timestamp = _stage_timestamp(status)
    results: list[PostAttackNetworkExposureResult] = []
    for row in selected:
        assessment = _assessment(row, "post_attack_network_exposure_assessment")
        baseline = _assessment(row, "baseline_assessment")
        post = _assessment(row, "post_attack_assessment")
        if assessment is None or baseline is None or post is None:
            continue
        context = (
            row.get("post_attack_network_exposure_context")
            if isinstance(row.get("post_attack_network_exposure_context"), dict)
            else {}
        )
        spec = row.get("attack_vector_spec") if isinstance(row.get("attack_vector_spec"), dict) else {}
        adv = int(spec.get("adversarial_direction", row.get("adversarial_direction", 0)) or 0)
        profile_id = str(row["profile"]["profile_id"])
        results.append(
            PostAttackNetworkExposureResult(
                profile_id=profile_id,
                scenario_id=str(assessment.scenario_id),
                opinion_leaf=opinion_leaf,
                attack_leaf=attack_leaf,
                baseline_score=int(baseline.score),
                private_post_score=int(post.score),
                post_attack_network_score=int(assessment.score),
                delta_from_baseline=int(assessment.score) - int(baseline.score),
                increment_from_private_post=int(assessment.score) - int(post.score),
                adversarial_direction=adv,
                confidence=float(assessment.confidence),
                reasoning=str(assessment.reasoning),
                model_name=str(assessment.model_name),
                call_id=f"pipeline:{assessment.scenario_id}:post_attack_network",
                timestamp=timestamp,
                post_attack_network_context=dict(context),
                heuristic_checks=dict(row.get("post_attack_network_exposure_heuristic_checks") or {}),
            )
        )
    top_k = int((results[0].post_attack_network_context or {}).get("top_k", 8) or 8) if results else 8
    return PostAttackNetworkExposureRunResponse(
        post_network_run_id=f"pipeline_{run_id}_04b",
        status=_mapped_run_status(status),
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        post_run_id=post_run_id,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        model_name=results[0].model_name if results else "pipeline_artifact",
        profile_count=profile_count,
        completed_count=len(results),
        failed_count=0,
        top_k=top_k,
        results=results,
        errors=[],
        artifact_dir=_artifact_dir(status),
        raw_llm_dir="",
        post_attack_network_contexts_path=(
            str(Path(_artifact_dir(status)) / "post_attack_network_contexts.jsonl") if status else ""
        ),
        created_at=timestamp,
        updated_at=timestamp,
    )


def _build_pipeline_view_response(
    run_id: str = "run_1",
    artifacts: _PipelineArtifacts | None = None,
    rows_by_stage: dict[str, list[dict[str, Any]]] | None = None,
    edge_limit_per_node: int = 6,
    opinion_leaf: str | None = None,
    attack_leaf: str | None = None,
) -> PipelineViewResponse:
    if artifacts is None:
        artifacts = _pipeline_artifacts(run_id)
    rows_by_stage = rows_by_stage or {}
    stage01b_rows = rows_by_stage.get("01b") or _stage_rows(artifacts, "01b")
    if not stage01b_rows:
        stage01b_status = artifacts.statuses.get("01b")
        raise WorkbenchNotFoundError(
            f"Pipeline Stage 01b exposure-network assignment output is required for pipeline_view and was not found. "
            f"Regenerate the run with Stage 01b enabled: "
            f"{stage01b_status.primary_output_path if stage01b_status else artifacts.stage_outputs_root}"
        )

    warnings = list(artifacts.warnings)
    network, opinion_leaves, network_warnings = _build_network(run_id, artifacts, stage01b_rows, edge_limit_per_node)
    warnings.extend(network_warnings)
    selected_opinion = opinion_leaf if opinion_leaf in opinion_leaves else (opinion_leaves[0] if opinion_leaves else "")
    if opinion_leaf and opinion_leaf not in opinion_leaves:
        warnings.append(f"Requested opinion leaf is not present in Stage 01 output: {opinion_leaf}")

    stage02_rows = rows_by_stage.get("02") or _stage_rows(artifacts, "02")
    stage02b_rows = rows_by_stage.get("02b") or _stage_rows(artifacts, "02b")
    stage04_rows = rows_by_stage.get("04") or _stage_rows(artifacts, "04")
    stage04b_rows = rows_by_stage.get("04b") or _stage_rows(artifacts, "04b")
    attack_options = _attack_options_from_rows(stage04_rows or stage01b_rows, selected_opinion)

    fallback_attack = _first_attack_leaf(stage04_rows, selected_opinion) or _first_attack_leaf(stage01b_rows, selected_opinion)
    requested_attack_available = bool(
        attack_leaf
        and (
            _has_attack_leaf(stage04_rows, selected_opinion, attack_leaf)
            or _has_attack_leaf(stage01b_rows, selected_opinion, attack_leaf)
        )
    )
    selected_attack = attack_leaf if requested_attack_available else fallback_attack
    if attack_leaf and not requested_attack_available:
        warnings.append(f"Requested attack leaf is not present for selected opinion: {attack_leaf}")
    if not selected_attack:
        warnings.append("No attack leaf found for selected opinion; post-exposure pipeline results are unavailable.")

    profile_count = network.diagnostics.profile_count
    baseline = _baseline_run(
        run_id,
        stage02_rows,
        selected_opinion,
        profile_count,
        artifacts.statuses.get("02"),
        warnings,
    )
    network_run = (
        _network_run(
            run_id,
            stage02b_rows,
            selected_opinion,
            profile_count,
            baseline.baseline_run_id if baseline else f"pipeline_{run_id}_02",
            artifacts.statuses.get("02b"),
            warnings,
        )
        if stage02b_rows
        else None
    )
    post = (
        _post_run(
            run_id,
            stage04_rows,
            selected_opinion,
            selected_attack,
            profile_count,
            baseline.baseline_run_id if baseline else f"pipeline_{run_id}_02",
            artifacts.statuses.get("04"),
            warnings,
        )
        if selected_attack and stage04_rows
        else None
    )
    post_network = (
        _post_network_run(
            run_id,
            stage04b_rows,
            selected_opinion,
            selected_attack,
            profile_count,
            baseline.baseline_run_id if baseline else f"pipeline_{run_id}_02",
            post.post_run_id if post else f"pipeline_{run_id}_04",
            artifacts.statuses.get("04b"),
            warnings,
        )
        if selected_attack and stage04b_rows
        else None
    )

    for stage_id in ("02", "02b", "04", "04b"):
        status = artifacts.statuses.get(stage_id, PipelineViewStageStatus(stage_id=stage_id, stage_name="", available=False))
        if status.source == "live_sidecar":
            warnings.append(f"Stage {stage_id} is using live sidecar rows; canonical manifest is not available yet.")
        elif not status.available:
            warnings.append(f"Stage {stage_id} artifact is not available for pipeline_view.")

    return PipelineViewResponse(
        run_id=run_id,
        network=network,
        attack_options=attack_options,
        baseline_run=baseline,
        network_run=network_run,
        post_run=post,
        post_network_run=post_network,
        stage_status=list(artifacts.statuses.values()),
        selected_opinion_leaf=selected_opinion or None,
        selected_attack_leaf=selected_attack,
        warnings=warnings,
    )


def get_pipeline_view(
    run_id: str = "run_1",
    edge_limit_per_node: int = 6,
    opinion_leaf: str | None = None,
    attack_leaf: str | None = None,
) -> PipelineViewResponse:
    db_warning: str | None = None
    try:
        db_data = _load_db_pipeline_view_data(safe_run_id(run_id))
        if db_data is not None:
            return _build_pipeline_view_response(
                run_id=run_id,
                artifacts=_pipeline_artifacts_from_db_data(db_data),
                rows_by_stage=db_data.rows_by_stage,
                edge_limit_per_node=edge_limit_per_node,
                opinion_leaf=opinion_leaf,
                attack_leaf=attack_leaf,
            )
    except Exception as exc:
        db_warning = f"Postgres pipeline_view unavailable; using artifact reader: {exc}"

    response = _build_pipeline_view_response(
        run_id=run_id,
        edge_limit_per_node=edge_limit_per_node,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
    )
    if db_warning:
        response.warnings.append(db_warning)
    return response
