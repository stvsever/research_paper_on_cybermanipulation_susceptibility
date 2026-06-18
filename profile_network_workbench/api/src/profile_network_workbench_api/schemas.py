from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


OntologyMode = Literal["test", "production"]
BaselineRunStatus = Literal["queued", "running", "completed", "failed", "completed_with_errors"]
PostExposureRunStatus = BaselineRunStatus
NetworkExposureRunStatus = BaselineRunStatus
PostAttackNetworkExposureRunStatus = BaselineRunStatus


class OpinionLeafOption(BaseModel):
    path: str
    label: str
    domain: str | None = None


class AttackOption(BaseModel):
    path: str
    label: str
    family: str
    complexity_tier: str
    temporal_horizon: str
    epistemic_target: str
    compatible: bool
    notes: list[str] = Field(default_factory=list)


class AttackOptionsResponse(BaseModel):
    run_id: str
    opinion_leaf: str
    attack_options: list[AttackOption]


class AffinityComponents(BaseModel):
    categorical_similarity: float = Field(ge=0.0, le=1.0)
    personality_similarity: float = Field(ge=0.0, le=1.0)
    age_context_similarity: float = Field(ge=0.0, le=1.0)
    ontology_leaf_overlap: float = Field(ge=0.0, le=1.0)


class AffinityFormulaWeights(BaseModel):
    personality_similarity: float = Field(ge=0.0, le=1.0)
    ontology_leaf_overlap: float = Field(ge=0.0, le=1.0)
    age_context_similarity: float = Field(ge=0.0, le=1.0)
    categorical_similarity: float = Field(ge=0.0, le=1.0)


class AffinityFormulaComponent(BaseModel):
    key: str
    label: str
    description: str


class AffinityFormulaMetadata(BaseModel):
    label: str
    default_weights: AffinityFormulaWeights
    components: list[AffinityFormulaComponent]
    warning: str
    note: str


class BaselineResult(BaseModel):
    profile_id: str
    scenario_id: str
    opinion_leaf: str
    score: int
    confidence: float
    reasoning: str
    model_name: str
    call_id: str
    timestamp: str


class ProfileNetworkNode(BaseModel):
    id: str
    label: str
    cluster_id: str
    categorical_attributes: dict[str, str] = Field(default_factory=dict)
    continuous_attributes: dict[str, float] = Field(default_factory=dict)
    selected_leaf_nodes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    centrality: float = Field(ge=0.0, le=1.0)
    baseline: BaselineResult | None = None


class ProfileNetworkEdge(BaseModel):
    source: str
    target: str
    affinity: float = Field(ge=0.0, le=1.0)
    normalized_affinity: float = Field(ge=0.0, le=1.0)
    components: AffinityComponents
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_kind: str = "profile_affinity"
    directed: bool = False
    source_position_id: str | None = None
    target_position_id: str | None = None
    exposure_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    interaction_types: str | None = None
    rank_for_receiver: int | None = None


class ProfileLayoutAffinity(BaseModel):
    source: str
    target: str
    affinity: float = Field(ge=0.0, le=1.0)
    components: AffinityComponents
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_kind: str = "profile_affinity"
    directed: bool = False
    source_position_id: str | None = None
    target_position_id: str | None = None
    exposure_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    interaction_types: str | None = None
    rank_for_receiver: int | None = None


class ProfileNetworkDiagnostics(BaseModel):
    profile_count: int
    full_pair_count: int
    displayed_edge_count: int
    edge_limit_per_node: int
    affinity_min: float | None = None
    affinity_max: float | None = None
    affinity_mean: float | None = None
    edge_semantics: str = "profile_affinity_not_social_tie"
    empirical_edge_count: int | None = None
    assigned_profile_count: int | None = None
    community_count: int | None = None
    prompt_ready_count: int | None = None


class ProfileNetworkProvenance(BaseModel):
    run_id: str
    mode: OntologyMode
    config_path: str
    ontology_root: str
    source: str = "stage01_reconstructed_profiles"
    seed: int
    model_name: str | None = None
    graph_id: str | None = None
    graph_root: str | None = None
    network_basis: str | None = None


class ProfileNetworkResponse(BaseModel):
    run_id: str
    mode: OntologyMode
    nodes: list[ProfileNetworkNode]
    edges: list[ProfileNetworkEdge]
    layout_affinities: list[ProfileLayoutAffinity]
    affinity_formula: AffinityFormulaMetadata
    opinion_leaves: list[OpinionLeafOption]
    diagnostics: ProfileNetworkDiagnostics
    provenance: ProfileNetworkProvenance
    warnings: list[str] = Field(default_factory=list)


class BaselineRunCreateRequest(BaseModel):
    run_id: str = "run_1"
    opinion_leaf: str
    profile_ids: list[str] | None = None
    model_name: str | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=64)


class BaselineRunCreateResponse(BaseModel):
    baseline_run_id: str
    status: BaselineRunStatus
    run_id: str
    opinion_leaf: str
    profile_count: int
    model_name: str


class BaselineRunError(BaseModel):
    profile_id: str
    scenario_id: str
    message: str
    timestamp: str


class BaselineRunResponse(BaseModel):
    baseline_run_id: str
    status: BaselineRunStatus
    run_id: str
    opinion_leaf: str
    model_name: str
    profile_count: int
    completed_count: int
    failed_count: int
    results: list[BaselineResult] = Field(default_factory=list)
    errors: list[BaselineRunError] = Field(default_factory=list)
    artifact_dir: str
    raw_llm_dir: str
    created_at: str
    updated_at: str


class BaselinePromptMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class BaselinePromptPreviewResponse(BaseModel):
    run_id: str
    profile_id: str
    opinion_leaf: str
    prompt_name: str = "baseline_opinion.md"
    model_name: str
    system_prompt: str
    user_payload: dict[str, Any]
    messages: list[BaselinePromptMessage]


class PostExposureResult(BaseModel):
    profile_id: str
    scenario_id: str
    opinion_leaf: str
    attack_leaf: str
    baseline_score: int
    post_score: int
    delta_score: int
    adversarial_direction: int
    confidence: float
    reasoning: str
    model_name: str
    call_id: str
    timestamp: str
    heuristic_checks: dict[str, Any] = Field(default_factory=dict)
    post_direction_clamped: bool = False


class PostExposureRunCreateRequest(BaseModel):
    run_id: str = "run_1"
    baseline_run_id: str
    opinion_leaf: str
    attack_leaf: str
    profile_ids: list[str] | None = None
    model_name: str | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=64)


class PostExposureRunCreateResponse(BaseModel):
    post_run_id: str
    status: PostExposureRunStatus
    run_id: str
    baseline_run_id: str
    opinion_leaf: str
    attack_leaf: str
    profile_count: int
    model_name: str


class PostExposureRunError(BaseModel):
    profile_id: str
    scenario_id: str
    message: str
    timestamp: str


class PostExposureRunResponse(BaseModel):
    post_run_id: str
    status: PostExposureRunStatus
    run_id: str
    baseline_run_id: str
    opinion_leaf: str
    attack_leaf: str
    model_name: str
    profile_count: int
    completed_count: int
    failed_count: int
    results: list[PostExposureResult] = Field(default_factory=list)
    errors: list[PostExposureRunError] = Field(default_factory=list)
    artifact_dir: str
    raw_llm_dir: str
    attack_specs_path: str
    created_at: str
    updated_at: str


class PostExposurePromptPreviewResponse(BaseModel):
    run_id: str
    baseline_run_id: str
    profile_id: str
    opinion_leaf: str
    attack_leaf: str
    prompt_name: str = "post_attack_opinion.md"
    model_name: str
    system_prompt: str
    user_payload: dict[str, Any]
    messages: list[BaselinePromptMessage]


class NetworkExposureResult(BaseModel):
    profile_id: str
    scenario_id: str
    opinion_leaf: str
    baseline_score: int
    network_score: int
    delta_score: int
    confidence: float
    reasoning: str
    model_name: str
    call_id: str
    timestamp: str
    network_context: dict[str, Any] = Field(default_factory=dict)


class NetworkExposureRunCreateRequest(BaseModel):
    run_id: str = "run_1"
    baseline_run_id: str
    opinion_leaf: str
    profile_ids: list[str] | None = None
    model_name: str | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=64)
    top_k: int | None = Field(default=8, ge=1, le=32)


class NetworkExposureRunCreateResponse(BaseModel):
    network_run_id: str
    status: NetworkExposureRunStatus
    run_id: str
    baseline_run_id: str
    opinion_leaf: str
    profile_count: int
    model_name: str
    top_k: int


class NetworkExposureRunError(BaseModel):
    profile_id: str
    scenario_id: str
    message: str
    timestamp: str


class NetworkExposureRunResponse(BaseModel):
    network_run_id: str
    status: NetworkExposureRunStatus
    run_id: str
    baseline_run_id: str
    opinion_leaf: str
    model_name: str
    profile_count: int
    completed_count: int
    failed_count: int
    top_k: int
    results: list[NetworkExposureResult] = Field(default_factory=list)
    errors: list[NetworkExposureRunError] = Field(default_factory=list)
    artifact_dir: str
    raw_llm_dir: str
    network_contexts_path: str
    created_at: str
    updated_at: str


class NetworkExposurePromptPreviewResponse(BaseModel):
    run_id: str
    baseline_run_id: str
    profile_id: str
    opinion_leaf: str
    prompt_name: str = "network_exposure_opinion.md"
    model_name: str
    system_prompt: str
    user_payload: dict[str, Any]
    messages: list[BaselinePromptMessage]


class PostAttackNetworkExposureResult(BaseModel):
    profile_id: str
    scenario_id: str
    opinion_leaf: str
    attack_leaf: str
    baseline_score: int
    private_post_score: int
    post_attack_network_score: int
    delta_from_baseline: int
    increment_from_private_post: int
    adversarial_direction: int
    confidence: float
    reasoning: str
    model_name: str
    call_id: str
    timestamp: str
    post_attack_network_context: dict[str, Any] = Field(default_factory=dict)
    heuristic_checks: dict[str, Any] = Field(default_factory=dict)


class PostAttackNetworkExposureRunCreateRequest(BaseModel):
    run_id: str = "run_1"
    baseline_run_id: str
    post_run_id: str
    opinion_leaf: str
    attack_leaf: str
    profile_ids: list[str] | None = None
    model_name: str | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=64)
    top_k: int | None = Field(default=8, ge=1, le=32)


class PostAttackNetworkExposureRunCreateResponse(BaseModel):
    post_network_run_id: str
    status: PostAttackNetworkExposureRunStatus
    run_id: str
    baseline_run_id: str
    post_run_id: str
    opinion_leaf: str
    attack_leaf: str
    profile_count: int
    model_name: str
    top_k: int


class PostAttackNetworkExposureRunError(BaseModel):
    profile_id: str
    scenario_id: str
    message: str
    timestamp: str


class PostAttackNetworkExposureRunResponse(BaseModel):
    post_network_run_id: str
    status: PostAttackNetworkExposureRunStatus
    run_id: str
    baseline_run_id: str
    post_run_id: str
    opinion_leaf: str
    attack_leaf: str
    model_name: str
    profile_count: int
    completed_count: int
    failed_count: int
    top_k: int
    results: list[PostAttackNetworkExposureResult] = Field(default_factory=list)
    errors: list[PostAttackNetworkExposureRunError] = Field(default_factory=list)
    artifact_dir: str
    raw_llm_dir: str
    post_attack_network_contexts_path: str
    created_at: str
    updated_at: str


class PostAttackNetworkExposurePromptPreviewResponse(BaseModel):
    run_id: str
    baseline_run_id: str
    post_run_id: str
    profile_id: str
    opinion_leaf: str
    attack_leaf: str
    prompt_name: str = "post_attack_network_exposure_opinion.md"
    model_name: str
    system_prompt: str
    user_payload: dict[str, Any]
    messages: list[BaselinePromptMessage]


class PipelineViewStageStatus(BaseModel):
    stage_id: str
    stage_name: str
    available: bool
    manifest_path: str | None = None
    primary_output_path: str | None = None
    record_count: int | None = None
    created_at_utc: str | None = None
    live_available: bool = False
    live_status: str | None = None
    live_result_count: int | None = None
    source: Literal["canonical", "live_sidecar", "missing"] = "missing"


class PipelineViewResponse(BaseModel):
    run_id: str
    network: ProfileNetworkResponse
    attack_options: list[AttackOption] = Field(default_factory=list)
    baseline_run: BaselineRunResponse | None = None
    network_run: NetworkExposureRunResponse | None = None
    post_run: PostExposureRunResponse | None = None
    post_network_run: PostAttackNetworkExposureRunResponse | None = None
    stage_status: list[PipelineViewStageStatus] = Field(default_factory=list)
    selected_opinion_leaf: str | None = None
    selected_attack_leaf: str | None = None
    warnings: list[str] = Field(default_factory=list)
