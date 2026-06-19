from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


SCORE_MIN = -1000
SCORE_MAX = 1000


class ProfileConfiguration(BaseModel):
    profile_id: str
    categorical_attributes: Dict[str, str] = Field(default_factory=dict)
    continuous_attributes: Dict[str, float] = Field(default_factory=dict)
    selected_leaf_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OpinionClusterLeaf(BaseModel):
    """One issue-position leaf inside an opinion parent cluster.

    Used by the integrated-scenario design (run_2+) where a single scenario
    targets an entire issue domain (the parent) and all of its directional
    leaves at once, rather than one leaf per scenario.
    """

    leaf: str
    path: str = ""
    adversarial_direction: int = 0


class OpinionCluster(BaseModel):
    """An opinion parent cluster: one issue domain and its directional leaves.

    Carried on a ScenarioRecord so stages 02/04 can assess every leaf of the
    domain in a single agent call (compute-saving cluster batching), and stage
    05 can expand the per-leaf scores back into the standard per-leaf long
    table the rest of the pipeline expects.
    """

    key: str
    family: str = ""
    parent_name: str = ""
    n_leaves: int = 0
    direction_summary: Dict[str, int] = Field(default_factory=dict)
    leaves: List[OpinionClusterLeaf] = Field(default_factory=list)


class ScenarioRecord(BaseModel):
    scenario_id: str
    scenario_index: int
    random_seed: int
    profile: ProfileConfiguration
    opinion_leaf: str
    # Optional opinion parent cluster (integrated-scenario design). When set,
    # `opinion_leaf` holds the cluster key (parent path) and the per-leaf items
    # live in `opinion_cluster.leaves`. Legacy single-leaf runs leave it None.
    opinion_cluster: Optional[OpinionCluster] = None
    attack_present: bool
    attack_leaf: Optional[str] = None
    attack_primary_node: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OpinionAssessment(BaseModel):
    scenario_id: str
    # "network_exposure_baseline" / "post_attack_network_exposure" are emitted only by
    # the additive empirical exposure-network phases (stages 02b / 04b). The core
    # individual-layer run (stages 02 / 04) only ever emits "baseline" / "post_attack".
    phase: Literal["baseline", "network_exposure_baseline", "post_attack", "post_attack_network_exposure"]
    opinion_leaf: str
    score: int
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    model_name: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(f"score must be in [{SCORE_MIN}, {SCORE_MAX}]")
        return value


class ClusterLeafScore(BaseModel):
    """One leaf's opinion score inside a cluster-level assessment."""

    leaf: str
    score: int
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""
    adversarial_direction: int = 0

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if value < SCORE_MIN or value > SCORE_MAX:
            raise ValueError(f"score must be in [{SCORE_MIN}, {SCORE_MAX}]")
        return value


class OpinionClusterAssessment(BaseModel):
    """A single agent call's scores for every leaf of an opinion cluster.

    This is the compute-saving counterpart to OpinionAssessment: one call
    covers the whole issue domain. Stage 05 expands `leaf_scores` into the
    standard per-leaf DeltaRecord / SemRow long table.
    """

    scenario_id: str
    # Network phases are emitted only by the additive cluster network-exposure
    # stages (02b / 04b); the core run only emits "baseline" / "post_attack".
    phase: Literal["baseline", "network_exposure_baseline", "post_attack", "post_attack_network_exposure"]
    cluster_key: str
    leaf_scores: List[ClusterLeafScore] = Field(default_factory=list)
    model_name: str


class AttackExposure(BaseModel):
    scenario_id: str
    attack_present: bool
    attack_leaf: Optional[str] = None
    exposure_text: str
    platform: str
    persuasion_strategy: str
    intensity_hint: float = Field(ge=0.0, le=1.0)
    model_name: str


class DeltaRecord(BaseModel):
    scenario_id: str
    opinion_leaf: str
    baseline_score: int
    post_score: int
    delta_score: int
    abs_delta_score: int
    adversarial_effectivity: Optional[float] = None
    attack_present: bool
    attack_leaf: Optional[str] = None
    profile_id: str
    profile_categorical: Dict[str, str] = Field(default_factory=dict)
    profile_continuous: Dict[str, float] = Field(default_factory=dict)


class SemRow(BaseModel):
    scenario_id: str
    opinion_leaf: str
    baseline_score: float
    post_score: float
    delta_score: float
    abs_delta_score: float
    adversarial_effectivity: Optional[float] = None
    attack_present: int
    attack_leaf: str
    profile_id: str
    profile_features: Dict[str, float] = Field(default_factory=dict)


class SemCoefficient(BaseModel):
    lhs: str
    op: str
    rhs: str
    estimate: float
    std_error: Optional[float] = None
    z_value: Optional[float] = None
    p_value: Optional[float] = None


class SemFitResult(BaseModel):
    model_name: str
    model_formula: str
    converged: bool
    n_obs: int
    fit_indices: Dict[str, Any] = Field(default_factory=dict)
    coefficients: List[SemCoefficient] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ConditionalSusceptibilityTaskModel(BaseModel):
    task_key: str
    attack_leaf: str
    opinion_leaf: str
    outcome_metric: str
    n_obs: int
    alpha: float
    cv_mse: float
    reliability_weight: float
    intercept: float
    coefficients: Dict[str, float] = Field(default_factory=dict)


class ConditionalSusceptibilityArtifact(BaseModel):
    model_name: str = "conditional_profile_susceptibility_index"
    outcome_metric: str = "abs_delta_score"
    attack_leaves: List[str] = Field(default_factory=list)
    opinion_leaves: List[str] = Field(default_factory=list)
    task_weighting_scheme: str = "n_obs_over_cv_mse"
    feature_columns: List[str] = Field(default_factory=list)
    continuous_feature_columns: List[str] = Field(default_factory=list)
    categorical_feature_columns: List[str] = Field(default_factory=list)
    excluded_feature_columns: List[str] = Field(default_factory=list)
    feature_means: Dict[str, float] = Field(default_factory=dict)
    feature_stds: Dict[str, float] = Field(default_factory=dict)
    task_models: List[ConditionalSusceptibilityTaskModel] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class StageArtifactManifest(BaseModel):
    stage_id: str
    stage_name: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    input_path: Optional[str] = None
    primary_output_path: str
    output_files: List[str] = Field(default_factory=list)
    record_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StageConfig(BaseModel):
    stage_name: str
    run_id: str = "run_1"
    seed: int = 42
    use_test_ontology: bool = True
    ontology_root: Optional[str] = None
    openrouter_model: Optional[str] = None
    temperature: float = 0.2
    max_repair_iter: int = 2
    save_raw_llm: bool = False
    raw_llm_dir: Optional[str] = None
    timeout_sec: int = 90
    max_concurrency: int = 1
    primary_moderator: Optional[str] = None
    bootstrap_samples: int = 500
    paper_title: Optional[str] = None
    report_root: Optional[str] = None
    report_assets_root: Optional[str] = None
    export_static_figures: bool = True
    build_report: bool = True
