export interface AffinityComponents {
  categorical_similarity: number;
  personality_similarity: number;
  age_context_similarity: number;
  ontology_leaf_overlap: number;
}

export interface AffinityFormulaWeights {
  personality_similarity: number;
  ontology_leaf_overlap: number;
  age_context_similarity: number;
  categorical_similarity: number;
}

export interface AffinityFormulaComponent {
  key: keyof AffinityFormulaWeights;
  label: string;
  description: string;
}

export interface AffinityFormulaMetadata {
  label: string;
  default_weights: AffinityFormulaWeights;
  components: AffinityFormulaComponent[];
  warning: string;
  note: string;
}

export interface BaselineResult {
  profile_id: string;
  scenario_id: string;
  opinion_leaf: string;
  score: number;
  confidence: number;
  reasoning: string;
  model_name: string;
  call_id: string;
  timestamp: string;
}

export interface AttackOption {
  path: string;
  label: string;
  family: string;
  complexity_tier: string;
  temporal_horizon: string;
  epistemic_target: string;
  compatible: boolean;
  notes: string[];
}

export interface AttackOptionsResponse {
  run_id: string;
  opinion_leaf: string;
  attack_options: AttackOption[];
}

export interface ProfileMeasurementResult {
  phase: "baseline" | "network" | "post" | "post_network";
  profile_id: string;
  scenario_id: string;
  opinion_leaf: string;
  score: number;
  confidence: number;
  reasoning: string;
  model_name: string;
  call_id: string;
  timestamp: string;
  attack_leaf?: string;
  baseline_score?: number;
  network_score?: number;
  post_score?: number;
  private_post_score?: number;
  post_attack_network_score?: number;
  delta_score?: number;
  delta_from_baseline?: number;
  increment_from_private_post?: number;
  adversarial_direction?: number;
  post_direction_clamped?: boolean;
  network_context?: Record<string, unknown>;
  post_attack_network_context?: Record<string, unknown>;
}

export interface ProfileNetworkNode {
  id: string;
  label: string;
  cluster_id: string;
  categorical_attributes: Record<string, string>;
  continuous_attributes: Record<string, number>;
  selected_leaf_nodes: string[];
  metadata: Record<string, unknown>;
  centrality: number;
  baseline?: BaselineResult | null;
}

export interface ProfileNetworkEdge {
  source: string;
  target: string;
  affinity: number;
  normalized_affinity: number;
  components: AffinityComponents;
  weight?: number | null;
  normalized_weight?: number | null;
  edge_kind?: string;
  directed?: boolean;
  source_position_id?: string | null;
  target_position_id?: string | null;
  exposure_weight?: number | null;
  interaction_types?: string | null;
  rank_for_receiver?: number | null;
}

export interface ProfileLayoutAffinity {
  source: string;
  target: string;
  affinity: number;
  components: AffinityComponents;
  weight?: number | null;
  edge_kind?: string;
  directed?: boolean;
  source_position_id?: string | null;
  target_position_id?: string | null;
  exposure_weight?: number | null;
  interaction_types?: string | null;
  rank_for_receiver?: number | null;
}

export interface OpinionLeafOption {
  path: string;
  label: string;
  domain?: string | null;
}

export interface ProfileNetworkResponse {
  run_id: string;
  mode: "test" | "production";
  nodes: ProfileNetworkNode[];
  edges: ProfileNetworkEdge[];
  layout_affinities: ProfileLayoutAffinity[];
  affinity_formula: AffinityFormulaMetadata;
  opinion_leaves: OpinionLeafOption[];
  diagnostics: {
    profile_count: number;
    full_pair_count: number;
    displayed_edge_count: number;
    edge_limit_per_node: number;
    affinity_min: number | null;
    affinity_max: number | null;
    affinity_mean: number | null;
    edge_semantics: string;
    empirical_edge_count?: number | null;
    assigned_profile_count?: number | null;
    community_count?: number | null;
    prompt_ready_count?: number | null;
  };
  provenance: {
    run_id: string;
    mode: "test" | "production";
    config_path: string;
    ontology_root: string;
    source: string;
    seed: number;
    model_name?: string | null;
    graph_id?: string | null;
    graph_root?: string | null;
    network_basis?: string | null;
  };
  warnings: string[];
}

export interface PipelineViewStageStatus {
  stage_id: string;
  stage_name: string;
  available: boolean;
  manifest_path?: string | null;
  primary_output_path?: string | null;
  record_count?: number | null;
  created_at_utc?: string | null;
  live_available: boolean;
  live_status?: string | null;
  live_result_count?: number | null;
  source: "canonical" | "live_sidecar" | "missing";
}

export interface BaselineRunCreateResponse {
  baseline_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  opinion_leaf: string;
  profile_count: number;
  model_name: string;
}

export type BaselineRunStatus = "queued" | "running" | "completed" | "failed" | "completed_with_errors";

export interface BaselineRunError {
  profile_id: string;
  scenario_id: string;
  message: string;
  timestamp: string;
}

export interface BaselineRunResponse {
  baseline_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  opinion_leaf: string;
  model_name: string;
  profile_count: number;
  completed_count: number;
  failed_count: number;
  results: BaselineResult[];
  errors: BaselineRunError[];
  artifact_dir: string;
  raw_llm_dir: string;
  created_at: string;
  updated_at: string;
}

export interface BaselinePromptMessage {
  role: "system" | "user";
  content: string;
}

export interface BaselinePromptPreviewResponse {
  run_id: string;
  profile_id: string;
  opinion_leaf: string;
  prompt_name: "baseline_opinion.md";
  model_name: string;
  system_prompt: string;
  user_payload: Record<string, unknown>;
  messages: BaselinePromptMessage[];
}

export interface PostExposureResult {
  profile_id: string;
  scenario_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  baseline_score: number;
  post_score: number;
  delta_score: number;
  adversarial_direction: number;
  confidence: number;
  reasoning: string;
  model_name: string;
  call_id: string;
  timestamp: string;
  heuristic_checks: Record<string, unknown>;
  post_direction_clamped: boolean;
}

export interface PostExposureRunCreateResponse {
  post_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  baseline_run_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  profile_count: number;
  model_name: string;
}

export interface PostExposureRunError {
  profile_id: string;
  scenario_id: string;
  message: string;
  timestamp: string;
}

export interface PostExposureRunResponse {
  post_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  baseline_run_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  model_name: string;
  profile_count: number;
  completed_count: number;
  failed_count: number;
  results: PostExposureResult[];
  errors: PostExposureRunError[];
  artifact_dir: string;
  raw_llm_dir: string;
  attack_specs_path: string;
  created_at: string;
  updated_at: string;
}

export interface PostExposurePromptPreviewResponse {
  run_id: string;
  baseline_run_id: string;
  profile_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  prompt_name: "post_attack_opinion.md";
  model_name: string;
  system_prompt: string;
  user_payload: Record<string, unknown>;
  messages: BaselinePromptMessage[];
}

export interface NetworkExposureResult {
  profile_id: string;
  scenario_id: string;
  opinion_leaf: string;
  baseline_score: number;
  network_score: number;
  delta_score: number;
  confidence: number;
  reasoning: string;
  model_name: string;
  call_id: string;
  timestamp: string;
  network_context: Record<string, unknown>;
}

export interface NetworkExposureRunCreateResponse {
  network_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  baseline_run_id: string;
  opinion_leaf: string;
  profile_count: number;
  model_name: string;
  top_k: number;
}

export interface NetworkExposureRunError {
  profile_id: string;
  scenario_id: string;
  message: string;
  timestamp: string;
}

export interface NetworkExposureRunResponse {
  network_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  baseline_run_id: string;
  opinion_leaf: string;
  model_name: string;
  profile_count: number;
  completed_count: number;
  failed_count: number;
  top_k: number;
  results: NetworkExposureResult[];
  errors: NetworkExposureRunError[];
  artifact_dir: string;
  raw_llm_dir: string;
  network_contexts_path: string;
  created_at: string;
  updated_at: string;
}

export interface NetworkExposurePromptPreviewResponse {
  run_id: string;
  baseline_run_id: string;
  profile_id: string;
  opinion_leaf: string;
  prompt_name: "network_exposure_opinion.md";
  model_name: string;
  system_prompt: string;
  user_payload: Record<string, unknown>;
  messages: BaselinePromptMessage[];
}

export interface PostAttackNetworkExposureResult {
  profile_id: string;
  scenario_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  baseline_score: number;
  private_post_score: number;
  post_attack_network_score: number;
  delta_from_baseline: number;
  increment_from_private_post: number;
  adversarial_direction: number;
  confidence: number;
  reasoning: string;
  model_name: string;
  call_id: string;
  timestamp: string;
  post_attack_network_context: Record<string, unknown>;
  heuristic_checks: Record<string, unknown>;
}

export interface PostAttackNetworkExposureRunCreateResponse {
  post_network_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  baseline_run_id: string;
  post_run_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  profile_count: number;
  model_name: string;
  top_k: number;
}

export interface PostAttackNetworkExposureRunError {
  profile_id: string;
  scenario_id: string;
  message: string;
  timestamp: string;
}

export interface PostAttackNetworkExposureRunResponse {
  post_network_run_id: string;
  status: BaselineRunStatus;
  run_id: string;
  baseline_run_id: string;
  post_run_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  model_name: string;
  profile_count: number;
  completed_count: number;
  failed_count: number;
  top_k: number;
  results: PostAttackNetworkExposureResult[];
  errors: PostAttackNetworkExposureRunError[];
  artifact_dir: string;
  raw_llm_dir: string;
  post_attack_network_contexts_path: string;
  created_at: string;
  updated_at: string;
}

export interface PostAttackNetworkExposurePromptPreviewResponse {
  run_id: string;
  baseline_run_id: string;
  post_run_id: string;
  profile_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  prompt_name: "post_attack_network_exposure_opinion.md";
  model_name: string;
  system_prompt: string;
  user_payload: Record<string, unknown>;
  messages: BaselinePromptMessage[];
}

export interface PipelineViewResponse {
  run_id: string;
  network: ProfileNetworkResponse;
  attack_options: AttackOption[];
  baseline_run: BaselineRunResponse | null;
  network_run: NetworkExposureRunResponse | null;
  post_run: PostExposureRunResponse | null;
  post_network_run: PostAttackNetworkExposureRunResponse | null;
  stage_status: PipelineViewStageStatus[];
  selected_opinion_leaf?: string | null;
  selected_attack_leaf?: string | null;
  warnings: string[];
}

export type PromptPreviewResponse =
  | BaselinePromptPreviewResponse
  | NetworkExposurePromptPreviewResponse
  | PostExposurePromptPreviewResponse
  | PostAttackNetworkExposurePromptPreviewResponse;
