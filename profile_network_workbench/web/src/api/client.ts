import type {
  AttackOptionsResponse,
  BaselinePromptPreviewResponse,
  BaselineRunCreateResponse,
  BaselineRunResponse,
  NetworkExposurePromptPreviewResponse,
  NetworkExposureRunCreateResponse,
  NetworkExposureRunResponse,
  PostAttackNetworkExposurePromptPreviewResponse,
  PostAttackNetworkExposureRunCreateResponse,
  PostAttackNetworkExposureRunResponse,
  PostExposurePromptPreviewResponse,
  PostExposureRunCreateResponse,
  PostExposureRunResponse,
  PipelineViewResponse,
  ProfileNetworkResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8013";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // Keep the HTTP status text when the server does not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function fetchProfileNetwork(runId: string, edgeLimitPerNode: number) {
  const params = new URLSearchParams({ run_id: runId, edge_limit_per_node: String(edgeLimitPerNode) });
  return request<ProfileNetworkResponse>(`/api/profile-network?${params.toString()}`);
}

export function fetchPipelineView(
  runId: string,
  edgeLimitPerNode: number,
  opinionLeaf?: string,
  attackLeaf?: string
) {
  const params = new URLSearchParams({ run_id: runId, edge_limit_per_node: String(edgeLimitPerNode) });
  if (opinionLeaf) params.set("opinion_leaf", opinionLeaf);
  if (attackLeaf) params.set("attack_leaf", attackLeaf);
  return request<PipelineViewResponse>(`/api/pipeline-view?${params.toString()}`);
}

export function createBaselineRun(payload: {
  run_id: string;
  opinion_leaf: string;
  profile_ids?: string[];
  model_name?: string;
  max_concurrency?: number;
}) {
  return request<BaselineRunCreateResponse>("/api/baseline-runs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchBaselineRun(baselineRunId: string) {
  return request<BaselineRunResponse>(`/api/baseline-runs/${baselineRunId}`);
}

export function fetchBaselinePromptPreview(runId: string, opinionLeaf: string, profileId: string) {
  const params = new URLSearchParams({ run_id: runId, opinion_leaf: opinionLeaf, profile_id: profileId });
  return request<BaselinePromptPreviewResponse>(`/api/baseline-prompt-preview?${params.toString()}`);
}

export function createNetworkExposureRun(payload: {
  run_id: string;
  baseline_run_id: string;
  opinion_leaf: string;
  profile_ids?: string[];
  model_name?: string;
  max_concurrency?: number;
  top_k?: number;
}) {
  return request<NetworkExposureRunCreateResponse>("/api/network-exposure-runs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchNetworkExposureRun(networkRunId: string) {
  return request<NetworkExposureRunResponse>(`/api/network-exposure-runs/${networkRunId}`);
}

export function fetchNetworkExposurePromptPreview(
  runId: string,
  baselineRunId: string,
  opinionLeaf: string,
  profileId: string,
  topK = 8
) {
  const params = new URLSearchParams({
    run_id: runId,
    baseline_run_id: baselineRunId,
    opinion_leaf: opinionLeaf,
    profile_id: profileId,
    top_k: String(topK)
  });
  return request<NetworkExposurePromptPreviewResponse>(`/api/network-exposure-prompt-preview?${params.toString()}`);
}

export function fetchAttackOptions(runId: string, opinionLeaf: string) {
  const params = new URLSearchParams({ run_id: runId, opinion_leaf: opinionLeaf });
  return request<AttackOptionsResponse>(`/api/attack-options?${params.toString()}`);
}

export function createPostExposureRun(payload: {
  run_id: string;
  baseline_run_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  profile_ids?: string[];
  model_name?: string;
  max_concurrency?: number;
}) {
  return request<PostExposureRunCreateResponse>("/api/post-exposure-runs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchPostExposureRun(postRunId: string) {
  return request<PostExposureRunResponse>(`/api/post-exposure-runs/${postRunId}`);
}

export function fetchPostExposurePromptPreview(
  runId: string,
  baselineRunId: string,
  opinionLeaf: string,
  attackLeaf: string,
  profileId: string
) {
  const params = new URLSearchParams({
    run_id: runId,
    baseline_run_id: baselineRunId,
    opinion_leaf: opinionLeaf,
    attack_leaf: attackLeaf,
    profile_id: profileId
  });
  return request<PostExposurePromptPreviewResponse>(`/api/post-exposure-prompt-preview?${params.toString()}`);
}

export function createPostAttackNetworkExposureRun(payload: {
  run_id: string;
  baseline_run_id: string;
  post_run_id: string;
  opinion_leaf: string;
  attack_leaf: string;
  profile_ids?: string[];
  model_name?: string;
  max_concurrency?: number;
  top_k?: number;
}) {
  return request<PostAttackNetworkExposureRunCreateResponse>("/api/post-attack-network-exposure-runs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchPostAttackNetworkExposureRun(postNetworkRunId: string) {
  return request<PostAttackNetworkExposureRunResponse>(`/api/post-attack-network-exposure-runs/${postNetworkRunId}`);
}

export function fetchPostAttackNetworkExposurePromptPreview(
  runId: string,
  baselineRunId: string,
  postRunId: string,
  opinionLeaf: string,
  attackLeaf: string,
  profileId: string,
  topK = 8
) {
  const params = new URLSearchParams({
    run_id: runId,
    baseline_run_id: baselineRunId,
    post_run_id: postRunId,
    opinion_leaf: opinionLeaf,
    attack_leaf: attackLeaf,
    profile_id: profileId,
    top_k: String(topK)
  });
  return request<PostAttackNetworkExposurePromptPreviewResponse>(
    `/api/post-attack-network-exposure-prompt-preview?${params.toString()}`
  );
}
