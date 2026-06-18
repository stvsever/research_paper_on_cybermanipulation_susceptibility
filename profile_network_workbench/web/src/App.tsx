import { useEffect, useMemo, useState } from "react";
import { Network, RefreshCw, RotateCcw, X } from "lucide-react";

import {
  createBaselineRun,
  createNetworkExposureRun,
  createPostAttackNetworkExposureRun,
  createPostExposureRun,
  fetchAttackOptions,
  fetchBaselineRun,
  fetchNetworkExposureRun,
  fetchPipelineView,
  fetchPostAttackNetworkExposureRun,
  fetchPostExposureRun,
  fetchProfileNetwork
} from "./api/client";
import type {
  AffinityFormulaWeights,
  AttackOption,
  BaselineResult,
  BaselineRunResponse,
  NetworkExposureResult,
  NetworkExposureRunResponse,
  PostAttackNetworkExposureResult,
  PostAttackNetworkExposureRunResponse,
  PostExposureResult,
  PostExposureRunResponse,
  PipelineViewStageStatus,
  ProfileMeasurementResult,
  ProfileNetworkEdge,
  ProfileNetworkNode,
  ProfileNetworkResponse
} from "./api/types";
import { MeasurementElicitationCard, type MeasurementMode } from "./components/MeasurementElicitationCard";
import { ProfileNetworkCanvas } from "./components/ProfileNetworkCanvas";
import { PromptPreviewModal } from "./components/PromptPreviewModal";
import { buildWeightedNetwork, DEFAULT_AFFINITY_WEIGHTS, normalizeAffinityWeights } from "./lib/affinity";

const DEFAULT_RUN_ID = "run_1";
type ExecutionMode = "workbench_live" | "pipeline_view";
const FORMULA_KEYS: Array<keyof AffinityFormulaWeights> = [
  "personality_similarity",
  "ontology_leaf_overlap",
  "age_context_similarity",
  "categorical_similarity"
];
const FORMULA_LABELS: Record<keyof AffinityFormulaWeights, string> = {
  personality_similarity: "Personality",
  ontology_leaf_overlap: "Ontology overlap",
  age_context_similarity: "Age/context",
  categorical_similarity: "Categorical"
};

function leafLabel(path: string) {
  const parts = path.split(">").map((part) => part.trim()).filter(Boolean);
  return parts.at(-1) ?? path;
}

function formatScore(score?: number) {
  if (score === undefined) return "--";
  return score > 0 ? `+${score}` : String(score);
}

function formatDelta(delta?: number) {
  if (delta === undefined) return "--";
  return delta > 0 ? `+${delta}` : String(delta);
}

function numericContextValue(context: Record<string, unknown> | undefined, key: string) {
  const value = context?.[key];
  return typeof value === "number" ? value : undefined;
}

function firstNumericContextValue(context: Record<string, unknown> | undefined, keys: string[]) {
  for (const key of keys) {
    const value = numericContextValue(context, key);
    if (value !== undefined) return value;
  }
  return undefined;
}

function peerCount(context: Record<string, unknown> | undefined) {
  const peers = context?.peer_exemplars ?? context?.peer_assessments;
  return Array.isArray(peers) ? peers.length : undefined;
}

function edgeWeight(edge: ProfileNetworkEdge) {
  return edge.weight ?? edge.exposure_weight ?? edge.affinity;
}

function formatMetric(value: unknown, digits = 3) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string" && value) return value;
  return "--";
}

function exposureAssignment(node: ProfileNetworkNode | undefined) {
  const assignment = node?.metadata?.exposure_network_assignment;
  return assignment && typeof assignment === "object" && !Array.isArray(assignment) ? (assignment as Record<string, unknown>) : undefined;
}

function terminal(status?: string) {
  return status === "completed" || status === "failed" || status === "completed_with_errors";
}

function pipelineStageDisplay(stage: PipelineViewStageStatus) {
  if (stage.available) return { className: "ready", label: "available" };
  if (stage.live_available) {
    const status = stage.live_status ?? "running";
    const count = typeof stage.live_result_count === "number" ? ` ${stage.live_result_count}` : "";
    if (status === "completed") return { className: "live", label: `live done${count}` };
    if (status === "completed_with_errors") return { className: "warning", label: `live errors${count}` };
    if (status === "failed") return { className: "missing", label: "failed" };
    return { className: "running", label: `${status}${count}` };
  }
  return { className: "missing", label: "missing" };
}

const MEASUREMENT_STAGE_ID: Record<MeasurementMode, string> = {
  baseline: "02",
  network: "02b",
  post: "04",
  post_network: "04b"
};

function useDebouncedValue<T>(value: T, delayMs: number) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AttributeList({ title, items, max = 10 }: { title: string; items: Record<string, string | number>; max?: number }) {
  const rows = Object.entries(items).slice(0, max);
  return (
    <section className="panel-section">
      <h3>{title}</h3>
      {rows.length ? (
        <div className="attribute-list">
          {rows.map(([key, value]) => (
            <div className="attribute-row" key={key}>
              <span>{key}</span>
              <strong>{typeof value === "number" ? value.toFixed(2) : value}</strong>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">No attributes available.</p>
      )}
    </section>
  );
}

export function App() {
  const [runId, setRunId] = useState(DEFAULT_RUN_ID);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>("workbench_live");
  const [displayDataSource, setDisplayDataSource] = useState<ExecutionMode>("workbench_live");
  const [edgeLimit, setEdgeLimit] = useState(6);
  const [network, setNetwork] = useState<ProfileNetworkResponse | null>(null);
  const [formulaWeights, setFormulaWeights] = useState<AffinityFormulaWeights>(DEFAULT_AFFINITY_WEIGHTS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [selectedOpinion, setSelectedOpinion] = useState("");
  const [measurementMode, setMeasurementMode] = useState<MeasurementMode>("baseline");
  const [attackOptions, setAttackOptions] = useState<AttackOption[]>([]);
  const [selectedAttack, setSelectedAttack] = useState("");
  const [baselineRun, setBaselineRun] = useState<BaselineRunResponse | null>(null);
  const [networkRun, setNetworkRun] = useState<NetworkExposureRunResponse | null>(null);
  const [postRun, setPostRun] = useState<PostExposureRunResponse | null>(null);
  const [postNetworkRun, setPostNetworkRun] = useState<PostAttackNetworkExposureRunResponse | null>(null);
  const [baselineError, setBaselineError] = useState("");
  const [networkError, setNetworkError] = useState("");
  const [postError, setPostError] = useState("");
  const [postNetworkError, setPostNetworkError] = useState("");
  const [detailProfileId, setDetailProfileId] = useState("");
  const [promptPreviewOpen, setPromptPreviewOpen] = useState(false);
  const [pipelineStageStatus, setPipelineStageStatus] = useState<PipelineViewStageStatus[]>([]);
  const [pipelineWarnings, setPipelineWarnings] = useState<string[]>([]);

  const matchingBaselineRun = baselineRun?.opinion_leaf === selectedOpinion ? baselineRun : null;
  const matchingNetworkRun =
    networkRun?.opinion_leaf === selectedOpinion && networkRun.baseline_run_id === matchingBaselineRun?.baseline_run_id
      ? networkRun
      : null;
  const matchingPostRun =
    postRun?.opinion_leaf === selectedOpinion && postRun.attack_leaf === selectedAttack ? postRun : null;
  const matchingPostNetworkRun =
    postNetworkRun?.opinion_leaf === selectedOpinion &&
    postNetworkRun.attack_leaf === selectedAttack &&
    postNetworkRun.post_run_id === matchingPostRun?.post_run_id
      ? postNetworkRun
      : null;

  const baselineByProfile = useMemo(() => {
    const map = new Map<string, BaselineResult>();
    matchingBaselineRun?.results.forEach((result) => map.set(result.profile_id, result));
    return map;
  }, [matchingBaselineRun]);

  const postByProfile = useMemo(() => {
    const map = new Map<string, PostExposureResult>();
    matchingPostRun?.results.forEach((result) => map.set(result.profile_id, result));
    return map;
  }, [matchingPostRun]);

  const networkByProfile = useMemo(() => {
    const map = new Map<string, NetworkExposureResult>();
    matchingNetworkRun?.results.forEach((result) => map.set(result.profile_id, result));
    return map;
  }, [matchingNetworkRun]);

  const postNetworkByProfile = useMemo(() => {
    const map = new Map<string, PostAttackNetworkExposureResult>();
    matchingPostNetworkRun?.results.forEach((result) => map.set(result.profile_id, result));
    return map;
  }, [matchingPostNetworkRun]);

  const measurementByProfile = useMemo(() => {
    const map = new Map<string, ProfileMeasurementResult>();
    if (measurementMode === "post_network") {
      postNetworkByProfile.forEach((result, profileId) => {
        map.set(profileId, {
          phase: "post_network",
          profile_id: result.profile_id,
          scenario_id: result.scenario_id,
          opinion_leaf: result.opinion_leaf,
          attack_leaf: result.attack_leaf,
          baseline_score: result.baseline_score,
          private_post_score: result.private_post_score,
          post_attack_network_score: result.post_attack_network_score,
          delta_score: result.delta_from_baseline,
          delta_from_baseline: result.delta_from_baseline,
          increment_from_private_post: result.increment_from_private_post,
          adversarial_direction: result.adversarial_direction,
          score: result.post_attack_network_score,
          confidence: result.confidence,
          reasoning: result.reasoning,
          model_name: result.model_name,
          call_id: result.call_id,
          timestamp: result.timestamp,
          post_attack_network_context: result.post_attack_network_context
        });
      });
      return map;
    }
    if (measurementMode === "network") {
      networkByProfile.forEach((result, profileId) => {
        map.set(profileId, {
          phase: "network",
          profile_id: result.profile_id,
          scenario_id: result.scenario_id,
          opinion_leaf: result.opinion_leaf,
          baseline_score: result.baseline_score,
          network_score: result.network_score,
          delta_score: result.delta_score,
          score: result.network_score,
          confidence: result.confidence,
          reasoning: result.reasoning,
          model_name: result.model_name,
          call_id: result.call_id,
          timestamp: result.timestamp,
          network_context: result.network_context
        });
      });
      return map;
    }
    if (measurementMode === "post") {
      postByProfile.forEach((result, profileId) => {
        map.set(profileId, {
          phase: "post",
          profile_id: result.profile_id,
          scenario_id: result.scenario_id,
          opinion_leaf: result.opinion_leaf,
          attack_leaf: result.attack_leaf,
          baseline_score: result.baseline_score,
          post_score: result.post_score,
          delta_score: result.delta_score,
          adversarial_direction: result.adversarial_direction,
          post_direction_clamped: result.post_direction_clamped,
          score: result.post_score,
          confidence: result.confidence,
          reasoning: result.reasoning,
          model_name: result.model_name,
          call_id: result.call_id,
          timestamp: result.timestamp
        });
      });
      return map;
    }
    baselineByProfile.forEach((result, profileId) => {
      map.set(profileId, {
        phase: "baseline",
        profile_id: result.profile_id,
        scenario_id: result.scenario_id,
        opinion_leaf: result.opinion_leaf,
        score: result.score,
        confidence: result.confidence,
        reasoning: result.reasoning,
        model_name: result.model_name,
        call_id: result.call_id,
        timestamp: result.timestamp
      });
    });
    return map;
  }, [baselineByProfile, measurementMode, networkByProfile, postByProfile, postNetworkByProfile]);

  const defaultFormulaWeights = network?.affinity_formula.default_weights ?? DEFAULT_AFFINITY_WEIGHTS;
  const normalizedFormulaWeights = useMemo(
    () => normalizeAffinityWeights(formulaWeights, defaultFormulaWeights),
    [formulaWeights, defaultFormulaWeights]
  );
  const debouncedFormulaWeights = useDebouncedValue(formulaWeights, 90);
  const appliedFormulaWeights = useMemo(
    () => normalizeAffinityWeights(debouncedFormulaWeights, defaultFormulaWeights),
    [debouncedFormulaWeights, defaultFormulaWeights]
  );
  const activeNetwork = useMemo(
    () => (network ? (executionMode === "pipeline_view" ? network : buildWeightedNetwork(network, appliedFormulaWeights, edgeLimit)) : null),
    [network, appliedFormulaWeights, edgeLimit, executionMode]
  );

  const selectedNode: ProfileNetworkNode | undefined = useMemo(() => {
    if (!activeNetwork) return undefined;
    return activeNetwork.nodes.find((node) => node.id === selectedId) ?? activeNetwork.nodes[0];
  }, [activeNetwork, selectedId]);

  const selectedBaseline = selectedNode ? baselineByProfile.get(selectedNode.id) : undefined;
  const selectedNetwork = selectedNode ? networkByProfile.get(selectedNode.id) : undefined;
  const selectedPost = selectedNode ? postByProfile.get(selectedNode.id) : undefined;
  const selectedPostNetwork = selectedNode ? postNetworkByProfile.get(selectedNode.id) : undefined;
  const selectedMeasurement = selectedNode ? measurementByProfile.get(selectedNode.id) : undefined;
  const selectedExposureAssignment = exposureAssignment(selectedNode);
  const detailMeasurement = detailProfileId ? measurementByProfile.get(detailProfileId) : undefined;
  const detailNode = detailProfileId && activeNetwork ? activeNetwork.nodes.find((node) => node.id === detailProfileId) : undefined;
  const selectedAttackOption = attackOptions.find((option) => option.path === selectedAttack);
  const nearestNeighbors = useMemo(() => {
    if (!activeNetwork || !selectedNode) return [];
    return activeNetwork.edges
      .filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id)
      .sort((left, right) => edgeWeight(right) - edgeWeight(left))
      .slice(0, 8)
      .map((edge) => ({
        id: edge.source === selectedNode.id ? edge.target : edge.source,
        weight: edgeWeight(edge),
        direction: edge.source === selectedNode.id ? "outgoing" : "incoming"
      }));
  }, [activeNetwork, selectedNode]);
  const activePipelineStage = pipelineStageStatus.find((stage) => stage.stage_id === MEASUREMENT_STAGE_ID[measurementMode]);
  const pipelineSourceLabel =
    executionMode === "pipeline_view" && activePipelineStage?.source === "live_sidecar"
      ? "live pipeline sidecar"
      : undefined;

  async function loadNetwork(overrides?: { opinionLeaf?: string; attackLeaf?: string }, options?: { quiet?: boolean }) {
    if (!options?.quiet) setLoading(true);
    setError("");
    try {
      const requestedOpinion = overrides?.opinionLeaf ?? selectedOpinion;
      const requestedAttack = overrides?.attackLeaf ?? selectedAttack;
      if (executionMode === "pipeline_view") {
        const payload = await fetchPipelineView(runId, edgeLimit, requestedOpinion || undefined, requestedAttack || undefined);
        setNetwork(payload.network);
        setFormulaWeights(payload.network.affinity_formula.default_weights ?? DEFAULT_AFFINITY_WEIGHTS);
        setSelectedId((current) => current || payload.network.nodes[0]?.id || "");
        setSelectedOpinion(payload.selected_opinion_leaf || payload.network.opinion_leaves[0]?.path || "");
        setSelectedAttack(payload.selected_attack_leaf || "");
        setAttackOptions(payload.attack_options);
        setBaselineRun(payload.baseline_run);
        setNetworkRun(payload.network_run);
        setPostRun(payload.post_run);
        setPostNetworkRun(payload.post_network_run);
        setPipelineStageStatus(payload.stage_status);
        setPipelineWarnings(payload.warnings);
        setDisplayDataSource("pipeline_view");
      } else {
        const payload = await fetchProfileNetwork(runId, edgeLimit);
        setNetwork(payload);
        setFormulaWeights(payload.affinity_formula.default_weights ?? DEFAULT_AFFINITY_WEIGHTS);
        setSelectedId((current) => current || payload.nodes[0]?.id || "");
        setSelectedOpinion((current) => current || payload.opinion_leaves[0]?.path || "");
        setPipelineStageStatus([]);
        setPipelineWarnings([]);
        if (displayDataSource === "pipeline_view") {
          setBaselineRun(null);
          setNetworkRun(null);
          setPostRun(null);
          setPostNetworkRun(null);
        }
        setDisplayDataSource("workbench_live");
      }
    } catch (exc) {
      if (executionMode === "pipeline_view") {
        setNetwork(null);
        setSelectedId("");
        setAttackOptions([]);
        setBaselineRun(null);
        setNetworkRun(null);
        setPostRun(null);
        setPostNetworkRun(null);
        setPipelineStageStatus([]);
        setPipelineWarnings([]);
        setDisplayDataSource("pipeline_view");
      }
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      if (!options?.quiet) setLoading(false);
    }
  }

  useEffect(() => {
    void loadNetwork();
  }, [executionMode]);

  const pipelineLiveActive = useMemo(
    () =>
      executionMode === "pipeline_view" &&
      pipelineStageStatus.some(
        (stage) =>
          stage.live_available &&
          stage.live_status !== "completed" &&
          stage.live_status !== "failed" &&
          stage.live_status !== "completed_with_errors"
      ),
    [executionMode, pipelineStageStatus]
  );

  useEffect(() => {
    if (!pipelineLiveActive) return;
    const timer = window.setInterval(() => {
      void loadNetwork(undefined, { quiet: true });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [pipelineLiveActive, runId, edgeLimit, selectedOpinion, selectedAttack]);

  function handleOpinionChange(opinionLeaf: string) {
    setSelectedOpinion(opinionLeaf);
    if (executionMode === "pipeline_view") {
      void loadNetwork({ opinionLeaf });
    }
  }

  function handleAttackChange(attackLeaf: string) {
    setSelectedAttack(attackLeaf);
    if (executionMode === "pipeline_view") {
      void loadNetwork({ attackLeaf });
    }
  }

  useEffect(() => {
    if (!baselineRun || terminal(baselineRun.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const payload = await fetchBaselineRun(baselineRun.baseline_run_id);
        setBaselineRun(payload);
      } catch (exc) {
        setBaselineError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [baselineRun]);

  useEffect(() => {
    if (!postRun || terminal(postRun.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const payload = await fetchPostExposureRun(postRun.post_run_id);
        setPostRun(payload);
      } catch (exc) {
        setPostError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [postRun]);

  useEffect(() => {
    if (!networkRun || terminal(networkRun.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const payload = await fetchNetworkExposureRun(networkRun.network_run_id);
        setNetworkRun(payload);
      } catch (exc) {
        setNetworkError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [networkRun]);

  useEffect(() => {
    if (!postNetworkRun || terminal(postNetworkRun.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const payload = await fetchPostAttackNetworkExposureRun(postNetworkRun.post_network_run_id);
        setPostNetworkRun(payload);
      } catch (exc) {
        setPostNetworkError(exc instanceof Error ? exc.message : String(exc));
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [postNetworkRun]);

  useEffect(() => {
    if (!network || !selectedOpinion) return;
    if (executionMode === "pipeline_view") return;
    let cancelled = false;
    setPostError("");
    void fetchAttackOptions(network.run_id, selectedOpinion)
      .then((payload) => {
        if (cancelled) return;
        setAttackOptions(payload.attack_options);
        setSelectedAttack((current) => {
          if (payload.attack_options.some((attack) => attack.path === current)) return current;
          return payload.attack_options.find((attack) => attack.compatible)?.path ?? payload.attack_options[0]?.path ?? "";
        });
      })
      .catch((exc) => {
        if (!cancelled) setPostError(exc instanceof Error ? exc.message : String(exc));
      });
    return () => {
      cancelled = true;
    };
  }, [network, selectedOpinion, executionMode]);

  async function runBaseline() {
    if (!network || !selectedOpinion) return;
    setBaselineError("");
    try {
      const created = await createBaselineRun({
        run_id: network.run_id,
        opinion_leaf: selectedOpinion,
        max_concurrency: 6
      });
      const status = await fetchBaselineRun(created.baseline_run_id);
      setBaselineRun(status);
    } catch (exc) {
      setBaselineError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function runPostExposure() {
    if (!network || !selectedOpinion || !selectedAttack || !matchingBaselineRun) return;
    setPostError("");
    try {
      const created = await createPostExposureRun({
        run_id: network.run_id,
        baseline_run_id: matchingBaselineRun.baseline_run_id,
        opinion_leaf: selectedOpinion,
        attack_leaf: selectedAttack,
        max_concurrency: 6
      });
      const status = await fetchPostExposureRun(created.post_run_id);
      setPostRun(status);
    } catch (exc) {
      setPostError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function runNetworkExposure() {
    if (!network || !selectedOpinion || !matchingBaselineRun) return;
    setNetworkError("");
    try {
      const created = await createNetworkExposureRun({
        run_id: network.run_id,
        baseline_run_id: matchingBaselineRun.baseline_run_id,
        opinion_leaf: selectedOpinion,
        max_concurrency: 6,
        top_k: 8
      });
      const status = await fetchNetworkExposureRun(created.network_run_id);
      setNetworkRun(status);
    } catch (exc) {
      setNetworkError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function runPostNetworkExposure() {
    if (!network || !selectedOpinion || !selectedAttack || !matchingBaselineRun || !matchingPostRun) return;
    setPostNetworkError("");
    try {
      const created = await createPostAttackNetworkExposureRun({
        run_id: network.run_id,
        baseline_run_id: matchingBaselineRun.baseline_run_id,
        post_run_id: matchingPostRun.post_run_id,
        opinion_leaf: selectedOpinion,
        attack_leaf: selectedAttack,
        max_concurrency: 6,
        top_k: 8
      });
      const status = await fetchPostAttackNetworkExposureRun(created.post_network_run_id);
      setPostNetworkRun(status);
    } catch (exc) {
      setPostNetworkError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  const baselineProgress = matchingBaselineRun
    ? Math.round(((matchingBaselineRun.completed_count + matchingBaselineRun.failed_count) / Math.max(1, matchingBaselineRun.profile_count)) * 100)
    : 0;
  const postProgress = matchingPostRun
    ? Math.round(((matchingPostRun.completed_count + matchingPostRun.failed_count) / Math.max(1, matchingPostRun.profile_count)) * 100)
    : 0;
  const networkProgress = matchingNetworkRun
    ? Math.round(((matchingNetworkRun.completed_count + matchingNetworkRun.failed_count) / Math.max(1, matchingNetworkRun.profile_count)) * 100)
    : 0;
  const postNetworkProgress = matchingPostNetworkRun
    ? Math.round(
        ((matchingPostNetworkRun.completed_count + matchingPostNetworkRun.failed_count) /
          Math.max(1, matchingPostNetworkRun.profile_count)) *
          100
      )
    : 0;
  const baselineRunning = matchingBaselineRun ? !terminal(matchingBaselineRun.status) : false;
  const networkRunning = matchingNetworkRun ? !terminal(matchingNetworkRun.status) : false;
  const postRunning = matchingPostRun ? !terminal(matchingPostRun.status) : false;
  const postNetworkRunning = matchingPostNetworkRun ? !terminal(matchingPostNetworkRun.status) : false;
  const networkReady = matchingBaselineRun?.status === "completed";
  const postReady = matchingBaselineRun?.status === "completed" && Boolean(selectedAttack);
  const postNetworkReady = matchingBaselineRun?.status === "completed" && matchingPostRun?.status === "completed" && Boolean(selectedAttack);
  const formulaLine = `affinity = ${normalizedFormulaWeights.personality_similarity.toFixed(2)} personality + ${normalizedFormulaWeights.ontology_leaf_overlap.toFixed(2)} ontology + ${normalizedFormulaWeights.age_context_similarity.toFixed(2)} age/context + ${normalizedFormulaWeights.categorical_similarity.toFixed(2)} categorical`;

  function updateFormulaWeight(key: keyof AffinityFormulaWeights, value: number) {
    setFormulaWeights((current) => {
      const next = { ...current, [key]: value };
      const total = FORMULA_KEYS.reduce((sum, item) => sum + Math.max(0, next[item]), 0);
      return total <= 0 ? defaultFormulaWeights : next;
    });
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Network size={22} />
          <div>
            <span>Profile Network</span>
            <strong>Experiment Workbench</strong>
          </div>
        </div>
        <div className="topbar-controls">
          <div className="source-toggle" role="tablist" aria-label="Execution source">
            <button
              className={executionMode === "workbench_live" ? "active" : ""}
              onClick={() => setExecutionMode("workbench_live")}
            >
              Live
            </button>
            <button
              className={executionMode === "pipeline_view" ? "active" : ""}
              onClick={() => setExecutionMode("pipeline_view")}
            >
              Pipeline view
            </button>
          </div>
          <label>
            Run
            <input value={runId} onChange={(event) => setRunId(event.target.value)} onBlur={() => void loadNetwork()} />
          </label>
          <button className="icon-button" onClick={() => void loadNetwork()} aria-label="Refresh network">
            <RefreshCw size={16} />
          </button>
        </div>
      </header>

      <section className="workbench-grid">
        <aside className="left-rail">
          <div className="panel-heading">
            <span>Evidence</span>
            <h1>{executionMode === "pipeline_view" ? "Empirical Exposure" : "Profile Affinity"}</h1>
          </div>
          <div className="status-pill">Ready</div>
          <div className="stat-grid">
            <Stat label="Profiles" value={activeNetwork?.diagnostics.profile_count ?? "--"} />
            <Stat
              label={executionMode === "pipeline_view" ? "Edges" : "Backbone"}
              value={activeNetwork?.diagnostics.empirical_edge_count ?? activeNetwork?.diagnostics.displayed_edge_count ?? "--"}
            />
            <Stat
              label={executionMode === "pipeline_view" ? "Communities" : "Pairs"}
              value={activeNetwork?.diagnostics.community_count ?? activeNetwork?.diagnostics.full_pair_count ?? "--"}
            />
            <Stat label="Mode" value={activeNetwork?.mode ?? "--"} />
          </div>
          {executionMode === "pipeline_view" ? (
            <div className="pipeline-status-list">
              {pipelineStageStatus.slice(0, 8).map((stage) => {
                const stageDisplay = pipelineStageDisplay(stage);
                return (
                  <div key={stage.stage_id} className={stageDisplay.className}>
                    <span>{stage.stage_id}</span>
                    <strong>{stageDisplay.label}</strong>
                  </div>
                );
              })}
            </div>
          ) : null}

          {executionMode === "pipeline_view" ? (
            <section className="formula-card">
              <div className="formula-card-heading">
                <div>
                  <span>Exposure Network</span>
                  <strong>{activeNetwork?.provenance.graph_id ?? "politisky24_bluesky_v1"}</strong>
                </div>
              </div>
              <p className="formula-line">edge = visible peer profile {"->"} exposed receiver profile</p>
              <div className="baseline-meta-list compact-meta">
                <div>
                  <span>Basis</span>
                  <strong>{activeNetwork?.provenance.network_basis ?? "empirical_exposure"}</strong>
                </div>
                <div>
                  <span>Prompt-ready</span>
                  <strong>{activeNetwork?.diagnostics.prompt_ready_count ?? "--"}</strong>
                </div>
                <div>
                  <span>Edge weight</span>
                  <strong>0.35 Like + 0.80 Repost + 0.90 Quote</strong>
                </div>
                <div>
                  <span>Direction</span>
                  <strong>{activeNetwork?.diagnostics.edge_semantics ?? "visible_peer_to_exposed_receiver"}</strong>
                </div>
              </div>
              <div className="rail-note">Pipeline view is canonical and uses Stage 01b profile-position assignments.</div>
            </section>
          ) : (
            <section className="formula-card">
              <div className="formula-card-heading">
                <div>
                  <span>Affinity Formula</span>
                  <strong>Susceptibility similarity</strong>
                </div>
                <button
                  className="mini-button"
                  onClick={() => setFormulaWeights(defaultFormulaWeights)}
                  aria-label="Reset to scientific default"
                  title="Scientific default"
                >
                  <RotateCcw size={13} />
                </button>
              </div>
              <p className="formula-line">{formulaLine}</p>
              <div className="weight-list">
                {FORMULA_KEYS.map((key) => (
                  <label className="weight-row" key={key}>
                    <span>
                      {FORMULA_LABELS[key]}
                      <strong>{Math.round(normalizedFormulaWeights[key] * 100)}%</strong>
                    </span>
                    <input
                      className="range"
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={Math.round(formulaWeights[key] * 100)}
                      onInput={(event) => updateFormulaWeight(key, Number(event.currentTarget.value) / 100)}
                      onChange={(event) => updateFormulaWeight(key, Number(event.target.value) / 100)}
                    />
                  </label>
                ))}
              </div>
              <div className="rail-note">{network?.affinity_formula.warning ?? "Categorical demographics are deliberately low-weighted."}</div>
              <div className="rail-note">
                {network?.affinity_formula.note ?? "Map updates from full pairwise affinities; visible edges are only the backbone."}
              </div>
            </section>
          )}

          {executionMode === "workbench_live" ? (
            <section className="panel-section">
              <h3>Visible backbone</h3>
              <input
                className="range"
                type="range"
                min={2}
                max={14}
                value={edgeLimit}
                onChange={(event) => setEdgeLimit(Number(event.target.value))}
              />
              <div className="rail-note">{edgeLimit} nearest affinity edges per profile</div>
              <div className="rail-note">
                Layout uses all {activeNetwork?.diagnostics.full_pair_count ?? "--"} profile-pair affinities; drawn lines are a readable backbone.
              </div>
            </section>
          ) : null}

          <MeasurementElicitationCard
            mode={measurementMode}
            network={activeNetwork}
            selectedNode={selectedNode}
            selectedOpinion={selectedOpinion}
            attackOptions={attackOptions}
            selectedAttack={selectedAttack}
            baselineRun={matchingBaselineRun}
            networkRun={matchingNetworkRun}
            postRun={matchingPostRun}
            postNetworkRun={matchingPostNetworkRun}
            baselineError={baselineError}
            networkError={networkError}
            postError={postError}
            postNetworkError={postNetworkError}
            baselineProgress={baselineProgress}
            networkProgress={networkProgress}
            postProgress={postProgress}
            postNetworkProgress={postNetworkProgress}
            baselineRunning={baselineRunning}
            networkRunning={networkRunning}
            postRunning={postRunning}
            postNetworkRunning={postNetworkRunning}
            networkReady={networkReady}
            postReady={postReady}
            postNetworkReady={postNetworkReady}
            readOnly={executionMode === "pipeline_view"}
            pipelineSourceLabel={pipelineSourceLabel}
            onModeChange={setMeasurementMode}
            onOpinionChange={handleOpinionChange}
            onAttackChange={handleAttackChange}
            onRunBaseline={runBaseline}
            onRunNetworkExposure={runNetworkExposure}
            onRunPostExposure={runPostExposure}
            onRunPostNetworkExposure={runPostNetworkExposure}
            onOpenPromptPreview={() => {
              if (executionMode === "workbench_live") setPromptPreviewOpen(true);
            }}
          />
          {executionMode === "pipeline_view" && pipelineWarnings.length ? (
            <div className="pipeline-warning-list">
              {pipelineWarnings.slice(0, 3).map((warning) => (
                <div key={warning}>{warning}</div>
              ))}
            </div>
          ) : null}
        </aside>

        <section className="canvas-panel">
          <ProfileNetworkCanvas
            network={activeNetwork}
            measurementByProfile={measurementByProfile}
            measurementMode={measurementMode}
            loading={loading}
            error={error}
            selectedId={selectedNode?.id ?? selectedId}
            onSelect={setSelectedId}
            onOpenMeasurement={setDetailProfileId}
          />
          {detailMeasurement && detailNode ? (
            <div className="baseline-detail-card">
              <button className="detail-close" onClick={() => setDetailProfileId("")} aria-label="Close measurement detail">
                <X size={15} />
              </button>
              <div className="detail-heading">
                <span>{detailNode.label}</span>
                <strong>{formatScore(detailMeasurement.score)}</strong>
              </div>
              <div className="detail-meta">
                <span>
                  {measurementMode === "baseline"
                    ? "Baseline"
                    : measurementMode === "network"
                      ? "Network exposure"
                      : measurementMode === "post"
                        ? "Post exposure"
                        : "Post network"}
                </span>
                <span>{Math.round(detailMeasurement.confidence * 100)}% confidence</span>
                <span>{leafLabel(detailMeasurement.opinion_leaf)}</span>
                {detailMeasurement.phase === "post_network" ? (
                  <span>inc {formatDelta(detailMeasurement.increment_from_private_post)}</span>
                ) : detailMeasurement.phase !== "baseline" ? (
                  <span>d {formatDelta(detailMeasurement.delta_score)}</span>
                ) : null}
              </div>
              <p>{detailMeasurement.reasoning}</p>
            </div>
          ) : null}
          <div className="legend">
            <span>
              <i className="dot neutral" />{" "}
              {measurementMode === "baseline"
                ? "no baseline"
                : measurementMode === "network"
                  ? "no network exposure"
                  : measurementMode === "post"
                    ? "no post"
                    : "no post network"}
            </span>
            <span><i className="dot positive" /> support</span>
            <span><i className="dot negative" /> opposition</span>
            <span>
              <i className="line" /> {executionMode === "pipeline_view" ? "visible -> exposed" : "profile affinity"}
            </span>
          </div>
        </section>

        <aside className="right-rail">
          <div className="panel-heading">
            <span>Inspector</span>
            <h2>{selectedNode?.label ?? "Select profile"}</h2>
          </div>
          <div className="score-card">
            <span>
              {measurementMode === "baseline"
                ? "Baseline score"
                : measurementMode === "network"
                  ? "Network-exposure score"
                  : measurementMode === "post"
                    ? "Post-exposure score"
                    : "Post-network score"}
            </span>
            <strong>{formatScore(selectedMeasurement?.score)}</strong>
            <small>
              {selectedMeasurement
                ? measurementMode === "network" && selectedNetwork
                  ? `d ${formatDelta(selectedNetwork.delta_score)} from baseline, ${Math.round(selectedNetwork.confidence * 100)}% confidence`
                  : measurementMode === "post" && selectedPost
                  ? `d ${formatDelta(selectedPost.delta_score)} from baseline, ${Math.round(selectedPost.confidence * 100)}% confidence`
                  : measurementMode === "post_network" && selectedPostNetwork
                  ? `inc ${formatDelta(selectedPostNetwork.increment_from_private_post)} from private post, ${Math.round(selectedPostNetwork.confidence * 100)}% confidence`
                  : `${Math.round(selectedMeasurement.confidence * 100)}% confidence`
                : measurementMode === "network"
                  ? "No network-exposure result yet"
                  : measurementMode === "post"
                  ? "No post-exposure result yet"
                  : measurementMode === "post_network"
                  ? "No post-network result yet"
                  : "No baseline result yet"}
            </small>
          </div>
          {measurementMode === "network" ? (
            <section className="panel-section">
              <h3>Network context</h3>
              <div className="attribute-list">
                <div className="attribute-row">
                  <span>Baseline</span>
                  <strong>{formatScore(selectedBaseline?.score)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Delta</span>
                  <strong>{formatDelta(selectedNetwork?.delta_score)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Peer mean</span>
                  <strong>{formatScore(numericContextValue(selectedNetwork?.network_context, "peer_score_mean"))}</strong>
                </div>
                <div className="attribute-row">
                  <span>Weighted mean</span>
                  <strong>
                    {formatScore(
                      firstNumericContextValue(selectedNetwork?.network_context, [
                        "exposure_weighted_peer_mean",
                        "peer_exposure_weighted_mean",
                        "affinity_weighted_peer_mean"
                      ])
                    )}
                  </strong>
                </div>
                <div className="attribute-row">
                  <span>Peers</span>
                  <strong>{peerCount(selectedNetwork?.network_context) ?? "--"}</strong>
                </div>
              </div>
            </section>
          ) : null}
          {measurementMode === "post" ? (
            <section className="panel-section">
              <h3>Exposure condition</h3>
              <div className="attribute-list">
                <div className="attribute-row">
                  <span>Baseline</span>
                  <strong>{formatScore(selectedBaseline?.score)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Delta</span>
                  <strong>{formatDelta(selectedPost?.delta_score)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Direction</span>
                  <strong>{selectedPost?.adversarial_direction ?? "--"}</strong>
                </div>
                <div className="attribute-row">
                  <span>Attack</span>
                  <strong>{selectedAttackOption?.label ?? "--"}</strong>
                </div>
              </div>
            </section>
          ) : null}
          {measurementMode === "post_network" ? (
            <section className="panel-section">
              <h3>Post-network condition</h3>
              <div className="attribute-list">
                <div className="attribute-row">
                  <span>Baseline</span>
                  <strong>{formatScore(selectedPostNetwork?.baseline_score ?? selectedBaseline?.score)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Private post</span>
                  <strong>{formatScore(selectedPostNetwork?.private_post_score ?? selectedPost?.post_score)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Increment</span>
                  <strong>{formatDelta(selectedPostNetwork?.increment_from_private_post)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Baseline delta</span>
                  <strong>{formatDelta(selectedPostNetwork?.delta_from_baseline)}</strong>
                </div>
                <div className="attribute-row">
                  <span>Peer post mean</span>
                  <strong>
                    {formatScore(numericContextValue(selectedPostNetwork?.post_attack_network_context, "peer_post_mean"))}
                  </strong>
                </div>
                <div className="attribute-row">
                  <span>Peer delta mean</span>
                  <strong>
                    {formatDelta(numericContextValue(selectedPostNetwork?.post_attack_network_context, "peer_delta_mean"))}
                  </strong>
                </div>
                <div className="attribute-row">
                  <span>Weighted post</span>
                  <strong>
                    {formatScore(
                      firstNumericContextValue(selectedPostNetwork?.post_attack_network_context, [
                        "exposure_weighted_peer_post_mean",
                        "peer_exposure_weighted_post_mean",
                        "affinity_weighted_peer_post_mean"
                      ])
                    )}
                  </strong>
                </div>
                <div className="attribute-row">
                  <span>Weighted delta</span>
                  <strong>
                    {formatDelta(
                      firstNumericContextValue(selectedPostNetwork?.post_attack_network_context, [
                        "exposure_weighted_peer_delta_mean",
                        "peer_exposure_weighted_delta_mean",
                        "affinity_weighted_peer_delta_mean"
                      ])
                    )}
                  </strong>
                </div>
                <div className="attribute-row">
                  <span>Peers</span>
                  <strong>{peerCount(selectedPostNetwork?.post_attack_network_context) ?? "--"}</strong>
                </div>
                <div className="attribute-row">
                  <span>Attack</span>
                  <strong>{selectedAttackOption?.label ?? "--"}</strong>
                </div>
              </div>
            </section>
          ) : null}
          {selectedMeasurement ? (
            <section className="panel-section">
              <h3>Reasoning</h3>
              <p className="reasoning">{selectedMeasurement.reasoning}</p>
            </section>
          ) : null}

          {selectedNode ? (
            <>
              {selectedExposureAssignment ? (
                <section className="panel-section">
                  <h3>Exposure position</h3>
                  <div className="attribute-list">
                    <div className="attribute-row">
                      <span>Position</span>
                      <strong>{formatMetric(selectedExposureAssignment.position_id)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Community</span>
                      <strong>{formatMetric(selectedExposureAssignment.community_id)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Role</span>
                      <strong>{formatMetric(selectedExposureAssignment.display_role)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Prompt-ready</span>
                      <strong>{formatMetric(selectedExposureAssignment.prompt_ready)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Incoming peers</span>
                      <strong>{formatMetric(selectedExposureAssignment.incoming_peer_count, 0)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Incoming weight</span>
                      <strong>{formatMetric(selectedExposureAssignment.incoming_exposure_weight)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Outgoing receivers</span>
                      <strong>{formatMetric(selectedExposureAssignment.outgoing_receiver_count, 0)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Outgoing visibility</span>
                      <strong>{formatMetric(selectedExposureAssignment.outgoing_visibility_weight)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Bridge score</span>
                      <strong>{formatMetric(selectedExposureAssignment.bridge_score)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Eigenvector</span>
                      <strong>{formatMetric(selectedExposureAssignment.eigenvector_centrality)}</strong>
                    </div>
                    <div className="attribute-row">
                      <span>Cascade reach</span>
                      <strong>{formatMetric(selectedExposureAssignment.cascade_reach_potential)}</strong>
                    </div>
                  </div>
                </section>
              ) : null}
              <AttributeList title="Categorical profile" items={selectedNode.categorical_attributes} max={8} />
              <AttributeList title="Continuous profile" items={selectedNode.continuous_attributes} max={8} />
              <section className="panel-section">
                <h3>{executionMode === "pipeline_view" ? "Exposure neighbors" : "Nearest profiles"}</h3>
                <div className="neighbor-list">
                  {nearestNeighbors.map((neighbor) => (
                    <button key={neighbor.id} onClick={() => setSelectedId(neighbor.id)}>
                      <span>
                        {executionMode === "pipeline_view"
                          ? `${neighbor.direction === "incoming" ? "in" : "out"} ${neighbor.id.replace("_", " ")}`
                          : neighbor.id.replace("_", " ")}
                      </span>
                      <strong>{neighbor.weight.toFixed(3)}</strong>
                    </button>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <p className="muted">Click a profile node to inspect attributes and measurement response.</p>
          )}
        </aside>
      </section>
      <PromptPreviewModal
        open={promptPreviewOpen}
        mode={measurementMode}
        network={activeNetwork}
        selectedNode={selectedNode}
        selectedOpinion={selectedOpinion}
        selectedAttack={selectedAttack}
        baselineRun={matchingBaselineRun}
        postRun={matchingPostRun}
        onClose={() => setPromptPreviewOpen(false)}
      />
    </main>
  );
}
