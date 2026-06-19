import { FileText, Play } from "lucide-react";

import type {
  AttackOption,
  BaselineRunResponse,
  NetworkExposureRunResponse,
  PostAttackNetworkExposureRunResponse,
  PostExposureRunResponse,
  ProfileNetworkNode,
  ProfileNetworkResponse
} from "../api/types";

export type MeasurementMode = "baseline" | "network" | "post" | "post_network";

interface MeasurementElicitationCardProps {
  mode: MeasurementMode;
  network: ProfileNetworkResponse | null;
  selectedNode: ProfileNetworkNode | undefined;
  selectedOpinion: string;
  attackOptions: AttackOption[];
  selectedAttack: string;
  baselineRun: BaselineRunResponse | null;
  networkRun: NetworkExposureRunResponse | null;
  postRun: PostExposureRunResponse | null;
  postNetworkRun: PostAttackNetworkExposureRunResponse | null;
  baselineError: string;
  networkError: string;
  postError: string;
  postNetworkError: string;
  baselineProgress: number;
  networkProgress: number;
  postProgress: number;
  postNetworkProgress: number;
  baselineRunning: boolean;
  networkRunning: boolean;
  postRunning: boolean;
  postNetworkRunning: boolean;
  networkReady: boolean;
  postReady: boolean;
  postNetworkReady: boolean;
  readOnly?: boolean;
  pipelineSourceLabel?: string;
  onModeChange: (mode: MeasurementMode) => void;
  onOpinionChange: (opinionLeaf: string) => void;
  onAttackChange: (attackLeaf: string) => void;
  onRunBaseline: () => void;
  onRunNetworkExposure: () => void;
  onRunPostExposure: () => void;
  onRunPostNetworkExposure: () => void;
  onOpenPromptPreview: () => void;
}

function titleCaseStatus(status: string) {
  return status
    .replaceAll("_", " ")
    .split(" ")
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function statusLabel(
  run:
    | BaselineRunResponse
    | NetworkExposureRunResponse
    | PostExposureRunResponse
    | PostAttackNetworkExposureRunResponse
    | null
) {
  return run ? titleCaseStatus(run.status) : "Ready";
}

function shortAttackLabel(option: AttackOption | undefined, fallback: string) {
  return option?.label ?? fallback.split(">").at(-1)?.trim() ?? fallback;
}

export function MeasurementElicitationCard({
  mode,
  network,
  selectedNode,
  selectedOpinion,
  attackOptions,
  selectedAttack,
  baselineRun,
  networkRun,
  postRun,
  postNetworkRun,
  baselineError,
  networkError,
  postError,
  postNetworkError,
  baselineProgress,
  networkProgress,
  postProgress,
  postNetworkProgress,
  baselineRunning,
  networkRunning,
  postRunning,
  postNetworkRunning,
  networkReady,
  postReady,
  postNetworkReady,
  readOnly = false,
  pipelineSourceLabel,
  onModeChange,
  onOpinionChange,
  onAttackChange,
  onRunBaseline,
  onRunNetworkExposure,
  onRunPostExposure,
  onRunPostNetworkExposure,
  onOpenPromptPreview
}: MeasurementElicitationCardProps) {
  const selectedLeaf = network?.opinion_leaves.find((opinion) => opinion.path === selectedOpinion);
  const selectedAttackOption = attackOptions.find((option) => option.path === selectedAttack);
  const isPostCondition = mode === "post" || mode === "post_network";
  const activeRun =
    mode === "baseline" ? baselineRun : mode === "network" ? networkRun : mode === "post" ? postRun : postNetworkRun;
  const activeProgress =
    mode === "baseline"
      ? baselineProgress
      : mode === "network"
        ? networkProgress
        : mode === "post"
          ? postProgress
          : postNetworkProgress;
  const activeError =
    mode === "baseline" ? baselineError : mode === "network" ? networkError : mode === "post" ? postError : postNetworkError;
  const activeRunning =
    mode === "baseline"
      ? baselineRunning
      : mode === "network"
        ? networkRunning
        : mode === "post"
          ? postRunning
          : postNetworkRunning;
  const modelName = activeRun?.model_name ?? network?.provenance.model_name ?? "--";
  const profileCount = activeRun?.profile_count ?? network?.diagnostics.profile_count ?? "--";
  const rawPath = activeRun?.raw_llm_dir ?? "";
  const baselineComplete = baselineRun?.status === "completed";
  const postComplete = postRun?.status === "completed";
  const modeTitle =
    mode === "baseline"
      ? "Pre-exposure measurement"
      : mode === "network"
        ? "Network exposure baseline"
        : mode === "post"
          ? "Post-exposure state"
          : "Post-network state";
  const taskText =
    mode === "baseline"
      ? "Measure pre-exposure opinion for the selected profile panel."
      : mode === "network"
        ? "Re-assess pre-attack opinion with baseline peer context."
        : mode === "post"
          ? "Estimate post-exposure opinion after the selected ontology attack vector."
          : "Estimate post-attack opinion with same-condition peer post-exposure context.";
  const currentLabel =
    mode === "baseline"
      ? "Current scenario"
      : mode === "network"
        ? "Current peer-context condition"
        : mode === "post"
          ? "Current attack condition"
          : "Current post-network condition";
  const currentValue =
    isPostCondition
      ? `${selectedLeaf?.label ?? "--"} × ${shortAttackLabel(selectedAttackOption, selectedAttack || "--")}`
      : selectedLeaf?.label ?? (selectedOpinion || "--");
  const sourceLabel =
    pipelineSourceLabel ??
    (mode === "baseline"
      ? network?.provenance.source ?? "pending"
      : mode === "network"
        ? "baseline_peer_context"
        : mode === "post"
          ? "stage03_attack_spec"
          : "same_condition_post_peer_context");
  const runButtonLabel =
    mode === "baseline" ? "Run baseline" : mode === "network" ? "Run network" : mode === "post" ? "Run post" : "Run post network";
  const completedNote =
    mode === "baseline"
      ? "Completed outputs appear as baseline profile bubbles."
      : mode === "network"
        ? "Completed outputs appear as network-exposure profile bubbles."
        : mode === "post"
          ? "Completed outputs appear as post-exposure profile bubbles."
          : "Completed outputs appear as post-network profile bubbles.";
  const disabledRun =
    readOnly
      ? true
      : mode === "baseline"
      ? !network || !selectedOpinion || activeRunning
      : mode === "network"
        ? !network || !selectedOpinion || !networkReady || activeRunning
        : mode === "post"
          ? !network || !selectedOpinion || !selectedAttack || !postReady || activeRunning
          : !network || !selectedOpinion || !selectedAttack || !postNetworkReady || activeRunning;
  const runAction =
    mode === "baseline"
      ? onRunBaseline
      : mode === "network"
        ? onRunNetworkExposure
        : mode === "post"
          ? onRunPostExposure
          : onRunPostNetworkExposure;
  const promptEnabled =
    readOnly
      ? false
      : mode === "baseline"
      ? Boolean(network && selectedOpinion && selectedNode)
      : mode === "network"
        ? Boolean(network && selectedOpinion && selectedNode && networkReady)
        : mode === "post"
          ? Boolean(network && selectedOpinion && selectedNode && selectedAttack && postReady)
          : Boolean(network && selectedOpinion && selectedNode && selectedAttack && postNetworkReady);

  return (
    <section className="baseline-card">
      <div className="baseline-card-heading">
        <div>
          <span>Opinion Elicitation</span>
          <strong>{modeTitle}</strong>
        </div>
        <div className={`baseline-status ${activeRun?.status ?? "ready"}`}>{statusLabel(activeRun)}</div>
      </div>

      <div className="measurement-mode-toggle" role="tablist" aria-label="Measurement mode">
        <button className={mode === "baseline" ? "active" : ""} onClick={() => onModeChange("baseline")}>
          Baseline
        </button>
        <button className={mode === "network" ? "active" : ""} onClick={() => onModeChange("network")}>
          Network exposure
        </button>
        <button className={mode === "post" ? "active" : ""} onClick={() => onModeChange("post")}>
          Post exposure
        </button>
        <button className={mode === "post_network" ? "active" : ""} onClick={() => onModeChange("post_network")}>
          Post network
        </button>
      </div>

      <p className="baseline-task">{taskText}</p>

      <label className="baseline-selector">
        <span>Opinion leaf</span>
        <select value={selectedOpinion} onChange={(event) => onOpinionChange(event.target.value)}>
          {network?.opinion_leaves.map((opinion) => (
            <option key={opinion.path} value={opinion.path}>
              {opinion.label}
            </option>
          ))}
        </select>
      </label>

      {isPostCondition ? (
        <label className="baseline-selector">
          <span>Attack vector</span>
          <select value={selectedAttack} onChange={(event) => onAttackChange(event.target.value)}>
            {attackOptions.map((attack) => (
              <option key={attack.path} value={attack.path}>
                {attack.label}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      <div className="selected-opinion-line">
        <span>{currentLabel}</span>
        <strong>{currentValue}</strong>
      </div>

      {mode === "network" ? (
        <div className={`dependency-line ${networkReady ? "ready" : "blocked"}`}>
          <span>Baseline dependency</span>
          <strong>{baselineComplete ? "Completed baseline linked" : "Run a completed baseline first"}</strong>
        </div>
      ) : null}

      {mode === "post" ? (
        <div className={`dependency-line ${postReady ? "ready" : "blocked"}`}>
          <span>Baseline dependency</span>
          <strong>{baselineComplete ? "Completed baseline linked" : "Run a completed baseline first"}</strong>
        </div>
      ) : null}

      {mode === "post_network" ? (
        <div className={`dependency-line ${postNetworkReady ? "ready" : "blocked"}`}>
          <span>Post dependency</span>
          <strong>{postComplete ? "Completed private post linked" : "Run completed post exposure first"}</strong>
        </div>
      ) : null}

      {readOnly ? (
        <div className="rail-note">Read-only pipeline artifact view; live run controls are available only in Live workbench.</div>
      ) : (
        <div className="baseline-actions">
          <button
            className="primary-button"
            disabled={disabledRun}
            onClick={runAction}
          >
            <Play size={15} />
            {runButtonLabel}
          </button>
          <button className="secondary-button" disabled={!promptEnabled} onClick={onOpenPromptPreview}>
            <FileText size={14} />
            View prompt
          </button>
        </div>
      )}

      <div className="baseline-meta-list">
        <div>
          <span>Model</span>
          <strong>{modelName}</strong>
        </div>
        <div>
          <span>Profiles</span>
          <strong>{profileCount}</strong>
        </div>
        <div>
          <span>Source</span>
          <strong>{sourceLabel}</strong>
        </div>
        <div>
          <span>Run</span>
          <strong>{network?.run_id ?? "--"}</strong>
        </div>
      </div>

      {activeRun ? (
        <div className="progress-block">
          <div className="progress-meta">
            <strong>{statusLabel(activeRun)}</strong>
            <span>{activeProgress}%</span>
          </div>
          <div className="progress-track">
            <div style={{ width: `${activeProgress}%` }} />
          </div>
          <div className="rail-note">
            {activeRun.completed_count} complete, {activeRun.failed_count} failed
          </div>
          <div className="rail-note">{completedNote}</div>
          {(rawPath || (readOnly && activeRun.artifact_dir)) ? (
            <div className="artifact-path">
              <span>{readOnly ? "Pipeline artifacts" : "Raw artifacts"}</span>
              <code>{rawPath || activeRun.artifact_dir}</code>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="rail-note">
          {readOnly
            ? "Pipeline view is read-only and displays canonical stage artifacts."
            : "Prompt preview is available before running and does not call the model."}
        </div>
      )}

      {isPostCondition && selectedAttackOption?.notes.length ? (
        <div className="rail-note">Attack notes: {selectedAttackOption.notes.join(" ")}</div>
      ) : null}
      {activeError ? <p className="error-text">{activeError}</p> : null}
    </section>
  );
}
