import { useEffect, useMemo, useState } from "react";
import { Copy, X } from "lucide-react";

import {
  fetchBaselinePromptPreview,
  fetchNetworkExposurePromptPreview,
  fetchPostAttackNetworkExposurePromptPreview,
  fetchPostExposurePromptPreview
} from "../api/client";
import type {
  BaselineRunResponse,
  PostExposureRunResponse,
  ProfileNetworkNode,
  ProfileNetworkResponse,
  PromptPreviewResponse
} from "../api/types";
import type { MeasurementMode } from "./MeasurementElicitationCard";

type PromptTab = "system" | "user" | "combined";

interface PromptPreviewModalProps {
  open: boolean;
  mode: MeasurementMode;
  network: ProfileNetworkResponse | null;
  selectedNode: ProfileNetworkNode | undefined;
  selectedOpinion: string;
  selectedAttack: string;
  baselineRun: BaselineRunResponse | null;
  postRun: PostExposureRunResponse | null;
  onClose: () => void;
}

function prettyJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function combinedMessages(preview: PromptPreviewResponse) {
  return prettyJson(preview.messages);
}

export function PromptPreviewModal({
  open,
  mode,
  network,
  selectedNode,
  selectedOpinion,
  selectedAttack,
  baselineRun,
  postRun,
  onClose
}: PromptPreviewModalProps) {
  const [preview, setPreview] = useState<PromptPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<PromptTab>("system");
  const [copied, setCopied] = useState(false);

  const profileId = selectedNode?.id ?? network?.nodes[0]?.id ?? "";

  useEffect(() => {
    if (!open || !network || !profileId || !selectedOpinion) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    setCopied(false);
    setPreview(null);
    const request =
      mode === "baseline"
        ? fetchBaselinePromptPreview(network.run_id, selectedOpinion, profileId)
        : mode === "network"
          ? baselineRun
            ? fetchNetworkExposurePromptPreview(network.run_id, baselineRun.baseline_run_id, selectedOpinion, profileId)
            : Promise.reject(new Error("A completed baseline run is required for network-exposure preview."))
          : mode === "post"
            ? baselineRun && selectedAttack
              ? fetchPostExposurePromptPreview(network.run_id, baselineRun.baseline_run_id, selectedOpinion, selectedAttack, profileId)
              : Promise.reject(new Error("A completed baseline run and selected attack are required for post-exposure preview."))
            : baselineRun && postRun && selectedAttack
              ? fetchPostAttackNetworkExposurePromptPreview(
                  network.run_id,
                  baselineRun.baseline_run_id,
                  postRun.post_run_id,
                  selectedOpinion,
                  selectedAttack,
                  profileId
                )
              : Promise.reject(new Error("A completed post-exposure run is required for post-network preview."));
    void request
      .then((payload) => {
        if (!cancelled) setPreview(payload);
      })
      .catch((exc) => {
        if (!cancelled) setError(exc instanceof Error ? exc.message : String(exc));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, mode, network, profileId, selectedOpinion, selectedAttack, baselineRun, postRun]);

  const visibleContent = useMemo(() => {
    if (!preview) return "";
    if (activeTab === "system") return preview.system_prompt;
    if (activeTab === "user") return prettyJson(preview.user_payload);
    return combinedMessages(preview);
  }, [activeTab, preview]);

  async function copyVisibleContent() {
    if (!visibleContent) return;
    try {
      await navigator.clipboard.writeText(visibleContent);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="prompt-modal" role="dialog" aria-modal="true" aria-labelledby="prompt-preview-title" onClick={(event) => event.stopPropagation()}>
        <button className="detail-close" onClick={onClose} aria-label="Close prompt preview">
          <X size={15} />
        </button>
        <div className="prompt-modal-heading">
          <span>LLM Prompt Preview</span>
          <h2 id="prompt-preview-title">
            {mode === "baseline"
              ? "Exact baseline request"
              : mode === "network"
                ? "Exact network-exposure request"
                : mode === "post"
                  ? "Exact post-exposure request"
                  : "Exact post-network request"}
          </h2>
          <p>
            {preview
              ? `${preview.prompt_name} for ${preview.profile_id} on ${preview.opinion_leaf}`
              : "The preview reconstructs the same system prompt and user JSON without calling the model."}
          </p>
        </div>

        <div className="prompt-tabs" role="tablist" aria-label="Prompt preview sections">
          <button className={activeTab === "system" ? "active" : ""} onClick={() => setActiveTab("system")}>
            System prompt
          </button>
          <button className={activeTab === "user" ? "active" : ""} onClick={() => setActiveTab("user")}>
            User JSON
          </button>
          <button className={activeTab === "combined" ? "active" : ""} onClick={() => setActiveTab("combined")}>
            Combined messages
          </button>
          <button className="copy-button" disabled={!visibleContent} onClick={copyVisibleContent}>
            <Copy size={13} />
            {copied ? "Copied" : "Copy"}
          </button>
        </div>

        <div className="prompt-content">
          {loading ? <div className="canvas-state"><strong>Loading prompt preview</strong></div> : null}
          {error ? <p className="error-text">{error}</p> : null}
          {!loading && !error && preview ? <pre>{visibleContent}</pre> : null}
        </div>
      </section>
    </div>
  );
}
