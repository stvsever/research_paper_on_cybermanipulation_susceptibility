from __future__ import annotations

import json
import os
import statistics
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from profile_network_workbench_api.backend_adapter import canonical_leaf_path, reconstruct_profile_bundles
from profile_network_workbench_api.schemas import (
    BaselinePromptMessage,
    BaselineResult,
    NetworkExposurePromptPreviewResponse,
    NetworkExposureResult,
    NetworkExposureRunCreateRequest,
    NetworkExposureRunCreateResponse,
    NetworkExposureRunError,
    NetworkExposureRunResponse,
    NetworkExposureRunStatus,
)
from profile_network_workbench_api.services.baseline_runs import BaselineRunNotFoundError, get_baseline_run
from profile_network_workbench_api.settings import load_settings


class NetworkExposureRunNotFoundError(KeyError):
    pass


class NetworkExposureRunConfigurationError(RuntimeError):
    pass


AGENT_FACTORY_CLASS: Any | None = None

_RUNS: dict[str, NetworkExposureRunResponse] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _persist_state(state: NetworkExposureRunResponse) -> None:
    artifact_dir = Path(state.artifact_dir)
    _write_json(artifact_dir / "status.json", state.model_dump())
    _write_jsonl(artifact_dir / "results.jsonl", [item.model_dump() for item in state.results])
    _write_jsonl(artifact_dir / "errors.jsonl", [item.model_dump() for item in state.errors])


def _set_state(run_id: str, **updates: Any) -> NetworkExposureRunResponse:
    with _LOCK:
        current = _RUNS[run_id]
        payload = current.model_dump()
        payload.update(updates)
        payload["updated_at"] = _now()
        next_state = NetworkExposureRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)
        return next_state


def _append_result(
    run_id: str,
    result: NetworkExposureResult | None = None,
    error: NetworkExposureRunError | None = None,
) -> None:
    with _LOCK:
        current = _RUNS[run_id]
        results = list(current.results)
        errors = list(current.errors)
        if result is not None:
            results.append(result)
        if error is not None:
            errors.append(error)
        status: NetworkExposureRunStatus = "running"
        payload = current.model_dump()
        payload.update(
            {
                "results": [item.model_dump() for item in results],
                "errors": [item.model_dump() for item in errors],
                "completed_count": len(results),
                "failed_count": len(errors),
                "status": status,
                "updated_at": _now(),
            }
        )
        next_state = NetworkExposureRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)


def _agent_factory_class() -> Any:
    if AGENT_FACTORY_CLASS is not None:
        return AGENT_FACTORY_CLASS
    from src.backend.agentic_framework.factory import AgentFactory

    return AgentFactory


def _profile_context(run_id: str) -> tuple[dict[str, Any], list[Any], list[str]]:
    payload, _, _, bundles, opinions = reconstruct_profile_bundles(run_id)
    profiles = [bundle["profile_result"].profile for bundle in bundles]
    if not profiles:
        raise NetworkExposureRunConfigurationError("No profiles selected for network exposure run")
    return payload, profiles, opinions


def _validate_opinion_leaf(opinion_leaf: str, configured_opinions: list[str]) -> str:
    canonical_opinion_leaf = canonical_leaf_path(opinion_leaf, configured_opinions)
    if canonical_opinion_leaf is None:
        raise NetworkExposureRunConfigurationError("Opinion leaf must be one of the run-configured opinion leaves")
    return canonical_opinion_leaf


def _baseline_result_by_profile(
    baseline_run_id: str,
    run_id: str,
    opinion_leaf: str,
) -> dict[str, BaselineResult]:
    try:
        baseline_run = get_baseline_run(baseline_run_id)
    except BaselineRunNotFoundError as exc:
        raise NetworkExposureRunConfigurationError(f"Baseline run not found: {baseline_run_id}") from exc

    if baseline_run.status != "completed":
        raise NetworkExposureRunConfigurationError("Network exposure requires a completed baseline run")
    if baseline_run.run_id != run_id:
        raise NetworkExposureRunConfigurationError("Baseline run belongs to a different run_id")
    if baseline_run.opinion_leaf != opinion_leaf:
        raise NetworkExposureRunConfigurationError("Baseline run uses a different opinion leaf")
    return {item.profile_id: item for item in baseline_run.results}


def _select_profiles_with_baseline(
    profiles: list[Any],
    baseline_by_profile: dict[str, BaselineResult],
    requested_profile_ids: list[str] | None,
) -> list[Any]:
    baseline_profile_ids = set(baseline_by_profile)
    if requested_profile_ids:
        requested = set(requested_profile_ids)
        profile_by_id = {str(profile.profile_id): profile for profile in profiles}
        unknown = sorted(requested - set(profile_by_id))
        if unknown:
            raise NetworkExposureRunConfigurationError(f"Unknown profile id: {unknown[0]}")
        missing_baseline = sorted(requested - baseline_profile_ids)
        if missing_baseline:
            raise NetworkExposureRunConfigurationError(f"Baseline result missing for profile: {missing_baseline[0]}")
        return [profile_by_id[profile_id] for profile_id in requested_profile_ids]
    selected = [profile for profile in profiles if str(profile.profile_id) in baseline_profile_ids]
    if not selected:
        raise NetworkExposureRunConfigurationError("No profiles with completed baseline results are available")
    return selected


def _peer_context(
    *,
    target_profile: Any,
    all_profiles: list[Any],
    baseline_by_profile: dict[str, BaselineResult],
    opinion_leaf: str,
    top_k: int,
) -> dict[str, Any]:
    from src.backend.utils.profile_affinity import compute_profile_affinities, rank_profile_neighbors

    available_profiles = [
        profile for profile in all_profiles if str(profile.profile_id) in baseline_by_profile
    ]
    affinities = compute_profile_affinities(available_profiles)
    target_id = str(target_profile.profile_id)
    peers: list[dict[str, Any]] = []
    for affinity in rank_profile_neighbors(target_id, affinities, limit=max(0, len(available_profiles) - 1)):
        peer_id = affinity.other(target_id)
        baseline = baseline_by_profile.get(peer_id)
        if baseline is None:
            continue
        peers.append(
            {
                "profile_id": peer_id,
                "affinity": round(float(affinity.affinity), 6),
                "baseline_score": int(baseline.score),
                "confidence": round(float(baseline.confidence), 4),
                "reasoning": str(baseline.reasoning),
            }
        )
        if len(peers) >= top_k:
            break

    peer_scores = [float(peer["baseline_score"]) for peer in peers]
    affinity_sum = sum(float(peer["affinity"]) for peer in peers)
    weighted_mean = (
        sum(float(peer["baseline_score"]) * float(peer["affinity"]) for peer in peers) / affinity_sum
        if peers and affinity_sum > 0.0
        else None
    )
    target_baseline = baseline_by_profile[target_id]
    return {
        "top_k": int(top_k),
        "opinion_leaf": opinion_leaf,
        "target_baseline_score": int(target_baseline.score),
        "target_baseline_confidence": round(float(target_baseline.confidence), 4),
        "peer_score_mean": round(float(statistics.mean(peer_scores)), 4) if peer_scores else None,
        "peer_score_sd": round(float(statistics.pstdev(peer_scores)), 4) if len(peer_scores) > 1 else (0.0 if peer_scores else None),
        "affinity_weighted_peer_mean": round(float(weighted_mean), 4) if weighted_mean is not None else None,
        "peer_assessments": peers,
    }


def _run_inputs(
    run_id: str,
    baseline_run_id: str,
    opinion_leaf: str,
    profile_ids: list[str] | None,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    payload, all_profiles, opinions = _profile_context(run_id)
    opinion_leaf = _validate_opinion_leaf(opinion_leaf, opinions)
    baseline_by_profile = _baseline_result_by_profile(baseline_run_id, run_id, opinion_leaf)
    target_profiles = _select_profiles_with_baseline(all_profiles, baseline_by_profile, profile_ids)
    inputs: list[dict[str, Any]] = []
    for profile in target_profiles:
        profile_id = str(profile.profile_id)
        scenario_id = f"network_{run_id}_{profile_id}_{_safe_filename(opinion_leaf)}"
        network_context = _peer_context(
            target_profile=profile,
            all_profiles=all_profiles,
            baseline_by_profile=baseline_by_profile,
            opinion_leaf=opinion_leaf,
            top_k=top_k,
        )
        inputs.append(
            {
                "profile": profile,
                "baseline": baseline_by_profile[profile_id],
                "scenario_id": scenario_id,
                "network_context": network_context,
            }
        )
    return payload, inputs, opinion_leaf


def get_network_exposure_prompt_preview(
    run_id: str,
    baseline_run_id: str,
    opinion_leaf: str,
    profile_id: str,
    top_k: int = 8,
) -> NetworkExposurePromptPreviewResponse:
    payload, inputs, opinion_leaf = _run_inputs(run_id, baseline_run_id, opinion_leaf, [profile_id], max(1, int(top_k)))
    model_name = str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise NetworkExposureRunConfigurationError("No model name configured for network exposure elicitation")

    item = inputs[0]
    profile = item["profile"]
    prompt_name = "network_exposure_opinion.md"
    prompts_dir = load_settings().lab_root / "src" / "backend" / "agentic_framework" / "prompts"
    system_prompt = (prompts_dir / prompt_name).read_text(encoding="utf-8")
    user_payload = {
        "scenario_id": f"preview_{run_id}_{profile.profile_id}_network",
        "opinion_leaf": opinion_leaf,
        "profile": profile.model_dump(),
        "baseline_score": int(item["baseline"].score),
        "network_context": item["network_context"],
    }
    user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return NetworkExposurePromptPreviewResponse(
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        profile_id=str(profile.profile_id),
        opinion_leaf=opinion_leaf,
        prompt_name=prompt_name,
        model_name=model_name,
        system_prompt=system_prompt,
        user_payload=user_payload,
        messages=[
            BaselinePromptMessage(role="system", content=system_prompt),
            BaselinePromptMessage(role="user", content=user_content),
        ],
    )


def start_network_exposure_run(request: NetworkExposureRunCreateRequest) -> NetworkExposureRunCreateResponse:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise NetworkExposureRunConfigurationError("OPENROUTER_API_KEY is required for real network exposure elicitation")

    top_k = max(1, int(request.top_k or 8))
    payload, profile_inputs, opinion_leaf = _run_inputs(
        request.run_id,
        request.baseline_run_id,
        request.opinion_leaf,
        request.profile_ids,
        top_k,
    )
    model_name = request.model_name or str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise NetworkExposureRunConfigurationError("No model name configured for network exposure elicitation")

    network_run_id = f"network_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    artifact_dir = load_settings().runs_root / network_run_id
    raw_llm_dir = artifact_dir / "raw_llm"
    network_contexts_path = artifact_dir / "network_contexts.jsonl"
    created_at = _now()
    state = NetworkExposureRunResponse(
        network_run_id=network_run_id,
        status="queued",
        run_id=request.run_id,
        baseline_run_id=request.baseline_run_id,
        opinion_leaf=opinion_leaf,
        model_name=model_name,
        profile_count=len(profile_inputs),
        completed_count=0,
        failed_count=0,
        top_k=top_k,
        artifact_dir=str(artifact_dir),
        raw_llm_dir=str(raw_llm_dir),
        network_contexts_path=str(network_contexts_path),
        created_at=created_at,
        updated_at=created_at,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_llm_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        network_contexts_path,
        [
            {
                "scenario_id": item["scenario_id"],
                "profile_id": str(item["profile"].profile_id),
                "opinion_leaf": opinion_leaf,
                "network_context": item["network_context"],
            }
            for item in profile_inputs
        ],
    )
    _write_json(
        artifact_dir / "metadata.json",
        {
            "network_run_id": network_run_id,
            "baseline_run_id": request.baseline_run_id,
            "run_id": request.run_id,
            "opinion_leaf": opinion_leaf,
            "model_name": model_name,
            "top_k": top_k,
            "profile_ids": [str(item["profile"].profile_id) for item in profile_inputs],
            "network_exposure_semantics": "network_exposure_agent_with_baseline_peer_context_no_fallback",
            "network_contexts_path": str(network_contexts_path),
        },
    )
    with _LOCK:
        _RUNS[network_run_id] = state
        _persist_state(state)

    thread = threading.Thread(
        target=_run_network_batch,
        args=(network_run_id, profile_inputs, opinion_leaf, model_name, api_key, payload, raw_llm_dir, request.max_concurrency),
        daemon=True,
    )
    thread.start()

    return NetworkExposureRunCreateResponse(
        network_run_id=network_run_id,
        status="queued",
        run_id=request.run_id,
        baseline_run_id=request.baseline_run_id,
        opinion_leaf=opinion_leaf,
        profile_count=len(profile_inputs),
        model_name=model_name,
        top_k=top_k,
    )


def get_network_exposure_run(network_run_id: str) -> NetworkExposureRunResponse:
    with _LOCK:
        state = _RUNS.get(network_run_id)
    if state is not None:
        return state

    status_path = load_settings().runs_root / _safe_filename(network_run_id) / "status.json"
    if not status_path.exists():
        raise NetworkExposureRunNotFoundError(network_run_id)
    with status_path.open(encoding="utf-8") as handle:
        return NetworkExposureRunResponse.model_validate(json.load(handle))


def _run_network_batch(
    network_run_id: str,
    profile_inputs: list[dict[str, Any]],
    opinion_leaf: str,
    model_name: str,
    api_key: str,
    config_payload: dict[str, Any],
    raw_llm_dir: Path,
    max_concurrency: int | None,
) -> None:
    _set_state(network_run_id, status="running")
    prompts_dir = load_settings().lab_root / "src" / "backend" / "agentic_framework" / "prompts"
    max_workers = max(1, int(max_concurrency or config_payload.get("max_concurrency") or 1))
    max_workers = min(max_workers, len(profile_inputs))
    thread_local = threading.local()

    def agent_for_thread() -> Any:
        if not hasattr(thread_local, "agent"):
            factory = _agent_factory_class()(
                prompts_dir=prompts_dir,
                openrouter_api_key=api_key,
                openrouter_model=model_name,
                max_repair_iter=int(config_payload.get("max_repair_iter") or 2),
                temperature=float(config_payload.get("temperature") or 0.2),
                timeout_sec=int(config_payload.get("timeout_sec") or 90),
                save_raw_dir=str(raw_llm_dir),
            )
            thread_local.agent = factory.network_exposure_opinion_agent()
        return thread_local.agent

    def process(item: dict[str, Any]) -> NetworkExposureResult:
        profile = item["profile"]
        baseline = item["baseline"]
        scenario_id = str(item["scenario_id"])
        network_context = dict(item["network_context"])
        call_id = f"{scenario_id}_network_exposure"
        assessment = agent_for_thread().assess(
            run_id=network_run_id,
            call_id=call_id,
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            profile=profile,
            baseline_score=int(baseline.score),
            network_context=network_context,
        )
        network_score = int(assessment.score)
        return NetworkExposureResult(
            profile_id=str(profile.profile_id),
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            baseline_score=int(baseline.score),
            network_score=network_score,
            delta_score=int(network_score - int(baseline.score)),
            confidence=float(assessment.confidence),
            reasoning=str(assessment.reasoning),
            model_name=str(assessment.model_name),
            call_id=call_id,
            timestamp=_now(),
            network_context=network_context,
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process, item): item for item in profile_inputs}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                profile = item["profile"]
                scenario_id = str(item["scenario_id"])
                try:
                    _append_result(network_run_id, result=future.result())
                except Exception as exc:
                    _append_result(
                        network_run_id,
                        error=NetworkExposureRunError(
                            profile_id=str(profile.profile_id),
                            scenario_id=scenario_id,
                            message=str(exc),
                            timestamp=_now(),
                        ),
                    )
        final = get_network_exposure_run(network_run_id)
        if final.failed_count and final.completed_count:
            _set_state(network_run_id, status="completed_with_errors")
        elif final.failed_count:
            _set_state(network_run_id, status="failed")
        else:
            _set_state(network_run_id, status="completed")
    except Exception as exc:
        _set_state(network_run_id, status="failed")
        _write_json(Path(get_network_exposure_run(network_run_id).artifact_dir) / "fatal_error.json", {"message": str(exc), "timestamp": _now()})
