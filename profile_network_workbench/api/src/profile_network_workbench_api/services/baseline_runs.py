from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from profile_network_workbench_api.backend_adapter import reconstruct_profile_bundles
from profile_network_workbench_api.schemas import (
    BaselinePromptMessage,
    BaselinePromptPreviewResponse,
    BaselineResult,
    BaselineRunCreateRequest,
    BaselineRunCreateResponse,
    BaselineRunError,
    BaselineRunResponse,
    BaselineRunStatus,
)
from profile_network_workbench_api.settings import load_settings


class BaselineRunNotFoundError(KeyError):
    pass


class BaselineRunConfigurationError(RuntimeError):
    pass


AGENT_FACTORY_CLASS: Any | None = None

_RUNS: dict[str, BaselineRunResponse] = {}
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


def _persist_state(state: BaselineRunResponse) -> None:
    artifact_dir = Path(state.artifact_dir)
    _write_json(artifact_dir / "status.json", state.model_dump())
    _write_jsonl(artifact_dir / "results.jsonl", [item.model_dump() for item in state.results])
    _write_jsonl(artifact_dir / "errors.jsonl", [item.model_dump() for item in state.errors])


def _set_state(run_id: str, **updates: Any) -> BaselineRunResponse:
    with _LOCK:
        current = _RUNS[run_id]
        payload = current.model_dump()
        payload.update(updates)
        payload["updated_at"] = _now()
        next_state = BaselineRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)
        return next_state


def _append_result(run_id: str, result: BaselineResult | None = None, error: BaselineRunError | None = None) -> None:
    with _LOCK:
        current = _RUNS[run_id]
        results = list(current.results)
        errors = list(current.errors)
        if result is not None:
            results.append(result)
        if error is not None:
            errors.append(error)
        status: BaselineRunStatus = "running"
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
        next_state = BaselineRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)


def _agent_factory_class() -> Any:
    if AGENT_FACTORY_CLASS is not None:
        return AGENT_FACTORY_CLASS
    from src.backend.agentic_framework.factory import AgentFactory

    return AgentFactory


def _profile_records(run_id: str, requested_profile_ids: list[str] | None) -> tuple[dict[str, Any], list[Any], list[str]]:
    payload, _, _, bundles, opinions = reconstruct_profile_bundles(run_id)
    profiles = [bundle["profile_result"].profile for bundle in bundles]
    if requested_profile_ids:
        requested = set(requested_profile_ids)
        profiles = [profile for profile in profiles if profile.profile_id in requested]
        missing = sorted(requested - {profile.profile_id for profile in profiles})
        if missing:
            raise BaselineRunConfigurationError(f"Unknown profile id: {missing[0]}")
    if not profiles:
        raise BaselineRunConfigurationError("No profiles selected for baseline run")
    return payload, profiles, opinions


def _validate_opinion_leaf(opinion_leaf: str, configured_opinions: list[str]) -> None:
    if opinion_leaf not in configured_opinions:
        raise BaselineRunConfigurationError("Opinion leaf must be one of the run-configured opinion leaves")


def get_baseline_prompt_preview(run_id: str, opinion_leaf: str, profile_id: str) -> BaselinePromptPreviewResponse:
    payload, profiles, opinions = _profile_records(run_id, [profile_id])
    _validate_opinion_leaf(opinion_leaf, opinions)

    model_name = str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise BaselineRunConfigurationError("No model name configured for baseline elicitation")

    profile = profiles[0]
    prompt_name = "baseline_opinion.md"
    prompts_dir = load_settings().lab_root / "src" / "backend" / "agentic_framework" / "prompts"
    system_prompt = (prompts_dir / prompt_name).read_text(encoding="utf-8")
    user_payload = {
        "scenario_id": f"preview_{run_id}_{profile.profile_id}",
        "opinion_leaf": opinion_leaf,
        "profile": profile.model_dump(),
    }
    user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return BaselinePromptPreviewResponse(
        run_id=run_id,
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


def start_baseline_run(request: BaselineRunCreateRequest) -> BaselineRunCreateResponse:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise BaselineRunConfigurationError("OPENROUTER_API_KEY is required for real baseline elicitation")

    payload, profiles, opinions = _profile_records(request.run_id, request.profile_ids)
    _validate_opinion_leaf(request.opinion_leaf, opinions)

    model_name = request.model_name or str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise BaselineRunConfigurationError("No model name configured for baseline elicitation")

    baseline_run_id = f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    artifact_dir = load_settings().runs_root / baseline_run_id
    raw_llm_dir = artifact_dir / "raw_llm"
    created_at = _now()
    state = BaselineRunResponse(
        baseline_run_id=baseline_run_id,
        status="queued",
        run_id=request.run_id,
        opinion_leaf=request.opinion_leaf,
        model_name=model_name,
        profile_count=len(profiles),
        completed_count=0,
        failed_count=0,
        artifact_dir=str(artifact_dir),
        raw_llm_dir=str(raw_llm_dir),
        created_at=created_at,
        updated_at=created_at,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_llm_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        artifact_dir / "metadata.json",
        {
            "baseline_run_id": baseline_run_id,
            "run_id": request.run_id,
            "opinion_leaf": request.opinion_leaf,
            "model_name": model_name,
            "profile_ids": [profile.profile_id for profile in profiles],
            "baseline_semantics": "exact_baseline_agent_no_fallback",
        },
    )
    with _LOCK:
        _RUNS[baseline_run_id] = state
        _persist_state(state)

    thread = threading.Thread(
        target=_run_baseline_batch,
        args=(baseline_run_id, profiles, request.opinion_leaf, model_name, api_key, payload, raw_llm_dir, request.max_concurrency),
        daemon=True,
    )
    thread.start()

    return BaselineRunCreateResponse(
        baseline_run_id=baseline_run_id,
        status="queued",
        run_id=request.run_id,
        opinion_leaf=request.opinion_leaf,
        profile_count=len(profiles),
        model_name=model_name,
    )


def get_baseline_run(baseline_run_id: str) -> BaselineRunResponse:
    with _LOCK:
        state = _RUNS.get(baseline_run_id)
    if state is not None:
        return state

    status_path = load_settings().runs_root / _safe_filename(baseline_run_id) / "status.json"
    if not status_path.exists():
        raise BaselineRunNotFoundError(baseline_run_id)
    with status_path.open(encoding="utf-8") as handle:
        return BaselineRunResponse.model_validate(json.load(handle))


def _run_baseline_batch(
    baseline_run_id: str,
    profiles: list[Any],
    opinion_leaf: str,
    model_name: str,
    api_key: str,
    config_payload: dict[str, Any],
    raw_llm_dir: Path,
    max_concurrency: int | None,
) -> None:
    _set_state(baseline_run_id, status="running")
    prompts_dir = load_settings().lab_root / "src" / "backend" / "agentic_framework" / "prompts"
    max_workers = max(1, int(max_concurrency or config_payload.get("max_concurrency") or 1))
    max_workers = min(max_workers, len(profiles))
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
            thread_local.agent = factory.baseline_opinion_agent()
        return thread_local.agent

    def process(profile: Any) -> BaselineResult:
        scenario_id = f"{baseline_run_id}_{profile.profile_id}"
        call_id = f"{scenario_id}_baseline"
        assessment = agent_for_thread().assess(
            run_id=baseline_run_id,
            call_id=call_id,
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            profile=profile,
        )
        return BaselineResult(
            profile_id=str(profile.profile_id),
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            score=int(assessment.score),
            confidence=float(assessment.confidence),
            reasoning=str(assessment.reasoning),
            model_name=str(assessment.model_name),
            call_id=call_id,
            timestamp=_now(),
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_profile = {executor.submit(process, profile): profile for profile in profiles}
            for future in as_completed(future_to_profile):
                profile = future_to_profile[future]
                scenario_id = f"{baseline_run_id}_{profile.profile_id}"
                try:
                    _append_result(baseline_run_id, result=future.result())
                except Exception as exc:
                    _append_result(
                        baseline_run_id,
                        error=BaselineRunError(
                            profile_id=str(profile.profile_id),
                            scenario_id=scenario_id,
                            message=str(exc),
                            timestamp=_now(),
                        ),
                    )
        final = get_baseline_run(baseline_run_id)
        if final.failed_count and final.completed_count:
            _set_state(baseline_run_id, status="completed_with_errors")
        elif final.failed_count:
            _set_state(baseline_run_id, status="failed")
        else:
            _set_state(baseline_run_id, status="completed")
    except Exception as exc:
        _set_state(baseline_run_id, status="failed")
        _write_json(Path(get_baseline_run(baseline_run_id).artifact_dir) / "fatal_error.json", {"message": str(exc), "timestamp": _now()})
