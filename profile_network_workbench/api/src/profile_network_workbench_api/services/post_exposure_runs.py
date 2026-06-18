from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from profile_network_workbench_api.backend_adapter import (
    canonical_leaf_path,
    load_backend_helpers,
    reconstruct_profile_bundles,
    selected_attack_leaves,
)
from profile_network_workbench_api.schemas import (
    BaselinePromptMessage,
    BaselineResult,
    PostExposurePromptPreviewResponse,
    PostExposureResult,
    PostExposureRunCreateRequest,
    PostExposureRunCreateResponse,
    PostExposureRunError,
    PostExposureRunResponse,
    PostExposureRunStatus,
)
from profile_network_workbench_api.services.baseline_runs import BaselineRunNotFoundError, get_baseline_run
from profile_network_workbench_api.settings import load_settings


class PostExposureRunNotFoundError(KeyError):
    pass


class PostExposureRunConfigurationError(RuntimeError):
    pass


AGENT_FACTORY_CLASS: Any | None = None

_RUNS: dict[str, PostExposureRunResponse] = {}
_LOCK = threading.Lock()

_SPEC_CONTEXT_KEYS = (
    "opinion_domain",
    "opinion_leaf_label",
    "attack_leaf_label",
    "adversarial_direction",
    "adversarial_direction_label",
    "baseline_vs_goal",
    "persuasion_goal",
    "motivational_lever",
    "emotional_register",
    "issue_frame",
    "attack_mechanism",
    "attack_primary_system",
    "attack_platform_hint",
    "attack_complexity_tier",
    "attack_temporal_horizon",
    "attack_epistemic_target",
    "attack_requires_personalization",
    "attack_agent_orchestration_required",
)


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


def _persist_state(state: PostExposureRunResponse) -> None:
    artifact_dir = Path(state.artifact_dir)
    _write_json(artifact_dir / "status.json", state.model_dump())
    _write_jsonl(artifact_dir / "results.jsonl", [item.model_dump() for item in state.results])
    _write_jsonl(artifact_dir / "errors.jsonl", [item.model_dump() for item in state.errors])


def _set_state(run_id: str, **updates: Any) -> PostExposureRunResponse:
    with _LOCK:
        current = _RUNS[run_id]
        payload = current.model_dump()
        payload.update(updates)
        payload["updated_at"] = _now()
        next_state = PostExposureRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)
        return next_state


def _append_result(
    run_id: str,
    result: PostExposureResult | None = None,
    error: PostExposureRunError | None = None,
) -> None:
    with _LOCK:
        current = _RUNS[run_id]
        results = list(current.results)
        errors = list(current.errors)
        if result is not None:
            results.append(result)
        if error is not None:
            errors.append(error)
        status: PostExposureRunStatus = "running"
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
        next_state = PostExposureRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)


def _agent_factory_class() -> Any:
    if AGENT_FACTORY_CLASS is not None:
        return AGENT_FACTORY_CLASS
    from src.backend.agentic_framework.factory import AgentFactory

    return AgentFactory


def _tier_intensity_proxy(complexity_tier: str) -> float:
    tier = (complexity_tier or "").upper()
    if tier.startswith("T4"):
        return 0.70
    if tier.startswith("T3"):
        return 0.60
    if tier.startswith("T2"):
        return 0.50
    if tier.startswith("T1"):
        return 0.35
    return 0.50


def _profile_context(run_id: str, requested_profile_ids: list[str] | None) -> tuple[dict[str, Any], Any, dict[str, Any], list[Any], list[str]]:
    payload, config, ontologies, bundles, opinions = reconstruct_profile_bundles(run_id)
    profiles = [bundle["profile_result"].profile for bundle in bundles]
    if requested_profile_ids:
        requested = set(requested_profile_ids)
        profiles = [profile for profile in profiles if profile.profile_id in requested]
        missing = sorted(requested - {profile.profile_id for profile in profiles})
        if missing:
            raise PostExposureRunConfigurationError(f"Unknown profile id: {missing[0]}")
    if not profiles:
        raise PostExposureRunConfigurationError("No profiles selected for post-exposure run")
    return payload, config, ontologies, profiles, opinions


def _validate_opinion_leaf(opinion_leaf: str, configured_opinions: list[str]) -> str:
    canonical_opinion_leaf = canonical_leaf_path(opinion_leaf, configured_opinions)
    if canonical_opinion_leaf is None:
        raise PostExposureRunConfigurationError("Opinion leaf must be one of the run-configured opinion leaves")
    return canonical_opinion_leaf


def _validate_attack_leaf(attack_leaf: str, config: Any, ontologies: dict[str, Any]) -> str:
    configured_attacks = selected_attack_leaves(ontologies, config)
    canonical_attack_leaf = canonical_leaf_path(attack_leaf, configured_attacks)
    if canonical_attack_leaf is None:
        raise PostExposureRunConfigurationError("Attack leaf must be one of the run-configured attack leaves")
    return canonical_attack_leaf


def _baseline_result_by_profile(baseline_run_id: str, run_id: str, opinion_leaf: str, profiles: list[Any]) -> dict[str, BaselineResult]:
    try:
        baseline_run = get_baseline_run(baseline_run_id)
    except BaselineRunNotFoundError as exc:
        raise PostExposureRunConfigurationError(f"Baseline run not found: {baseline_run_id}") from exc

    if baseline_run.status not in {"completed", "completed_with_errors"}:
        raise PostExposureRunConfigurationError("Post-exposure requires a completed baseline run")
    if baseline_run.run_id != run_id:
        raise PostExposureRunConfigurationError("Baseline run belongs to a different run_id")
    if baseline_run.opinion_leaf != opinion_leaf:
        raise PostExposureRunConfigurationError("Baseline run uses a different opinion leaf")

    by_profile = {item.profile_id: item for item in baseline_run.results}
    missing = [str(profile.profile_id) for profile in profiles if str(profile.profile_id) not in by_profile]
    if missing:
        raise PostExposureRunConfigurationError(f"Baseline result missing for profile: {missing[0]}")
    return by_profile


def _attack_metadata_dict(meta: Any) -> dict[str, Any]:
    requires = tuple(getattr(meta, "requires_capability", tuple()) or tuple())
    family = str(getattr(meta, "family", "") or "")
    return {
        "mechanism": family,
        "primary_system": family,
        "platform_hint": str(getattr(meta, "scenario_role", "") or ""),
        "complexity_tier": str(getattr(meta, "complexity_tier", "") or ""),
        "temporal_horizon": str(getattr(meta, "temporal_horizon", "") or ""),
        "epistemic_target": str(getattr(meta, "epistemic_target", "") or ""),
        "requires_personalization": "profile_personalisation" in requires,
        "agent_orchestration_required": "agent_orchestration" in requires,
    }


def _adversarial_direction(opinion_leaf: str, ontologies: dict[str, Any]) -> int:
    backend = load_backend_helpers()
    opinion_index = dict(backend["load_opinion_metadata_index"](ontologies["OPINION"]))
    if opinion_leaf in opinion_index:
        return int(opinion_index[opinion_leaf].adversarial_direction)
    directions, _ = backend["load_adversarial_directions_from_opinion"](ontologies["OPINION"])
    return int(directions.get(opinion_leaf.split(">")[-1].strip(), 0))


def _compile_attack_spec(
    *,
    ontologies: dict[str, Any],
    profile: Any,
    opinion_leaf: str,
    attack_leaf: str,
    baseline_score: int,
) -> tuple[dict[str, Any], int]:
    backend = load_backend_helpers()
    attack_index = dict(backend["load_attack_metadata_index"](ontologies["ATTACK"]))
    attack_meta = attack_index.get(attack_leaf)
    attack_meta_dict = _attack_metadata_dict(attack_meta) if attack_meta is not None else {}
    adv_direction = _adversarial_direction(opinion_leaf, ontologies)
    full_context = backend["build_attack_context"](
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        profile=profile,
        baseline_score=baseline_score,
        adversarial_direction=adv_direction,
        attack_metadata=attack_meta_dict,
    )
    attack_context = {key: full_context.get(key) for key in _SPEC_CONTEXT_KEYS}
    spec = {
        "attack_present": True,
        "attack_leaf": attack_leaf,
        "attack_leaf_label": attack_context.get("attack_leaf_label", attack_leaf.split(">")[-1].strip()),
        "attack_context": attack_context,
        "intensity_proxy": _tier_intensity_proxy(str(attack_context.get("attack_complexity_tier", ""))),
        "spec_source": "deterministic_ontology_v1",
    }
    return spec, adv_direction


def _clamp_post_score(baseline_score: int, post_score: int, adversarial_direction: int) -> tuple[int, bool]:
    bounded = max(-1000, min(1000, int(post_score)))
    if adversarial_direction > 0 and bounded < baseline_score:
        return int(baseline_score), True
    if adversarial_direction < 0 and bounded > baseline_score:
        return int(baseline_score), True
    if adversarial_direction == 0 and abs(bounded - baseline_score) > 120:
        sign = 1 if bounded > baseline_score else -1
        return max(-1000, min(1000, int(baseline_score + sign * 120))), True
    return bounded, bounded != int(post_score)


def get_post_exposure_prompt_preview(
    run_id: str,
    baseline_run_id: str,
    opinion_leaf: str,
    attack_leaf: str,
    profile_id: str,
) -> PostExposurePromptPreviewResponse:
    payload, config, ontologies, profiles, opinions = _profile_context(run_id, [profile_id])
    opinion_leaf = _validate_opinion_leaf(opinion_leaf, opinions)
    attack_leaf = _validate_attack_leaf(attack_leaf, config, ontologies)
    baseline_by_profile = _baseline_result_by_profile(baseline_run_id, run_id, opinion_leaf, profiles)

    model_name = str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise PostExposureRunConfigurationError("No model name configured for post-exposure elicitation")

    profile = profiles[0]
    baseline = baseline_by_profile[str(profile.profile_id)]
    attack_spec, adv_direction = _compile_attack_spec(
        ontologies=ontologies,
        profile=profile,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        baseline_score=baseline.score,
    )
    prompt_name = "post_attack_opinion.md"
    prompts_dir = load_settings().lab_root / "src" / "backend" / "agentic_framework" / "prompts"
    system_prompt = (prompts_dir / prompt_name).read_text(encoding="utf-8")
    user_payload = {
        "scenario_id": f"preview_{run_id}_{profile.profile_id}_post",
        "opinion_leaf": opinion_leaf,
        "profile": profile.model_dump(),
        "baseline_score": int(baseline.score),
        "attack_present": True,
        "adversarial_direction": adv_direction,
        "attack_leaf": attack_leaf,
        "attack_vector_spec": attack_spec,
    }
    user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return PostExposurePromptPreviewResponse(
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        profile_id=str(profile.profile_id),
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        prompt_name=prompt_name,
        model_name=model_name,
        system_prompt=system_prompt,
        user_payload=user_payload,
        messages=[
            BaselinePromptMessage(role="system", content=system_prompt),
            BaselinePromptMessage(role="user", content=user_content),
        ],
    )


def start_post_exposure_run(request: PostExposureRunCreateRequest) -> PostExposureRunCreateResponse:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise PostExposureRunConfigurationError("OPENROUTER_API_KEY is required for real post-exposure elicitation")

    payload, config, ontologies, profiles, opinions = _profile_context(request.run_id, request.profile_ids)
    opinion_leaf = _validate_opinion_leaf(request.opinion_leaf, opinions)
    attack_leaf = _validate_attack_leaf(request.attack_leaf, config, ontologies)
    baseline_by_profile = _baseline_result_by_profile(request.baseline_run_id, request.run_id, opinion_leaf, profiles)

    model_name = request.model_name or str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise PostExposureRunConfigurationError("No model name configured for post-exposure elicitation")

    post_run_id = f"post_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    artifact_dir = load_settings().runs_root / post_run_id
    raw_llm_dir = artifact_dir / "raw_llm"
    attack_specs_path = artifact_dir / "attack_specs.jsonl"
    created_at = _now()
    profile_inputs: list[dict[str, Any]] = []
    spec_records: list[dict[str, Any]] = []
    for profile in profiles:
        profile_id = str(profile.profile_id)
        baseline = baseline_by_profile[profile_id]
        scenario_id = f"{post_run_id}_{profile_id}"
        attack_spec, adv_direction = _compile_attack_spec(
            ontologies=ontologies,
            profile=profile,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            baseline_score=int(baseline.score),
        )
        profile_inputs.append(
            {
                "profile": profile,
                "baseline": baseline,
                "scenario_id": scenario_id,
                "attack_spec": attack_spec,
                "adversarial_direction": adv_direction,
            }
        )
        spec_records.append(
            {
                "scenario_id": scenario_id,
                "profile_id": profile_id,
                "baseline_score": int(baseline.score),
                "adversarial_direction": adv_direction,
                **attack_spec,
            }
        )

    state = PostExposureRunResponse(
        post_run_id=post_run_id,
        status="queued",
        run_id=request.run_id,
        baseline_run_id=request.baseline_run_id,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        model_name=model_name,
        profile_count=len(profiles),
        completed_count=0,
        failed_count=0,
        artifact_dir=str(artifact_dir),
        raw_llm_dir=str(raw_llm_dir),
        attack_specs_path=str(attack_specs_path),
        created_at=created_at,
        updated_at=created_at,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_llm_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(attack_specs_path, spec_records)
    _write_json(
        artifact_dir / "metadata.json",
        {
            "post_run_id": post_run_id,
            "baseline_run_id": request.baseline_run_id,
            "run_id": request.run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "model_name": model_name,
            "profile_ids": [str(profile.profile_id) for profile in profiles],
            "post_exposure_semantics": "post_attack_agent_with_deterministic_attack_spec_no_fallback",
            "attack_specs_path": str(attack_specs_path),
        },
    )
    with _LOCK:
        _RUNS[post_run_id] = state
        _persist_state(state)

    thread = threading.Thread(
        target=_run_post_batch,
        args=(post_run_id, profile_inputs, opinion_leaf, attack_leaf, model_name, api_key, payload, raw_llm_dir, request.max_concurrency),
        daemon=True,
    )
    thread.start()

    return PostExposureRunCreateResponse(
        post_run_id=post_run_id,
        status="queued",
        run_id=request.run_id,
        baseline_run_id=request.baseline_run_id,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        profile_count=len(profiles),
        model_name=model_name,
    )


def get_post_exposure_run(post_run_id: str) -> PostExposureRunResponse:
    with _LOCK:
        state = _RUNS.get(post_run_id)
    if state is not None:
        return state

    status_path = load_settings().runs_root / _safe_filename(post_run_id) / "status.json"
    if not status_path.exists():
        raise PostExposureRunNotFoundError(post_run_id)
    with status_path.open(encoding="utf-8") as handle:
        return PostExposureRunResponse.model_validate(json.load(handle))


def _run_post_batch(
    post_run_id: str,
    profile_inputs: list[dict[str, Any]],
    opinion_leaf: str,
    attack_leaf: str,
    model_name: str,
    api_key: str,
    config_payload: dict[str, Any],
    raw_llm_dir: Path,
    max_concurrency: int | None,
) -> None:
    _set_state(post_run_id, status="running")
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
            thread_local.agent = factory.post_attack_opinion_agent()
        return thread_local.agent

    def process(item: dict[str, Any]) -> PostExposureResult:
        profile = item["profile"]
        baseline = item["baseline"]
        scenario_id = str(item["scenario_id"])
        attack_spec = dict(item["attack_spec"])
        adv_direction = int(item["adversarial_direction"])
        call_id = f"{scenario_id}_post_attack"
        assessment = agent_for_thread().assess(
            run_id=post_run_id,
            call_id=call_id,
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            profile=profile,
            baseline_score=int(baseline.score),
            attack_present=True,
            adversarial_direction=adv_direction,
            attack_leaf=attack_leaf,
            attack_vector_spec=attack_spec,
        )
        post_score, clamped = _clamp_post_score(int(baseline.score), int(assessment.score), adv_direction)

        from src.backend.utils.scenario_realism import assess_post_opinion_heuristics

        heuristics = assess_post_opinion_heuristics(
            baseline_score=int(baseline.score),
            post_score=post_score,
            attack_present=True,
            intensity_hint=float(attack_spec.get("intensity_proxy", 0.5) or 0.5),
            shift_sensitivity_proxy=float(profile.continuous_attributes.get("heuristic_shift_sensitivity_proxy", 0.5)),
            adversarial_direction=adv_direction,
        )
        reasoning = str(assessment.reasoning)
        if clamped:
            reasoning = f"{reasoning} [Directional guard: counter-goal or out-of-bounds movement corrected.]"
        return PostExposureResult(
            profile_id=str(profile.profile_id),
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            baseline_score=int(baseline.score),
            post_score=post_score,
            delta_score=int(post_score - int(baseline.score)),
            adversarial_direction=adv_direction,
            confidence=float(assessment.confidence),
            reasoning=reasoning.strip(),
            model_name=str(assessment.model_name),
            call_id=call_id,
            timestamp=_now(),
            heuristic_checks=heuristics,
            post_direction_clamped=clamped,
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process, item): item for item in profile_inputs}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                profile = item["profile"]
                scenario_id = str(item["scenario_id"])
                try:
                    _append_result(post_run_id, result=future.result())
                except Exception as exc:
                    _append_result(
                        post_run_id,
                        error=PostExposureRunError(
                            profile_id=str(profile.profile_id),
                            scenario_id=scenario_id,
                            message=str(exc),
                            timestamp=_now(),
                        ),
                    )
        final = get_post_exposure_run(post_run_id)
        if final.failed_count and final.completed_count:
            _set_state(post_run_id, status="completed_with_errors")
        elif final.failed_count:
            _set_state(post_run_id, status="failed")
        else:
            _set_state(post_run_id, status="completed")
    except Exception as exc:
        _set_state(post_run_id, status="failed")
        _write_json(Path(get_post_exposure_run(post_run_id).artifact_dir) / "fatal_error.json", {"message": str(exc), "timestamp": _now()})
