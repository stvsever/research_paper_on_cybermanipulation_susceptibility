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

from profile_network_workbench_api.backend_adapter import (
    canonical_leaf_path,
    reconstruct_profile_bundles,
    selected_attack_leaves,
)
from profile_network_workbench_api.schemas import (
    BaselinePromptMessage,
    BaselineResult,
    PostAttackNetworkExposurePromptPreviewResponse,
    PostAttackNetworkExposureResult,
    PostAttackNetworkExposureRunCreateRequest,
    PostAttackNetworkExposureRunCreateResponse,
    PostAttackNetworkExposureRunError,
    PostAttackNetworkExposureRunResponse,
    PostAttackNetworkExposureRunStatus,
    PostExposureResult,
)
from profile_network_workbench_api.services.baseline_runs import BaselineRunNotFoundError, get_baseline_run
from profile_network_workbench_api.services.post_exposure_runs import (
    PostExposureRunNotFoundError,
    _compile_attack_spec,
    get_post_exposure_run,
)
from profile_network_workbench_api.settings import load_settings


class PostAttackNetworkExposureRunNotFoundError(KeyError):
    pass


class PostAttackNetworkExposureRunConfigurationError(RuntimeError):
    pass


AGENT_FACTORY_CLASS: Any | None = None

_RUNS: dict[str, PostAttackNetworkExposureRunResponse] = {}
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


def _persist_state(state: PostAttackNetworkExposureRunResponse) -> None:
    artifact_dir = Path(state.artifact_dir)
    _write_json(artifact_dir / "status.json", state.model_dump())
    _write_jsonl(artifact_dir / "results.jsonl", [item.model_dump() for item in state.results])
    _write_jsonl(artifact_dir / "errors.jsonl", [item.model_dump() for item in state.errors])


def _set_state(run_id: str, **updates: Any) -> PostAttackNetworkExposureRunResponse:
    with _LOCK:
        current = _RUNS[run_id]
        payload = current.model_dump()
        payload.update(updates)
        payload["updated_at"] = _now()
        next_state = PostAttackNetworkExposureRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)
        return next_state


def _append_result(
    run_id: str,
    result: PostAttackNetworkExposureResult | None = None,
    error: PostAttackNetworkExposureRunError | None = None,
) -> None:
    with _LOCK:
        current = _RUNS[run_id]
        results = list(current.results)
        errors = list(current.errors)
        if result is not None:
            results.append(result)
        if error is not None:
            errors.append(error)
        status: PostAttackNetworkExposureRunStatus = "running"
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
        next_state = PostAttackNetworkExposureRunResponse.model_validate(payload)
        _RUNS[run_id] = next_state
        _persist_state(next_state)


def _agent_factory_class() -> Any:
    if AGENT_FACTORY_CLASS is not None:
        return AGENT_FACTORY_CLASS
    from src.backend.agentic_framework.factory import AgentFactory

    return AgentFactory


def _profile_context(run_id: str) -> tuple[dict[str, Any], Any, dict[str, Any], list[Any], list[str]]:
    payload, config, ontologies, bundles, opinions = reconstruct_profile_bundles(run_id)
    profiles = [bundle["profile_result"].profile for bundle in bundles]
    if not profiles:
        raise PostAttackNetworkExposureRunConfigurationError("No profiles available for post-network run")
    return payload, config, ontologies, profiles, opinions


def _validate_opinion_leaf(opinion_leaf: str, configured_opinions: list[str]) -> str:
    canonical_opinion_leaf = canonical_leaf_path(opinion_leaf, configured_opinions)
    if canonical_opinion_leaf is None:
        raise PostAttackNetworkExposureRunConfigurationError("Opinion leaf must be one of the run-configured opinion leaves")
    return canonical_opinion_leaf


def _validate_attack_leaf(attack_leaf: str, config: Any, ontologies: dict[str, Any]) -> str:
    configured_attacks = selected_attack_leaves(ontologies, config)
    canonical_attack_leaf = canonical_leaf_path(attack_leaf, configured_attacks)
    if canonical_attack_leaf is None:
        raise PostAttackNetworkExposureRunConfigurationError("Attack leaf must be one of the run-configured attack leaves")
    return canonical_attack_leaf


def _baseline_result_by_profile(
    baseline_run_id: str,
    run_id: str,
    opinion_leaf: str,
) -> dict[str, BaselineResult]:
    try:
        baseline_run = get_baseline_run(baseline_run_id)
    except BaselineRunNotFoundError as exc:
        raise PostAttackNetworkExposureRunConfigurationError(f"Baseline run not found: {baseline_run_id}") from exc

    if baseline_run.status != "completed":
        raise PostAttackNetworkExposureRunConfigurationError("Post-network exposure requires a completed baseline run")
    if baseline_run.run_id != run_id:
        raise PostAttackNetworkExposureRunConfigurationError("Baseline run belongs to a different run_id")
    if baseline_run.opinion_leaf != opinion_leaf:
        raise PostAttackNetworkExposureRunConfigurationError("Baseline run uses a different opinion leaf")
    return {item.profile_id: item for item in baseline_run.results}


def _post_result_by_profile(
    post_run_id: str,
    baseline_run_id: str,
    run_id: str,
    opinion_leaf: str,
    attack_leaf: str,
) -> dict[str, PostExposureResult]:
    try:
        post_run = get_post_exposure_run(post_run_id)
    except PostExposureRunNotFoundError as exc:
        raise PostAttackNetworkExposureRunConfigurationError(f"Post-exposure run not found: {post_run_id}") from exc

    if post_run.status != "completed":
        raise PostAttackNetworkExposureRunConfigurationError("Post-network exposure requires a completed post-exposure run")
    if post_run.run_id != run_id:
        raise PostAttackNetworkExposureRunConfigurationError("Post-exposure run belongs to a different run_id")
    if post_run.baseline_run_id != baseline_run_id:
        raise PostAttackNetworkExposureRunConfigurationError("Post-exposure run is linked to a different baseline run")
    if post_run.opinion_leaf != opinion_leaf:
        raise PostAttackNetworkExposureRunConfigurationError("Post-exposure run uses a different opinion leaf")
    if post_run.attack_leaf != attack_leaf:
        raise PostAttackNetworkExposureRunConfigurationError("Post-exposure run uses a different attack leaf")
    return {item.profile_id: item for item in post_run.results}


def _select_profiles(
    profiles: list[Any],
    baseline_by_profile: dict[str, BaselineResult],
    post_by_profile: dict[str, PostExposureResult],
    requested_profile_ids: list[str] | None,
) -> list[Any]:
    profile_by_id = {str(profile.profile_id): profile for profile in profiles}
    eligible = set(baseline_by_profile).intersection(post_by_profile)
    if requested_profile_ids:
        requested = set(requested_profile_ids)
        unknown = sorted(requested - set(profile_by_id))
        if unknown:
            raise PostAttackNetworkExposureRunConfigurationError(f"Unknown profile id: {unknown[0]}")
        missing_baseline = sorted(requested - set(baseline_by_profile))
        if missing_baseline:
            raise PostAttackNetworkExposureRunConfigurationError(f"Baseline result missing for profile: {missing_baseline[0]}")
        missing_post = sorted(requested - set(post_by_profile))
        if missing_post:
            raise PostAttackNetworkExposureRunConfigurationError(f"Post-exposure result missing for profile: {missing_post[0]}")
        return [profile_by_id[profile_id] for profile_id in requested_profile_ids]

    selected = [profile for profile in profiles if str(profile.profile_id) in eligible]
    if not selected:
        raise PostAttackNetworkExposureRunConfigurationError("No profiles with completed post-exposure results are available")
    return selected


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _peer_context(
    *,
    target_profile: Any,
    all_profiles: list[Any],
    baseline_by_profile: dict[str, BaselineResult],
    post_by_profile: dict[str, PostExposureResult],
    opinion_leaf: str,
    attack_leaf: str,
    adversarial_direction: int,
    top_k: int,
) -> dict[str, Any]:
    from src.backend.utils.profile_affinity import compute_profile_affinities, rank_profile_neighbors

    available_profiles = [
        profile
        for profile in all_profiles
        if str(profile.profile_id) in baseline_by_profile and str(profile.profile_id) in post_by_profile
    ]
    if len(available_profiles) < 2:
        raise PostAttackNetworkExposureRunConfigurationError(
            "Post-network exposure requires at least one same-condition peer"
        )

    affinities = compute_profile_affinities(available_profiles)
    target_id = str(target_profile.profile_id)
    peers: list[dict[str, Any]] = []
    for affinity in rank_profile_neighbors(target_id, affinities, limit=max(0, len(available_profiles) - 1)):
        peer_id = affinity.other(target_id)
        baseline = baseline_by_profile.get(peer_id)
        post = post_by_profile.get(peer_id)
        if baseline is None or post is None:
            continue
        peers.append(
            {
                "profile_id": peer_id,
                "affinity": round(float(affinity.affinity), 6),
                "baseline_score": int(baseline.score),
                "post_score": int(post.post_score),
                "attack_delta": int(post.post_score - baseline.score),
                "confidence": round(float(post.confidence), 4),
                "reasoning": str(post.reasoning),
            }
        )
        if len(peers) >= top_k:
            break

    if not peers:
        raise PostAttackNetworkExposureRunConfigurationError(
            "Post-network exposure requires at least one same-condition peer"
        )

    peer_post_scores = [float(peer["post_score"]) for peer in peers]
    peer_deltas = [float(peer["attack_delta"]) for peer in peers]
    affinity_weight_sum = sum(float(peer["affinity"]) for peer in peers)
    weighted_post_mean = (
        sum(float(peer["post_score"]) * float(peer["affinity"]) for peer in peers) / affinity_weight_sum
        if affinity_weight_sum > 0.0
        else None
    )
    weighted_delta_mean = (
        sum(float(peer["attack_delta"]) * float(peer["affinity"]) for peer in peers) / affinity_weight_sum
        if affinity_weight_sum > 0.0
        else None
    )
    target_baseline = baseline_by_profile[target_id]
    target_post = post_by_profile[target_id]
    return {
        "top_k": int(top_k),
        "opinion_leaf": opinion_leaf,
        "attack_present": True,
        "attack_leaf": attack_leaf,
        "adversarial_direction": int(adversarial_direction),
        "target_baseline_score": int(target_baseline.score),
        "target_baseline_confidence": round(float(target_baseline.confidence), 4),
        "target_private_post_score": int(target_post.post_score),
        "target_private_post_confidence": round(float(target_post.confidence), 4),
        "target_private_attack_delta": int(target_post.post_score - target_baseline.score),
        "peer_post_mean": round(float(_safe_mean(peer_post_scores)), 4) if peer_post_scores else None,
        "peer_delta_mean": round(float(_safe_mean(peer_deltas)), 4) if peer_deltas else None,
        "peer_post_sd": round(float(_safe_pstdev(peer_post_scores)), 4) if peer_post_scores else None,
        "peer_delta_sd": round(float(_safe_pstdev(peer_deltas)), 4) if peer_deltas else None,
        "affinity_weighted_peer_post_mean": round(float(weighted_post_mean), 4) if weighted_post_mean is not None else None,
        "affinity_weighted_peer_delta_mean": round(float(weighted_delta_mean), 4) if weighted_delta_mean is not None else None,
        "peer_assessments": peers,
    }


def _run_inputs(
    run_id: str,
    baseline_run_id: str,
    post_run_id: str,
    opinion_leaf: str,
    attack_leaf: str,
    profile_ids: list[str] | None,
    top_k: int,
    scenario_id_prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    payload, config, ontologies, all_profiles, opinions = _profile_context(run_id)
    opinion_leaf = _validate_opinion_leaf(opinion_leaf, opinions)
    attack_leaf = _validate_attack_leaf(attack_leaf, config, ontologies)
    baseline_by_profile = _baseline_result_by_profile(baseline_run_id, run_id, opinion_leaf)
    post_by_profile = _post_result_by_profile(post_run_id, baseline_run_id, run_id, opinion_leaf, attack_leaf)
    target_profiles = _select_profiles(all_profiles, baseline_by_profile, post_by_profile, profile_ids)

    inputs: list[dict[str, Any]] = []
    for profile in target_profiles:
        profile_id = str(profile.profile_id)
        baseline = baseline_by_profile[profile_id]
        private_post = post_by_profile[profile_id]
        attack_spec, adversarial_direction = _compile_attack_spec(
            ontologies=ontologies,
            profile=profile,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            baseline_score=int(baseline.score),
        )
        scenario_id = f"{scenario_id_prefix}_{profile_id}"
        post_attack_network_context = _peer_context(
            target_profile=profile,
            all_profiles=all_profiles,
            baseline_by_profile=baseline_by_profile,
            post_by_profile=post_by_profile,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            adversarial_direction=adversarial_direction,
            top_k=top_k,
        )
        inputs.append(
            {
                "profile": profile,
                "baseline": baseline,
                "private_post": private_post,
                "scenario_id": scenario_id,
                "attack_spec": attack_spec,
                "adversarial_direction": adversarial_direction,
                "post_attack_network_context": post_attack_network_context,
            }
        )
    return payload, inputs, opinion_leaf, attack_leaf


def get_post_attack_network_exposure_prompt_preview(
    run_id: str,
    baseline_run_id: str,
    post_run_id: str,
    opinion_leaf: str,
    attack_leaf: str,
    profile_id: str,
    top_k: int = 8,
) -> PostAttackNetworkExposurePromptPreviewResponse:
    payload, inputs, opinion_leaf, attack_leaf = _run_inputs(
        run_id,
        baseline_run_id,
        post_run_id,
        opinion_leaf,
        attack_leaf,
        [profile_id],
        max(1, int(top_k)),
        f"preview_{run_id}_post_network",
    )
    model_name = str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise PostAttackNetworkExposureRunConfigurationError("No model name configured for post-network exposure")

    item = inputs[0]
    profile = item["profile"]
    prompt_name = "post_attack_network_exposure_opinion.md"
    prompts_dir = load_settings().lab_root / "src" / "backend" / "agentic_framework" / "prompts"
    system_prompt = (prompts_dir / prompt_name).read_text(encoding="utf-8")
    user_payload = {
        "scenario_id": f"preview_{run_id}_{profile.profile_id}_post_network",
        "opinion_leaf": opinion_leaf,
        "profile": profile.model_dump(),
        "baseline_score": int(item["baseline"].score),
        "private_post_score": int(item["private_post"].post_score),
        "attack_present": True,
        "adversarial_direction": int(item["adversarial_direction"]),
        "attack_leaf": attack_leaf,
        "attack_vector_spec": item["attack_spec"],
        "post_attack_network_context": item["post_attack_network_context"],
    }
    user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return PostAttackNetworkExposurePromptPreviewResponse(
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        post_run_id=post_run_id,
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


def start_post_attack_network_exposure_run(
    request: PostAttackNetworkExposureRunCreateRequest,
) -> PostAttackNetworkExposureRunCreateResponse:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise PostAttackNetworkExposureRunConfigurationError(
            "OPENROUTER_API_KEY is required for real post-network exposure elicitation"
        )

    top_k = max(1, int(request.top_k or 8))
    post_network_run_id = f"post_network_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    payload, profile_inputs, opinion_leaf, attack_leaf = _run_inputs(
        request.run_id,
        request.baseline_run_id,
        request.post_run_id,
        request.opinion_leaf,
        request.attack_leaf,
        request.profile_ids,
        top_k,
        post_network_run_id,
    )
    model_name = request.model_name or str(payload.get("openrouter_model") or "").strip()
    if not model_name:
        raise PostAttackNetworkExposureRunConfigurationError("No model name configured for post-network exposure")

    artifact_dir = load_settings().runs_root / post_network_run_id
    raw_llm_dir = artifact_dir / "raw_llm"
    contexts_path = artifact_dir / "post_attack_network_contexts.jsonl"
    created_at = _now()
    state = PostAttackNetworkExposureRunResponse(
        post_network_run_id=post_network_run_id,
        status="queued",
        run_id=request.run_id,
        baseline_run_id=request.baseline_run_id,
        post_run_id=request.post_run_id,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        model_name=model_name,
        profile_count=len(profile_inputs),
        completed_count=0,
        failed_count=0,
        top_k=top_k,
        artifact_dir=str(artifact_dir),
        raw_llm_dir=str(raw_llm_dir),
        post_attack_network_contexts_path=str(contexts_path),
        created_at=created_at,
        updated_at=created_at,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_llm_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        contexts_path,
        [
            {
                "scenario_id": item["scenario_id"],
                "profile_id": str(item["profile"].profile_id),
                "opinion_leaf": opinion_leaf,
                "attack_leaf": attack_leaf,
                "post_attack_network_context": item["post_attack_network_context"],
            }
            for item in profile_inputs
        ],
    )
    _write_json(
        artifact_dir / "metadata.json",
        {
            "post_network_run_id": post_network_run_id,
            "baseline_run_id": request.baseline_run_id,
            "post_run_id": request.post_run_id,
            "run_id": request.run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "model_name": model_name,
            "top_k": top_k,
            "profile_ids": [str(item["profile"].profile_id) for item in profile_inputs],
            "post_attack_network_exposure_semantics": "post_attack_network_agent_with_same_condition_post_peer_context_no_fallback",
            "post_attack_network_contexts_path": str(contexts_path),
        },
    )
    with _LOCK:
        _RUNS[post_network_run_id] = state
        _persist_state(state)

    thread = threading.Thread(
        target=_run_post_attack_network_batch,
        args=(
            post_network_run_id,
            profile_inputs,
            opinion_leaf,
            attack_leaf,
            model_name,
            api_key,
            payload,
            raw_llm_dir,
            request.max_concurrency,
        ),
        daemon=True,
    )
    thread.start()

    return PostAttackNetworkExposureRunCreateResponse(
        post_network_run_id=post_network_run_id,
        status="queued",
        run_id=request.run_id,
        baseline_run_id=request.baseline_run_id,
        post_run_id=request.post_run_id,
        opinion_leaf=opinion_leaf,
        attack_leaf=attack_leaf,
        profile_count=len(profile_inputs),
        model_name=model_name,
        top_k=top_k,
    )


def get_post_attack_network_exposure_run(post_network_run_id: str) -> PostAttackNetworkExposureRunResponse:
    with _LOCK:
        state = _RUNS.get(post_network_run_id)
    if state is not None:
        return state

    status_path = load_settings().runs_root / _safe_filename(post_network_run_id) / "status.json"
    if not status_path.exists():
        raise PostAttackNetworkExposureRunNotFoundError(post_network_run_id)
    with status_path.open(encoding="utf-8") as handle:
        return PostAttackNetworkExposureRunResponse.model_validate(json.load(handle))


def _run_post_attack_network_batch(
    post_network_run_id: str,
    profile_inputs: list[dict[str, Any]],
    opinion_leaf: str,
    attack_leaf: str,
    model_name: str,
    api_key: str,
    config_payload: dict[str, Any],
    raw_llm_dir: Path,
    max_concurrency: int | None,
) -> None:
    _set_state(post_network_run_id, status="running")
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
            thread_local.agent = factory.post_attack_network_exposure_opinion_agent()
        return thread_local.agent

    def process(item: dict[str, Any]) -> PostAttackNetworkExposureResult:
        profile = item["profile"]
        baseline = item["baseline"]
        private_post = item["private_post"]
        scenario_id = str(item["scenario_id"])
        attack_spec = dict(item["attack_spec"])
        adversarial_direction = int(item["adversarial_direction"])
        context = dict(item["post_attack_network_context"])
        call_id = f"{scenario_id}_post_attack_network"
        assessment = agent_for_thread().assess(
            run_id=post_network_run_id,
            call_id=call_id,
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            profile=profile,
            baseline_score=int(baseline.score),
            private_post_score=int(private_post.post_score),
            attack_present=True,
            adversarial_direction=adversarial_direction,
            attack_leaf=attack_leaf,
            attack_vector_spec=attack_spec,
            post_attack_network_context=context,
        )
        network_score = int(assessment.score)

        from src.backend.utils.scenario_realism import assess_post_attack_network_exposure_heuristics

        heuristics = assess_post_attack_network_exposure_heuristics(
            private_post_score=int(private_post.post_score),
            network_score=network_score,
            confidence=float(assessment.confidence),
            peer_count=len(context.get("peer_assessments", [])),
            min_peers=1,
            adversarial_direction=adversarial_direction,
        )
        return PostAttackNetworkExposureResult(
            profile_id=str(profile.profile_id),
            scenario_id=scenario_id,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            baseline_score=int(baseline.score),
            private_post_score=int(private_post.post_score),
            post_attack_network_score=network_score,
            delta_from_baseline=int(network_score - int(baseline.score)),
            increment_from_private_post=int(network_score - int(private_post.post_score)),
            adversarial_direction=adversarial_direction,
            confidence=float(assessment.confidence),
            reasoning=str(assessment.reasoning),
            model_name=str(assessment.model_name),
            call_id=call_id,
            timestamp=_now(),
            post_attack_network_context=context,
            heuristic_checks=heuristics,
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process, item): item for item in profile_inputs}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                profile = item["profile"]
                scenario_id = str(item["scenario_id"])
                try:
                    _append_result(post_network_run_id, result=future.result())
                except Exception as exc:
                    _append_result(
                        post_network_run_id,
                        error=PostAttackNetworkExposureRunError(
                            profile_id=str(profile.profile_id),
                            scenario_id=scenario_id,
                            message=str(exc),
                            timestamp=_now(),
                        ),
                    )
        final = get_post_attack_network_exposure_run(post_network_run_id)
        if final.failed_count and final.completed_count:
            _set_state(post_network_run_id, status="completed_with_errors")
        elif final.failed_count:
            _set_state(post_network_run_id, status="failed")
        else:
            _set_state(post_network_run_id, status="completed")
    except Exception as exc:
        _set_state(post_network_run_id, status="failed")
        _write_json(
            Path(get_post_attack_network_exposure_run(post_network_run_id).artifact_dir) / "fatal_error.json",
            {"message": str(exc), "timestamp": _now()},
        )
