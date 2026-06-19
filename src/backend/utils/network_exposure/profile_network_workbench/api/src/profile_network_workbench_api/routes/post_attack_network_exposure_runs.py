from __future__ import annotations

from fastapi import APIRouter, HTTPException

from profile_network_workbench_api.schemas import (
    PostAttackNetworkExposurePromptPreviewResponse,
    PostAttackNetworkExposureRunCreateRequest,
    PostAttackNetworkExposureRunCreateResponse,
    PostAttackNetworkExposureRunResponse,
)
from profile_network_workbench_api.services.post_attack_network_exposure_runs import (
    PostAttackNetworkExposureRunConfigurationError,
    PostAttackNetworkExposureRunNotFoundError,
    get_post_attack_network_exposure_prompt_preview,
    get_post_attack_network_exposure_run,
    start_post_attack_network_exposure_run,
)

router = APIRouter(prefix="/api", tags=["post-attack-network-exposure-runs"])


@router.post("/post-attack-network-exposure-runs", response_model=PostAttackNetworkExposureRunCreateResponse)
def create_post_attack_network_exposure_run(
    request: PostAttackNetworkExposureRunCreateRequest,
) -> PostAttackNetworkExposureRunCreateResponse:
    try:
        return start_post_attack_network_exposure_run(request)
    except PostAttackNetworkExposureRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/post-attack-network-exposure-runs/{post_network_run_id}", response_model=PostAttackNetworkExposureRunResponse)
def read_post_attack_network_exposure_run(post_network_run_id: str) -> PostAttackNetworkExposureRunResponse:
    try:
        return get_post_attack_network_exposure_run(post_network_run_id)
    except PostAttackNetworkExposureRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Post-attack network exposure run not found: {post_network_run_id}",
        ) from exc


@router.get("/post-attack-network-exposure-prompt-preview", response_model=PostAttackNetworkExposurePromptPreviewResponse)
def read_post_attack_network_exposure_prompt_preview(
    run_id: str,
    baseline_run_id: str,
    post_run_id: str,
    opinion_leaf: str,
    attack_leaf: str,
    profile_id: str,
    top_k: int = 8,
) -> PostAttackNetworkExposurePromptPreviewResponse:
    try:
        return get_post_attack_network_exposure_prompt_preview(
            run_id=run_id,
            baseline_run_id=baseline_run_id,
            post_run_id=post_run_id,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            profile_id=profile_id,
            top_k=top_k,
        )
    except PostAttackNetworkExposureRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
