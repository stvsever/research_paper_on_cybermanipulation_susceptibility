from __future__ import annotations

from fastapi import APIRouter, HTTPException

from profile_network_workbench_api.schemas import (
    PostExposurePromptPreviewResponse,
    PostExposureRunCreateRequest,
    PostExposureRunCreateResponse,
    PostExposureRunResponse,
)
from profile_network_workbench_api.services.post_exposure_runs import (
    PostExposureRunConfigurationError,
    PostExposureRunNotFoundError,
    get_post_exposure_prompt_preview,
    get_post_exposure_run,
    start_post_exposure_run,
)

router = APIRouter(prefix="/api", tags=["post-exposure-runs"])


@router.post("/post-exposure-runs", response_model=PostExposureRunCreateResponse)
def create_post_exposure_run(request: PostExposureRunCreateRequest) -> PostExposureRunCreateResponse:
    try:
        return start_post_exposure_run(request)
    except PostExposureRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/post-exposure-runs/{post_run_id}", response_model=PostExposureRunResponse)
def read_post_exposure_run(post_run_id: str) -> PostExposureRunResponse:
    try:
        return get_post_exposure_run(post_run_id)
    except PostExposureRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Post-exposure run not found: {post_run_id}") from exc


@router.get("/post-exposure-prompt-preview", response_model=PostExposurePromptPreviewResponse)
def read_post_exposure_prompt_preview(
    run_id: str,
    baseline_run_id: str,
    opinion_leaf: str,
    attack_leaf: str,
    profile_id: str,
) -> PostExposurePromptPreviewResponse:
    try:
        return get_post_exposure_prompt_preview(
            run_id=run_id,
            baseline_run_id=baseline_run_id,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
            profile_id=profile_id,
        )
    except PostExposureRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
