from __future__ import annotations

from fastapi import APIRouter, HTTPException

from profile_network_workbench_api.schemas import (
    NetworkExposurePromptPreviewResponse,
    NetworkExposureRunCreateRequest,
    NetworkExposureRunCreateResponse,
    NetworkExposureRunResponse,
)
from profile_network_workbench_api.services.network_exposure_runs import (
    NetworkExposureRunConfigurationError,
    NetworkExposureRunNotFoundError,
    get_network_exposure_prompt_preview,
    get_network_exposure_run,
    start_network_exposure_run,
)

router = APIRouter(prefix="/api", tags=["network-exposure-runs"])


@router.post("/network-exposure-runs", response_model=NetworkExposureRunCreateResponse)
def create_network_exposure_run(request: NetworkExposureRunCreateRequest) -> NetworkExposureRunCreateResponse:
    try:
        return start_network_exposure_run(request)
    except NetworkExposureRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/network-exposure-runs/{network_run_id}", response_model=NetworkExposureRunResponse)
def read_network_exposure_run(network_run_id: str) -> NetworkExposureRunResponse:
    try:
        return get_network_exposure_run(network_run_id)
    except NetworkExposureRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Network exposure run not found: {network_run_id}") from exc


@router.get("/network-exposure-prompt-preview", response_model=NetworkExposurePromptPreviewResponse)
def read_network_exposure_prompt_preview(
    run_id: str,
    baseline_run_id: str,
    opinion_leaf: str,
    profile_id: str,
    top_k: int = 8,
) -> NetworkExposurePromptPreviewResponse:
    try:
        return get_network_exposure_prompt_preview(
            run_id=run_id,
            baseline_run_id=baseline_run_id,
            opinion_leaf=opinion_leaf,
            profile_id=profile_id,
            top_k=top_k,
        )
    except NetworkExposureRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
