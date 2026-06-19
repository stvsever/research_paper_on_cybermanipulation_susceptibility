from __future__ import annotations

from fastapi import APIRouter, HTTPException

from profile_network_workbench_api.schemas import (
    BaselinePromptPreviewResponse,
    BaselineRunCreateRequest,
    BaselineRunCreateResponse,
    BaselineRunResponse,
)
from profile_network_workbench_api.services.baseline_runs import (
    BaselineRunConfigurationError,
    BaselineRunNotFoundError,
    get_baseline_prompt_preview,
    get_baseline_run,
    start_baseline_run,
)

router = APIRouter(prefix="/api", tags=["baseline-runs"])


@router.post("/baseline-runs", response_model=BaselineRunCreateResponse)
def create_baseline_run(request: BaselineRunCreateRequest) -> BaselineRunCreateResponse:
    try:
        return start_baseline_run(request)
    except BaselineRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/baseline-runs/{baseline_run_id}", response_model=BaselineRunResponse)
def read_baseline_run(baseline_run_id: str) -> BaselineRunResponse:
    try:
        return get_baseline_run(baseline_run_id)
    except BaselineRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Baseline run not found: {baseline_run_id}") from exc


@router.get("/baseline-prompt-preview", response_model=BaselinePromptPreviewResponse)
def read_baseline_prompt_preview(run_id: str, opinion_leaf: str, profile_id: str) -> BaselinePromptPreviewResponse:
    try:
        return get_baseline_prompt_preview(run_id=run_id, opinion_leaf=opinion_leaf, profile_id=profile_id)
    except BaselineRunConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
