from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from profile_network_workbench_api.backend_adapter import WorkbenchNotFoundError
from profile_network_workbench_api.schemas import PipelineViewResponse
from profile_network_workbench_api.services.pipeline_view import get_pipeline_view

router = APIRouter(prefix="/api", tags=["pipeline-view"])


@router.get("/pipeline-view", response_model=PipelineViewResponse)
def pipeline_view(
    run_id: str = "run_1",
    edge_limit_per_node: int = Query(default=6, ge=1, le=20),
    opinion_leaf: str | None = None,
    attack_leaf: str | None = None,
) -> PipelineViewResponse:
    try:
        return get_pipeline_view(
            run_id=run_id,
            edge_limit_per_node=edge_limit_per_node,
            opinion_leaf=opinion_leaf,
            attack_leaf=attack_leaf,
        )
    except WorkbenchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
