from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from profile_network_workbench_api.backend_adapter import WorkbenchNotFoundError, attack_options
from profile_network_workbench_api.schemas import AttackOptionsResponse, ProfileNetworkResponse
from profile_network_workbench_api.services.profile_network import get_profile_network

router = APIRouter(prefix="/api", tags=["profile-network"])


@router.get("/profile-network", response_model=ProfileNetworkResponse)
def profile_network(
    run_id: str = "run_1",
    edge_limit_per_node: int = Query(default=6, ge=1, le=20),
) -> ProfileNetworkResponse:
    try:
        return get_profile_network(run_id=run_id, edge_limit_per_node=edge_limit_per_node)
    except WorkbenchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/attack-options", response_model=AttackOptionsResponse)
def read_attack_options(run_id: str = "run_1", opinion_leaf: str = "") -> AttackOptionsResponse:
    try:
        return attack_options(run_id=run_id, opinion_leaf=opinion_leaf)
    except WorkbenchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
