from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from profile_network_workbench_api.routes import (
    baseline_runs,
    network_exposure_runs,
    pipeline_view,
    post_attack_network_exposure_runs,
    post_exposure_runs,
    profile_network,
)
from profile_network_workbench_api.settings import load_settings


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="Profile Network Experiment Workbench API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "lab_root": str(settings.lab_root),
            "evaluation_available": settings.evaluation_path.exists(),
            "runs_root": str(settings.runs_root),
        }

    app.include_router(profile_network.router)
    app.include_router(baseline_runs.router)
    app.include_router(network_exposure_runs.router)
    app.include_router(post_exposure_runs.router)
    app.include_router(post_attack_network_exposure_runs.router)
    app.include_router(pipeline_view.router)
    return app


app = create_app()
