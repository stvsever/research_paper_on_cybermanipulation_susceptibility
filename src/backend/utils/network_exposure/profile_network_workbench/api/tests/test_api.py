from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from profile_network_workbench_api.main import create_app
from profile_network_workbench_api.services import (
    baseline_runs,
    network_exposure_runs,
    pipeline_view,
    post_attack_network_exposure_runs,
    post_exposure_runs,
)
from src.backend.utils import live_artifacts
from src.backend.utils.network_exposure import load_exposure_network_package
from src.backend.utils.schemas import OpinionAssessment


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_stage(stage_outputs: Path, stage_id: str, stage_name: str, filename: str, rows: list[dict]) -> Path:
    stage_dir = stage_outputs / f"{stage_id}_{stage_name}"
    output_path = stage_dir / filename
    _write_jsonl(output_path, rows)
    _write_json(
        stage_dir / "manifest.json",
        {
            "stage_id": stage_id,
            "stage_name": stage_name,
            "created_at_utc": "2026-06-17T12:00:00+00:00",
            "primary_output_path": str(output_path),
            "output_files": [str(output_path)],
            "record_count": len(rows),
            "metadata": {},
        },
    )
    return output_path


def _pipeline_row(profile_id: str, scenario_index: int, opinion_leaf: str, attack_leaf: str) -> dict:
    return {
        "scenario_id": f"scenario_{scenario_index:04d}",
        "scenario_index": scenario_index,
        "random_seed": 120,
        "profile": {
            "profile_id": profile_id,
            "categorical_attributes": {"sex": "Other"},
            "continuous_attributes": {
                "chronological_age": 35 + scenario_index,
                "big_five_openness_mean_pct": 70 + scenario_index,
                "big_five_agreeableness_mean_pct": 60 + scenario_index,
            },
            "selected_leaf_nodes": [f"profile_leaf_{profile_id}"],
            "metadata": {"source": "test_stage01"},
        },
        "opinion_leaf": opinion_leaf,
        "attack_present": True,
        "attack_leaf": attack_leaf,
        "attack_primary_node": attack_leaf,
        "metadata": {},
    }


def _assignment_context(profile_id: str, position_id: str, assignment_rank: int) -> dict:
    package = load_exposure_network_package()
    metrics = dict(package.metrics_for_position(position_id).values)
    position_rows = {str(row["position_id"]): row for row in package.assignment_candidates()}
    metrics.update(position_rows[position_id])
    metrics.update(
        {
            "profile_id": profile_id,
            "position_id": position_id,
            "assignment_rank": assignment_rank,
            "graph_id": package.graph_id,
            "network_basis": "empirical_politisky24_bluesky_exposure",
            "edge_direction": package.manifest.get("edge_direction"),
            "edge_meaning": package.manifest.get("edge_meaning"),
        }
    )
    return metrics


def _with_exposure_assignments(rows: list[dict]) -> list[dict]:
    # Connected empirical positions from the stable graph package, so tests exercise real directed edges.
    assignments = {
        "profile_0001": _assignment_context("profile_0001", "550", 47),
        "profile_0002": _assignment_context("profile_0002", "169089", 57),
    }
    enriched: list[dict] = []
    for row in rows:
        next_row = json.loads(json.dumps(row))
        profile_id = next_row["profile"]["profile_id"]
        assignment = assignments[profile_id]
        next_row["profile"].setdefault("metadata", {})["exposure_network_assignment"] = assignment
        next_row.setdefault("metadata", {})["exposure_network_assignment"] = assignment
        enriched.append(next_row)
    return enriched


def _assessment(scenario_id: str, phase: str, opinion_leaf: str, score: int) -> dict:
    return {
        "scenario_id": scenario_id,
        "phase": phase,
        "opinion_leaf": opinion_leaf,
        "score": score,
        "confidence": 0.71,
        "reasoning": f"{phase} reasoning",
        "model_name": "mock/pipeline",
    }


def _create_pipeline_artifacts(tmp_path: Path) -> tuple[Path, str, str, str]:
    lab_root = tmp_path / "lab"
    run_id = "run_pipeline"
    run_root = lab_root / "evaluation" / run_id
    stage_outputs = run_root / "stage_outputs"
    opinion_leaf = "Opinion > Alliance_Commitment_Support"
    attack_leaf = "Attack > Headline_And_Lede_Misframing"
    stage01 = [
        _pipeline_row("profile_0001", 1, opinion_leaf, attack_leaf),
        _pipeline_row("profile_0002", 2, opinion_leaf, attack_leaf),
    ]
    stage01b = _with_exposure_assignments(stage01)
    _write_stage(stage_outputs, "01", "create_scenarios", "scenarios.jsonl", stage01)
    _write_stage(
        stage_outputs,
        "01b",
        "assign_exposure_network_positions",
        "scenarios_with_exposure_positions.jsonl",
        stage01b,
    )
    stage02 = []
    stage02b = []
    stage04 = []
    for row, baseline_score, network_score, post_score in zip(stage01b, [10, -20], [35, -5], [80, 25]):
        baseline = _assessment(row["scenario_id"], "baseline", opinion_leaf, baseline_score)
        stage02.append({**row, "baseline_assessment": baseline})
        stage02b.append(
            {
                **row,
                "baseline_assessment": baseline,
                "network_exposure_assessment": _assessment(
                    f"{row['scenario_id']}_network", "network_exposure_baseline", opinion_leaf, network_score
                ),
                "network_exposure_context": {
                    "max_exemplars": 8,
                    "target_baseline_score": baseline_score,
                    "peer_count": 1,
                    "peer_exemplars": [{"profile_id": "peer", "baseline_score": 12, "exposure_weight": 0.7}],
                    "peer_assessments": [{"profile_id": "peer", "baseline_score": 12, "exposure_weight": 0.7}],
                },
            }
        )
        stage04.append(
            {
                **row,
                "baseline_assessment": baseline,
                "attack_vector_spec": {"adversarial_direction": 1},
                "post_attack_assessment": _assessment(f"{row['scenario_id']}_post", "post_attack", opinion_leaf, post_score),
                "post_heuristic_checks": {"valid": True},
            }
        )
    _write_stage(stage_outputs, "02", "assess_baseline_opinions", "scenarios_with_baseline.jsonl", stage02)
    _write_stage(stage_outputs, "02b", "assess_network_exposure_opinions", "scenarios_with_network_exposure.jsonl", stage02b)
    _write_stage(stage_outputs, "04", "assess_post_attack_opinions", "scenarios_with_post.jsonl", stage04)
    _write_json(
        run_root / "provenance" / "run_manifest.json",
        {
            "stage_outputs_root": str(stage_outputs),
            "pipeline_config": {"use_test_ontology": True, "openrouter_model": "mock/pipeline"},
        },
    )
    return lab_root, run_id, opinion_leaf, attack_leaf


def _create_stage01_with_live_stage02(tmp_path: Path) -> tuple[Path, str, str, str]:
    lab_root = tmp_path / "lab"
    run_id = "run_pipeline_live"
    run_root = lab_root / "evaluation" / run_id
    stage_outputs = run_root / "stage_outputs"
    opinion_leaf = "Opinion > Alliance_Commitment_Support"
    attack_leaf = "Attack > Headline_And_Lede_Misframing"
    stage01 = [
        _pipeline_row("profile_0001", 1, opinion_leaf, attack_leaf),
        _pipeline_row("profile_0002", 2, opinion_leaf, attack_leaf),
    ]
    stage01b = _with_exposure_assignments(stage01)
    _write_stage(stage_outputs, "01", "create_scenarios", "scenarios.jsonl", stage01)
    _write_stage(
        stage_outputs,
        "01b",
        "assign_exposure_network_positions",
        "scenarios_with_exposure_positions.jsonl",
        stage01b,
    )

    live_dir = stage_outputs / "02_assess_baseline_opinions"
    live_artifacts.init_live_stage(
        live_dir,
        run_id=run_id,
        stage_id="02",
        stage_name="assess_baseline_opinions",
        phase="baseline",
        total_count=2,
    )
    live_artifacts.append_live_result(
        live_dir,
        {**stage01b[0], "baseline_assessment": _assessment(stage01b[0]["scenario_id"], "baseline", opinion_leaf, 123)},
    )
    # Simulate a half-written line from a concurrently observed file; the reader should ignore it.
    with (live_dir / "live_results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{partial\n")
    live_artifacts.update_live_status(live_dir, completed_count=1, failed_count=0, status="running")

    _write_json(
        run_root / "provenance" / "run_manifest.json",
        {
            "stage_outputs_root": str(stage_outputs),
            "pipeline_config": {"use_test_ontology": True, "openrouter_model": "mock/pipeline"},
        },
    )
    return lab_root, run_id, opinion_leaf, attack_leaf


def client() -> TestClient:
    return TestClient(create_app())


def _poll_baseline(test_client: TestClient, baseline_run_id: str) -> dict:
    payload = {}
    for _ in range(50):
        payload = test_client.get(f"/api/baseline-runs/{baseline_run_id}").json()
        if payload["status"] in {"completed", "failed", "completed_with_errors"}:
            break
        time.sleep(0.05)
    return payload


def _poll_post(test_client: TestClient, post_run_id: str) -> dict:
    payload = {}
    for _ in range(50):
        payload = test_client.get(f"/api/post-exposure-runs/{post_run_id}").json()
        if payload["status"] in {"completed", "failed", "completed_with_errors"}:
            break
        time.sleep(0.05)
    return payload


def _poll_network(test_client: TestClient, network_run_id: str) -> dict:
    payload = {}
    for _ in range(50):
        payload = test_client.get(f"/api/network-exposure-runs/{network_run_id}").json()
        if payload["status"] in {"completed", "failed", "completed_with_errors"}:
            break
        time.sleep(0.05)
    return payload


def _poll_post_network(test_client: TestClient, post_network_run_id: str) -> dict:
    payload = {}
    for _ in range(50):
        payload = test_client.get(f"/api/post-attack-network-exposure-runs/{post_network_run_id}").json()
        if payload["status"] in {"completed", "failed", "completed_with_errors"}:
            break
        time.sleep(0.05)
    return payload


def _install_fake_baseline(monkeypatch) -> None:
    class FakeBaselineAgent:
        def assess(self, run_id, call_id, scenario_id, opinion_leaf, profile, review_feedback=None):
            return OpinionAssessment(
                scenario_id=scenario_id,
                phase="baseline",
                opinion_leaf=opinion_leaf,
                score=137,
                confidence=0.81,
                reasoning=f"Mocked baseline for {profile.profile_id}",
                model_name="mock/model",
            )

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def baseline_opinion_agent(self) -> FakeBaselineAgent:
            return FakeBaselineAgent()

    monkeypatch.setattr(baseline_runs, "AGENT_FACTORY_CLASS", FakeFactory)


def _create_mock_baseline_run(monkeypatch, tmp_path, profile_count: int = 1) -> tuple[TestClient, dict, str, str, list[str]]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PROFILE_NETWORK_RUNS_ROOT", str(tmp_path))
    _install_fake_baseline(monkeypatch)
    test_client = client()
    network = test_client.get("/api/profile-network", params={"run_id": "run_1"}).json()
    opinion_leaf = network["opinion_leaves"][0]["path"]
    selected_profile_ids = [node["id"] for node in network["nodes"][:profile_count]]
    response = test_client.post(
        "/api/baseline-runs",
        json={
            "run_id": "run_1",
            "opinion_leaf": opinion_leaf,
            "profile_ids": selected_profile_ids,
            "model_name": "mock/model",
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    baseline_run_id = response.json()["baseline_run_id"]
    payload = _poll_baseline(test_client, baseline_run_id)
    assert payload["status"] == "completed"
    return test_client, network, baseline_run_id, opinion_leaf, selected_profile_ids


def _install_fake_post(monkeypatch) -> None:
    class FakePostAgent:
        def assess(
            self,
            run_id,
            call_id,
            scenario_id,
            opinion_leaf,
            profile,
            baseline_score,
            attack_present,
            adversarial_direction,
            attack_leaf,
            attack_vector_spec,
        ):
            shift = 75 if adversarial_direction >= 0 else -75
            return OpinionAssessment(
                scenario_id=scenario_id,
                phase="post_attack",
                opinion_leaf=opinion_leaf,
                score=int(baseline_score + shift),
                confidence=0.77,
                reasoning=f"Mocked post exposure for {profile.profile_id}",
                model_name="mock/model",
            )

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_attack_opinion_agent(self) -> FakePostAgent:
            return FakePostAgent()

    monkeypatch.setattr(post_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)


def _create_mock_post_run(monkeypatch, tmp_path, profile_count: int = 3) -> tuple[TestClient, dict, str, str, str, str, list[str], dict]:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(
        monkeypatch,
        tmp_path,
        profile_count=profile_count,
    )
    attack_leaf = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": opinion_leaf},
    ).json()["attack_options"][0]["path"]
    _install_fake_post(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = test_client.post(
        "/api/post-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    post_run_id = response.json()["post_run_id"]
    post_payload = _poll_post(test_client, post_run_id)
    assert post_payload["status"] == "completed"
    return test_client, network, baseline_run_id, post_run_id, opinion_leaf, attack_leaf, profile_ids, post_payload


def test_profile_network_is_deterministic_for_run_1() -> None:
    first = client().get("/api/profile-network", params={"run_id": "run_1"}).json()
    second = client().get("/api/profile-network", params={"run_id": "run_1"}).json()
    assert first == second
    assert first["diagnostics"]["profile_count"] == 60
    assert len(first["layout_affinities"]) == first["diagnostics"]["full_pair_count"]
    assert first["diagnostics"]["full_pair_count"] == 1770
    assert all(node["cluster_id"].startswith("cluster_") for node in first["nodes"])
    assert first["provenance"]["source"] == "stage01_reconstructed_profiles"
    weights = first["affinity_formula"]["default_weights"]
    assert round(sum(weights.values()), 6) == 1.0
    assert weights["categorical_similarity"] < 0.10


def test_profile_network_affinities_are_bounded() -> None:
    response = client().get("/api/profile-network", params={"run_id": "run_1", "edge_limit_per_node": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["edges"]
    for edge in payload["edges"]:
        assert 0.0 <= edge["affinity"] <= 1.0
        assert 0.0 <= edge["normalized_affinity"] <= 1.0
        for value in edge["components"].values():
            assert 0.0 <= value <= 1.0
    for item in payload["layout_affinities"]:
        assert 0.0 <= item["affinity"] <= 1.0
        assert set(item["components"]) == {
            "categorical_similarity",
            "personality_similarity",
            "age_context_similarity",
            "ontology_leaf_overlap",
        }
        for value in item["components"].values():
            assert 0.0 <= value <= 1.0


def test_pipeline_view_requires_stage01b(monkeypatch, tmp_path) -> None:
    lab_root = tmp_path / "lab"
    (lab_root / "evaluation" / "run_empty").mkdir(parents=True)
    monkeypatch.setenv("PROFILE_NETWORK_LAB_ROOT", str(lab_root))

    response = client().get("/api/pipeline-view", params={"run_id": "run_empty"})

    assert response.status_code == 404
    assert "Stage 01b exposure-network assignment output is required" in response.json()["detail"]


def test_pipeline_view_reads_canonical_stage_artifacts(monkeypatch, tmp_path) -> None:
    lab_root, run_id, opinion_leaf, attack_leaf = _create_pipeline_artifacts(tmp_path)
    monkeypatch.setenv("PROFILE_NETWORK_LAB_ROOT", str(lab_root))

    response = client().get(
        "/api/pipeline-view",
        params={"run_id": run_id, "opinion_leaf": opinion_leaf, "attack_leaf": attack_leaf},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["network"]["provenance"]["source"] == "stage01b_exposure_network_assignments"
    assert payload["network"]["provenance"]["network_basis"] == "politisky24_bluesky_v1_empirical_exposure"
    assert payload["network"]["diagnostics"]["profile_count"] == 2
    assert payload["network"]["diagnostics"]["edge_semantics"] == "visible_peer_to_exposed_receiver"
    assert payload["network"]["edges"][0]["edge_kind"] == "empirical_exposure"
    assert payload["network"]["edges"][0]["directed"] is True
    assert "source_position_id" in payload["network"]["edges"][0]
    assert "target_position_id" in payload["network"]["edges"][0]
    assert "exposure_weight" in payload["network"]["edges"][0]
    assert payload["attack_options"][0]["path"] == attack_leaf
    assert payload["baseline_run"]["status"] == "completed"
    assert payload["baseline_run"]["completed_count"] == 2
    assert payload["baseline_run"]["results"][0]["score"] == 10
    assert payload["network_run"]["results"][0]["network_score"] == 35
    assert payload["post_run"]["attack_leaf"] == attack_leaf
    assert payload["post_run"]["results"][0]["post_score"] == 80
    assert payload["post_network_run"] is None
    assert any(stage["stage_id"] == "04b" and not stage["available"] for stage in payload["stage_status"])
    assert any("Stage 04b" in warning for warning in payload["warnings"])


def test_live_artifacts_write_status_results_and_errors(tmp_path) -> None:
    output_dir = tmp_path / "stage"
    live_artifacts.init_live_stage(
        output_dir,
        run_id="run_live",
        stage_id="02",
        stage_name="assess_baseline_opinions",
        phase="baseline",
        total_count=2,
    )
    live_artifacts.append_live_result(output_dir, {"scenario_id": "scenario_0001", "score": 10})
    live_artifacts.append_live_error(output_dir, {"scenario_id": "scenario_0002", "message": "mock failure"})
    live_artifacts.update_live_status(output_dir, completed_count=1, failed_count=1, status="completed_with_errors")

    status = json.loads((output_dir / "live_status.json").read_text(encoding="utf-8"))
    result_rows = _read_jsonl(output_dir / "live_results.jsonl")
    error_rows = _read_jsonl(output_dir / "live_errors.jsonl")

    assert status["run_id"] == "run_live"
    assert status["status"] == "completed_with_errors"
    assert status["completed_count"] == 1
    assert status["failed_count"] == 1
    assert status["source"] == "live_sidecar"
    assert result_rows == [{"scenario_id": "scenario_0001", "score": 10}]
    assert error_rows[0]["message"] == "mock failure"


def test_pipeline_view_reads_live_stage02_sidecar_when_canonical_missing(monkeypatch, tmp_path) -> None:
    lab_root, run_id, opinion_leaf, attack_leaf = _create_stage01_with_live_stage02(tmp_path)
    monkeypatch.setenv("PROFILE_NETWORK_LAB_ROOT", str(lab_root))

    response = client().get(
        "/api/pipeline-view",
        params={"run_id": run_id, "opinion_leaf": opinion_leaf, "attack_leaf": attack_leaf},
    )

    assert response.status_code == 200
    payload = response.json()
    stage02 = next(stage for stage in payload["stage_status"] if stage["stage_id"] == "02")
    assert stage02["available"] is False
    assert stage02["source"] == "live_sidecar"
    assert stage02["live_available"] is True
    assert stage02["live_status"] == "running"
    assert stage02["live_result_count"] == 1
    assert payload["network"]["diagnostics"]["profile_count"] == 2
    assert payload["baseline_run"]["status"] == "running"
    assert payload["baseline_run"]["completed_count"] == 1
    assert payload["baseline_run"]["results"][0]["profile_id"] == "profile_0001"
    assert payload["baseline_run"]["results"][0]["score"] == 123
    assert any("Stage 02 is using live sidecar rows" in warning for warning in payload["warnings"])


def test_pipeline_view_prefers_canonical_stage_over_live_sidecar(monkeypatch, tmp_path) -> None:
    lab_root, run_id, opinion_leaf, attack_leaf = _create_pipeline_artifacts(tmp_path)
    monkeypatch.setenv("PROFILE_NETWORK_LAB_ROOT", str(lab_root))
    stage_outputs = lab_root / "evaluation" / run_id / "stage_outputs"
    stage02_rows = _read_jsonl(stage_outputs / "02_assess_baseline_opinions" / "scenarios_with_baseline.jsonl")
    live_dir = stage_outputs / "02_assess_baseline_opinions"
    live_artifacts.init_live_stage(
        live_dir,
        run_id=run_id,
        stage_id="02",
        stage_name="assess_baseline_opinions",
        phase="baseline",
        total_count=2,
    )
    live_artifacts.append_live_result(
        live_dir,
        {
            **stage02_rows[0],
            "baseline_assessment": _assessment(stage02_rows[0]["scenario_id"], "baseline", opinion_leaf, 999),
        },
    )
    live_artifacts.update_live_status(live_dir, completed_count=1, failed_count=0, status="running")

    response = client().get(
        "/api/pipeline-view",
        params={"run_id": run_id, "opinion_leaf": opinion_leaf, "attack_leaf": attack_leaf},
    )

    assert response.status_code == 200
    payload = response.json()
    stage02 = next(stage for stage in payload["stage_status"] if stage["stage_id"] == "02")
    assert stage02["available"] is True
    assert stage02["source"] == "canonical"
    assert stage02["live_available"] is True
    assert payload["baseline_run"]["status"] == "completed"
    assert payload["baseline_run"]["results"][0]["score"] == 10


def test_pipeline_view_prefers_ingested_db_projection(monkeypatch, tmp_path) -> None:
    lab_root, run_id, opinion_leaf, attack_leaf = _create_pipeline_artifacts(tmp_path)
    monkeypatch.setenv("PROFILE_NETWORK_LAB_ROOT", str(lab_root))
    run_root = lab_root / "evaluation" / run_id
    stage_outputs = run_root / "stage_outputs"

    stage01b = _read_jsonl(stage_outputs / "01b_assign_exposure_network_positions" / "scenarios_with_exposure_positions.jsonl")
    stage02 = _read_jsonl(stage_outputs / "02_assess_baseline_opinions" / "scenarios_with_baseline.jsonl")
    stage02[0]["baseline_assessment"]["score"] = 444

    def fake_db_loader(requested_run_id: str):
        assert requested_run_id == run_id
        return SimpleNamespace(
            run_id=run_id,
            run_root=run_root,
            stage_outputs_root=stage_outputs,
            warnings=["Loaded pipeline_view from Postgres artifact projection."],
            statuses=[
                {
                    "stage_id": "01b",
                    "stage_name": "assign_exposure_network_positions",
                    "available": True,
                    "manifest_path": str(stage_outputs / "01b_assign_exposure_network_positions" / "manifest.json"),
                    "primary_output_path": str(
                        stage_outputs / "01b_assign_exposure_network_positions" / "scenarios_with_exposure_positions.jsonl"
                    ),
                    "record_count": 2,
                    "created_at_utc": "2026-06-17T12:00:00+00:00",
                },
                {
                    "stage_id": "02",
                    "stage_name": "assess_baseline_opinions",
                    "available": True,
                    "manifest_path": str(stage_outputs / "02_assess_baseline_opinions" / "manifest.json"),
                    "primary_output_path": str(stage_outputs / "02_assess_baseline_opinions" / "scenarios_with_baseline.jsonl"),
                    "record_count": 2,
                    "created_at_utc": "2026-06-17T12:00:00+00:00",
                },
            ],
            rows_by_stage={"01b": stage01b, "02": stage02},
        )

    monkeypatch.setattr(pipeline_view, "_load_db_pipeline_view_data", fake_db_loader)

    response = client().get(
        "/api/pipeline-view",
        params={"run_id": run_id, "opinion_leaf": opinion_leaf, "attack_leaf": attack_leaf},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_run"]["results"][0]["score"] == 444
    assert any("Postgres" in warning for warning in payload["warnings"])


def test_baseline_run_requires_openrouter_key(monkeypatch) -> None:
    network = client().get("/api/profile-network", params={"run_id": "run_1"}).json()
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    response = client().post(
        "/api/baseline-runs",
        json={
            "run_id": "run_1",
            "opinion_leaf": network["opinion_leaves"][0]["path"],
            "profile_ids": [network["nodes"][0]["id"]],
        },
    )
    assert response.status_code == 400
    assert "OPENROUTER_API_KEY" in response.json()["detail"]


def test_baseline_prompt_preview_uses_real_prompt_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    test_client = client()
    network = test_client.get("/api/profile-network", params={"run_id": "run_1"}).json()
    profile_id = network["nodes"][0]["id"]
    opinion_leaf = network["opinion_leaves"][0]["path"]
    response = test_client.get(
        "/api/baseline-prompt-preview",
        params={"run_id": "run_1", "opinion_leaf": opinion_leaf, "profile_id": profile_id},
    )
    assert response.status_code == 200
    payload = response.json()
    prompt_path = Path(__file__).parents[3] / "src" / "backend" / "agentic_framework" / "prompts" / "baseline_opinion.md"
    assert payload["prompt_name"] == "baseline_opinion.md"
    assert payload["system_prompt"] == prompt_path.read_text(encoding="utf-8")
    assert payload["messages"][0] == {"role": "system", "content": payload["system_prompt"]}
    assert payload["messages"][1]["role"] == "user"
    assert payload["user_payload"]["scenario_id"] == f"preview_run_1_{profile_id}"
    assert payload["user_payload"]["opinion_leaf"] == opinion_leaf
    assert payload["user_payload"]["profile"]["profile_id"] == profile_id
    assert '"profile_id":' in payload["messages"][1]["content"]


def test_baseline_prompt_preview_rejects_unknown_profile() -> None:
    network = client().get("/api/profile-network", params={"run_id": "run_1"}).json()
    response = client().get(
        "/api/baseline-prompt-preview",
        params={
            "run_id": "run_1",
            "opinion_leaf": network["opinion_leaves"][0]["path"],
            "profile_id": "profile_missing",
        },
    )
    assert response.status_code == 400
    assert "Unknown profile id" in response.json()["detail"]


def test_attack_options_for_run_1_include_configured_attacks() -> None:
    test_client = client()
    network = test_client.get("/api/profile-network", params={"run_id": "run_1"}).json()
    response = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": network["opinion_leaves"][0]["path"]},
    )
    assert response.status_code == 200
    payload = response.json()
    paths = {item["path"] for item in payload["attack_options"]}
    assert len(payload["attack_options"]) == 4
    assert any(path.endswith("Headline_And_Lede_Misframing") for path in paths)
    assert any(path.endswith("Personal_Safety_Fear_Appeal") for path in paths)
    assert all("complexity_tier" in item for item in payload["attack_options"])


def test_post_exposure_requires_completed_baseline(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    test_client = client()
    network = test_client.get("/api/profile-network", params={"run_id": "run_1"}).json()
    attack_leaf = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": network["opinion_leaves"][0]["path"]},
    ).json()["attack_options"][0]["path"]
    response = test_client.post(
        "/api/post-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": "missing-baseline",
            "opinion_leaf": network["opinion_leaves"][0]["path"],
            "attack_leaf": attack_leaf,
            "profile_ids": [network["nodes"][0]["id"]],
            "model_name": "mock/model",
        },
    )
    assert response.status_code == 400
    assert "Baseline run not found" in response.json()["detail"]


def test_network_exposure_requires_completed_baseline(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    test_client = client()
    network = test_client.get("/api/profile-network", params={"run_id": "run_1"}).json()
    response = test_client.post(
        "/api/network-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": "missing-baseline",
            "opinion_leaf": network["opinion_leaves"][0]["path"],
            "profile_ids": [network["nodes"][0]["id"]],
            "model_name": "mock/model",
        },
    )
    assert response.status_code == 400
    assert "Baseline run not found" in response.json()["detail"]


def test_network_prompt_preview_uses_real_prompt_without_api_key(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path, profile_count=3)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    response = test_client.get(
        "/api/network-exposure-prompt-preview",
        params={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "profile_id": profile_ids[0],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    prompt_path = Path(__file__).parents[3] / "src" / "backend" / "agentic_framework" / "prompts" / "network_exposure_opinion.md"
    assert payload["prompt_name"] == "network_exposure_opinion.md"
    assert payload["system_prompt"] == prompt_path.read_text(encoding="utf-8")
    assert payload["user_payload"]["baseline_score"] == 137
    assert payload["user_payload"]["network_context"]["target_baseline_score"] == 137
    assert payload["user_payload"]["network_context"]["peer_assessments"]
    assert payload["user_payload"]["network_context"]["peer_assessments"][0]["profile_id"] != profile_ids[0]
    assert len(payload["user_payload"]["network_context"]["peer_assessments"]) <= 8
    assert payload["messages"][1]["role"] == "user"


def test_mocked_network_exposure_run_persists_results(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path, profile_count=3)

    class FakeNetworkAgent:
        def assess(self, run_id, call_id, scenario_id, opinion_leaf, profile, baseline_score, network_context, review_feedback=None):
            return OpinionAssessment(
                scenario_id=scenario_id,
                phase="network_exposure_baseline",
                opinion_leaf=opinion_leaf,
                score=baseline_score + 23,
                confidence=0.77,
                reasoning=f"Mocked network exposure for {profile.profile_id}",
                model_name="mock/model",
            )

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def network_exposure_opinion_agent(self) -> FakeNetworkAgent:
            return FakeNetworkAgent()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(network_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)
    response = test_client.post(
        "/api/network-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
            "top_k": 2,
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    payload = _poll_network(test_client, response.json()["network_run_id"])
    assert payload["status"] == "completed"
    assert payload["completed_count"] == 3
    assert payload["failed_count"] == 0
    assert {item["profile_id"] for item in payload["results"]} == set(profile_ids)
    first = payload["results"][0]
    assert first["baseline_score"] == 137
    assert first["network_score"] == 160
    assert first["delta_score"] == 23
    assert first["network_context"]["peer_assessments"]
    assert len(first["network_context"]["peer_assessments"]) <= 2
    assert (tmp_path / response.json()["network_run_id"] / "network_contexts.jsonl").exists()


def test_network_exposure_failed_agent_records_errors_without_fallback(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path, profile_count=1)

    class FailingNetworkAgent:
        def assess(self, *args, **kwargs):
            raise RuntimeError("mock network failure")

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def network_exposure_opinion_agent(self) -> FailingNetworkAgent:
            return FailingNetworkAgent()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(network_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)
    response = test_client.post(
        "/api/network-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
        },
    )
    assert response.status_code == 200
    payload = _poll_network(test_client, response.json()["network_run_id"])
    assert payload["status"] == "failed"
    assert payload["completed_count"] == 0
    assert payload["failed_count"] == 1
    assert payload["results"] == []
    assert "mock network failure" in payload["errors"][0]["message"]


def test_post_prompt_preview_uses_real_prompt_without_api_key(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    attack_leaf = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": opinion_leaf},
    ).json()["attack_options"][0]["path"]
    response = test_client.get(
        "/api/post-exposure-prompt-preview",
        params={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_id": profile_ids[0],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    prompt_path = Path(__file__).parents[3] / "src" / "backend" / "agentic_framework" / "prompts" / "post_attack_opinion.md"
    assert payload["prompt_name"] == "post_attack_opinion.md"
    assert payload["system_prompt"] == prompt_path.read_text(encoding="utf-8")
    assert payload["user_payload"]["baseline_score"] == 137
    assert payload["user_payload"]["attack_present"] is True
    assert payload["user_payload"]["attack_leaf"] == attack_leaf
    assert payload["user_payload"]["attack_vector_spec"]["spec_source"] == "deterministic_ontology_v1"
    assert payload["messages"][1]["role"] == "user"


def test_mocked_post_exposure_run_persists_results(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path, profile_count=2)
    attack_leaf = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": opinion_leaf},
    ).json()["attack_options"][0]["path"]

    class FakePostAgent:
        def assess(
            self,
            run_id,
            call_id,
            scenario_id,
            opinion_leaf,
            profile,
            baseline_score,
            attack_present,
            adversarial_direction,
            attack_leaf,
            attack_vector_spec,
        ):
            return OpinionAssessment(
                scenario_id=scenario_id,
                phase="post_attack",
                opinion_leaf=opinion_leaf,
                score=int(baseline_score + (75 if adversarial_direction >= 0 else -75)),
                confidence=0.77,
                reasoning=f"Mocked post exposure for {profile.profile_id}",
                model_name="mock/model",
            )

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_attack_opinion_agent(self) -> FakePostAgent:
            return FakePostAgent()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(post_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)
    response = test_client.post(
        "/api/post-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    post_run_id = response.json()["post_run_id"]
    payload = _poll_post(test_client, post_run_id)
    assert payload["status"] == "completed"
    assert payload["completed_count"] == 2
    assert payload["failed_count"] == 0
    assert {item["profile_id"] for item in payload["results"]} == set(profile_ids)
    assert all(item["baseline_score"] == 137 for item in payload["results"])
    assert all(item["attack_leaf"] == attack_leaf for item in payload["results"])
    assert (tmp_path / post_run_id / "attack_specs.jsonl").exists()


def test_post_directional_guard_clamps_counter_goal_shift(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path)
    attack_leaf = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": opinion_leaf},
    ).json()["attack_options"][0]["path"]

    class CounterGoalAgent:
        def assess(
            self,
            run_id,
            call_id,
            scenario_id,
            opinion_leaf,
            profile,
            baseline_score,
            attack_present,
            adversarial_direction,
            attack_leaf,
            attack_vector_spec,
        ):
            counter_goal_score = baseline_score - 100 if adversarial_direction > 0 else baseline_score + 100
            return OpinionAssessment(
                scenario_id=scenario_id,
                phase="post_attack",
                opinion_leaf=opinion_leaf,
                score=int(counter_goal_score),
                confidence=0.65,
                reasoning="Counter-goal mock movement.",
                model_name="mock/model",
            )

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_attack_opinion_agent(self) -> CounterGoalAgent:
            return CounterGoalAgent()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(post_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)
    response = test_client.post(
        "/api/post-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
        },
    )
    assert response.status_code == 200
    payload = _poll_post(test_client, response.json()["post_run_id"])
    assert payload["status"] == "completed"
    result = payload["results"][0]
    if result["adversarial_direction"] == 0:
        assert result["post_score"] in {17, 257}
        assert abs(result["delta_score"]) == 120
    else:
        assert result["post_score"] == 137
        assert result["delta_score"] == 0
    assert result["post_direction_clamped"] is True


def test_post_network_requires_completed_post_run(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, opinion_leaf, profile_ids = _create_mock_baseline_run(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    attack_leaf = test_client.get(
        "/api/attack-options",
        params={"run_id": "run_1", "opinion_leaf": opinion_leaf},
    ).json()["attack_options"][0]["path"]
    response = test_client.post(
        "/api/post-attack-network-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "post_run_id": "missing-post",
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
        },
    )
    assert response.status_code == 400
    assert "Post-exposure run not found" in response.json()["detail"]


def test_post_network_prompt_preview_uses_real_prompt_without_api_key(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, post_run_id, opinion_leaf, attack_leaf, profile_ids, post_payload = _create_mock_post_run(
        monkeypatch,
        tmp_path,
        profile_count=3,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    private_post_by_profile = {item["profile_id"]: item for item in post_payload["results"]}
    response = test_client.get(
        "/api/post-attack-network-exposure-prompt-preview",
        params={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "post_run_id": post_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_id": profile_ids[0],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    prompt_path = Path(__file__).parents[3] / "src" / "backend" / "agentic_framework" / "prompts" / "post_attack_network_exposure_opinion.md"
    assert payload["prompt_name"] == "post_attack_network_exposure_opinion.md"
    assert payload["system_prompt"] == prompt_path.read_text(encoding="utf-8")
    assert payload["user_payload"]["baseline_score"] == 137
    assert payload["user_payload"]["private_post_score"] == private_post_by_profile[profile_ids[0]]["post_score"]
    assert payload["user_payload"]["attack_present"] is True
    assert payload["user_payload"]["attack_leaf"] == attack_leaf
    assert payload["user_payload"]["attack_vector_spec"]["spec_source"] == "deterministic_ontology_v1"
    context = payload["user_payload"]["post_attack_network_context"]
    assert context["attack_leaf"] == attack_leaf
    assert context["peer_assessments"]
    assert all(peer["profile_id"] != profile_ids[0] for peer in context["peer_assessments"])
    assert all("post_score" in peer for peer in context["peer_assessments"])
    assert payload["messages"][1]["role"] == "user"


def test_mocked_post_network_run_persists_results(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, post_run_id, opinion_leaf, attack_leaf, profile_ids, post_payload = _create_mock_post_run(
        monkeypatch,
        tmp_path,
        profile_count=3,
    )
    private_post_by_profile = {item["profile_id"]: item for item in post_payload["results"]}

    class FakePostNetworkAgent:
        def assess(
            self,
            run_id,
            call_id,
            scenario_id,
            opinion_leaf,
            profile,
            baseline_score,
            private_post_score,
            attack_present,
            adversarial_direction,
            attack_leaf,
            attack_vector_spec,
            post_attack_network_context,
            review_feedback=None,
        ):
            return OpinionAssessment(
                scenario_id=scenario_id,
                phase="post_attack_network_exposure",
                opinion_leaf=opinion_leaf,
                score=int(private_post_score - 31),
                confidence=0.73,
                reasoning=f"Mocked post-network exposure for {profile.profile_id}",
                model_name="mock/model",
            )

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_attack_network_exposure_opinion_agent(self) -> FakePostNetworkAgent:
            return FakePostNetworkAgent()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(post_attack_network_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)
    response = test_client.post(
        "/api/post-attack-network-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "post_run_id": post_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
            "top_k": 2,
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    payload = _poll_post_network(test_client, response.json()["post_network_run_id"])
    assert payload["status"] == "completed"
    assert payload["completed_count"] == 3
    assert payload["failed_count"] == 0
    assert {item["profile_id"] for item in payload["results"]} == set(profile_ids)
    first = payload["results"][0]
    private_post = private_post_by_profile[first["profile_id"]]["post_score"]
    assert "Issue_Position_Taxonomy" not in first["scenario_id"]
    assert "Political_Opinion_Cybermanipulation_Ontology" not in first["scenario_id"]
    assert len(first["scenario_id"]) < 80
    assert first["baseline_score"] == 137
    assert first["private_post_score"] == private_post
    assert first["post_attack_network_score"] == private_post - 31
    assert first["increment_from_private_post"] == -31
    assert first["delta_from_baseline"] == first["post_attack_network_score"] - 137
    assert first["post_attack_network_context"]["peer_assessments"]
    assert all(peer["profile_id"] != first["profile_id"] for peer in first["post_attack_network_context"]["peer_assessments"])
    assert len(first["post_attack_network_context"]["peer_assessments"]) <= 2
    assert (tmp_path / response.json()["post_network_run_id"] / "post_attack_network_contexts.jsonl").exists()


def test_post_network_failed_agent_records_errors_without_fallback(monkeypatch, tmp_path) -> None:
    test_client, network, baseline_run_id, post_run_id, opinion_leaf, attack_leaf, profile_ids, post_payload = _create_mock_post_run(
        monkeypatch,
        tmp_path,
        profile_count=2,
    )

    class FailingPostNetworkAgent:
        def assess(self, *args, **kwargs):
            raise RuntimeError("mock post-network failure")

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_attack_network_exposure_opinion_agent(self) -> FailingPostNetworkAgent:
            return FailingPostNetworkAgent()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(post_attack_network_exposure_runs, "AGENT_FACTORY_CLASS", FakeFactory)
    response = test_client.post(
        "/api/post-attack-network-exposure-runs",
        json={
            "run_id": "run_1",
            "baseline_run_id": baseline_run_id,
            "post_run_id": post_run_id,
            "opinion_leaf": opinion_leaf,
            "attack_leaf": attack_leaf,
            "profile_ids": profile_ids,
            "model_name": "mock/model",
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    payload = _poll_post_network(test_client, response.json()["post_network_run_id"])
    assert payload["status"] == "failed"
    assert payload["completed_count"] == 0
    assert payload["failed_count"] == 2
    assert payload["results"] == []
    assert "mock post-network failure" in payload["errors"][0]["message"]


def test_mocked_baseline_run_persists_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("PROFILE_NETWORK_RUNS_ROOT", str(tmp_path))
    _install_fake_baseline(monkeypatch)
    test_client = client()
    network = test_client.get("/api/profile-network", params={"run_id": "run_1"}).json()
    selected_profile_ids = [network["nodes"][0]["id"], network["nodes"][1]["id"]]
    response = test_client.post(
        "/api/baseline-runs",
        json={
            "run_id": "run_1",
            "opinion_leaf": network["opinion_leaves"][0]["path"],
            "profile_ids": selected_profile_ids,
            "model_name": "mock/model",
            "max_concurrency": 2,
        },
    )
    assert response.status_code == 200
    baseline_run_id = response.json()["baseline_run_id"]

    payload = _poll_baseline(test_client, baseline_run_id)

    assert payload["status"] == "completed"
    assert payload["completed_count"] == 2
    assert payload["failed_count"] == 0
    assert {item["profile_id"] for item in payload["results"]} == set(selected_profile_ids)
    assert (tmp_path / baseline_run_id / "results.jsonl").exists()
