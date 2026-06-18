from __future__ import annotations

"""
Technical overview
------------------
Stage 01b assigns every generated profile to one empirical PolitiSky24
exposure-network position. It does not change the scenario design created by
Stage 01; it adds stable profile-level network-position metadata that is reused
across all opinion leaves, attack leaves, and control rows.

This stage is the handoff between the factorial scenario panel and the empirical
directed exposure graph. Later network-exposure stages use the assigned
positions to resolve who can see whose outputs through directed edges:

    visible peer profile -> exposed target profile

The assignment is deterministic for a fixed run seed and deliberately
independent of profile traits, opinions, and attacks. This keeps individual
susceptibility and empirical network position analytically separable.
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.io import abs_path, read_jsonl, stage_manifest_path, write_json, write_jsonl
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.network_exposure import (
    assign_profiles_to_positions,
    build_assignment_summary,
    load_exposure_network_package,
)
from src.backend.utils.network_exposure.defaults import DEFAULT_ASSIGNMENT_MODE, DEFAULT_GRAPH_ID
from src.backend.utils.schemas import StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


class Stage01bConfig(StageConfig):
    exposure_network_root: Optional[str] = None
    exposure_graph_id: str = DEFAULT_GRAPH_ID
    exposure_assignment_mode: str = DEFAULT_ASSIGNMENT_MODE


def _profile_id(row: dict[str, Any]) -> str:
    profile = row.get("profile")
    if not isinstance(profile, dict) or not profile.get("profile_id"):
        raise ValueError(f"Scenario row is missing profile.profile_id: {row.get('scenario_id')}")
    return str(profile["profile_id"])


def _unique_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile = row.get("profile")
        if not isinstance(profile, dict) or not profile.get("profile_id"):
            raise ValueError(f"Scenario row is missing a valid profile: {row.get('scenario_id')}")
        profile_id = str(profile["profile_id"])
        existing = profiles_by_id.get(profile_id)
        if existing is not None and existing != profile:
            raise ValueError(f"Profile {profile_id} has inconsistent payloads across scenario rows")
        profiles_by_id[profile_id] = profile
    return [profiles_by_id[profile_id] for profile_id in sorted(profiles_by_id)]


def _add_assignment(row: dict[str, Any], assignment_payload: dict[str, Any]) -> dict[str, Any]:
    enriched = copy.deepcopy(row)
    profile = enriched.setdefault("profile", {})
    if not isinstance(profile, dict):
        raise ValueError(f"Scenario row profile is not an object: {row.get('scenario_id')}")
    profile_metadata = profile.setdefault("metadata", {})
    if not isinstance(profile_metadata, dict):
        profile_metadata = {}
        profile["metadata"] = profile_metadata
    profile_metadata["exposure_network_assignment"] = copy.deepcopy(assignment_payload)

    row_metadata = enriched.setdefault("metadata", {})
    if not isinstance(row_metadata, dict):
        row_metadata = {}
        enriched["metadata"] = row_metadata
    row_metadata["exposure_network_assignment"] = copy.deepcopy(assignment_payload)
    return enriched


def run_stage(input_path: str, output_dir: str, config: Stage01bConfig) -> StageArtifactManifest:
    if not input_path:
        raise RuntimeError("Stage 01b requires Stage 01 scenarios.jsonl as --input-path")

    scenario_rows = [dict(row) for row in read_jsonl(input_path)]
    if not scenario_rows:
        raise RuntimeError(f"Stage 01b received no scenario rows: {input_path}")

    package = load_exposure_network_package(
        graph_root=config.exposure_network_root,
        graph_id=config.exposure_graph_id,
        validate=True,
    )
    unique_profiles = _unique_profiles(scenario_rows)
    assignments = assign_profiles_to_positions(
        unique_profiles,
        package,
        seed=config.seed,
        assignment_mode=config.exposure_assignment_mode,
    )

    enriched_rows: list[dict[str, Any]] = []
    for row in scenario_rows:
        profile_id = _profile_id(row)
        enriched_rows.append(_add_assignment(row, assignments[profile_id].to_context()))

    enriched_jsonl = Path(output_dir) / "scenarios_with_exposure_positions.jsonl"
    assignments_jsonl = Path(output_dir) / "profile_position_assignments.jsonl"
    summary_json = Path(output_dir) / "exposure_network_assignment_summary.json"

    assignment_rows = [assignments[profile_id].to_context() for profile_id in sorted(assignments)]
    summary = build_assignment_summary(
        assignments,
        package,
        seed=config.seed,
        assignment_mode=config.exposure_assignment_mode,
    )
    summary.update(
        {
            "scenario_count": len(enriched_rows),
            "input_path": abs_path(input_path),
            "primary_output_path": abs_path(enriched_jsonl),
        }
    )

    write_jsonl(enriched_jsonl, enriched_rows)
    write_jsonl(assignments_jsonl, assignment_rows)
    write_json(summary_json, summary)

    manifest = StageArtifactManifest(
        stage_id="01b",
        stage_name="assign_exposure_network_positions",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(enriched_jsonl),
        output_files=[abs_path(enriched_jsonl), abs_path(assignments_jsonl), abs_path(summary_json)],
        record_count=len(enriched_rows),
        metadata=summary,
    )
    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01b - Assign empirical exposure-network positions")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exposure-network-root", default=None)
    parser.add_argument("--exposure-graph-id", default=DEFAULT_GRAPH_ID)
    parser.add_argument("--exposure-assignment-mode", default=DEFAULT_ASSIGNMENT_MODE)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")
    config = Stage01bConfig(
        stage_name="assign_exposure_network_positions",
        run_id=args.run_id,
        seed=args.seed,
        exposure_network_root=args.exposure_network_root,
        exposure_graph_id=args.exposure_graph_id,
        exposure_assignment_mode=args.exposure_assignment_mode,
    )
    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 01b completed: %s scenario rows", manifest.record_count)


if __name__ == "__main__":
    main()
