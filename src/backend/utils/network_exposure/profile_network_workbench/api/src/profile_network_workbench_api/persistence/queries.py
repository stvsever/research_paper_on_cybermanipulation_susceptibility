from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import connect, database_url


STAGES: tuple[tuple[str, str], ...] = (
    ("01", "create_scenarios"),
    ("02", "assess_baseline_opinions"),
    ("02b", "assess_network_exposure_opinions"),
    ("04", "assess_post_attack_opinions"),
    ("04b", "assess_post_attack_network_exposure_opinions"),
)


@dataclass(frozen=True)
class DbPipelineViewData:
    run_id: str
    run_root: Path
    stage_outputs_root: Path
    statuses: list[dict[str, Any]]
    rows_by_stage: dict[str, list[dict[str, Any]]]
    warnings: list[str]


def configured(database_url_override: str | None = None) -> bool:
    return database_url(database_url_override) is not None


def load_pipeline_view_data(run_id: str, database_url_override: str | None = None) -> DbPipelineViewData | None:
    if not configured(database_url_override):
        return None

    with connect(database_url_override) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT output_root, stage_outputs_root FROM pipeline_runs WHERE run_id = %s",
                (run_id,),
            )
            run_row = cur.fetchone()
            if run_row is None:
                return None
            output_root, stage_outputs_root = run_row

            cur.execute(
                """
                SELECT stage_id, stage_name, manifest_path, primary_output_path, record_count, created_at_utc
                FROM pipeline_stage_artifacts
                WHERE run_id = %s
                """,
                (run_id,),
            )
            existing = {
                row[0]: {
                    "stage_id": row[0],
                    "stage_name": row[1],
                    "available": True,
                    "manifest_path": row[2],
                    "primary_output_path": row[3],
                    "record_count": row[4],
                    "created_at_utc": row[5],
                }
                for row in cur.fetchall()
            }

            statuses = [
                existing.get(
                    stage_id,
                    {
                        "stage_id": stage_id,
                        "stage_name": stage_name,
                        "available": False,
                        "manifest_path": str(Path(stage_outputs_root) / f"{stage_id}_{stage_name}" / "manifest.json"),
                    },
                )
                for stage_id, stage_name in STAGES
            ]

            cur.execute(
                """
                SELECT raw_json
                FROM scenarios
                WHERE run_id = %s
                ORDER BY scenario_index, scenario_id
                """,
                (run_id,),
            )
            rows_by_stage: dict[str, list[dict[str, Any]]] = {"01": [dict(row[0]) for row in cur.fetchall()]}

            phase_to_stage = {
                "baseline": "02",
                "network_exposure_baseline": "02b",
                "post_attack": "04",
                "post_attack_network_exposure": "04b",
            }
            cur.execute(
                """
                SELECT phase, row_json
                FROM opinion_assessments
                WHERE run_id = %s
                ORDER BY scenario_id, phase
                """,
                (run_id,),
            )
            for phase, row_json in cur.fetchall():
                stage_id = phase_to_stage.get(str(phase))
                if stage_id:
                    rows_by_stage.setdefault(stage_id, []).append(dict(row_json))

    return DbPipelineViewData(
        run_id=run_id,
        run_root=Path(output_root),
        stage_outputs_root=Path(stage_outputs_root),
        statuses=statuses,
        rows_by_stage=rows_by_stage,
        warnings=["Loaded pipeline_view from Postgres artifact projection."],
    )

