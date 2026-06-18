from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.backend.persistence.db import connect, lab_root
from src.backend.persistence.migrate import migrate_up
from src.backend.utils.io import read_json, read_jsonl


STAGES: tuple[tuple[str, str], ...] = (
    ("01", "create_scenarios"),
    ("02", "assess_baseline_opinions"),
    ("02b", "assess_network_exposure_opinions"),
    ("03", "run_opinion_attacks"),
    ("04", "assess_post_attack_opinions"),
    ("04b", "assess_post_attack_network_exposure_opinions"),
    ("05", "compute_effectivity_deltas"),
)

ASSESSMENT_KEYS: dict[str, tuple[str, str]] = {
    "02": ("baseline", "baseline_assessment"),
    "02b": ("network_exposure_baseline", "network_exposure_assessment"),
    "04": ("post_attack", "post_attack_assessment"),
    "04b": ("post_attack_network_exposure", "post_attack_network_exposure_assessment"),
}

CONTEXT_KEYS: dict[str, tuple[str, str]] = {
    "02b": ("network_exposure_baseline", "network_exposure_context"),
    "04b": ("post_attack_network_exposure", "post_attack_network_exposure_context"),
}


@dataclass(frozen=True)
class StageSource:
    stage_id: str
    stage_name: str
    manifest_path: Path
    manifest: dict[str, Any]
    primary_output_path: Path | None
    rows: list[dict[str, Any]]


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"run_[A-Za-z0-9_-]+", run_id):
        raise RuntimeError(f"Unsupported run id: {run_id}")
    return run_id


def _json(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value if value is not None else {})


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists() or path.suffix.lower() != ".jsonl":
        return []
    return [dict(row) for row in read_jsonl(path)]


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _load_sources(run_root: Path) -> tuple[dict[str, Any], Path, dict[str, StageSource]]:
    run_manifest_path = run_root / "provenance" / "run_manifest.json"
    if not run_manifest_path.exists():
        raise RuntimeError(f"Run manifest not found: {run_manifest_path}")
    run_manifest = dict(read_json(run_manifest_path))
    stage_outputs_root = _resolve_path(str(run_manifest.get("stage_outputs_root") or "")) or run_root / "stage_outputs"

    sources: dict[str, StageSource] = {}
    for stage_id, stage_name in STAGES:
        manifest_path = stage_outputs_root / f"{stage_id}_{stage_name}" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = dict(read_json(manifest_path))
        primary = _resolve_path(str(manifest.get("primary_output_path") or ""))
        sources[stage_id] = StageSource(
            stage_id=stage_id,
            stage_name=stage_name,
            manifest_path=manifest_path,
            manifest=manifest,
            primary_output_path=primary,
            rows=_read_rows(primary),
        )
    if "01" not in sources or not sources["01"].rows:
        raise RuntimeError("Stage 01 artifact is required for database ingestion.")
    return run_manifest, stage_outputs_root, sources


def _profile_id(row: dict[str, Any]) -> str | None:
    profile = row.get("profile")
    if not isinstance(profile, dict):
        return None
    profile_id = profile.get("profile_id")
    return str(profile_id) if profile_id else None


def _fallback_used(assessment: dict[str, Any]) -> bool:
    return str(assessment.get("model_name") or "") == "fallback_deterministic"


def _upsert_run(cur: Any, run_id: str, run_root: Path, stage_outputs_root: Path, run_manifest: dict[str, Any]) -> None:
    config = dict(run_manifest.get("pipeline_config") or {})
    cur.execute(
        """
        INSERT INTO pipeline_runs(run_id, output_root, stage_outputs_root, config_json, run_manifest_json, ingested_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (run_id) DO UPDATE SET
            output_root = EXCLUDED.output_root,
            stage_outputs_root = EXCLUDED.stage_outputs_root,
            config_json = EXCLUDED.config_json,
            run_manifest_json = EXCLUDED.run_manifest_json,
            ingested_at = now()
        """,
        (run_id, str(run_root), str(stage_outputs_root), _json(config), _json(run_manifest)),
    )


def _upsert_stage(cur: Any, run_id: str, source: StageSource) -> None:
    cur.execute(
        """
        INSERT INTO pipeline_stage_artifacts(
            run_id, stage_id, stage_name, manifest_path, primary_output_path,
            record_count, created_at_utc, manifest_json, artifact_checksum, ingested_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (run_id, stage_id) DO UPDATE SET
            stage_name = EXCLUDED.stage_name,
            manifest_path = EXCLUDED.manifest_path,
            primary_output_path = EXCLUDED.primary_output_path,
            record_count = EXCLUDED.record_count,
            created_at_utc = EXCLUDED.created_at_utc,
            manifest_json = EXCLUDED.manifest_json,
            artifact_checksum = EXCLUDED.artifact_checksum,
            ingested_at = now()
        """,
        (
            run_id,
            source.stage_id,
            source.stage_name,
            str(source.manifest_path),
            str(source.primary_output_path) if source.primary_output_path else None,
            int(source.manifest.get("record_count") or len(source.rows)),
            source.manifest.get("created_at_utc"),
            _json(source.manifest),
            _sha256(source.primary_output_path),
        ),
    )


def _ingest_stage01(cur: Any, run_id: str, source: StageSource) -> None:
    for row in source.rows:
        scenario_id = str(row.get("scenario_id") or "")
        profile_id = _profile_id(row)
        if not scenario_id or profile_id is None:
            continue
        profile = dict(row.get("profile") or {})
        cur.execute(
            """
            INSERT INTO profiles(
                run_id, profile_id, stage_id, scenario_id, manifest_path, primary_output_path,
                categorical_attributes, continuous_attributes, selected_leaf_nodes, metadata, raw_json, ingested_at
            )
            VALUES (%s, %s, '01', %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (run_id, profile_id) DO UPDATE SET
                scenario_id = EXCLUDED.scenario_id,
                manifest_path = EXCLUDED.manifest_path,
                primary_output_path = EXCLUDED.primary_output_path,
                categorical_attributes = EXCLUDED.categorical_attributes,
                continuous_attributes = EXCLUDED.continuous_attributes,
                selected_leaf_nodes = EXCLUDED.selected_leaf_nodes,
                metadata = EXCLUDED.metadata,
                raw_json = EXCLUDED.raw_json,
                ingested_at = now()
            """,
            (
                run_id,
                profile_id,
                scenario_id,
                str(source.manifest_path),
                str(source.primary_output_path) if source.primary_output_path else None,
                _json(profile.get("categorical_attributes") or {}),
                _json(profile.get("continuous_attributes") or {}),
                _json(profile.get("selected_leaf_nodes") or []),
                _json(profile.get("metadata") or {}),
                _json(profile),
            ),
        )
        cur.execute(
            """
            INSERT INTO scenarios(
                run_id, scenario_id, scenario_index, profile_id, opinion_leaf, attack_present, attack_leaf,
                stage_id, manifest_path, primary_output_path, metadata, raw_json, ingested_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, '01', %s, %s, %s, %s, now())
            ON CONFLICT (run_id, scenario_id) DO UPDATE SET
                scenario_index = EXCLUDED.scenario_index,
                profile_id = EXCLUDED.profile_id,
                opinion_leaf = EXCLUDED.opinion_leaf,
                attack_present = EXCLUDED.attack_present,
                attack_leaf = EXCLUDED.attack_leaf,
                manifest_path = EXCLUDED.manifest_path,
                primary_output_path = EXCLUDED.primary_output_path,
                metadata = EXCLUDED.metadata,
                raw_json = EXCLUDED.raw_json,
                ingested_at = now()
            """,
            (
                run_id,
                scenario_id,
                int(row.get("scenario_index") or 0),
                profile_id,
                str(row.get("opinion_leaf") or ""),
                bool(row.get("attack_present", False)),
                row.get("attack_leaf"),
                str(source.manifest_path),
                str(source.primary_output_path) if source.primary_output_path else None,
                _json(row.get("metadata") or {}),
                _json(row),
            ),
        )


def _ingest_assessments(cur: Any, run_id: str, source: StageSource) -> None:
    phase_and_key = ASSESSMENT_KEYS.get(source.stage_id)
    if phase_and_key is None:
        return
    phase, key = phase_and_key
    for row in source.rows:
        scenario_id = str(row.get("scenario_id") or "")
        profile_id = _profile_id(row)
        assessment = row.get(key)
        if not scenario_id or profile_id is None or not isinstance(assessment, dict):
            continue
        cur.execute(
            """
            INSERT INTO opinion_assessments(
                run_id, scenario_id, assessment_scenario_id, profile_id, phase, opinion_leaf,
                attack_present, attack_leaf, score, confidence, reasoning, model_name, fallback_used,
                stage_id, manifest_path, primary_output_path, assessment_json, row_json, ingested_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (run_id, scenario_id, phase) DO UPDATE SET
                assessment_scenario_id = EXCLUDED.assessment_scenario_id,
                profile_id = EXCLUDED.profile_id,
                opinion_leaf = EXCLUDED.opinion_leaf,
                attack_present = EXCLUDED.attack_present,
                attack_leaf = EXCLUDED.attack_leaf,
                score = EXCLUDED.score,
                confidence = EXCLUDED.confidence,
                reasoning = EXCLUDED.reasoning,
                model_name = EXCLUDED.model_name,
                fallback_used = EXCLUDED.fallback_used,
                stage_id = EXCLUDED.stage_id,
                manifest_path = EXCLUDED.manifest_path,
                primary_output_path = EXCLUDED.primary_output_path,
                assessment_json = EXCLUDED.assessment_json,
                row_json = EXCLUDED.row_json,
                ingested_at = now()
            """,
            (
                run_id,
                scenario_id,
                assessment.get("scenario_id"),
                profile_id,
                phase,
                str(row.get("opinion_leaf") or assessment.get("opinion_leaf") or ""),
                bool(row.get("attack_present", False)),
                row.get("attack_leaf"),
                int(assessment.get("score")),
                float(assessment.get("confidence")),
                str(assessment.get("reasoning") or ""),
                str(assessment.get("model_name") or ""),
                _fallback_used(assessment),
                source.stage_id,
                str(source.manifest_path),
                str(source.primary_output_path) if source.primary_output_path else None,
                _json(assessment),
                _json(row),
            ),
        )


def _ingest_contexts(cur: Any, run_id: str, source: StageSource) -> None:
    phase_and_key = CONTEXT_KEYS.get(source.stage_id)
    if phase_and_key is None:
        return
    phase, key = phase_and_key
    for row in source.rows:
        scenario_id = str(row.get("scenario_id") or "")
        profile_id = _profile_id(row)
        context = row.get(key)
        if not scenario_id or profile_id is None or not isinstance(context, dict):
            continue
        cur.execute(
            """
            INSERT INTO network_contexts(
                run_id, scenario_id, profile_id, phase, opinion_leaf, attack_leaf,
                stage_id, manifest_path, primary_output_path, context_json, raw_json, ingested_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (run_id, scenario_id, phase) DO UPDATE SET
                profile_id = EXCLUDED.profile_id,
                opinion_leaf = EXCLUDED.opinion_leaf,
                attack_leaf = EXCLUDED.attack_leaf,
                stage_id = EXCLUDED.stage_id,
                manifest_path = EXCLUDED.manifest_path,
                primary_output_path = EXCLUDED.primary_output_path,
                context_json = EXCLUDED.context_json,
                raw_json = EXCLUDED.raw_json,
                ingested_at = now()
            """,
            (
                run_id,
                scenario_id,
                profile_id,
                phase,
                str(row.get("opinion_leaf") or ""),
                row.get("attack_leaf"),
                source.stage_id,
                str(source.manifest_path),
                str(source.primary_output_path) if source.primary_output_path else None,
                _json(context),
                _json(row),
            ),
        )


def _ingest_attack_specs(cur: Any, run_id: str, sources: dict[str, StageSource]) -> None:
    for stage_id in ("03", "04", "04b"):
        source = sources.get(stage_id)
        if source is None:
            continue
        for row in source.rows:
            scenario_id = str(row.get("scenario_id") or "")
            spec = row.get("attack_vector_spec")
            if not scenario_id or not isinstance(spec, dict):
                continue
            cur.execute(
                """
                INSERT INTO attack_specs(
                    run_id, scenario_id, profile_id, opinion_leaf, attack_leaf, stage_id,
                    manifest_path, primary_output_path, attack_vector_spec, raw_json, ingested_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (run_id, scenario_id) DO UPDATE SET
                    profile_id = EXCLUDED.profile_id,
                    opinion_leaf = EXCLUDED.opinion_leaf,
                    attack_leaf = EXCLUDED.attack_leaf,
                    stage_id = EXCLUDED.stage_id,
                    manifest_path = EXCLUDED.manifest_path,
                    primary_output_path = EXCLUDED.primary_output_path,
                    attack_vector_spec = EXCLUDED.attack_vector_spec,
                    raw_json = EXCLUDED.raw_json,
                    ingested_at = now()
                """,
                (
                    run_id,
                    scenario_id,
                    _profile_id(row),
                    row.get("opinion_leaf"),
                    row.get("attack_leaf"),
                    stage_id,
                    str(source.manifest_path),
                    str(source.primary_output_path) if source.primary_output_path else None,
                    _json(spec),
                    _json(row),
                ),
            )


def _stage05_delta_path(source: StageSource) -> Path | None:
    for item in source.manifest.get("output_files", []):
        path = _resolve_path(str(item))
        if path is not None and path.name == "effectivity_deltas.jsonl":
            return path
    return source.primary_output_path if source.primary_output_path and source.primary_output_path.suffix == ".jsonl" else None


def _ingest_effectivity(cur: Any, run_id: str, source: StageSource) -> int:
    path = _stage05_delta_path(source)
    rows = _read_rows(path)
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            continue
        cur.execute(
            """
            INSERT INTO effectivity_deltas(
                run_id, scenario_id, profile_id, opinion_leaf, attack_leaf, baseline_score, post_score,
                delta_score, abs_delta_score, adversarial_effectivity, stage_id, manifest_path,
                primary_output_path, raw_json, ingested_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '05', %s, %s, %s, now())
            ON CONFLICT (run_id, scenario_id) DO UPDATE SET
                profile_id = EXCLUDED.profile_id,
                opinion_leaf = EXCLUDED.opinion_leaf,
                attack_leaf = EXCLUDED.attack_leaf,
                baseline_score = EXCLUDED.baseline_score,
                post_score = EXCLUDED.post_score,
                delta_score = EXCLUDED.delta_score,
                abs_delta_score = EXCLUDED.abs_delta_score,
                adversarial_effectivity = EXCLUDED.adversarial_effectivity,
                manifest_path = EXCLUDED.manifest_path,
                primary_output_path = EXCLUDED.primary_output_path,
                raw_json = EXCLUDED.raw_json,
                ingested_at = now()
            """,
            (
                run_id,
                scenario_id,
                row.get("profile_id"),
                row.get("opinion_leaf"),
                row.get("attack_leaf"),
                row.get("baseline_score"),
                row.get("post_score"),
                row.get("delta_score"),
                row.get("abs_delta_score"),
                row.get("adversarial_effectivity"),
                str(source.manifest_path),
                str(path) if path else None,
                _json(row),
            ),
        )
    return len(rows)


def ingest_pipeline_run(run_id: str, database_url: str | None = None, root: Path | None = None, apply_migrations: bool = True) -> dict[str, int]:
    safe_run_id = _safe_run_id(run_id)
    project_root = root or lab_root()
    run_root = project_root / "evaluation" / safe_run_id
    if not run_root.exists():
        raise RuntimeError(f"Pipeline run root not found: {run_root}")
    run_manifest, stage_outputs_root, sources = _load_sources(run_root)
    if apply_migrations:
        migrate_up(database_url)

    counts: dict[str, int] = {"stages": len(sources), "effectivity_deltas": 0}
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            _upsert_run(cur, safe_run_id, run_root, stage_outputs_root, run_manifest)
            for source in sources.values():
                _upsert_stage(cur, safe_run_id, source)
            _ingest_stage01(cur, safe_run_id, sources["01"])
            counts["profiles"] = len({row.get("profile", {}).get("profile_id") for row in sources["01"].rows})
            counts["scenarios"] = len(sources["01"].rows)
            for source in sources.values():
                _ingest_assessments(cur, safe_run_id, source)
                _ingest_contexts(cur, safe_run_id, source)
            _ingest_attack_specs(cur, safe_run_id, sources)
            if "05" in sources:
                counts["effectivity_deltas"] = _ingest_effectivity(cur, safe_run_id, sources["05"])
        conn.commit()
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest canonical pipeline artifacts into Postgres.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--lab-root", default=None)
    parser.add_argument("--no-migrate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.lab_root).resolve() if args.lab_root else None
    counts = ingest_pipeline_run(
        run_id=args.run_id,
        database_url=args.database_url,
        root=root,
        apply_migrations=not args.no_migrate,
    )
    print(json.dumps({"run_id": args.run_id, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()

