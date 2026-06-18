from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover - lean local environments
    class _OrjsonCompat:
        @staticmethod
        def dumps(data: Any) -> bytes:
            return json.dumps(data, ensure_ascii=False).encode("utf-8")

    orjson = _OrjsonCompat()


LOGGER = logging.getLogger(__name__)

LIVE_STATUS_FILENAME = "live_status.json"
LIVE_RESULTS_FILENAME = "live_results.jsonl"
LIVE_ERRORS_FILENAME = "live_errors.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_path(output_dir: str | Path) -> Path:
    return _output_dir(output_dir) / LIVE_STATUS_FILENAME


def _safe_write_status(output_dir: str | Path, payload: dict[str, Any]) -> None:
    try:
        path = _status_path(output_dir)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception as exc:  # pragma: no cover - best effort by design
        LOGGER.warning("Failed to write live status sidecar for %s: %s", output_dir, exc)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
    except Exception as exc:  # pragma: no cover - best effort by design
        LOGGER.warning("Failed to append live sidecar row to %s: %s", path, exc)


def init_live_stage(
    output_dir: str | Path,
    *,
    run_id: str,
    stage_id: str,
    stage_name: str,
    phase: str,
    total_count: int,
) -> None:
    output_path = _output_dir(output_dir)
    for filename in (LIVE_RESULTS_FILENAME, LIVE_ERRORS_FILENAME):
        try:
            path = output_path / filename
            if path.exists():
                path.unlink()
        except Exception as exc:  # pragma: no cover - best effort by design
            LOGGER.warning("Failed to clear live sidecar %s: %s", output_path / filename, exc)
    _safe_write_status(
        output_path,
        {
            "run_id": run_id,
            "stage_id": stage_id,
            "stage_name": stage_name,
            "phase": phase,
            "status": "running",
            "total_count": int(total_count),
            "completed_count": 0,
            "failed_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "source": "live_sidecar",
        },
    )


def update_live_status(
    output_dir: str | Path,
    *,
    completed_count: int,
    failed_count: int,
    status: str,
) -> None:
    path = _status_path(output_dir)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(
        {
            "status": status,
            "completed_count": int(completed_count),
            "failed_count": int(failed_count),
            "updated_at": _now(),
            "source": "live_sidecar",
        }
    )
    _safe_write_status(output_dir, payload)


def append_live_result(output_dir: str | Path, row: dict[str, Any]) -> None:
    _append_jsonl(_output_dir(output_dir) / LIVE_RESULTS_FILENAME, row)


def append_live_error(output_dir: str | Path, error: dict[str, Any]) -> None:
    payload = dict(error)
    payload.setdefault("timestamp", _now())
    _append_jsonl(_output_dir(output_dir) / LIVE_ERRORS_FILENAME, payload)
