from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean local envs
    class _OrjsonCompat:
        @staticmethod
        def loads(data: bytes | str) -> Any:
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)

        @staticmethod
        def dumps(data: Any) -> bytes:
            return json.dumps(data, ensure_ascii=False).encode("utf-8")

    orjson = _OrjsonCompat()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return orjson.loads(handle.read())


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)
    return target


def read_jsonl(path: str | Path) -> List[Any]:
    rows: List[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
    return target


def write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(content, encoding="utf-8")
    return target


def abs_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def bool_from_str(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def stage_manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "manifest.json"


def clean_filename(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def env_get_required(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
