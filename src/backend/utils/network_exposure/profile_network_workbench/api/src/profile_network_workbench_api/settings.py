from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class WorkbenchSettings:
    lab_root: Path
    workbench_root: Path
    evaluation_path: Path
    test_ontology_root: Path
    production_ontology_root: Path
    runs_root: Path
    allowed_origins: tuple[str, ...]


def _default_lab_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_settings() -> WorkbenchSettings:
    lab_root = Path(os.getenv("PROFILE_NETWORK_LAB_ROOT", _default_lab_root())).resolve()
    workbench_root = lab_root / "profile_network_workbench"

    load_dotenv(lab_root / ".env")
    load_dotenv(workbench_root / "api" / ".env")

    if str(lab_root) not in sys.path:
        sys.path.insert(0, str(lab_root))

    ontology_root = lab_root / "src" / "backend" / "ontology" / "separate"
    allowed_origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "PROFILE_NETWORK_ALLOWED_ORIGINS",
            "http://127.0.0.1:5176,http://localhost:5176,"
            "http://127.0.0.1:5177,http://localhost:5177,"
            "http://127.0.0.1:5180,http://localhost:5180,"
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if origin.strip()
    )
    runs_root = Path(os.getenv("PROFILE_NETWORK_RUNS_ROOT", workbench_root / "runs")).resolve()

    return WorkbenchSettings(
        lab_root=lab_root,
        workbench_root=workbench_root,
        evaluation_path=lab_root / "evaluation",
        test_ontology_root=ontology_root / "test",
        production_ontology_root=ontology_root / "production",
        runs_root=runs_root,
        allowed_origins=allowed_origins,
    )
