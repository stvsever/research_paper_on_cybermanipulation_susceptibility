from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root(current_file: str, levels_up: int = 4) -> Path:
    project_root = Path(current_file).resolve().parents[levels_up]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
