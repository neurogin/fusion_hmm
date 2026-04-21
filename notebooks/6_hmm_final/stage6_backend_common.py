"""Shared utilities for the public Stage-6 backend modules.

These helpers keep the lightweight path, JSON, and display logic in one place
so the Stage-6 backends can focus on the scientific steps they implement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def display_or_print(obj: Any) -> None:
    """Display rich notebook output when available, otherwise print the object."""
    try:
        from IPython.display import display as ipy_display

        ipy_display(obj)
    except Exception:
        print(obj)


def resolve_existing_path(*candidates: str | Path | None) -> Path | None:
    """Return the first existing path from a list of candidates, or `None`."""
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file into a Python dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_result_file(result_root: Path, final_dir: Path, name: str) -> Path:
    """Find a required Stage-6 result file under the root or `final/` directory."""
    path = resolve_existing_path(result_root / name, final_dir / name, Path(name))
    if path is None:
        raise FileNotFoundError(f"Could not find required file: {name}")
    return path
