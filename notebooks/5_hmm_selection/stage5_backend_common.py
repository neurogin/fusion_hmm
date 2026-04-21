"""Shared utilities for the public Stage-5 model-selection backends.

This module keeps the repeated manifest and path-resolution logic in one place
for the Stage-5 K-sweep and shortlist backends.

The public notebooks still call the stage-specific backend modules directly.
This file only factors out the repeated housekeeping that both backends need:
- finding the canonical Stage-4 `segments_manifest.tsv`
- extracting held-out subject IDs from run names
- resolving segment paths from the manifest
"""

from __future__ import annotations

from pathlib import Path


def auto_find_manifest(final_root: Path, feature_mode: str, minlen: int) -> Path:
    """Locate the retained-segment manifest for the requested Stage-4 branch."""
    mode = feature_mode.lower()
    candidates = [
        final_root / f"hmm_segments_minlen{minlen}_{mode}" / "segments_manifest.tsv",
        final_root / f"hmm_segments_minlen{minlen}" / "segments_manifest.tsv",
    ]
    for manifest_tsv in candidates:
        if manifest_tsv.exists():
            return manifest_tsv

    hits = list(final_root.rglob("segments_manifest.tsv"))
    if hits:
        def score(path: Path) -> int:
            path_str = str(path)
            score_value = 0
            if f"minlen{minlen}" in path_str:
                score_value += 10
            if mode in path_str:
                score_value += 5
            return score_value

        return sorted(hits, key=score, reverse=True)[0]

    raise FileNotFoundError(f"Could not find segments_manifest.tsv under {final_root}")


def parse_subject_from_run(run: str) -> str:
    """Extract the held-out subject ID from a run label such as `sub-01_ses-01`."""
    for part in str(run).split("_"):
        if part.startswith("sub-"):
            return part
    return str(run).split("_")[0]


def resolve_segment_path(seg_root: Path, path_like: str) -> Path:
    """Resolve a manifest segment path that may be absolute or relative."""
    path = Path(path_like)
    return path if path.is_absolute() else (seg_root / path)
