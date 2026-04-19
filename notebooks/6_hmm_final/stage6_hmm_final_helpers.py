"""Helper functions for the cleaned Stage-6 final K=3 workflow.

This module supports the public Stage-6 notebooks that:
- run the final full-data K=3 fit
- review saved final-fit QC and state-dynamics outputs
- reconstruct BOLD, cross-modal, and cortical-map summaries
- optionally export Figure 4 and Figure 5 panels

Important note:
- the public notebooks now keep the user-facing setup and scientific notes visible
- the dense runtime code lives in same-directory Python backend modules
- the old preserved PipelineE notebooks remain provenance copies rather than active backends
"""

from __future__ import annotations

from pathlib import Path

from stage6_bold_state_backend import run_bold_state_reconstruction_backend
from stage6_brainmaps_backend import run_brainmaps_backend
from stage6_crossmodal_backend import run_crossmodal_reconstruction_backend
from stage6_final_fit_backend import run_final_fit_backend
from stage6_panel_export_backend import run_panel_export_backend
from stage6_review_backend import run_final_review_backend


def ensure_configured_path(path_value, label: str, *, must_exist: bool = False) -> Path:
    """Validate a user-supplied notebook path and raise a plain-language error if needed."""
    path = Path(path_value)
    if "<SET_" in str(path):
        raise ValueError(f"{label} still uses a placeholder path. Edit the notebook before running.")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def run_public_final_fit_step(**kwargs):
    """Run Step 60 through the active final-fit backend module."""
    return run_final_fit_backend(**kwargs)


def run_public_review_step(**kwargs):
    """Run Step 61 through the active review backend module."""
    return run_final_review_backend(**kwargs)


def run_public_bold_state_reconstruction_step(**kwargs):
    """Run Step 62 through the active BOLD-state reconstruction backend module."""
    return run_bold_state_reconstruction_backend(**kwargs)


def run_public_crossmodal_reconstruction_step(**kwargs):
    """Run Step 63 through the active cross-modal reconstruction backend module."""
    return run_crossmodal_reconstruction_backend(**kwargs)


def run_public_brainmap_step(**kwargs):
    """Run Step 64 through the active cortical-map backend module."""
    return run_brainmaps_backend(**kwargs)


def run_public_panel_export_step(**kwargs):
    """Run Step 65 through the active optional panel-export backend module."""
    return run_panel_export_backend(**kwargs)

