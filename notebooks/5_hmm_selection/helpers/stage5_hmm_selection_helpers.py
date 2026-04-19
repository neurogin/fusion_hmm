"""Compatibility shim for the Stage-5 public helper layer.

The active public Stage-5 helper modules now live directly in
`notebooks/5_hmm_selection/`:
- `stage5_hmm_selection_helpers.py`
- `stage5_k_sweep_backend.py`
- `stage5_shortlist_backend.py`

This compatibility file remains only so older imports from the historical
`helpers/` subfolder do not break. It forwards to the active same-directory
public helper layer and does not execute preserved provenance notebooks.
"""

from __future__ import annotations

import sys
from pathlib import Path


STAGE5_DIR = Path(__file__).resolve().parents[1]
if str(STAGE5_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE5_DIR))

from stage5_hmm_selection_helpers import *  # noqa: F401,F403
