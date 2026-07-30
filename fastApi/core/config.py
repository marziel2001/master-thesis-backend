"""Paths and tunables used across the API.

Previously these were scattered as module-level constants and inline literals
in ``main.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The ``fastApi`` package directory.
API_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: Repository root, i.e. the directory holding ``fastApi/`` and ``transcribe/``.
BACKEND_ROOT = API_PACKAGE_DIR.parent

OUTPUTS_DIR = BACKEND_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"

#: Editable list of selectable models and their variants.
MODEL_CATALOG_PATH = API_PACKAGE_DIR / "models.json"

#: File name holding a run's metadata inside its directory.
RUN_METADATA_FILENAME = "run.json"

DEFAULT_FRONTEND_ORIGIN = "http://localhost:5173"

#: Formats that the speech backends cannot read directly.
CONVERTIBLE_AUDIO_SUFFIXES = frozenset({".m4a", ".aac", ".mp3"})

WAV_SAMPLE_RATE = 44_100
WAV_CHANNELS = 1

#: Windows can hold a lock on a just-closed temp file for a short while.
TEMP_FILE_REMOVE_ATTEMPTS = 10
TEMP_FILE_REMOVE_DELAY_SECONDS = 0.1

#: Indentation used for every JSON file this API writes.
JSON_INDENT = 2


def get_frontend_origin() -> str:
    """CORS origin allowed to call this API."""
    return os.getenv("FRONTEND_ORIGIN", DEFAULT_FRONTEND_ORIGIN)
