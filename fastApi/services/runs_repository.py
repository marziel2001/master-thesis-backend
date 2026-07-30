"""Filesystem storage for saved runs.

Each run is a directory under ``outputs/runs/`` holding a ``run.json`` plus one
JSON file per model result.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from fastApi.core.config import RUN_METADATA_FILENAME, RUNS_DIR
from fastApi.services.outputs import write_json

logger = logging.getLogger(__name__)


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def get_run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def generate_run_id() -> str:
    """Timestamp plus a sub-second suffix, for runs saved without a name."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{time.time_ns() % 1_000_000:06d}"


def sanitize_run_id(name: str) -> str:
    """Reduce a user-supplied name to something usable as a directory name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip(" ._")


def iter_run_payloads() -> Iterator[dict[str, Any]]:
    """Yield each stored run's raw payload, skipping unreadable ones."""
    ensure_runs_dir()

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / RUN_METADATA_FILENAME
        if not metadata_path.exists():
            continue

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable run file %s", metadata_path)
            continue

        if isinstance(payload, dict):
            yield payload


def read_run_payload(run_id: str) -> dict[str, Any] | None:
    """Raw payload for one run, or ``None`` when it does not exist."""
    metadata_path = get_run_dir(run_id) / RUN_METADATA_FILENAME
    if not metadata_path.exists():
        return None

    payload = json.loads(metadata_path.read_text(encoding="utf-8") or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Invalid run file contents.")

    return payload


def create_run_dir(run_id: str) -> Path:
    """Fresh directory for a run, replacing any existing one with that id."""
    ensure_runs_dir()

    run_dir = get_run_dir(run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def write_run_payload(run_id: str, payload: dict[str, Any]) -> None:
    write_json(get_run_dir(run_id) / RUN_METADATA_FILENAME, payload)


def delete_run(run_id: str) -> None:
    shutil.rmtree(get_run_dir(run_id))
