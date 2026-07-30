from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from fastApi.core.config import MODEL_CATALOG_PATH

logger = logging.getLogger(__name__)

EMPTY_CATALOG: dict[str, Any] = {"models": []}


@lru_cache
def load_model_catalog() -> dict[str, Any]:
    """Parsed ``models.json``, cached for the process lifetime."""
    if not MODEL_CATALOG_PATH.exists():
        logger.warning("Model catalog not found at %s", MODEL_CATALOG_PATH)
        return EMPTY_CATALOG

    return json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))


def available_models() -> list[str]:
    """Ids of every catalog entry that declares one."""
    models = load_model_catalog().get("models")
    if not isinstance(models, list):
        return []

    return [
        entry["id"]
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]
