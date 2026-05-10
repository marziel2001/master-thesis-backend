from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parent / "models.json"


@lru_cache
def load_model_catalog() -> dict[str, Any]:
    if not CATALOG_PATH.exists():
        return {"models": []}
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def available_models() -> list[str]:
    catalog = load_model_catalog()
    models = catalog.get("models")
    if not isinstance(models, list):
        return []

    ids: list[str] = []
    for entry in models:
        if isinstance(entry, dict):
            model_id = entry.get("id")
            if isinstance(model_id, str):
                ids.append(model_id)
    return ids
