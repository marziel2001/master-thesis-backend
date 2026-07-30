from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastApi.services.model_catalog import load_model_catalog

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def models() -> dict[str, Any]:
    return load_model_catalog()
