from __future__ import annotations

from fastapi import APIRouter
from fastApi.diff_html import normalize_for_metrics
from fastApi.schemas.metrics import NormalizeTextRequest, NormalizeTextResponse

router = APIRouter(prefix="/api", tags=["text"])


@router.post("/normalize-text", response_model=NormalizeTextResponse)
def normalize_text(payload: NormalizeTextRequest) -> NormalizeTextResponse:
    """Return the text in the same normalised form used for metrics."""
    return NormalizeTextResponse(text=normalize_for_metrics(payload.text))
