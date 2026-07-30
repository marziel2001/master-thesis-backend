from __future__ import annotations

from fastapi import APIRouter
from fastApi.schemas.metrics import MetricsRequest, MetricsResponse
from fastApi.services.metrics import calculate_metrics

router = APIRouter(prefix="/api", tags=["metrics"])


@router.post("/metrics", response_model=MetricsResponse)
def metrics(payload: MetricsRequest) -> MetricsResponse:
    wer_value, cer_value = calculate_metrics(
        reference_text=payload.reference_text,
        hypothesis_text=payload.hypothesis_text,
        normalize=payload.normalize,
    )

    return MetricsResponse(wer=wer_value, cer=cer_value)
