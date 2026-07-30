from __future__ import annotations

from fastApi.schemas.base import ApiModel


class MetricsRequest(ApiModel):
    reference_text: str
    hypothesis_text: str
    normalize: bool = True


class MetricsResponse(ApiModel):
    wer: float
    cer: float


class NormalizeTextRequest(ApiModel):
    text: str


class NormalizeTextResponse(ApiModel):
    text: str
