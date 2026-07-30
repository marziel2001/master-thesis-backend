from __future__ import annotations

from fastApi.schemas.base import ApiModel


class UpdateOutputRequest(ApiModel):
    output_file: str
    wer: float | None = None
    cer: float | None = None
    reference_text: str = ""


class UpdateOutputResponse(ApiModel):
    output_file: str
