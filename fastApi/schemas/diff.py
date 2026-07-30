from __future__ import annotations

from fastApi.schemas.base import ApiModel


class DiffHtmlRequest(ApiModel):
    reference_text: str
    hypothesis_text: str
    model_name: str
    normalize: bool = True


class DiffHtmlResponse(ApiModel):
    ref_html: str
    hyp_html: str
    title_html: str | None = None
