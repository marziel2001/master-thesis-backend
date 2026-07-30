from __future__ import annotations

from fastapi import APIRouter
from fastApi.diff_html import build_colored_diff_html, normalize_for_metrics
from fastApi.schemas.diff import DiffHtmlRequest, DiffHtmlResponse

router = APIRouter(prefix="/api", tags=["diff"])


@router.post("/diff-html", response_model=DiffHtmlResponse)
def diff_html(payload: DiffHtmlRequest) -> DiffHtmlResponse:
    reference_text = payload.reference_text
    hypothesis_text = payload.hypothesis_text

    if payload.normalize:
        reference_text = normalize_for_metrics(reference_text)
        hypothesis_text = normalize_for_metrics(hypothesis_text)

    ref_html, hyp_html = build_colored_diff_html(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        model_name=payload.model_name,
    )

    return DiffHtmlResponse(ref_html=ref_html, hyp_html=hyp_html)
