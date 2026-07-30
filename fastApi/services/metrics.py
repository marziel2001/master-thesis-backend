from __future__ import annotations

from jiwer import cer, wer

from fastApi.diff_html import normalize_for_metrics


def calculate_metrics(
    reference_text: str,
    hypothesis_text: str,
    normalize: bool = True,
) -> tuple[float, float]:
    """Word and character error rate, in that order."""
    if normalize:
        reference_text = normalize_for_metrics(reference_text)
        hypothesis_text = normalize_for_metrics(hypothesis_text)

    return (
        wer(reference_text, hypothesis_text),
        cer(reference_text, hypothesis_text),
    )
