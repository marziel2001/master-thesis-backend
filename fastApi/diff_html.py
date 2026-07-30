"""Renders a reference/hypothesis word alignment as two HTML fragments."""

from __future__ import annotations

import html
import logging

from jiwer import process_words

from scripts.remove_punctuation import strip_punctuation_text

logger = logging.getLogger(__name__)

#: jiwer chunk types, grouped by how they should be highlighted.
_EQUAL_TYPES = frozenset({"equal", "hit", "correct"})
_SUBSTITUTE_TYPES = frozenset({"substitute", "substitution", "replace"})
_DELETE_TYPES = frozenset({"delete", "deletion"})
_INSERT_TYPES = frozenset({"insert", "insertion"})

#: Placeholder shown opposite an inserted or deleted word.
_GAP_SPAN = "<span class='token gap'>∅</span> "

_BASE_STYLE = """
        <style>
          .diff-wrap {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin: 8px 0 14px 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 12px;
            line-height: 1.7;
            white-space: normal;
            word-break: break-word;
          }
          .token { display:inline; margin:0; padding:0; }
          .eq { background: transparent; }
          .sub-ref, .sub-hyp, .del, .ins {
            padding: 0 4px;
            border-radius: 4px;
          }
          .sub-ref { background:#ffe1e1; color:#8a1c1c; }
          .sub-hyp { background:#fff1cc; color:#7a5b00; }
          .del { background:#ffd6d6; color:#8a1c1c; text-decoration:line-through; }
          .ins { background:#d9f8d9; color:#1d6f1d; font-weight:600; }
          .gap { color:#888; }
        </style>
        """


def normalize_for_metrics(text: str) -> str:
    """Lower-cased, punctuation-free form used for both metrics and diffing."""
    return strip_punctuation_text(text).lower()


def _chunk_class(chunk_type: object) -> str:
    normalized = str(chunk_type).lower()

    if normalized in _SUBSTITUTE_TYPES:
        return "sub"
    if normalized in _DELETE_TYPES:
        return "del"
    if normalized in _INSERT_TYPES:
        return "ins"
    if normalized not in _EQUAL_TYPES:
        logger.debug("Unrecognised alignment chunk type %r; treating as equal", chunk_type)

    return "eq"


def _tokens_to_spans(tokens: list[str], css_class: str) -> str:
    return "".join(
        f"<span class='token {css_class}'>{html.escape(token)}</span> "
        for token in tokens
    )


def _wrap(title: str, body_parts: list[str]) -> str:
    return f"""
        {_BASE_STYLE}
        <div class='diff-wrap'>
          <div><b>{title}:</b></div>
          <div>{''.join(body_parts)}</div>
        </div>
        """


def build_colored_diff_html(
    reference_text: str,
    hypothesis_text: str,
    model_name: str,
) -> tuple[str, str]:
    """Return ``(reference_html, hypothesis_html)`` with aligned words marked up.

    Every token is HTML-escaped. On failure both fragments carry the same error
    message, so the caller always has something to display.
    """
    try:
        processed = process_words(reference_text, hypothesis_text)

        reference_parts: list[str] = []
        hypothesis_parts: list[str] = []

        for sentence_index, chunks in enumerate(processed.alignments):
            reference_tokens = processed.references[sentence_index]
            hypothesis_tokens = processed.hypotheses[sentence_index]

            for chunk in chunks:
                css_class = _chunk_class(chunk.type)
                reference_slice = reference_tokens[
                    chunk.ref_start_idx : chunk.ref_end_idx
                ]
                hypothesis_slice = hypothesis_tokens[
                    chunk.hyp_start_idx : chunk.hyp_end_idx
                ]

                if css_class == "del":
                    reference_parts.append(_tokens_to_spans(reference_slice, "del"))
                    hypothesis_parts.append(_GAP_SPAN)
                elif css_class == "ins":
                    reference_parts.append(_GAP_SPAN)
                    hypothesis_parts.append(_tokens_to_spans(hypothesis_slice, "ins"))
                elif css_class == "sub":
                    reference_parts.append(
                        _tokens_to_spans(reference_slice, "sub-ref")
                    )
                    hypothesis_parts.append(
                        _tokens_to_spans(hypothesis_slice, "sub-hyp")
                    )
                else:
                    reference_parts.append(_tokens_to_spans(reference_slice, "eq"))
                    hypothesis_parts.append(
                        _tokens_to_spans(hypothesis_slice, "eq")
                    )

            reference_parts.append("<br/>")
            hypothesis_parts.append("<br/>")

        return _wrap("Wzorzec", reference_parts), _wrap("Wynik", hypothesis_parts)
    except Exception as exc:
        logger.exception("Could not build diff for model %s", model_name)
        error_html = (
            f"<div style='color:#b00020;'>Błąd: {html.escape(str(exc))}</div>"
        )
        return error_html, error_html
