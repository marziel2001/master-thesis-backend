from __future__ import annotations

import html

from jiwer import process_words
from remove_punctuation import strip_punctuation_text


def normalize_for_metrics(text: str) -> str:
    return strip_punctuation_text(text).lower()


def _chunk_type_to_class(chunk_type: object) -> str:
    normalized = str(chunk_type).lower()
    if normalized in ("equal", "hit", "correct"):
        return "eq"
    if normalized in ("substitute", "substitution", "replace"):
        return "sub"
    if normalized in ("delete", "deletion"):
        return "del"
    if normalized in ("insert", "insertion"):
        return "ins"
    return "eq"


def _tokens_to_spans(tokens: list[str], css_class: str) -> str:
    if not tokens:
        return ""
    return "".join(
        f"<span class='token {css_class}'>{html.escape(token)}</span> " for token in tokens
    )


def build_colored_diff_html(reference_text: str, hypothesis_text: str, model_name: str) -> str:
    try:
        processed = process_words(reference_text, hypothesis_text)

        ref_line: list[str] = []
        hyp_line: list[str] = []

        for sent_idx, chunks in enumerate(processed.alignments):
            ref_tokens = processed.references[sent_idx]
            hyp_tokens = processed.hypotheses[sent_idx]

            for chunk in chunks:
                css_class = _chunk_type_to_class(chunk.type)
                ref_part = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
                hyp_part = hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]

                if css_class == "del":
                    ref_line.append(_tokens_to_spans(ref_part, "del"))
                    hyp_line.append("<span class='token gap'>∅</span> ")
                elif css_class == "ins":
                    ref_line.append("<span class='token gap'>∅</span> ")
                    hyp_line.append(_tokens_to_spans(hyp_part, "ins"))
                elif css_class == "sub":
                    ref_line.append(_tokens_to_spans(ref_part, "sub-ref"))
                    hyp_line.append(_tokens_to_spans(hyp_part, "sub-hyp"))
                else:
                    ref_line.append(_tokens_to_spans(ref_part, "eq"))
                    hyp_line.append(_tokens_to_spans(hyp_part, "eq"))

            ref_line.append("<br/>")
            hyp_line.append("<br/>")

        return f"""
        <style>
          .diff-wrap {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin: 8px 0 14px 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 12px;
            line-height: 1.7;
            white-space: normal;
            word-break: break-word;
          }}
          .diff-title {{ font-weight: 700; margin-bottom: 6px; }}
          .diff-row-label {{ font-weight: 700; color: #333; margin-right: 8px; }}
          .token {{
            display: inline-block;
            margin: 1px 2px;
            padding: 0 4px;
            border-radius: 4px;
          }}
          .eq {{ background: #f5f5f5; color: #222; }}
          .sub-ref {{ background: #ffe1e1; color: #8a1c1c; }}
          .sub-hyp {{ background: #fff1cc; color: #7a5b00; }}
          .del {{ background: #ffd6d6; color: #8a1c1c; text-decoration: line-through; }}
          .ins {{ background: #d9f8d9; color: #1d6f1d; font-weight: 600; }}
          .gap {{ background: #ececec; color: #888; }}
          .legend {{ margin-top: 8px; color: #555; font-size: 11px; }}
        </style>
        <div class='diff-wrap'>
          <div class='diff-title'>Różnice względem referencji - {html.escape(model_name)}</div>
          <div><span class='diff-row-label'>REF:</span>{''.join(ref_line)}</div>
          <div><span class='diff-row-label'>HYP:</span>{''.join(hyp_line)}</div>
          <div class='legend'>
            <span class='token sub-ref'>zamiana (ref)</span>
            <span class='token sub-hyp'>zamiana (model)</span>
            <span class='token del'>usuniete</span>
            <span class='token ins'>dodane</span>
          </div>
        </div>
        """
    except Exception as exc:
        return (
            "<div style='color:#b00020;'>"
            f"Nie udalo sie wygenerowac diff HTML dla {html.escape(model_name)}: {html.escape(str(exc))}"
            "</div>"
        )
