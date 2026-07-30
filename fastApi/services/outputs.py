"""Reading and writing the per-result transcription JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastApi.core.config import JSON_INDENT, OUTPUTS_DIR


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT),
        encoding="utf-8",
    )


def write_transcription_output(
    *,
    output_path: Path,
    model_name: str,
    model_version: str,
    compute_time: float | None,
    audio_duration: float | None,
    filename: str,
    transcription: str,
    reference_text: str,
    wer_value: float | None,
    cer_value: float | None,
) -> Path:
    """Persist one model's result. Key order is part of the file format."""
    write_json(
        output_path,
        {
            "modelName": model_name,
            "modelVersion": model_version,
            "computeTime": compute_time,
            "audioDuration": audio_duration,
            "filename": filename,
            "transcription": transcription,
            "reference": reference_text,
            "wer": wer_value,
            "cer": cer_value,
        },
    )
    return output_path


def resolve_output_path(raw_path: str) -> Path:
    """Resolve a client-supplied output path, refusing anything outside
    ``outputs/``.

    Raises ``ValueError`` when the path cannot be resolved or escapes the
    directory.
    """
    try:
        resolved = Path(raw_path).expanduser().resolve()
    except OSError as exc:
        raise ValueError("Invalid output file path.") from exc

    outputs_root = OUTPUTS_DIR.resolve()
    if outputs_root not in resolved.parents and resolved != outputs_root:
        raise ValueError("Output file is outside outputs directory.")

    return resolved


def read_output_file(path: Path) -> dict[str, Any]:
    """Load an output file. Raises ``ValueError`` if it is not a JSON object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid output file contents.") from exc

    if not isinstance(data, dict):
        raise ValueError("Output file must contain a JSON object.")

    return data
