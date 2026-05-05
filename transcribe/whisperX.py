from __future__ import annotations

import json
import time
import os
from typing import Any
import argparse
from datetime import datetime

try:
    import whisperx
except Exception as exc:
    raise ImportError(
        "Failed to import 'whisperx'. Install with: pip install -U whisperx "
        "and ensure ffmpeg is installed and on PATH."
    ) from exc

try:
    import torch
except Exception:
    torch = None

_MODEL_CACHE: dict[str, Any] = {}


def _resolve_device(prefer_device: str | None) -> str:
    if prefer_device:
        return prefer_device
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_compute_type(device: str, prefer_compute_type: str | None) -> str:
    if prefer_compute_type:
        return prefer_compute_type
    if device == "cuda":
        return "float16"
    return "int8"


def _resolve_language(prefer_language: str | None) -> str | None:
    if prefer_language is None:
        prefer_language = os.getenv("WHISPERX_LANGUAGE", "pl")
    language = prefer_language.strip()
    return language or None


def _resolve_batch_size(prefer_batch_size: int | None) -> int:
    if prefer_batch_size is not None and prefer_batch_size > 0:
        return prefer_batch_size
    env_value = os.getenv("WHISPERX_BATCH_SIZE")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return 16


def _get_model(model_size: str, device: str, compute_type: str, language: str | None) -> Any:
    cache_key = f"{model_size}:{device}:{compute_type}:{language or 'auto'}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = whisperx.load_model(
            model_size,
            device,
            compute_type=compute_type,
            language=language,
        )
    return _MODEL_CACHE[cache_key]


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text.strip()
        segments = result.get("segments")
        if isinstance(segments, list):
            parts: list[str] = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                segment_text = str(segment.get("text", "")).strip()
                if segment_text:
                    parts.append(segment_text)
            return " ".join(parts).strip()
    return ""


def transcribe_file(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    batch_size: int | None = None,
) -> str:
    resolved_language = _resolve_language(language)
    resolved_device = _resolve_device(device or os.getenv("WHISPERX_DEVICE"))
    resolved_compute_type = _resolve_compute_type(
        resolved_device,
        compute_type or os.getenv("WHISPERX_COMPUTE_TYPE"),
    )
    resolved_batch_size = _resolve_batch_size(batch_size)

    model = _get_model(
        model_size=model_size,
        device=resolved_device,
        compute_type=resolved_compute_type,
        language=resolved_language,
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(
        audio,
        batch_size=resolved_batch_size,
        language=resolved_language,
    )
    return _extract_text(result)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WhisperX transcription helper")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--language", default=None, help="Language code, e.g. pl")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument(
        "--compute-type",
        default=None,
        dest="compute_type",
        help="float16/int8 or other WhisperX-supported type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Batch size for transcription",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path for transcription text",
    )
    return parser


def _main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()
    started_at = time.perf_counter()
    text = transcribe_file(
        audio_path=args.audio_path,
        model_size=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
    )
    compute_time = time.perf_counter() - started_at
    output_path = args.output
    if not output_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        outputs_dir = os.path.join(base_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        output_path = os.path.join(
            outputs_dir,
            f"transcription_whisperx_{audio_name}_{timestamp}.json",
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "modelName": "whisperx",
        "modelVersion": args.model,
        "computeTime": compute_time,
        "filename": os.path.basename(args.audio_path),
        "transcription": text,
        "language": args.language,
        "device": args.device,
        "computeType": args.compute_type,
        "batchSize": args.batch_size,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(text)
    print(f"Saved transcription to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
