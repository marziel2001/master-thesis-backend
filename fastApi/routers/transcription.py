from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastApi.core.config import OUTPUTS_DIR
from fastApi.schemas.transcription import TranscriptionResponse
from fastApi.services.audio import get_audio_duration, prepared_audio_file
from fastApi.services.metrics import calculate_metrics
from fastApi.services.outputs import write_transcription_output
from fastApi.services.transcription_service import (
    resolve_model_name,
    resolve_model_version,
    transcribe_audio,
)
from transcribe.DEFAULT_MODELS import DEFAULT_WHISPER_OFFLINE_MODEL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcription"])


def _safe_metrics(
    reference_text: str, hypothesis_text: str
) -> tuple[float | None, float | None]:
    """Metrics for the request, or ``(None, None)`` if they cannot be computed.

    A transcription is still worth returning even when scoring it fails.
    """
    if not reference_text.strip():
        return None, None

    try:
        return calculate_metrics(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
            normalize=True,
        )
    except Exception:
        logger.warning("Could not compute metrics for this request", exc_info=True)
        return None, None


@router.post("/transcribe/{model_name}", response_model=TranscriptionResponse)
async def transcribe(
    model_name: str,
    file: UploadFile = File(...),
    whisper_model: str = Form(DEFAULT_WHISPER_OFFLINE_MODEL),
    model_variant: str = Form(""),
    reference_text: str = Form(""),
) -> TranscriptionResponse:
    try:
        resolved_model = resolve_model_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_version = resolve_model_version(
        resolved_model, model_variant, whisper_model
    )
    original_filename = file.filename or ""

    try:
        upload_bytes = await file.read()
    finally:
        await file.close()

    try:
        with prepared_audio_file(upload_bytes, original_filename) as audio_path:
            started_at = time.perf_counter()
            transcript = await run_in_threadpool(
                transcribe_audio,
                resolved_model,
                str(audio_path),
                model_version,
            )
            rt_time = time.perf_counter() - started_at
            audio_duration = get_audio_duration(audio_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed for model %s", model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    transcription = transcript or ""
    wer_value, cer_value = _safe_metrics(reference_text, transcription)

    audio_stem = Path(original_filename or "audio.bin").stem or "audio"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / (
        f"transcription_{resolved_model}_{audio_stem}_{timestamp}.json"
    )
    write_transcription_output(
        output_path=output_path,
        model_name=resolved_model,
        model_version=model_version,
        compute_time=rt_time,
        audio_duration=audio_duration,
        filename=original_filename,
        transcription=transcription,
        reference_text=reference_text,
        wer_value=wer_value,
        cer_value=cer_value,
    )

    return TranscriptionResponse(
        requested_model=model_name,
        model=resolved_model,
        model_name=resolved_model,
        model_version=model_version,
        compute_time=rt_time,
        audio_duration=audio_duration,
        filename=original_filename,
        transcription=transcription,
        wer=wer_value,
        cer=cer_value,
        rt_time=rt_time,
        output_file=str(output_path),
    )
