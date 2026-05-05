from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jiwer import cer, wer
from pydantic import BaseModel
from fastApi.diff_html import build_colored_diff_html, normalize_for_metrics
from fastApi.transcription_service import available_models, resolve_model_name, transcribe_audio

app = FastAPI(title="Transcription API", version="1.0.0")
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/api/models")
def models() -> dict[str, list[str]]:
    return {"models": available_models()}


class DiffHtmlRequest(BaseModel):
    reference_text: str
    hypothesis_text: str
    model_name: str
    normalize: bool = True


class DiffHtmlResponse(BaseModel):
    html: str


class MetricsRequest(BaseModel):
    reference_text: str
    hypothesis_text: str
    normalize: bool = True


class MetricsResponse(BaseModel):
    wer: float
    cer: float


class TranscriptionResponse(BaseModel):
    requested_model: str
    model: str
    model_name: str
    model_version: str
    compute_time: float
    filename: str
    transcription: str
    wer: float | None = None
    cer: float | None = None
    rt_time: float | None = None
    output_file: str | None = None


def _resolve_model_version(model_name: str, whisper_model: str) -> str:
    if model_name in {"whisper_offline", "whisperx"}:
        return whisper_model
    if model_name == "openai":
        return "gpt-4o-transcribe"
    return model_name


def _write_transcription_output(
    *,
    model_name: str,
    model_version: str,
    compute_time: float,
    filename: str,
    transcription: str,
    output_path: Path,
    wer_value: float | None,
    cer_value: float | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modelName": model_name,
        "modelVersion": model_version,
        "computeTime": compute_time,
        "filename": filename,
        "transcription": transcription,
        "wer": wer_value,
        "cer": cer_value,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def calculate_metrics(
    reference_text: str,
    hypothesis_text: str,
    normalize: bool = True,
) -> tuple[float, float]:
    if normalize:
        reference_text = normalize_for_metrics(reference_text)
        hypothesis_text = normalize_for_metrics(hypothesis_text)

    return wer(reference_text, hypothesis_text), cer(reference_text, hypothesis_text)


@app.post("/api/diff-html", response_model=DiffHtmlResponse)
def diff_html(payload: DiffHtmlRequest) -> DiffHtmlResponse:
    reference_text = payload.reference_text
    hypothesis_text = payload.hypothesis_text

    if payload.normalize:
        reference_text = normalize_for_metrics(reference_text)
        hypothesis_text = normalize_for_metrics(hypothesis_text)

    html_output = build_colored_diff_html(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        model_name=payload.model_name,
    )
    return DiffHtmlResponse(html=html_output)


@app.post("/api/metrics", response_model=MetricsResponse)
def metrics(payload: MetricsRequest) -> MetricsResponse:
    wer_value, cer_value = calculate_metrics(
        reference_text=payload.reference_text,
        hypothesis_text=payload.hypothesis_text,
        normalize=payload.normalize,
    )
    return MetricsResponse(wer=wer_value, cer=cer_value)


@app.post("/api/transcribe/{model_name}", response_model=TranscriptionResponse)
async def transcribe(
    model_name: str,
    file: UploadFile = File(...),
    whisper_model: str = Form("large-v3"),
    reference_text: str = Form(""),
) -> TranscriptionResponse:
    temp_path = None

    try:
        normalized_model = resolve_model_name(model_name)
        suffix = os.path.splitext(file.filename or "audio.bin")[1]
        model_version = _resolve_model_version(normalized_model, whisper_model)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        start_time = time.perf_counter()
        transcript = transcribe_audio(
            model=normalized_model,
            audio_path=temp_path,
            whisper_model=whisper_model,
        )
        rt_time = time.perf_counter() - start_time

        wer_value: float | None = None
        cer_value: float | None = None
        if reference_text.strip():
            try:
                wer_value, cer_value = calculate_metrics(
                    reference_text=reference_text,
                    hypothesis_text=transcript or "",
                    normalize=True,
                )
            except Exception:
                wer_value, cer_value = None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_stem = Path(file.filename or "audio.bin").stem or "audio"
        output_path = OUTPUTS_DIR / f"transcription_{normalized_model}_{audio_stem}_{timestamp}.json"
        saved_output = _write_transcription_output(
            model_name=normalized_model,
            model_version=model_version,
            compute_time=rt_time,
            filename=file.filename or "",
            transcription=transcript or "",
            output_path=output_path,
            wer_value=wer_value,
            cer_value=cer_value,
        )

        return TranscriptionResponse(
            requested_model=model_name,
            model=normalized_model,
            model_name=normalized_model,
            model_version=model_version,
            compute_time=rt_time,
            filename=file.filename or "",
            transcription=transcript or "",
            wer=wer_value,
            cer=cer_value,
            rt_time=rt_time,
            output_file=str(saved_output),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
