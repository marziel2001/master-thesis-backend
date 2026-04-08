from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastApi.diff_html import build_colored_diff_html, normalize_for_metrics
from fastApi.transcription_service import available_models, resolve_model_name, transcribe_audio

app = FastAPI(title="Transcription API", version="1.0.0")

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


@app.post("/api/transcribe/{model_name}")
async def transcribe(
    model_name: str,
    file: UploadFile = File(...),
    whisper_model: str = Form("large-v3"),
) -> dict[str, str]:
    temp_path = None

    try:
        normalized_model = resolve_model_name(model_name)
        suffix = os.path.splitext(file.filename or "audio.bin")[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        transcript = transcribe_audio(
            model=normalized_model,
            audio_path=temp_path,
            whisper_model=whisper_model,
        )

        return {
            "requested_model": model_name,
            "model": normalized_model,
            "filename": file.filename or "",
            "transcription": transcript or "",
        }
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
