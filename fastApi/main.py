from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
