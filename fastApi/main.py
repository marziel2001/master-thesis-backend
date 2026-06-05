from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from contextlib import closing
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from jiwer import cer, wer
from pydantic import BaseModel
from fastApi.diff_html import build_colored_diff_html, normalize_for_metrics
from fastApi.model_catalog import load_model_catalog
from fastApi.transcription_service import resolve_model_name, transcribe_audio
from transcribe.DEFAULT_MODELS import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_WHISPER_OFFLINE_MODEL,
    DEFAULT_WHISPERX_MODEL,
)

app = FastAPI(title="Transcription API", version="1.0.0")
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"

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
def models() -> dict[str, object]:
    return load_model_catalog()


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


class NormalizeTextRequest(BaseModel):
    text: str


class NormalizeTextResponse(BaseModel):
    text: str


class UpdateOutputRequest(BaseModel):
    output_file: str
    wer: float | None = None
    cer: float | None = None
    reference_text: str = ""


class UpdateOutputResponse(BaseModel):
    output_file: str


class RunResultPayload(BaseModel):
    model: str
    model_version: str | None = None
    transcription: str
    wer: float | None = None
    cer: float | None = None
    rt_time: float | None = None
    rtf: float | None = None
    audio_duration: float | None = None
    output_file: str | None = None


class RunCreateRequest(BaseModel):
    name: str | None = None
    reference_text: str
    audio_filename: str | None = None
    results: list[RunResultPayload]


class RunResponse(BaseModel):
    id: str
    created_at: str
    name: str | None = None
    reference_text: str
    audio_filename: str | None = None
    results: list[RunResultPayload]


class TranscriptionResponse(BaseModel):
    requested_model: str
    model: str
    model_name: str
    model_version: str
    compute_time: float
    audio_duration: float | None = None
    filename: str
    transcription: str
    wer: float | None = None
    cer: float | None = None
    rt_time: float | None = None
    output_file: str | None = None


def _resolve_model_version(
    model_name: str,
    model_variant: str,
    whisper_model: str,
) -> str:
    requested_variant = model_variant.strip()
    if model_name == "openai":
        return requested_variant or DEFAULT_OPENAI_MODEL
    if model_name == "whisper_offline":
        return requested_variant or whisper_model or DEFAULT_WHISPER_OFFLINE_MODEL
    if model_name == "whisperx":
        return requested_variant or whisper_model or DEFAULT_WHISPERX_MODEL
    return model_name


def _write_transcription_output(
    *,
    model_name: str,
    model_version: str,
    compute_time: float | None,
    audio_duration: float | None,
    filename: str,
    transcription: str,
    output_path: Path,
    wer_value: float | None,
    cer_value: float | None,
    reference_text: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modelName": model_name,
        "modelVersion": model_version,
        "computeTime": compute_time,
        "audioDuration": audio_duration,
        "filename": filename,
        "transcription": transcription,
        "reference": reference_text,
        "wer": wer_value,
        "cer": cer_value,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = f"{time.time_ns() % 1000000:06d}"
    return f"{stamp}_{random_part}"


def _sanitize_run_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return cleaned.strip(" ._")


def _get_audio_duration_ffprobe(audio_path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        duration_raw = data.get("format", {}).get("duration")
        duration = float(duration_raw)
        if duration > 0:
            return duration
    except Exception:
        return None

    return None


def _get_audio_duration_wave(audio_path: str) -> float | None:
    try:
        with closing(wave.open(audio_path, "rb")) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                return None
            duration = frames / float(rate)
            return duration if duration > 0 else None
    except Exception:
        return None


def get_audio_duration(audio_path: str) -> float | None:
    duration = _get_audio_duration_ffprobe(audio_path)
    if duration is not None:
        return duration
    return _get_audio_duration_wave(audio_path)


def _should_convert_to_wav(file_name: str) -> bool:
    suffix = Path(file_name).suffix.lower()
    return suffix in {".m4a", ".aac", ".mp3"}


def _convert_audio_to_wav(source_path: str, target_path: str) -> None:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                source_path,
                "-ar",
                "44100",
                "-ac",
                "1",
                target_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to convert m4a/mp3 files to wav."
        ) from exc

    if result.returncode != 0 or not os.path.exists(target_path):
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            "Failed to convert audio file to wav."
            + (f" ffmpeg output: {stderr}" if stderr else "")
        )


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


@app.post("/api/normalize-text", response_model=NormalizeTextResponse)
def normalize_text(payload: NormalizeTextRequest) -> NormalizeTextResponse:
    return NormalizeTextResponse(text=normalize_for_metrics(payload.text))


@app.get("/api/runs", response_model=list[RunResponse])
def list_runs() -> list[RunResponse]:
    _ensure_runs_dir()
    runs: list[RunResponse] = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8") or "{}")
            runs.append(RunResponse(**payload))
        except Exception:
            continue

    runs.sort(key=lambda run: run.created_at, reverse=True)
    return runs


@app.get("/api/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    run_file = RUNS_DIR / run_id / "run.json"
    if not run_file.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        payload = json.loads(run_file.read_text(encoding="utf-8") or "{}")
        return RunResponse(**payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid run file contents.",
        ) from exc


@app.post("/api/runs", response_model=RunResponse)
def create_run(payload: RunCreateRequest) -> RunResponse:
    if not payload.results:
        raise HTTPException(status_code=400, detail="Run must contain results.")

    _ensure_runs_dir()
    run_name = (payload.name or "").strip()
    run_id = _sanitize_run_id(run_name) if run_name else ""
    if not run_id:
        run_id = _generate_run_id()
        run_name = None
    else:
        run_name = run_id
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    created_at = datetime.now().isoformat()
    audio_stem = Path(payload.audio_filename or "audio").stem or "audio"
    output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[RunResultPayload] = []

    for result in payload.results:
        output_name = f"transcription_{result.model}_{audio_stem}_{output_timestamp}.json"
        output_path = run_dir / output_name
        _write_transcription_output(
            model_name=result.model,
            model_version=result.model_version or "",
            compute_time=result.rt_time,
            audio_duration=result.audio_duration,
            filename=payload.audio_filename or "",
            transcription=result.transcription,
            output_path=output_path,
            wer_value=result.wer,
            cer_value=result.cer,
            reference_text=payload.reference_text,
        )
        results.append(
            RunResultPayload(
                model=result.model,
                model_version=result.model_version,
                transcription=result.transcription,
                wer=result.wer,
                cer=result.cer,
                rt_time=result.rt_time,
                rtf=result.rtf,
                audio_duration=result.audio_duration,
                output_file=str(output_path),
            )
        )

    run_payload = RunResponse(
        id=run_id,
        created_at=created_at,
        name=run_name,
        reference_text=payload.reference_text,
        audio_filename=payload.audio_filename,
        results=results,
    )
    (run_dir / "run.json").write_text(
        json.dumps(run_payload.dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_payload


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, str]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        shutil.rmtree(run_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete run.",
        ) from exc
    return {"status": "ok"}


@app.post("/api/output/update", response_model=UpdateOutputResponse)
def update_output(payload: UpdateOutputRequest) -> UpdateOutputResponse:
    output_path = Path(payload.output_file).expanduser()
    try:
        resolved_path = output_path.resolve()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid output file path.",
        ) from exc

    outputs_root = OUTPUTS_DIR.resolve()
    if outputs_root not in resolved_path.parents and resolved_path != outputs_root:
        raise HTTPException(
            status_code=400,
            detail="Output file is outside outputs directory.",
        )

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")

    try:
        data = json.loads(resolved_path.read_text(encoding="utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("Output file must contain a JSON object.")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid output file contents.",
        ) from exc

    data["wer"] = payload.wer
    data["cer"] = payload.cer
    data["reference"] = payload.reference_text

    resolved_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return UpdateOutputResponse(output_file=str(resolved_path))


@app.post("/api/transcribe/{model_name}", response_model=TranscriptionResponse)
async def transcribe(
    model_name: str,
    file: UploadFile = File(...),
    whisper_model: str = Form(DEFAULT_WHISPER_OFFLINE_MODEL),
    model_variant: str = Form(""),
    reference_text: str = Form(""),
) -> TranscriptionResponse:
    temp_path = None
    converted_path = None

    try:
        normalized_model = resolve_model_name(model_name)
        suffix = os.path.splitext(file.filename or "audio.bin")[1]
        model_version = _resolve_model_version(
            normalized_model,
            model_variant,
            whisper_model,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        audio_path = temp_path
        if _should_convert_to_wav(file.filename or ""):
            wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav_temp.close()
            converted_path = wav_temp.name
            _convert_audio_to_wav(temp_path, converted_path)
            audio_path = converted_path

        start_time = time.perf_counter()
        transcript = await run_in_threadpool(
            transcribe_audio,
            normalized_model,
            audio_path,
            model_version,
        )
        rt_time = time.perf_counter() - start_time
        audio_duration = get_audio_duration(audio_path)

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
            audio_duration=audio_duration,
            filename=file.filename or "",
            transcription=transcript or "",
            output_path=output_path,
            wer_value=wer_value,
            cer_value=cer_value,
            reference_text=reference_text,
        )

        return TranscriptionResponse(
            requested_model=model_name,
            model=normalized_model,
            model_name=normalized_model,
            model_version=model_version,
            compute_time=rt_time,
            audio_duration=audio_duration,
            filename=file.filename or "",
            transcription=transcript or "",
            wer=wer_value,
            cer=cer_value,
            rt_time=rt_time,
            output_file=str(saved_output),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        await file.close()
        if converted_path and os.path.exists(converted_path):
            for _ in range(10):
                try:
                    os.remove(converted_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
            else:
                pass
        if temp_path and os.path.exists(temp_path):
            for _ in range(10):
                try:
                    os.remove(temp_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
            else:
                pass
