from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, Literal

from transcribe.amazon_stt import transcribe_file

ModelName = Literal[
    "openai",
    "whisper_offline",
    "whisperx",
    "google",
    "azure",
    "amazon",
]

_whisper_clients: Dict[str, object] = {}

_PATH_MODEL_NAMES = [
    "openai",
    "whisperOffline",
    "whisperX",
    "googleStt",
    "azureStt",
    "amazonStt",
]
_MODEL_ALIASES: dict[str, ModelName] = {
    "openai": "openai",
    "openaiwhisper": "openai",
    "whisperoffline": "whisper_offline",
    "whisperofflinestt": "whisper_offline",
    "localwhisper": "whisper_offline",
    "whisperx": "whisperx",
    "whisperxstt": "whisperx",
    "googlestt": "google",
    "google": "google",
    "azurestt": "azure",
    "azure": "azure",
    "msazurestt": "azure",
    "amazonstt": "amazon",
    "amazon": "amazon",
    "awstranscribe": "amazon",
}


def available_models() -> list[str]:
    return _PATH_MODEL_NAMES


def resolve_model_name(model_name: str) -> ModelName:
    normalized = "".join(ch for ch in model_name.strip().lower() if ch.isalnum())
    resolved = _MODEL_ALIASES.get(normalized)
    if not resolved:
        raise ValueError(
            f"Unsupported model '{model_name}'. Use one of: {', '.join(_PATH_MODEL_NAMES)}"
        )
    return resolved


def _transcribe_with_google(audio_path: str) -> str:
    from transcribe.google_stt import transcribe_file

    response = transcribe_file(audio_path)
    if not getattr(response, "results", None):
        return ""

    parts: list[str] = []
    for result in response.results:
        if not result.alternatives:
            continue
        parts.append(result.alternatives[0].transcript)

    return " ".join(parts).strip()


def _transcribe_with_openai(audio_path: str) -> str:
    from transcribe.openAiWhisper import transcribe_file

    return transcribe_file(audio_path)


def _resolve_whisperx_python() -> str:
    python_override = os.getenv("WHISPERX_PYTHON")
    if python_override:
        return python_override

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if sys.platform.startswith("win"):
        candidate = os.path.join(base_dir, ".venv-whisperx", "Scripts", "python.exe")
    else:
        candidate = os.path.join(base_dir, ".venv-whisperx", "bin", "python")

    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        "WhisperX Python not found. Set WHISPERX_PYTHON to the python.exe from "
        "the WhisperX venv (e.g. backend/.venv-whisperx)."
    )


def _resolve_whisperx_script() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(base_dir, "transcribe", "whisperX.py")
    if os.path.exists(script_path):
        return script_path
    raise FileNotFoundError(f"WhisperX script not found: {script_path}")


def _transcribe_with_whisperx(audio_path: str, whisper_model: str) -> str:
    python_exe = _resolve_whisperx_python()
    script_path = _resolve_whisperx_script()

    output_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            output_file = tmp.name

        command = [
            python_exe,
            script_path,
            audio_path,
            "--model",
            whisper_model,
            "--output",
            output_file,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        with open(output_file, "r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                handle.seek(0)
                return handle.read().strip()

        if isinstance(payload, dict):
            transcription = payload.get("transcription")
            if isinstance(transcription, str):
                return transcription.strip()

        return ""
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or "Unknown error"
        raise RuntimeError(f"WhisperX subprocess failed: {details}") from exc
    finally:
        if output_file and os.path.exists(output_file):
            os.remove(output_file)


def _transcribe_with_azure(audio_path: str) -> str:
    from transcribe.ms_azure_stt import transcribe_file

    return transcribe_file(audio_path)


def _transcribe_with_amazon(audio_path: str) -> str:
    from transcribe.amazon_stt import transcribe_file

    return transcribe_file(audio_path)


def _transcribe_with_local_whisper(audio_path: str, whisper_model: str) -> str:
    from transcribe.whisperOffline import LocalWhisperClient
    if whisper_model not in _whisper_clients:
        _whisper_clients[whisper_model] = LocalWhisperClient(model_size=whisper_model)

    result = _whisper_clients[whisper_model].transcribe(audio_path)
    return result.get("text", "") if isinstance(result, dict) else ""


def transcribe_audio(model: ModelName, audio_path: str, whisper_model: str = "large-v3") -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model == "openai":
        return "Open ai called"
        return _transcribe_with_openai(audio_path)
    if model == "whisper_offline":
        # return "local whisper called"
        return _transcribe_with_local_whisper(audio_path, whisper_model=whisper_model)
    if model == "whisperx":
        return _transcribe_with_whisperx(audio_path, whisper_model=whisper_model)
    if model == "google":
        return "Google called"
        return _transcribe_with_google(audio_path)
    if model == "azure":
        return "Azure called"
        return _transcribe_with_azure(audio_path)
    if model == "amazon":
        return "Amazon called"
        return _transcribe_with_amazon(audio_path)

    raise ValueError(f"Unsupported model: {model}")
