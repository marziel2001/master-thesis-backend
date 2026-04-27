from __future__ import annotations

import os
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


def _transcribe_with_whisperx(audio_path: str, whisper_model: str) -> str:
    from transcribe.whisperX import transcribe_file

    return transcribe_file(audio_path, model_size=whisper_model)


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
