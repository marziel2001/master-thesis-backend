from __future__ import annotations

import os
import time
from typing import Dict, Literal

from fastApi.model_catalog import available_models as catalog_available_models
from transcribe.DEFAULT_MODELS import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_WHISPER_OFFLINE_MODEL,
    DEFAULT_WHISPERX_MODEL,
)

ModelName = Literal[
    "openai",
    "whisper_offline",
    "whisperx",
    "google",
    "azure",
    "amazon",
]

_whisper_clients: Dict[str, object] = {}

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
    return catalog_available_models()


def resolve_model_name(model_name: str) -> ModelName:
    normalized = "".join(ch for ch in model_name.strip().lower() if ch.isalnum())
    resolved = _MODEL_ALIASES.get(normalized)
    if not resolved:
        raise ValueError(
            f"Unsupported model '{model_name}'. Use one of: {', '.join(available_models())}"
        )
    return resolved


def _transcribe_with_google(audio_path: str) -> str:
    from transcribe.google_stt import transcribe_file

    response = transcribe_file(audio_path)

    if not getattr(response, "results", None):
        return ""

    parts: list[str] = []

    for file_result in response.results.values():

        # 1. próbuj inline_result (często najprostsze)
        inline = getattr(file_result, "inline_result", None)
        if inline and inline.transcript:
            for r in inline.transcript.results:
                for alt in r.alternatives:
                    if alt.transcript:
                        parts.append(alt.transcript)
            continue

        # 2. fallback: transcript field
        transcript_obj = getattr(file_result, "transcript", None)
        if transcript_obj:
            for r in transcript_obj.results:
                for alt in r.alternatives:
                    if alt.transcript:
                        parts.append(alt.transcript)

    return " ".join(parts).strip()

def _transcribe_with_openai(audio_path: str, model_variant: str) -> str:
    from transcribe.openAiWhisper import transcribe_file

    return transcribe_file(audio_path, model=model_variant)


def _transcribe_with_whisperx(audio_path: str, whisper_model: str) -> str:
    from transcribe.whisperX import transcribe_file

    return transcribe_file(audio_path=audio_path, model_size=whisper_model)


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


def transcribe_audio(model: ModelName, audio_path: str, model_variant: str | None = None) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    resolved_variant = (model_variant or "").strip()

    mock_responses = False

    def _mock_response(text: str) -> str:
        time.sleep(1)
        return text

    if model == "openai":
        if mock_responses:
            return _mock_response("Open ai called")
        return _transcribe_with_openai(
            audio_path,
            resolved_variant or DEFAULT_OPENAI_MODEL,
        )
    if model == "whisper_offline":
        if mock_responses:
            return _mock_response("Local whisper called")
        return _transcribe_with_local_whisper(
            audio_path,
            whisper_model=resolved_variant or DEFAULT_WHISPER_OFFLINE_MODEL,
        )
    if model == "whisperx":
        if mock_responses:
            return _mock_response("WhisperX called")
        return _transcribe_with_whisperx(
            audio_path,
            whisper_model=resolved_variant or DEFAULT_WHISPERX_MODEL,
        )
    if model == "google":
        if mock_responses:
            return _mock_response("Google called")

        return _transcribe_with_google(audio_path)
    if model == "azure":
        if mock_responses:
            return _mock_response("Azure called")
        return _transcribe_with_azure(audio_path)
    if model == "amazon":
        if mock_responses:
            return _mock_response("Amazon called")
        return _transcribe_with_amazon(audio_path)

    raise ValueError(f"Unsupported model: {model}")
