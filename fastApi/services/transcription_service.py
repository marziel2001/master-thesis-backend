"""Dispatch to the per-provider speech-to-text adapters in ``transcribe/``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fastApi.services.model_catalog import available_models
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

#: Accepted spellings for each canonical model id. Keys are compared after
#: lower-casing and stripping every non-alphanumeric character, so the catalog
#: ids (``whisperOffline``, ``googleStt``, ...) resolve here too.
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

#: Loading a local Whisper model is expensive, so instances are reused.
_local_whisper_clients: dict[str, Any] = {}


def resolve_model_name(model_name: str) -> ModelName:
    """Map any accepted alias onto a canonical model id."""
    normalized = "".join(ch for ch in model_name.strip().lower() if ch.isalnum())
    resolved = _MODEL_ALIASES.get(normalized)

    if resolved is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Use one of: {', '.join(available_models())}"
        )

    return resolved


def resolve_model_version(
    model: ModelName,
    requested_variant: str,
    whisper_model: str,
) -> str:
    """The variant that will actually run, falling back to the defaults.

    Providers without selectable variants report their own id as the version.
    """
    variant = requested_variant.strip()

    if model == "openai":
        return variant or DEFAULT_OPENAI_MODEL
    if model == "whisper_offline":
        return variant or whisper_model or DEFAULT_WHISPER_OFFLINE_MODEL
    if model == "whisperx":
        return variant or whisper_model or DEFAULT_WHISPERX_MODEL

    return model


def _transcribe_with_openai(audio_path: str, variant: str) -> str:
    from transcribe.openAiWhisper import transcribe_file

    return transcribe_file(audio_path, model=variant or DEFAULT_OPENAI_MODEL)


def _transcribe_with_local_whisper(audio_path: str, variant: str) -> str:
    from transcribe.whisperOffline import LocalWhisperClient

    model_size = variant or DEFAULT_WHISPER_OFFLINE_MODEL
    if model_size not in _local_whisper_clients:
        _local_whisper_clients[model_size] = LocalWhisperClient(
            model_size=model_size
        )

    result = _local_whisper_clients[model_size].transcribe(audio_path)

    return result.get("text", "") if isinstance(result, dict) else ""


def _transcribe_with_whisperx(audio_path: str, variant: str) -> str:
    from transcribe.whisperX import transcribe_file

    return transcribe_file(
        audio_path=audio_path,
        model_size=variant or DEFAULT_WHISPERX_MODEL,
    )


def _transcribe_with_google(audio_path: str, _variant: str) -> str:
    from transcribe.google_stt import transcribe_file

    response = transcribe_file(audio_path)
    results = getattr(response, "results", None)
    if not results:
        return ""

    parts: list[str] = []

    for file_result in results.values():
        # `inline_result` is present for inline-output requests; older responses
        # expose the transcript directly.
        inline = getattr(file_result, "inline_result", None)
        transcript = (
            inline.transcript
            if inline and inline.transcript
            else getattr(file_result, "transcript", None)
        )
        if not transcript:
            continue

        parts.extend(
            alternative.transcript
            for result in transcript.results
            for alternative in result.alternatives
            if alternative.transcript
        )

    return " ".join(parts).strip()


def _transcribe_with_azure(audio_path: str, _variant: str) -> str:
    from transcribe.ms_azure_stt import transcribe_file

    return transcribe_file(audio_path)


def _transcribe_with_amazon(audio_path: str, _variant: str) -> str:
    from transcribe.amazon_stt import transcribe_file

    return transcribe_file(audio_path)


#: Adapters are imported lazily inside each function so that a missing optional
#: dependency only breaks the provider that needs it.
_TRANSCRIBERS: dict[ModelName, Callable[[str, str], str]] = {
    "openai": _transcribe_with_openai,
    "whisper_offline": _transcribe_with_local_whisper,
    "whisperx": _transcribe_with_whisperx,
    "google": _transcribe_with_google,
    "azure": _transcribe_with_azure,
    "amazon": _transcribe_with_amazon,
}


def transcribe_audio(
    model: ModelName,
    audio_path: str,
    model_variant: str | None = None,
) -> str:
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcriber = _TRANSCRIBERS.get(model)
    if transcriber is None:
        raise ValueError(f"Unsupported model: {model}")

    return transcriber(audio_path, (model_variant or "").strip())
