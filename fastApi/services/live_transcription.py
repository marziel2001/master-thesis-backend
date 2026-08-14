"""Fan-out of live microphone chunks to every catalog model at once.

The browser records short, independently-decodable audio chunks (a few
seconds each) and streams them over a websocket. Each chunk is converted to
WAV once and handed to one queue per model; a dedicated worker task per model
transcribes its queue in order, so a slow provider (a polling job, a large
local model) never blocks the others. Queues are capped at one pending chunk:
once a model falls behind, newly arriving chunks are dropped for it rather
than queued up, which keeps every model "live" instead of replaying a growing
backlog.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from fastApi.services.audio import convert_to_wav, remove_file_with_retry
from fastApi.services.model_catalog import load_model_catalog
from fastApi.services.transcription_service import (
    ModelName,
    resolve_model_name,
    resolve_model_version,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

#: Chunks queued per model before new ones are dropped so the model can catch up.
MODEL_QUEUE_MAXSIZE = 1


@dataclass
class ChunkRef:
    """A chunk's WAV file, removed once every model has consumed or dropped it."""

    path: Path
    pending: int

    def release(self) -> None:
        self.pending -= 1
        if self.pending <= 0:
            remove_file_with_retry(self.path)


@dataclass
class LiveModel:
    """One catalog model's live-session state: its queue and its identity."""

    catalog_id: str
    resolved_model: ModelName
    variant: str
    queue: asyncio.Queue[tuple[int, ChunkRef]]


def build_live_models() -> list[LiveModel]:
    """One entry per catalog model that can actually be dispatched to."""
    catalog_entries = load_model_catalog().get("models", [])
    live_models: list[LiveModel] = []

    for entry in catalog_entries:
        if not isinstance(entry, dict):
            continue

        catalog_id = entry.get("id")
        if not isinstance(catalog_id, str) or not catalog_id:
            continue

        try:
            resolved_model = resolve_model_name(catalog_id)
        except ValueError:
            logger.warning("Skipping unresolvable catalog model %s", catalog_id)
            continue

        default_variant = entry.get("defaultVariant")
        variant = resolve_model_version(
            resolved_model,
            default_variant if isinstance(default_variant, str) else "",
            "",
        )

        live_models.append(
            LiveModel(
                catalog_id=catalog_id,
                resolved_model=resolved_model,
                variant=variant,
                queue=asyncio.Queue(maxsize=MODEL_QUEUE_MAXSIZE),
            )
        )

    return live_models


def convert_chunk_to_wav(raw_bytes: bytes, suffix: str) -> Path:
    """Transcodes one recorded chunk to mono WAV.

    Browser chunks are webm/ogg containers, never WAV, so this always shells
    out to ffmpeg rather than checking the extension first.
    """
    source_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".webm") as source_file:
            source_file.write(raw_bytes)
            source_path = Path(source_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            wav_path = Path(wav_file.name)

        convert_to_wav(source_path, wav_path)
        return wav_path
    finally:
        if source_path is not None:
            remove_file_with_retry(source_path)


async def transcribe_chunk_for_model(
    live_model: LiveModel, chunk_index: int, chunk_ref: ChunkRef
) -> dict[str, Any]:
    """Runs one model on one chunk, always returning a client-ready message.

    Provider SDKs raise all sorts of exceptions (network, auth, decoding); a
    blind catch here is intentional so one model's failure becomes a message
    for that model instead of tearing down the whole live session.
    """
    started_at = time.perf_counter()
    try:
        text = await run_in_threadpool(
            transcribe_audio,
            live_model.resolved_model,
            str(chunk_ref.path),
            live_model.variant,
        )
    except Exception as exc:
        logger.warning(
            "Live transcription failed for %s (chunk %s)",
            live_model.catalog_id,
            chunk_index,
            exc_info=True,
        )
        return {
            "type": "error",
            "model": live_model.catalog_id,
            "index": chunk_index,
            "message": str(exc) or "Transcription failed.",
        }
    else:
        return {
            "type": "result",
            "model": live_model.catalog_id,
            "index": chunk_index,
            "text": text or "",
            "computeTime": time.perf_counter() - started_at,
        }
    finally:
        chunk_ref.release()
