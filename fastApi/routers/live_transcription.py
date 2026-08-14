"""Websocket endpoint that fans a live microphone feed out to every model."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastApi.services.live_transcription import (
    ChunkRef,
    LiveModel,
    build_live_models,
    convert_chunk_to_wav,
    transcribe_chunk_for_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live-transcription"])

#: Container format the frontend's MediaRecorder is asked to produce.
INCOMING_CHUNK_SUFFIX = ".webm"


def _drain_queue(live_model: LiveModel) -> None:
    """Releases every chunk still waiting in a model's queue on shutdown."""
    while not live_model.queue.empty():
        try:
            _chunk_index, chunk_ref = live_model.queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        chunk_ref.release()


async def _model_worker(
    live_model: LiveModel, outbox: asyncio.Queue[dict[str, Any]]
) -> None:
    while True:
        chunk_index, chunk_ref = await live_model.queue.get()
        result = await transcribe_chunk_for_model(live_model, chunk_index, chunk_ref)
        await outbox.put(result)


async def _outbox_writer(
    websocket: WebSocket, outbox: asyncio.Queue[dict[str, Any] | None]
) -> None:
    while True:
        message = await outbox.get()
        if message is None:
            return
        await websocket.send_json(message)


def _parse_control_message(raw_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


@router.websocket("/ws/live-transcription")
async def live_transcription(websocket: WebSocket) -> None:
    await websocket.accept()

    live_models = build_live_models()
    if not live_models:
        await websocket.send_json(
            {"type": "fatal", "message": "No transcription models are configured."}
        )
        await websocket.close()
        return

    outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    writer_task = asyncio.create_task(_outbox_writer(websocket, outbox))
    worker_tasks = [
        asyncio.create_task(_model_worker(live_model, outbox))
        for live_model in live_models
    ]

    await outbox.put(
        {
            "type": "ready",
            "models": [live_model.catalog_id for live_model in live_models],
        }
    )

    chunk_index = 0
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            raw_bytes = message.get("bytes")
            if raw_bytes is None:
                raw_text = message.get("text")
                control = _parse_control_message(raw_text) if raw_text else None
                if control is not None and control.get("type") == "stop":
                    break
                continue

            chunk_index += 1
            current_index = chunk_index

            try:
                wav_path = await run_in_threadpool(
                    convert_chunk_to_wav, raw_bytes, INCOMING_CHUNK_SUFFIX
                )
            except RuntimeError as exc:
                await outbox.put(
                    {"type": "chunk_error", "index": current_index, "message": str(exc)}
                )
                continue

            chunk_ref = ChunkRef(path=wav_path, pending=len(live_models))
            await outbox.put({"type": "chunk_received", "index": current_index})

            for live_model in live_models:
                try:
                    live_model.queue.put_nowait((current_index, chunk_ref))
                except asyncio.QueueFull:
                    chunk_ref.release()
                    await outbox.put(
                        {
                            "type": "chunk_skipped",
                            "model": live_model.catalog_id,
                            "index": current_index,
                        }
                    )
    except WebSocketDisconnect:
        pass
    finally:
        for live_model in live_models:
            _drain_queue(live_model)

        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        await outbox.put(None)
        await writer_task

        with contextlib.suppress(RuntimeError):
            await websocket.close()
