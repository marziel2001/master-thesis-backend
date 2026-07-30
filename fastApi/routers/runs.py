from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from fastapi import APIRouter, HTTPException
from fastApi.schemas.runs import RunCreateRequest, RunResponse, RunResultPayload
from fastApi.services.outputs import write_transcription_output
from fastApi.services.runs_repository import (
    create_run_dir,
    delete_run,
    generate_run_id,
    get_run_dir,
    iter_run_payloads,
    read_run_payload,
    sanitize_run_id,
    write_run_payload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["runs"])

INVALID_RUN_DETAIL = "Invalid run file contents."
RUN_NOT_FOUND_DETAIL = "Run not found."


@router.get("/runs", response_model=list[RunResponse])
def list_runs() -> list[RunResponse]:
    runs: list[RunResponse] = []

    for payload in iter_run_payloads():
        try:
            runs.append(RunResponse(**payload))
        except ValidationError:
            logger.warning("Skipping run with unexpected contents: %s", payload.get("id"))

    runs.sort(key=lambda run: run.name, reverse=True)

    return runs


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    try:
        payload = read_run_payload(run_id)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=INVALID_RUN_DETAIL) from exc

    if payload is None:
        raise HTTPException(status_code=404, detail=RUN_NOT_FOUND_DETAIL)

    try:
        return RunResponse(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=INVALID_RUN_DETAIL) from exc


@router.post("/runs", response_model=RunResponse)
def create_run(payload: RunCreateRequest) -> RunResponse:
    """Store a run and write one output file per model result.

    A run named by the user is stored under a sanitised version of that name,
    which means saving twice under the same name replaces the earlier run.
    Unnamed runs get a timestamp id and no name.
    """
    if not payload.results:
        raise HTTPException(status_code=400, detail="Run must contain results.")

    requested_name = (payload.name or "").strip()
    sanitized_name = sanitize_run_id(requested_name) if requested_name else ""

    run_id = sanitized_name or generate_run_id()
    run_name = sanitized_name or None

    run_dir = create_run_dir(run_id)

    audio_stem = Path(payload.audio_filename or "audio").stem or "audio"
    output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: list[RunResultPayload] = []
    for result in payload.results:
        output_path = run_dir / (
            f"transcription_{result.model}_{audio_stem}_{output_timestamp}.json"
        )
        write_transcription_output(
            output_path=output_path,
            model_name=result.model,
            model_version=result.model_version or "",
            compute_time=result.rt_time,
            audio_duration=result.audio_duration,
            filename=payload.audio_filename or "",
            transcription=result.transcription,
            reference_text=payload.reference_text,
            wer_value=result.wer,
            cer_value=result.cer,
        )
        results.append(result.model_copy(update={"output_file": str(output_path)}))

    run = RunResponse(
        id=run_id,
        created_at=datetime.now().isoformat(),
        name=run_name,
        reference_text=payload.reference_text,
        audio_filename=payload.audio_filename,
        results=results,
    )
    write_run_payload(run_id, run.model_dump())

    return run


@router.delete("/runs/{run_id}")
def remove_run(run_id: str) -> dict[str, str]:
    if not get_run_dir(run_id).exists():
        raise HTTPException(status_code=404, detail=RUN_NOT_FOUND_DETAIL)

    try:
        delete_run(run_id)
    except OSError as exc:
        logger.exception("Could not delete run %s", run_id)
        raise HTTPException(
            status_code=500, detail="Failed to delete run."
        ) from exc

    return {"status": "ok"}
