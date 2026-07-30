from __future__ import annotations

from fastApi.schemas.base import ApiModel


class RunResultPayload(ApiModel):
    """One model's result inside a saved run."""

    model: str
    model_version: str | None = None
    transcription: str
    wer: float | None = None
    cer: float | None = None
    rt_time: float | None = None
    rtf: float | None = None
    audio_duration: float | None = None
    output_file: str | None = None


class RunCreateRequest(ApiModel):
    name: str | None = None
    reference_text: str
    audio_filename: str | None = None
    results: list[RunResultPayload]


class RunResponse(ApiModel):
    """Also the on-disk shape of ``run.json``."""

    id: str
    created_at: str
    name: str | None = None
    reference_text: str
    audio_filename: str | None = None
    results: list[RunResultPayload]
