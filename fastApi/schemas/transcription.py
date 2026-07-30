from __future__ import annotations

from fastApi.schemas.base import ApiModel


class TranscriptionResponse(ApiModel):
    #: Model id exactly as it appeared in the request path.
    requested_model: str
    #: Canonical model id the alias resolved to.
    model: str
    model_name: str
    model_version: str
    compute_time: float
    audio_duration: float | None = None
    filename: str
    transcription: str
    wer: float | None = None
    cer: float | None = None
    #: Duplicate of ``compute_time``; kept for the existing frontend contract.
    rt_time: float | None = None
    output_file: str | None = None
