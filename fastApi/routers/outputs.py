from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastApi.schemas.outputs import UpdateOutputRequest, UpdateOutputResponse
from fastApi.services.outputs import (
    read_output_file,
    resolve_output_path,
    write_json,
)

router = APIRouter(prefix="/api", tags=["outputs"])


@router.post("/output/update", response_model=UpdateOutputResponse)
def update_output(payload: UpdateOutputRequest) -> UpdateOutputResponse:
    """Write recomputed metrics back into an existing output file."""
    try:
        output_path = resolve_output_path(payload.output_file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")

    try:
        data = read_output_file(output_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data["wer"] = payload.wer
    data["cer"] = payload.cer
    data["reference"] = payload.reference_text

    write_json(output_path, data)

    return UpdateOutputResponse(output_file=str(output_path))
