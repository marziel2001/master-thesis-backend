# FastAPI backend for React transcription app

## What this API supports
- File upload + transcription only (no live transcription).
- Uses your existing adapters from `transcribe/`.

## Endpoints
- `GET /health`
- `GET /api/models`
- `POST /api/transcribe`

## Request format for transcription
`multipart/form-data` fields:
- `file` (required): audio file
- `model` (required): one of `openai`, `whisper_offline`, `google`, `azure`, `amazon`
- `whisper_model` (optional): e.g. `small`, used only for `whisper_offline`

## Run
From backend root:

```powershell
pip install -r requirements.txt
uvicorn fastApi.main:app --reload --host 0.0.0.0 --port 8000
```

## CORS
By default, allowed frontend origin is `http://localhost:3000`.
Set env var `FRONTEND_ORIGIN` to change it.
