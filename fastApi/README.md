# FastAPI backend for the React transcription app

File-upload transcription, error-rate metrics, saved-run storage, and a
websocket endpoint that fans a live microphone feed out to every catalog
model at once.

## Layout

```
fastApi/
  main.py          app factory + router registration
  core/config.py   paths, CORS origin, ffmpeg settings
  routers/         one module per endpoint group
  schemas/         Pydantic request/response models
  services/        audio, metrics, output files, run storage, model dispatch,
                    live-transcription fan-out
  diff_html.py     word-alignment HTML renderer
  models.json      editable catalog of selectable models
transcribe/        one adapter per speech-to-text provider
scripts/           standalone text utilities
```

## Install

```powershell
pip install -r requirements.txt
```

That covers the API itself. Each provider SDK is imported lazily by its adapter,
so install only what you plan to call:

```powershell
pip install -r requirements-providers.txt
```

`ffmpeg` and `ffprobe` must be on `PATH`: ffmpeg converts m4a/aac/mp3 uploads to
WAV, and ffprobe reads audio duration (with a WAV-header fallback).

## Run

From the backend root:

```powershell
python -m uvicorn fastApi.main:app --reload
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

## Endpoints

| Method   | Path                        | Purpose                                        |
| -------- | --------------------------- | ---------------------------------------------- |
| `GET`    | `/health`                   | Liveness probe.                                |
| `GET`    | `/api/models`              | Contents of `models.json`.                     |
| `POST`   | `/api/transcribe/{model_name}`  | Transcribe an upload with one model.           |
| `POST`   | `/api/metrics`             | WER / CER for a reference and hypothesis.      |
| `POST`   | `/api/normalize-text`      | The normalised form used for metrics.          |
| `POST`   | `/api/diff-html`           | Word alignment as two HTML fragments.          |
| `GET`    | `/api/runs`                | List saved runs.                               |
| `GET`    | `/api/runs/{run_id}`       | One saved run.                                 |
| `POST`   | `/api/runs`                | Save a run and write its output files.         |
| `DELETE` | `/api/runs/{run_id}`       | Delete a run and its directory.                |
| `POST`   | `/api/output/update`       | Write recomputed metrics into an output file.   |
| `WS`     | `/ws/live-transcription`   | Live microphone feed, fanned out to every model. |

### `POST /api/transcribe/{model_name}`

`{model_name}` is a model id from `models.json` (`openai`, `whisperOffline`,
`whisperX`, `googleStt`, `azureStt`, `amazonStt`). Aliases are accepted: the id
is lower-cased and stripped of non-alphanumeric characters before lookup.

`multipart/form-data` fields:

| Field            | Required | Notes                                                     |
| ---------------- | -------- | --------------------------------------------------------- |
| `file`           | yes      | Audio file.                                               |
| `model_variant`  | no       | e.g. `large-v3`; defaults per model.                      |
| `whisper_model`  | no       | Legacy variant field for the Whisper models.              |
| `reference_text` | no       | When given, WER / CER are computed and returned.          |

### `WS /ws/live-transcription`

Every catalog model transcribes the same live microphone feed at once, each
at its own pace. There is no native low-latency streaming API wired up for
any provider here; instead, the client records short, independently
decodable audio chunks (a few seconds each) and sends one per binary frame.
The server converts each chunk to WAV and queues it for every model. A
model that falls behind (a slow local model, a provider whose adapter here
is job/polling-based rather than streaming, e.g. Amazon Transcribe or Google's
batch recognizer) has new chunks dropped for it rather than queued up, so it
stays "live" instead of replaying a growing backlog — expect it to visibly
lag or skip chunks compared to the faster providers.

Client → server:
- Binary frame: one chunk's raw audio bytes (the browser's `MediaRecorder`
  container format, e.g. webm/opus).
- Text frame `{"type": "stop"}`: ends the session; the server drains and
  closes the connection.

Server → client (JSON text frames):
- `{"type": "ready", "models": [...]}` — catalog ids taking part, sent once.
- `{"type": "chunk_received", "index": N}` — chunk N was decoded and queued.
- `{"type": "chunk_error", "index": N, "message": "..."}` — chunk N failed to
  decode (e.g. ffmpeg missing); that chunk is skipped for every model.
- `{"type": "chunk_skipped", "model": "...", "index": N}` — model fell behind
  and chunk N was dropped for it specifically.
- `{"type": "result", "model": "...", "index": N, "text": "...", "computeTime": 1.23}`
  — model finished transcribing chunk N.
- `{"type": "error", "model": "...", "index": N, "message": "..."}` — model
  failed on chunk N (e.g. missing credentials); other models are unaffected.

## Storage

- `outputs/` — one JSON file per transcription request.
- `outputs/runs/<run_id>/` — a saved run: `run.json` plus one file per result.

A run saved with a name is stored under a sanitised version of that name, so
saving twice under the same name **replaces** the earlier run. Unnamed runs get
a timestamp id.

## CORS

Only `http://localhost:5173` (the Vite dev server) is allowed by default. Set
the `FRONTEND_ORIGIN` environment variable to change it.

## Lint

```powershell
python -m ruff check .
```
