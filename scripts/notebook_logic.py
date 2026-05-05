import json
import time
from pathlib import Path

from fastApi.diff_html import normalize_for_metrics
from scripts.count_metrics import calculate_metrics_from_text
from transcribe import amazon_stt, google_stt, ms_azure_stt, whisperOffline


def list_input_files(inputs_dir: Path, audio_exts: list[str]) -> tuple[list[str], list[str]]:
    audio_files = [f.name for f in inputs_dir.iterdir() if f.suffix.lower() in audio_exts]
    text_files = [f.name for f in inputs_dir.iterdir() if f.suffix.lower() == ".txt"]
    return audio_files, text_files


def resolve_audio_path(audio_path_value: str, audio_dropdown_value: str | None, inputs_dir: Path) -> str | None:
    if audio_path_value.strip():
        return audio_path_value.strip()
    if audio_dropdown_value and audio_dropdown_value != "--choose--":
        return str(inputs_dir / audio_dropdown_value)
    return None


def resolve_reference(
    ref_text_value: str,
    ref_dropdown_value: str | None,
    base_dir: Path,
    inputs_dir: Path,
) -> str | None:
    if ref_text_value.strip():
        temp_dir = base_dir / "temp_refs"
        temp_dir.mkdir(exist_ok=True)
        ref_path = temp_dir / f"ref_{int(time.time())}.txt"
        ref_path.write_text(ref_text_value, encoding="utf-8")
        return str(ref_path)
    if ref_dropdown_value and ref_dropdown_value != "--choose--":
        return str(inputs_dir / ref_dropdown_value)
    return None


def run_transcription_for_model(model_key: str, audio_path: str) -> str:
    try:
        if model_key == "amazon":
            return amazon_stt.transcribe_file(audio_path) or ""
        if model_key == "google":
            resp = google_stt.transcribe_file(audio_path)
            try:
                texts = [r.alternatives[0].transcript for r in resp.results]
                return " ".join(texts)
            except Exception:
                return ""
        if model_key == "azure":
            return ms_azure_stt.transcribe_file(audio_path) or ""
        if model_key == "openai_offline":
            client = whisperOffline.LocalWhisperClient(model_size="small")
            res = client.transcribe(audio_path)
            return res.get("text", "")
        if model_key == "openai_online":
            try:
                from openai import OpenAI

                client = OpenAI()
                with open(audio_path, "rb") as af:
                    transcription = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=af)
                return getattr(transcription, "text", "") or transcription.get("text", "")
            except Exception as err:
                print("OpenAI online transcription failed:", err)
                return ""
    except Exception as err:
        print(f"Error running model {model_key}:", err)
        return ""
    return ""


def normalize_text(text: str) -> str:
    return normalize_for_metrics(text)


def save_transcription(
    output_dir: Path,
    model_key: str,
    basename: str,
    transcription: str,
    compute_time: float,
) -> Path:
    out_path = output_dir / f"{model_key}_{basename}.json"
    payload = {
        "modelName": model_key,
        "modelVersion": model_key,
        "computeTime": compute_time,
        "transcription": transcription,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def compute_metrics(reference_text: str, transcription_text: str) -> tuple[float | None, float | None]:
    try:
        metrics = calculate_metrics_from_text(reference_text, transcription_text)
        return metrics["wer"], metrics["cer"]
    except Exception:
        return None, None


def execute_transcription_workflow(
    audio_path: str,
    reference_path: str,
    selected_models: list[str],
    output_dir: Path,
) -> dict:
    reference_text_raw = Path(reference_path).read_text(encoding="utf-8")
    reference_text = normalize_text(reference_text_raw)

    output_dir.mkdir(exist_ok=True)
    basename = Path(audio_path).stem
    results = {}

    for model_key in selected_models:
        started_at = time.perf_counter()
        text_raw = run_transcription_for_model(model_key, audio_path)
        compute_time = time.perf_counter() - started_at
        text = normalize_text(text_raw)
        save_transcription(output_dir, model_key, basename, text, compute_time)
        wer, cer = compute_metrics(reference_text, text)

        results[model_key] = {
            "text": text,
            "wer": wer,
            "cer": cer,
            "compute_time": compute_time,
        }

    return {
        "reference_text": reference_text,
        "basename": basename,
        "results": results,
    }
