import argparse
import io
import importlib
import json
import re
import time
from datetime import datetime
import os
from typing import Any

try:
    import whisper
except Exception as e:
    raise ImportError(
        "Failed to import the 'whisper' library. "
        "On Windows the package named 'whisper' from PyPI is sometimes a different project that fails at import.\n"
        "Recommended fix: uninstall the wrong package and install OpenAI's Whisper or an alternative:\n"
        "  C:/.../python.exe -m pip uninstall -y whisper\n"
        "  C:/.../python.exe -m pip install -U openai-whisper\n"
        "Or consider 'faster-whisper' for better Windows support. Also ensure 'ffmpeg' is installed and on PATH.\n"
        f"Original import error: {e}"
    )


class LocalWhisperClient:
    def __init__(self, model_size="small"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> dict:
        start = time.perf_counter()

        result = self.model.transcribe(
            audio_path,
            language="pl",
            fp16=False,
            verbose=True
        )

        end = time.perf_counter()

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "rt_time": end - start
        }


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token

    token_path = os.path.join(os.path.dirname(__file__), "..", "credentials", "diarizationTokens.json")
    if not os.path.exists(token_path):
        return None

    try:
        with open(token_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
            return tokens.get("PYANNOTE_TOKEN") or None
    except Exception:
        return None


def _load_diarization_pipeline(token: str):
    try:
        pyannote_audio = importlib.import_module("pyannote.audio")
        Pipeline = getattr(pyannote_audio, "Pipeline")
    except Exception as e:
        raise ImportError(
            "pyannote.audio is required for diarization. Install with: pip install pyannote.audio"
        ) from e

    try:
        try:
            return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    except Exception as e:
        raise RuntimeError(
            "Failed to load diarization pipeline. Ensure you accepted model terms on Hugging Face "
            "and provided a valid token via --hf-token or backend/credentials/diarizationTokens.json."
        ) from e


def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _assign_speakers_to_asr_segments(
    asr_segments: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for seg in asr_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()

        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0.0
        for dseg in diarization_segments:
            overlap = _segment_overlap(start, end, dseg["start"], dseg["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        aligned.append(
            {
                "start": start,
                "end": end,
                "speaker": best_speaker,
                "text": text,
            }
        )

    return aligned


def _merge_speaker_blocks(
    segments: list[dict[str, Any]],
    max_gap_seconds: float = 1.5,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    current = {
        "start": float(segments[0].get("start", 0.0)),
        "end": float(segments[0].get("end", 0.0)),
        "speaker": str(segments[0].get("speaker", "SPEAKER_UNKNOWN")),
        "text": str(segments[0].get("text", "")).strip(),
    }

    for segment in segments[1:]:
        speaker = str(segment.get("speaker", "SPEAKER_UNKNOWN"))
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        gap = max(0.0, start - float(current["end"]))

        same_speaker = speaker == current["speaker"]
        if same_speaker and gap <= max_gap_seconds:
            if text:
                if current["text"]:
                    current["text"] = f"{current['text']} {text}".strip()
                else:
                    current["text"] = text
            current["end"] = max(float(current["end"]), end)
            continue

        merged.append(current)
        current = {
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text,
        }

    merged.append(current)
    return merged


def _extract_diarization_segments(diarization: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    def _segments_from_rttm_writer(obj: Any) -> list[dict[str, Any]]:
        if not hasattr(obj, "write_rttm"):
            return []

        parsed: list[dict[str, Any]] = []
        try:
            buffer = io.StringIO()
            obj.write_rttm(buffer)
            buffer.seek(0)
            for line in buffer.readlines():
                line = line.strip()
                if not line or not line.startswith("SPEAKER"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                parsed.append(
                    {
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker,
                    }
                )
        except Exception:
            return []

        return parsed

    # Prefer the official RTTM export when available (as in pyannote docs).
    segments.extend(_segments_from_rttm_writer(diarization))
    if segments:
        return segments

    # Some DiarizeOutput wrappers hold an inner object that can write RTTM.
    for field_name in ("speaker_diarization", "diarization", "annotation", "output"):
        inner = getattr(diarization, field_name, None)
        if inner is None and isinstance(diarization, dict):
            inner = diarization.get(field_name)
        if inner is None:
            continue
        inner_segments = _segments_from_rttm_writer(inner)
        if inner_segments:
            return inner_segments

    def _as_mapping(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return vars(value)
            except Exception:
                pass
        return value

    def _segment_from_item(item: Any) -> dict[str, Any] | None:
        item = _as_mapping(item)

        if isinstance(item, (list, tuple)) and len(item) >= 3:
            start_raw, end_raw, speaker_raw = item[0], item[1], item[2]
            try:
                return {
                    "start": float(start_raw),
                    "end": float(end_raw),
                    "speaker": str(speaker_raw),
                }
            except Exception:
                return None

        def _find_speaker_deep(value: Any) -> Any:
            value = _as_mapping(value)
            if isinstance(value, dict):
                direct = (
                    value.get("speaker")
                    or value.get("speaker_id")
                    or value.get("speakerId")
                    or value.get("speaker_label")
                    or value.get("speakerLabel")
                    or value.get("label")
                    or value.get("cluster")
                    or value.get("cluster_id")
                    or value.get("clusterId")
                    or value.get("name")
                )
                if isinstance(direct, str) and re.search(r"speaker|spk|cluster", direct, re.IGNORECASE):
                    return direct
                if isinstance(direct, str) and direct.strip():
                    return direct

                for nested in value.values():
                    nested_speaker = _find_speaker_deep(nested)
                    if nested_speaker not in (None, ""):
                        return nested_speaker

            if isinstance(value, list):
                for nested in value:
                    nested_speaker = _find_speaker_deep(nested)
                    if nested_speaker not in (None, ""):
                        return nested_speaker

            if isinstance(value, str) and re.search(r"speaker|spk|cluster", value, re.IGNORECASE):
                return value

            return None

        def _extract_speaker_from_dict(value: dict[str, Any]) -> Any:
            speaker = (
                value.get("speaker")
                or value.get("speaker_id")
                or value.get("speakerId")
                or value.get("speaker_label")
                or value.get("speakerLabel")
                or value.get("label")
                or value.get("cluster")
                or value.get("cluster_id")
                or value.get("clusterId")
                or value.get("id")
            )

            if isinstance(speaker, dict):
                speaker = speaker.get("id") or speaker.get("label") or speaker.get("name")

            if speaker in (None, ""):
                speaker = _find_speaker_deep(value)

            return speaker

        if isinstance(item, dict):
            start = (
                item.get("start")
                or item.get("begin")
                or item.get("start_time")
                or item.get("startTime")
                or item.get("t_start")
                or item.get("onset")
            )
            end = (
                item.get("end")
                or item.get("stop")
                or item.get("end_time")
                or item.get("endTime")
                or item.get("t_end")
                or item.get("offset")
            )

            if (start is None or end is None) and isinstance(item.get("timestamp"), (list, tuple)) and len(item.get("timestamp")) >= 2:
                start = start if start is not None else item.get("timestamp")[0]
                end = end if end is not None else item.get("timestamp")[1]

            if (start is None or end is None) and isinstance(item.get("times"), (list, tuple)) and len(item.get("times")) >= 2:
                start = start if start is not None else item.get("times")[0]
                end = end if end is not None else item.get("times")[1]

            if (start is None or end is None) and isinstance(item.get("segment"), dict):
                segment = item.get("segment", {})
                start = (
                    start
                    if start is not None
                    else segment.get("start")
                    or segment.get("begin")
                    or segment.get("start_time")
                    or segment.get("onset")
                )
                end = (
                    end
                    if end is not None
                    else segment.get("end")
                    or segment.get("stop")
                    or segment.get("end_time")
                    or segment.get("offset")
                )

            if (start is None or end is None) and isinstance(item.get("segment"), (list, tuple)) and len(item.get("segment")) >= 2:
                start = start if start is not None else item.get("segment")[0]
                end = end if end is not None else item.get("segment")[1]

            # If times are in milliseconds, normalize to seconds.
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > 10000:
                start = float(start) / 1000.0
                end = float(end) / 1000.0

            speaker = _extract_speaker_from_dict(item)
        else:
            start = (
                getattr(item, "start", None)
                or getattr(item, "begin", None)
                or getattr(item, "start_time", None)
                or getattr(item, "startTime", None)
                or getattr(item, "t_start", None)
                or getattr(item, "onset", None)
            )
            end = (
                getattr(item, "end", None)
                or getattr(item, "stop", None)
                or getattr(item, "end_time", None)
                or getattr(item, "endTime", None)
                or getattr(item, "t_end", None)
                or getattr(item, "offset", None)
            )
            if start is None or end is None:
                segment_obj = getattr(item, "segment", None)
                if segment_obj is not None:
                    start = (
                        start
                        if start is not None
                        else getattr(segment_obj, "start", None)
                        or getattr(segment_obj, "begin", None)
                        or getattr(segment_obj, "start_time", None)
                        or getattr(segment_obj, "onset", None)
                    )
                    end = (
                        end
                        if end is not None
                        else getattr(segment_obj, "end", None)
                        or getattr(segment_obj, "stop", None)
                        or getattr(segment_obj, "end_time", None)
                        or getattr(segment_obj, "offset", None)
                    )

            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > 10000:
                start = float(start) / 1000.0
                end = float(end) / 1000.0
            speaker = (
                getattr(item, "speaker", None)
                or getattr(item, "speaker_id", None)
                or getattr(item, "speakerId", None)
                or getattr(item, "speaker_label", None)
                or getattr(item, "speakerLabel", None)
                or getattr(item, "label", None)
                or getattr(item, "cluster", None)
                or getattr(item, "cluster_id", None)
                or getattr(item, "clusterId", None)
                or getattr(item, "id", None)
            )
            if speaker in (None, ""):
                speaker = _find_speaker_deep(item)

        if start is None or end is None or speaker in (None, ""):
            return None

        try:
            return {
                "start": float(start),
                "end": float(end),
                "speaker": str(speaker),
            }
        except Exception:
            return None

    def _collect_segments_deep(value: Any) -> list[dict[str, Any]]:
        value = _as_mapping(value)
        found: list[dict[str, Any]] = []

        if isinstance(value, list):
            for item in value:
                parsed = _segment_from_item(item)
                if parsed is not None:
                    found.append(parsed)
                else:
                    found.extend(_collect_segments_deep(item))
            return found

        if isinstance(value, dict):
            for nested in value.values():
                found.extend(_collect_segments_deep(nested))
            return found

        return found

    # Legacy pyannote Annotation output.
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )
        return segments

    # Newer SDK/outputs may expose diarization segments directly.
    candidate = None
    if hasattr(diarization, "speaker_diarization"):
        candidate = getattr(diarization, "speaker_diarization")
    elif hasattr(diarization, "diarization"):
        candidate = getattr(diarization, "diarization")
    elif hasattr(diarization, "segments"):
        candidate = getattr(diarization, "segments")
    elif isinstance(diarization, dict):
        candidate = (
            diarization.get("speaker_diarization")
            or diarization.get("diarization")
            or diarization.get("segments")
            or diarization.get("output")
        )

    if candidate is None:
        candidate = _as_mapping(diarization)

    if candidate is not None:
        segments.extend(_collect_segments_deep(candidate))

        if segments:
            return segments

    # Helpful diagnostics: we reached here because no speaker labels were found.
    debug_preview = _as_mapping(diarization)
    debug_keys = list(debug_preview.keys()) if isinstance(debug_preview, dict) else []
    raise TypeError(
        "Diarization output did not contain speaker labels. "
        f"Output type: {type(diarization).__name__}; preview type: {type(debug_preview).__name__}; "
        f"preview keys: {debug_keys}."
    )


def _run_diarization(
    audio_path: str,
    hf_token: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    pipeline = _load_diarization_pipeline(hf_token)

    kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        sf = importlib.import_module("soundfile")
        torch = importlib.import_module("torch")
    except Exception as e:
        raise ImportError(
            "Diarization requires 'soundfile' and 'torch' packages to load audio in-memory. "
            "Install with: pip install soundfile torch"
        ) from e

    # Load audio ourselves to bypass pyannote's torchcodec-based decoder on Windows.
    waveform_np, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform_np = waveform_np.T  # (time, channels) -> (channels, time)
    waveform = torch.from_numpy(waveform_np)

    diarization_input = {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }

    diarization = pipeline(diarization_input, **kwargs)
    return _extract_diarization_segments(diarization)


def _format_diarized_text(aligned_segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for seg in aligned_segments:
        text = seg["text"]
        if not text:
            continue
        lines.append(
            f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {text}"
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Whisper transcription with optional speaker diarization."
    )
    parser.add_argument(
        "audio_inputs",
        nargs="*",
        help="Path(s) to audio file(s). If empty, backend/inputs/test1.wav is used.",
    )
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="Whisper model size, e.g. small, medium, large-v3.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization and speaker-labeled output.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote pipeline.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--speaker-block-gap",
        type=float,
        default=1.5,
        help="Merge consecutive segments from the same speaker if silence gap is <= this many seconds.",
    )
    parser.add_argument(
        "--no-merge-speaker-blocks",
        action="store_true",
        help="Disable merging consecutive segments from the same speaker.",
    )
    return parser


def test_local_whisper():
    args = _build_arg_parser().parse_args()

    print(f"Dostepne modele: {whisper.available_models()}")
    client = LocalWhisperClient(model_size="large-v3")

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    print("=== TEST LOKALNEGO KLIENTA ===")
    audio_inputs = args.audio_inputs if args.audio_inputs else [os.path.join(base_dir, "inputs", "test1.wav")]

    hf_token = _resolve_token(args.hf_token)
    if args.diarize and not hf_token:
        print(
            "[WARN] Diarization requested, but no Hugging Face token was provided. "
            "Falling back to plain transcription."
        )

    missing_files = [audio for audio in audio_inputs if not audio or not os.path.exists(audio)]
    if missing_files:
        raise FileNotFoundError(
            "Audio file(s) not found: "
            f"{', '.join(missing_files)}. "
            "Provide valid path(s) as arguments or place the default file in backend/inputs/test1.wav."
        )

    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    total_rt_time = 0.0
    for idx, audio in enumerate(audio_inputs, start=1):
        print(f"[{idx}/{len(audio_inputs)}] Przetwarzanie: {audio}")
        result = client.transcribe(audio)
        print(f"Transkrypcja: {result}")

        total_rt_time += result["rt_time"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = os.path.splitext(os.path.basename(audio))[0]
        output_path = os.path.join(outputs_dir, f"transcription_{audio_name}_{timestamp}.txt")

        output_text = result["text"]
        diarized_segments: list[dict[str, Any]] = []
        diarization_succeeded = False
        if args.diarize:
            if not hf_token:
                print("[WARN] Skipping diarization and saving plain transcription.")
            else:
                print("Uruchamianie diarization...")
                try:
                    diarization_segments = _run_diarization(
                        audio_path=audio,
                        hf_token=hf_token,
                        min_speakers=args.min_speakers,
                        max_speakers=args.max_speakers,
                    )
                    diarization_labels = sorted({str(seg.get("speaker", "SPEAKER_UNKNOWN")) for seg in diarization_segments})
                    print(f"Diarization labels (raw): {diarization_labels}")
                    print(f"Diarization segments count: {len(diarization_segments)}")
                    print("Diarization sample (first 5):")
                    for sample in diarization_segments[:5]:
                        print(
                            f"  [{sample.get('start', 0.0):.2f}-{sample.get('end', 0.0):.2f}] "
                            f"{sample.get('speaker', 'SPEAKER_UNKNOWN')}"
                        )

                    diarized_segments = _assign_speakers_to_asr_segments(
                        asr_segments=result.get("segments", []),
                        diarization_segments=diarization_segments,
                    )
                    if args.no_merge_speaker_blocks:
                        print("Scalanie blokow mówców: WYŁĄCZONE")
                    else:
                        diarized_segments = _merge_speaker_blocks(
                            diarized_segments,
                            max_gap_seconds=args.speaker_block_gap,
                        )
                        print(
                            "Scalanie blokow mówców: WŁĄCZONE "
                            f"(max gap: {args.speaker_block_gap:.2f}s)"
                        )
                    output_text = _format_diarized_text(diarized_segments)
                    diarization_succeeded = True
                except Exception as diarization_error:
                    print(
                        "[WARN] Diarization failed; saving plain transcription instead. "
                        f"Reason: {diarization_error}"
                    )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        if args.diarize and diarization_succeeded:
            json_path = os.path.join(outputs_dir, f"transcription_{audio_name}_{timestamp}.json")
            payload = {
                "text": result["text"],
                "segments": diarized_segments,
                "model": args.model_size,
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(payload, jf, ensure_ascii=False, indent=2)
            print(f"Zapisano diarization JSON do: {json_path}")

        print(f"Zapisano transkrypcję do: {output_path}")
        print(f"Tekst: {result['text']}")
        print(f"Czas przetwarzania: {result['rt_time']:.2f} s")

    print(f"Przetworzono {len(audio_inputs)} plik(ów). Łączny czas przetwarzania: {total_rt_time:.2f} s")


if __name__ == "__main__":
    test_local_whisper()