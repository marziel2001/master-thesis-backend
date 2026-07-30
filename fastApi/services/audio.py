"""Audio probing, format conversion and temp-file lifecycle."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
import wave
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from fastApi.core.config import (
    CONVERTIBLE_AUDIO_SUFFIXES,
    TEMP_FILE_REMOVE_ATTEMPTS,
    TEMP_FILE_REMOVE_DELAY_SECONDS,
    WAV_CHANNELS,
    WAV_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


def _probe_duration_with_ffprobe(audio_path: Path) -> float | None:
    """Duration via ffprobe, or ``None`` if it is unavailable or fails."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("ffprobe is not available", exc_info=True)
        return None

    if result.returncode != 0:
        logger.debug("ffprobe exited with %s: %s", result.returncode, result.stderr)
        return None

    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        logger.debug("Could not parse ffprobe output: %s", result.stdout)
        return None

    return duration if duration > 0 else None


def _probe_duration_with_wave(audio_path: Path) -> float | None:
    """Duration via the stdlib WAV reader. Only works for real WAV files."""
    try:
        with closing(wave.open(str(audio_path), "rb")) as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                return None
            duration = handle.getnframes() / float(frame_rate)
    except (OSError, wave.Error):
        logger.debug("Could not read %s as WAV", audio_path, exc_info=True)
        return None

    return duration if duration > 0 else None


def get_audio_duration(audio_path: Path) -> float | None:
    """Duration in seconds, preferring ffprobe and falling back to WAV headers."""
    return _probe_duration_with_ffprobe(audio_path) or _probe_duration_with_wave(
        audio_path
    )


def needs_wav_conversion(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in CONVERTIBLE_AUDIO_SUFFIXES


def convert_to_wav(source_path: Path, target_path: Path) -> None:
    """Transcode to mono WAV. Raises ``RuntimeError`` when ffmpeg cannot."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ar",
                str(WAV_SAMPLE_RATE),
                "-ac",
                str(WAV_CHANNELS),
                str(target_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to convert m4a/mp3 files to wav."
        ) from exc

    if result.returncode != 0 or not target_path.exists():
        stderr = (result.stderr or "").strip()
        detail = f" ffmpeg output: {stderr}" if stderr else ""
        raise RuntimeError(f"Failed to convert audio file to wav.{detail}")


def remove_file_with_retry(path: Path) -> None:
    """Delete a temp file, retrying while another process still holds it."""
    for _ in range(TEMP_FILE_REMOVE_ATTEMPTS):
        if not path.exists():
            return
        try:
            path.unlink()
            return
        except PermissionError:
            time.sleep(TEMP_FILE_REMOVE_DELAY_SECONDS)
        except OSError:
            logger.warning("Could not remove temp file %s", path, exc_info=True)
            return

    logger.warning("Gave up removing temp file %s", path)


@contextmanager
def prepared_audio_file(data: bytes, file_name: str) -> Iterator[Path]:
    """Spill an upload to disk, converting to WAV when the format needs it.

    Yields the path the speech backends should read, and removes both the
    original and any converted file on the way out. This replaces the two
    duplicated retry loops that used to sit in the endpoint's ``finally``.
    """
    temp_path: Path | None = None
    converted_path: Path | None = None

    try:
        suffix = Path(file_name or "audio.bin").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)

        if not needs_wav_conversion(file_name):
            yield temp_path
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            converted_path = Path(wav_file.name)

        convert_to_wav(temp_path, converted_path)
        yield converted_path
    finally:
        for path in (converted_path, temp_path):
            if path is not None:
                remove_file_with_retry(path)
