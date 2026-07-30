import json
import os
import threading

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception as e:
    raise ImportError(
        "Azure Speech SDK not found. Install with: pip install azure-cognitiveservices-speech"
    ) from e


def transcribe_file(audio_path: str):
    """Transcribe an audio file using Microsoft Azure Speech SDK.

    Expects environment variables: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION.
    Returns the recognized text (empty string on no match or error).
    """
    key = None
    region = None
    config_path = os.path.join(os.path.dirname(__file__), "..", "credentials", "azure_credentials.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
                key = cfg.get("AZURE_SPEECH_KEY") or cfg.get("key")
                region = cfg.get("AZURE_SPEECH_REGION") or cfg.get("region")
        except Exception:
            key = None
            region = None

    if not region:
        region = os.getenv("AZURE_SPEECH_REGION") or "northeurope"

    if not key or not region:
        raise ValueError(
            "Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in azure_credentials.json or environment variables"
        )

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = "pl-PL"

    audio_input = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_input
    )

    recognized_parts: list[str] = []
    done = threading.Event()
    failure: dict[str, str] = {}

    def _on_recognized(event) -> None:
        if event.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (event.result.text or "").strip()
            if text:
                recognized_parts.append(text)

    def _on_canceled(event) -> None:
        cancellation_details = speechsdk.CancellationDetails(event.result)
        if cancellation_details.reason == speechsdk.CancellationReason.EndOfStream:
            done.set()
            return
        failure["reason"] = str(cancellation_details.reason)
        failure["details"] = cancellation_details.error_details or ""
        done.set()

    def _on_session_stopped(_: object) -> None:
        done.set()

    recognizer.recognized.connect(_on_recognized)
    recognizer.canceled.connect(_on_canceled)
    recognizer.session_stopped.connect(_on_session_stopped)

    recognizer.start_continuous_recognition()
    done.wait()
    recognizer.stop_continuous_recognition()

    if failure:
        reason = failure.get("reason", "Unknown")
        details = failure.get("details", "")
        raise RuntimeError(
            "Azure recognition was canceled: "
            f"{reason}. {details}".strip()
        )

    transcript = " ".join(recognized_parts).strip()
    if transcript:
        return transcript

    raise RuntimeError("Azure returned no recognized speech for the full audio file.")


if __name__ == "__main__":
    # Hardcoded filename for quick testing
    audio_file = os.path.join(os.path.dirname(__file__), "..", "inputs", "test1.wav")
    print("Recognizing...")
    text = transcribe_file(audio_file)
    if text:
        print("=== Transkrypcja ===")
        print(text)
    else:
        print("No speech could be recognized or an error occurred.")
