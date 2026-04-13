import os
import json

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
    config_path = os.path.join(os.path.dirname(__file__), "..", "credentials", "azure_credentials.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                key = cfg.get("AZURE_SPEECH_KEY") or cfg.get("key")
        except Exception:
            key = None
    region = "northeurope"
    if not key or not region:
        raise ValueError(
            "Set AZURE_SPEECH_KEY environment variable"
        )

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = "pl-PL"

    audio_input = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_input
    )

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return ""
    else:
        # Canceled or error
        return ""


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
