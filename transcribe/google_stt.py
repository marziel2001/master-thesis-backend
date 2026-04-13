import os
import importlib
import sys

try:
    from google.cloud import speech_v1 as speech
except Exception:
    speech = None


def ensure_requirements():
    missing = []
    try:
        importlib.import_module("google.cloud.speech_v1")
    except Exception:
        missing.append("google-cloud-speech")
    base_dir = os.path.dirname(__file__)
    creds_path = os.path.join(base_dir, "..", "credentials", "google_credentials.json")
    if not os.path.exists(creds_path):
        missing.append(f"credentials file not found: {creds_path}")
    if missing:
        lines = ["Missing requirements:"]
        for m in missing:
            lines.append(" - " + m)
        lines.append("")
        lines.append("Install packages: pip install google-cloud-speech")
        lines.append(f"Place credentials JSON at: {creds_path}")
        raise RuntimeError("\n".join(lines))


def transcribe_file(audio_path: str):
    ensure_requirements()
    base_dir = os.path.dirname(__file__)
    creds_path = os.path.join(base_dir, "..", "credentials", "google_credentials.json")
    client = speech.SpeechClient.from_service_account_file(creds_path)

    # Wczytanie pliku audio
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,  # dla m4a zmień na LINEAR16 po konwersji
        language_code="pl-PL",
        enable_automatic_punctuation=True,
        sample_rate_hertz=44100,  # dostosuj do swojego pliku audio
    )

    response = client.recognize(config=config, audio=audio)

    return response

if __name__ == "__main__":
    default_audio = os.path.join(os.path.dirname(__file__), "..", "inputs", "test1.mp3")
    response = transcribe_file(default_audio)

    print("=== Transkrypcja ===")
    for result in response.results:
        print(result.alternatives[0].transcript)
