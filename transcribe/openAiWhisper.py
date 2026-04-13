import json
import os
from openai import OpenAI


def load_api_key() -> str | None:
    creds_path = os.path.join(os.path.dirname(__file__), "..", "credentials", "openai_credentials.json")
    api_key = None
    try:
        with open(creds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            api_key = data.get("openai_api_key") or data.get("api_key")
    except Exception:
        pass
    return api_key or os.environ.get("OPENAI_API_KEY")


def get_client() -> OpenAI:
    api_key = load_api_key()
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def transcribe_file(audio_path: str) -> str:
    client = get_client()
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
    return transcription.text


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    audio_path = os.path.join(base_dir, "inputs", "test1.wav")
    try:
        text = transcribe_file(audio_path)
        print(text)
    except Exception as e:
        print(f"Transcription error: {e}")