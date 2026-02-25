import json
import os
from openai import OpenAI

# Load API key from credentials JSON, fallback to OPENAI_API_KEY env var
creds_path = os.path.join(os.path.dirname(__file__), "credentials", "openai_credentials.json")
api_key = None
try:
    with open(creds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        api_key = data.get("openai_api_key") or data.get("api_key")
except FileNotFoundError:
    pass

api_key = api_key or os.environ.get("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = OpenAI()

audio_file = open("C:\\Users\\Marcel\\Desktop\\MAGISTERKA\\projekt\\backend\\test1.wav", "rb")

transcription = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file
)

print(transcription.text)