import os
import importlib

from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions


BUCKET_NAME = "magisterka-stt-marziel"
BUCKET_FOLDER = "transcriptions"

REGION = "europe-west3"
API_ENDPOINT = "europe-west3-speech.googleapis.com"
RECOGNIZER_ID = "magisterka-recognizer"


def ensure_requirements():
    missing = []

    try:
        importlib.import_module("google.cloud.speech_v2")
    except Exception:
        missing.append("google-cloud-speech")

    try:
        importlib.import_module("google.cloud.storage")
    except Exception:
        missing.append("google-cloud-storage")

    base_dir = os.path.dirname(__file__)

    creds_path = os.path.join(
        base_dir,
        "..",
        "credentials",
        "google_credentials.json",
    )

    if not os.path.exists(creds_path):
        missing.append(f"credentials file not found: {creds_path}")

    if missing:
        lines = ["Missing requirements:"]
        for m in missing:
            lines.append(" - " + m)

        lines.append("")
        lines.append(
            "Install packages: pip install google-cloud-speech google-cloud-storage"
        )

        raise RuntimeError("\n".join(lines))


def upload_to_bucket(audio_path: str, creds_path: str) -> str:
    credentials = service_account.Credentials.from_service_account_file(
        creds_path
    )

    storage_client = storage.Client(
        credentials=credentials,
        project=credentials.project_id,
    )

    file_name = os.path.basename(audio_path)
    blob_name = f"{BUCKET_FOLDER}/{file_name}"

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(audio_path)

    return f"gs://{BUCKET_NAME}/{blob_name}"


def transcribe_file(audio_path: str):

    ensure_requirements()

    base_dir = os.path.dirname(__file__)

    creds_path = os.path.join(
        base_dir,
        "..",
        "credentials",
        "google_credentials.json",
    )

    credentials = service_account.Credentials.from_service_account_file(
        creds_path
    )

    # 🔥 REGION-BASED CLIENT (KLUCZOWE)
    client = SpeechClient(
        credentials=credentials,
        client_options=ClientOptions(
            api_endpoint=API_ENDPOINT
        )
    )

    audio_uri = upload_to_bucket(audio_path, creds_path)

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),

        language_codes=["pl-PL"],

        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True
        ),

        model="chirp_3",  # OK w regionach EU/US/etc
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(
        uri=audio_uri
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=(
            f"projects/{credentials.project_id}"
            f"/locations/{REGION}/recognizers/{RECOGNIZER_ID}"
        ),
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig()
        ),
    )

    operation = client.batch_recognize(request=request)

    response = operation.result(timeout=3600)

    return response


if __name__ == "__main__":

    default_audio = os.path.join(
        os.path.dirname(__file__),
        "..",
        "inputs",
        "test1.mp3",
    )

    response = transcribe_file(default_audio)

    print("=== Transkrypcja ===")

    audio_uri = (
        f"gs://{BUCKET_NAME}/transcriptions/"
        f"{os.path.basename(default_audio)}"
    )

    for result in response.results[audio_uri].transcript.results:
        print(result.alternatives[0].transcript)