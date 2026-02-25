import os
import json
import time
import uuid
from typing import Optional
import sys

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception as e:
    raise ImportError("boto3 not found. Install with: pip install boto3") from e

import requests


def _load_aws_credentials(cfg_path: str) -> dict:
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def transcribe_file(audio_path: str, bucket: Optional[str] = None, region: str = "eu-central-1") -> str:
    """Transcribe an audio file using Amazon Transcribe.

    Workflow:
    - Read optional credentials/config from backend/credentials/aws_credentials.json.
    - Upload the local audio file to the configured S3 bucket.
    - Start a Transcribe job and poll until completion.
    - Download the transcript JSON and return the recognized text.

    Returns empty string on failure.
    """
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "credentials/aws_credentials.json")
    cfg = _load_aws_credentials(cfg_path)

    aws_key = cfg.get("AWS_ACCESS_KEY_ID")
    aws_secret = cfg.get("AWS_SECRET_ACCESS_KEY")
    cfg_region = cfg.get("AWS_REGION")
    cfg_bucket = cfg.get("S3_BUCKET")

    region = cfg_region or region
    bucket = bucket or cfg_bucket

    if not bucket:
        raise ValueError("No S3 bucket configured. Set S3_BUCKET in credentials/aws_credentials.json or pass bucket argument")

    # Check that audio file exists early to provide clearer error messages
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return ""

    # Create clients (if credentials omitted boto3 will use env or ~/.aws)
    try:
        s3 = boto3.client("s3", region_name=region, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
        transcribe = boto3.client("transcribe", region_name=region, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    except (BotoCoreError, ClientError) as e:
        print("Failed to create AWS clients:", e)
        return ""

    # Upload file to S3
    filename = os.path.basename(audio_path)
    object_key = f"transcribe_uploads/{filename}"
    try:
        s3.upload_file(audio_path, bucket, object_key)
    except Exception as e:
        print("Failed to upload file to S3:", e)
        return ""

    s3_uri = f"s3://{bucket}/{object_key}"

    # Determine media format from file extension
    _, ext = os.path.splitext(filename)
    fmt = ext.lstrip(".").lower()
    if fmt == "m4a":
        fmt = "mp4"

    job_name = f"job-{uuid.uuid4()}"

    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat=fmt,
            LanguageCode="pl-PL",
        )
    except Exception as e:
        print("Failed to start Transcribe job:", e)
        return ""

    # Poll for job completion
    for _ in range(0, 120):  # up to ~10 minutes depending on file length
        try:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        except Exception as e:
            print("Error fetching transcription job status:", e)
            return ""

        job = status.get("TranscriptionJob", {})
        state = job.get("TranscriptionJobStatus")
        if state == "COMPLETED":
            transcript_uri = job.get("Transcript", {}).get("TranscriptFileUri")
            if not transcript_uri:
                print("No transcript URI returned by Transcribe")
                return ""
            try:
                r = requests.get(transcript_uri)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", {})
                transcripts = results.get("transcripts", [])
                if transcripts:
                    return transcripts[0].get("transcript", "")
                return ""
            except Exception as e:
                print("Failed to download or parse transcript:", e)
                return ""
        elif state in ("FAILED",):
            print("Transcription job failed")
            return ""

        time.sleep(5)

    print("Transcription job did not complete in time")
    return ""


if __name__ == "__main__":
    # Quick local test (adjust bucket/region in credentials/aws_credentials.json)
    # Accept a file path as first arg, otherwise default to test1.wav
    audio = sys.argv[1] if len(sys.argv) > 1 else "test1.wav"
    if not os.path.exists(audio):
        print(f"Audio file not found: {audio}")
    else:
        print(transcribe_file(audio))
