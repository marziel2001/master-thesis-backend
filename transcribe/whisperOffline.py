import time
from datetime import datetime
import os

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
            "text": result["text"],
            "rt_time": end - start
        }


def test_local_whisper():
    print(f"Dostepne modele: {whisper.available_models()}")
    client = LocalWhisperClient(model_size="large-v3")

    import sys
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    # allow passing audio path as first argument, otherwise default to inputs/test1.wav
    audio = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "inputs", "test1.wav")
    if not audio or not os.path.exists(audio):
        raise FileNotFoundError(
            f"Audio file not found: {audio}. Provide a valid path as the first argument or place the file in the current directory."
        )

    result = client.transcribe(audio)

    print("=== TEST LOKALNEGO KLIENTA ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, f"transcription_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Zapisano transkrypcję do: {output_path}")
    print(f"Tekst: {result['text']}")
    print(f"Czas przetwarzania: {result['rt_time']:.2f} s")


if __name__ == "__main__":
    test_local_whisper()