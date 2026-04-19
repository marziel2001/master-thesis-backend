import time
from datetime import datetime
import os
import sys

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

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    print("=== TEST LOKALNEGO KLIENTA ===")
    # allow passing many audio paths; if none provided use default input
    audio_inputs = sys.argv[1:] if len(sys.argv) > 1 else [os.path.join(base_dir, "inputs", "test1.wav")]

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
        total_rt_time += result["rt_time"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = os.path.splitext(os.path.basename(audio))[0]
        output_path = os.path.join(outputs_dir, f"transcription_{audio_name}_{timestamp}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Zapisano transkrypcję do: {output_path}")
        print(f"Tekst: {result['text']}")
        print(f"Czas przetwarzania: {result['rt_time']:.2f} s")

    print(f"Przetworzono {len(audio_inputs)} plik(ów). Łączny czas przetwarzania: {total_rt_time:.2f} s")


if __name__ == "__main__":
    test_local_whisper()