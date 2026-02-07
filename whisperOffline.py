import whisper
import time
from datetime import datetime

class LocalWhisperClient:
    def __init__(self, model_size="small"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> dict:
        start = time.perf_counter()

        result = self.model.transcribe(
            audio_path,
            language="pl",
            fp16=False
        )

        end = time.perf_counter()

        return {
            "text": result["text"],
            "rt_time": end - start
        }

def test_local_whisper():
    client = LocalWhisperClient(model_size="large-v3")

    result = client.transcribe("test1.wav")

    print("=== TEST LOKALNEGO KLIENTA ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"transcription_{timestamp}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Zapisano transkrypcję do: {output_path}")
    print(f"Tekst: {result['text']}")
    # print(f"Tekst: {result['text']}")
    print(f"Czas przetwarzania: {result['rt_time']:.2f} s")

if __name__ == "__main__":
    test_local_whisper()