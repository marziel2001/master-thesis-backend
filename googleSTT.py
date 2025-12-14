# from pydub import AudioSegment
from google.cloud import speech_v1 as speech

def transcribe_file(audio_path: str):
    client = speech.SpeechClient.from_service_account_file("./magisterkastt-06b1d98aebd0.json")

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

    print("=== Transkrypcja ===")
    for result in response.results:
        print(result.alternatives[0].transcript)

    return response

if __name__ == "__main__":
    transcribe_file("test1.mp3")
