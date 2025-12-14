from openai import OpenAI

client = OpenAI(api_key="")
audio_file = open("test1.m4a", "rb")

transcription = client.audio.transcriptions.create(
    file=audio_file,
    model="whisper-1"   # dostępny w OpenAI API
)

print("Transkrypcja:")
print(transcription.text)

