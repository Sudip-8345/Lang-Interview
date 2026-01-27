import assemblyai as aai
from dotenv import load_dotenv
import os
load_dotenv()

base_url = "https://api.assemblyai.com"
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# upload file to AssemblyAI
audio_file = "./output.mp3"

# transcribe file using AssemblyAI
transcriber = aai.Transcriber()
transcript = transcriber.transcribe(audio_file)

# poll for transcription result
if transcript.status == aai.TranscriptStatus.error:
    print(f"Transcription failed: {transcript.error}")
    exit(1)

# save transcription to text file
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript.text)
print("\nTranscription saved to transcript.txt")
print(f"Transcription text:\n{transcript.text}")