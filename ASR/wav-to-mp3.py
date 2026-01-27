from pydub import AudioSegment

AudioSegment.converter = r"C:\Users\Sudip Das\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\Sudip Das\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffprobe.exe"

# Load the WAV file
audio = AudioSegment.from_wav("record.wav")

audio = audio + 6
audio *= 2
audio = audio.fade_in(2000).fade_out(3000)
# Export as MP3
audio.export("output.mp3", format="mp3")

print("Conversion complete!")