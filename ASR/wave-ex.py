import wave

obj = wave.open('record.wav', 'rb')
print("Number of channels:", obj.getnchannels())
print("Sample width (in bytes):", obj.getsampwidth())
print("Frame rate (samples per second):", obj.getframerate())
print("Number of frames:", obj.getnframes())
t_audio = obj.getnframes() / obj.getframerate()
print("Duration (seconds):", t_audio)
print("Parameters:", obj.getparams())
obj.close()