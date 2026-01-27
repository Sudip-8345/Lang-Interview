import pyaudio
import wave

FRAMES_PER_BUFFER = 3200 # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16 bits per sample
CHANNELS = 1
RATE = 16000

P = pyaudio.PyAudio() # Create a PyAudio instance

stream = P.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER)

print("Recording...")  
seconds = 5
frames = []

# Record audio in chunks for the specified number of seconds
for _ in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)
    
print("Finished recording.")
stream.stop_stream()
stream.close()
P.terminate()

# Save the recorded data as a WAV file
wf = wave.open('record.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(P.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()