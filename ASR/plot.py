import wave
import matplotlib.pyplot as plt
import numpy as np

obj = wave.open('record.wav', 'rb')
sample_freq = obj.getframerate()
n_samples = obj.getnframes()
t_audio = n_samples / sample_freq
signal_wave = obj.readframes(-1)
obj.close()

print('t_audio =', t_audio)

signal_array = np.frombuffer(signal_wave, dtype=np.int16)
time_array = np.linspace(0, t_audio, num=n_samples)

plt.figure(figsize=(12, 6))
plt.plot(time_array, signal_array)
plt.title('Audio Signal Waveform')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.xlim(0, t_audio)
plt.grid()
plt.show()