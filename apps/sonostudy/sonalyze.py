import argparse
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import read

parser = argparse.ArgumentParser(description="Play and visualize a WAV audio file.")
parser.add_argument("wavfile", help="Path to the .wav file")
args = parser.parse_args()

print(f"Reading in .wav file: {args.wavfile}")
rate, data = read(args.wavfile)

if data.ndim > 1:
    data = np.mean(data, axis=1)

if np.issubdtype(data.dtype, np.integer):
    data = data / np.iinfo(data.dtype).max

print("Playing back...")
sd.play(data, rate)
sd.wait()

N = len(data)
freqs = fftfreq(N, d=1/rate)[:N//2]
fft_magnitude = np.abs(fft(data)[:N//2])

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

time = np.arange(N) / rate
axs[0].plot(time, data, color='black')
axs[0].set_title("Time-Domain Waveform")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

axs[1].plot(freqs, fft_magnitude, color='blue')
axs[1].set_title("FFT of the Waveform")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Magnitude")
axs[1].set_xlim([0, rate / 2])
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

axs[2].specgram(data, NFFT=1024, Fs=rate, cmap='inferno', noverlap=512)
axs[2].set_title("Spectrogram")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Frequency (Hz)")

plt.tight_layout()
plt.show()
