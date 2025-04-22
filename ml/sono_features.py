import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# Import WAV files
fs, tap = wav.read("output.wav")
t = np.arange(tap.size) / fs

# Raw audio:
plt.figure()
plt.plot(t, tap)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Raw Audio")

# Fourier Transform:
tap_fft = fft(tap)
f = fs * np.arange(tap.size) / tap.size

plt.figure()
plt.plot(f, tap_fft)
plt.xlim((0, fs/2))
plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT Amplitude")
plt.title("FFT")

# Spectrogram:
f, t, tap_stft = signal.spectrogram(tap, fs)
tap_stft = 10*np.log10(np.abs(tap_stft))  # dB scale

plt.figure()
plt.pcolormesh(t, f, tap_stft)
# plt.ylim((0, 2000))
plt.colorbar(label='Power (dB)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram (STFT) of Recording")

# Amplitude Spectral Density:
f_psd, tap_psd = signal.welch(tap, fs)
tap_asd = np.sqrt(tap_psd)

plt.figure()
plt.plot(f_psd, tap_asd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("ASD")
plt.title("Amplitude Spectral Density")

# PCA: