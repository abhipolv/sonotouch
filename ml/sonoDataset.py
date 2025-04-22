import torch
import numpy as np
import os
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt, spectrogram
from torch.utils.data import Dataset

# Custom PyTorch Dataset
class SonoDataset(Dataset):
    def __init__(self, path):
        self.wav_dir = path
        self.wav_labels = os.listdir(self.wav_dir)

    def __len__(self):
        return len(self.wav_labels)

    def extract_label(self, filename):
        # Return single character after first '_'
        idx_ = filename.find('_')
        return int(filename[idx_+1:idx_+2])
    
    def __getitem__(self, idx):
        # Load WAV file
        path = os.path.join(self.wav_dir, self.wav_labels[idx])
        fs, rec = read(path)
        label = self.extract_label(self.wav_labels[idx])

        # Convert to spectrogram
        # -- Normalize
        rec = rec.astype(np.float32)
        rec = rec / np.max(np.abs(rec))

        # -- Highpass filter out noise
        fc = 100
        order = 5
        b, a = butter(order, fc / (fs / 2), btype='high')
        rec = filtfilt(b, a, rec)

        # -- STFT
        window_size = 1024
        overlap = window_size - 128
        nfft = 1024

        f, t, Sxx = spectrogram(rec, fs, nperseg=window_size,
                                    noverlap=overlap, nfft=nfft, mode='complex')
        Sxx_dB = 20 * np.log10(Sxx + 1e-10)

        # Return as Torch tensor
        return torch.from_numpy(Sxx_dB.real), torch.tensor(label, dtype=torch.long)
    
