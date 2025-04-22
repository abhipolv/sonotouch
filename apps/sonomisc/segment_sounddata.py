import os
import time
import hashlib
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Inputs
input_dir = os.getcwd() + "/data/abhi_04122025T0124_44100hz/"
output_dir = os.getcwd() + "/data/abhi_04122025T0124_44100hz_centered/"

# Parameters
mag_thresh = 1000  # naively chosen
width = 0.2  # seconds, naively chosen
debug = False

# Segment around maximum peak values
def save_tap(filename, mag_thresh=1000, width_thresh=0.2, debug=False):
    # Import WAV file
    path = input_dir+filename
    fs, rec = wav.read(path)
    t = np.arange(rec.size) / fs

    if debug:
        plt.figure()
        plt.plot(t,rec,'--k')
        plt.show()

    idx_ = filename.find('_')
    cl = int(filename[idx_+1:idx_+2]) - 1  # account for no neutral class
    cl = str(cl)

    # If neutral class, no peaks
    if cl == "-1":
        print("\tAborted: Neutral class (0)")
        return

    # Identify and extract peak
    width = int(width_thresh*fs)  
    idx_peaks, _ = signal.find_peaks(rec, height=mag_thresh)

    if len(idx_peaks) == 0:
        print("\tAborted: no peaks found.")
        return

    idx_max = idx_peaks[np.argmax(rec[idx_peaks])]
    peak = rec[(idx_max-width):(idx_max+width)]

    if peak.size == 0:
        print("\tAborted: peak cut off.")
        return

    # Format filename
    hash = hashlib.sha256(filename.encode()).hexdigest()
    outname = "recording_"+cl+"_"+hash+".wav"

    if debug:
        plt.figure()
        plt.plot(t[(idx_max-width):(idx_max+width)],peak,'--k')
        plt.show()

    # Save as WAV
    wav.write(output_dir+outname, fs, peak)

    # peaks, _ = signal.find_peaks(data, height=mag_thresh)
    # classes = np.repeat([cl],peaks.size)

    # data = np.zeros((peaks.size, width*2))
    # for i, p in enumerate(peaks):
    #     data[i,:] = data[(p-width):(p+width)]

    #     if debug:
    #         plt.plot(t[(p-width):(p+width)],data[i,:],label=i)

    # if debug:
    #     plt.show()

    # # Export as Numpy zip
    # datetime = time.strftime("%Y%m%d_%H%M%S")
    # np.savez(datetime+"_sonodata", classes, data)

# Run
for file in os.listdir(input_dir):
    print(file)
    save_tap(file, mag_thresh, width, debug=debug)
