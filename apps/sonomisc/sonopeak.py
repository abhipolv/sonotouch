from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import socket
import struct
import time
import os
from sonoserver import TcpServer

# DEFAULT CONFIGURATION
# HOST = '192.168.50.56'  # IPv4 address, AJA
# HOST = '192.168.50.155'  # IPv4 address, ADV
PORT = 3333
CHANNELS = 2

# RECORDING SETUP
SAMPLERATE = 16000  # Hz
CLASS = "thumb"
recording_dir = os.path.join(os.path.dirname(__file__), "../data/recordings")
os.makedirs(recording_dir, exist_ok=True)
timestamp = datetime.datetime.now().current_time.strftime("%Y%m%d_%H%M%S")
recording_file = f"recordings/recording_{f}hz_{timestamp}.wav"

# CLASS DECLARATIONS:
class StreamBuffer: # Rolling buffer class to capture incoming data
    def __init__(self, size_max, channels=1, dtype=np.float32):
        self.size_max = size_max
        self.channels = channels
        self._data = np.zeros((size_max, channels), dtype=dtype)

    def append(self, value):
        num_samples = value.shape[0]
        self._data = np.roll(self._data, -num_samples, axis=0)
        self._data[-num_samples:, :] = value

    def get_data(self):
        return self._data

# HELPER FUNCTIONS:
def update(frame): # Function to update plot animation
    data = buffer.get_data()[:, 0]
    frequencies, times, Sxx = spectrogram(data, fs=SAMPLERATE, nperseg=2048)
    Sxx = 10 * np.log10(Sxx + 1e-10)
    image.set_data(Sxx)
    image.set_extent([times.min(), times.max(), frequencies.min(), frequencies.max()])
    image.set_clim(np.min(Sxx), np.max(Sxx))
    return [image]

def record(data, fs, mag_thresh=1000, width_thresh=0.1, debug=False): # Function to identify taps (peaks) and save to NPZ
    width = int(width_thresh*fs)

    # Identify peaks
    peaks, _ = signal.find_peaks(data, height=mag_thresh)
    classes = np.repeat([cl],peaks.size)
 
    # Segment data around peaks
    data = np.zeros((peaks.size, width*2))
    for i, p in enumerate(peaks):
        data[i,:] = data[(p-width):(p+width)]

        if debug:
            t = np.arange(data.size) / fs
            plt.plot(t[(p-width):(p+width)],data[i,:],label=i)
    if debug:
        plt.show()

    # Export as Numpy zip
    datetime = time.strftime("%Y%m%d_%H%M%S")
    np.savez("sonodata_"+cl+"_"+fs/1e3+"kHz_"+datetime, classes, data)

# - Function to save buffer to WAV on 'r' keypress
def on_press(event):
    if event.key == 'r':
        save_taps(mic_buffer,fs)
        print("Saved!")

# VISUALIZATION SETUP:

# - Initialize plot for live updating
fig, axs = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

f, t, stft = spectrogram(mic_buffer, fs)
mesh = axs.pcolormesh(t, f, stft, shading='nearest')

# - Start audio I/O stream:
mic_indata = []

ani = FuncAnimation(fig, update, interval=200, blit=True)

# RUN SERVER:
# - Start server
s = socket.socket()
s.bind((HOST, PORT))
s.listen()
print('Sever listening...')

# Connect & read from client
while True:
    client, addr = s.accept()
    data = client.recv(1024)
    print("Data received!")
    if len(data) > 0:
        data = struct.unpack('>f', data)[0]  # convert from binary -> float
        mic_indata.append(data)
    client.close()
    plt.show()


def main():
        ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)
        plt.show()

        print('Recording and playing... Press Ctrl+C to stop.')
        while True:
            sd.sleep(1000)

    except KeyboardInterrupt:
        print('')
        print('Recording interrupted by user.')

    finally:
        output_stream.stop()
        input_stream.stop()
        input_stream.close()
        output_stream.close()
        wavfile.write(recording_file, compromised_samplerate, buffer.get_data())

    print('')
    print(f'Audio saved in recordings/{recording_file}')

if __name__ == "__main__":
    main()
    
################################################################################
