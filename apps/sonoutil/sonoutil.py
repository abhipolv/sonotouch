from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram, butter, filtfilt
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import datetime
import hashlib
import socket
import struct
import torch
import os
import yaml
from threading import Event, Thread
from queue import Queue

from sonoModel import SonoModel

TCP_PORT = 3333
SAMPLERATE = 44100
DURATION = 1
SAMPLE_COUNT = 4096
SAMPLE_SIZE = 2
SAMPLE_BYTES = SAMPLE_COUNT * SAMPLE_SIZE
ANIM_INTERVAL = 5
TALKBACK = False
SOCKET_QUEUE = Queue()
INFERENCE_EN = False

CONFIG_PATH = "config.yaml"

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["user"], config["class"]

def generate_hash(data):
    return hashlib.sha1(data.tobytes()).hexdigest()[:8]

class TcpServer:
    def __init__(self, port, family_addr, timeout=60):
        self.port = port
        self.socket = socket.socket(family_addr, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(timeout)
        self.socket.bind(('192.168.50.155', self.port))
        self.socket.listen(1)
        print(f'Started server on port={self.port} family_addr={family_addr}')
        self.shutdown = Event()
        self.server_thread = Thread(target=self.run_server)
        self.server_thread.start()

    def run_server(self):
        while not self.shutdown.is_set():
            try:
                conn, _ = self.socket.accept()
                conn.setblocking(1)
                while not self.shutdown.is_set():
                    data = conn.recv(SAMPLE_BYTES)
                    if not data:
                        continue
                    SOCKET_QUEUE.put(data)
                    if TALKBACK:
                        first_sample = struct.unpack('<h', data[0:2])[0]
                        conn.send(f"OK {first_sample}".encode())
            except (socket.timeout, socket.error):
                continue

class SonoViz:
    def __init__(self, user, class_label):
        self.user = user
        self.class_label = class_label
        self.buffer = np.zeros(SAMPLERATE * DURATION, dtype=np.int16)
        self.record_count = 0

        self.recording_dir = self._create_recording_dir()
        self.shutdown = Event()

        self.fig, self.ax = plt.subplots()
        self.Sxx = np.zeros((1024, 1024))
        self.filt_param_b, self.filt_param_a = butter(5, 100 / (SAMPLERATE / 2), btype='high')
        
        self.img = self.ax.imshow(self.Sxx, extent=[0, 1, 0, SAMPLERATE/2], origin='lower', aspect='auto', cmap='jet')
        self.img.set_clim(vmin=-100, vmax=20)

        self.prediction = "neutral"
        self.classes = {
            0 : "neutral",
            1 : "index-thumb pinch",
            2 : "middle-thumb snap",
            3 : "index-thumb flick",
            4 : "middle-thumb pinch"
        }

        if INFERENCE_EN:
            DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.model = SonoModel().to(DEVICE)
            self.model.load_state_dict(torch.load("../../model/sono_model.pt", weights_only=True, map_location=torch.device('cpu')), strict=False)

        self._set_title()
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.cbar = plt.colorbar(self.img, ax=self.ax)
        self.cbar.set_label("Amplitude (dB)", rotation=90)
        self.fig.canvas.mpl_connect('key_press_event', self.save_recording)
        self.viz_engine_thread = Thread(target=self.process_data)
        self.viz_engine_thread.start()

    def _create_recording_dir(self):
        base = os.path.join(os.path.dirname(__file__), "../../data")
        timestamp = datetime.datetime.now().strftime("%m%d%YT%H%M")
        dirname = f"{self.user}_{timestamp}_{SAMPLERATE}hz"
        path = os.path.join(base, dirname)
        os.makedirs(path, exist_ok=True)
        return path

    def _set_title(self):
        title = f"User: {self.user} | Class: {self.class_label} | Recordings: {self.record_count} | Prediction: {self.prediction}"
        self.ax.set_title(title)

    def _interpret_buffer(self):
        out = self.model(torch.from_numpy(self.Sxx).unsqueeze(0).float())
        _, prediction = torch.max(out, 1)
        self.prediction = self.classes[prediction.item()]

    def process_data(self):
        while not self.shutdown.is_set():
            data = SOCKET_QUEUE.get()
            samples = np.frombuffer(data, dtype='<i2')
            self.buffer = np.roll(self.buffer, -len(samples))
            self.buffer[-len(samples):] = samples
            # self.buffer = filtfilt(self.filt_param_b, self.filt_param_a, self.buffer)

    def update_plot(self, frame):
        f, t, Sxx = spectrogram(self.buffer.astype(np.float32), fs=SAMPLERATE, nperseg=1024, noverlap=896, scaling='density', detrend=False)
        self.Sxx = 20 * np.log10(Sxx + 1e-10)
        if INFERENCE_EN:
            self._interpret_buffer()

        self.img.set_data(self.Sxx)
        self.img.set_extent([t.min(), t.max(), f.min(), f.max()])
        self._set_title()
        return (self.img,)

    def save_recording(self, event):
        if event.key != 'r':
            return
        buffer_copy = self.buffer.copy()
        hash_str = generate_hash(buffer_copy)
        filename = f"recording_{self.class_label}_{hash_str}.wav"
        path = os.path.join(self.recording_dir, filename)
        wav.write(path, SAMPLERATE, buffer_copy)
        self.record_count += 1
        print(f"[Saved] {filename}")

def main():
    user, class_label = load_config(CONFIG_PATH)
    viz = SonoViz(user, class_label)
    server = TcpServer(TCP_PORT, socket.AF_INET)
    ani = FuncAnimation(viz.fig, viz.update_plot, interval=ANIM_INTERVAL, blit=False)

    try:
        print("Server and Viz running. Press 'r' to save. Close the plot or Ctrl+C to stop.")
        plt.show()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        viz.shutdown.set()
        server.shutdown.set()
        viz.viz_engine_thread.join()
        server.server_thread.join()

if __name__ == '__main__':
    main()
