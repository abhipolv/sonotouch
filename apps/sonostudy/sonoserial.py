import serial
import wave
import struct
import hashlib
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 921600
SAMPLE_RATE = 44100
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1
BUFF_IN_BYTES = 4096

timestamp = str(time.time()).encode()
h = hashlib.sha1(timestamp).hexdigest()[:8]
wav_filename = f"recording_{h}.wav"
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

print("Recording audio... Press Ctrl+C to stop.")

with wave.open(wav_filename, "wb") as wav_mic:
    wav_mic.setnchannels(NUM_CHANNELS)
    wav_mic.setsampwidth(BITS_PER_SAMPLE // 8)
    wav_mic.setframerate(SAMPLE_RATE)

    try:
        while True:
            raw_data = ser.read(BUFF_IN_BYTES)
            if len(raw_data) < 2:
                continue

            samples = struct.unpack("<" + "h" * (len(raw_data) // 2), raw_data)
            wav_mic.writeframes(struct.pack("<" + "h" * len(samples), *samples))

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

print(f"Saved audio to {wav_filename}")
