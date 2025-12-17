import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import resample
import sounddevice as sd

# === CONFIGURATION ===
CENTER_FREQ = 700e6      # Match LimeSDR TX
SAMPLE_RATE = 2.4e6      # RTL-SDR max
AUDIO_RATE = 48000       # Audio playback rate
GAIN = 'auto'
CHUNK_SIZE = 256 * 10240  # Big chunk for smoother audio

# === INIT RTL-SDR ===
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

print(f"[INFO] Listening at {CENTER_FREQ / 1e6} MHz (AM mode)")

# === AUDIO INIT ===
sd.default.samplerate = AUDIO_RATE
sd.default.channels = 1

def am_demod(iq_data):
    envelope = np.abs(iq_data)
    audio = resample(envelope, int(AUDIO_RATE / SAMPLE_RATE * len(envelope)))
    return audio.astype(np.float32)

try:
    with sd.OutputStream(dtype='float32') as stream:
        while True:
            samples = sdr.read_samples(CHUNK_SIZE)
            audio = am_demod(samples)
            stream.write(audio)
except KeyboardInterrupt:
    print("\n[INFO] Stopping receiver.")
finally:
    sdr.close()



