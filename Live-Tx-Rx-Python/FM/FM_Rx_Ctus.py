import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import decimate
import sounddevice as sd

# === CONFIGURATION ===
CENTER_FREQ = 700e6
SAMPLE_RATE = 2.4e6
AUDIO_RATE = 48000
GAIN = 40.2 #'auto'
DECIM = int(SAMPLE_RATE // AUDIO_RATE)
CHUNK_SIZE = 256 * 1024

# === INIT SDR ===
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

sd.default.samplerate = AUDIO_RATE
sd.default.channels = 1

# === WBFM Demodulation ===
def fm_demod(iq):
    phase = np.angle(iq[1:] * np.conj(iq[:-1]))
    return phase

print(f"[INFO] Receiving FM from {CENTER_FREQ / 1e6} MHz...")

try:
    with sd.OutputStream(dtype='float32') as stream:
        while True:
            samples = sdr.read_samples(CHUNK_SIZE)
            demod = fm_demod(samples)
            audio = decimate(demod, DECIM)
            audio /= np.max(np.abs(audio) + 1e-6)
            stream.write(audio.astype(np.float32))
except KeyboardInterrupt:
    print("\n[INFO] Stopping receiver.")
finally:
    sdr.close()

