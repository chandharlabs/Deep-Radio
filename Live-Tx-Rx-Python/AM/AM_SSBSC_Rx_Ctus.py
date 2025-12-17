mport numpy as np
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, resample
import sounddevice as sd

# === CONFIGURATION ===
CENTER_FREQ = 700e6       # SSB signal center
SAMPLE_RATE = 2.4e6       # RTL-SDR max
AUDIO_RATE = 48000        # Audio playback rate
SSB_OFFSET = 5000         # SSB offset in Hz (USB: +ve, LSB: -ve)
GAIN = 40.2 #'auto'
CHUNK_SIZE = 256 * 10240

# === SDR INIT ===
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

# === AUDIO INIT ===
sd.default.samplerate = AUDIO_RATE
sd.default.channels = 1

# === FILTER ===
def lowpass_filter(signal, cutoff=6000, fs=SAMPLE_RATE, order=6):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low')
    return lfilter(b, a, signal)

# === SSB Demodulation ===
def ssb_demod(iq_data, ssb_offset):
    n = len(iq_data)
    t = np.arange(n) / SAMPLE_RATE
    oscillator = np.exp(-1j * 2 * np.pi * ssb_offset * t)  # product detection
    mixed = iq_data * oscillator  # shift sideband to 0 Hz
    audio_baseband = np.real(lowpass_filter(mixed))       # extract audio
    audio = resample(audio_baseband, int(AUDIO_RATE / SAMPLE_RATE * len(audio_baseband)))
    return audio.astype(np.float32)

print(f"[INFO] Listening at {CENTER_FREQ / 1e6} MHz (SSB mode)")

try:
    with sd.OutputStream(dtype='float32') as stream:
        while True:
            samples = sdr.read_samples(CHUNK_SIZE)
            audio = ssb_demod(samples, ssb_offset=SSB_OFFSET)
            stream.write(audio)
except KeyboardInterrupt:
    print("\n[INFO] Stopping receiver.")
finally:
    sdr.close()

                
