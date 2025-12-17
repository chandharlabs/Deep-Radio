import numpy as np
import time
from scipy.io import wavfile
from scipy.signal import resample
from SoapySDR import Device, SOAPY_SDR_TX

# === CONFIGURATION ===
CENTER_FREQ = 100e6       # Transmit frequency (e.g. 100 MHz)
SAMPLE_RATE = 1e6         # SDR sample rate (1 MSPS)
TX_GAIN = 60              # Transmit gain
MODULATION_INDEX = 0.5    # AM modulation index
AMPLITUDE = 0.7           # Carrier amplitude
WAV_FILE = "sampled_audio_44100.wav"  # Replace with your actual file

# === SDR INIT ===
sdr = Device(dict(driver="lime"))
tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)
sdr.activateStream(tx_stream)

# === LOAD AUDIO FILE ===
fs, audio = wavfile.read(WAV_FILE)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # Convert to mono

audio = audio.astype(np.float32)
audio = audio / np.max(np.abs(audio))  # Normalize

# === RESAMPLE TO SDR RATE ===
num_samples = int(len(audio) * SAMPLE_RATE / fs)
audio_resampled = resample(audio, num_samples)

# === GENERATE AM SIGNAL ===
am_signal = AMPLITUDE * (1 + MODULATION_INDEX * audio_resampled)
am_signal = am_signal.astype(np.float32)
complex_signal = am_signal.astype(np.complex64)

# === TRANSMIT LOOP ===
print(f"[INFO] Transmitting audio on {CENTER_FREQ/1e6:.1f} MHz AM. Press Ctrl+C to stop.")
try:
    i = 0
    CHUNK = 1024
    while True:
        if i + CHUNK > len(complex_signal):
            i = 0  # Loop back to start
        chunk = complex_signal[i:i+CHUNK]
        sdr.writeStream(tx_stream, [chunk], len(chunk))
        i += CHUNK
except KeyboardInterrupt:
    print("\n[INFO] Stopping transmission.")
finally:
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)

