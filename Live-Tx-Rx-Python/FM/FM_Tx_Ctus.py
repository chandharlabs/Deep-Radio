import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from SoapySDR import Device, SOAPY_SDR_TX

# === CONFIGURATION ===
CENTER_FREQ = 700e6         # FM frequency (e.g., 100 MHz)
SAMPLE_RATE = 1e6           # SDR sample rate
AUDIO_FILE = "sampled_audio_44100.wav"
AUDIO_RATE = 44100
FM_DEVIATION = 75000        # Max frequency deviation (Hz)
GAIN = 60
CHUNK = 1024

# === SDR SETUP ===
sdr = Device(dict(driver="lime"))
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, GAIN)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND1")
tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
sdr.activateStream(tx_stream)

# === LOAD AUDIO ===
fs, audio = wavfile.read(AUDIO_FILE)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
audio = audio.astype(np.float32)
audio /= np.max(np.abs(audio))

# === RESAMPLE AUDIO TO SDR RATE ===
num_samples = int(len(audio) * SAMPLE_RATE / fs)
audio_resampled = resample(audio, num_samples)

# === INTEGRATE AUDIO → PHASE FOR FM MODULATION ===
k = 2 * np.pi * FM_DEVIATION / SAMPLE_RATE
phase = np.cumsum(audio_resampled) * k
fm_signal = np.exp(1j * phase).astype(np.complex64)

# === TRANSMIT ===
print(f"[INFO] Transmitting FM at {CENTER_FREQ/1e6:.1f} MHz... Press Ctrl+C to stop.")
try:
    i = 0
    while True:
        if i + CHUNK > len(fm_signal):
            i = 0
        chunk = fm_signal[i:i+CHUNK]
        sdr.writeStream(tx_stream, [chunk], len(chunk))
        i += CHUNK
except KeyboardInterrupt:
    print("\n[INFO] Transmission stopped.")
finally:
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)




