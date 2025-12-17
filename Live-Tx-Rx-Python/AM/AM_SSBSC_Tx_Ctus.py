import numpy as np
import time
from scipy.io import wavfile
from scipy.signal import resample, firwin, lfilter
from SoapySDR import Device, SOAPY_SDR_TX

# === CONFIGURATION ===
CENTER_FREQ = 700e6         # Carrier frequency in Hz
SAMPLE_RATE = 1e6           # LimeSDR TX sample rate
TX_GAIN = 60                # TX gain
AUDIO_FILE = "sampled_audio_44100.wav"
INTERMEDIATE_FREQ = 5000    # Audio modulated to 5 kHz
AMPLITUDE = 0.5             # Output amplitude
CHUNK = 1024

# === SDR INIT ===
sdr = Device(dict(driver="lime"))
tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)
sdr.activateStream(tx_stream)

# === LOAD AND PROCESS AUDIO ===
fs, audio = wavfile.read(AUDIO_FILE)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # Convert to mono
audio = audio.astype(np.float32)
audio = audio / np.max(np.abs(audio))  # Normalize

# === RESAMPLE TO SDR RATE ===
num_samples = int(len(audio) * SAMPLE_RATE / fs)
audio_resampled = resample(audio, num_samples)

# === MODULATE AUDIO TO INTERMEDIATE FREQUENCY (e.g., 5 kHz) ===
t = np.arange(len(audio_resampled)) / SAMPLE_RATE
modulated = AMPLITUDE * audio_resampled * np.exp(1j * 2 * np.pi * INTERMEDIATE_FREQ * t)

# === FILTER OUT ONE SIDEBAND (LOW-PASS) ===
cutoff = 8000  # Retain only upper sideband (e.g., 0–8 kHz)
num_taps = 101
lpf = firwin(num_taps, cutoff / (SAMPLE_RATE / 2))  # Normalized
filtered_i = lfilter(lpf, 1.0, np.real(modulated))
filtered_q = lfilter(lpf, 1.0, np.imag(modulated))
ssb_signal = (filtered_i + 1j * filtered_q).astype(np.complex64)

# === TRANSMIT LOOP ===
print(f"[INFO] Transmitting SSB (no Hilbert) at {CENTER_FREQ/1e6:.1f} MHz... Press Ctrl+C to stop.")
try:
    i = 0
    while True:
        if i + CHUNK > len(ssb_signal):
            i = 0
        chunk = ssb_signal[i:i+CHUNK]
        sdr.writeStream(tx_stream, [chunk], len(chunk))
        i += CHUNK
except KeyboardInterrupt:
    print("\n[INFO] Transmission stopped.")
finally:
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)


