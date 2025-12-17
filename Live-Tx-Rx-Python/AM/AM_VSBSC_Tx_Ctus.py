import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, firwin, lfilter
from SoapySDR import Device, SOAPY_SDR_TX

# === CONFIG ===
CENTER_FREQ = 700e6         # LimeSDR TX center frequency
SAMPLE_RATE = 1e6           # LimeSDR sample rate
TX_GAIN = 60
AMPLITUDE = 0.5
WAV_FILE = "sampled_audio_44100.wav"
CHUNK = 1024

# === SDR INIT ===
sdr = Device(dict(driver="lime"))
tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)
sdr.activateStream(tx_stream)

# === LOAD AUDIO ===
fs, audio = wavfile.read(WAV_FILE)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
audio = audio.astype(np.float32)
audio /= np.max(np.abs(audio))

# === RESAMPLE AUDIO TO SDR RATE ===
num_samples = int(len(audio) * SAMPLE_RATE / fs)
audio_resampled = resample(audio, num_samples)

# === MODULATE TO IF (e.g., 10 kHz) ===
IF_FREQ = 10000  # Intermediate frequency
t = np.arange(len(audio_resampled)) / SAMPLE_RATE
complex_signal = AMPLITUDE * audio_resampled * np.exp(1j * 2 * np.pi * IF_FREQ * t)

# === VSB FILTER (bandpass) ===
# Keep 10 kHz to 15 kHz (upper sideband) + 3 kHz to 10 kHz (vestigial LSB)
bpf = firwin(129, [3000, 15000], pass_zero=False, fs=SAMPLE_RATE)
vsb_signal = lfilter(bpf, 1.0, complex_signal).astype(np.complex64)

# === TRANSMIT ===
print(f"[INFO] Transmitting VSB-SC at {CENTER_FREQ/1e6:.1f} MHz... Press Ctrl+C to stop.")
try:
    i = 0
    while True:
        if i + CHUNK > len(vsb_signal):
            i = 0
        chunk = vsb_signal[i:i+CHUNK]
        sdr.writeStream(tx_stream, [chunk], len(chunk))
        i += CHUNK
except KeyboardInterrupt:
    print("\n[INFO] Transmission stopped.")
finally:
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)



