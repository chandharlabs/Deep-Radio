  GNU nano 6.2                                                                            Downloads/YesDR-BPSK-main/AM_VSBSC-Rx_Ctus.py                                                                                     
import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, resample
import sounddevice as sd

# === CONFIGURATION ===
CENTER_FREQ = 700e6
SAMPLE_RATE = 2.4e6
AUDIO_RATE = 48000
VSB_OFFSET = 10000  # Match transmitter IF
GAIN = 40.2 #'auto'
CHUNK_SIZE = 2* 256 * 10240

# === SDR INIT ===
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

sd.default.samplerate = AUDIO_RATE
sd.default.channels = 1

# === FILTER ===
def lowpass(signal, cutoff=15000, fs=SAMPLE_RATE, order=6):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low')
    return lfilter(b, a, signal)

# === VSB Demodulator ===
def vsb_demod(iq_data, offset):
    n = len(iq_data)
    t = np.arange(n) / SAMPLE_RATE
    oscillator = np.exp(-1j * 2 * np.pi * offset * t)
    mixed = iq_data * oscillator
    audio_baseband = np.real(lowpass(mixed))
    audio = resample(audio_baseband, int(AUDIO_RATE / SAMPLE_RATE * len(audio_baseband)))
    return audio.astype(np.float32)

print(f"[INFO] Receiving VSB-SC from {CENTER_FREQ/1e6} MHz...")
try:
    with sd.OutputStream(dtype='float32') as stream:
        while True:
            samples = sdr.read_samples(CHUNK_SIZE)
            audio = vsb_demod(samples, offset=VSB_OFFSET)
            stream.write(audio)
except KeyboardInterrupt:
    print("\n[INFO] Receiver stopped.")
finally:
    sdr.close()

