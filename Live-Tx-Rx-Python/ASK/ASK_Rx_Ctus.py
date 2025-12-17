import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt

# === CONFIG ===
CENTER_FREQ = 700e6      # RF carrier freq = same as LimeSDR TX
SAMPLE_RATE = 1e6        # 1 MHz sampling rate
GAIN = 40.2              # or 'auto'
BITRATE = 1000           # bits per second
CARRIER_FREQ = 10e3      # baseband tone in TX
CHUNK = 256 * 1024       # samples to read

# === SDR INIT ===
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

print(f"[INFO] Receiving ASK at {CENTER_FREQ/1e6:.1f} MHz, baseband tone {CARRIER_FREQ/1e3:.1f} kHz")

# === FIR BANDPASS ===
def bandpass(signal, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut/nyq, highcut/nyq], pass_zero=False)
    return lfilter(taps, 1.0, signal)

# === FIR LOWPASS ===
def lowpass(signal, cutoff, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, cutoff/nyq)
    return lfilter(taps, 1.0, signal)

# === RECEIVE + DEMODULATE ===
try:
    samples = sdr.read_samples(CHUNK)

    # 1) Bandpass to isolate 10 kHz tone (±3 kHz)
    bpf = bandpass(samples, CARRIER_FREQ - 3000, CARRIER_FREQ + 3000, SAMPLE_RATE)

    # 2) Rectifier: take magnitude (envelope)
    rectified = np.abs(bpf)

    # 3) Low-pass to smooth envelope (cutoff ~2*BITRATE)
    smoothed = lowpass(rectified, 2 * BITRATE, SAMPLE_RATE)

    # 4) Comparator: threshold to detect ON/OFF
    smoothed /= np.max(smoothed) + 1e-12  # normalize
    threshold = 0.3
    comparator = np.where(smoothed > threshold, 1, 0)

    # 5) Bit slicing: average over each bit time
    samples_per_bit = int(SAMPLE_RATE // BITRATE)
    bits = []
    for i in range(0, len(comparator) - samples_per_bit, samples_per_bit):
        avg = np.mean(comparator[i:i+samples_per_bit])
        bit = 1 if avg > 0.5 else 0
        bits.append(bit)

    print(f"[DETECTED BITS]: {''.join(map(str, bits))}")

    # === Plot ===
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(np.real(samples[:5000]))
    plt.title("Received IQ (Real Part)")

    plt.subplot(3, 1, 2)
    plt.plot(smoothed[:5000])
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Rectified & Low-pass Filtered Envelope")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(comparator[:5000])
    plt.title("Comparator Output (Raw Bits)")

    plt.tight_layout()
    plt.show()

except KeyboardInterrupt:
    print("\n[INFO] Reception stopped.")
finally:
    sdr.close()







