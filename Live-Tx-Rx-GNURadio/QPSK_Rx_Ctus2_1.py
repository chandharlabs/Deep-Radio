# RX_QPSK_RTLSdr.py

import numpy as np
from rtlsdr import RtlSdr
import time
from scipy.signal import correlate, fir_filter_design, lfilter, upfirdn

# --- Config ---
sample_rate = 2.4e6
center_freq = 915.014e6
gain = 40.2
symbol_rate = 80e3
sps = int(sample_rate / symbol_rate)
buffer_len = 65536
threshold = 0.2
detect_length = 1000

symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)
preamble_indices = [0, 1, 2, 3, 0, 1]
preamble_symbols = symbol_table[preamble_indices]

# --- RRC Filter ---
def rrc_filter(beta=0.35, span=10):
    from scipy.signal import kaiser
    N = span * sps
    t = np.arange(-N//2, N//2 + 1)
    t = t / sps
    denom = 1 - (4*beta*t)**2
    h = np.sinc(t) * np.cos(np.pi * beta * t) / denom
    h[t == 0] = 1.0
    h[np.abs(4*beta*t) == 1] = np.pi/4 * np.sinc(1/(2*beta))
    h /= np.sqrt(np.sum(h**2))  # Normalize
    return h.astype(np.float32)

rrc_taps = rrc_filter()

# --- Demapper ---
def qpsk_demod(sym):
    distances = np.abs(sym - symbol_table)
    return np.argmin(distances)

# --- RTL-SDR Setup ---
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = gain

print("📡 RTL-SDR listening at 915.014 MHz...")

try:
    while True:
        raw = np.array(sdr.read_samples(buffer_len))
        amplitude = np.abs(raw)
        if np.sum(amplitude > threshold) < detect_length:
            print("… No signal")
            continue

        print("🚨 Transmission Detected!")
        filtered = lfilter(rrc_taps, 1.0, raw)

        # Downsample using slicing (fixed offset for now)
        symbols_rx = filtered[::sps]

        # Correlate with preamble
        corr = correlate(symbols_rx[:300], preamble_symbols)
        peak = np.argmax(np.abs(corr))
        print(f"🔍 Preamble peak at {peak}")

        start = peak + 1
        length_index = start + len(preamble_symbols)

        frame_len = qpsk_demod(symbols_rx[length_index])
        print(f"📦 Frame length: {frame_len}")

        payload_start = length_index + 1
        payload_syms = symbols_rx[payload_start : payload_start + frame_len]
        payload_syms /= np.abs(payload_syms)

        decoded = [qpsk_demod(sym) for sym in payload_syms]
        print("🧩 Decoded Payload:", decoded)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Receiver stopped.")
    sdr.close()
