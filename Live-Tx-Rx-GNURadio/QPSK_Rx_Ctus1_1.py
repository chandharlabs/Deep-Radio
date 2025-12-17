import numpy as np
from rtlsdr import RtlSdr
import time

# ------------------------
# RTL-SDR Parameters
# ------------------------
sample_rate = 2.4e6
center_freq = 915.014e6
gain = 40.2
threshold = 0.2
detect_length = 1000
buffer_len = 65536
sps = int(sample_rate / 80000)  # 30 samples/symbol for 80kSym/s

# QPSK symbol map
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)
preamble_indices = [0, 1, 2, 3, 0, 1]
preamble_symbols = symbol_table[preamble_indices]

# ------------------------
# RTL-SDR Setup
# ------------------------
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = gain

print("📡 RTL-SDR listening at 915.014 MHz...")
print("🔍 Awaiting transmission (amplitude > threshold)...")

def qpsk_demod(symbol):
    distances = np.abs(symbol - symbol_table)
    return np.argmin(distances)

try:
    while True:
        samples = np.array(sdr.read_samples(buffer_len))
        amplitude = np.abs(samples)

        if np.sum(amplitude > threshold) >= detect_length:
            print("🚨 Transmission Detected!")

            # Downsample
            symbols_rx = samples[::sps]

            # Preamble correlation
            corr = np.correlate(symbols_rx[:200], preamble_symbols, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            print(f"🔍 Preamble found at index: {peak_idx}")

            # Get data after preamble
            data_start = peak_idx + len(preamble_symbols)
            payload_symbols = symbols_rx[data_start:data_start + 4]  # Adjust 100 as needed

            # Demodulate
            decoded = [qpsk_demod(sym) for sym in payload_symbols]
            print("🧩 Decoded Symbols:", decoded)

        else:
            print("… No signal detected")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
    sdr.close()
