import numpy as np
from rtlsdr import RtlSdr
import time

# ------------------------
# RTL-SDR Parameters
# ------------------------
sample_rate = 1e6           # Updated to 1 MSPS
center_freq = 915.014e6     # Same as TX
gain = 40.2                 # Adjust as needed
threshold = 0.2             # Amplitude threshold for detection
detect_length = 500         # Samples above threshold to trigger detection
buffer_len = 32768          # Shorter buffer for quicker response
sps = int(sample_rate / 100)  # 100 symbols/sec => 10,000 samples/symbol (too high!), so adjust

sps = int(sample_rate / 100)  # = 10,000 samples/symbol
downsample_factor = sps // 10  # We'll use 10 samples per symbol after decimation

# QPSK Constellation
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)

# ------------------------
# RTL-SDR Setup
# ------------------------
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = gain

print("📡 RTL-SDR listening at 915.014 MHz...")
print("🔍 Awaiting transmission...")

def qpsk_demod(symbol):
    distances = np.abs(symbol - symbol_table)
    return np.argmin(distances)

try:
    while True:
        samples = np.array(sdr.read_samples(buffer_len))
        amplitude = np.abs(samples)

        if np.sum(amplitude > threshold) >= detect_length:
            print("🚨 Transmission Detected!")

            # Decimate to roughly 10 samples per symbol
            symbols_rx = samples[::downsample_factor]

            # Demodulate first 100 symbols
            payload_symbols = symbols_rx[:5]

            decoded = [qpsk_demod(sym) for sym in payload_symbols]
            print("🧩 Decoded Symbols:", decoded)
        else:
            print("… No signal detected")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Receiver stopped.")
    sdr.close()
