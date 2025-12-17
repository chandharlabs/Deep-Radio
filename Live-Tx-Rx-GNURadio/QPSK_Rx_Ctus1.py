import numpy as np
from rtlsdr import RtlSdr
import time

# ------------------------
# RTL-SDR Parameters
# ------------------------
sample_rate = 2.4e6           # Fixed by RTL-SDR
center_freq = 915.014e6
gain = 40.2 #'auto'
threshold = 0.2               # Amplitude threshold
detect_length = 1000          # Minimum samples above threshold
buffer_len = 65536            # ~27 ms of samples at 2.4 MS/s
sps = int(sample_rate / 80000)  # Samples per symbol (30)

# QPSK Constellation
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)

# ------------------------
# RTL-SDR Setup
# ------------------------
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = gain

print("📡 RTL-SDR receiver ready at 915.014 MHz")
print("🔍 Detecting transmission (amplitude > threshold)...")

try:
    while True:
        samples = sdr.read_samples(buffer_len)
        samples = np.array(samples)
        amplitude = np.abs(samples)

        # Basic transmission detection
        if np.sum(amplitude > threshold) >= detect_length:
            print("🚨 Transmission Detected!")

            # Downsample: 30 samples/symbol expected
            symbols_rx = samples[::sps]

            # Demap to QPSK symbols
            def qpsk_demod(symbol):
                distances = np.abs(symbol - symbol_table)
                return np.argmin(distances)

            decoded = [qpsk_demod(sym) for sym in symbols_rx[:len(symbols_rx)//2]]  # Limit output
            print("🧩 Decoded Symbols:", decoded)

        else:
            print("… No signal detected")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Receiver stopped by user.")
    sdr.close()
