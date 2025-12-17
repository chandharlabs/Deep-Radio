import numpy as np
from rtlsdr import RtlSdr
import time

# ------------------------
# RTL-SDR Parameters
# ------------------------
sample_rate = 1e6
center_freq = 915.014e6
gain = 40.2 #'auto'
threshold = 0.2         # Amplitude threshold
detect_length = 1000    # Number of samples above threshold
buffer_len = 65536      # ~27 ms at 2.4 MS/s

# QPSK symbol table
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)

# ------------------------
# Setup RTL-SDR
# ------------------------
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = gain

print("📡 RTL-SDR listening at 915.014 MHz...")

try:
    while True:
        # Receive raw samples
        samples = sdr.read_samples(buffer_len)
        samples = np.array(samples)

        # Amplitude detection
        amplitude = np.abs(samples)
        above = amplitude > threshold

        if np.sum(above) >= detect_length:
            print("🚨 Transmission Detected!")

            # Optional: print portion of detected signal
            # Downsample: assuming symbol rate was 80kSym/s, and sample_rate = 2.4e6 → sps = 30
            sps = int(sample_rate / 80000)

            # Prevent out-of-range
            sps = max(sps, 1)
            downsampled = samples[::sps]

            # Demap
            def qpsk_demod(sym):
                distances = np.abs(sym - symbol_table)
                return np.argmin(distances)

            decoded = [qpsk_demod(s) for s in downsampled[:len(downsampled)//2]]  # limit to first symbols
            print("🧩 Decoded Symbol Sequence:", decoded)
        else:
            print("… idle")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
    sdr.close()
