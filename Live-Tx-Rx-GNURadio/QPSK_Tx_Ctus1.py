import numpy as np
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time

# ------------------------
# SDR Parameters
# ------------------------
sample_rate = 8e6         # 8 MS/s
center_freq = 915.014e6   # 915.014 MHz
tx_gain = 60              # dB
symbol_rate = 80e3        # 80 kSym/s
sps = int(sample_rate / symbol_rate)  # Samples per symbol = 100

# ------------------------
# Ask User for QPSK Symbol Sequence
# ------------------------
print("🔢 Enter QPSK symbol sequence (0, 1, 2, 3 separated by spaces):")
user_input = input(">> ")

try:
    input_sequence = [int(x) for x in user_input.strip().split()]
    if not all(0 <= x <= 3 for x in input_sequence):
        raise ValueError("Only integers 0–3 are allowed.")
except Exception as e:
    print(f"❌ Invalid input: {e}")
    exit(1)

print(f"✅ Transmitting sequence: {input_sequence}")

# ------------------------
# QPSK Mapping
# ------------------------
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)
symbols = symbol_table[input_sequence]

# Upsample (repeat each symbol sps times)
samples = np.repeat(symbols, sps).astype(np.complex64)

# ------------------------
# LimeSDR Setup
# ------------------------
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)

tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

print("🚀 Transmitting QPSK stream continuously... (Ctrl+C to stop)")

# ------------------------
# Streaming
# ------------------------
try:
    while True:
        sr = sdr.writeStream(tx_stream, [samples], len(samples))
        if sr.ret != len(samples):
            print("⚠️ Partial or failed write:", sr)
except KeyboardInterrupt:
    print("🛑 Transmission stopped by user.")

# ------------------------
# Cleanup
# ------------------------
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
