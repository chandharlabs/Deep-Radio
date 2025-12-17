import numpy as np
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time

# ------------------------
# SDR Parameters (match GRC)
# ------------------------
sample_rate = 8e6       # 8 MS/s
center_freq = 915.014e6 # 915.014 MHz
tx_gain = 60            # dB
symbol_rate = 80e3      # 80 kSym/s
sps = int(sample_rate / symbol_rate)  # Samples per symbol = 100

# ------------------------
# Your QPSK Input Data
# ------------------------
input_sequence = [0,1,2,3,3,3,3,3,2,2,2,2,2,2,2,3,3]
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)

# Map to QPSK symbols
symbols = symbol_table[input_sequence]

# Upsample (repeat each symbol sps times)
samples = np.repeat(symbols, sps).astype(np.complex64)

# ------------------------
# LimeSDR Setup via SoapySDR
# ------------------------
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)

# Setup stream
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

print("🔊 Transmitting QPSK sequence... Press Ctrl+C to stop.")

try:
    while True:
        sr = sdr.writeStream(tx_stream, [samples], len(samples))
        if sr.ret != len(samples):
            print("⚠️ Partial or failed write:", sr)
except KeyboardInterrupt:
    print("🛑 Transmission stopped by user.")

# Cleanup
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
