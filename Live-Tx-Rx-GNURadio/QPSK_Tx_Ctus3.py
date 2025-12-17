# ✅ Transmitter (TX) Script with 1 MSPS and 100 Symbols/sec
import numpy as np
import SoapySDR
from SoapySDR import *
import time

# SDR configuration
sample_rate = 1e6            # 1 MSPS
symbol_rate = 100            # 100 symbols/sec
sps = int(sample_rate / symbol_rate)
center_freq = 915.014e6
tx_gain = 60

# QPSK symbol map
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)

# Connect to LimeSDR
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)

tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)
time.sleep(0.1)

print("✅ SDR Ready @ 1 MSPS. Enter symbols 0–3 (Ctrl+C to stop):")

try:
    while True:
        user_input = input("\n🔢 Enter QPSK sequence: ")
        try:
            input_sequence = [int(x) for x in user_input.strip().split()]
            if not all(0 <= x <= 3 for x in input_sequence):
                raise ValueError("Only integers 0 to 3 allowed")
        except Exception as e:
            print(f"❌ Invalid input: {e}")
            continue

        symbols = symbol_table[input_sequence]
        samples = np.repeat(symbols, sps).astype(np.complex64)

        if len(samples) < 10000:
            samples = np.tile(samples, int(10000 / len(samples)))

        print("📡 Transmitting...")
        for _ in range(10):
            sr = sdr.writeStream(tx_stream, [samples], len(samples))
            if sr.ret != len(samples):
                print(f"⚠️ Incomplete TX ({sr.ret} of {len(samples)} samples)")
            time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
