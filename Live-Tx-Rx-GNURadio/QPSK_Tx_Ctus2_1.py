# TX_QPSK_LimeSDR_Base4Length.py
import numpy as np
import SoapySDR
from SoapySDR import *
import time
from scipy.signal import upfirdn

# --- Config ---
symbol_rate = 80e3
sample_rate = 8e6
center_freq = 915.014e6
tx_gain = 60
sps = int(sample_rate / symbol_rate)

# --- QPSK Mapping ---
symbol_table = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=np.complex64)

def int_to_base4(n):
    return [n // 4, n % 4]  # supports up to 15

# --- RRC Filter ---
def rrc_filter(beta=0.35, span=10):
    N = span * sps
    t = np.arange(-N//2, N//2 + 1) / sps
    denom = 1 - (4 * beta * t)**2
    h = np.sinc(t) * np.cos(np.pi * beta * t) / denom
    h[t == 0] = 1.0
    h[np.abs(4 * beta * t) == 1] = np.pi / 4 * np.sinc(1 / (2 * beta))
    h /= np.sqrt(np.sum(h**2))
    return h.astype(np.float32)

rrc_taps = rrc_filter()

# --- SDR Setup ---
sdr = SoapySDR.Device(dict(driver="lime"))
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

# --- Frame Format ---
preamble = [0, 1, 2, 3, 0, 1]

def create_frame(payload):
    length_symbols = int_to_base4(len(payload))
    full_msg = preamble + length_symbols + payload
    return symbol_table[full_msg]

# --- Main Loop ---
print("🔊 Transmitting. Press Ctrl+C to exit.")

try:
    while True:
        user_input = input("📥 Enter numbers (0–3 space-separated, max 15): ").strip()
        if not user_input:
            continue
        payload = list(map(int, user_input.split()))
        if any(p not in [0, 1, 2, 3] for p in payload):
            print("❌ Invalid symbol. Only 0,1,2,3 allowed.")
            continue
        if len(payload) > 15:
            print("❌ Max payload length is 15 symbols.")
            continue

        symbols = create_frame(payload)
        samples = upfirdn(rrc_taps, symbols, up=sps).astype(np.complex64)

        for _ in range(10):
            sdr.writeStream(tx_stream, [samples], len(samples))
            time.sleep(0.5)

except KeyboardInterrupt:
    print("🛑 Transmission ended.")
finally:
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)


