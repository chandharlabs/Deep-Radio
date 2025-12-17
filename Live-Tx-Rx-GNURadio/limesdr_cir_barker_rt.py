import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import SoapySDR
from SoapySDR import *
import time

# --------------------- CONFIG ---------------------
SAMPLE_RATE = 5e6           # 5 MSps for stability
CENTER_FREQ = 915e6         # ISM band
TX_GAIN = 60
RX_GAIN = 60
BARKER_SEQ = [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]  # Barker-11
PULSE_LEN = len(BARKER_SEQ)
RX_CAPTURE_LEN = 8192*10
# --------------------------------------------------

# ------------- SDR Setup (TX & RX) ----------------
sdr = SoapySDR.Device(dict(driver="lime"))

# TX setup
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)

# RX setup
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_RX, 0, RX_GAIN)

# ------------------- Signal Prep -------------------
pulse = np.array(BARKER_SEQ, dtype=np.complex64)
# Repeat pulse to ensure enough energy
tx_signal = np.tile(pulse, 50)

# -------------------- Transmit ---------------------
print("[TX] Sending Barker sequence...")
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)
sdr.writeStream(tx_stream, [tx_signal], len(tx_signal))
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)

time.sleep(0.05)

# -------------------- Receive ----------------------
print("[RX] Capturing response...")
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
sdr.activateStream(rx_stream)
rx_buf = np.empty(RX_CAPTURE_LEN, dtype=np.complex64)
sr = sdr.readStream(rx_stream, [rx_buf], RX_CAPTURE_LEN)
rx_data = rx_buf[:sr.ret] if sr.ret > 0 else np.zeros(RX_CAPTURE_LEN, dtype=np.complex64)
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

# -------------------- CIR Analysis -----------------
print("[PROCESS] Computing CIR via cross-correlation...")
cir = correlate(rx_data, pulse, mode='full')
delay_axis = np.arange(-len(pulse) + 1, len(rx_data))

# -------------------- Plot CIR ---------------------
plt.figure(figsize=(12, 5))
plt.plot(delay_axis, np.abs(cir))
plt.title("Channel Impulse Response (Barker Sequence, Magnitude)")
plt.xlabel("Sample Delay")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

