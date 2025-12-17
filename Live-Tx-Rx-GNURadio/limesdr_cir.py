import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time

# --------------------------- CONFIG ---------------------------
SAMPLE_RATE = 1e6          # 1 MSps
CENTER_FREQ = 700e6        # 915 MHz ISM band
TX_GAIN = 50               # dB
RX_GAIN = 40               # dB
PULSE_LEN = 1024           # samples
RX_CAPTURE_LEN = 4096      # samples
# --------------------------------------------------------------

# -------------------- INITIALIZE SDR DEVICE --------------------
print("[INFO] Initializing LimeSDR...")
sdr = SoapySDR.Device(dict(driver="lime"))

# Set up TX
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)

# Set up RX
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_RX, 0, RX_GAIN)

# -------------------- GENERATE IMPULSE -------------------------
pulse = np.zeros(PULSE_LEN, dtype=np.complex64)
pulse[0] = 1 + 1j  # Impulse at start

# -------------------- TRANSMIT IMPULSE -------------------------
print("[INFO] Transmitting pulse...")
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)
sdr.writeStream(tx_stream, [pulse], len(pulse))
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)

time.sleep(0.1)  # short pause

# -------------------- RECEIVE SAMPLES --------------------------
print("[INFO] Receiving signal...")
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
sdr.activateStream(rx_stream)

rx_buf = np.empty(RX_CAPTURE_LEN, dtype=np.complex64)
sr = sdr.readStream(rx_stream, [rx_buf], RX_CAPTURE_LEN)
if sr.ret > 0:
    rx_data = rx_buf[:sr.ret]
    print(f"[INFO] Received {sr.ret} samples.")
else:
    print("[ERROR] Failed to receive samples.")
    rx_data = np.zeros(RX_CAPTURE_LEN, dtype=np.complex64)

sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

# -------------------- CHANNEL IMPULSE RESPONSE -----------------
print("[INFO] Calculating CIR...")
cir = correlate(rx_data, pulse, mode='full')
delay_axis = np.arange(-len(pulse)+1, len(rx_data))

# -------------------- PLOT CIR ---------------------------------
plt.figure(figsize=(12, 5))
plt.plot(delay_axis, np.abs(cir))
plt.title("Channel Impulse Response (Magnitude)")
plt.xlabel("Sample Delay")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time

# --------------------------- CONFIG ---------------------------
SAMPLE_RATE = 1e6          # 1 MSps
CENTER_FREQ = 915e6        # 915 MHz ISM band
TX_GAIN = 50               # dB
RX_GAIN = 40               # dB
PULSE_LEN = 1024           # samples
RX_CAPTURE_LEN = 4096      # samples
# --------------------------------------------------------------

# -------------------- INITIALIZE SDR DEVICE --------------------
print("[INFO] Initializing LimeSDR...")
sdr = SoapySDR.Device(dict(driver="lime"))

# Set up TX
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)

# Set up RX
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_RX, 0, RX_GAIN)

# -------------------- GENERATE IMPULSE -------------------------
pulse = np.zeros(PULSE_LEN, dtype=np.complex64)
pulse[0] = 1 + 1j  # Impulse at start

# -------------------- TRANSMIT IMPULSE -------------------------
print("[INFO] Transmitting pulse...")
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)
sdr.writeStream(tx_stream, [pulse], len(pulse))
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)

time.sleep(0.1)  # short pause

# -------------------- RECEIVE SAMPLES --------------------------
print("[INFO] Receiving signal...")
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
sdr.activateStream(rx_stream)

rx_buf = np.empty(RX_CAPTURE_LEN, dtype=np.complex64)
sr = sdr.readStream(rx_stream, [rx_buf], RX_CAPTURE_LEN)
if sr.ret > 0:
    rx_data = rx_buf[:sr.ret]
    print(f"[INFO] Received {sr.ret} samples.")
else:
    print("[ERROR] Failed to receive samples.")
    rx_data = np.zeros(RX_CAPTURE_LEN, dtype=np.complex64)

sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

# -------------------- CHANNEL IMPULSE RESPONSE -----------------
print("[INFO] Calculating CIR...")
cir = correlate(rx_data, pulse, mode='full')
delay_axis = np.arange(-len(pulse)+1, len(rx_data))

# -------------------- PLOT CIR ---------------------------------
plt.figure(figsize=(12, 5))
plt.plot(delay_axis, np.abs(cir))
plt.title("Channel Impulse Response (Magnitude)")
plt.xlabel("Sample Delay")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

