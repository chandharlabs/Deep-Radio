import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
import time

# === SDR Params ===
sample_rate = 1e6         # 1 MHz
center_freq = 915e6       # 915 MHz
tx_gain = 50
rx_gain = 60
num_samples = 40960
tone_freq = 50e3          # 50 kHz sine wave

# === Generate Sine Wave for TX ===
t = np.arange(num_samples) / sample_rate
tx_waveform = 0.5 * np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)

# === Initialize LimeSDR ===
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# TX0 Setup
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND1")

# RX0 and RX1 Setup
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_RX, ch, rx_gain)
    sdr.setAntenna(SOAPY_SDR_RX, ch, "LNAW")

# === Setup Streams ===
txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])

# === Activate Streams ===
sdr.activateStream(rxStream)
sdr.activateStream(txStream)

time.sleep(0.1)  # Give time to settle

# === Transmit the Sine Wave ===
sr = sdr.writeStream(txStream, [tx_waveform], len(tx_waveform))
print(f"TX samples sent: {sr.ret}")

# === Receive on RX0 and RX1 ===
rx0 = np.empty(num_samples, dtype=np.complex64)
rx1 = np.empty(num_samples, dtype=np.complex64)
sr = sdr.readStream(rxStream, [rx0, rx1], num_samples)
print(f"RX samples received: {sr.ret}")

# === Cleanup ===
sdr.deactivateStream(txStream)
sdr.deactivateStream(rxStream)
sdr.closeStream(txStream)
sdr.closeStream(rxStream)

# === MRC Combining ===
# Phase-conjugate weighting (simplified MRC)
w0 = np.conj(rx0)
w1 = np.conj(rx1)
mrc = (w0 * rx0 + w1 * rx1) / (np.abs(w0)**2 + np.abs(w1)**2 + 1e-12)

# === Plot ===
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.title("RX0 (Antenna 0) - Real Part")
plt.plot(np.real(rx0))
plt.grid()

plt.subplot(4, 1, 2)
plt.title("RX1 (Antenna 1) - Real Part")
plt.plot(np.real(rx1))
plt.grid()

plt.subplot(4, 1, 3)
plt.title("MRC Combined Output - Real Part")
plt.plot(np.real(mrc))
plt.grid()

plt.subplot(4, 1, 4)
plt.title("Tx Waveform - Real Part")
plt.plot(np.real(tx_waveform))
plt.grid()



plt.tight_layout()
plt.show()
