import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 50
rx_gain = 60
num_samples = 8192
pilot_len = 128

# SDR Setup
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# Setup TX channels (TX1_2 → ch 0, TX2_1 → ch 1)
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")

sdr.setSampleRate(SOAPY_SDR_TX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 1, center_freq)
sdr.setGain(SOAPY_SDR_TX, 1, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 1, "BAND2")

# Setup RX channels (RX1_H → ch 0, RX2_H → ch 1)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")

sdr.setSampleRate(SOAPY_SDR_RX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 1, center_freq)
sdr.setGain(SOAPY_SDR_RX, 1, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")

# Create pilot (impulse)
pilot = np.zeros(num_samples, dtype=np.complex64)
pilot[100:100+pilot_len] = 1.0

silent = np.zeros(num_samples, dtype=np.complex64)

# Create streams
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# --- Transmit Pilot from TX1_2 only ---
print("Transmitting from TX1_2 only...")
sdr.writeStream(tx_stream, [pilot, silent], num_samples)
time.sleep(0.1)
rx1a = np.zeros(num_samples, dtype=np.complex64)
rx2a = np.zeros(num_samples, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1a, rx2a], num_samples)

# --- Transmit Pilot from TX2_1 only ---
print("Transmitting from TX2_1 only...")
sdr.writeStream(tx_stream, [silent, pilot], num_samples)
time.sleep(0.1)
rx1b = np.zeros(num_samples, dtype=np.complex64)
rx2b = np.zeros(num_samples, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1b, rx2b], num_samples)

# Cleanup
sdr.deactivateStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(tx_stream)
sdr.closeStream(rx_stream)

# Estimate CSI via correlation
def estimate_cir(rx, pilot):
    return np.correlate(rx, pilot, mode='full')

h11 = estimate_cir(rx1a, pilot)  # TX1_2 → RX1_H
h21 = estimate_cir(rx2a, pilot)  # TX1_2 → RX2_H
h12 = estimate_cir(rx1b, pilot)  # TX2_1 → RX1_H
h22 = estimate_cir(rx2b, pilot)  # TX2_1 → RX2_H

# Plot CSI results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("h11: TX1_2 → RX1_H (Mag)")
plt.plot(np.abs(h11))
plt.grid()

plt.subplot(2, 2, 2)
plt.title("h12: TX2_1 → RX1_H (Mag)")
plt.plot(np.abs(h12))
plt.grid()

plt.subplot(2, 2, 3)
plt.title("h21: TX1_2 → RX2_H (Mag)")
plt.plot(np.abs(h21))
plt.grid()

plt.subplot(2, 2, 4)
plt.title("h22: TX2_1 → RX2_H (Mag)")
plt.plot(np.abs(h22))
plt.grid()

plt.tight_layout()
plt.show()

