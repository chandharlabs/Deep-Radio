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
num_samples = 4096
pilot_len = 128

# Generate pilot (impulse)
pilot = np.zeros(num_samples, dtype=np.complex64)
pilot[300:300+pilot_len] = 1.0

silent = np.zeros_like(pilot)

# Initialize SDR
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# TX Configuration
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_TX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_TX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_TX, ch, tx_gain)
    sdr.setAntenna(SOAPY_SDR_TX, ch, "BAND2")

# RX Configuration
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_RX, ch, rx_gain)
    sdr.setAntenna(SOAPY_SDR_RX, ch, "LNAH")

# Stream setup
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# Step 1: TX1_2 only
sdr.writeStream(tx_stream, [pilot, silent], num_samples)
time.sleep(0.1)
rx1a = np.zeros(num_samples, dtype=np.complex64)
rx2a = np.zeros(num_samples, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1a, rx2a], num_samples)

# Step 2: TX2_1 only
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

# CIR estimation via correlation
def estimate_cir(rx, tx_pilot):
    return np.correlate(rx, tx_pilot, mode='full')

h11 = estimate_cir(rx1a, pilot)  # TX1_2 → RX1_H
h21 = estimate_cir(rx2a, pilot)  # TX1_2 → RX2_H
h12 = estimate_cir(rx1b, pilot)  # TX2_1 → RX1_H
h22 = estimate_cir(rx2b, pilot)  # TX2_1 → RX2_H

# Plot results
plt.figure(figsize=(14, 12))

# Channel impulse responses
plt.subplot(3, 2, 1)
plt.title("h11: TX1_2 → RX1_H")
plt.plot(np.abs(h11)); plt.grid()

plt.subplot(3, 2, 2)
plt.title("h12: TX2_1 → RX1_H")
plt.plot(np.abs(h12)); plt.grid()

plt.subplot(3, 2, 3)
plt.title("h21: TX1_2 → RX2_H")
plt.plot(np.abs(h21)); plt.grid()

plt.subplot(3, 2, 4)
plt.title("h22: TX2_1 → RX2_H")
plt.plot(np.abs(h22)); plt.grid()

# Raw RX signal magnitudes
plt.subplot(3, 2, 5)
plt.title("RX1_H Raw Signal (TX1_2 then TX2_1)")
plt.plot(np.abs(rx1a), label="From TX1_2")
plt.plot(np.abs(rx1b), label="From TX2_1", linestyle='--')
plt.legend(); plt.grid()

plt.subplot(3, 2, 6)
plt.title("RX2_H Raw Signal (TX1_2 then TX2_1)")
plt.plot(np.abs(rx2a), label="From TX1_2")
plt.plot(np.abs(rx2b), label="From TX2_1", linestyle='--')
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()

