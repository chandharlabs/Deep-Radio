import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 50
rx_gain = 60
num_samples = 4096

# Generate pilot (simple impulse)
pilot_len = 128
pilot_tx1 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx2 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx1[300:300+pilot_len] = 1.0  # TX1_2 impulse
pilot_tx2[500:500+pilot_len] = 1.0  # TX2_1 impulse

# Initialize SDR
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# --- TX Configuration ---
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")  # Your setting

sdr.setSampleRate(SOAPY_SDR_TX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 1, center_freq)
sdr.setGain(SOAPY_SDR_TX, 1, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 1, "BAND2")  # Your setting

# --- RX Configuration ---
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")  # RX1_H

sdr.setSampleRate(SOAPY_SDR_RX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 1, center_freq)
sdr.setGain(SOAPY_SDR_RX, 1, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")  # RX2_H

# Stream setup
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# Transmit pilots
sdr.writeStream(tx_stream, [pilot_tx1, pilot_tx2], num_samples)

time.sleep(0.1)

# Receive
rx1 = np.zeros(num_samples, dtype=np.complex64)
rx2 = np.zeros(num_samples, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1, rx2], num_samples)

# Stop streams
sdr.deactivateStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(tx_stream)
sdr.closeStream(rx_stream)

# Estimate CIRs using correlation
def estimate_cir(rx, tx_pilot):
    return np.correlate(rx, tx_pilot, mode='full')

h11 = estimate_cir(rx1, pilot_tx1)  # TX1_2 → RX1_H
h12 = estimate_cir(rx1, pilot_tx2)  # TX2_1 → RX1_H
h21 = estimate_cir(rx2, pilot_tx1)  # TX1_2 → RX2_H
h22 = estimate_cir(rx2, pilot_tx2)  # TX2_1 → RX2_H

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("h11 (TX1_2 → RX1_H)")
plt.plot(np.abs(h11)); plt.grid()

plt.subplot(2, 2, 2)
plt.title("h12 (TX2_1 → RX1_H)")
plt.plot(np.abs(h12)); plt.grid()

plt.subplot(2, 2, 3)
plt.title("h21 (TX1_2 → RX2_H)")
plt.plot(np.abs(h21)); plt.grid()

plt.subplot(2, 2, 4)
plt.title("h22 (TX2_1 → RX2_H)")
plt.plot(np.abs(h22)); plt.grid()

plt.tight_layout()
plt.show()
