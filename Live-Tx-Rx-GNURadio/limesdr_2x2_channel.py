import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
sample_rate = 1e6
center_freq = 2.7e9  # 2.4 GHz
tx_gain = 60
rx_gain = 60
pilot_len = 128
num_samples = 1024
num_iterations = 100
tracking_interval = 0.2  # seconds

# Generate orthogonal pilots
pilot_tx1 = np.random.choice([-1, 1], pilot_len)
pilot_tx2 = np.random.choice([-1, 1], pilot_len)

# LimeSDR Setup
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# Set sample rates
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setSampleRate(SOAPY_SDR_RX, 1, sample_rate)
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setSampleRate(SOAPY_SDR_TX, 1, sample_rate)

# Set frequencies
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setFrequency(SOAPY_SDR_RX, 1, center_freq)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setFrequency(SOAPY_SDR_TX, 1, center_freq)

# Set antennas
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")
sdr.setAntenna(SOAPY_SDR_TX, 1, "BAND2")
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")

# Set gains
sdr.setGain(SOAPY_SDR_RX, 0, rx_gain)
sdr.setGain(SOAPY_SDR_RX, 1, rx_gain)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setGain(SOAPY_SDR_TX, 1, tx_gain)

# Stream setup
rx_stream_0 = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
rx_stream_1 = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [1])
tx_stream_0 = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
tx_stream_1 = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [1])

sdr.activateStream(rx_stream_0)
sdr.activateStream(rx_stream_1)

# Tracking data
H_t = []

for i in range(num_iterations):
    print(f"[{i}] Estimating MIMO channel H(t)...")

    # --- TX1 pilot transmission ---
    sdr.activateStream(tx_stream_0)
    sdr.deactivateStream(tx_stream_1)

    tx_buf0 = pilot_tx1.astype(np.complex64)
    tx_buf1 = np.zeros_like(tx_buf0)

    sdr.writeStream(tx_stream_0, [tx_buf0], len(tx_buf0))
    sdr.writeStream(tx_stream_1, [tx_buf1], len(tx_buf1))

    time.sleep(0.01)

    rx_buf0 = np.zeros(num_samples, np.complex64)
    rx_buf1 = np.zeros(num_samples, np.complex64)

    sdr.readStream(rx_stream_0, [rx_buf0], num_samples)
    sdr.readStream(rx_stream_1, [rx_buf1], num_samples)

    h11 = np.correlate(rx_buf0, pilot_tx1, mode='valid')[0] / pilot_len
    h21 = np.correlate(rx_buf1, pilot_tx1, mode='valid')[0] / pilot_len

    # --- TX2 pilot transmission ---
    sdr.deactivateStream(tx_stream_0)
    sdr.activateStream(tx_stream_1)

    tx_buf0 = np.zeros_like(pilot_tx2).astype(np.complex64)
    tx_buf1 = pilot_tx2.astype(np.complex64)

    sdr.writeStream(tx_stream_0, [tx_buf0], len(tx_buf0))
    sdr.writeStream(tx_stream_1, [tx_buf1], len(tx_buf1))

    time.sleep(0.01)

    rx_buf0 = np.zeros(num_samples, np.complex64)
    rx_buf1 = np.zeros(num_samples, np.complex64)

    sdr.readStream(rx_stream_0, [rx_buf0], num_samples)
    sdr.readStream(rx_stream_1, [rx_buf1], num_samples)

    h12 = np.correlate(rx_buf0, pilot_tx2, mode='valid')[0] / pilot_len
    h22 = np.correlate(rx_buf1, pilot_tx2, mode='valid')[0] / pilot_len

    # Store channel matrix
    H = np.array([[h11, h12],
                  [h21, h22]])
    H_t.append(H)

    time.sleep(tracking_interval)

# Cleanup
sdr.deactivateStream(tx_stream_0)
sdr.deactivateStream(tx_stream_1)
sdr.closeStream(tx_stream_0)
sdr.closeStream(tx_stream_1)
sdr.closeStream(rx_stream_0)
sdr.closeStream(rx_stream_1)

# Plot channel tracking
H_array = np.array(H_t)
plt.figure(figsize=(12, 6))
for i in range(2):
    for j in range(2):
        plt.plot(np.abs(H_array[:, i, j]), label=f"|h{i+1}{j+1}|")
plt.xlabel("Time (samples)")
plt.ylabel("Magnitude")
plt.title("Time-Varying 2x2 MIMO Channel Tracking @ 2.4 GHz")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

