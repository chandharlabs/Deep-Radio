import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

import time

# Parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 50
rx_gain = 30
num_samples = 4096
tone_freq = 10e3

# Initialize SDR device
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# Setup TX
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)

# Setup RX0 and RX1
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_RX, ch, rx_gain)

# Generate sine wave
t = np.arange(num_samples) / sample_rate
sine_wave = 0.7 * np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)

# Setup TX stream
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

# Setup RX stream for channels 0 and 1
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(rx_stream)

# Transmit continuously in a separate thread or loop
sdr.writeStream(tx_stream, [sine_wave], len(sine_wave))

# Small delay to allow propagation
time.sleep(0.1)

# Receive samples
buff = np.empty((2, num_samples), dtype=np.complex64)
sr = sdr.readStream(rx_stream, [buff[0], buff[1]], num_samples)
rx0, rx1 = buff[0], buff[1]

# Deactivate and close streams
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

# Delay Estimation via Cross-correlation
corr = correlate(rx0, rx1, mode='full')
lag = np.argmax(np.abs(corr)) - (len(rx0) - 1)
time_delay = lag / sample_rate

print(f"Estimated Sample Delay (RX1 relative to RX0): {lag} samples")
print(f"Time Delay: {time_delay:.6e} seconds")

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.title("Received Signals")
plt.plot(np.real(rx0), label="RX0")
plt.plot(np.real(rx1), label="RX1")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.title("Cross-Correlation")
lags = np.arange(-len(rx0)+1, len(rx0))
plt.plot(lags, np.abs(corr))
plt.axvline(lag, color='r', linestyle='--', label=f"Lag: {lag}")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

