import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import time

# Parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 60
rx_gain = 50
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

# Setup streams
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# Start live plot
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
line_rx0, = ax1.plot(np.zeros(num_samples), label="RX0")
line_rx1, = ax1.plot(np.zeros(num_samples), label="RX1")
ax1.set_ylim(-1, 1)
ax1.set_title("Live RX Signals")
ax1.legend()
ax1.grid()

corr_line, = ax2.plot(np.zeros(2 * num_samples - 1), label="Cross-Correlation")
lag_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)
ax2.set_title("Live Cross-Correlation RX0 vs RX1")
ax2.grid()
ax2.legend()

try:
    while True:
        # Transmit (looped)
        sdr.writeStream(tx_stream, [sine_wave], len(sine_wave))

        # Receive
        buff = np.empty((2, num_samples), dtype=np.complex64)
        sr = sdr.readStream(rx_stream, [buff[0], buff[1]], num_samples)
        rx0, rx1 = buff[0], buff[1]

        # Cross-correlation delay estimation
        corr = correlate(rx0, rx1, mode='full')
        lag = np.argmax(np.abs(corr)) - (len(rx0) - 1)
        time_delay = lag / sample_rate

        # Update plots
        line_rx0.set_ydata(np.real(rx0))
        line_rx1.set_ydata(np.real(rx1))
        corr_line.set_ydata(np.abs(corr))
        lag_text.set_text(f"Lag: {lag} samples\nDelay: {time_delay:.2e} s")
        ax2.set_ylim(0, np.max(np.abs(corr)) + 1)

        plt.pause(0.01)

except KeyboardInterrupt:
    print("Interrupted by user.")

# Cleanup
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)
plt.ioff()
plt.show()

