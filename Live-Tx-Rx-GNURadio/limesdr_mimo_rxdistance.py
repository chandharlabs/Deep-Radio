import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.constants import speed_of_light
import time
import threading

# SDR Parameters
sample_rate = 5e6
center_freq = 915e6
tx_gain = 50
rx_gain = 60
num_samples = 81920
tx_tone_freq = 100e3

# Create SDR device
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# Configure TX
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")

# Configure RX1 (CH0)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")

# Configure RX2 (CH1)
sdr.setSampleRate(SOAPY_SDR_RX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 1, center_freq)
sdr.setGain(SOAPY_SDR_RX, 1, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")

# Create TX signal (sine wave)
t = np.arange(num_samples) / sample_rate
tx_waveform = 0.5 * np.exp(2j * np.pi * tx_tone_freq * t).astype(np.complex64)

# TX Stream setup
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

# Continuous TX thread
tx_running = True
def tx_loop():
    while tx_running:
        sr = sdr.writeStream(tx_stream, [tx_waveform], len(tx_waveform))
        if sr.ret != len(tx_waveform):
            print("TX write error:", sr)

tx_thread = threading.Thread(target=tx_loop)
tx_thread.start()

# Give TX some time to start
time.sleep(0.2)

# RX setup
rx_stream_0 = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
rx_stream_1 = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [1])
sdr.activateStream(rx_stream_0)
sdr.activateStream(rx_stream_1)

# Read samples
rx_buffer_0 = np.empty(num_samples, dtype=np.complex64)
rx_buffer_1 = np.empty(num_samples, dtype=np.complex64)
sr0 = sdr.readStream(rx_stream_0, [rx_buffer_0], num_samples)
sr1 = sdr.readStream(rx_stream_1, [rx_buffer_1], num_samples)

# Stop TX thread and streams
tx_running = False
tx_thread.join()
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
sdr.deactivateStream(rx_stream_0)
sdr.deactivateStream(rx_stream_1)
sdr.closeStream(rx_stream_0)
sdr.closeStream(rx_stream_1)

# Correlation for delay
correlation = correlate(rx_buffer_0, rx_buffer_1, mode='full')
lag = np.argmax(np.abs(correlation)) - (len(rx_buffer_0) - 1)
time_delay = lag / sample_rate
distance = abs(time_delay * speed_of_light)

print(f"Estimated sample lag: {lag}")
print(f"Time delay: {time_delay * 1e9:.2f} ns")
print(f"Estimated distance between RX1_H and RX2_H: {distance:.3f} meters")

# Plot TX + RX signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(np.real(tx_waveform), label="TX Signal")
plt.title("TX1_1 - Transmitted Sine Wave (Real)")
plt.grid(); plt.legend()

plt.subplot(3, 1, 2)
plt.plot(np.real(rx_buffer_0), label="RX1_H")
plt.title("RX1_H - Received Signal")
plt.grid(); plt.legend()

plt.subplot(3, 1, 3)
plt.plot(np.real(rx_buffer_1), label="RX2_H")
plt.title("RX2_H - Received Signal")
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()

