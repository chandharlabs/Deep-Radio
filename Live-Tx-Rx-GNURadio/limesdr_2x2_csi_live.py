import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# SDR parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 50
rx_gain = 60
num_samples = 4096
pilot_len = 128

# Create pilot impulses
pilot_tx1 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx2 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx1[100:100+pilot_len] = 1.0
pilot_tx2[300:300+pilot_len] = 1.0

# Init SDR
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# TX Configuration
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")  # TX1_2

sdr.setSampleRate(SOAPY_SDR_TX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 1, center_freq)
sdr.setGain(SOAPY_SDR_TX, 1, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 1, "BAND2")  # TX2_1

# RX Configuration
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")  # RX1_H

sdr.setSampleRate(SOAPY_SDR_RX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 1, center_freq)
sdr.setGain(SOAPY_SDR_RX, 1, rx_gain)
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")  # RX2_H

# Setup streams
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# Create plots
fig, axs = plt.subplots(4, 2, figsize=(10, 8))
fig.suptitle("Live MIMO CSI - Amplitude and Phase (TX1_2, TX2_1 to RX1_H, RX2_H)")

lines = []
for ax in axs.flatten():
    line, = ax.plot([], [])
    lines.append(line)
    ax.grid(True)

# Correlation for CIR
def correlate(rx, tx):
    return np.correlate(rx, tx, mode='full')

# Plot updater
def update(frame):
    sdr.writeStream(tx_stream, [pilot_tx1, pilot_tx2], num_samples)
    time.sleep(0.01)

    rx1 = np.zeros(num_samples, dtype=np.complex64)
    rx2 = np.zeros(num_samples, dtype=np.complex64)
    sdr.readStream(rx_stream, [rx1, rx2], num_samples)

    # Estimate CIRs
    h11 = correlate(rx1, pilot_tx1)
    h12 = correlate(rx1, pilot_tx2)
    h21 = correlate(rx2, pilot_tx1)
    h22 = correlate(rx2, pilot_tx2)

    x = np.arange(len(h11))

    for i, (h, ax_row) in enumerate(zip([h11, h12, h21, h22], axs)):
        mag = np.abs(h)
        phase = np.angle(h)
        lines[2*i].set_data(x, mag)
        lines[2*i+1].set_data(x, phase)

        ax_row[0].set_xlim(0, len(h))
        ax_row[0].set_ylim(0, np.max(mag)*1.1)
        ax_row[0].set_title(f"h{i//2+1}{i%2+1} Magnitude")

        ax_row[1].set_xlim(0, len(h))
        ax_row[1].set_ylim(-np.pi, np.pi)
        ax_row[1].set_title(f"h{i//2+1}{i%2+1} Phase")

    return lines

# Start live animation
ani = FuncAnimation(fig, update, interval=500, blit=False)
plt.tight_layout()
plt.show()

# Optional cleanup after window closed
sdr.deactivateStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(tx_stream)
sdr.closeStream(rx_stream)

