import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from matplotlib.animation import FuncAnimation
import SoapySDR
from SoapySDR import *
import time

# ---------------- SDR + Signal Setup ----------------
SAMPLE_RATE = 5e6
CENTER_FREQ = 915e6
TX_GAIN = 60
RX_GAIN = 60
RX_CAPTURE_LEN = 8192
NUM_FRAMES = 50

BARKER_SEQ = [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
pulse = np.array(BARKER_SEQ, dtype=np.complex64)

# ---------------- SDR Init ----------------
sdr = SoapySDR.Device(dict(driver="lime"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_RX, 0, RX_GAIN)
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
sdr.activateStream(rx_stream)

# ---------------- Data Buffers ----------------
cir_matrix = []
rms_delays = []
frame_idx = []
pdp_latest = []
delay_axis = np.arange(-len(pulse)+1, RX_CAPTURE_LEN)

# ---------------- Plot Setup ----------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Heatmap
img = ax1.imshow(np.zeros((1, len(delay_axis))), aspect='auto', cmap='inferno',
                 extent=[delay_axis[0], delay_axis[-1], 0, 1], origin='lower', vmin=-120, vmax=-10)
ax1.set_title("CIR Heatmap Over Time")
ax1.set_ylabel("Frame Index")
ax1.set_xlabel("Sample Delay")
cbar = fig.colorbar(img, ax=ax1, label="Magnitude (dB)")

# RMS Delay Plot
rms_plot, = ax2.plot([], [], label="RMS Delay Spread (µs)")
ax2.set_xlim(0, NUM_FRAMES)
ax2.set_ylim(0, 10)
ax2.set_xlabel("Frame Index")
ax2.set_ylabel("Delay Spread (µs)")
ax2.grid()
ax2.legend()

# Power Delay Profile Plot
pdp_plot, = ax3.plot([], [], color='green')
ax3.set_title("Power Delay Profile (Current Frame)")
ax3.set_xlabel("Sample Delay")
ax3.set_ylabel("Power (dB)")
ax3.set_xlim(delay_axis[0], delay_axis[-1])
ax3.set_ylim(-120, 0)
ax3.grid()

# ---------------- Update Function ----------------
def update(frame):
    # Read RX samples
    rx_buf = np.empty(RX_CAPTURE_LEN, dtype=np.complex64)
    sr = sdr.readStream(rx_stream, [rx_buf], RX_CAPTURE_LEN)
    rx_data = rx_buf[:sr.ret] if sr.ret > 0 else np.zeros(RX_CAPTURE_LEN, dtype=np.complex64)

    # CIR and PDP
    cir = correlate(rx_data, pulse, mode='full')
    cir_mag = np.abs(cir)
    cir_matrix.append(cir_mag)
    pdp = cir_mag**2
    pdp_latest.clear()
    pdp_latest.extend(pdp)

    # RMS Delay Spread
    power = pdp / np.sum(pdp)
    delays = np.arange(len(pdp)) / SAMPLE_RATE
    mean_delay = np.sum(delays * power)
    rms = np.sqrt(np.sum((delays - mean_delay)**2 * power))
    rms_delays.append(rms * 1e6)  # µs

    # Update heatmap
    img.set_data(20 * np.log10(np.array(cir_matrix) + 1e-8))
    img.set_extent([delay_axis[0], delay_axis[-1], 0, len(cir_matrix)])

    # Update RMS plot
    rms_plot.set_data(np.arange(len(rms_delays)), rms_delays)
    ax2.set_xlim(0, len(rms_delays))
    ax2.set_ylim(0, max(1, np.max(rms_delays) * 1.2))

    # Update PDP
    pdp_plot.set_data(delay_axis, 10 * np.log10(np.array(pdp_latest) + 1e-12))

    return img, rms_plot, pdp_plot

# ---------------- Run Animation ----------------
ani = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=500, repeat=False)
plt.tight_layout()
plt.show()

# ---------------- Cleanup ----------------
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

