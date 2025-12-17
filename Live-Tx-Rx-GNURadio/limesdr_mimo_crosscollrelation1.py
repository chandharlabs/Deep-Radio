import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from collections import deque
import time

# ==== Parameters ====
sample_rate = 60e6
center_freq = 915e6
tx_gain = 50
rx_gain = 60
n_samples = 32768
antenna_spacing = 4  # meters
c = 3e8
wavelength = c / center_freq
plot_window = 100
plot_every_n = 1
median_filter_len = 5
avg_window_len = 30

# ==== Generate PN Sequence for TX ====
def generate_pn_sequence(length=511):
    state = 0b1111111
    taps = [7, 1]
    seq = []
    for _ in range(length):
        bit = state & 1
        seq.append(1 if bit else -1)
        feedback = 0
        for t in taps:
            feedback ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (feedback << 6)
    return np.array(seq, dtype=np.float32)

pn_seq = generate_pn_sequence()
tx_signal = np.tile(np.concatenate([pn_seq, np.zeros(1000)]), 10).astype(np.complex64)

# ==== SDR Setup ====
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# TX Setup
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")  # TX1_2
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)

# RX Setup
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")  # RX1_H
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")  # RX2_H
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_RX, ch, rx_gain)

rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(rx_stream)

# ==== Buffers ====
buff1 = np.zeros(n_samples, np.complex64)
buff2 = np.zeros(n_samples, np.complex64)
buffers = [buff1, buff2]

# ==== Plot Setup ====
plt.ion()
fig, (ax_delay, ax_signal) = plt.subplots(2, 1, figsize=(12, 6))

# Delay plot
delay_series = []
avg_delay_series = []
line_delay, = ax_delay.plot([], [], label="Instantaneous Delay (samples)", color='blue')
line_avg_delay, = ax_delay.plot([], [], label="Smoothed Avg Delay", color='orange')
ax_delay.set_ylabel("Delay (samples)")
ax_delay.set_xlabel("Frame #")
ax_delay.grid(True)
ax_delay.legend()

# Signal plot
line_sig0, = ax_signal.plot([], [], label="Rx1 |mag|", alpha=0.7)
line_sig1, = ax_signal.plot([], [], label="Rx2 |mag|", alpha=0.7)
ax_signal.set_ylabel("Magnitude")
ax_signal.set_xlabel("Sample Index")
ax_signal.grid(True)
ax_signal.legend()

# ==== State ====
frame = 0
total_phase = 0
recent_lags = deque(maxlen=median_filter_len)
recent_avg = deque(maxlen=avg_window_len)

# ==== Main Loop ====
try:
    while True:
        # Transmit signal once
        sdr.activateStream(tx_stream)
        sdr.writeStream(tx_stream, [tx_signal], len(tx_signal))
        sdr.deactivateStream(tx_stream)

        sr = sdr.readStream(rx_stream, buffers, n_samples)
        if sr.ret != n_samples:
            print("Stream underrun or error. Skipping frame...")
            continue

        x0 = np.real(buff1)
        x1 = np.real(buff2)

        # Cross-correlation
        corr = correlate(x1, x0, mode='full')
        lags = np.arange(-len(x0) + 1, len(x0))
        lag_samples = lags[np.argmax(np.abs(corr))]
        recent_lags.append(lag_samples)
        filtered_lag = int(np.median(recent_lags))

        # Phase difference
        cross = buff2 * np.conj(buff1)
        mean_phase = np.angle(np.mean(cross))
        total_phase += mean_phase

        frame += 1
        recent_avg.append(filtered_lag)
        moving_avg_delay = np.mean(recent_avg)
        avg_delay_sec = moving_avg_delay / sample_rate

        # AoA estimation
        try:
            aoa_rad = np.arcsin((mean_phase * wavelength) / (2 * np.pi * antenna_spacing))
            aoa_deg = np.degrees(aoa_rad)
        except ValueError:
            aoa_deg = np.nan

        print(f"[Frame {frame:>4}] Delay: {filtered_lag:+3d} | Smoothed Avg: {moving_avg_delay:+.2f} "
              f"samples | Time: {avg_delay_sec * 1e9:.2f} ns | AoA: {aoa_deg:+.2f}°")

        # ==== Plot update ====
        if frame % plot_every_n == 0:
            delay_series.append(filtered_lag)
            avg_delay_series.append(moving_avg_delay)

            if len(delay_series) > plot_window:
                delay_series = delay_series[-plot_window:]
                avg_delay_series = avg_delay_series[-plot_window:]

            line_delay.set_xdata(np.arange(len(delay_series)))
            line_delay.set_ydata(delay_series)
            line_avg_delay.set_xdata(np.arange(len(avg_delay_series)))
            line_avg_delay.set_ydata(avg_delay_series)
            ax_delay.set_xlim(0, len(delay_series))
            ax_delay.set_ylim(min(delay_series + avg_delay_series) - 1,
                              max(delay_series + avg_delay_series) + 1)

            n_disp = 1024
            mag0 = np.abs(buff1[:n_disp])
            mag1 = np.abs(buff2[:n_disp])
            line_sig0.set_data(np.arange(n_disp), mag0)
            line_sig1.set_data(np.arange(n_disp), mag1)
            ax_signal.set_xlim(0, n_disp)
            ax_signal.set_ylim(0, max(np.max(mag0), np.max(mag1)) * 1.1)

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)
    sdr.closeStream(tx_stream)
    print("Streams closed.")
