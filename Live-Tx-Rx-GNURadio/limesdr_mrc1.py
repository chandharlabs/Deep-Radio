import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
import time



# === SDR Parameters ===
sample_rate = 1e6
center_freq = 915e6
tx_gain = 50
rx_gain0 = 60
rx_gain1 = 60
num_samples = 4096
tone_freq = 5e3  # 50 kHz sine tone

# === Generate Sine Wave ===
t = np.arange(num_samples) / sample_rate
tx_waveform = 0.5 * np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)

# === Initialize LimeSDR USB ===
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

#print("Available RX antennas:")
#print(sdr.listAntennas(SOAPY_SDR_RX, 0))
#print(sdr.listAntennas(SOAPY_SDR_RX, 1))

# === TX Setup (TX0 → TX1_1) ===
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, tx_gain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND1")

# === RX Setup ===
# RX0 → RX1_H
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, rx_gain0)
sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAH")

# RX1 → RX2_H
sdr.setSampleRate(SOAPY_SDR_RX, 1, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 1, center_freq)
sdr.setGain(SOAPY_SDR_RX, 1, rx_gain1)
sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")

# === Optional Calibration ===
sdr.writeSetting("CALIBRATE", "1")

# === Setup Streams ===
txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])

# === Activate Streams ===
sdr.activateStream(txStream)
sdr.activateStream(rxStream)

# === Transmit in a Loop (Non-blocking) ===
print("Transmitting sine wave...")
for _ in range(100):
    sr = sdr.writeStream(txStream, [tx_waveform], len(tx_waveform))
    if sr.ret != len(tx_waveform):
        print(f"TX Warning: Sent {sr.ret} samples")

# === Wait for RX to catch signal ===
time.sleep(0.1)

# === Receive from RX0 and RX1 ===
rx0 = np.empty(num_samples, dtype=np.complex64)
rx1 = np.empty(num_samples, dtype=np.complex64)
sr = sdr.readStream(rxStream, [rx0, rx1], num_samples)
print(f"Received {sr.ret} samples")

# === Cleanup Streams ===
sdr.deactivateStream(txStream)
sdr.deactivateStream(rxStream)
sdr.closeStream(txStream)
sdr.closeStream(rxStream)

# === Power Measurements ===
pwr_rx0 = 10 * np.log10(np.mean(np.abs(rx0)**2) + 1e-12)
pwr_rx1 = 10 * np.log10(np.mean(np.abs(rx1)**2) + 1e-12)
print(f"RX0 (RX1_H) Power: {pwr_rx0:.2f} dB")
print(f"RX1 (RX2_H) Power: {pwr_rx1:.2f} dB")

# === MRC Combining (with phase alignment) ===
# Estimate phase offset between RX0 and RX1
phase_diff = np.angle(np.vdot(rx0, rx1))  # complex dot product
rx1_aligned = rx1 * np.exp(-1j * phase_diff)  # align phase to rx0
mrc = (rx0 + rx1_aligned) / 2

# === Plot TX, RX0, RX1, MRC ===
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.title("TX Signal (Real Part)")
plt.plot(np.real(tx_waveform))
plt.grid()

plt.subplot(4, 1, 2)
plt.title(f"RX0 (RX1_H) - Real Part | Power: {pwr_rx0:.2f} dB")
plt.plot(np.real(rx0))
plt.grid()

plt.subplot(4, 1, 3)
plt.title(f"RX1 (RX2_H) - Real Part | Power: {pwr_rx1:.2f} dB")
plt.plot(np.real(rx1))
plt.grid()

plt.subplot(4, 1, 4)
plt.title("MRC Combined Output - Real Part")
plt.plot(np.real(mrc))
plt.grid()

plt.tight_layout()
plt.show()

