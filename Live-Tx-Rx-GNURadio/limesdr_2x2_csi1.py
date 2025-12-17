
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 60
rx_gain = 60
num_samples = 4096

# Generate pilot (simple impulses)
pilot_len = 128
pilot_tx1 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx2 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx1[200:200+pilot_len] = 1 # TX1_2 impulse
pilot_tx2[600:600+pilot_len] = 1  # TX2_1 impulse

# Initialize SDR
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

# Stream setup
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# Transmit pilot signals
sdr.writeStream(tx_stream, [pilot_tx1, pilot_tx2], num_samples)
time.sleep(0.1)

# Receive
rx1 = np.zeros(num_samples, dtype=np.complex64)
rx2 = np.zeros(num_samples, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1, rx2], num_samples)

# Stop and close streams
sdr.deactivateStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(tx_stream)
sdr.closeStream(rx_stream)

# Estimate CIRs using correlation
def estimate_cir(rx, tx):
    h = np.correlate(rx, tx, mode='full')
    # Normalize by pilot energy
    return h / 128 #np.linalg.norm(tx)

h11 = estimate_cir(rx1, pilot_tx1)  # TX1_2 → RX1_H
h12 = estimate_cir(rx1, pilot_tx2)  # TX2_1 → RX1_H
h21 = estimate_cir(rx2, pilot_tx1)  # TX1_2 → RX2_H
h22 = estimate_cir(rx2, pilot_tx2)  # TX2_1 → RX2_H


# -------------------------------
# ✅ Compute CSI as complex numbers (a + bj)
# Pick the peak correlation tap as CSI value
# -------------------------------

def extract_csis(rx, tx_pilot):
    h = np.correlate(rx, tx_pilot, mode='full') / 128 #np.linalg.norm(tx_pilot)
    # Find two largest peaks
    peak_indices = np.argpartition(np.abs(h), -2)[-2:]
    peak_indices = sorted(peak_indices)  # First = TX1, Second = TX2
    csi1 = h[peak_indices[0]]
    csi2 = h[peak_indices[1]]
    return csi1, csi2, peak_indices

# rx1 = RX1_H signal, rx2 = RX2_H signal (already captured from readStream)

csi_h11, csi_h12, idx1 = extract_csis(rx1, pilot_tx1)  # RX1_H: TX1_2, TX2_1
csi_h21, csi_h22, idx2 = extract_csis(rx2, pilot_tx2)  # RX2_H: TX1_2, TX2_1

print("Estimated Complex CSI:")
print(f"h11 (TX1_2 → RX1_H): {csi_h11:.4f}")
print(f"h12 (TX2_1 → RX1_H): {csi_h12:.4f}")
print(f"h21 (TX1_2 → RX2_H): {csi_h21:.4f}")
print(f"h22 (TX2_1 → RX2_H): {csi_h22:.4f}")


# Plot CSI and raw RX signals
plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
plt.title("h11: TX1_2 → RX1_H")
plt.plot(np.abs(h11[4000:]))
plt.grid()

plt.subplot(3, 2, 2)
plt.title("h12: TX2_1 → RX1_H")
plt.plot(np.abs(h12[4000:]))
plt.grid()

plt.subplot(3, 2, 3)
plt.title("h21: TX1_2 → RX2_H")
plt.plot(np.abs(h21[4000:]))
plt.grid()

plt.subplot(3, 2, 4)
plt.title("h22: TX2_1 → RX2_H")
plt.plot(np.abs(h22[4000:]))
plt.grid()

plt.subplot(3, 2, 5)
plt.title("RX1_H Raw Signal (|x|)")
plt.plot(np.abs(rx1))
plt.grid()

plt.subplot(3, 2, 6)
plt.title("RX2_H Raw Signal (|x|)")
plt.plot(np.abs(rx2))
plt.grid()

plt.tight_layout()
plt.show()

