import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
sample_rate = 1e6
center_freq = 915e6
tx_gain = 60
rx_gain = 60
num_samples = 4096
pilot_len = 128

# Generate PN sequence
def generate_pn_sequence(length, seed=1):
    np.random.seed(seed)
    seq = 2 * np.random.randint(0, 2, length) - 1  # 0/1 → -1/+1
    return seq.astype(np.complex64)

pilot_pn_tx1 = generate_pn_sequence(pilot_len, seed=1)
pilot_pn_tx2 = generate_pn_sequence(pilot_len, seed=2)

# Embed pilots in TX frame
pilot_tx1 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx2 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx1[300:300 + pilot_len] = pilot_pn_tx1  # TX1_2
pilot_tx2[800:800 + pilot_len] = pilot_pn_tx2  # TX2_1

# Initialize SDR
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# TX Configuration
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_TX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_TX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_TX, ch, tx_gain)
    sdr.setAntenna(SOAPY_SDR_TX, ch, "BAND2")

# RX Configuration
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_RX, ch, rx_gain)
    sdr.setAntenna(SOAPY_SDR_RX, ch, "LNAH")

# Stream setup
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# Transmit both pilots in same frame
sdr.writeStream(tx_stream, [pilot_tx1, pilot_tx2], num_samples)
time.sleep(0.1)

# Receive
rx1 = np.zeros(num_samples, dtype=np.complex64)
rx2 = np.zeros(num_samples, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1, rx2], num_samples)

# Cleanup
sdr.deactivateStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(tx_stream)
sdr.closeStream(rx_stream)

# Estimate CSI using correlation and normalization
def estimate_csi(rx, tx_pilot):
    h = np.correlate(rx, tx_pilot, mode='full')
    return h / np.sum(np.abs(tx_pilot)**2)

h11 = estimate_csi(rx1, pilot_pn_tx1)
h12 = estimate_csi(rx1, pilot_pn_tx2)
h21 = estimate_csi(rx2, pilot_pn_tx1)
h22 = estimate_csi(rx2, pilot_pn_tx2)

# Extract peak CSI values
def extract_csi(h):
    idx = np.argmax(np.abs(h))
    return h[idx], idx

csi_h11, i11 = extract_csi(h11)
csi_h12, i12 = extract_csi(h12)
csi_h21, i21 = extract_csi(h21)
csi_h22, i22 = extract_csi(h22)

# Print complex CSI
print("Estimated CSI values (PN sequence):")
print(f"h11 (TX1_2 → RX1_H): {csi_h11:.4f} @ index {i11}")
print(f"h12 (TX2_1 → RX1_H): {csi_h12:.4f} @ index {i12}")
print(f"h21 (TX1_2 → RX2_H): {csi_h21:.4f} @ index {i21}")
print(f"h22 (TX2_1 → RX2_H): {csi_h22:.4f} @ index {i22}")

# Plot
plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
plt.title("h11 (TX1_2 → RX1_H)")
plt.plot(np.abs(h11)); plt.grid()

plt.subplot(3, 2, 2)
plt.title("h12 (TX2_1 → RX1_H)")
plt.plot(np.abs(h12)); plt.grid()

plt.subplot(3, 2, 3)
plt.title("h21 (TX1_2 → RX2_H)")
plt.plot(np.abs(h21)); plt.grid()

plt.subplot(3, 2, 4)
plt.title("h22 (TX2_1 → RX2_H)")
plt.plot(np.abs(h22)); plt.grid()

plt.subplot(3, 2, 5)
plt.title("RX1 Magnitude")
plt.plot(np.abs(rx1)); plt.grid()

plt.subplot(3, 2, 6)
plt.title("RX2 Magnitude")
plt.plot(np.abs(rx2)); plt.grid()

plt.tight_layout()
plt.show()

