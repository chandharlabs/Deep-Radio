import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
sample_rate = 10e6  # 10 MHz
center_freq = 915e6
tx_gain = 60
rx_gain = 60
num_samples = 8192
pilot_len = 128

# Generate PN sequence
def generate_pn_sequence(length, seed=1):
    np.random.seed(seed)
    seq = 2 * np.random.randint(0, 2, length) - 1  # 0/1 → -1/+1
    return seq.astype(np.complex64)

pilot_pn_tx1 = generate_pn_sequence(pilot_len, seed=1)
pilot_pn_tx2 = generate_pn_sequence(pilot_len, seed=2)

# Build TX frames with pilots at different offsets
pilot_tx1 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx2 = np.zeros(num_samples, dtype=np.complex64)
pilot_tx1[1000:1000 + pilot_len] = pilot_pn_tx1  # TX1_2
pilot_tx2[4000:4000 + pilot_len] = pilot_pn_tx2  # TX2_1

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

# Transmit both pilots in one shot
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

# CSI estimation
def estimate_csi(rx, tx_pilot):
    h = np.correlate(rx, tx_pilot, mode='full')
    return h / np.sum(np.abs(tx_pilot)**2)

h11 = estimate_csi(rx1, pilot_pn_tx1)
h12 = estimate_csi(rx1, pilot_pn_tx2)
h21 = estimate_csi(rx2, pilot_pn_tx1)
h22 = estimate_csi(rx2, pilot_pn_tx2)

# Extract peak tap
def extract_csi(h):
    idx = np.argmax(np.abs(h))
    return h[idx], idx

csi_h11, i11 = extract_csi(h11)
csi_h12, i12 = extract_csi(h12)
csi_h21, i21 = extract_csi(h21)
csi_h22, i22 = extract_csi(h22)

print("Estimated CSI values (PN):")
print(f"h11 (TX1_2 → RX1_H): {csi_h11:.4f} @ index {i11}")
print(f"h12 (TX2_1 → RX1_H): {csi_h12:.4f} @ index {i12}")
print(f"h21 (TX1_2 → RX2_H): {csi_h21:.4f} @ index {i21}")
print(f"h22 (TX2_1 → RX2_H): {csi_h22:.4f} @ index {i22}")

# Power Delay Profiles
pdp_h11 = np.abs(h11)**2
pdp_h12 = np.abs(h12)**2
pdp_h21 = np.abs(h21)**2
pdp_h22 = np.abs(h22)**2

# Delay axis in microseconds
delay_axis = np.arange(-len(pilot_pn_tx1)+1, len(rx1)) / sample_rate * 1e6  # in µs

def rms_delay_spread(pdp, delay_axis):
    pdp = np.array(pdp)
    delay_axis = np.array(delay_axis)
    
    pdp = pdp / np.sum(pdp)  # Normalize
    mean_delay = np.sum(delay_axis * pdp)
    mean_delay_sq = np.sum(((delay_axis - mean_delay)**2) * pdp)
    return np.sqrt(mean_delay_sq)

rms_h11 = rms_delay_spread(pdp_h11, delay_axis)
rms_h12 = rms_delay_spread(pdp_h12, delay_axis)
rms_h21 = rms_delay_spread(pdp_h21, delay_axis)
rms_h22 = rms_delay_spread(pdp_h22, delay_axis)

print("\nRMS Delay Spreads (in microseconds):")
print(f"TX1_2 → RX1_H: {rms_h11:.3f} µs")
print(f"TX2_1 → RX1_H: {rms_h12:.3f} µs")
print(f"TX1_2 → RX2_H: {rms_h21:.3f} µs")
print(f"TX2_1 → RX2_H: {rms_h22:.3f} µs")






# Plot PDP
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.title("PDP: TX1_2 → RX1_H")
plt.plot(delay_axis, pdp_h11); plt.xlabel("Delay (µs)"); plt.grid()

plt.subplot(2, 2, 2)
plt.title("PDP: TX2_1 → RX1_H")
plt.plot(delay_axis, pdp_h12); plt.xlabel("Delay (µs)"); plt.grid()

plt.subplot(2, 2, 3)
plt.title("PDP: TX1_2 → RX2_H")
plt.plot(delay_axis, pdp_h21); plt.xlabel("Delay (µs)"); plt.grid()

plt.subplot(2, 2, 4)
plt.title("PDP: TX2_1 → RX2_H")
plt.plot(delay_axis, pdp_h22); plt.xlabel("Delay (µs)"); plt.grid()

plt.tight_layout()
plt.show()




