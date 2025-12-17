import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
import time

# === SDR Parameters ===
sample_rate = 5e6
center_freq = 915e6
tx_gain = 60
rx_gain = 60
fft_size = 512
cp_len = 72
num_samples = fft_size + cp_len+10000
subcarrier_index = 257  # center subcarrier

# === Generate QPSK-modulated OFDM pilot ===
def generate_ofdm_symbol(subcarrier_index, qpsk_symbol, fft_size, cp_len):
    freq_bins = np.zeros(fft_size, dtype=np.complex64)
    freq_bins[subcarrier_index] = qpsk_symbol
    time_domain = np.fft.ifft(np.fft.ifftshift(freq_bins))
    with_cp = np.concatenate([time_domain[-cp_len:], time_domain])
    return with_cp.astype(np.complex64)

# === TX Buffers (One subcarrier modulated by QPSK) ===
np.random.seed(42)
qpsk1 = ((1 - 2*np.random.randint(0,2)) + 1j*(1 - 2*np.random.randint(0,2))) / np.sqrt(2)
qpsk2 = ((1 - 2*np.random.randint(0,2)) + 1j*(1 - 2*np.random.randint(0,2))) / np.sqrt(2)

ofdm_tx1 = generate_ofdm_symbol(subcarrier_index, qpsk1, fft_size, cp_len)
ofdm_tx2 = generate_ofdm_symbol(subcarrier_index, qpsk2, fft_size, cp_len)

# Padding into buffers
tx_buf1 = np.zeros(4096, dtype=np.complex64)
tx_buf2 = np.zeros(4096, dtype=np.complex64)
tx_buf1[100:100+num_samples] = ofdm_tx1
tx_buf2[100:100+num_samples] = ofdm_tx2

# === SDR Setup ===
args = dict(driver="lime")
sdr = SoapySDR.Device(args)

# TX configuration
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_TX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_TX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_TX, ch, tx_gain)
    sdr.setAntenna(SOAPY_SDR_TX, ch, "BAND2")

# RX configuration
for ch in [0, 1]:
    sdr.setSampleRate(SOAPY_SDR_RX, ch, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    sdr.setGain(SOAPY_SDR_RX, ch, rx_gain)
    sdr.setAntenna(SOAPY_SDR_RX, ch, "LNAH")

# Setup streams
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0, 1])
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1])
sdr.activateStream(tx_stream)
sdr.activateStream(rx_stream)

# === TX1 active, TX2 silent ===
sdr.writeStream(tx_stream, [tx_buf1, np.zeros_like(tx_buf1)], len(tx_buf1))
time.sleep(0.05)
rx1a = np.zeros(4096, dtype=np.complex64)
rx2a = np.zeros(4096, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1a, rx2a], len(rx1a))

# === TX2 active, TX1 silent ===
sdr.writeStream(tx_stream, [np.zeros_like(tx_buf2), tx_buf2], len(tx_buf2))
time.sleep(0.05)
rx1b = np.zeros(4096, dtype=np.complex64)
rx2b = np.zeros(4096, dtype=np.complex64)
sdr.readStream(rx_stream, [rx1b, rx2b], len(rx1b))

# Cleanup
sdr.deactivateStream(tx_stream)
sdr.deactivateStream(rx_stream)
sdr.closeStream(tx_stream)
sdr.closeStream(rx_stream)

# === Extract CSI ===
def estimate_csi(rx, symbol, qpsk_ref, subcarrier_index):
    rx_ofdm = rx[100+cp_len:100+num_samples]
    rx_fft = np.fft.fftshift(np.fft.fft(rx_ofdm, fft_size))
    rx_sym = rx_fft[subcarrier_index]
    return rx_sym / qpsk_ref

h11 = estimate_csi(rx1a, ofdm_tx1, qpsk1, subcarrier_index)
h21 = estimate_csi(rx2a, ofdm_tx1, qpsk1, subcarrier_index)
h12 = estimate_csi(rx1b, ofdm_tx2, qpsk2, subcarrier_index)
h22 = estimate_csi(rx2b, ofdm_tx2, qpsk2, subcarrier_index)

# === Print CSI Matrix ===
print("\nEstimated 2x2 MIMO CSI Matrix H:")
print(f"h11 (TX1→RX1): {h11:.4f}")
print(f"h12 (TX2→RX1): {h12:.4f}")
print(f"h21 (TX1→RX2): {h21:.4f}")
print(f"h22 (TX2→RX2): {h22:.4f}")

# === Plot RX Signals (Magnitude) ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("RX1 Signal")
plt.plot(np.abs(rx1a), label="From TX1")
plt.plot(np.abs(rx1b), label="From TX2", linestyle='--')
plt.grid(); plt.legend()

plt.subplot(1, 2, 2)
plt.title("RX2 Signal")
plt.plot(np.abs(rx2a), label="From TX1")
plt.plot(np.abs(rx2b), label="From TX2", linestyle='--')
plt.grid(); plt.legend()
plt.tight_layout()
plt.show()

