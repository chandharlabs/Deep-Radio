import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import SoapySDR
from SoapySDR import *
import time

# ----------------- SDR + Signal Settings ----------------
SAMPLE_RATE = 5e6
CENTER_FREQ = 915e6
TX_GAIN = 60
RX_GAIN = 60
NUM_FRAMES = 50
RX_CAPTURE_LEN = 8192

BARKER_SEQ = [1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
pulse = np.array(BARKER_SEQ, dtype=np.complex64)

# ----------------- SDR Setup ----------------
sdr = SoapySDR.Device(dict(driver="lime"))
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_RX, 0, RX_GAIN)

rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
sdr.activateStream(rx_stream)

# ----------------- Repeated CIR Capture ----------------
cir_matrix = []

print("Starting CIR heatmap capture...")
for i in range(NUM_FRAMES):
    # --- Receive ---
    rx_buf = np.empty(RX_CAPTURE_LEN, dtype=np.complex64)
    sr = sdr.readStream(rx_stream, [rx_buf], RX_CAPTURE_LEN)
    rx_data = rx_buf[:sr.ret] if sr.ret > 0 else np.zeros(RX_CAPTURE_LEN, dtype=np.complex64)

    # --- CIR Calculation ---
    cir = correlate(rx_data, pulse, mode='full')
    cir_mag = np.abs(cir)
    cir_matrix.append(cir_mag)

    print(f"Captured frame {i+1}/{NUM_FRAMES}")
    time.sleep(0.5)  # Adjust as needed

sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

# ----------------- Heatmap Plot ----------------
cir_matrix = np.array(cir_matrix)
delay_axis = np.arange(-len(pulse) + 1, RX_CAPTURE_LEN)

plt.figure(figsize=(12, 6))
plt.imshow(20*np.log10(cir_matrix + 1e-6), aspect='auto', cmap='inferno',
           extent=[delay_axis[0], delay_axis[-1], 0, NUM_FRAMES], origin='lower')
plt.colorbar(label="Magnitude (dB)")
plt.xlabel("Sample Delay")
plt.ylabel("Frame Index")
plt.title("CIR Heatmap Over Time (Barker Sequence)")
plt.tight_layout()
plt.show()

