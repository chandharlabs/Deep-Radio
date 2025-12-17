import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import sys

# ===== PARAMETERS =====
SAMPLE_RATE = 60e6
NUM_SAMPLES = 32768*5
MAX_FRAMES = 50

# ===== TX SIGNAL: PN SEQUENCE =====
def generate_pn_sequence(length=127):
    """Generate a maximal-length PN sequence using an LFSR"""
    state = 0b1111111  # 7-bit LFSR seed (non-zero)
    taps = [7, 1]      # Polynomial x^7 + x + 1
    seq = []
    for _ in range(length):
        output = state & 1
        seq.append(1 if output else -1)
        feedback = 0
        for t in taps:
            feedback ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (feedback << 6)
    return np.array(seq)

pn_seq = generate_pn_sequence(length=127)
pn_upsampled = np.repeat(pn_seq, 10)
tx_signal = np.concatenate([pn_upsampled, np.zeros(500)])  # total length ~1770 samples

# ===== GUI CLASS =====
class CIRAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live CIR Analyzer with PN-Sequence & PDP")

        # Main layout
        layout = QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(plt.figure(figsize=(10, 10)))
        layout.addWidget(self.canvas)

        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Subplots
        self.ax1 = self.canvas.figure.add_subplot(311)
        self.ax2 = self.canvas.figure.add_subplot(312)
        self.ax3 = self.canvas.figure.add_subplot(313)

        # Buffers
        self.cir_matrix = []
        self.rms_list = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        self.init_sdr()

    def init_sdr(self):
        self.sdr = SoapySDR.Device(dict(driver="lime"))
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, 700e6)
        self.sdr.setGain(SOAPY_SDR_RX, 0, 50)
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    def capture_samples(self):
        self.sdr.activateStream(self.rx_stream)
        buff = np.zeros(NUM_SAMPLES, np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [buff], NUM_SAMPLES)
        self.sdr.deactivateStream(self.rx_stream)
        return buff[:sr.ret] if sr.ret > 0 else np.zeros(NUM_SAMPLES, dtype=np.complex64)

    def compute_cir(self, rx):
        corr = np.correlate(rx, tx_signal, mode='full')
        start = len(tx_signal) - 1
        return corr[start:start + NUM_SAMPLES]

    def start_capture(self):
        self.timer.start(300)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_capture(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_plot(self):
        rx = self.capture_samples()
        cir = self.compute_cir(rx)
        cir_mag = np.abs(cir)
        pdp = cir_mag ** 2

        # Store CIR
        self.cir_matrix.append(cir_mag)
        if len(self.cir_matrix) > MAX_FRAMES:
            self.cir_matrix.pop(0)

        # --- RMS Delay Spread ---
        power = pdp / (np.sum(pdp) + 1e-12)
        delays = np.arange(len(pdp))
        mu = np.sum(delays * power)
        sigma = np.sqrt(np.sum(power * (delays - mu) ** 2)) * (1 / SAMPLE_RATE) * 1e6
        self.rms_list.append(sigma)
        if len(self.rms_list) > MAX_FRAMES:
            self.rms_list.pop(0)

        # ----------- PLOT 1: CIR Heatmap -----------
        self.ax1.clear()
        cir_array = np.array(self.cir_matrix)
        self.ax1.imshow(20 * np.log10(cir_array + 1e-8),
                        aspect='auto', origin='lower', cmap='inferno')
        self.ax1.set_title("CIR Heatmap (dB)")
        self.ax1.set_ylabel("Frame Index")

        # ----------- PLOT 2: RMS Delay Spread -----------
        self.ax2.clear()
        self.ax2.plot(self.rms_list, label="RMS Delay Spread (µs)")
        self.ax2.set_ylabel("µs")
        self.ax2.set_xlabel("Frame Index")
        self.ax2.grid(True)
        self.ax2.legend()

        # ----------- PLOT 3: PDP with Multiple Paths -----------
        self.ax3.clear()
        time_axis = np.arange(len(pdp)) / SAMPLE_RATE * 1e6
        pdp_db = 10 * np.log10(pdp + 1e-12)

        # Detect significant paths above threshold
        threshold_db = np.max(pdp_db) - 6
        strong_indices = np.where(pdp_db >= threshold_db)[0]

        self.ax3.plot(time_axis, pdp_db, label="PDP", color='green')
        self.ax3.scatter(time_axis[strong_indices],
                         pdp_db[strong_indices],
                         color='red', s=60, label='Strong Paths')

        max_idx = np.argmax(pdp_db)
        max_val = pdp_db[max_idx]
        self.ax3.text(time_axis[max_idx], max_val + 2,
                      f"{time_axis[max_idx]:.1f} µs", color='red', fontsize=9)

        self.ax3.set_title("Instantaneous Power Delay Profile (PDP)")
        self.ax3.set_xlabel("Delay (µs)")
        self.ax3.set_ylabel("Power (dB)")
        self.ax3.set_ylim(-120, 15)
        self.ax3.grid(True)
        self.ax3.legend()

        self.canvas.draw()

# ===== MAIN APP =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CIRAnalyzer()
    window.show()
    sys.exit(app.exec_())

