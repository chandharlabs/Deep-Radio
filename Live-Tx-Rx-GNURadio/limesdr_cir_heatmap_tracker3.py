import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer
from mpl_toolkits.mplot3d import Axes3D
import SoapySDR
from SoapySDR import *
import sys

# ----- Parameters -----
SAMPLE_RATE = 1e6
NUM_SAMPLES = 4000
MAX_FRAMES = 50

# ----- Transmit Sequence -----
barker = np.array([1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1])
barker_upsampled = np.repeat(barker, 10)
tx_signal = np.concatenate([barker_upsampled, np.zeros(500)])

class CIRAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live CIR Analyzer with PDP + Start/Stop")

        # Main layout
        layout = QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(plt.figure(figsize=(10, 10)))
        layout.addWidget(self.canvas)

        # Control buttons
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
        self.ax3 = self.canvas.figure.add_subplot(313, projection='3d')

        # Buffers
        self.cir_matrix = []
        self.pdp_matrix = []
        self.rms_list = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        self.init_sdr()

    def init_sdr(self):
        self.sdr = SoapySDR.Device(dict(driver="lime"))
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, 915e6)
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
        pdp = cir_mag**2

        # Store
        self.cir_matrix.append(cir_mag)
        self.pdp_matrix.append(pdp)

        # Keep fixed length
        if len(self.cir_matrix) > MAX_FRAMES:
            self.cir_matrix.pop(0)
            self.pdp_matrix.pop(0)

        # --- RMS Delay Spread ---
        power = pdp / np.sum(pdp + 1e-12)
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

        # ----------- PLOT 3: 3D PDP Waterfall -----------
        self.ax3.clear()
        pdp_db = 10 * np.log10(np.array(self.pdp_matrix) + 1e-12)
        Y = np.arange(len(pdp_db))
        X = np.arange(NUM_SAMPLES)
        X, Y = np.meshgrid(X, Y)

        for i in range(0, len(pdp_db), 2):  # step to avoid clutter
            self.ax3.plot(X[i], Y[i], pdp_db[i], color='blue', alpha=0.6)

        # Highlight strongest path of latest frame
        last_pdp = pdp_db[-1]
        max_idx = np.argmax(last_pdp)
        self.ax3.scatter(max_idx, len(pdp_db)-1, last_pdp[max_idx], color='red', s=50, label='Strongest Path')

        self.ax3.set_xlabel("Sample Delay")
        self.ax3.set_ylabel("Frame Index")
        self.ax3.set_zlabel("Power (dB)")
        self.ax3.set_title("3D Waterfall PDP")
        self.ax3.legend()

        self.canvas.draw()

# ----- Main App -----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CIRAnalyzer()
    window.show()
    sys.exit(app.exec_())

