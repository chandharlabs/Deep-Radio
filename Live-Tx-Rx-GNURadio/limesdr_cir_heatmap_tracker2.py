import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import sys

# ----- Transmit sequence -----
barker = np.array([1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1])
barker_upsampled = np.repeat(barker, 10)
tx_signal = np.concatenate([barker_upsampled, np.zeros(1000)])

class CIRAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live CIR + RMS + PDP Analyzer")

        # GUI Layout
        self.canvas = FigureCanvasQTAgg(plt.figure(figsize=(10, 8)))
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Add 3 subplots
        self.ax1 = self.canvas.figure.add_subplot(311)
        self.ax2 = self.canvas.figure.add_subplot(312)
        self.ax3 = self.canvas.figure.add_subplot(313)

        # Data buffers
        self.cir_matrix = []
        self.sample_rate = 1e6
        self.Ts = 1 / self.sample_rate

        # Init SDR
        self.init_sdr()

        # Start live update
        self.update_plot()

    def init_sdr(self):
        args = dict(driver="lime")
        self.sdr = SoapySDR.Device(args)
        self.rx_chan = 0
        self.sdr.setSampleRate(SOAPY_SDR_RX, self.rx_chan, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, self.rx_chan, 915e6)
        self.sdr.setGain(SOAPY_SDR_RX, self.rx_chan, 50)
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    def capture_samples(self, num_samples=8192):
        self.sdr.activateStream(self.rx_stream)
        buff = np.array([0]*num_samples, np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [buff], num_samples)
        self.sdr.deactivateStream(self.rx_stream)
        return buff[:sr.ret] if sr.ret > 0 else np.zeros(num_samples, dtype=np.complex64)

    def compute_cir(self, rx):
        corr = np.correlate(rx, tx_signal, mode='full')
        start = len(tx_signal) - 1
        return corr[start:start+1024]

    def update_plot(self):
        rx = self.capture_samples()
        cir = self.compute_cir(rx)
        cir_mag = np.abs(cir)
        pdp = cir_mag**2

        # Update CIR buffer
        self.cir_matrix.append(cir_mag)
        if len(self.cir_matrix) > 50:
            self.cir_matrix.pop(0)

        cir_array = np.array(self.cir_matrix)

        # --- Heatmap (ax1) ---
        self.ax1.clear()
        self.ax1.imshow(20 * np.log10(cir_array + 1e-12),
                        aspect='auto', origin='lower', cmap='inferno')
        self.ax1.set_title("CIR Heatmap")
        self.ax1.set_ylabel("Frame Index")

        # --- RMS Delay Spread (ax2) ---
        rms = []
        for frame in cir_array:
            power = frame**2
            power /= np.sum(power) + 1e-12
            delay_idx = np.arange(len(power))
            mu = np.sum(power * delay_idx)
            sigma = np.sqrt(np.sum(power * (delay_idx - mu)**2)) * self.Ts * 1e6
            rms.append(sigma)

        self.ax2.clear()
        self.ax2.plot(rms, label="RMS Delay Spread (µs)")
        self.ax2.set_ylabel("Delay Spread (µs)")
        self.ax2.set_xlabel("Frame Index")
        self.ax2.set_ylim(0, max(1, np.max(rms) * 1.2))
        self.ax2.grid(True)
        self.ax2.legend()

        # --- Power Delay Profile (ax3) ---
        pdp_db = 10 * np.log10(pdp + 1e-12)
        self.ax3.clear()
        self.ax3.plot(pdp_db, color='green')
        self.ax3.set_title("Instantaneous Power Delay Profile (PDP)")
        self.ax3.set_xlabel("Sample Delay")
        self.ax3.set_ylabel("Power (dB)")
        self.ax3.set_ylim(-120, 0)
        self.ax3.grid(True)

        # Redraw all
        self.canvas.draw()
        QTimer.singleShot(300, self.update_plot)  # Refresh every 300ms

# --- Run the app ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CIRAnalyzer()
    window.show()
    sys.exit(app.exec_())

