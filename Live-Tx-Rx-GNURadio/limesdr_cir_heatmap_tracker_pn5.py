import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
import SoapySDR
from SoapySDR import *
import sys
import time
from PyQt5.QtCore import QTimer

# ===== CONFIGURATION =====
SAMPLE_RATE = 60e6
CENTER_FREQ = 3200e6
TX_GAIN = 60
RX_GAIN = 60

# ===== Generate PN Sequence =====
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
    return np.array(seq)

pn_seq = generate_pn_sequence()
pn_upsampled = np.repeat(pn_seq, 4)
zero_padding = np.zeros(2048)
tx_burst = np.concatenate([pn_upsampled, zero_padding])
tx_signal = np.tile(tx_burst, 20).astype(np.complex64)
NUM_SAMPLES = len(tx_signal)

class CIRAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live CIR + PDP (TX0→RX1_H)")

        layout = QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(plt.figure(figsize=(10, 8)))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_live)
        button_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_live)
        button_layout.addWidget(self.pause_button)

        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.ax1 = self.canvas.figure.add_subplot(211)
        self.ax2 = self.canvas.figure.add_subplot(212)

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_and_plot)

        self.init_sdr()

    def init_sdr(self):
        self.sdr = SoapySDR.Device(dict(driver="lime"))
        self.sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")  # TX1_2 antenna
        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
        self.sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)
        self.tx_stream = self.sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)

        self.sdr.setSampleRate(SOAPY_SDR_RX, 1, SAMPLE_RATE)
        self.sdr.setFrequency(SOAPY_SDR_RX, 1, CENTER_FREQ)
        self.sdr.setGain(SOAPY_SDR_RX, 1, RX_GAIN)
        self.sdr.setAntenna(SOAPY_SDR_RX, 1, "LNAH")
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [1])

    def tx_once(self):
        self.sdr.activateStream(self.tx_stream)
        sr = self.sdr.writeStream(self.tx_stream, [tx_signal], len(tx_signal))
        self.sdr.deactivateStream(self.tx_stream)

    def capture_samples(self):
        self.sdr.activateStream(self.rx_stream)
        buff = np.zeros(NUM_SAMPLES, np.complex64)
        sr = self.sdr.readStream(self.rx_stream, [buff], NUM_SAMPLES)
        self.sdr.deactivateStream(self.rx_stream)
        return buff[:sr.ret] if sr.ret > 0 else np.zeros(NUM_SAMPLES, dtype=np.complex64)

    def compute_cir(self, rx):
        pn_ref = tx_burst
        corr = np.correlate(rx, pn_ref, mode='full')
        corr = corr / (np.linalg.norm(pn_ref)**2 + 1e-12)
        peak_index = np.argmax(np.abs(corr))
        start = peak_index
        end = start + NUM_SAMPLES
        if end > len(corr):
            cir = corr[start:]
            cir = np.pad(cir, (0, NUM_SAMPLES - len(cir)), 'constant')
        else:
            cir = corr[start:end]
        return cir

    def capture_and_plot(self):
        self.tx_once()
        time.sleep(0.05)
        rx = self.capture_samples()
        cir = self.compute_cir(rx)
        cir_mag = np.abs(cir)
        pdp = cir_mag ** 2
        pdp /= np.sum(pdp) + 1e-12

        delays = np.arange(len(pdp))
        mu = np.sum(delays * pdp)
        sigma = np.sqrt(np.sum(pdp * (delays - mu) ** 2)) * (1 / SAMPLE_RATE) * 1e6

        self.ax1.clear()
        self.ax1.plot(np.arange(len(cir_mag)) / SAMPLE_RATE * 1e6, 20 * np.log10(cir_mag + 1e-12))
        self.ax1.set_ylabel("CIR (dB)")
        self.ax1.set_title(f"CIR | RMS Delay Spread = {sigma:.2f} µs")
        self.ax1.grid(True)

        self.ax2.clear()
        time_axis = np.arange(len(pdp)) / SAMPLE_RATE * 1e6
        pdp_db = 10 * np.log10(pdp + 1e-12)

        threshold_db = np.max(pdp_db) - 6
        strong_indices = np.where(pdp_db >= threshold_db)[0]

        self.ax2.plot(time_axis, pdp_db, label="Normalized PDP", color='green')
        self.ax2.scatter(time_axis[strong_indices], pdp_db[strong_indices], color='red', s=60, label='Strong Paths')

        self.ax2.set_title("Normalized Power Delay Profile (PDP)")
        self.ax2.set_xlabel("Delay (µs)")
        self.ax2.set_ylabel("Power (dB)")
        self.ax2.set_ylim(-60, 0)
        self.ax2.grid(True)
        self.ax2.legend()

        self.canvas.draw()

    def start_live(self):
        self.timer.start(1000)  # every 1 second

    def pause_live(self):
        self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CIRAnalyzer()
    window.show()
    sys.exit(app.exec_())

