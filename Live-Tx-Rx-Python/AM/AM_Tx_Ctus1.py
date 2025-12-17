import numpy as np
import time
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, Device

def generate_am_signal(frequency=1000, duration=5, sample_rate=1e6, modulation_index=0.5):
    t = np.arange(0, duration, 1/sample_rate)
    message = np.sin(2 * np.pi * frequency * t)
    carrier = 1 + modulation_index * message
    return carrier.astype(np.complex64)

class AMTransmitter:
    def __init__(self, sample_rate=1e6, center_freq=700e6, gain=60):
        self.sample_rate = sample_rate

        self.sdr = Device(dict(driver="lime"))
        self.tx_stream = self.sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])

        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
        self.sdr.setGain(SOAPY_SDR_TX, 0, gain)
        self.sdr.activateStream(self.tx_stream)

    def transmit_once(self, signal):
        num_tx = len(signal)
        self.sdr.writeStream(self.tx_stream, [signal], num_tx)
        time.sleep(len(signal) / self.sample_rate)

    def close(self):
        self.sdr.deactivateStream(self.tx_stream)
        self.sdr.closeStream(self.tx_stream)
        print("Transmission stopped.")

if __name__ == '__main__':
    tx = AMTransmitter()
    try:
        signal = generate_am_signal()
        print("[INFO] Transmitting 1 kHz AM tone on 700 MHz...")
        while True:
            tx.transmit_once(signal)
    except KeyboardInterrupt:
        tx.close()

