from gnuradio import gr
import numpy as np
import time
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, Device

def generate_bpsk_symbols(bits):
    return np.array([1 if b == 1 else -1 for b in bits], dtype=np.complex64)

class BPSKTransmitter:
    def __init__(self, sample_rate=.1e6, symbol_period=.01, center_freq=700e6, gain=60):
        self.bits = [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0]
        self.symbols = generate_bpsk_symbols(self.bits)
        
        # Upsample to match the sample rate
        self.samples_per_symbol = int(sample_rate * symbol_period)
        self.upsampled_symbols = np.repeat(self.symbols, self.samples_per_symbol)
        
        # Configure SoapySDR device
        self.sdr = Device(dict(driver="lime"))
        self.sdr_tx = self.sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
        self.sdr.setGain(SOAPY_SDR_TX, 0, gain)
        self.sdr.activateStream(self.sdr_tx)

    def transmit_once(self):
        num_tx = len(self.upsampled_symbols)
        buff = self.upsampled_symbols.astype(np.complex64)
        self.sdr.writeStream(self.sdr_tx, [buff], num_tx)
        time.sleep(len(self.bits) * 0.01)  # Wait for transmission duration

    def close(self):
        self.sdr.deactivateStream(self.sdr_tx)
        self.sdr.closeStream(self.sdr_tx)
        print("Transmission stopped.")

if __name__ == '__main__':
    tb = BPSKTransmitter()
    try:
        print("Press Ctrl+C to stop transmission...")
        while True:
            tb.transmit_once()
            time.sleep(1)  # Wait 1 second before retransmitting
    except KeyboardInterrupt:
        tb.close()

