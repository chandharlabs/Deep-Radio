from gnuradio import gr
import numpy as np
import time
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, Device


def generate_bpsk_symbols(bits):
    """Maps bits (0 → -1, 1 → +1) to BPSK symbols."""
    x = np.array([1 if b == 1 else -1 for b in bits], dtype=np.complex64)
    return x

def string_to_bits(string):
    """Converts a string to a binary sequence with a single '010' prefix."""
    bits = [0, 1, 0, 1, 1, 1]  # Add '010' prefix only once before the entire string
    for char in string:
        bits.extend([int(b) for b in format(ord(char), '08b')])  # Convert each character to 8-bit ASCII
    return bits

class BPSKTransmitter:
    def __init__(self, text, sample_rate=1e6, symbol_period=.01, center_freq=700e6, gain=64):
        # Convert the string to a bit sequence with a single '010' prefix
        raw_bits = string_to_bits(text)
        self.symbol_period = symbol_period
        self.bits = raw_bits
        self.symbols = generate_bpsk_symbols(self.bits)

        # Print transmission details
        print(f"\nTransmitting String: '{text}'")
        print(f"Prefixed Binary:        {''.join(map(str, raw_bits))}")
        print(f"BPSK Symbols:           {self.symbols}\n")

        # Upsample to match sample rate
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
        time.sleep(self.symbol_period*len(self.bits))  # Small delay for transmission

    def close(self):
        self.sdr.deactivateStream(self.sdr_tx)
        self.sdr.closeStream(self.sdr_tx)
        print("Transmission stopped.")

if __name__ == '__main__':
    text_to_transmit = input("Enter a string to transmit: ").strip()  # Ask user for input
    tb = BPSKTransmitter(text=text_to_transmit)
    
    try:
        print("Press Ctrl+C to stop transmission...")
        while True:
            tb.transmit_once()
            time.sleep(1)  # Transmit once per second
    except KeyboardInterrupt:
        tb.close()

