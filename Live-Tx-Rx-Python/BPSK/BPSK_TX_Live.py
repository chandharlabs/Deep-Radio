import numpy as np
import time
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, Device
import sys
import select


def generate_bpsk_symbols(bits):
    return np.array([1 if b == 1 else -1 for b in bits], dtype=np.complex64)

def string_to_bits(string):
    bits = [0, 1, 0, 1, 1, 1]
    for char in string:
        bits.extend([int(b) for b in format(ord(char), '08b')])
    return bits

class BPSKTransmitter:
    def __init__(self, sample_rate=1e6, symbol_period=0.01, center_freq=700e6, gain=64):
        self.sample_rate = sample_rate
        self.symbol_period = symbol_period
        self.samples_per_symbol = int(sample_rate * symbol_period)

        self.sdr = Device(dict(driver="lime"))
        self.sdr_tx = self.sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
        self.sdr.setGain(SOAPY_SDR_TX, 0, gain)
        self.sdr.activateStream(self.sdr_tx)

        self.upsampled_symbols = np.zeros(1024, dtype=np.complex64)

    def update_text(self, text):
        text = f"~{text}#"
        raw_bits = string_to_bits(text)
        symbols = generate_bpsk_symbols(raw_bits)
        self.upsampled_symbols = np.repeat(symbols, self.samples_per_symbol)

        print(f"\n[Updated Transmission]")
        print(f"Text:       {text}")
        print(f"Bitstream:  {''.join(map(str, raw_bits))}")
        print(f"Symbols:    {symbols[:10]}... (total {len(symbols)} symbols)\n")

    def transmit_once(self):
        num_tx = len(self.upsampled_symbols)
        buff = self.upsampled_symbols.astype(np.complex64)
        self.sdr.writeStream(self.sdr_tx, [buff], num_tx)
        time.sleep(self.symbol_period * len(buff) / self.samples_per_symbol)

    def close(self):
        self.sdr.deactivateStream(self.sdr_tx)
        self.sdr.closeStream(self.sdr_tx)
        print("Transmission stopped.")

def input_available():
    return select.select([sys.stdin], [], [], 0.0)[0]



if __name__ == '__main__':
    tb = BPSKTransmitter()

    try:
        while True:
            user_input = input("Enter string to transmit once (or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                break
            tb.update_text(user_input)
            tb.transmit_once()
            print("Transmission completed.\n")
    except KeyboardInterrupt:
        pass
    finally:
        tb.close()

