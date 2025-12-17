from threading import Thread, Event
import time
from gnuradio import gr
import numpy as np
# from SoapySDR import SOAPY_SDR_TX, Device

transmitter_thread = None
stop_event = Event()

modulations = ["BPSK", "QPSK", "16QAM", "64QAM", "GMSK", "CPFSK", "FSK"]


class SDRTransmitter:
    def __init__(
        self,
        char_to_send="A",
        modulation="BPSK",
        sample_rate=1e6,
        symbol_period=0.01,
        center_freq=700e6,
        gain=64,
    ):
        bit_length = (
            6
            if modulation == "64QAM"
            else 4 if modulation == "16QAM" else 2 if modulation == "QPSK" else 1
        )
        self.bits = self.char_to_bits(char_to_send, bit_length)
        self.symbols = self.generate_symbols(self.bits, modulation)

        self.samples_per_symbol = int(sample_rate * symbol_period)
        self.upsampled_symbols = np.repeat(self.symbols, self.samples_per_symbol)

        self.sdr = Device(dict(driver="lime"))
        self.sdr_tx = self.sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
        self.sdr.setGain(SOAPY_SDR_TX, 0, gain)
        self.sdr.activateStream(self.sdr_tx)

    def generate_symbols(self, bits, modulation):
        if modulation == "BPSK":
            return np.where(np.array(bits) == 1, 1, -1).astype(np.complex64)
        elif modulation == "QPSK":
            bit_pairs = np.reshape(bits, (-1, 2))
            mapping = {
                (0, 0): 1 + 1j,
                (0, 1): -1 + 1j,
                (1, 0): 1 - 1j,
                (1, 1): -1 - 1j,
            }
            return np.array([mapping[tuple(b)] for b in bit_pairs], dtype=np.complex64)
        elif modulation == "16QAM":
            bit_quads = np.reshape(bits, (-1, 4))
            mapping = {
                (0, 0, 0, 0): -3 - 3j,
                (0, 0, 0, 1): -3 - 1j,
                (0, 0, 1, 0): -3 + 3j,
                (0, 0, 1, 1): -3 + 1j,
                (0, 1, 0, 0): -1 - 3j,
                (0, 1, 0, 1): -1 - 1j,
                (0, 1, 1, 0): -1 + 3j,
                (0, 1, 1, 1): -1 + 1j,
                (1, 0, 0, 0): 3 - 3j,
                (1, 0, 0, 1): 3 - 1j,
                (1, 0, 1, 0): 3 + 3j,
                (1, 0, 1, 1): 3 + 1j,
                (1, 1, 0, 0): 1 - 3j,
                (1, 1, 0, 1): 1 - 1j,
                (1, 1, 1, 0): 1 + 3j,
                (1, 1, 1, 1): 1 + 1j,
            }
            return np.array([mapping[tuple(b)] for b in bit_quads], dtype=np.complex64)
        elif modulation == "64QAM":
            bit_six = np.reshape(bits, (-1, 6))
            mapping = {
                (tuple(format(i, "06b"))): (
                    2 * (i % 2) - 1 + (2 * ((i // 2) % 2) - 1) * 1j
                )
                * (2 * ((i // 4) % 2) - 1 + (2 * ((i // 8) % 2) - 1) * 1j)
                * (2 * ((i // 16) % 2) - 1 + (2 * ((i // 32) % 2) - 1) * 1j)
                for i in range(64)
            }
            return np.array([mapping[tuple(b)] for b in bit_six], dtype=np.complex64)
        elif modulation == "GMSK":
            return np.exp(1j * np.pi * np.cumsum(bits) / 2).astype(np.complex64)
        elif modulation == "CPFSK":
            return np.exp(1j * np.pi * np.cumsum(bits)).astype(np.complex64)
        elif modulation == "FSK":
            return np.exp(1j * np.pi * np.array(bits)).astype(np.complex64)
        else:
            raise ValueError("Unsupported modulation type")

    def char_to_bits(self, char, num_bits):
        return [int(b) for b in format(ord(char), f"0{num_bits}b")]

    def transmit_once(self):
        self.sdr.writeStream(
            self.sdr_tx,
            [self.upsampled_symbols.astype(np.complex64)],
            len(self.upsampled_symbols),
        )
        time.sleep(0.1)

    def close(self):
        self.sdr.deactivateStream(self.sdr_tx)
        self.sdr.closeStream(self.sdr_tx)
        print("Transmission stopped.")


def start_transmitting_bpsk(char_to_transmit, modulation, gain, symbol_period):
    global transmitter_thread, stop_event
    stop_event.clear()
    tb = SDRTransmitter(
        char_to_send=char_to_transmit,
        modulation=modulation,
        gain=gain,
        symbol_period=symbol_period,
    )

    def transmit():
        try:
            while not stop_event.is_set():
                tb.transmit_once()
                time.sleep(1)
        finally:
            tb.close()

    transmitter_thread = Thread(target=transmit)
    transmitter_thread.start()


def stop_transmitting_bpsk():
    global stop_event
    stop_event.set()
    if transmitter_thread:
        transmitter_thread.join()
