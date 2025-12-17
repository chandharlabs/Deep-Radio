import numpy as np
from SoapySDR import Device, SOAPY_SDR_TX

# ==============================
# CONFIGURATION
# ==============================
CENTER_FREQ = 700e6
SAMPLE_RATE = 1e6
TX_GAIN = 60
SYMBOL_RATE = 1000
AMPLITUDE = 0.5
CARRIER_FREQ = 10000  # Hz
CHUNK = 1024

# ==============================
# MODULATION UTILITY FUNCTIONS
# ==============================
def bpsk(bits):
    """Map 0->-1, 1->1"""
    return 2 * np.array(bits) - 1

def qpsk(bits):
    """Map pairs of bits to QPSK constellation."""
    mapping = {
        (0,0): 1 + 1j,
        (0,1): -1 + 1j,
        (1,0): 1 - 1j,
        (1,1): -1 - 1j
    }
    symbols = []
    for i in range(0, len(bits), 2):
        b0, b1 = bits[i], bits[i+1]
        symbols.append(mapping[(b0, b1)])
    return np.array(symbols)

def gmsk(bits, BT=0.3, samples_per_symbol=1000):
    """Very basic GMSK modulator."""
    # Frequency shift for '1' vs '0'
    phase = np.pi * np.cumsum(2 * np.array(bits) - 1) / 2
    t = np.arange(samples_per_symbol) / samples_per_symbol
    shaped = np.repeat(np.exp(1j * phase), samples_per_symbol)
    return shaped

def cpfsk(bits, h=0.5, samples_per_symbol=1000):
    """Continuous phase FSK."""
    phase = 2 * np.pi * h * np.cumsum(np.array(bits) * 2 - 1) / samples_per_symbol
    shaped = np.exp(1j * np.repeat(phase, samples_per_symbol))
    return shaped

def psk(bits, order):
    """M-PSK mapping."""
    symbols = []
    for i in range(0, len(bits), int(np.log2(order))):
        chunk = bits[i:i + int(np.log2(order))]
        decimal = int("".join(str(b) for b in chunk), 2)
        angle = 2 * np.pi * decimal / order
        symbols.append(np.exp(1j * angle))
    return np.array(symbols)

def qam(bits, order):
    """M-QAM mapping."""
    import itertools
    m = int(np.sqrt(order))
    levels = np.arange(-m + 1, m, 2)  # e.g., [-3, -1, 1, 3]
    symbols = []
    for i in range(0, len(bits), int(np.log2(order))):
        chunk = bits[i:i + int(np.log2(order))]
        decimal = int("".join(str(b) for b in chunk), 2)
        I = levels[decimal % m]
        Q = levels[decimal // m]
        symbols.append(I + 1j * Q)
    return np.array(symbols)

# ==============================
# MAIN FUNCTION
# ==============================
if __name__ == "__main__":
    import itertools
    import time

    MODULATION = input("[INPUT] Enter modulation (bpsk/qpsk/gmsk/cpfsk/8psk/16qam/64qam): ").lower()

    # GENERATION PARAMETERS
    samples_per_symbol = int(SAMPLE_RATE / SYMBOL_RATE)
    BIT_LENGTH = 1024
    BITS = np.random.randint(0, 2, BIT_LENGTH)

    if MODULATION == "bpsk":
        symbols = bpsk(BITS)
    elif MODULATION == "qpsk":
        symbols = qpsk(BITS)
    elif MODULATION == "gmsk":
        symbols = gmsk(BITS, samples_per_symbol=samples_per_symbol)
    elif MODULATION == "cpfsk":
        symbols = cpfsk(BITS, samples_per_symbol=samples_per_symbol)
    elif MODULATION == "8psk":
        symbols = psk(BITS, order=8)
    elif MODULATION == "16qam":
        symbols = qam(BITS, order=16)
    elif MODULATION == "64qam":
        symbols = qam(BITS, order=64)
    else:
        raise ValueError(f"Unsupported modulation: {MODULATION}")

    # UPSAMPLE IF NEEDED
    if MODULATION not in ["gmsk", "cpfsk"]:
        signal = np.repeat(symbols, samples_per_symbol)
    else:
        signal = symbols

    # CARRIER MODULATION
    t = np.arange(len(signal)) / SAMPLE_RATE
    carrier = np.exp(1j * 2 * np.pi * CARRIER_FREQ * t)
    signal = AMPLITUDE * signal * carrier
    signal = signal.astype(np.complex64)

    # SDR INIT
    sdr = Device(dict(driver="lime"))
    sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
    sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)
    sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND1")
    tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
    sdr.activateStream(tx_stream)

    # CONTINUOUS TRANSMISSION
    print(f"[INFO] Transmitting {MODULATION.upper()} at {CENTER_FREQ / 1e6} MHz...")
    try:
        i = 0
        while True:
            chunk = signal[i:i + CHUNK]
            if len(chunk) < CHUNK:
                chunk = np.concatenate([chunk, signal[:CHUNK - len(chunk)]])
                i = CHUNK - len(chunk)
            sdr.writeStream(tx_stream, [chunk], len(chunk))
            i += CHUNK
            if i >= len(signal): i = 0
    except KeyboardInterrupt:
        print("\n[INFO] Transmission stopped.")
    finally:
        sdr.deactivateStream(tx_stream)
        sdr.closeStream(tx_stream)
