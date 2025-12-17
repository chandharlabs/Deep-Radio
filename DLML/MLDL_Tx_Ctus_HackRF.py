#!/usr/bin/env python3
"""
HackRF transmit example using SoapySDR (CF32).
Improved modulation helpers + robust HackRF initialization.
"""
import numpy as np
from SoapySDR import Device, SOAPY_SDR_TX
import time
import sys

# ==============================
# CONFIGURATION
# ==============================
CENTER_FREQ = 700e6        # TX center frequency (Hz)
SAMPLE_RATE = 2e6         # HackRF supports up to ~20e6, pick something reasonable
TX_GAIN = 40              # initial gain value (dB) - HackRF gain range differs; we try best-effort
SYMBOL_RATE = 1e3         # symbol rate (symbols/sec)
AMPLITUDE = 0.3           # base amplitude (scale to avoid clipping)
CARRIER_FREQ = 0          # if >0, will multiply by a complex carrier (Hz). 0 = no carrier (baseband)
CHUNK = 4096              # samples per write
BITS_TO_GENERATE = 16384  # number of random bits to create (must be >= required)

# ==============================
# MODULATION UTILITY FUNCTIONS
# ==============================
def pad_bits(bits, block):
    """Pad bits length to next multiple of block by repeating zeros."""
    if len(bits) % block == 0:
        return bits
    pad_len = block - (len(bits) % block)
    return np.concatenate([bits, np.zeros(pad_len, dtype=int)])

def bpsk(bits):
    """Map 0->-1, 1->+1 (real)"""
    return 2 * bits.astype(np.int8) - 1

def qpsk(bits):
    """Map pairs of bits to normalized QPSK constellation (complex)."""
    bits = pad_bits(bits, 2)
    mapping = {
        (0, 0): (1 + 1j) / np.sqrt(2),
        (0, 1): (-1 + 1j) / np.sqrt(2),
        (1, 0): (1 - 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
    }
    symbols = []
    for i in range(0, len(bits), 2):
        b0, b1 = int(bits[i]), int(bits[i+1])
        symbols.append(mapping[(b0, b1)])
    return np.array(symbols, dtype=np.complex64)

def gmsk(bits, BT=0.3, samples_per_symbol=8):
    """Basic GMSK: Gaussian filter + continuous phase frequency modulation.
    This is a simple implementation for demo purposes (not highly optimized)."""
    bits = pad_bits(bits, 1)
    # NRZ mapping (-1, +1)
    nrz = 2 * bits.astype(np.int8) - 1
    # upsample
    up = np.repeat(nrz, samples_per_symbol).astype(np.float64)
    # gaussian filter design (alpha from BT)
    # approximate filter length
    span = 4  # symbols
    N = span * samples_per_symbol
    t = np.linspace(-span/2, span/2, N)
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
    g = np.exp(-t**2/(2*sigma**2))
    g /= np.sum(g)
    filtered = np.convolve(up, g, mode='same')
    # integrate to get phase (h = 0.5 frequency sensitivity)
    h = 0.5
    phase = np.cumsum(filtered) * (np.pi * h / samples_per_symbol)
    return np.exp(1j * phase).astype(np.complex64)

def cpfsk(bits, h=0.5, samples_per_symbol=8):
    """Continuous phase FSK simple implementation."""
    bits = pad_bits(bits, 1)
    nrz = 2 * bits.astype(np.int8) - 1
    up = np.repeat(nrz, samples_per_symbol).astype(np.float64)
    # instantaneous frequency = h * nrz (scaled)
    phase = 2 * np.pi * h * np.cumsum(up) / samples_per_symbol
    return np.exp(1j * phase).astype(np.complex64)

def psk(bits, order):
    """M-PSK mapping. Pads bits as needed."""
    k = int(np.log2(order))
    bits = pad_bits(bits, k)
    symbols = []
    for i in range(0, len(bits), k):
        chunk = bits[i:i+k]
        decimal = int("".join(str(int(b)) for b in chunk), 2)
        angle = 2 * np.pi * decimal / order
        symbols.append(np.exp(1j * angle))
    # normalize power
    arr = np.array(symbols, dtype=np.complex64)
    arr /= np.sqrt(np.mean(np.abs(arr)**2))
    return arr

def qam(bits, order):
    """Square M-QAM mapping (e.g., 16, 64). Gray coding not implemented here."""
    k = int(np.log2(order))
    m = int(np.sqrt(order))
    if m * m != order:
        raise ValueError("Only square QAM orders supported (e.g., 16, 64).")
    bits = pad_bits(bits, k)
    levels = np.arange(-m + 1, m, 2)  # e.g., [-3, -1, 1, 3] for 16QAM
    symbols = []
    for i in range(0, len(bits), k):
        chunk = bits[i:i+k]
        decimal = int("".join(str(int(b)) for b in chunk), 2)
        I = levels[decimal % m]
        Q = levels[decimal // m]
        symbols.append(I + 1j * Q)
    arr = np.array(symbols, dtype=np.complex64)
    arr /= np.sqrt(np.mean(np.abs(arr)**2))  # power normalize
    return arr

# ==============================
# Helper: pick device
# ==============================
def open_sdr(preferred_driver="hackrf"):
    # Try to open a hackrf device, else fallback to any Soapy device
    args = dict(driver=preferred_driver)
    try:
        sdr = Device(args)
        print(f"[INFO] Opened SoapySDR device with driver='{preferred_driver}'")
        return sdr
    except Exception as e:
        print(f"[WARN] Could not open driver='{preferred_driver}': {e}. Trying any available device...")
        try:
            sdr = Device()  # let Soapy pick
            print("[INFO] Opened default SoapySDR device")
            return sdr
        except Exception as e2:
            print("[ERROR] No SoapySDR device found:", e2)
            raise

# ==============================
# MAIN
# ==============================
def main():
    MODULATION = input("[INPUT] Enter modulation (bpsk/qpsk/gmsk/cpfsk/8psk/16qam/64qam): ").strip().lower()

    # compute upsampling ratio (samples per symbol)
    samples_per_symbol = max(1, int(round(SAMPLE_RATE / SYMBOL_RATE)))
    if samples_per_symbol < 1:
        samples_per_symbol = 1

    # generate random bits
    bits = np.random.randint(0, 2, BITS_TO_GENERATE, dtype=np.int8)

    # create symbols according to modulation
    if MODULATION == "bpsk":
        syms = bpsk(bits)
    elif MODULATION == "qpsk":
        syms = qpsk(bits)
    elif MODULATION == "gmsk":
        syms = gmsk(bits, samples_per_symbol=samples_per_symbol)
    elif MODULATION == "cpfsk":
        syms = cpfsk(bits, samples_per_symbol=samples_per_symbol)
    elif MODULATION == "8psk":
        syms = psk(bits, order=8)
    elif MODULATION == "16qam":
        syms = qam(bits, order=16)
    elif MODULATION == "64qam":
        syms = qam(bits, order=64)
    else:
        print("[ERROR] Unsupported modulation:", MODULATION)
        sys.exit(1)

    # upsample non-CPM signals if returned symbols are per-symbol
    if MODULATION not in ("gmsk", "cpfsk"):
        # if syms is real for BPSK, convert to complex
        syms = np.asarray(syms, dtype=np.complex64)
        signal = np.repeat(syms, samples_per_symbol).astype(np.complex64)
    else:
        signal = np.asarray(syms, dtype=np.complex64)

    # optionally multiply by a low-frequency complex carrier (rarely needed for HackRF baseband)
    if CARRIER_FREQ and CARRIER_FREQ != 0:
        t = np.arange(len(signal)) / SAMPLE_RATE
        carrier = np.exp(1j * 2 * np.pi * CARRIER_FREQ * t).astype(np.complex64)
        signal = AMPLITUDE * signal * carrier
    else:
        # scale amplitude to avoid clipping; ensure complex dtype
        avg_power = np.mean(np.abs(signal)**2)
        if avg_power == 0:
            scale = 0.0
        else:
            scale = AMPLITUDE / np.sqrt(avg_power)
        signal = (signal * scale).astype(np.complex64)

    # open SDR
    sdr = open_sdr("hackrf")

    # configure TX channel 0 (error handling for different hardware)
    chan = 0
    try:
        sdr.setSampleRate(SOAPY_SDR_TX, chan, float(SAMPLE_RATE))
        sdr.setFrequency(SOAPY_SDR_TX, chan, float(CENTER_FREQ))
    except Exception as e:
        print("[WARN] setSampleRate/setFrequency failed:", e)

    # Try to set gain; different devices have different gain elements.
    try:
        sdr.setGain(SOAPY_SDR_TX, chan, float(TX_GAIN))
    except Exception:
        # try named elements
        try:
            sdr.setGain(SOAPY_SDR_TX, chan, "TX", float(TX_GAIN))
        except Exception:
            try:
                sdr.setGain(SOAPY_SDR_TX, chan, "AMP", float(TX_GAIN))
            except Exception as e:
                print("[WARN] Could not set gain cleanly:", e)

    # Try to set antenna if available
    try:
        sdr.setAntenna(SOAPY_SDR_TX, chan, "TX")
    except Exception:
        try:
            sdr.setAntenna(SOAPY_SDR_TX, chan, "ANT")
        except Exception:
            pass  # ignore if not applicable

    # setup stream and activate
    tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [chan])
    sdr.activateStream(tx_stream)

    print(f"[INFO] Transmitting {MODULATION.upper()} at {CENTER_FREQ/1e6:.6f} MHz (SR={SAMPLE_RATE/1e6:.3f} Msps) ...")
    try:
        ptr = 0
        total_len = len(signal)
        while True:
            # build chunk
            if ptr + CHUNK <= total_len:
                chunk = signal[ptr:ptr + CHUNK]
                ptr += CHUNK
            else:
                # wrap-around copy to produce continuous stream
                first = signal[ptr:]
                need = CHUNK - len(first)
                second = signal[:need]
                chunk = np.concatenate([first, second])
                ptr = need
            # ensure dtype
            chunk = np.asarray(chunk, dtype=np.complex64)
            sr = sdr.writeStream(tx_stream, [chunk], len(chunk))
            # sr is a tuple or int depending on soapy version; ignore return unless error
            # small sleep optional to avoid hogging CPU
            # time.sleep(0)  # not required; writeStream blocks
    except KeyboardInterrupt:
        print("\n[INFO] Transmission stopped by user.")
    finally:
        try:
            sdr.deactivateStream(tx_stream)
            sdr.closeStream(tx_stream)
        except Exception:
            pass

if __name__ == "__main__":
    main()
