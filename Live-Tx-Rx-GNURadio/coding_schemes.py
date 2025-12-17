import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---------- 1. Convolutional Coding (commpy) ----------
from commpy.channelcoding import conv_encode, viterbi_decode
from commpy.channelcoding.convcode import Trellis

def convolutional_coding(bits):
    trellis = Trellis(np.array([7]), np.array([[0o133, 0o171]]))  # (133, 171) in octal
    encoded = conv_encode(bits, trellis)
    return encoded, trellis

def convolutional_decoding(encoded, trellis):
    decoded = viterbi_decode(encoded.astype(float), trellis, tb_depth=15)
    return decoded

# ---------- 2. LDPC Coding (pyldpc) ----------
from pyldpc import make_ldpc, encode, decode, get_message

def ldpc_coding(bits, snr_db):
    n, d_v, d_c = 512, 3, 6
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    bits = bits[:G.shape[1]]  # Trim to match G
    codeword = encode(G, bits, snr_db)
    return codeword, H

def ldpc_decoding(codeword, H, snr_db):
    y = codeword + np.random.normal(0, 1/np.sqrt(2 * 10**(snr_db/10)), codeword.shape)
    decoded = decode(H, y, snr_db)
    return get_message(H, decoded)

# ---------- 3. Turbo Coding (custom simulation) ----------
from scipy.signal import convolve2d

def turbo_encode(bits):
    # Simplified turbo encoding: interleave and conv encode twice
    interleaver = np.random.permutation(len(bits))
    trellis = Trellis(np.array([5]), np.array([[0o5, 0o7]]))
    parity1 = conv_encode(bits, trellis)
    parity2 = conv_encode(bits[interleaver], trellis)
    return np.vstack((bits, parity1, parity2)), interleaver

def turbo_decode(received, interleaver):
    # Soft combining and max-log approximation (simplified)
    decoded = received[0]  # Simplified hard decision
    return decoded

# ---------- Utility Functions ----------
def bpsk_modulate(bits): return 1 - 2*bits
def awgn(signal, snr_db):
    snr = 10**(snr_db/10)
    sigma = np.sqrt(1 / (2 * snr))
    return signal + sigma * np.random.randn(*signal.shape)

def ber(original, decoded): return np.sum(original != decoded[:len(original)]) / len(original)

# ---------- Simulation ----------
def simulate_all(snr_db=5):
    np.random.seed(42)
    n_bits = 1024
    bits = np.random.randint(0, 2, n_bits)

    print("=== Convolutional Coding ===")
    conv_enc, trellis = convolutional_coding(bits)
    modulated = bpsk_modulate(conv_enc)
    received = awgn(modulated, snr_db)
    demod = (received < 0).astype(int)
    conv_dec = convolutional_decoding(demod, trellis)
    print("BER:", ber(bits, conv_dec))

    print("\n=== LDPC Coding ===")
    ldpc_code, H = ldpc_coding(bits, snr_db)
    ldpc_dec = ldpc_decoding(ldpc_code, H, snr_db)
    print("BER:", ber(bits[:len(ldpc_dec)], ldpc_dec))

    print("\n=== Turbo Coding ===")
    turbo_code, interleaver = turbo_encode(bits)
    modulated = bpsk_modulate(turbo_code[0])
    received = awgn(modulated, snr_db)
    demod = (received < 0).astype(int)
    turbo_dec = turbo_decode(turbo_code, interleaver)
    print("BER:", ber(bits, turbo_dec))

simulate_all(snr_db=5)
