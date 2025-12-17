############ LIVE RTL-SDR PSD + FEATURE VIEWER ############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import time

# ------------------- SDR SETTINGS --------------------
freq = 957e6          # change frequency here
sample_rate = 2.4e6   # RTL-SDR raw sample rate
ppm = 56              # Adjust for your dongle
gain = 40.2           # or 'auto'
f_offset = 250_000    # to dodge DC spike before mixing back

# ------------------- CAPTURE FROM RTL-SDR --------------------
def read_samples_sdr(freq_hz, sample_rate_hz):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate_hz
    sdr.err_ppm = ppm
    sdr.gain = gain
    sdr.center_freq = freq_hz - f_offset   # tune with DC offset
    time.sleep(0.06)

    iq = sdr.read_samples(1_221_376)[:600_000]  # take 600k samples
    sdr.close()

    # Mix back DC offset
    n = np.arange(len(iq))
    mixer = np.exp(-1j * 2.0 * np.pi * f_offset / sample_rate_hz * n)
    return (iq * mixer).astype(np.complex64)

print("Reading from RTL-SDR...")
iq = read_samples_sdr(freq, sample_rate)
print("IQ samples shape:", iq.shape)

# ------------------- WELCH PSD + FEATURES --------------
FEATURE_CFG = {
    "nperseg": 4096,
    "noverlap": 2048,
    "window": "hann",
    "scaling": "density",
    "return_onesided": True,
    "n_features": 1024,
    "eps": 1e-12
}

def compute_psd_and_features(iq_samples, cfg):
    f, Pxx = signal.welch(
        iq_samples,
        window=cfg["window"],
        nperseg=cfg["nperseg"],
        noverlap=cfg["noverlap"],
        scaling=cfg["scaling"],
        return_onesided=cfg["return_onesided"],
        detrend=False
    )
    Pxx_dB = 10 * np.log10(Pxx + cfg["eps"])
    feat = signal.resample(Pxx_dB, cfg["n_features"])
    fmin, fmax = feat.min(), feat.max()
    feat_norm = ((feat - fmin) / (fmax - fmin)).astype(np.float32)
    return f, Pxx_dB, feat_norm

# compute PSD + features
f, Pxx_dB, features = compute_psd_and_features(iq, FEATURE_CFG)

# ------------------- PRINT FEATURES ---------------------
print("\nFirst 20 normalized FFT-PSD features:\n", features[:20])

# ------------------- PLOT PSD ---------------------------
plt.figure(figsize=(8,4))
plt.plot(f, Pxx_dB)
plt.title("Welch Power Spectral Density (from RTL-SDR)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid(True)
plt.show()

