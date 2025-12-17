import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ------------------- LOAD YOUR FILE --------------------
fname = "samples-etgkqhmozvoxrttz.npy"   # change only if filename differs
iq = np.load(fname)

print("Loaded IQ shape:", iq.shape)

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

# compute spectrum + features
f, Pxx_dB, features = compute_psd_and_features(iq, FEATURE_CFG)

# ------------------- PRINT FEATURES ---------------------
print("\nFirst 20 normalized FFT-PSD features:\n", features[:20])

# ------------------- PLOT PSD ---------------------------
plt.figure(figsize=(8,4))
plt.plot(f, Pxx_dB)
plt.title("Welch Power Spectral Density (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid(True)
plt.show()


# ----- PRINT FEATURE NAMES (FREQUENCY + NORMALIZED FEATURE VALUE) -----
print("\n--- First 20 FFT-based PSD Features ---")
for i in range(20):  # change 20 to any number if needed
    print(f"Feature {i:3d}  |  Freq = {f[i]:10.2f} Hz  |  Value = {features[i]:.6f}")

