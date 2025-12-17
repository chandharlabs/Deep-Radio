############ Predict with Welch-PSD (log FFT) features #############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time
import os, sys
import pickle

# RTL-SDR
from rtlsdr import RtlSdr

t1 = time.time()

# =========================
# Welch-PSD Feature Config
# (must match training)
# =========================
FEATURE_CFG = {
    "nperseg": 4096,       # Welch window length
    "noverlap": 2048,      # 50% overlap
    "window": "hann",
    "scaling": "density",
    "return_onesided": True,
    "n_features": 1024,    # final feature vector length after resampling
    "eps": 1e-12,          # numerical stability for log
}

DECIMATION_RATE = 12      # you requested to keep this

def compute_psd_features(iq_samples, cfg=FEATURE_CFG):
    """
    Compute Welch PSD (log power) features and resample to fixed length.
      1) Welch PSD
      2) 10*log10(PSD + eps)
      3) Resample to cfg['n_features']
      4) Per-sample min-max normalize to [0,1]
    Returns float32 vector of length n_features.
    """
    f, Pxx = signal.welch(
        x=iq_samples,
        window=cfg["window"],
        nperseg=cfg["nperseg"],
        noverlap=cfg["noverlap"],
        return_onesided=cfg["return_onesided"],
        scaling=cfg["scaling"],
        detrend=False
    )

    # Log power (dB)
    Pxx_dB = 10.0 * np.log10(Pxx + cfg["eps"])

    # Fixed-length feature vector
    if Pxx_dB.size != cfg["n_features"]:
        feat = signal.resample(Pxx_dB, cfg["n_features"])
    else:
        feat = Pxx_dB

    # Per-sample min-max normalization
    fmin, fmax = feat.min(), feat.max()
    if fmax - fmin < 1e-9:
        feat_norm = np.zeros_like(feat, dtype=np.float32)
    else:
        feat_norm = ((feat - fmin) / (fmax - fmin)).astype(np.float32)

    return feat_norm

def read_samples_sdr(freq_hz, sample_rate_hz):
    """
    Capture IQ from RTL-SDR, shift by +250k to dodge DC, then mix back.
    Returns complex64 numpy array.
    """
    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate_hz
        sdr.err_ppm = 56       # adjust to your dongle
        sdr.gain = 40.2        # or 'auto'
        f_offset = 250_000     # 250 kHz offset
        sdr.center_freq = freq_hz - f_offset
        time.sleep(0.06)

        # capture a bit over 1.2M samples then trim (same as your original)
        iq_samples = sdr.read_samples(1_221_376)
        iq_samples = iq_samples[:600_000]

        # mix back the offset
        n = np.arange(len(iq_samples))
        fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate_hz * n)
        iq_samples = iq_samples * fc1
        return iq_samples.astype(np.complex64)
    finally:
        sdr.close()

def predict_mod(freq_hz, model, classes, sample_rate_hz):
    """
    Capture → decimate ×12 → Welch PSD log-power → resample → normalize → predict
    """
    # Capture from SDR
    iq = read_samples_sdr(freq_hz, sample_rate_hz)

    # Decimate (keep as requested)
    iq_dec = signal.decimate(iq, DECIMATION_RATE, zero_phase=True)

    # Welch PSD features (on decimated IQ)
    feat = compute_psd_features(iq_dec, FEATURE_CFG).reshape(1, -1)

    # Predict
    pred = model["model"].predict(feat)
    pred_idx = int(pred[0])
    pred_label = classes[pred_idx]

    # Confidence (if RF with predict_proba)
    conf_txt = ""
    if hasattr(model["model"], "predict_proba"):
        proba = model["model"].predict_proba(feat)[0]
        conf = float(np.max(proba))
        conf_txt = "  (confidence: {:.2f})".format(conf)

    print("Predicted:", pred_label + conf_txt)
    return pred_label

########### Main #################################

if __name__ == "__main__":
    # Classes (must match training order)
    class_array = ['BPSK', 'QPSK', 'GMSK']

    # Load trained model (Welch PSD RF)
    # You confirmed this filename:
    MODEL_PATH = 'model_rf_welchpsd.sav'
    model = pickle.load(open(MODEL_PATH, 'rb'))

    # SDR params (same as your script)
    sample_rate = 2.4e6
    freq = 957e6

    # Optional: allow CLI overrides
    if len(sys.argv) >= 2:
        try:
            freq = float(sys.argv[1])
        except:
            pass
    if len(sys.argv) >= 3:
        try:
            sample_rate = float(sys.argv[2])
        except:
            pass

    predict_mod(freq, model, class_array, sample_rate)
    print("Time taken: {:.2f}s".format(time.time() - t1))

