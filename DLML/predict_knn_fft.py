############ RTL-SDR Predict with Welch-PSD (log FFT) + KNN #############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time
import os, sys
import pickle

# RTL-SDR
from rtlsdr import RtlSdr

t1 = time.time()

# Keep your requested decimation
DECIMATION_RATE = 12

def compute_psd_features(iq_samples, cfg):
    """
    Welch PSD (density) -> 10*log10 -> resample to n_features -> per-sample min-max [0,1]
    Uses the same config saved with the model.
    """
    f, Pxx = signal.welch(
        x=iq_samples,
        window=cfg.get("window", "hann"),
        nperseg=cfg.get("nperseg", 4096),
        noverlap=cfg.get("noverlap", 2048),
        return_onesided=cfg.get("return_onesided", True),
        scaling=cfg.get("scaling", "density"),
        detrend=False
    )

    # Log power (dB)
    eps = cfg.get("eps", 1e-12)
    Pxx_dB = 10.0 * np.log10(Pxx + eps)

    # Fixed-length features
    n_features = cfg.get("n_features", 1024)
    if Pxx_dB.size != n_features:
        feat = signal.resample(Pxx_dB, n_features)
    else:
        feat = Pxx_dB

    # Normalize 0–1 (per sample)
    fmin, fmax = feat.min(), feat.max()
    if fmax - fmin < 1e-9:
        feat_norm = np.zeros_like(feat, dtype=np.float32)
    else:
        feat_norm = ((feat - fmin) / (fmax - fmin)).astype(np.float32)

    return feat_norm

def read_samples_sdr(freq_hz, sample_rate_hz, ppm=56, gain=40.2, f_offset=250_000):
    """
    Capture IQ from RTL-SDR, tune freq - f_offset to dodge DC, mix back by +f_offset.
    Returns complex64 numpy array.
    """
    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate_hz
        sdr.err_ppm = ppm
        sdr.gain = gain  # or 'auto'
        sdr.center_freq = freq_hz - f_offset
        time.sleep(0.06)

        # Grab ~1.22M samples then trim to 600k (same as your original)
        iq_samples = sdr.read_samples(1_221_376)
        iq_samples = iq_samples[:600_000]

        # Mix back the offset
        n = np.arange(len(iq_samples))
        mixer = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate_hz * n)
        iq_samples = (iq_samples * mixer).astype(np.complex64)
        return iq_samples
    finally:
        sdr.close()

def predict_live(freq_hz, sample_rate_hz, model_bundle):
    """
    Live pipeline:
      SDR capture -> decimate x12 -> Welch PSD features -> predict with KNN
    """
    classes = model_bundle.get("class_array", ['BPSK','QPSK','GMSK'])
    cfg = model_bundle["feature_cfg"]
    model = model_bundle["model"]

    # 1) Capture
    iq = read_samples_sdr(freq_hz, sample_rate_hz)

    # 2) Decimate (as requested)
    iq_dec = signal.decimate(iq, DECIMATION_RATE, zero_phase=True)


    # 3) Features (Welch PSD log-power)
    feat = compute_psd_features(iq_dec, cfg).reshape(1, -1)

    # ✅ SAFETY FIX (remove NaN / inf)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    # 4) Predict
    pred_idx = int(model.predict(feat)[0])

    pred_label = classes[pred_idx]

    # Confidence (KNN supports predict_proba)
    conf_txt = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        conf = float(np.max(proba))
        conf_txt = " (confidence: {:.2f})".format(conf)

    print("Predicted:", pred_label + conf_txt)
    return pred_label

########### Main #################################
if __name__ == "__main__":
    # Load the FFT-based KNN model you confirmed
    MODEL_PATH = "model_knn_welchpsd.sav"
    model_bundle = pickle.load(open(MODEL_PATH, "rb"))

    # Defaults (override via CLI: python predict.py <freq_Hz> <samp_Hz>)
    sample_rate = 2.4e6
    freq = 957e6

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

    predict_live(freq, sample_rate, model_bundle)
    print("Time taken: {:.2f}s".format(time.time() - t1))

