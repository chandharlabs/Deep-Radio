############ RTL-SDR Predict with Welch-PSD (log FFT) + SVM #############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time, sys, pickle
from rtlsdr import RtlSdr  # Ensure installed: pip install pyrtlsdr

t1 = time.time()

def compute_psd_features(iq_samples, cfg):
    """ Welch PSD -> log10 -> resample -> 0–1 normalize (per sample). """
    f, Pxx = signal.welch(
        x=iq_samples,
        window=cfg.get("window", "hann"),
        nperseg=cfg.get("nperseg", 4096),
        noverlap=cfg.get("noverlap", 2048),
        return_onesided=cfg.get("return_onesided", True),
        scaling=cfg.get("scaling", "density"),
        detrend=False
    )
    eps = cfg.get("eps", 1e-12)
    Pxx_dB = 10.0 * np.log10(Pxx + eps)

    nf = cfg.get("n_features", 1024)
    feat = signal.resample(Pxx_dB, nf) if Pxx_dB.size != nf else Pxx_dB

    fmin, fmax = feat.min(), feat.max()
    return np.zeros_like(feat, dtype=np.float32) if fmax - fmin < 1e-9 else \
           ((feat - fmin) / (fmax - fmin)).astype(np.float32)

def read_samples_sdr(freq_hz, sample_rate_hz, ppm=56, gain=40.2, f_offset=250_000):
    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate_hz
        sdr.err_ppm = ppm
        sdr.gain = gain
        sdr.center_freq = freq_hz - f_offset
        time.sleep(0.06)

        iq = sdr.read_samples(1_221_376)[:600_000]
        mixer = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate_hz * np.arange(len(iq)))
        return (iq * mixer).astype(np.complex64)
    finally:
        sdr.close()

def predict_sdr(freq_hz, sample_rate_hz, model_bundle):
    classes = model_bundle["class_array"]
    cfg = model_bundle["feature_cfg"]
    decim = model_bundle.get("decimation_rate", 12)
    model = model_bundle["model"]

    iq = read_samples_sdr(freq_hz, sample_rate_hz)   
    iq_dec = signal.decimate(iq, decim, zero_phase=True)
    feat = compute_psd_features(iq, cfg).reshape(1, -1)

    pred_idx = int(model.predict(feat)[0])
    pred_label = classes[pred_idx]

    conf_txt = ""
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(feat)[0]))
        conf_txt = f" (confidence: {conf:.2f})"

    print("Predicted:", pred_label + conf_txt)
    return pred_label

########### MAIN #################################
if __name__ == "__main__":
    model = pickle.load(open("model_svm_welchpsd.sav", "rb"))

    sample_rate = 2.4e6
    freq = 957e6

    if len(sys.argv) >= 2:
        freq = float(sys.argv[1])
    if len(sys.argv) >= 3:
        sample_rate = float(sys.argv[2])

    predict_sdr(freq, sample_rate, model)
    print("Time taken: %.2fs" % (time.time() - t1))

