############ RTL-SDR Predict with Decimation x12 + Welch PSD (1024) + Decision Tree ############

from __future__ import division, print_function
import numpy as np
import scipy.signal as signal
import time, sys, pickle
from rtlsdr import RtlSdr  # pip install pyrtlsdr

t1 = time.time()

def compute_psd_features(iq_samples, cfg):
    """Welch PSD -> log10 -> resample -> per-sample min-max [0,1]."""
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
    return np.zeros_like(feat, dtype=np.float32) if (fmax - fmin) < 1e-9 else \
           ((feat - fmin) / (fmax - fmin)).astype(np.float32)

def read_samples_sdr(freq_hz, sample_rate_hz, ppm=56, gain=40.2, f_offset=250_000):
    """
    Capture IQ from RTL-SDR, tune to (freq - f_offset) to dodge DC, then mix back by +f_offset.
    """
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

def predict_live(freq_hz, sample_rate_hz, bundle):
    classes = bundle["class_array"]
    cfg = bundle["feature_cfg"]
    decim = bundle.get("decimation_rate", 12)
    model = bundle["model"]

    # 1) Capture
    iq = read_samples_sdr(freq_hz, sample_rate_hz)

    # 2) Decimate x12 (matching training)
    iq_dec = signal.decimate(iq, decim, zero_phase=True)

    # 3) Welch PSD features
    feat = compute_psd_features(iq_dec, cfg).reshape(1, -1)

    # 4) Predict
    pred_idx = int(model.predict(feat)[0])
    print("Predicted:", classes[pred_idx])
    return classes[pred_idx]

########### MAIN ###########
if __name__ == "__main__":
    # Load the decimated-FFT model
    MODEL_PATH = "model_dt_welchpsd.sav"
    bundle = pickle.load(open(MODEL_PATH, "rb"))

    sample_rate = 2.4e6
    freq = 957e6

    if len(sys.argv) >= 2:
        freq = float(sys.argv[1])
    if len(sys.argv) >= 3:
        sample_rate = float(sys.argv[2])

    predict_live(freq, sample_rate, bundle)
    print("Time taken: %.2fs" % (time.time() - t1))

