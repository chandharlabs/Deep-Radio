############ RTL-SDR Predict (Decimation x12 + Welch PSD 1024) + Naive Bayes ############

import numpy as np, scipy.signal as signal, time, sys, pickle
from rtlsdr import RtlSdr  # pip install pyrtlsdr

DECIMATION_RATE = 12  # REQUIRED — your SDR is raw, training .npy was already decimated

def compute_psd_features(iq, cfg):
    f, Pxx = signal.welch(iq, window=cfg["window"], nperseg=cfg["nperseg"],
                          noverlap=cfg["noverlap"], scaling=cfg["scaling"],
                          return_onesided=cfg["return_onesided"], detrend=False)
    Pxx_dB = 10*np.log10(Pxx + cfg["eps"])
    feat = signal.resample(Pxx_dB, cfg["n_features"])
    fmin, fmax = feat.min(), feat.max()
    return ((feat - fmin)/(fmax - fmin)).astype(np.float32)

def read_sdr(freq, sr):
    sdr = RtlSdr()
    sdr.sample_rate = sr
    sdr.err_ppm = 56
    sdr.gain = 40.2
    f_offset = 250e3
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq = sdr.read_samples(1_221_376)[:600_000]
    sdr.close()
    return (iq * np.exp(-1j*2*np.pi*f_offset/sr*np.arange(len(iq)))).astype(np.complex64)

if __name__ == "__main__":
    bundle = pickle.load(open("model_nb_welchpsd.sav","rb"))
    model = bundle["model"]
    cfg   = bundle["feature_cfg"]
    classes = bundle["class_array"]

    sample_rate = 2.4e6
    freq = 957e6

    iq = read_sdr(freq, sample_rate)
    print("Decimating x12 ...")
    iq = signal.decimate(iq, DECIMATION_RATE, zero_phase=True)
    # 3) Welch PSD features
    feat = compute_psd_features(iq, cfg).reshape(1, -1)

    # ✅ FIX FOR NaN / INF
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    # now prediction is safe
    pred = classes[int(model.predict(feat)[0])]

    print("\nPredicted:", pred)

