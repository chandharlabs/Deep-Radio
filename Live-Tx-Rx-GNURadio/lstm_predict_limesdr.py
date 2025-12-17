import numpy as np
from keras.models import load_model
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time

# --------- Function to Capture New Samples Using LimeSDR ---------
def capture_limesdr_samples(sample_rate=1e6, center_freq=881e6, gain=60, num_samples=1000000):
    args = dict(driver="lime")
    sdr = SoapySDR.Device(args)

    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
    sdr.setGain(SOAPY_SDR_RX, 0, gain)
    sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAW")

    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)

    buff = np.empty(num_samples, dtype=np.complex64)
    num_recv = 0
    print("Receiving new samples...")
    while num_recv < num_samples:
        sr = sdr.readStream(rxStream, [buff[num_recv:]], num_samples - num_recv)
        if sr.ret > 0:
            num_recv += sr.ret
        else:
            print("Stream read error:", sr.ret)

    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

    print("New capture complete.")
    return buff

# --------- Energy Detection and Label-Free Windowing ---------
def get_windows(samples, window_size=1000):
    mags = np.abs(samples)
    windows = []
    for i in range(0, len(mags) - window_size, window_size):
        window = mags[i:i + window_size]
        windows.append(window)
    return np.array(windows)

# --------- Prepare LSTM Input Sequences ---------
def prepare_lstm_input(windows, seq_len=10):
    X = []
    for i in range(len(windows) - seq_len):
        X.append(windows[i:i + seq_len])
    return np.array(X)

# --------- Predict Activity (ON/OFF) Using Trained Model ---------
def predict_activity(model_path="limesdr_lstm_model.h5"):
    model = load_model(model_path)

    samples = capture_limesdr_samples()
    windows = get_windows(samples)
    X = prepare_lstm_input(windows)
    #X = X.reshape((X.shape[0], X.shape[1], 1))

    print(f"Predicting on {X.shape[0]} sequences...")
    predictions = model.predict(X)

    for i, p in enumerate(predictions):
        status = "ON" if p[0] > 0.5 else "OFF"
        print(f"Time Window {i}: Activity {status} (Confidence: {p[0]:.2f})")

# --------- Run ---------
if __name__ == "__main__":
    predict_activity()

