import numpy as np
from keras.models import load_model
from rtlsdr import RtlSdr
import time

# --------- Function to Capture New Samples Using RTL-SDR ---------

def capture_rtlsdr_samples(sample_rate=240e3, center_freq=947.2e6, gain=40.2, num_samples=1_000_000, chunk_size=256*1024):
    from rtlsdr import RtlSdr
    sdr = RtlSdr()

    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain

    print(f"Receiving {num_samples} samples in chunks...")

    samples = []
    remaining = num_samples

    try:
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            chunk = sdr.read_samples(to_read)
            samples.append(chunk)
            remaining -= to_read
    except Exception as e:
        print("Error during SDR read:", e)
    finally:
        sdr.close()

    all_samples = np.concatenate(samples)
    print(f"Capture complete. Received {len(all_samples)} samples.")
    return all_samples


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
def predict_activity(model_path="rtlsdr_lstm_model.h5"):
    model = load_model(model_path)

    samples = capture_rtlsdr_samples()
    windows = get_windows(samples)
    X = prepare_lstm_input(windows)

    print(f"Predicting on {X.shape[0]} sequences...")
    predictions = model.predict(X)

    for i, p in enumerate(predictions):
        status = "ON" if p[0] > 0.5 else "OFF"
        print(f"Time Window {i}: Activity {status} (Confidence: {p[0]:.2f})")

# --------- Run ---------
if __name__ == "__main__":
    predict_activity()

