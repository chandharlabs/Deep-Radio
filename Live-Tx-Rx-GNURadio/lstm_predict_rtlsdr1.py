import numpy as np
from keras.models import load_model
from rtlsdr import RtlSdr
import time
import csv

# --------- Function to Capture New Samples Using RTL-SDR ---------
def capture_rtlsdr_samples(sample_rate=240e3, center_freq=947.2e6, gain=40.2,
                           num_samples=200_000, chunk_size=65536):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain

    print(f"Receiving {num_samples} samples in real time...")
    samples = []
    remaining = num_samples

    try:
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            try:
                chunk = sdr.read_samples(to_read)
                samples.append(chunk)
                remaining -= to_read
            except Exception as e:
                print("Error during SDR read:", e)
                break
    finally:
        sdr.close()

    all_samples = np.concatenate(samples)
    print(f"Capture complete. Received {len(all_samples)} samples.")
    return all_samples

# --------- Energy Detection and Windowing ---------
def get_windows(samples, window_size=200):
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

# --------- Predict Activity and Save to CSV ---------
def predict_activity(model_path="rtlsdr_lstm_model.h5", output_csv="predictions.csv"):
    model = load_model(model_path)

    samples = capture_rtlsdr_samples()
    windows = get_windows(samples, window_size=200)
    X = prepare_lstm_input(windows, seq_len=10)

    print(f"Predicting on {X.shape[0]} sequences... (shape = {X.shape})")
    predictions = model.predict(X)

    # Calculate timestamps
    sample_rate = 240e3
    window_size = 200
    window_duration_sec = window_size / sample_rate
    sequence_duration_sec = window_duration_sec * 10
    start_time = time.time()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Time", "Status", "Confidence", "MeanEnergy"])

        for i, (seq, p) in enumerate(zip(X, predictions)):
            status = "ON" if p[0] > 0.5 else "OFF"
            timestamp = start_time + i * sequence_duration_sec
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            mean_energy = np.mean(seq)  # Overall energy in 10-window sequence

            print(f"[{time_str}] Time Window {i}: Activity {status} (Confidence: {p[0]:.2f})")
            writer.writerow([i, time_str, status, f"{p[0]:.4f}", f"{mean_energy:.6f}"])

    print(f"\nPredictions saved to: {output_csv}")

# --------- Run ---------
if __name__ == "__main__":
    predict_activity()

