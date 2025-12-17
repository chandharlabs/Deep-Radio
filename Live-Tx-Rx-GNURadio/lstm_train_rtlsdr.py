import numpy as np
from rtlsdr import RtlSdr
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import time

# ------------ 1. Capture Samples from RTL-SDR ------------
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

# ------------ 2. Energy Histogram and Threshold Input ------------
def choose_threshold(samples, window_size=1000):
    mags = np.abs(samples)
    energies = []
    windows = []

    for i in range(0, len(mags) - window_size, window_size):
        window = mags[i:i + window_size]
        energy = np.mean(window**2)
        energies.append(energy)
        windows.append(window)

    # Plot histogram
    plt.hist(energies, bins=100, color='blue', alpha=0.7)
    plt.xlabel("Mean Energy per Window")
    plt.ylabel("Count")
    plt.title("Energy Distribution - Choose Threshold")
    plt.grid(True)
    plt.show()

    # Ask user for threshold input
    while True:
        try:
            threshold = float(input("Enter threshold value based on histogram: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid float.")
    
    labels = [1 if e > threshold else 0 for e in energies]
    return np.array(windows), np.array(labels), threshold

# ------------ 3. Prepare LSTM Data ------------
def prepare_lstm_data(windows, labels, seq_len=10):
    X, y = [], []
    for i in range(len(windows) - seq_len):
        X.append(windows[i:i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X), np.array(y)

# ------------ 4. Train LSTM Model ------------
def train_lstm_model(X, y, epochs=10):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    return model, history

# ------------ 5. Main Pipeline ------------
def main():
    samples = capture_rtlsdr_samples()
    print(f"Captured {len(samples)} samples.")

    windows, labels, threshold = choose_threshold(samples)
    print(f"Using threshold = {threshold}")
    print(f"Windows: {windows.shape}, Labels: {labels.shape}")

    X, y = prepare_lstm_data(windows, labels)
    print(f"LSTM input shape: {X.shape}, y shape: {y.shape}")

    print("Training LSTM...")
    model, history = train_lstm_model(X, y, epochs=10)

    model.save("rtlsdr_lstm_model.h5")
    print("Model saved as 'rtlsdr_lstm_model.h5'")

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

