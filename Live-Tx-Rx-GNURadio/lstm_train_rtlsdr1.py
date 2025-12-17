import numpy as np
from rtlsdr import RtlSdr
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# ------------ 1. Capture Samples with Live Plotting ------------
def capture_rtlsdr_samples(sample_rate=240e3, center_freq=947.2e6, gain=40.2, num_samples=2_00_000, window_size=1000):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain

    print(f"Receiving {num_samples} samples in real time...")
    samples = []
    mags_live = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1)
    ax.set_title("Live Signal Magnitude")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Magnitude")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, window_size)

    def update_plot(window_mags):
        line.set_ydata(window_mags)
        line.set_xdata(np.arange(len(window_mags)))
        fig.canvas.draw()
        fig.canvas.flush_events()

    received = 0
    while received < num_samples:
        to_read = min(window_size, num_samples - received)
        chunk = sdr.read_samples(64*1024)
        samples.extend(chunk)
        window_mags = np.abs(chunk)
        update_plot(window_mags)
        received += len(chunk)

    sdr.close()
    print(f"Capture complete. Received {len(samples)} samples.")
    return np.array(samples)

# ------------ 2. Energy Histogram + Time Plot + Labeling ------------
def choose_threshold(samples, window_size=200):
    mags = np.abs(samples)
    energies = []
    windows = []

    for i in range(0, len(mags) - window_size, window_size):
        window = mags[i:i + window_size]
        energy = np.mean(window**2)
        energies.append(energy)
        windows.append(window)

    # Plot histogram of energy
    plt.figure()
    plt.hist(energies, bins=100, color='blue', alpha=0.7)
    plt.xlabel("Mean Energy per Window")
    plt.ylabel("Count")
    plt.title("Energy Distribution - Choose Threshold")
    plt.grid(True)
    plt.show()

    # Input threshold from user
    while True:
        try:
            threshold = float(input("Enter threshold value based on histogram: "))
            break
        except ValueError:
            print("Invalid input. Please enter a float.")

    # Label data based on threshold
    labels = [1 if e > threshold else 0 for e in energies]
    times = np.arange(len(energies)) * (window_size / 240000.0)  # seconds

    # Plot energy over time with labels
    plt.figure()
    plt.plot(times, energies, label="Energy")
    plt.plot(times, labels, label="ON/OFF", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Energy / Status")
    plt.title("Energy over Time with ON/OFF Labels")
    plt.legend()
    plt.grid(True)
    plt.show()

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
    #time.sleep(10)
    windows, labels, threshold = choose_threshold(samples)
    print(f"Using threshold = {threshold}")
    print(f"Windows: {windows.shape}, Labels: {labels.shape}")

    X, y = prepare_lstm_data(windows, labels)
    print(f"LSTM input shape: {X.shape}, y shape: {y.shape}")

    print("Training LSTM...")
    model, history = train_lstm_model(X, y, epochs=10)

    model.save("rtlsdr_lstm_model.h5")
    print("Model saved as 'rtlsdr_lstm_model.h5'")

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

