import numpy as np
from rtlsdr import RtlSdr
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

# ------------ 1. Capture Samples with Live Plotting and STOP LIVE ------------
def capture_rtlsdr_samples(sample_rate=240e3, center_freq=957e6, gain=40.2, num_samples=2_000_000, window_size=1000):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain

    print(f"Receiving {num_samples} samples in real time...")
    samples = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))

    # Plot 1 — Magnitude
    line1, = ax1.plot([], [], lw=1)
    ax1.set_title("Live Signal Magnitude")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Magnitude")
    ax1.set_ylim(0, .1)
    ax1.set_xlim(0, window_size)
    ax1.grid(True)  
    
    # Plot 2 — Energy (live smooth trace)
    line2, = ax2.plot([], [], lw=1, color='green')
    ax2.set_title("Live Energy per Window")
    ax2.set_xlabel("Window #")
    ax2.set_ylabel("Energy")
    ax2.set_ylim(0, 0.0002)  # adjust based on energy scale
    ax2.grid(True)  
    energies_live = []  # energy history

    
    # STOP LIVE button (top-right)
    stop_ax = plt.axes([0.82, 0.93, 0.16, 0.05])
    stop_button = Button(stop_ax, 'STOP LIVE', color='#cc0000', hovercolor='#ff3333')
    stop_flag = {"stop": False}

    def on_stop(event):
        stop_flag["stop"] = True
        stop_button.ax.set_facecolor("#990000")
        fig.canvas.draw_idle()
        print("\n[INFO] STOP LIVE pressed. Finishing capture...")

    stop_button.on_clicked(on_stop)

    received = 0
    chunk_size = 64 * 1024

    try:
        while received < num_samples and not stop_flag["stop"]:
            chunk = sdr.read_samples(chunk_size)
            samples.extend(chunk)

            # LIVE PLOT: Magnitude
            window_mags = np.abs(chunk)
            line1.set_data(np.arange(len(window_mags)), window_mags)

            # LIVE PLOT: Energy
            energy = np.mean(window_mags ** 2)
            energies_live.append(energy)
            line2.set_data(np.arange(len(energies_live)), energies_live)
            ax2.set_xlim(max(0, len(energies_live) - 100), len(energies_live))  # scroll forward

            fig.canvas.draw()
            fig.canvas.flush_events()

            received += len(chunk)

    finally:
        sdr.close()

    print(f"\n[INFO] Capture complete. Received {len(samples)} samples.")
    plt.ioff()
    plt.show(block=False)
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

    # ---- NEW: Show energy over time BEFORE threshold selection ----
    times = np.arange(len(energies)) * (window_size / 240000.0)  # time in seconds

    plt.figure(figsize=(10,4))
    plt.plot(times, energies, lw=1)
    plt.title("Energy Over Time (Before Threshold Selection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.show()
    # ---------------------------------------------------------------

    # Continue exactly as before...
    plt.figure()
    plt.hist(energies, bins=100, color='blue', alpha=0.7)
    plt.xlabel("Mean Energy per Window")
    plt.ylabel("Count")
    plt.title("Energy Distribution - Choose Threshold")
    plt.grid(True)
    plt.show()

    while True:
        try:
            threshold = float(input("Enter threshold value based on histogram: "))
            break
        except ValueError:
            print("Invalid input. Please enter a float.")


    labels = [1 if e > threshold else 0 for e in energies]
    times = np.arange(len(energies)) * (window_size / 240000.0)

    #plt.figure()
    #plt.plot(times, energies, label="Energy")
    #plt.plot(times, labels, label="ON/OFF", linestyle='--')
    #plt.xlabel("Time (s)")
    #plt.ylabel("Energy / Status")
    #plt.title("Energy over Time with ON/OFF Labels")
    #plt.legend()
    #plt.grid(True)
    #plt.show()
    
    
    plt.figure(figsize=(10,6))

    # Plot 1 — Energy vs Time
    plt.subplot(2,1,1)
    plt.plot(times, energies, label="Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Energy over Time")
    plt.grid(True)

    # Plot 2 — ON/OFF vs Time
    plt.subplot(2,1,2)
    plt.plot(times, labels, label="ON/OFF", linestyle='--', color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Status (0 OFF / 1 ON)")
    plt.title("Detected ON/OFF Activity")
    plt.grid(True)

    plt.tight_layout()
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

    windows, labels, threshold = choose_threshold(samples)
    print(f"Using threshold = {threshold}")
    print(f"Windows: {windows.shape}, Labels: {labels.shape}")

    X, y = prepare_lstm_data(windows, labels)
    print(f"LSTM input shape: {X.shape}, y shape: {y.shape}")

    print("Training LSTM...")
    model, history = train_lstm_model(X, y, epochs=100)

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

