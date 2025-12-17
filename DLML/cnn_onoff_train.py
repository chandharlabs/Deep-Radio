import numpy as np
from rtlsdr import RtlSdr
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

# ------------ 1) Capture with STOP LIVE + dual live plots ------------
def capture_rtlsdr_samples(sample_rate=240e3, center_freq=957e6, gain=40.2,
                           num_samples=2_000_000, window_size=1000):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain

    print(f"Receiving {num_samples} samples in real time...")
    samples = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

    # Plot 1 — Magnitude of current chunk
    line1, = ax1.plot([], [], lw=1)
    ax1.set_title("Live Signal Magnitude")
    ax1.set_xlabel("Sample Index (chunk)")
    ax1.set_ylabel("Magnitude")
    ax1.set_ylim(0, 0.1)
    ax1.set_xlim(0, window_size)
    ax1.grid(True)

    # Plot 2 — Energy per chunk (scrolling)
    line2, = ax2.plot([], [], lw=1, color='green')
    ax2.set_title("Live Energy per Chunk")
    ax2.set_xlabel("Chunk #")
    ax2.set_ylabel("Energy")
    ax2.set_ylim(0, 2e-4)  # adjust if needed
    ax2.grid(True)
    energies_live = []

    # STOP LIVE button (top-right)
    stop_ax = plt.axes([0.82, 0.93, 0.16, 0.05])
    stop_button = Button(stop_ax, "STOP LIVE", color="#cc0000", hovercolor="#ff3333")
    stop_flag = {"stop": False}

    def on_stop(_):
        stop_flag["stop"] = True
        stop_button.ax.set_facecolor("#990000")
        fig.canvas.draw_idle()
        print("\n[INFO] STOP LIVE pressed. Finishing capture...")

    stop_button.on_clicked(on_stop)

    received = 0
    chunk_size = 65536  # adjust to your dongle; smaller (4096-16384) reduces overflows

    try:
        while received < num_samples and not stop_flag["stop"]:
            to_read = min(chunk_size, num_samples - received)
            try:
                chunk = sdr.read_samples(to_read)
            except Exception as e:
                print("SDR Read Error:", e)
                time.sleep(0.05)
                continue

            samples.extend(chunk)

            # Live magnitude of this chunk
            window_mags = np.abs(chunk)
            n = len(window_mags)
            if n > 0:
                # LIVE PLOT: Magnitude (limit to window_size)
                n = min(len(window_mags), window_size)
                line1.set_data(np.arange(n), window_mags[:n])
                ax1.set_xlim(0, n)


            # Live energy trace
            energy = float(np.mean(window_mags ** 2)) if n > 0 else 0.0
            energies_live.append(energy)
            line2.set_data(np.arange(len(energies_live)), energies_live)
            # scroll last 200 points
            right = len(energies_live)
            left = max(0, right - 200)
            ax2.set_xlim(left, right)

            fig.canvas.draw()
            fig.canvas.flush_events()

            received += n
            # small pause helps avoid USB overflow on some hosts
            # time.sleep(0.001)

    finally:
        sdr.close()

    print(f"\n[INFO] Capture complete. Received {len(samples)} samples.")
    plt.ioff()
    plt.show(block=False)  # keep the window open / frozen
    return np.array(samples)

# ------------ 2) Energy over time (before threshold) + Histogram + Labels ------------
def choose_threshold(samples, window_size=200, sample_rate=240e3):
    mags = np.abs(samples)
    energies = []
    windows = []

    for i in range(0, len(mags) - window_size, window_size):
        window = mags[i:i + window_size]
        energy = float(np.mean(window**2))
        energies.append(energy)
        windows.append(window)

    energies = np.array(energies, dtype=np.float64)
    windows = np.array(windows, dtype=np.float32)

    # (a) Show energy over time BEFORE threshold selection
    times = np.arange(len(energies)) * (window_size / float(sample_rate))  # seconds
    plt.figure(figsize=(10, 4))
    plt.plot(times, energies, lw=1)
    plt.title("Energy Over Time (Before Threshold Selection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.show()

    # (b) Histogram for threshold picking
    plt.figure()
    plt.hist(energies, bins=100, color='blue', alpha=0.7)
    plt.xlabel("Mean Energy per Window")
    plt.ylabel("Count")
    plt.title("Energy Distribution - Choose Threshold")
    plt.grid(True)
    plt.show()

    # Manual threshold
    while True:
        try:
            threshold = float(input("Enter threshold value based on histogram: "))
            break
        except ValueError:
            print("Invalid input. Please enter a float.")

    labels = (energies > threshold).astype(np.float32)

    # (c) Split plots: Energy vs Time and ON/OFF vs Time
    plt.figure(figsize=(10, 6))
    # Energy vs Time
    plt.subplot(2, 1, 1)
    plt.plot(times, energies, label="Energy", lw=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Energy over Time")
    plt.grid(True)
    # ON/OFF vs Time
    plt.subplot(2, 1, 2)
    plt.plot(times, labels, label="ON/OFF", linestyle='--', color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Status (0 OFF / 1 ON)")
    plt.title("Detected ON/OFF Activity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return windows, labels, threshold

# ------------ 3) Prepare CNN data ------------
def prepare_cnn_data(windows, labels):
    # windows: (num_windows, 200) → (num_windows, 200, 1)
    X = windows[..., np.newaxis].astype(np.float32)
    y = labels.astype(np.float32)
    return X, y

# ------------ 4) Train CNN model ------------
def train_cnn_model(X, y, epochs=10):
    model = Sequential([
        Conv1D(32, 5, activation='relu', input_shape=(200, 1)),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # ON/OFF
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X, y, epochs=epochs, batch_size=64, validation_split=0.2)
    return model, history

# ------------ 5) Main pipeline ------------
def main():
    samples = capture_rtlsdr_samples()
    windows, labels, threshold = choose_threshold(samples)
    print(f"Using threshold = {threshold}")
    print(f"Windows: {windows.shape}, Labels: {labels.shape}")

    X, y = prepare_cnn_data(windows, labels)
    print(f"CNN input shape: {X.shape}, y shape: {y.shape}")

    print("Training 1D-CNN...")
    model, history = train_cnn_model(X, y, epochs=50)

    model.save("rtlsdr_cnn_onoff.h5")
    print("✅ Model saved as 'rtlsdr_cnn_onoff.h5'")

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("CNN Training Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

