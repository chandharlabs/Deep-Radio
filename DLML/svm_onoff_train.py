# svm_train.py
import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# ------------ 1) Capture with STOP LIVE + dual live plots ------------
def capture_rtlsdr_samples(sample_rate=240e3, center_freq=957e6, gain=40.2,
                           num_samples=2_000_000, window_size=1000, chunk_size=16384):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain

    print(f"Receiving {num_samples} samples in real time...")
    samples = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

    # Plot 1 — Magnitude (current chunk)
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
    ax2.set_ylim(0, 2e-4)  # adjust as needed
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

            # Live magnitude (cap to window_size for plotting)
            mags = np.abs(chunk)
            n = min(len(mags), window_size)
            if n > 0:
                line1.set_data(np.arange(n), mags[:n])
                ax1.set_xlim(0, n)

            # Live energy trace
            energy = float(np.mean(mags**2)) if len(mags) > 0 else 0.0
            energies_live.append(energy)
            line2.set_data(np.arange(len(energies_live)), energies_live)
            right = len(energies_live); left = max(0, right - 200)
            ax2.set_xlim(left, right)

            fig.canvas.draw(); fig.canvas.flush_events()

            received += len(chunk)
            # time.sleep(0.001)  # uncomment if you still see USB overflows

    finally:
        sdr.close()

    print(f"\n[INFO] Capture complete. Received {len(samples)} samples.")
    plt.ioff()
    plt.show(block=False)  # keep the window visible/frozen
    return np.array(samples)

# ------------ 2) Energy over time + Histogram + Threshold & Labels ------------
def label_by_threshold(samples, window_size=200, sample_rate=240e3):
    mags = np.abs(samples)
    energies = []
    for i in range(0, len(mags) - window_size, window_size):
        win = mags[i:i + window_size]
        energies.append(float(np.mean(win**2)))
    energies = np.array(energies, dtype=np.float64)

    # (a) Energy vs Time BEFORE threshold
    times = np.arange(len(energies)) * (window_size / float(sample_rate))
    plt.figure(figsize=(10, 4))
    plt.plot(times, energies, lw=1)
    plt.title("Energy Over Time (Before Threshold Selection)")
    plt.xlabel("Time (s)"); plt.ylabel("Energy"); plt.grid(True); plt.show()

    # (b) Histogram
    plt.figure()
    plt.hist(energies, bins=100, color='blue', alpha=0.7)
    plt.xlabel("Mean Energy per Window"); plt.ylabel("Count")
    plt.title("Energy Distribution - Choose Threshold")
    plt.grid(True); plt.show()

    # Ask for threshold
    while True:
        try:
            threshold = float(input("Enter threshold value based on histogram: "))
            break
        except ValueError:
            print("Invalid input. Please enter a float.")

    labels = (energies > threshold).astype(np.float32)

    # (c) Split plots: Energy vs Time and ON/OFF vs Time
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, energies, lw=1); plt.title("Energy over Time")
    plt.xlabel("Time (s)"); plt.ylabel("Energy"); plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(times, labels, linestyle='--', color='orange')
    plt.xlabel("Time (s)"); plt.ylabel("Status (0 OFF / 1 ON)")
    plt.title("Detected ON/OFF Activity"); plt.grid(True)
    plt.tight_layout(); plt.show()

    return energies, labels, threshold

# ------------ 3) Feature extraction (B-set) ------------
def extract_features(samples, window_size=200):
    """Return features per window:
       [mean_energy, std_dev, max_mag, variance]"""
    mags = np.abs(samples)
    feats = []
    for i in range(0, len(mags) - window_size, window_size):
        w = mags[i:i + window_size]
        mean_energy = float(np.mean(w**2))
        std_dev     = float(np.std(w))
        max_mag     = float(np.max(w))
        variance    = float(np.var(w))
        feats.append([mean_energy, std_dev, max_mag, variance])
    return np.array(feats, dtype=np.float32)

# ------------ 4) Train SVM ------------
def train_svm(X, y):
    # Pipeline: Standardize features → SVM classifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale"))
    ])
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xva, yva)
    print(f"[INFO] Validation accuracy: {acc:.4f}")
    return pipe

def main():
    samples = capture_rtlsdr_samples()
    energies, labels, threshold = label_by_threshold(samples)

    # Align features to labels (one per window)
    X = extract_features(samples)                 # shape: (Nwin, 4)
    y = labels[:len(X)]                           # safety: match lengths
    print(f"Features: {X.shape}, Labels: {y.shape}")

    model = train_svm(X, y)
    joblib.dump({"model": model,
                 "window_size": 200,
                 "sample_rate": 240e3}, "svm_onoff.pkl")
    print("✅ Saved SVM model to svm_onoff.pkl")

if __name__ == "__main__":
    main()
