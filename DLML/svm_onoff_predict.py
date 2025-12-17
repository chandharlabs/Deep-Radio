import numpy as np
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time, csv, joblib
from collections import deque

# -------- Extract window-based features --------
def extract_window_features(window):
    w = np.abs(window)
    mean_energy = float(np.mean(w**2))
    std_dev     = float(np.std(w))
    max_mag     = float(np.max(w))
    variance    = float(np.var(w))
    return np.array([mean_energy, std_dev, max_mag, variance], dtype=np.float32)

# -------- Live SDR Sample Generator --------
def live_rtlsdr_samples(sample_rate=240e3, center_freq=957e6, gain=40.2,
                        chunk_size=4096, sleep_sec=0.02):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain
    try:
        while True:
            try:
                yield sdr.read_samples(chunk_size)
            except Exception as e:
                print("SDR read warning:", e)
                time.sleep(0.05)
                continue
            time.sleep(sleep_sec)
    finally:
        sdr.close()

# -------- Live Prediction Pipeline --------
def live_predict_svm(model_path="svm_onoff.pkl",
                     output_csv="svm_live_predictions.csv",
                     sample_rate=240e3, center_freq=957e6, gain=40.2,
                     window_size=200, chunk_size=4096, x_window_seconds=30.0):

    # Load SVM model + metadata
    pack = joblib.load(model_path)
    model = pack["model"]
    if "window_size" in pack:
        window_size = int(pack["window_size"])
    threshold = pack.get("threshold", None)
    if threshold is None:
        threshold = float(input("Enter energy threshold used during training: "))
    print(f"[INFO] Using energy threshold = {threshold:.6e}")

    # Buffers & data history
    mag_buffer = deque()
    energy_times, energies = deque(maxlen=50000), deque(maxlen=50000)
    edet_times, edet_vals  = deque(maxlen=50000), deque(maxlen=50000)
    svm_times, svm_vals    = deque(maxlen=50000), deque(maxlen=50000)

    # CSV Logging
    f = open(output_csv, "w", newline="")
    wcsv = csv.writer(f)
    wcsv.writerow(["Time(s)", "Energy", "EnergyDetector", "SVM_Status", "SVM_Prob"])
    f.flush()

    # ---------- Live Plot Setup ----------
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    fig.suptitle("LIVE RF Activity — Magnitude | Energy | Energy Detector | SVM")

    # Plot 1 — Magnitude (current chunk)
    line_mag, = axs[0].plot([], [], lw=1)
    axs[0].set_title("Live Magnitude (current chunk)")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_ylim(0, 0.1)
    axs[0].set_xlim(0, window_size)
    axs[0].grid(True)

    # Plot 2 — Energy vs Time
    line_eng, = axs[1].plot([], [], lw=1, color='green')
    axs[1].set_title("Energy per Window")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Energy")
    axs[1].grid(True)

    # Plot 3 — Energy Detector Output
    line_edet, = axs[2].plot([], [], lw=2, drawstyle="steps-post", color='blue')
    axs[2].set_title("Energy Detector ON/OFF (Threshold-based)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Status")
    axs[2].set_ylim(-0.2, 1.2)
    axs[2].grid(True)

    # Plot 4 — SVM Output
    line_svm, = axs[3].plot([], [], lw=2, drawstyle="steps-post", color='orange')
    axs[3].set_title("SVM ON/OFF Prediction")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Status")
    axs[3].set_ylim(-0.2, 1.2)
    axs[3].grid(True)

    # STOP button (top-right)
    stop_ax = plt.axes([0.82, 0.94, 0.16, 0.05])
    stop_button = Button(stop_ax, "STOP LIVE", color="#cc0000", hovercolor="#ff3333")
    stop_flag = {"stop": False}
    def on_stop(_):
        stop_flag["stop"] = True
        stop_button.ax.set_facecolor("#990000")
        fig.canvas.draw_idle()
        print("[INFO] STOP LIVE pressed.")
    stop_button.on_clicked(on_stop)

    window_dt = window_size / float(sample_rate)
    total_win = 0

    try:
        for chunk in live_rtlsdr_samples(sample_rate, center_freq, gain, chunk_size, 0.02):
            if stop_flag["stop"]:
                break

            mags = np.abs(chunk)
            mag_buffer.extend(mags)

            # Plot magnitude
            n = min(len(mags), window_size)
            if n > 0:
                line_mag.set_data(np.arange(n), mags[:n])
                axs[0].set_xlim(0, n)

            # Process full windows
            while len(mag_buffer) >= window_size:
                if stop_flag["stop"]:
                    break

                win = np.array([mag_buffer.popleft() for _ in range(window_size)], dtype=np.float32)
                t_now = total_win * window_dt
                energy = float(np.mean(win**2))

                # --- Energy Detector output ---
                edet_status = 1 if energy > threshold else 0

                # --- SVM Prediction ---
                X = extract_window_features(win).reshape(1, -1)
                prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.decision_function(X))
                svm_status = 1 if prob > 0.5 else 0

                # Update time series
                energy_times.append(t_now); energies.append(energy)
                edet_times.append(t_now); edet_vals.append(edet_status)
                svm_times.append(t_now); svm_vals.append(svm_status)

                # Write CSV
                wcsv.writerow([f"{t_now:.6f}", f"{energy:.8e}", edet_status, svm_status, f"{prob:.6f}"])
                f.flush()

                total_win += 1

            # --- Update plots dynamically ---
            if energy_times:
                xmax = energy_times[-1]
                xmin = max(0, xmax - x_window_seconds)

                # Energy
                t_e, v_e = np.array(energy_times), np.array(energies)
                m_e = (t_e >= xmin) & (t_e <= xmax)
                if m_e.any():
                    line_eng.set_data(t_e[m_e], v_e[m_e])
                    axs[1].set_xlim(xmin, xmax)
                    ymin, ymax = np.percentile(v_e[m_e], [5, 95])
                    axs[1].set_ylim(max(0, ymin), ymax * 1.1)

                # Energy Detector
                t_d, v_d = np.array(edet_times), np.array(edet_vals)
                m_d = (t_d >= xmin) & (t_d <= xmax)
                if m_d.any():
                    line_edet.set_data(t_d[m_d], v_d[m_d])
                    axs[2].set_xlim(xmin, xmax)

                # SVM
                t_s, v_s = np.array(svm_times), np.array(svm_vals)
                m_s = (t_s >= xmin) & (t_s <= xmax)
                if m_s.any():
                    line_svm.set_data(t_s[m_s], v_s[m_s])
                    axs[3].set_xlim(xmin, xmax)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

    finally:
        f.close()
        plt.ioff()
        print(f"[INFO] Stopped. CSV saved to {output_csv}")
        plt.show()

# -------- Run --------
if __name__ == "__main__":
    live_predict_svm(
        model_path="svm_onoff.pkl",
        output_csv="svm_live_predictions.csv",
        sample_rate=240e3,
        center_freq=957e6,
        gain=40.2,
        window_size=200,
        chunk_size=4096,
        x_window_seconds=30.0
    )
