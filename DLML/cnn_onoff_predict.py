# live_cnn_onoff.py
import numpy as np
from keras.models import load_model
from rtlsdr import RtlSdr
import time, csv, warnings
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------- SDR chunked generator ----------
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
                print("SDR read warning:", e); time.sleep(0.05); continue
            time.sleep(sleep_sec)
    finally:
        sdr.close()

def live_predict_activity(model_path="rtlsdr_cnn_onoff.h5",
                          output_csv="live_cnn_predictions.csv",
                          sample_rate=240e3, center_freq=957e6, gain=40.2,
                          window_size=200, chunk_size=4096,
                          x_window_seconds=30.0,
                          threshold_percentile=80):
    # Load CNN
    model = load_model(model_path)

    # Buffers & histories
    mag_buffer = deque()
    energy_hist = deque(maxlen=50000)   # for dynamic threshold
    bin_hist    = deque(maxlen=50000)   # binary for ARIMA

    energy_times = deque(maxlen=50000); energies  = deque(maxlen=50000)
    det_times    = deque(maxlen=50000); det_vals  = deque(maxlen=50000)
    cnn_times    = deque(maxlen=50000); cnn_vals  = deque(maxlen=50000)


    # CSV
    csv_file = open(output_csv, "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Time(s)", "Energy", "Threshold", "EnergyDetector",
                     "CNN_Status", "CNN_Confidence"])
    csv_file.flush()

    # Plots
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    ax1, ax2, ax3 = axes
    fig.suptitle("LIVE RF Activity • Energy | Energy Detector | CNN")

    # Plot 1: Energy
    line_energy, = ax1.plot([], [], lw=1, label="Energy")
    thresh_line  = ax1.axhline(0.0, color="#aa0000", ls="--", lw=1, label="Threshold")
    ax1.set_ylabel("Energy"); ax1.grid(True); ax1.legend(loc="upper left")

    # Plot 2: Energy detector (binary)
    line_det, = ax2.plot([], [], lw=2, drawstyle="steps-post", color="#ff8c00", label="Energy Detector")
    ax2.set_ylabel("Detector"); ax2.set_ylim(-0.2,1.2); ax2.grid(True); ax2.legend(loc="upper left")

    # Plot 3: CNN (binary)
    line_cnn, = ax3.plot([], [], lw=2, drawstyle="steps-post", color="#1a7f37", label="CNN (200-sample window)")
    ax3.set_ylabel("CNN"); ax3.set_ylim(-0.2,1.2); ax3.grid(True); ax3.legend(loc="upper left")

    # STOP LIVE button
    stop_ax = plt.axes([0.82, 0.94, 0.16, 0.05])
    from matplotlib.widgets import Button
    stop_button = Button(stop_ax, "STOP LIVE", color="#cc0000", hovercolor="#ff3333")
    stop_flag = {"stop": False}
    def on_stop(_): stop_flag["stop"] = True; stop_button.ax.set_facecolor("#990000"); fig.canvas.draw_idle()
    stop_button.on_clicked(on_stop)

    window_dt = window_size / float(sample_rate)
    total_win = 0
    

    def dyn_threshold():
        if len(energy_hist) < 50:
            return (np.median(energy_hist) if energy_hist else 0.0)
        return float(np.percentile(np.array(energy_hist), threshold_percentile))

    def view_slice(tdeque, vdeque, xmin, xmax):
        if not tdeque: return np.array([]), np.array([])
        t_arr, v_arr = np.array(tdeque), np.array(vdeque)
        m = (t_arr >= xmin) & (t_arr <= xmax)
        if not m.any(): return np.array([]), np.array([])
        return t_arr[m], v_arr[m]

    try:
        for chunk in live_rtlsdr_samples(sample_rate, center_freq, gain, chunk_size, 0.02):
            if stop_flag["stop"]: break
            mag_buffer.extend(np.abs(chunk))

            while len(mag_buffer) >= window_size:
                if stop_flag["stop"]: break

                # Build one window (200 samples)
                window = np.array([mag_buffer.popleft() for _ in range(window_size)], dtype=np.float32)
                energy = float(np.mean(window**2))
                t_now  = total_win * window_dt

                # Energy + threshold
                energy_hist.append(energy); energies.append(energy); energy_times.append(t_now)
                thr = dyn_threshold(); thresh_line.set_ydata([thr, thr])

                # Energy detector (binary)
                det = 1 if energy > thr else 0
                det_vals.append(det); det_times.append(t_now)

                # CNN prediction from this window (standardize exactly like training)
                mu, sd = window.mean(), window.std() + 1e-6
                x_win = ((window - mu) / sd).reshape(1, window_size, 1)  # (1,200,1)
                p = float(model.predict(x_win, verbose=0)[0][0])
                cnn_status = 1 if p > 0.5 else 0
                cnn_times.append(t_now); cnn_vals.append(cnn_status)


                # CSV
                writer.writerow([
                    f"{t_now:.6f}",
                    f"{energy:.8f}",
                    f"{thr:.8f}",
                    ("1" if det==1 else "0"),
                    ("1" if cnn_status==1 else "0"),
                    f"{p:.6f}"
                ])
                csv_file.flush()

                total_win += 1

            # Update plots
            if energy_times:
                xmax = energy_times[-1]; xmin = max(0.0, xmax - x_window_seconds)

                t1,e1 = view_slice(energy_times, energies, xmin, xmax)
                line_energy.set_data(t1, e1); ax1.set_xlim(xmin, xmax)
                if e1.size>0:
                    ymin, ymax = float(np.percentile(e1,5)), float(np.percentile(e1,95))
                    if ymax<=ymin: ymax=ymin+1e-6
                    ax1.set_ylim(ymin, ymax)

                td,yd = view_slice(det_times, det_vals, xmin, xmax)
                line_det.set_data(td, yd); ax2.set_xlim(xmin, xmax)

                tc,yc = view_slice(cnn_times, cnn_vals, xmin, xmax)
                line_cnn.set_data(tc, yc); ax3.set_xlim(xmin, xmax)

                fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)
    finally:
        csv_file.close()
        plt.ioff()
        print(f"[INFO] Stopped. CSV saved to {output_csv}")
        plt.show()

if __name__ == "__main__":
    live_predict_activity(
        model_path="rtlsdr_cnn_onoff.h5",      # <-- CNN model you trained with script #1
        output_csv="live_cnn_predictions.csv",
        sample_rate=240e3,
        center_freq=957e6,
        gain=40.2,
        window_size=200,
        chunk_size=4096,
        x_window_seconds=30.0,
        threshold_percentile=80
    )

