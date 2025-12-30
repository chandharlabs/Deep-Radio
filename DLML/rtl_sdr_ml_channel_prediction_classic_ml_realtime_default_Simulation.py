#!/usr/bin/env python3
"""
python3 rtl_sdr_ml_channel_prediction_classic_ml_realtime.py --fc 956.4e6 --samp-rate 2.4e6 --gain 40.2  --frame 16384 --hop 8192 --horizon-ms 3 --win 16 --retrain-interval 1.0


Realtime ML-based channel power prediction using RTL-SDR (Classic ML)

Mode: A (Classic ML) + Data source: 2 (Realtime capture)

What it does
------------
- Captures IQ from RTL-SDR in short frames
- Extracts classic DSP features per frame (RSSI, variance, kurtosis, spectral centroid, bandwidth, etc.)
- Builds supervised samples with sliding windows to predict **future channel power** (RSSI) at a given horizon
- Trains lightweight ML models (LinearRegression + RandomForestRegressor)
- Runs a live loop: capture → feature → predict → (periodic) retrain
- Shows a live plot of actual vs predicted power and prints rolling metrics

If no RTL-SDR is available, add `--simulate` to use a fading simulator (Rayleigh/Rician with Doppler).

Dependencies
------------
python -m pip install numpy scipy scikit-learn matplotlib pyrtlsdr

Example
-------
python ml_channel_pred_rtlsdr.py \
  --fc 98.3e6 --samp-rate 2.4e6 --gain 20 \
  --frame 4096 --hop 2048 --horizon-ms 10 --win 8 --retrain-interval 3.0

"""
import argparse
import time
import sys
import threading
import queue
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import welch, get_window
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

try:
    from rtlsdr import RtlSdr
except Exception:
    RtlSdr = None

# ------------------------------
# Feature extraction
# ------------------------------

def feature_vector(iq: np.ndarray, fs: float):
    """Compute classic DSP features for a frame.
    Returns vector and a scalar target proxy (RSSI in dBFS).
    """
    if iq.size == 0:
        return None, None

    # Center/normalize
    x = iq.astype(np.complex64)
    # Power and RSSI proxy
    p = np.real(x*np.conj(x))
    p_mean = float(np.mean(p) + 1e-12)
    rssi = 10*np.log10(p_mean)

    # Higher order stats (real & imag)
    xr, xi = np.real(x), np.imag(x)
    v_r = float(np.var(xr))
    v_i = float(np.var(xi))
    k_r = float(kurtosis(xr, fisher=True, bias=False))
    k_i = float(kurtosis(xi, fisher=True, bias=False))
    s_r = float(skew(xr, bias=False))
    s_i = float(skew(xi, bias=False))

    # Spectrum features
    # Use Welch PSD for robustness
    f, psd = welch(x, fs=fs, window='hann', nperseg=min(1024, len(x)), noverlap=0, return_onesided=True, detrend=False, scaling='density')
    psd = np.abs(psd).astype(np.float64) + 1e-18
    psd_sum = float(np.sum(psd))
    # Spectral centroid & spread
    centroid = float(np.sum(f * psd) / psd_sum)
    spread = float(np.sqrt(np.sum(((f - centroid)**2) * psd) / psd_sum))
    # Spectral flatness (geometric / arithmetic mean)
    gmean = float(np.exp(np.mean(np.log(psd))))
    amean = float(np.mean(psd))
    flatness = float(gmean / (amean + 1e-18))
    # Occupied bandwidth (x% power)
    cdf = np.cumsum(psd)
    cdf = cdf / cdf[-1]
    try:
        low_idx = int(np.searchsorted(cdf, 0.05))
        high_idx = int(np.searchsorted(cdf, 0.95))
        obw = float(f[max(high_idx,1)-1] - f[max(low_idx,1)-1])
    except Exception:
        obw = 0.0

    # Simple Doppler proxy via lag-1 autocorr phase rate (if strong tone/motion)
    x1 = x[:-1]
    x2 = x[1:]
    if len(x1) > 0:
        r1 = np.vdot(x1, x2) / (np.linalg.norm(x1)*np.linalg.norm(x2) + 1e-12)
        doppler_hz = float(np.angle(r1) * fs / (2*np.pi))
        coh = float(np.abs(r1))
    else:
        doppler_hz, coh = 0.0, 0.0

    feats = np.array([
        rssi, v_r, v_i, k_r, k_i, s_r, s_i,
        centroid, spread, flatness, obw, doppler_hz, coh
    ], dtype=np.float32)

    return feats, rssi

# ------------------------------
# Sliding window dataset builder
# ------------------------------

def build_samples(feat_buf, target_buf, win, horizon):
    """Create X (win stacked frames) and y (future target at +horizon frames)."""
    X, y = [], []
    for t in range(len(feat_buf) - win - horizon):
        x = np.hstack(feat_buf[t:t+win])
        X.append(x)
        y.append(target_buf[t+win+horizon-1])
    if not X:
        return None, None
    return np.vstack(X), np.array(y, dtype=np.float32)

# ------------------------------
# Simulator for fallback / testing
# ------------------------------

def simulate_iq(n, fs, fd=30.0, K=0.0, snr_db=20.0, f0=0.0):
    """Rayleigh/Rician fading simulator with optional tone and AWGN.
    fd: max Doppler Hz, K: Rician K-factor (0=Rayleigh), f0: residual carrier."""
    t = np.arange(n)/fs
    # Clarke/Jakes-like fading (very rough)
    n_sin = 16
    w = 2*np.pi*fd*np.random.uniform(0.8, 1.2, size=n_sin)
    ph = 2*np.pi*np.random.rand(n_sin)
    ray = np.sum(np.exp(1j*(np.outer(t, w) + ph)), axis=1)/np.sqrt(n_sin)
    if K > 0:
        los = np.sqrt(K/(K+1)) * np.exp(1j*(2*np.pi*f0*t))
        ray = los + ray/np.sqrt(K+1)
    # AWGN
    sig_pow = np.mean(np.abs(ray)**2)
    snr_lin = 10**(snr_db/10)
    noise_pow = sig_pow / snr_lin
    noise = (np.sqrt(noise_pow/2)*(np.random.randn(n)+1j*np.random.randn(n))).astype(np.complex64)
    x = (ray + noise).astype(np.complex64)
    return x

# ------------------------------
# Main loop
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description='Realtime ML channel power prediction with RTL-SDR')
    ap.add_argument('--fc', type=float, default=98.3e6, help='Center frequency [Hz]')
    ap.add_argument('--samp-rate', type=float, default=2.4e6, help='Sample rate [S/s]')
    ap.add_argument('--gain', type=float, default=20, help='Tuner gain [dB] (use -1 for auto)')
    ap.add_argument('--frame', type=int, default=4096, help='Samples per frame')
    ap.add_argument('--hop', type=int, default=2048, help='Frame hop (overlap)')
    ap.add_argument('--win', type=int, default=8, help='Number of past frames in X')
    ap.add_argument('--horizon-ms', type=float, default=10.0, help='Prediction horizon in milliseconds')
    ap.add_argument('--retrain-interval', type=float, default=3.0, help='Seconds between retrains')
    ap.add_argument('--max-train-samples', type=int, default=4000, help='Cap training rows (older dropped)')
    ap.add_argument('--no-plot', action='store_true', help='Disable live plot')
    ap.add_argument('--simulate', action='store_true', help='Use simulator (no hardware)')
    args = ap.parse_args()

    fs = args.samp_rate
    frame = args.frame
    hop = args.hop
    assert hop <= frame, 'hop must be <= frame'

    horizon_frames = max(1, int(round(args.horizon_ms*1e-3 * fs / hop)))

    # Buffers
    feat_buf = []
    tgt_buf = []

    # Models
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    lr = LinearRegression()
    have_models = False

    last_retrain = time.time()

    # Live plot
    if not args.no_plot:
        import matplotlib.widgets as widgets
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Channel Power (RSSI) Actual vs Predicted (dB)')
        ax.set_xlabel('Frame index')
        ax.set_ylabel('dBFS (relative)')
        h_actual, = ax.plot([], [], label='Actual')
        h_pred_rf, = ax.plot([], [], label='RF Pred')
        #h_pred_lr, = ax.plot([], [], label='LinReg Pred')
        # Add STOP button
        stop_ax = fig.add_axes([0.8, 0.01, 0.15, 0.05])
        stop_button = widgets.Button(stop_ax, 'STOP')
        stop_flag = {'stop': False}
        def _stop(event): stop_flag['stop'] = True
        stop_button.on_clicked(_stop)
        ax.legend()

    # Source setup
    sdr = None
    if not args.simulate and RtlSdr is not None:
        try:
            sdr = RtlSdr()
            sdr.sample_rate = fs
            sdr.center_freq = args.fc
            if args.gain >= 0:
                sdr.gain = args.gain
            else:
                sdr.gain = 'auto'
        except Exception as e:
            print(f"[WARN] RTL-SDR init failed: {e}. Falling back to simulator.")
            sdr = None
            args.simulate = True
    else:
        args.simulate = True

    # Capture helper
    def get_frame():
        if args.simulate:
            # Simulate with time-varying Doppler and SNR
            fd = 30 + 20*np.sin(time.time()/5.0)
            snr = 15 + 5*np.cos(time.time()/7.0)
            x = simulate_iq(frame, fs, fd=fd, K=0.5, snr_db=snr, f0=200)
        else:
            # Read with overlap using hop
            # For efficiency, read 'hop' new samples and keep last (frame-hop)
            if not hasattr(get_frame, 'ring'):
                get_frame.ring = np.zeros(frame, dtype=np.complex64)
            new = np.array(sdr.read_samples(hop), dtype=np.complex64)
            get_frame.ring = np.roll(get_frame.ring, -hop)
            get_frame.ring[-hop:] = new
            x = get_frame.ring.copy()
        return x

    # Main streaming loop
    actual_hist = []
    pred_rf_hist = []
    pred_lr_hist = []

    try:
        # Press 'q' in terminal to stop gracefully
        import select, sys
        while True:
            if not args.no_plot and stop_flag.get('stop', False):
                print("STOP button pressed. Stopped updates — figure will remain open.")
                # stop updates but keep figure open
                while plt.fignum_exists(fig.number):
                    plt.pause(0.1)
                break
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == 'q':
                    print("Stop command received."); break
            x = get_frame()
            feats, rssi = feature_vector(x, fs)
            if feats is None:
                continue
            feat_buf.append(feats)
            tgt_buf.append(rssi)

            # Keep buffers reasonable
            if len(feat_buf) > 12000:
                feat_buf = feat_buf[-8000:]
                tgt_buf = tgt_buf[-8000:]

            # Build dataset
            X, y = build_samples(feat_buf, tgt_buf, args.win, horizon_frames)
            can_train = X is not None and len(y) >= max(50, args.win*5)

            # Retrain periodically
            now = time.time()
            if can_train and (not have_models or (now - last_retrain) >= args.retrain_interval):
                last_retrain = now
                # cap training size (use latest rows)
                if len(y) > args.max_train_samples:
                    X = X[-args.max_train_samples:]
                    y = y[-args.max_train_samples:]
                try:
                    rf.fit(X, y)
                    lr.fit(X, y)
                    have_models = True
                except Exception as e:
                    print(f"[WARN] training failed: {e}")

            # Predict next if models exist
            if have_models and len(feat_buf) >= args.win:
                x_last = np.hstack(feat_buf[-args.win:]).reshape(1, -1)
                prf = float(rf.predict(x_last)[0])
                plr = float(lr.predict(x_last)[0])
            else:
                prf = plr = rssi

            actual_hist.append(rssi)
            pred_rf_hist.append(prf)
            pred_lr_hist.append(plr)

            # Rolling MAE on recent window
            if len(actual_hist) > 200:
                a = np.array(actual_hist[-200:])
                pr = np.array(pred_rf_hist[-200:])
                pl = np.array(pred_lr_hist[-200:])
                mae_rf = mean_absolute_error(a, pr)
                mae_lr = mean_absolute_error(a, pl)
            else:
                mae_rf = mae_lr = np.nan

            # Console status
            sys.stdout.write(f"\rFrames: {len(tgt_buf):6d}  RSSI: {rssi:6.2f} dB  Pred(RF): {prf:6.2f}  Pred(LR): {plr:6.2f}  MAE_RF(200): {mae_rf:5.2f}")
            sys.stdout.flush()

            # Update plot
            if not args.no_plot and (len(actual_hist) % 2 == 0):
                idx0 = max(0, len(actual_hist)-600)
                h_actual.set_data(np.arange(idx0, len(actual_hist)), actual_hist[idx0:])
                h_pred_rf.set_data(np.arange(idx0, len(pred_rf_hist)), pred_rf_hist[idx0:])
                #h_pred_lr.set_data(np.arange(idx0, len(pred_lr_hist)), pred_lr_hist[idx0:])
                ax.relim(); ax.autoscale_view()
                fig.canvas.draw(); fig.canvas.flush_events()

    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        if sdr is not None:
            try:
                sdr.close()
            except Exception:
                pass

if __name__ == '__main__':
    main()

