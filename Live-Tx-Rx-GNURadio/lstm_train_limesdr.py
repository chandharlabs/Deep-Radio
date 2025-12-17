import numpy as np
import SoapySDR
from SoapySDR import *
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time
import matplotlib.pyplot as plt

# ------------ 1. Capture Samples from LimeSDR ------------
def capture_limesdr_samples(sample_rate=1e6, center_freq=881e6, gain=60, num_samples=1000000):
    args = dict(driver="lime")
    sdr = SoapySDR.Device(args)

    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
    sdr.setGain(SOAPY_SDR_RX, 0, gain)
    sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAW")

    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)

    buff = np.empty(num_samples, dtype=np.complex64)
    num_recv = 0
    print("Receiving samples...")
    while num_recv < num_samples:
        sr = sdr.readStream(rxStream, [buff[num_recv:]], num_samples - num_recv)
        if sr.ret > 0:
            num_recv += sr.ret
        else:
            print("Stream read error:", sr.ret)

    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

    print("Capture complete.")
    return buff

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
    samples = capture_limesdr_samples()
    print(f"Captured {len(samples)} samples.")

    windows, labels, threshold = choose_threshold(samples)
    print(f"Using threshold = {threshold}")
    print(f"Windows: {windows.shape}, Labels: {labels.shape}")

    X, y = prepare_lstm_data(windows, labels)
    print(f"LSTM input shape: {X.shape}, y shape: {y.shape}")

    print("Training LSTM...")
    model, history = train_lstm_model(X, y, epochs=10)

    model.save("limesdr_lstm_model.h5")
    print("Model saved as 'limesdr_lstm_model.h5'")

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("LSTM Training Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

