import numpy as np
import adi
import sounddevice as sd
import scipy.signal as signal

# FM Modulation parameters
carrier_freq = 915e6  # Carrier frequency in Hz
fm_deviation = 75e3   # Frequency deviation in Hz
audio_sample_rate = 44100  # Audio sample rate
sdr_sample_rate = int(2.5e6)  # SDR sample rate

# PlutoSDR configuration
sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = sdr_sample_rate
sdr.tx_rf_bandwidth = sdr_sample_rate  # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(carrier_freq)
sdr.tx_hardwaregain_chan0 = -10  # Adjust as necessary

def fm_modulate(audio_signal, sample_rate, fm_deviation):
    # Normalize audio signal to -1 to 1
    if audio_signal.dtype != np.float32:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))

    # Perform FM modulation
    integral_of_audio = np.cumsum(audio_signal) / sample_rate
    fm_signal = np.exp(2.0j * np.pi * (fm_deviation * integral_of_audio))

    return fm_signal

def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # Perform FM modulation on the captured audio signal
    fm_signal = fm_modulate(indata[:, 0], audio_sample_rate, fm_deviation)

    # Interpolate to match the SDR sample rate
    interp_factor = int(sdr_sample_rate / audio_sample_rate)
    fm_signal_interpolated = signal.resample(fm_signal, len(fm_signal) * interp_factor)

    # Scale the FM signal to the range expected by PlutoSDR
    fm_signal_interpolated *= 2**14

    # Transmit the FM-modulated audio signal
    sdr.tx(fm_signal_interpolated.astype(np.complex64))

# Capture audio from the microphone and transmit using PlutoSDR
with sd.InputStream(samplerate=audio_sample_rate, channels=2, callback=audio_callback):
    print("Transmitting voice. Press Ctrl+C to stop.")
    while True:
        sd.sleep(1000)

print("FM transmission of voice complete.")