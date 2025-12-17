import adi
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# Load WAV file
sample_rate, audio_signal = wavfile.read('file_example_WAV_1MG.wav')

# Normalize audio signal to -1 to 1
audio_signal = audio_signal / np.max(np.abs(audio_signal))
audio_signal=audio_signal[:1000]
# FM Modulation parameters
carrier_freq = 100e6  # Carrier frequency in Hz
fm_deviation = 75e3   # Frequency deviation in Hz

# Check if the carrier frequency is within the valid range for PlutoSDR
if not (70e6 <= carrier_freq <= 6e9):
    raise ValueError("Carrier frequency must be between 70 MHz and 6 GHz")

# Create time array for the original sample rate
t = np.arange(len(audio_signal)) / sample_rate

# Perform FM modulation
integral_of_audio = np.cumsum(audio_signal) / sample_rate
fm_signal = np.cos(2 * np.pi * carrier_freq * t + 2 * np.pi * fm_deviation * integral_of_audio)

# PlutoSDR configuration
sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = int(2.5e6)
sdr.tx_rf_bandwidth = int(2e6)  # Ensure this is an integer
sdr.tx_lo = int(carrier_freq)
sdr.tx_hardwaregain_chan0 = -10  # Adjust as necessary

# Prepare the signal for transmission
# Interpolate to match the SDR sample rate
interp_factor = int(sdr.sample_rate / sample_rate)
fm_signal_interpolated = signal.resample(fm_signal, len(fm_signal) * interp_factor)

# Transmit the signal
sdr.tx(fm_signal_interpolated)

print("Transmission complete.")