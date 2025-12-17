import numpy as np
import adi
import scipy.io.wavfile as wavfile
import scipy.signal as signal

# Load WAV file
sample_rate, audio_signal = wavfile.read('testMusic.wav')
print("Sample rate:", sample_rate)

# Normalize audio signal to -1 to 1
if audio_signal.dtype != np.float32:
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

# FM Modulation parameters
carrier_freq = 915e6  # Carrier frequency in Hz
fm_deviation = 75e3   # Frequency deviation in Hz

# Create time array for the original sample rate
t = np.arange(len(audio_signal)) / sample_rate

# Perform FM modulation
integral_of_audio = np.cumsum(audio_signal) / sample_rate

# PlutoSDR configuration
sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = int(2.5e6)
sdr.tx_rf_bandwidth = int(2.5e6)  # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(carrier_freq)
sdr.tx_hardwaregain_chan0 = -30  # Adjust as necessary

# Interpolate to match the SDR sample rate
interp_factor = int(sdr.sample_rate / sample_rate)
t_interpolated = np.arange(len(audio_signal) * interp_factor) / sdr.sample_rate
integral_of_audio_interpolated = signal.resample(integral_of_audio, len(t_interpolated))

# Perform FM modulation with interpolated signal
fm_signal = np.exp(2.0j * np.pi * (fm_deviation * integral_of_audio_interpolated))

# Scale the FM signal to the range expected by PlutoSDR
fm_signal *= 2**14  # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Transmit our batch of samples 100 times, so it should be enough for continuous transmission
for i in range(100):
    sdr.tx(fm_signal.astype(np.complex64))  # transmit the batch of samples

print("FM transmission of audio complete.")