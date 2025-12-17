import numpy as np
import adi
import scipy.signal as signal

sample_rate = 1e6  # Hz
center_freq = 915e6  # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate)  # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50  # Increase to increase tx power, valid range is -90 to 0 dB

N = 10000  # number of samples to transmit at once
t = np.arange(N) / sample_rate

# FM modulation parameters
fm_deviation = 75e3  # Frequency deviation in Hz
message_freq = 1e3  # Message frequency in Hz

# Create a message signal
message_signal = np.sin(2 * np.pi * message_freq * t)

# Perform FM modulation
integral_of_message = np.cumsum(message_signal) / sample_rate
fm_signal = np.exp(2.0j * np.pi * (center_freq * t + fm_deviation * integral_of_message))

# Scale the FM signal to the range expected by PlutoSDR
fm_signal *= 2**14  # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Transmit our batch of samples 10000 times, so it should be 10 seconds worth of samples total, if USB can keep up
for i in range(10000):
    sdr.tx(fm_signal)  # transmit the batch of samples once

print("FM transmission complete.")