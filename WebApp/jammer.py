import SoapySDR
import numpy as np
import time
import json
from scipy.signal import chirp
import sys

# Parse arguments
sample_rate = float(sys.argv[1])
tx_gain = float(sys.argv[2])
num_samples = int(sys.argv[3])
hop_interval = float(sys.argv[4])
freq_list = json.loads(sys.argv[5])

# Create SDR object
sdr = SoapySDR.Device({"driver": "lime"})

# Configure TX stream
sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, sample_rate)
sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, tx_gain)

# Create TX stream
tx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32, [0])
sdr.activateStream(tx_stream)

# Generate a 30 MHz wide chirp signal
t = np.linspace(0, 1, num_samples, endpoint=False)
tx_signal = chirp(t, f0=-15e6, f1=15e6, t1=1, method='linear')
tx_signal = np.exp(2j * np.pi * tx_signal)  # Convert to complex IQ samples
tx_signal = (tx_signal * 0.5).astype(np.complex64)  # Normalize amplitude

# Frequency hopping loop
try:
    while True:
        for freq in freq_list:
            print(f"Setting frequency to {freq} Hz")
            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, float(freq))
            
            sr = sdr.writeStream(tx_stream, [tx_signal], num_samples)
            if sr.ret != num_samples:
                print(f"Warning: {sr.ret} samples transmitted instead of {num_samples}")
            
            time.sleep(hop_interval)  # Wait before hopping to the next frequency
except KeyboardInterrupt:
    print("Transmission stopped.")

# Cleanup
sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
print("TX Stream closed.")