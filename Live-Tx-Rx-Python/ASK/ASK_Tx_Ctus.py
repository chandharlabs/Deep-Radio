import numpy as np
from SoapySDR import Device, SOAPY_SDR_TX

# === CONFIG ===
CENTER_FREQ = 700e6
SAMPLE_RATE = 1e6
TX_GAIN = 60
BITRATE = 1000
BITS = "10010000011000"
CHUNK = 1024

# === SDR INIT ===
sdr = Device(dict(driver="lime"))
sdr.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_TX, 0, CENTER_FREQ)
sdr.setGain(SOAPY_SDR_TX, 0, TX_GAIN)
sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND1")
tx_stream = sdr.setupStream(SOAPY_SDR_TX, "CF32", [0])
sdr.activateStream(tx_stream)

# === GENERATE OOK / ASK ===
samples_per_bit = int(SAMPLE_RATE // BITRATE)
bit_symbols = []
for bit in BITS:
    amp = 1.0 if bit == '1' else 0.0
    symbol = amp * np.ones(samples_per_bit, dtype=np.complex64)
    bit_symbols.append(symbol)
signal = np.concatenate(bit_symbols)

# === TRANSMIT LOOP ===
print(f"[INFO] Transmitting ASK/OOK at {CENTER_FREQ/1e6:.1f} MHz... Ctrl+C to stop")
try:
    i = 0
    while True:
        if i + CHUNK > len(signal):
            i = 0
        chunk = signal[i:i+CHUNK]
        sdr.writeStream(tx_stream, [chunk], len(chunk))
        i += CHUNK
except KeyboardInterrupt:
    print("\n[INFO] Transmission stopped.")
finally:
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)


