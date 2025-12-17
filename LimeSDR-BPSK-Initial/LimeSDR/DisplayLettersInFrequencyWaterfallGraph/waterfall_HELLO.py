import SoapySDR
import numpy as np
import time

# ---- SDR Configuration ----
sample_rate = 2e6        # 2 MHz bandwidth
center_freq = 650e6      # Center frequency at 650 MHz
tx_gain = 50             # High transmission gain for visibility
symbol_duration = 0.1    # Duration of each letter burst (seconds)

# ---- Letter Frequencies ----
letter_frequencies = {
    "H": {
        "left_vertical_bar": 649.9e6,
        "right_vertical_bar": 650.1e6,
        "horizontal_bar": np.linspace(649.9e6, 650.1e6, num=10)
    },
    "E": {
        "left_vertical_bar": 649.9e6,
        "horizontal_bar_top": np.linspace(649.9e6, 650.1e6, num=10),
        "horizontal_bar_middle": np.linspace(649.9e6, 650.1e6, num=10),
        "horizontal_bar_bottom": np.linspace(649.9e6, 650.1e6, num=10)
    },
    "L": {
        "left_vertical_bar": 649.9e6,
        "horizontal_bar_bottom": np.linspace(649.9e6, 650.1e6, num=10)
    },
    "O": {
        "left_vertical_bar": 649.9e6,
        "right_vertical_bar": 650.1e6,
        "horizontal_bar_top": np.linspace(649.9e6, 650.1e6, num=10),
        "horizontal_bar_bottom": np.linspace(649.9e6, 650.1e6, num=10)
    }
}

# ---- Initialize LimeSDR ----
sdr = SoapySDR.Device({"driver": "lime"})
sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, sample_rate)
sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, tx_gain)

tx = sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32, [0])

# ---- Generate CW Signal ----
num_samples = 1024
cw_signal = np.ones(num_samples, dtype=np.complex64) * 0.7  # Strong CW signal

def generate_wideband_signal(start_freq, stop_freq, sample_rate, num_samples=1024):
    """
    Generate a multitone signal covering a frequency range from start_freq to stop_freq.
    This ensures all frequencies in the range are transmitted simultaneously.
    """
    num_tones = 100  # Increase this for a smoother band
    freqs = np.linspace(start_freq, stop_freq, num=num_tones)
    
    t = np.arange(num_samples) / sample_rate
    signal = np.sum([np.exp(2j * np.pi * f * t) for f in freqs], axis=0)
    
    # Normalize to fit LimeSDR signal range
    signal /= np.max(np.abs(signal))
    
    return signal.astype(np.complex64)

# Generate wideband signal for the horizontal bar
horizontal_signal = generate_wideband_signal(649.9e6, 650.1e6, sample_rate)


def transmit_letter(letter):
    """ Transmit a letter by following its frequency pattern. """
    print(f"Transmitting letter '{letter}'...")

    if letter == "H":
        for x in range(10):
            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, letter_frequencies["H"]["left_vertical_bar"])
            sdr.writeStream(tx, [cw_signal], len(cw_signal))
            time.sleep(symbol_duration)

            if x == 5:  # Transmit continuous horizontal bar as a wideband signal
                sdr.writeStream(tx, [horizontal_signal], len(horizontal_signal))
                time.sleep(symbol_duration)

            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, letter_frequencies["H"]["right_vertical_bar"])
            sdr.writeStream(tx, [cw_signal], len(cw_signal))
            time.sleep(symbol_duration)

    elif letter == "E":
        for x in range(12):
            if x == 0 or x == 11:  # Top and bottom horizontal bars
                for freq in letter_frequencies["E"]["horizontal_bar_top"]:
                    sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, freq)
                    sdr.writeStream(tx, [cw_signal], len(cw_signal))
                time.sleep(symbol_duration)

            if x == 6:  # Middle horizontal bar
                sdr.writeStream(tx, [horizontal_signal], len(horizontal_signal))
                time.sleep(symbol_duration)

            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, letter_frequencies["E"]["left_vertical_bar"])
            sdr.writeStream(tx, [cw_signal], len(cw_signal))
            time.sleep(symbol_duration)

    elif letter == "L":
        for x in range(10):
            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, letter_frequencies["L"]["left_vertical_bar"])
            sdr.writeStream(tx, [cw_signal], len(cw_signal))
            time.sleep(symbol_duration)

            if x == 0:  # Bottom horizontal bar
                sdr.writeStream(tx, [horizontal_signal], len(horizontal_signal))
                time.sleep(symbol_duration)

    elif letter == "O":
        for x in range(10):
            if x == 0:  # Top horizontal bar
                sdr.writeStream(tx, [horizontal_signal], len(horizontal_signal))
                time.sleep(symbol_duration)

            if x == 9:  # Bottom horizontal bar
                sdr.writeStream(tx, [horizontal_signal], len(horizontal_signal))
                time.sleep(symbol_duration)

            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, letter_frequencies["O"]["left_vertical_bar"])
            sdr.writeStream(tx, [cw_signal], len(cw_signal))
            time.sleep(symbol_duration)

            sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, letter_frequencies["O"]["right_vertical_bar"])
            sdr.writeStream(tx, [cw_signal], len(cw_signal))
            time.sleep(symbol_duration)

# ---- Transmit "HELLO" ----
for letter in "HELLO":
    transmit_letter(letter)
    time.sleep(1)  # Pause between letters for visibility

# ---- Stop Transmission ----
sdr.deactivateStream(tx)
sdr.closeStream(tx)
print("Transmission complete!")
