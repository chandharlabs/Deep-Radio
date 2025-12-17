import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from rtlsdr import RtlSdr

def fm_demodulate(samples, sample_rate, freq, bandwidth):
    # Decimate the signal
    decimation_factor = int(sample_rate / bandwidth)
    samples = signal.decimate(samples, decimation_factor)
    sample_rate = sample_rate // decimation_factor
    
    # Perform FM demodulation
    angle = np.angle(samples[1:] * np.conj(samples[:-1]))
    audio_signal = signal.decimate(angle, 10)
    
    return audio_signal, sample_rate

def main():
    # Configure RTL-SDR
    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6  # Sample rate in Hz
    sdr.center_freq = 90.4e6  # Center frequency in Hz (e.g., 100 MHz for FM)
    sdr.gain = 25 #'auto'
    
    # Capture samples
    print("Capturing samples...")
    samples = sdr.read_samples(256*1024)
    
    # Perform FM demodulation
    print("Demodulating signal...")
    audio_signal, audio_rate = fm_demodulate(samples, sdr.sample_rate, sdr.center_freq, 200e3)
    
    # Normalize audio signal to 16-bit PCM format
    audio_signal = np.int16((audio_signal / np.max(np.abs(audio_signal))) * 32767)
    
    # Ensure the sample rate is an integer
    audio_rate = int(audio_rate)
    
    # Save audio to file
    print("Saving audio to file...")
    wavfile.write('fm_audio.wav', audio_rate, audio_signal)
    
    # Close RTL-SDR
    sdr.close()
    
if __name__ == "__main__":
    main()