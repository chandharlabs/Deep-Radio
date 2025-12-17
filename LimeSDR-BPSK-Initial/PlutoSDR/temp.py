#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# GNU Radio Python Flow Graph
# Title: PlutoSDR FM Transmit
# GNU Radio version: 3.8

import numpy as np
from gnuradio import gr, blocks, analog
import soapy
import scipy.io.wavfile as wavfile
import scipy.signal as signal

class pluto_fm_transmit(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "PlutoSDR FM Transmit")

        ##################################################
        # Variables
        ##################################################
        sample_rate = 1e6  # Hz
        center_freq = 915e6  # Hz
        fm_deviation = 75e3   # Frequency deviation in Hz

        ##################################################
        # Blocks
        ##################################################
        # SoapySDR Pluto Sink
        dev = 'driver=plutosdr,uri=ip:192.168.2.1'
        stream_args = 'fc32'
        tune_args = ['']
        settings = ['']
        self.soapy_plutosdr_sink_0 = soapy.sink(dev, "fc32", 1, "ip:192.168.2.1", stream_args, tune_args, settings)
        self.soapy_plutosdr_sink_0.set_sample_rate(0, sample_rate)
        self.soapy_plutosdr_sink_0.set_bandwidth(0, sample_rate)
        self.soapy_plutosdr_sink_0.set_frequency(0, center_freq)
        self.soapy_plutosdr_sink_0.set_gain(0, -10)  # Adjust as necessary

        # WAV file source
        sample_rate, audio_signal = wavfile.read('sample_audio.wav')

        # Normalize audio signal to -1 to 1
        if audio_signal.dtype != np.float32:
            audio_signal = audio_signal / np.max(np.abs(audio_signal))

        # Create time array for the original sample rate
        t = np.arange(len(audio_signal)) / sample_rate

        # Perform FM modulation
        integral_of_audio = np.cumsum(audio_signal) / sample_rate
        fm_signal = np.exp(2.0j * np.pi * (center_freq * t + fm_deviation * integral_of_audio))

        # Scale the FM signal to the range expected by PlutoSDR
        fm_signal *= 2**14  # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

        # Transmit the FM-modulated audio signal
        # Interpolate to match the SDR sample rate
        interp_factor = int(sample_rate / sample_rate)
        fm_signal_interpolated = signal.resample(fm_signal, len(fm_signal) * interp_factor)

        # Transmitting the signal using GNU Radio blocks
        self.blocks_vector_source_x_0 = blocks.vector_source_c(fm_signal_interpolated, True, 1, [])
        self.connect((self.blocks_vector_source_x_0, 0), (self.soapy_plutosdr_sink_0, 0))

if __name__ == '__main__':
    tb = pluto_fm_transmit()
    tb.start()
    tb.wait()