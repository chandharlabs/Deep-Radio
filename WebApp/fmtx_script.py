import sys
from gnuradio import gr, analog, filter, blocks, osmosdr
from gnuradio.filter import firdes
import signal

class FMTXHackRF(gr.top_block):

    def __init__(self, samp_rate, audio_rate, center_freq, gain, bb_gain, if_gain, bandwidth, wavfile):
        gr.top_block.__init__(self, "FMTX HackRF")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = int(samp_rate)
        self.audio_rate = int(audio_rate)
        self.center_freq = float(center_freq)
        self.gain = int(gain)
        self.bb_gain = int(bb_gain)
        self.if_gain = int(if_gain)
        self.bandwidth = int(bandwidth)
        self.wavfile = wavfile

        ##################################################
        # Blocks
        ##################################################
        self.rational_resampler_xxx_0_0 = filter.rational_resampler_ccc(
                interpolation=500000,
                decimation=480000,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
                interpolation=480000,
                decimation=self.audio_rate,
                taps=[],
                fractional_bw=0)
        self.osmosdr_sink_0 = osmosdr.sink(
            args="numchan=" + str(1) + " " + ''
        )
        self.osmosdr_sink_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_sink_0.set_sample_rate(500000)
        self.osmosdr_sink_0.set_center_freq(self.center_freq, 0)
        self.osmosdr_sink_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0.set_gain(self.gain, 0)
        self.osmosdr_sink_0.set_if_gain(self.if_gain, 0)
        self.osmosdr_sink_0.set_bb_gain(self.bb_gain, 0)
        self.osmosdr_sink_0.set_antenna('', 0)
        self.osmosdr_sink_0.set_bandwidth(self.bandwidth, 0)
        self.low_pass_filter_0 = filter.interp_fir_filter_fff(
            1,
            firdes.low_pass(
                1,
                self.audio_rate,
                24000,
                1000,
                firdes.WIN_HAMMING,
                6.76))
        self.blocks_wavfile_source_0 = blocks.wavfile_source(self.wavfile, True)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc(480000/(2*3.14*17000))

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.rational_resampler_xxx_0_0, 0))
        self.connect((self.blocks_wavfile_source_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.osmosdr_sink_0, 0))

def main():
    samp_rate = sys.argv[1]
    audio_rate = sys.argv[2]
    center_freq = sys.argv[3]
    gain = sys.argv[4]
    bb_gain = sys.argv[5]
    if_gain = sys.argv[6]
    bandwidth = sys.argv[7]
    wavfile = sys.argv[8]

    tb = FMTXHackRF(samp_rate, audio_rate, center_freq, gain, bb_gain, if_gain, bandwidth, wavfile)
    tb.start()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("FM Transmission started. Press Ctrl+C to stop.")
    signal.pause()

if __name__ == '__main__':
    main()