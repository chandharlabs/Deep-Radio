#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: BPSK Decoder
# GNU Radio version: v3.9.2.0-85-g08bb05c1

from distutils.version import StrictVersion
import sys

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import BPSK_message_decoder_epy_block_1 as epy_block_1  # embedded python block
import osmosdr
import time

class BPSK_message_decoder(gr.top_block):

    def __init__(self, samp_rate, center_freq, gain):
        gr.top_block.__init__(self, "BPSK Decoder", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.packet_len = packet_len = 10

        ##################################################
        # Blocks
        ##################################################
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + ""
        )
        self.rtlsdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(center_freq, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(0, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(0, 0)
        self.rtlsdr_source_0.set_gain_mode(False, 0)
        self.rtlsdr_source_0.set_gain(gain, 0)
        self.rtlsdr_source_0.set_if_gain(30, 0)
        self.rtlsdr_source_0.set_bb_gain(15, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                500e3,
                300e3,
                window.WIN_HAMMING,
                6.76))
        self.epy_block_1 = epy_block_1.bpsk_decoder(transmittedBitsLen=190, transition_threshold=0.5, bit_validation_threshold=0.7, silence_threshold=20000, bit_length=670, buffer_size=1000000, starting_sequence='0,1,0,1,1,1')
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(15, 0.02, firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32), 64, 16, 1.5, 1)
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(.02, 2, True)
        self.blocks_selector_0 = blocks.selector(gr.sizeof_gr_complex*1,0,1)
        self.blocks_selector_0.set_enabled(True)
        self.blocks_selector_0.set_max_output_buffer(1000000)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex*1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_selector_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.blocks_selector_0, 1), (self.epy_block_1, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.digital_pfb_clock_sync_xxx_0, 0))
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.blocks_selector_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.rtlsdr_source_0, 0), (self.low_pass_filter_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 500e3, 300e3, window.WIN_HAMMING, 6.76))
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len

def main():
    parser = ArgumentParser()
    parser.add_argument("samp_rate", type=float, help="Sample rate")
    parser.add_argument("packet_len", type=int, help="Packet length")
    parser.add_argument("center_freq", type=float, help="Center frequency")
    parser.add_argument("gain", type=int, help="Gain")
    args = parser.parse_args()

    tb = BPSK_message_decoder(args.samp_rate, args.center_freq, args.gain)
    tb.start()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()