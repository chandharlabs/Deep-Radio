#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# Generated: Thu Mar  7 04:06:02 2019
##################################################


if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio import wxgui
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.wxgui import fftsink2
from gnuradio.wxgui import forms
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import osmosdr
import time
import wx


class top_block(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Top Block")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 8000
        self.quad_rate = quad_rate = 32000
        self.ptt = ptt = 0
        self.out_rate = out_rate = 1000000
        self.enable_tone = enable_tone = 0
        self.carrier_freq = carrier_freq = 145.350e6

        ##################################################
        # Blocks
        ##################################################
        _ptt_sizer = wx.BoxSizer(wx.VERTICAL)
        self._ptt_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_ptt_sizer,
        	value=self.ptt,
        	callback=self.set_ptt,
        	label='ptt',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._ptt_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_ptt_sizer,
        	value=self.ptt,
        	callback=self.set_ptt,
        	minimum=0,
        	maximum=1,
        	num_steps=1,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_ptt_sizer)
        _enable_tone_sizer = wx.BoxSizer(wx.VERTICAL)
        self._enable_tone_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_enable_tone_sizer,
        	value=self.enable_tone,
        	callback=self.set_enable_tone,
        	label='enable_tone',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._enable_tone_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_enable_tone_sizer,
        	value=self.enable_tone,
        	callback=self.set_enable_tone,
        	minimum=0,
        	maximum=1,
        	num_steps=1,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_enable_tone_sizer)
        self._carrier_freq_text_box = forms.text_box(
        	parent=self.GetWin(),
        	value=self.carrier_freq,
        	callback=self.set_carrier_freq,
        	label='carrier_freq',
        	converter=forms.float_converter(),
        )
        self.Add(self._carrier_freq_text_box)
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
        	self.GetWin(),
        	baseband_freq=0,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=quad_rate,
        	fft_size=1024,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='FFT Plot',
        	peak_hold=False,
        )
        self.Add(self.wxgui_fftsink2_0.win)
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=out_rate,
                decimation=quad_rate,
                taps=None,
                fractional_bw=None,
        )
        self.osmosdr_sink_0 = osmosdr.sink( args="numchan=" + str(1) + " " + '' )
        self.osmosdr_sink_0.set_sample_rate(out_rate)
        self.osmosdr_sink_0.set_center_freq(carrier_freq, 0)
        self.osmosdr_sink_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0.set_gain(14, 0)
        self.osmosdr_sink_0.set_if_gain(47, 0)
        self.osmosdr_sink_0.set_bb_gain(20, 0)
        self.osmosdr_sink_0.set_antenna('', 0)
        self.osmosdr_sink_0.set_bandwidth(0, 0)

        self.low_pass_filter_0 = filter.fir_filter_fff(1, firdes.low_pass(
        	1, samp_rate, 4e3, 500, firdes.WIN_HAMMING, 6.76))
        self.blocks_multiply_const_vxx_2 = blocks.multiply_const_vcc((ptt, ))
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_vff((enable_tone, ))
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vff((1-enable_tone, ))
        self.blocks_add_xx_0 = blocks.add_vff(1)
        self.audio_source_0 = audio.source(samp_rate, '', True)
        self.analog_sig_source_x_0 = analog.sig_source_f(samp_rate, analog.GR_COS_WAVE, 1750, 0.1, 0)
        self.analog_nbfm_tx_0 = analog.nbfm_tx(
        	audio_rate=samp_rate,
        	quad_rate=quad_rate,
        	tau=75e-6,
        	max_dev=5e3,
        	fh=-1.0,
                )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_nbfm_tx_0, 0), (self.blocks_multiply_const_vxx_2, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.audio_source_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_2, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_2, 0), (self.wxgui_fftsink2_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.analog_nbfm_tx_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.osmosdr_sink_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 4e3, 500, firdes.WIN_HAMMING, 6.76))
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_quad_rate(self):
        return self.quad_rate

    def set_quad_rate(self, quad_rate):
        self.quad_rate = quad_rate
        self.wxgui_fftsink2_0.set_sample_rate(self.quad_rate)

    def get_ptt(self):
        return self.ptt

    def set_ptt(self, ptt):
        self.ptt = ptt
        self._ptt_slider.set_value(self.ptt)
        self._ptt_text_box.set_value(self.ptt)
        self.blocks_multiply_const_vxx_2.set_k((self.ptt, ))

    def get_out_rate(self):
        return self.out_rate

    def set_out_rate(self, out_rate):
        self.out_rate = out_rate
        self.osmosdr_sink_0.set_sample_rate(self.out_rate)

    def get_enable_tone(self):
        return self.enable_tone

    def set_enable_tone(self, enable_tone):
        self.enable_tone = enable_tone
        self._enable_tone_slider.set_value(self.enable_tone)
        self._enable_tone_text_box.set_value(self.enable_tone)
        self.blocks_multiply_const_vxx_1.set_k((self.enable_tone, ))
        self.blocks_multiply_const_vxx_0.set_k((1-self.enable_tone, ))

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq
        self._carrier_freq_text_box.set_value(self.carrier_freq)
        self.osmosdr_sink_0.set_center_freq(self.carrier_freq, 0)


def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
