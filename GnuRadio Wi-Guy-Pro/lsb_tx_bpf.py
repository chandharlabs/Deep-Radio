#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: LSB TX
# Author: Alexandru Csete OZ9AEC
# Description: LSB transmitter using complex band pass filter.
# Generated: Sun Nov 24 01:07:39 2019
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


class lsb_tx_bpf(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="LSB TX")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.usrp_freq = usrp_freq = 90.05
        self.offset = offset = 10000
        self.high = high = -300
        self.filter_width = filter_width = 2700
        self.carrier_level = carrier_level = 1
        self.audio_gain = audio_gain = 0.5
        self.tx_freq = tx_freq = usrp_freq+offset
        self.signal = signal = 1000
        self.samp_rate = samp_rate = 50000
        self.rf_gain = rf_gain = 0
        self.mod_index = mod_index = audio_gain/carrier_level
        self.low = low = high-filter_width

        ##################################################
        # Blocks
        ##################################################
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
        	self.GetWin(),
        	baseband_freq=0,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=samp_rate,
        	fft_size=512,
        	fft_rate=15,
        	average=True,
        	avg_alpha=0.5,
        	title='Modulator output',
        	peak_hold=False,
        	size=(800,300),
        )
        self.GridAdd(self.wxgui_fftsink2_0.win, 0, 0, 4, 4)
        self._usrp_freq_text_box = forms.text_box(
        	parent=self.GetWin(),
        	value=self.usrp_freq,
        	callback=self.set_usrp_freq,
        	label='USRP',
        	converter=forms.float_converter(),
        )
        self.GridAdd(self._usrp_freq_text_box, 4, 1, 1, 1)
        self._tx_freq_static_text = forms.static_text(
        	parent=self.GetWin(),
        	value=self.tx_freq,
        	callback=self.set_tx_freq,
        	label='TX',
        	converter=forms.float_converter(),
        )
        self.GridAdd(self._tx_freq_static_text, 4, 3, 1, 1)
        _signal_sizer = wx.BoxSizer(wx.VERTICAL)
        self._signal_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_signal_sizer,
        	value=self.signal,
        	callback=self.set_signal,
        	label='Signal level',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._signal_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_signal_sizer,
        	value=self.signal,
        	callback=self.set_signal,
        	minimum=0,
        	maximum=10000,
        	num_steps=1000,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_signal_sizer, 6, 3, 1, 1)
        _rf_gain_sizer = wx.BoxSizer(wx.VERTICAL)
        self._rf_gain_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_rf_gain_sizer,
        	value=self.rf_gain,
        	callback=self.set_rf_gain,
        	label='USRP gain',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._rf_gain_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_rf_gain_sizer,
        	value=self.rf_gain,
        	callback=self.set_rf_gain,
        	minimum=0,
        	maximum=20,
        	num_steps=200,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_rf_gain_sizer, 7, 3, 1, 1)
        self.rational_resampler_xxx_0_0 = filter.rational_resampler_ccf(
                interpolation=25,
                decimation=24,
                taps=None,
                fractional_bw=None,
        )
        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
                interpolation=25,
                decimation=24,
                taps=None,
                fractional_bw=None,
        )
        self.osmosdr_sink_0 = osmosdr.sink( args="numchan=" + str(1) + " " + '' )
        self.osmosdr_sink_0.set_sample_rate(500000)
        self.osmosdr_sink_0.set_center_freq(90.05e6, 0)
        self.osmosdr_sink_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0.set_gain(20, 0)
        self.osmosdr_sink_0.set_if_gain(20, 0)
        self.osmosdr_sink_0.set_bb_gain(20, 0)
        self.osmosdr_sink_0.set_antenna('', 0)
        self.osmosdr_sink_0.set_bandwidth(40e3, 0)

        _offset_sizer = wx.BoxSizer(wx.VERTICAL)
        self._offset_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_offset_sizer,
        	value=self.offset,
        	callback=self.set_offset,
        	label='Offset',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._offset_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_offset_sizer,
        	value=self.offset,
        	callback=self.set_offset,
        	minimum=-25000,
        	maximum=25000,
        	num_steps=1000,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_offset_sizer, 5, 0, 1, 4)
        self._mod_index_static_text = forms.static_text(
        	parent=self.GetWin(),
        	value=self.mod_index,
        	callback=self.set_mod_index,
        	label='Mod. index',
        	converter=forms.float_converter(),
        )
        self.GridAdd(self._mod_index_static_text, 7, 0, 1, 1)
        _filter_width_sizer = wx.BoxSizer(wx.VERTICAL)
        self._filter_width_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_filter_width_sizer,
        	value=self.filter_width,
        	callback=self.set_filter_width,
        	label='Filter width',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._filter_width_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_filter_width_sizer,
        	value=self.filter_width,
        	callback=self.set_filter_width,
        	minimum=1000,
        	maximum=5000,
        	num_steps=400,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_filter_width_sizer, 6, 0, 1, 1)
        _carrier_level_sizer = wx.BoxSizer(wx.VERTICAL)
        self._carrier_level_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_carrier_level_sizer,
        	value=self.carrier_level,
        	callback=self.set_carrier_level,
        	label='Carrier level',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._carrier_level_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_carrier_level_sizer,
        	value=self.carrier_level,
        	callback=self.set_carrier_level,
        	minimum=0,
        	maximum=10,
        	num_steps=100,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_carrier_level_sizer, 7, 1, 1, 1)
        self.blocks_wavfile_source_0 = blocks.wavfile_source('/home/prabhuchandhar/Desktop/a2002011001-e02.wav', True)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_vcc((0, ))
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vff((500e-3, ))
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.band_pass_filter_0 = filter.interp_fir_filter_ccc(1, firdes.complex_band_pass(
        	1, samp_rate, low, high, 100, firdes.WIN_HAMMING, 6.76))
        _audio_gain_sizer = wx.BoxSizer(wx.VERTICAL)
        self._audio_gain_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_audio_gain_sizer,
        	value=self.audio_gain,
        	callback=self.set_audio_gain,
        	label='Audio gain',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._audio_gain_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_audio_gain_sizer,
        	value=self.audio_gain,
        	callback=self.set_audio_gain,
        	minimum=0,
        	maximum=2,
        	num_steps=200,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_audio_gain_sizer, 6, 1, 1, 1)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, 0, 1, 0)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, 0, 1, 0)
        self.analog_const_source_x_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, 0)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_multiply_xx_0_0, 0))
        self.connect((self.band_pass_filter_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.osmosdr_sink_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.band_pass_filter_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.wxgui_fftsink2_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.rational_resampler_xxx_0_0, 0))
        self.connect((self.blocks_wavfile_source_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.blocks_float_to_complex_0, 1))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.blocks_multiply_const_vxx_1, 0))

    def get_usrp_freq(self):
        return self.usrp_freq

    def set_usrp_freq(self, usrp_freq):
        self.usrp_freq = usrp_freq
        self._usrp_freq_text_box.set_value(self.usrp_freq)
        self.set_tx_freq(self.usrp_freq+self.offset)

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset
        self.set_tx_freq(self.usrp_freq+self.offset)
        self._offset_slider.set_value(self.offset)
        self._offset_text_box.set_value(self.offset)

    def get_high(self):
        return self.high

    def set_high(self, high):
        self.high = high
        self.set_low(self.high-self.filter_width)
        self.band_pass_filter_0.set_taps(firdes.complex_band_pass(1, self.samp_rate, self.low, self.high, 100, firdes.WIN_HAMMING, 6.76))

    def get_filter_width(self):
        return self.filter_width

    def set_filter_width(self, filter_width):
        self.filter_width = filter_width
        self.set_low(self.high-self.filter_width)
        self._filter_width_slider.set_value(self.filter_width)
        self._filter_width_text_box.set_value(self.filter_width)

    def get_carrier_level(self):
        return self.carrier_level

    def set_carrier_level(self, carrier_level):
        self.carrier_level = carrier_level
        self.set_mod_index(self.audio_gain/self.carrier_level)
        self._carrier_level_slider.set_value(self.carrier_level)
        self._carrier_level_text_box.set_value(self.carrier_level)

    def get_audio_gain(self):
        return self.audio_gain

    def set_audio_gain(self, audio_gain):
        self.audio_gain = audio_gain
        self.set_mod_index(self.audio_gain/self.carrier_level)
        self._audio_gain_slider.set_value(self.audio_gain)
        self._audio_gain_text_box.set_value(self.audio_gain)

    def get_tx_freq(self):
        return self.tx_freq

    def set_tx_freq(self, tx_freq):
        self.tx_freq = tx_freq
        self._tx_freq_static_text.set_value(self.tx_freq)

    def get_signal(self):
        return self.signal

    def set_signal(self, signal):
        self.signal = signal
        self._signal_slider.set_value(self.signal)
        self._signal_text_box.set_value(self.signal)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.wxgui_fftsink2_0.set_sample_rate(self.samp_rate)
        self.band_pass_filter_0.set_taps(firdes.complex_band_pass(1, self.samp_rate, self.low, self.high, 100, firdes.WIN_HAMMING, 6.76))
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self._rf_gain_slider.set_value(self.rf_gain)
        self._rf_gain_text_box.set_value(self.rf_gain)

    def get_mod_index(self):
        return self.mod_index

    def set_mod_index(self, mod_index):
        self.mod_index = mod_index
        self._mod_index_static_text.set_value(self.mod_index)

    def get_low(self):
        return self.low

    def set_low(self, low):
        self.low = low
        self.band_pass_filter_0.set_taps(firdes.complex_band_pass(1, self.samp_rate, self.low, self.high, 100, firdes.WIN_HAMMING, 6.76))


def main(top_block_cls=lsb_tx_bpf, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
