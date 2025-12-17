#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: AM Receiver
# Generated: Sat Nov 23 20:13:34 2019
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
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from gnuradio.wxgui import forms
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import osmosdr
import time
import wx


class am_receiver(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="AM Receiver")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.vol = vol = 1
        self.samp_rate = samp_rate = 2e6
        self.rf_gain = rf_gain = 10
        self.passband = passband = 6
        self.freq = freq = 770
        self.decimate = decimate = 10
        self.convert = convert = 125e6

        ##################################################
        # Blocks
        ##################################################
        _rf_gain_sizer = wx.BoxSizer(wx.VERTICAL)
        self._rf_gain_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_rf_gain_sizer,
        	value=self.rf_gain,
        	callback=self.set_rf_gain,
        	label='RF Gain',
        	converter=forms.int_converter(),
        	proportion=0,
        )
        self._rf_gain_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_rf_gain_sizer,
        	value=self.rf_gain,
        	callback=self.set_rf_gain,
        	minimum=10,
        	maximum=70,
        	num_steps=12,
        	style=wx.SL_HORIZONTAL,
        	cast=int,
        	proportion=1,
        )
        self.GridAdd(_rf_gain_sizer, 0, 1, 1, 3)
        _passband_sizer = wx.BoxSizer(wx.VERTICAL)
        self._passband_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_passband_sizer,
        	value=self.passband,
        	callback=self.set_passband,
        	label='Filter kHz',
        	converter=forms.int_converter(),
        	proportion=0,
        )
        self._passband_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_passband_sizer,
        	value=self.passband,
        	callback=self.set_passband,
        	minimum=3,
        	maximum=10,
        	num_steps=7,
        	style=wx.SL_HORIZONTAL,
        	cast=int,
        	proportion=1,
        )
        self.GridAdd(_passband_sizer, 0, 4, 1, 3)
        _vol_sizer = wx.BoxSizer(wx.VERTICAL)
        self._vol_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_vol_sizer,
        	value=self.vol,
        	callback=self.set_vol,
        	label='Volume',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._vol_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_vol_sizer,
        	value=self.vol,
        	callback=self.set_vol,
        	minimum=0,
        	maximum=100,
        	num_steps=100,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_vol_sizer, 0, 8, 1, 3)
        self.rtlsdr_source_0 = osmosdr.source( args="numchan=" + str(1) + " " + 'rtl_tcp=192.168.1.41:1234' )
        self.rtlsdr_source_0.set_sample_rate(500e3)
        self.rtlsdr_source_0.set_center_freq(51e6, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(2, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(2, 0)
        self.rtlsdr_source_0.set_gain_mode(True, 0)
        self.rtlsdr_source_0.set_gain(rf_gain, 0)
        self.rtlsdr_source_0.set_if_gain(40, 0)
        self.rtlsdr_source_0.set_bb_gain(20, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)

        self.low_pass_filter_0 = filter.fir_filter_ccf(1, firdes.low_pass(
        	1, samp_rate/decimate, passband*1000, 200, firdes.WIN_HAMMING, 6.76))
        self._freq_text_box = forms.text_box(
        	parent=self.GetWin(),
        	value=self.freq,
        	callback=self.set_freq,
        	label='Frequency in kHz',
        	converter=forms.int_converter(),
        )
        self.GridAdd(self._freq_text_box, 0, 12, 1, 10)
        self.audio_sink_0 = audio.sink(48000, '', True)
        self.analog_am_demod_cf_0 = analog.am_demod_cf(
        	channel_rate=samp_rate/decimate,
        	audio_decim=1,
        	audio_pass=10000,
        	audio_stop=11000,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_am_demod_cf_0, 0), (self.audio_sink_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.analog_am_demod_cf_0, 0))
        self.connect((self.rtlsdr_source_0, 0), (self.low_pass_filter_0, 0))

    def get_vol(self):
        return self.vol

    def set_vol(self, vol):
        self.vol = vol
        self._vol_slider.set_value(self.vol)
        self._vol_text_box.set_value(self.vol)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate/self.decimate, self.passband*1000, 200, firdes.WIN_HAMMING, 6.76))

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain
        self._rf_gain_slider.set_value(self.rf_gain)
        self._rf_gain_text_box.set_value(self.rf_gain)
        self.rtlsdr_source_0.set_gain(self.rf_gain, 0)

    def get_passband(self):
        return self.passband

    def set_passband(self, passband):
        self.passband = passband
        self._passband_slider.set_value(self.passband)
        self._passband_text_box.set_value(self.passband)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate/self.decimate, self.passband*1000, 200, firdes.WIN_HAMMING, 6.76))

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self._freq_text_box.set_value(self.freq)

    def get_decimate(self):
        return self.decimate

    def set_decimate(self, decimate):
        self.decimate = decimate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate/self.decimate, self.passband*1000, 200, firdes.WIN_HAMMING, 6.76))

    def get_convert(self):
        return self.convert

    def set_convert(self, convert):
        self.convert = convert


def main(top_block_cls=am_receiver, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
