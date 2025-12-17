#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: AM Transmission
# Generated: Sat Nov 23 16:14:51 2019
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
from gnuradio.wxgui import scopesink2
from grc_gnuradio import blks2 as grc_blks2
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import wx


class amx(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="AM Transmission")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 480000
        self.vol = vol = 1
        self.transmit_type = transmit_type = 1

        self.taps_rx = taps_rx = firdes.band_pass(1.0, samp_rate, 120e3, 180e3, 2000, firdes.WIN_HAMMING, 6.76)

        self.source = source = 0
        self.level_signal = level_signal = 0
        self.level_carrier = level_carrier = 0.1

        self.fft_taps = fft_taps = firdes.low_pass(1.0, samp_rate, 180e3, 5000, firdes.WIN_HAMMING, 6.76)


        self.audio_taps = audio_taps = firdes.low_pass(2, samp_rate, 10e3, 1000, firdes.WIN_HAMMING, 6.76)


        ##################################################
        # Blocks
        ##################################################
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
        	maximum=2,
        	num_steps=20,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_vol_sizer, 0, 40, 1, 10)
        self._transmit_type_chooser = forms.radio_buttons(
        	parent=self.GetWin(),
        	value=self.transmit_type,
        	callback=self.set_transmit_type,
        	label='Type',
        	choices=[0,1],
        	labels=["SC", "TC"],
        	style=wx.RA_HORIZONTAL,
        )
        self.GridAdd(self._transmit_type_chooser, 0, 30, 1, 10)
        self.tabs = self.tabs = wx.Notebook(self.GetWin(), style=wx.NB_TOP)
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Audio")
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Signal")
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Spectrum")
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Trap")
        self.Add(self.tabs)
        self._source_chooser = forms.radio_buttons(
        	parent=self.GetWin(),
        	value=self.source,
        	callback=self.set_source,
        	label='Source',
        	choices=[0,1],
        	labels=['Tone', 'Audio'],
        	style=wx.RA_HORIZONTAL,
        )
        self.GridAdd(self._source_chooser, 0, 0, 1, 10)
        _level_signal_sizer = wx.BoxSizer(wx.VERTICAL)
        self._level_signal_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_level_signal_sizer,
        	value=self.level_signal,
        	callback=self.set_level_signal,
        	label='Signal',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._level_signal_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_level_signal_sizer,
        	value=self.level_signal,
        	callback=self.set_level_signal,
        	minimum=0,
        	maximum=1,
        	num_steps=10,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_level_signal_sizer, 0, 20, 1, 10)
        self.wxgui_scopesink2_2 = scopesink2.scope_sink_f(
        	self.tabs.GetPage(3).GetWin(),
        	title='Trapezoid',
        	sample_rate=samp_rate,
        	v_scale=0.5,
        	v_offset=0,
        	t_scale=4,
        	ac_couple=True,
        	xy_mode=True,
        	num_inputs=2,
        	trig_mode=wxgui.TRIG_MODE_AUTO,
        	y_axis_label='Counts',
        )
        self.tabs.GetPage(3).Add(self.wxgui_scopesink2_2.win)
        self.wxgui_scopesink2_1 = scopesink2.scope_sink_f(
        	self.tabs.GetPage(0).GetWin(),
        	title='Audio Source',
        	sample_rate=samp_rate,
        	v_scale=0,
        	v_offset=0,
        	t_scale=5e-3,
        	ac_couple=False,
        	xy_mode=False,
        	num_inputs=1,
        	trig_mode=wxgui.TRIG_MODE_AUTO,
        	y_axis_label='Amplitude',
        )
        self.tabs.GetPage(0).Add(self.wxgui_scopesink2_1.win)
        self.wxgui_scopesink2_0 = scopesink2.scope_sink_f(
        	self.tabs.GetPage(1).GetWin(),
        	title='Transmit Signal',
        	sample_rate=samp_rate,
        	v_scale=0,
        	v_offset=0,
        	t_scale=5e-3,
        	ac_couple=False,
        	xy_mode=False,
        	num_inputs=1,
        	trig_mode=wxgui.TRIG_MODE_NORM,
        	y_axis_label='Amplitude',
        )
        self.tabs.GetPage(1).Add(self.wxgui_scopesink2_0.win)
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
        	self.tabs.GetPage(2).GetWin(),
        	baseband_freq=150e3,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=samp_rate/8,
        	fft_size=512,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='Transmit Spectrum',
        	peak_hold=False,
        )
        self.tabs.GetPage(2).Add(self.wxgui_fftsink2_0.win)
        self.msg_tone = analog.sig_source_f(samp_rate, analog.GR_COS_WAVE, 800, level_signal, 0)
        _level_carrier_sizer = wx.BoxSizer(wx.VERTICAL)
        self._level_carrier_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_level_carrier_sizer,
        	value=self.level_carrier,
        	callback=self.set_level_carrier,
        	label='Carrier',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._level_carrier_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_level_carrier_sizer,
        	value=self.level_carrier,
        	callback=self.set_level_carrier,
        	minimum=0,
        	maximum=1,
        	num_steps=10,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_level_carrier_sizer, 0, 10, 1, 10)
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_fff(10, (audio_taps))
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_fcc(8, (fft_taps), 150e3, samp_rate)
        self.fir_filter_xxx_0 = filter.fir_filter_fcc(2, (taps_rx))
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.carrier = analog.sig_source_f(samp_rate, analog.GR_COS_WAVE, 150000, level_carrier, 0)
        self.blocks_wavfile_source_0 = blocks.wavfile_source('/home/prabhuchandhar/Desktop/a2002011001-e02.wav', True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_float*1, samp_rate,True)
        self.blocks_multiply_xx_0 = blocks.multiply_vff(1)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_vff((vol, ))
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vff((level_signal*35, ))
        self.blocks_add_const_vxx_0 = blocks.add_const_vff((transmit_type, ))
        self.blks2_selector_0 = grc_blks2.selector(
        	item_size=gr.sizeof_float*1,
        	num_inputs=2,
        	num_outputs=1,
        	input_index=source,
        	output_index=0,
        )
        self.audio_sink_0 = audio.sink(48000, '', True)
        self.analog_am_demod_cf_0 = analog.am_demod_cf(
        	channel_rate=samp_rate/2,
        	audio_decim=5,
        	audio_pass=10000,
        	audio_stop=11000,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_am_demod_cf_0, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.blks2_selector_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blks2_selector_0, 0), (self.wxgui_scopesink2_1, 0))
        self.connect((self.blks2_selector_0, 0), (self.wxgui_scopesink2_2, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.audio_sink_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.wxgui_scopesink2_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.wxgui_scopesink2_2, 1))
        self.connect((self.blocks_wavfile_source_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.carrier, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.fir_filter_xxx_0, 0), (self.analog_am_demod_cf_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.wxgui_fftsink2_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.blks2_selector_0, 1))
        self.connect((self.msg_tone, 0), (self.blks2_selector_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.wxgui_scopesink2_2.set_sample_rate(self.samp_rate)
        self.wxgui_scopesink2_1.set_sample_rate(self.samp_rate)
        self.wxgui_scopesink2_0.set_sample_rate(self.samp_rate)
        self.wxgui_fftsink2_0.set_sample_rate(self.samp_rate/8)
        self.msg_tone.set_sampling_freq(self.samp_rate)
        self.carrier.set_sampling_freq(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_vol(self):
        return self.vol

    def set_vol(self, vol):
        self.vol = vol
        self._vol_slider.set_value(self.vol)
        self._vol_text_box.set_value(self.vol)
        self.blocks_multiply_const_vxx_1.set_k((self.vol, ))

    def get_transmit_type(self):
        return self.transmit_type

    def set_transmit_type(self, transmit_type):
        self.transmit_type = transmit_type
        self._transmit_type_chooser.set_value(self.transmit_type)
        self.blocks_add_const_vxx_0.set_k((self.transmit_type, ))

    def get_taps_rx(self):
        return self.taps_rx

    def set_taps_rx(self, taps_rx):
        self.taps_rx = taps_rx
        self.fir_filter_xxx_0.set_taps((self.taps_rx))

    def get_source(self):
        return self.source

    def set_source(self, source):
        self.source = source
        self._source_chooser.set_value(self.source)
        self.blks2_selector_0.set_input_index(int(self.source))

    def get_level_signal(self):
        return self.level_signal

    def set_level_signal(self, level_signal):
        self.level_signal = level_signal
        self._level_signal_slider.set_value(self.level_signal)
        self._level_signal_text_box.set_value(self.level_signal)
        self.msg_tone.set_amplitude(self.level_signal)
        self.blocks_multiply_const_vxx_0.set_k((self.level_signal*35, ))

    def get_level_carrier(self):
        return self.level_carrier

    def set_level_carrier(self, level_carrier):
        self.level_carrier = level_carrier
        self._level_carrier_slider.set_value(self.level_carrier)
        self._level_carrier_text_box.set_value(self.level_carrier)
        self.carrier.set_amplitude(self.level_carrier)

    def get_fft_taps(self):
        return self.fft_taps

    def set_fft_taps(self, fft_taps):
        self.fft_taps = fft_taps
        self.freq_xlating_fir_filter_xxx_0.set_taps((self.fft_taps))

    def get_audio_taps(self):
        return self.audio_taps

    def set_audio_taps(self, audio_taps):
        self.audio_taps = audio_taps
        self.interp_fir_filter_xxx_0.set_taps((self.audio_taps))


def main(top_block_cls=amx, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
