#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Env Det Test
# Generated: Sat Nov 23 15:42:50 2019
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
from gnuradio.filter import firdes
from gnuradio.wxgui import forms
from gnuradio.wxgui import scopesink2
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import ed
import wx


class env_det_test(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Env Det Test")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.vol = vol = .1
        self.samp_rate = samp_rate = 96000
        self.rc_coeff = rc_coeff = 0.15
        self.mode = mode = 1
        self.level_mod = level_mod = 1
        self.level_carrier = level_carrier = 1
        self.freq = freq = 600
        self.carrier = carrier = 40000

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
        	maximum=1,
        	num_steps=10,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_vol_sizer, 0, 30, 1, 10)
        self.tabs = self.tabs = wx.Notebook(self.GetWin(), style=wx.NB_TOP)
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Signal")
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Detector")
        self.tabs.AddPage(grc_wxgui.Panel(self.tabs), "Audio")
        self.Add(self.tabs)
        _rc_coeff_sizer = wx.BoxSizer(wx.VERTICAL)
        self._rc_coeff_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_rc_coeff_sizer,
        	value=self.rc_coeff,
        	callback=self.set_rc_coeff,
        	label='RC Coefficient',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._rc_coeff_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_rc_coeff_sizer,
        	value=self.rc_coeff,
        	callback=self.set_rc_coeff,
        	minimum=0.01,
        	maximum=0.2,
        	num_steps=200,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_rc_coeff_sizer, 0, 50, 1, 10)
        self._mode_chooser = forms.radio_buttons(
        	parent=self.GetWin(),
        	value=self.mode,
        	callback=self.set_mode,
        	label='Detector',
        	choices=[0,1],
        	labels=['Half', 'Full'],
        	style=wx.RA_HORIZONTAL,
        )
        self.GridAdd(self._mode_chooser, 0, 40, 1, 10)
        _level_mod_sizer = wx.BoxSizer(wx.VERTICAL)
        self._level_mod_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_level_mod_sizer,
        	value=self.level_mod,
        	callback=self.set_level_mod,
        	label='Modulation Level',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._level_mod_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_level_mod_sizer,
        	value=self.level_mod,
        	callback=self.set_level_mod,
        	minimum=0,
        	maximum=2,
        	num_steps=20,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_level_mod_sizer, 0, 20, 1, 10)
        _level_carrier_sizer = wx.BoxSizer(wx.VERTICAL)
        self._level_carrier_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_level_carrier_sizer,
        	value=self.level_carrier,
        	callback=self.set_level_carrier,
        	label='Carrier Level',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._level_carrier_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_level_carrier_sizer,
        	value=self.level_carrier,
        	callback=self.set_level_carrier,
        	minimum=0,
        	maximum=2,
        	num_steps=20,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_level_carrier_sizer, 0, 10, 1, 10)
        _freq_sizer = wx.BoxSizer(wx.VERTICAL)
        self._freq_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_freq_sizer,
        	value=self.freq,
        	callback=self.set_freq,
        	label='Tone Freq',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._freq_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_freq_sizer,
        	value=self.freq,
        	callback=self.set_freq,
        	minimum=0,
        	maximum=2000,
        	num_steps=20,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.GridAdd(_freq_sizer, 0, 0, 1, 10)
        self.wxgui_scopesink2_2 = scopesink2.scope_sink_f(
        	self.tabs.GetPage(2).GetWin(),
        	title='Scope Plot',
        	sample_rate=24000,
        	v_scale=0,
        	v_offset=0,
        	t_scale=.02,
        	ac_couple=True,
        	xy_mode=False,
        	num_inputs=1,
        	trig_mode=wxgui.TRIG_MODE_AUTO,
        	y_axis_label='Counts',
        )
        self.tabs.GetPage(2).Add(self.wxgui_scopesink2_2.win)
        self.wxgui_scopesink2_1 = scopesink2.scope_sink_f(
        	self.tabs.GetPage(1).GetWin(),
        	title='Detector',
        	sample_rate=samp_rate,
        	v_scale=0,
        	v_offset=0,
        	t_scale=.002,
        	ac_couple=True,
        	xy_mode=False,
        	num_inputs=1,
        	trig_mode=wxgui.TRIG_MODE_NORM,
        	y_axis_label='Amplitude',
        )
        self.tabs.GetPage(1).Add(self.wxgui_scopesink2_1.win)
        self.wxgui_scopesink2_0 = scopesink2.scope_sink_f(
        	self.tabs.GetPage(0).GetWin(),
        	title='Signal',
        	sample_rate=samp_rate,
        	v_scale=0,
        	v_offset=0,
        	t_scale=.002,
        	ac_couple=True,
        	xy_mode=False,
        	num_inputs=1,
        	trig_mode=wxgui.TRIG_MODE_AUTO,
        	y_axis_label='Counts',
        )
        self.tabs.GetPage(0).Add(self.wxgui_scopesink2_0.win)
        self.low_pass_filter_0 = filter.fir_filter_fff(4, firdes.low_pass(
        	1, samp_rate, 5000, 500, firdes.WIN_HAMMING, 6.76))
        self.ed = ed.blk(threshold=0, mode=mode, coeff=rc_coeff)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_float*1, 100000,True)
        self.blocks_multiply_xx_0 = blocks.multiply_vff(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vff((vol, ))
        self.blocks_add_const_vxx_0 = blocks.add_const_vff((1, ))
        self.band_pass_filter_0 = filter.fir_filter_fff(1, firdes.band_pass(
        	1, samp_rate, 38000, 42000, 500, firdes.WIN_HAMMING, 6.76))
        self.audio_sink_0 = audio.sink(24000, '', True)
        self.analog_sig_source_x_1 = analog.sig_source_f(samp_rate, analog.GR_COS_WAVE, freq, level_mod, 0)
        self.analog_sig_source_x_0 = analog.sig_source_f(samp_rate, analog.GR_COS_WAVE, carrier, level_carrier, 0)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.analog_sig_source_x_1, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.band_pass_filter_0, 0), (self.ed, 0))
        self.connect((self.band_pass_filter_0, 0), (self.wxgui_scopesink2_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.audio_sink_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.wxgui_scopesink2_2, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.band_pass_filter_0, 0))
        self.connect((self.ed, 0), (self.low_pass_filter_0, 0))
        self.connect((self.ed, 0), (self.wxgui_scopesink2_1, 0))
        self.connect((self.low_pass_filter_0, 0), (self.blocks_multiply_const_vxx_0, 0))

    def get_vol(self):
        return self.vol

    def set_vol(self, vol):
        self.vol = vol
        self._vol_slider.set_value(self.vol)
        self._vol_text_box.set_value(self.vol)
        self.blocks_multiply_const_vxx_0.set_k((self.vol, ))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.wxgui_scopesink2_1.set_sample_rate(self.samp_rate)
        self.wxgui_scopesink2_0.set_sample_rate(self.samp_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 5000, 500, firdes.WIN_HAMMING, 6.76))
        self.band_pass_filter_0.set_taps(firdes.band_pass(1, self.samp_rate, 38000, 42000, 500, firdes.WIN_HAMMING, 6.76))
        self.analog_sig_source_x_1.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_rc_coeff(self):
        return self.rc_coeff

    def set_rc_coeff(self, rc_coeff):
        self.rc_coeff = rc_coeff
        self._rc_coeff_slider.set_value(self.rc_coeff)
        self._rc_coeff_text_box.set_value(self.rc_coeff)
        self.ed.coeff = self.rc_coeff

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode
        self._mode_chooser.set_value(self.mode)
        self.ed.mode = self.mode

    def get_level_mod(self):
        return self.level_mod

    def set_level_mod(self, level_mod):
        self.level_mod = level_mod
        self._level_mod_slider.set_value(self.level_mod)
        self._level_mod_text_box.set_value(self.level_mod)
        self.analog_sig_source_x_1.set_amplitude(self.level_mod)

    def get_level_carrier(self):
        return self.level_carrier

    def set_level_carrier(self, level_carrier):
        self.level_carrier = level_carrier
        self._level_carrier_slider.set_value(self.level_carrier)
        self._level_carrier_text_box.set_value(self.level_carrier)
        self.analog_sig_source_x_0.set_amplitude(self.level_carrier)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self._freq_slider.set_value(self.freq)
        self._freq_text_box.set_value(self.freq)
        self.analog_sig_source_x_1.set_frequency(self.freq)

    def get_carrier(self):
        return self.carrier

    def set_carrier(self, carrier):
        self.carrier = carrier
        self.analog_sig_source_x_0.set_frequency(self.carrier)


def main(top_block_cls=env_det_test, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
