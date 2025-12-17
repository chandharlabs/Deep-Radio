#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# Generated: Fri Oct 25 19:31:02 2019
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

from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
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
import numpy
import wx


class top_block(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Top Block")
        _icon_path = "/usr/share/icons/hicolor/32x32/apps/gnuradio-grc.png"
        self.SetIcon(wx.Icon(_icon_path, wx.BITMAP_TYPE_ANY))

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 8
        self.samp_rate = samp_rate = 100E3

        self.pam = pam = digital.constellation_calcdist(([-1, 1]), ([0, 1]), 4, 1).base()

        self.delay = delay = 78

        ##################################################
        # Blocks
        ##################################################
        self.wxgui_scopesink2_0 = scopesink2.scope_sink_f(
        	self.GetWin(),
        	title='Scope Plot',
        	sample_rate=samp_rate,
        	v_scale=0,
        	v_offset=0,
        	t_scale=0,
        	ac_couple=False,
        	xy_mode=False,
        	num_inputs=1,
        	trig_mode=wxgui.TRIG_MODE_FREE,
        	y_axis_label='Counts',
        )
        self.Add(self.wxgui_scopesink2_0.win)
        self.root_raised_cosine_filter_0 = filter.fir_filter_ccf(1, firdes.root_raised_cosine(
        	1, sps, 1, 0.35, 11*sps))
        self.digital_constellation_modulator_0 = digital.generic_mod(
          constellation=pam,
          differential=False,
          samples_per_symbol=sps,
          pre_diff_code=True,
          excess_bw=0.35,
          verbose=False,
          log=False,
          )
        _delay_sizer = wx.BoxSizer(wx.VERTICAL)
        self._delay_text_box = forms.text_box(
        	parent=self.GetWin(),
        	sizer=_delay_sizer,
        	value=self.delay,
        	callback=self.set_delay,
        	label='delay',
        	converter=forms.float_converter(),
        	proportion=0,
        )
        self._delay_slider = forms.slider(
        	parent=self.GetWin(),
        	sizer=_delay_sizer,
        	value=self.delay,
        	callback=self.set_delay,
        	minimum=0,
        	maximum=100,
        	num_steps=100,
        	style=wx.SL_HORIZONTAL,
        	cast=float,
        	proportion=1,
        )
        self.Add(_delay_sizer)
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=0.4,
        	frequency_offset=0.0,
        	epsilon=1.0,
        	taps=(1, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_char*1, samp_rate,True)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 255, 1000)), True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_complex_to_real_0, 0), (self.wxgui_scopesink2_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.root_raised_cosine_filter_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.blocks_complex_to_real_0, 0))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.sps, 1, 0.35, 11*self.sps))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.wxgui_scopesink2_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_pam(self):
        return self.pam

    def set_pam(self, pam):
        self.pam = pam

    def get_delay(self):
        return self.delay

    def set_delay(self, delay):
        self.delay = delay
        self._delay_slider.set_value(self.delay)
        self._delay_text_box.set_value(self.delay)


def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
