#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: BPSK Decoder
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import BPSK_message_decoder_epy_block_1 as epy_block_1  # embedded python block
import osmosdr
import time



from gnuradio import qtgui

class BPSK_message_decoder(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "BPSK Decoder", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("BPSK Decoder")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "BPSK_message_decoder")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.threshold = threshold = 0.5
        self.selector = selector = 0
        self.samp_rate = samp_rate = 1e6
        self.rf_gain_slider = rf_gain_slider = 17
        self.packet_len = packet_len = 10
        self.if_gain_slider = if_gain_slider = 30
        self.bit_len = bit_len = 670
        self.bb_gain_slider = bb_gain_slider = 15

        ##################################################
        # Blocks
        ##################################################
        self._threshold_range = Range(-1, 1, 0.01, 0.5, 200)
        self._threshold_win = RangeWidget(self._threshold_range, self.set_threshold, "'threshold'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._threshold_win)
        _selector_check_box = Qt.QCheckBox("selector")
        self._selector_choices = {True: 1, False: 0}
        self._selector_choices_inv = dict((v,k) for k,v in self._selector_choices.items())
        self._selector_callback = lambda i: Qt.QMetaObject.invokeMethod(_selector_check_box, "setChecked", Qt.Q_ARG("bool", self._selector_choices_inv[i]))
        self._selector_callback(self.selector)
        _selector_check_box.stateChanged.connect(lambda i: self.set_selector(self._selector_choices[bool(i)]))
        self.top_layout.addWidget(_selector_check_box)
        self._rf_gain_slider_range = Range(0, 100, 1, 17, 200)
        self._rf_gain_slider_win = RangeWidget(self._rf_gain_slider_range, self.set_rf_gain_slider, "'rf_gain_slider'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rf_gain_slider_win)
        self._if_gain_slider_range = Range(0, 100, 1, 30, 200)
        self._if_gain_slider_win = RangeWidget(self._if_gain_slider_range, self.set_if_gain_slider, "'if_gain_slider'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._if_gain_slider_win)
        self._bit_len_range = Range(0, 2000, 1, 670, 200)
        self._bit_len_win = RangeWidget(self._bit_len_range, self.set_bit_len, "Bit Length", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._bit_len_win)
        self._bb_gain_slider_range = Range(0, 100, 1, 15, 200)
        self._bb_gain_slider_win = RangeWidget(self._bb_gain_slider_range, self.set_bb_gain_slider, "'bb_gain_slider'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._bb_gain_slider_win)
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + ""
        )
        self.rtlsdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(700e6, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(0, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(0, 0)
        self.rtlsdr_source_0.set_gain_mode(False, 0)
        self.rtlsdr_source_0.set_gain(rf_gain_slider, 0)
        self.rtlsdr_source_0.set_if_gain(if_gain_slider, 0)
        self.rtlsdr_source_0.set_bb_gain(bb_gain_slider, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.qtgui_time_sink_x_1_1 = qtgui.time_sink_c(
            202400, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1_1.set_y_axis(-1.3, 1.2)

        self.qtgui_time_sink_x_1_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1_1.enable_tags(True)
        self.qtgui_time_sink_x_1_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1_1.enable_grid(False)
        self.qtgui_time_sink_x_1_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_1_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_1_win)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                500e3,
                300e3,
                window.WIN_HAMMING,
                6.76))
        self.epy_block_1 = epy_block_1.bpsk_decoder(transmittedBitsLen=126, transition_threshold=threshold, bit_validation_threshold=0.7, silence_threshold=20000, bit_length=bit_len, buffer_size=1000000, starting_sequence='0,1,0,1,1,1')
        self.digital_pfb_clock_sync_xxx_0 = digital.pfb_clock_sync_ccf(15, 0.02, firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32), 64, 16, 1.5, 1)
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(.02, 2, True)
        self.blocks_selector_0 = blocks.selector(gr.sizeof_gr_complex*1,0,selector)
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
        self.connect((self.digital_pfb_clock_sync_xxx_0, 0), (self.qtgui_time_sink_x_1_1, 0))
        self.connect((self.low_pass_filter_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.rtlsdr_source_0, 0), (self.low_pass_filter_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "BPSK_message_decoder")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.epy_block_1.transition_threshold = self.threshold

    def get_selector(self):
        return self.selector

    def set_selector(self, selector):
        self.selector = selector
        self._selector_callback(self.selector)
        self.blocks_selector_0.set_output_index(self.selector)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 500e3, 300e3, window.WIN_HAMMING, 6.76))
        self.qtgui_time_sink_x_1_1.set_samp_rate(self.samp_rate)
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)

    def get_rf_gain_slider(self):
        return self.rf_gain_slider

    def set_rf_gain_slider(self, rf_gain_slider):
        self.rf_gain_slider = rf_gain_slider
        self.rtlsdr_source_0.set_gain(self.rf_gain_slider, 0)

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len

    def get_if_gain_slider(self):
        return self.if_gain_slider

    def set_if_gain_slider(self, if_gain_slider):
        self.if_gain_slider = if_gain_slider
        self.rtlsdr_source_0.set_if_gain(self.if_gain_slider, 0)

    def get_bit_len(self):
        return self.bit_len

    def set_bit_len(self, bit_len):
        self.bit_len = bit_len
        self.epy_block_1.bit_length = self.bit_len

    def get_bb_gain_slider(self):
        return self.bb_gain_slider

    def set_bb_gain_slider(self, bb_gain_slider):
        self.bb_gain_slider = bb_gain_slider
        self.rtlsdr_source_0.set_bb_gain(self.bb_gain_slider, 0)




def main(top_block_cls=BPSK_message_decoder, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
