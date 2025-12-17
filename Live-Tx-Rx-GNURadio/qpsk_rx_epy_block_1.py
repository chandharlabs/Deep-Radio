import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self, threshold=0.3):
        gr.sync_block.__init__(
            self,
            name='AmplitudeGate',
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        self.threshold = threshold

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        avg_amplitude = np.mean(np.abs(in0))

        if avg_amplitude > self.threshold:
            out[:] = in0
        else:
            out[:] = np.zeros_like(in0)

        return len(out)

