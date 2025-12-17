import numpy as np
from gnuradio import gr

class PhaseCorrector(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self, 
                               name="Phase Corrector",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.last_symbol = None

    def work(self, input_items, output_items):
        in_data = input_items[0]
        out_data = output_items[0]

        # Detect if phase has flipped
        if self.last_symbol is not None:
            phase_diff = np.angle(in_data[0] * np.conj(self.last_symbol))
            if abs(phase_diff) > np.pi / 2:  # 180° phase flip
                out_data[:] = -in_data  # Correct the flip
            else:
                out_data[:] = in_data
        else:
            out_data[:] = in_data

        self.last_symbol = in_data[-1]  # Store last symbol for comparison
        return len(out_data)

