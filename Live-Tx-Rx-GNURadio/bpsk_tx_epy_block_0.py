import numpy as np
import time
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Text to Bits (Byte Output)',
            in_sig=None,
            out_sig=[np.uint8]  # Output individual bits as 0 or 1
        )

        # Load file as bytes
        with open("/home/chandharlabs/Downloads/HackrfONE_RTLSDR_demo-main(1)/HackrfONE_RTLSDR_demo-main/test.txt", "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Convert to bit array (uint8): each element is 0 or 1
        bit_array = np.unpackbits(byte_data)

        # Optional: prepend 100 zeros
        bit_array = np.concatenate((np.zeros(100, dtype=np.uint8), bit_array))

        self.symbol_values = bit_array  # Bits: 0 or 1
        self.index = 0
        self.last_repeat_time = time.time()
        self.repeat_delay = 0  # seconds

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        out_index = 0

        while out_index < n:
            # Delay between repeats
            if self.index == 0:
                current_time = time.time()
                if current_time < self.last_repeat_time + self.repeat_delay:
                    return out_index
                self.last_repeat_time = current_time

            out[out_index] = self.symbol_values[self.index]
            self.index += 1
            out_index += 1

            if self.index >= len(self.symbol_values):
                self.index = 0  # Loop again if needed

        return out_index

