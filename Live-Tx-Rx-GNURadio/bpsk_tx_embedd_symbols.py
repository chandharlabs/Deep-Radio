import numpy as np
import time
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Text to 2-bit Symbols with Safe Delay',
            in_sig=None,
            out_sig=[np.uint8]
        )

        # Load text file as bytes
        with open("/home/chandharlabs/Downloads/HackrfONE_RTLSDR_demo-main(1)/HackrfONE_RTLSDR_demo-main/test.txt", "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Convert to bits
        bit_array = np.unpackbits(byte_data)
        if len(bit_array) % 2 != 0:
            bit_array = np.append(bit_array, 0)

        # Convert bits to 2-bit symbols (00→0, 01→1, 10→2, 11→3)
        #self.symbol_values = (bit_array[::2] << 1) | bit_array[1::2]
        self.symbol_values = np.concatenate((     np.zeros(100, dtype=np.uint8),      (bit_array[::2] << 1) | bit_array[1::2] ))

        #self.symbol_values = [0,1,2,3,2,2,3,3,1,1,1,2,2,2,2,3,3] #(bit_array[::2] << 1) | bit_array[1::2]
        self.index = 0
        self.last_repeat_time = time.time()
        self.repeat_delay = 0 #.1  # seconds

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        out_index = 0

        while out_index < n:
            # If we're starting a new repeat, ensure delay has passed
            if self.index == 0:
                current_time = time.time()
                if current_time < self.last_repeat_time + self.repeat_delay:
                    # Don't output anything yet
                    return out_index  # return the number of items written (0 here if paused)

                self.last_repeat_time = current_time

            out[out_index] = self.symbol_values[self.index]

            self.index += 1
            out_index += 1

            if self.index >= len(self.symbol_values):
                self.index = 0  # Restart for next round

        return out_index

