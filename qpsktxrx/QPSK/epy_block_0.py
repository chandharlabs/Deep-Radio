import numpy as np
import time
from gnuradio import gr
import os

class blk(gr.sync_block):
    def __init__(self, file_path="C:\\Users\\admin\\Desktop\\QPSK\\test.txt", preamble_len=100):
        gr.sync_block.__init__(
            self,
            name='Text to 2-bit Symbols with Safe Delay',
            in_sig=None,
            out_sig=[np.uint8]
        )

        self.file_path = file_path
        self.preamble_len = preamble_len

        # Load text file as bytes
        with open(self.file_path, "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Convert to bits
        bit_array = np.unpackbits(byte_data)
        if len(bit_array) % 2 != 0:
            bit_array = np.append(bit_array, 0)

        save_dir = os.path.dirname(self.file_path)
        np.save(os.path.join(save_dir, "C:\\Users\\admin\\Desktop\\QPSK\\tx_bits.npy"), bit_array)
        with open("C:\\Users\\admin\\Desktop\\QPSK\\tx_bits.txt", "w") as f:
            np.savetxt(f, bit_array, fmt='%d', delimiter='')


        # Convert bits to 2-bit symbols (00→0, 01→1, 10→2, 11→3)
        symbol_values = (bit_array[::2] << 1) | bit_array[1::2]

        # Add preamble of zeros
        self.symbol_values = np.concatenate((
            np.zeros(self.preamble_len, dtype=np.uint8),
            symbol_values
        ))

        self.index = 0
        self.last_repeat_time = time.time()
        self.repeat_delay = 0  # seconds

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        out_index = 0

        while out_index < n:
            if self.index == 0:
                current_time = time.time()
                if current_time < self.last_repeat_time + self.repeat_delay:
                    return out_index

                self.last_repeat_time = current_time

            out[out_index] = self.symbol_values[self.index]
            self.index += 1
            out_index += 1

            if self.index >= len(self.symbol_values):
                self.index = 0  # restart

        return out_index

