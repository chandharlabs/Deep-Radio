import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Burst Capture after Long Zeros',
            in_sig=[np.uint8],
            out_sig=None
        )

        self.zero_count = 0
        self.zero_threshold = 10  # Number of zeros considered as 'long'
        self.capturing = False
        self.buffer = []
        self.file_path = "peak_capture_clean.txt"
        self.f = open(self.file_path, "w")

    def work(self, input_items, output_items):
        in0 = input_items[0]

        for val in in0:
            if val == 0:
                self.zero_count += 1
                if self.capturing:
                    self.buffer.append(val)
            else:
                if not self.capturing and self.zero_count >= self.zero_threshold:
                    # Start capture after long silence
                    self.capturing = True
                    self.buffer = []  # start fresh
                    self.buffer.append(val)
                elif self.capturing:
                    self.buffer.append(val)

                self.zero_count = 0  # Reset on non-zero

            # Stop capture if we've recorded some data and now hit another long zero
            if self.capturing and self.zero_count >= self.zero_threshold:
                self._save_buffer()
                self.buffer = []
                self.capturing = False

        return len(in0)

    def _save_buffer(self):
        # Remove leading/trailing zeros before saving
        trimmed = self._trim_zeros(self.buffer)
        if trimmed:
            self.f.write(" ".join(map(str, trimmed)) + "\n")
            self.f.flush()
            print(f"Captured burst of {len(trimmed)} symbols")

    def _trim_zeros(self, buf):
        # Trim leading and trailing zeros
        arr = np.array(buf)
        nonzero = np.nonzero(arr)[0]
        if nonzero.size == 0:
            return []
        start, end = nonzero[0], nonzero[-1] + 1
        return arr[start:end].tolist()

    def __del__(self):
        if self.buffer:
            self._save_buffer()
        self.f.close()

