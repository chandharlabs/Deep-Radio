import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Burst Capture for BPSK with ASCII Decode',
            in_sig=[np.uint8],  # Assuming input stream contains 0s and 1s
            out_sig=None
        )

        self.zero_count = 0
        self.zero_threshold = 1000  # Silence threshold to detect burst
        self.capturing = False
        self.buffer = []

        # Output files
        self.raw_file = open("peak_capture_clean.txt", "w")
        self.ascii_file = open("decoded_ascii.txt", "w")

    def work(self, input_items, output_items):
        in0 = input_items[0]

        for val in in0:
            if val == 0:
                self.zero_count += 1
                if self.capturing:
                    self.buffer.append(val)
            else:
                if not self.capturing and self.zero_count >= self.zero_threshold:
                    self.capturing = True
                    self.buffer = [val]
                elif self.capturing:
                    self.buffer.append(val)
                self.zero_count = 0

            if self.capturing and self.zero_count >= self.zero_threshold:
                self._save_and_decode()
                self.buffer = []
                self.capturing = False

        return len(in0)

    def _trim_zeros(self, buf):
        arr = np.array(buf)
        nonzero = np.nonzero(arr)[0]
        if nonzero.size == 0:
            return []
        start, end = nonzero[0], nonzero[-1] + 1
        return arr[start:end].tolist()

    def _burst_downsample_from_peak(self, values, zero_gap_threshold=120):
        result = []
        last_index = None
        i = 0
        n = len(values)

        while i < n:
            if values[i] != 0:
                start = i
                sym = values[i]
                while i < n and values[i] == sym:
                    i += 1
                if last_index is not None and (start - last_index) > zero_gap_threshold:
                    result.append(0)
                result.append(sym)
                last_index = start
            else:
                i += 1

        return result

    def _save_and_decode(self):
        trimmed = self._trim_zeros(self.buffer)
        if not trimmed:
            return

        downsampled = self._burst_downsample_from_peak(trimmed)
        if not downsampled:
            return

        self.raw_file.write(" ".join(map(str, downsampled)) + "\n")
        self.raw_file.flush()

        ascii_str = self._decode_symbols_to_ascii(downsampled)
        self.ascii_file.write(ascii_str + "\n")
        self.ascii_file.flush()

        print(f"Captured {len(downsampled)} bits → ASCII: {ascii_str.strip()}")

    def _decode_symbols_to_ascii(self, symbols):
        bits = list(symbols)

        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)

        bytes_out = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for b in bits[i:i+8]:
                byte_val = (byte_val << 1) | b
            bytes_out.append(byte_val)

        try:
            ascii_str = bytes(bytes_out).decode('utf-8', errors='replace')
        except Exception as e:
            ascii_str = f"[Decode Error: {e}]"

        return ascii_str

    def __del__(self):
        if self.buffer:
            self._save_and_decode()
        self.raw_file.close()
        self.ascii_file.close()

