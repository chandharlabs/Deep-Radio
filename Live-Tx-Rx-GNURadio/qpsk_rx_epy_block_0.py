import numpy as np
from gnuradio import gr
import os

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='QPSK Decoder',
            in_sig=[np.uint8],
            out_sig=None
        )

        self.zero_count = 0
        self.zero_threshold = 1000
        self.capturing = False
        self.buffer = []

        # Output files
        self.raw_file = open("peak_capture_clean.txt", "w")
        self.ascii_file = open("decoded_ascii.txt", "w")

        # Accumulated bitstream
        self.rx_bits = []

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

        bits = self._symbols_to_bits(downsampled)
        if not bits:
            return

        # Convert bits to ASCII and write
        ascii_str = self._decode_bits_to_ascii(bits)
        self.ascii_file.write(ascii_str + "\n")
        self.ascii_file.flush()

        # ✅ Extract bits from decoded ASCII (not from symbols)
        decoded_bytes = ascii_str.encode('utf-8', errors='replace')
        binary_bits = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8))

        # ✅ Save binary_bits as rx_bits
        np.save("rx_bits.npy", binary_bits)
        with open("rx_bits.txt", "w") as f:
            np.savetxt(f, binary_bits, fmt='%d', delimiter='')

       
        # ✅ Compute BER if tx_bits.npy exists
        if os.path.exists("tx_bits.npy"):
            tx_bits = np.load("tx_bits.npy")
            min_len = min(len(tx_bits), len(binary_bits))
            bit_errors = np.sum(tx_bits[:min_len] != binary_bits[:min_len])
            ber = bit_errors / min_len
            with open("ber_result.txt", "a") as f:
                f.write(f"Bit Errors: {bit_errors}\n")
                f.write(f"Total Bits Compared: {min_len}\n")
                f.write(f"BER: {ber:.8f}\n")
        else:
            with open("ber_result.txt", "a") as f:
                f.write("tx_bits.npy not found. BER not computed.\n")

        # Check if the CSV file already exists
        write_header = not os.path.exists("ber_log.csv")

         # Append the new BER entry
        with open("ber_log.csv", "a") as csv_file:
            if write_header:
                csv_file.write("Total Bits Compared,Bit Errors,BER\n")
            csv_file.write(f"{min_len},{bit_errors},{ber:.8f}\n")



    def _symbols_to_bits(self, symbols):
        bits = []
        for sym in symbols:
            bits.extend([(sym >> 1) & 1, sym & 1])  # MSB first
        return bits

    def _decode_bits_to_ascii(self, bits):
        padded_bits = bits.copy()
        while len(padded_bits) % 8 != 0:
            padded_bits.append(0)

        bytes_out = []
        for i in range(0, len(padded_bits), 8):
            byte_val = 0
            for b in padded_bits[i:i+8]:
                byte_val = (byte_val << 1) | b
            bytes_out.append(byte_val)

        try:
            return bytes(bytes_out).decode('utf-8', errors='replace')
        except Exception as e:
            return f"[Decode Error: {e}]"

    def _finalize_rx_data(self):
        if not self.rx_bits:
            with open("ber_result.txt", "w") as f:
                f.write("No received bits detected. BER not computed.\n")
            return

        rx_bits_array = np.array(self.rx_bits, dtype=np.uint8)
        np.save("rx_bits.npy", rx_bits_array)
        np.savetxt("rx_bits.txt", rx_bits_array, fmt='%d')

        if os.path.exists("tx_bits.npy"):
            tx_bits = np.load("tx_bits.npy")
            min_len = min(len(tx_bits), len(rx_bits_array))
            tx_trim = tx_bits[:min_len]
            rx_trim = rx_bits_array[:min_len]
            bit_errors = np.sum(tx_trim != rx_trim)
            ber = bit_errors / min_len
            with open("ber_result.txt", "w") as f:
                f.write(f"Bit Errors: {bit_errors}\n")
                f.write(f"Total Bits Compared: {min_len}\n")
                f.write(f"BER: {ber:.8f}\n")
        else:
            with open("ber_result.txt", "w") as f:
                f.write("tx_bits.npy not found. BER not computed.\n")

    def __del__(self):
        # Final capture check
        if self.capturing and self.buffer:
            self._save_and_decode()

        self._finalize_rx_data()
        self.raw_file.close()
        self.ascii_file.close()

