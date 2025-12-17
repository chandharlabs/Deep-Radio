import numpy as np
from collections import Counter
from gnuradio import gr
import threading

class bpsk_decoder(gr.sync_block):
    def __init__(self, transmittedBitsLen=19, transition_threshold=0.5, bit_validation_threshold=0.7, silence_threshold=30000, bit_length=1300, buffer_size=500000, starting_sequence="0,1,0,1,1,1"):
        gr.sync_block.__init__(self,
            name="BPSK Decoder",
            in_sig=[np.complex64],
            out_sig=None)
        
        self.transmittedBitsLen = transmittedBitsLen
        self.transition_threshold = transition_threshold
        self.bit_validation_threshold = bit_validation_threshold
        self.silence_threshold = silence_threshold
        self.bit_length = bit_length
        self.starting_sequence = list(map(int, starting_sequence.split(',')))
        
        self.buffer_size = buffer_size  # Half a million samples
        self.data_buffer = np.array([], dtype=np.complex64)
        self.bit_sequences = []
        self.lock = threading.Lock()
        self.processing_thread = None

    def process_data(self, data):
        real_part = np.real(data)
        half_bit_length = self.bit_length // 2
        
        # Find all transition indices where signal crosses ±0.5
        transition_indices = np.where((real_part[:-1] < self.transition_threshold) & (real_part[1:] > self.transition_threshold) |
                                      (real_part[:-1] > -self.transition_threshold) & (real_part[1:] < -self.transition_threshold))[0]
        
        if len(transition_indices) == 0:
            return

        start_index = transition_indices[0]

        while start_index < len(real_part):
            # Ensure no transitions in the next half bit length
            if np.any((transition_indices > start_index) & (transition_indices <= start_index + half_bit_length)):
                # Move to the next transition
#                print("Jerk Detected on : ", start_index)
                next_transition = transition_indices[transition_indices > start_index][0]
                start_index = next_transition
#                print("Moving to : ", start_index)

            bits = []
            for bit_count in range(self.transmittedBitsLen):
                if start_index >= len(real_part):
                    break

                # Move by half bit length and determine the bit value
                if bit_count == 1:
                    sample_index = start_index +  30
                else:
                    sample_index = start_index + (bit_count * self.bit_length) + 30


                if sample_index >= len(real_part):
                    break

                sample_value = real_part[sample_index]
                if sample_value > self.transition_threshold:
#                    print("Sample Index : ", sample_index, " | Sample Value : ", sample_value)
                    bits.append(1)
                elif sample_value < -self.transition_threshold:
                    bits.append(0)
#                else:
#                    # Undefined bit, skip this segment
#                    continue

            
            print("Bit Sequence : ", bits)
            # Extract the starting sequence and the actual message bits
            if bits[3:6] == [0, 0, 0]:  # Adjusted to compare the correct portion
                # Flip the bits
                bits = [1 - bit for bit in bits]

            if bits[3:len(self.starting_sequence)] == self.starting_sequence[3:6]:
                actual_bits = bits[len(self.starting_sequence):self.transmittedBitsLen]
                print("Actual lenth : ", len(actual_bits))
                self.bit_sequences.append(tuple(actual_bits))

            # Skip the silence period and move to the next transition
            next_transition = transition_indices[transition_indices > start_index + self.silence_threshold]
            if len(next_transition) == 0:
                break  # No more valid transitions found
            start_index = next_transition[0]

        if len(self.bit_sequences) > 0:
            message = ""
            for bit_sequence in self.bit_sequences:
                # Split the bit sequence into 8-bit chunks and convert to characters
                for i in range(0, len(bit_sequence), 8):
                    byte = bit_sequence[i:i + 8]
                    if len(byte) == 8:
                        ascii_char = chr(int(''.join(map(str, byte)), 2))
                        message += ascii_char
            
            with open("decoded_message.txt", "a") as file:
                file.write(f"{message}\n")

            self.bit_sequences = []

    def work(self, input_items, output_items):
        with self.lock:
            self.data_buffer = np.concatenate((self.data_buffer, input_items[0]))

        if len(self.data_buffer) >= self.buffer_size:
            data_to_process = self.data_buffer[:self.buffer_size]
            self.data_buffer = self.data_buffer[self.buffer_size:]

            # Ensure any previous processing thread has finished before starting a new one
            if self.processing_thread is not None and self.processing_thread.is_alive():
                self.processing_thread.join()

            # Start a new processing thread
            self.processing_thread = threading.Thread(target=self.process_data, args=(data_to_process,))
            self.processing_thread.start()

        return len(input_items[0])