import numpy as np
from collections import Counter

# Define file path
file_path = "output5.dat"

# Read complex data from file
data = np.fromfile(file_path, dtype=np.complex64)

# Extract real part
real_part = np.real(data)

# Define thresholds
transmittedBitsLen = 8
transition_threshold = 0.5  # Threshold to detect transitions
bit_validation_threshold = 0.7  # Threshold to determine valid bits
silence_threshold = 20000  # Number of consecutive zeros to identify silence
bit_length = 1300  # Approximate number of samples per bit

# Expected starting sequence (preamble)
starting_sequence = [0, 1, 0]  # First few bits of transmitted sequence

# Find all transition indices where signal crosses ±0.5
transition_indices = np.where((real_part[:-1] < transition_threshold) & (real_part[1:] > transition_threshold) |
                              (real_part[:-1] > -transition_threshold) & (real_part[1:] < -transition_threshold))[0]

# Print first detected transition
if len(transition_indices) > 0:
    print("First transition index:", transition_indices[0])
else:
    print("No valid transitions detected.")
    exit()

# Process detected bit sequences
bit_sequences = []
start_index = transition_indices[0]

while start_index < len(real_part):
    # Extract a window of data after the transition
    segment = real_part[start_index:start_index + (transmittedBitsLen+1)*1300]

    # Convert to binary values using thresholding
    arr_processed = np.where(segment > transition_threshold, 1, np.where(segment < -transition_threshold, 0, segment))

    # Check for long silence (more than silence_threshold zeros)
    if np.mean(arr_processed[:silence_threshold]) < 0.1:
        print("\n--- Silence detected. Moving to next sequence. ---\n")
        next_transition = transition_indices[transition_indices > start_index + silence_threshold]
        if len(next_transition) == 0:
            break  # No more valid transitions found
        start_index = next_transition[0]
        continue  # Move to the next transition sequence

    # Extract bits based on transition windows
    bits = []
    i = 0
    while i < len(arr_processed) - bit_length:
        bit_segment = arr_processed[i:i + bit_length]

        # Validate bit based on the average value of the segment
        avg_value = np.mean(bit_segment)
        if avg_value > bit_validation_threshold:
            bits.append(1)
        elif avg_value < (1 - bit_validation_threshold):
            bits.append(0)
        else:
            # Skip undefined bit regions (due to noise or overlap)
            i += bit_length
            continue

        i += bit_length  # Move to the next bit

    # Check if the sequence is flipped
    if bits[:len(starting_sequence)] != starting_sequence:
        print("Flipped sequence detected! Inverting bits...")
        bits = [1 - b for b in bits]  # Flip all bits

    # Store extracted bits
    if bits:
        bit_sequences.append(tuple(bits))  # Convert list to tuple for hashing
        print("Detected bit sequence:", bits)

    # Move to the next transition
    next_transition = transition_indices[transition_indices > start_index + len(segment)]
    if len(next_transition) == 0:
        break  # No more valid transitions found
    start_index = next_transition[0]

# Find the most common sequence
if bit_sequences:
    most_common_sequence, count = Counter(bit_sequences).most_common(1)[0]
    final_sequence = list(most_common_sequence)  # Convert back to list
else:
    final_sequence = []

# Print all extracted sequences
print("\nFinal Extracted Bit Sequences:")
for seq in bit_sequences:
    print(list(seq))

# Print the final most common sequence
print("\nMost Common Sequence (Final Output):", final_sequence)

# # Perform differential decoding
# decoded_sequence = [final_sequence[0]]  # Initialize with the first bit
# for i in range(1, len(final_sequence)):
#     decoded_bit = final_sequence[i] ^ decoded_sequence[i-1]  # XOR operation
#     decoded_sequence.append(decoded_bit)
    
# print("differential decoded sequence : ", decoded_sequence)

# Convert the decoded sequence to ASCII character
ascii_char = chr(int(''.join(map(str, final_sequence)), 2))

# Print the final ASCII character
print("\nFinal ASCII Character:", ascii_char)