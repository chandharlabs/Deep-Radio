import numpy as np

# Load file as unsigned 8-bit integers
data = np.fromfile("test.bin", dtype=np.uint8)

# Limit to first 100000 entries (optional)
x = data[:100000]

# Remove zero values
nzv = x[x != 0]
print("📦 Non-zero values from test.bin:\n", nzv)

# Define the known pattern to search
pattern = np.array([1, 2, 3, 2, 2, 3, 3, 1, 1, 1, 2, 2, 2, 2, 3, 3], dtype=np.uint8)
pattern_len = len(pattern)

# Find pattern matches
def find_repeats(data, pattern):
    matches = []
    for i in range(len(data) - pattern_len + 1):
        if np.array_equal(data[i:i+pattern_len], pattern):
            matches.append(i)
    return matches

# Find and print matches
match_indices = find_repeats(nzv, pattern)
print(f"\n🔁 Pattern found {len(match_indices)} times at indices: {match_indices}")

# Optionally print the full matched regions
if match_indices:
    repeating_regions = [nzv[i:i+pattern_len] for i in match_indices]
    print("\n🧩 First matched region:", repeating_regions[0])
