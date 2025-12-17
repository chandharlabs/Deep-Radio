import numpy as np

# Load file as unsigned 8-bit integers
data = np.fromfile("test.bin", dtype=np.uint8)

# Print or process
print("Integer values from test.bin:")
x=(data[:100000])
nzv = x[x != 0]
print(nzv)
