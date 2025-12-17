import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("ber_log.csv")

# Set threshold
threshold = .01

# Option 2: Replace values > threshold with NaN and compute average (ignores NaNs)
df["BER"] = df["BER"].apply(lambda x: x if x <= threshold else np.nan)
average_masked = df["BER"].mean()

# Output both

print("Average with NaNs for >", threshold, ":", average_masked)

