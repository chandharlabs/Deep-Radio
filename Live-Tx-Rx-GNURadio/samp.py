import numpy as np
import os

class TextTo2BitSymbols:
    def __init__(self, file_path):
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        # Read the file as binary
        with open(file_path, "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Convert bytes to bit array
        bit_array = np.unpackbits(byte_data)

        print("🔍 Bits (first 64):", bit_array[:64])

        # Pad to make length even
        if len(bit_array) % 2 != 0:
            bit_array = np.append(bit_array, 0)

        # Reshape to 2-bit groups and convert to integers (00=0, 01=1, 10=2, 11=3)
        symbols_2bit = bit_array.reshape(-1, 2)
        self.symbol_values = (symbols_2bit[:, 0] << 1) | symbols_2bit[:, 1]

        print("📦 2-bit symbols (first 64):", self.symbol_values[:64])
        print(f"📏 Total 2-bit symbols: {len(self.symbol_values)}")

    def get_symbols(self):
        return self.symbol_values

# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_file = "test.txt"  # 🔁 Replace with your actual text file path
    output_file = "symbols_output.txt"

    try:
        converter = TextTo2BitSymbols(input_file)
        symbols = converter.get_symbols()

        # Save to file
        np.savetxt(output_file, symbols, fmt='%d')
        print(f"✅ 2-bit symbols saved to '{output_file}'")
    except FileNotFoundError as e:
        print(e)

