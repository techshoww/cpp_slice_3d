# tests/generate_test_data.py
import numpy as np
import os

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define test cases: (name, dim0, dim1, dim2, dtype)
test_cases = [
    ("case1_float_small", 5, 4, 6, np.float32),
    ("case2_int_medium", 10, 8, 12, np.int32),
    ("case3_double_large", 20, 15, 25, np.float64),
]

# --- Step 1: Generate and save random data ---
def generate_and_save_data():
    print("Step 1: Generating test data...")
    for name, dim0, dim1, dim2, dtype in test_cases:
        # Use a fixed seed for reproducibility within each case
        np.random.seed(sum(ord(c) for c in name)) 
        
        if np.issubdtype(dtype, np.integer):
            # For integers, use a reasonable range
            data = np.random.randint(-100, 100, size=(dim0, dim1, dim2), dtype=dtype)
        else: # For floats
            data = np.random.rand(dim0, dim1, dim2).astype(dtype)
            data = (data - 0.5) * 200 # Scale to -100 to 100 range

        # Save the 3D array as a flattened 1D array to a text file
        flat_data = data.flatten()
        filename = os.path.join(DATA_DIR, f"{name}_data.txt")
        np.savetxt(filename, flat_data, fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')
        print(f"  - Saved {name} data (shape {data.shape}, dtype {data.dtype}) to {filename}")

    print("Step 1 completed.\n")

if __name__ == "__main__":
    generate_and_save_data()