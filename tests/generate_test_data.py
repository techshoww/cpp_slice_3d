# tests/generate_test_data.py
import numpy as np
import os

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Define Test Cases ---
# Structure: (name, dim0, dim1, dim2, dtype)
# Aim for variety in size and type
test_cases = [
    # Original Cases
    ("case1_float_small", 5, 4, 6, np.float32),
    ("case2_int_medium", 10, 8, 12, np.int32),
    ("case3_double_large", 20, 15, 25, np.float64),
    # Edge Cases
    ("case4_float_edge_small_dims", 1, 1, 5, np.float32),  # Very small dims
    ("case5_int_edge_unit_dim1", 3, 1, 4, np.int32),       # Unit middle dim
    ("case6_double_edge_unit_dim2", 2, 5, 1, np.float64),  # Unit last dim
    ("case7_int_edge_large_dim0", 50, 3, 2, np.int32),     # Large first dim (reduced from 100 for speed)
    ("case8_float_mixed_types", 6, 7, 8, np.float32),      # Mixed, common-ish size
]

def generate_and_save_data():
    print("Step 1: Generating test data...")
    # Use a single RNG for consistency
    rng = np.random.default_rng(seed=12345) 

    for name, dim0, dim1, dim2, dtype in test_cases:
        print(f"  - Generating {name} (shape {dim0},{dim1},{dim2}, dtype {dtype})...")
        
        if np.issubdtype(dtype, np.integer):
            # For integers, use a reasonable range
            data = rng.integers(-50, 51, size=(dim0, dim1, dim2), dtype=dtype)
        else:  # For floats
            # Use random values in a range
            data = (rng.random((dim0, dim1, dim2), dtype=np.float64).astype(dtype) - 0.5) * 100

        # Save the 3D array as a flattened 1D array to a text file
        flat_data = data.flatten()
        filename = os.path.join(DATA_DIR, f"{name}_data.txt")
        # Use appropriate format string
        fmt_str = '%.8f' if np.issubdtype(dtype, np.floating) else '%d'
        np.savetxt(filename, flat_data, fmt=fmt_str)
        print(f"    Saved to {filename}")

    print("Step 1 completed.\n")

if __name__ == "__main__":
    generate_and_save_data()