# tests/run_python_tests.py
import numpy as np
import os

DATA_DIR = "./data"
OUTPUT_DIR = "./data/python_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define test cases matching C++
test_cases = [
    ("case1_float_small", 5, 4, 6, np.float32),
    ("case2_int_medium", 10, 8, 12, np.int32),
    ("case3_double_large", 20, 15, 25, np.float64),
]

def load_flat_data(filename, shape, dtype):
    """Load flat data from txt and reshape."""
    flat_data = np.loadtxt(filename, dtype=dtype)
    return flat_data.reshape(shape)

# --- Step 2: Run Python slicing tests ---
def run_python_tests():
    print("Step 2: Running Python slicing tests...")
    for name, dim0, dim1, dim2, dtype in test_cases:
        print(f"  - Processing {name}...")
        data_filename = os.path.join(DATA_DIR, f"{name}_data.txt")
        original_data_3d = load_flat_data(data_filename, (dim0, dim1, dim2), dtype)

        # --- Define slicing operations to test (matching C++) ---
        # 1. Slice last dim from index (float and double)
        if name in ["case1_float_small", "case3_double_large"]:
            result = original_data_3d[:, :, 2:]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_dim2_from2.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')

        # 2. Slice last dim last N (float and double)
        if name in ["case1_float_small", "case3_double_large"]:
            result = original_data_3d[:, :, -3:]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_dim2_last3.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')

        # 3. Slice first dim (float and double)
        if name in ["case1_float_small", "case3_double_large"]:
            result = original_data_3d[1:4, :, :]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_dim0_1to4.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')

        # 4. Multi-dim slice (float and double)
        if name in ["case1_float_small", "case3_double_large"]:
            result = original_data_3d[1:3, 1:3, 1:4]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_multidim_13_13_14.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')

        # 5. Edge case: Empty slice (float and double)
        if name in ["case1_float_small", "case3_double_large"]:
            result = original_data_3d[2:2, :, :] # Empty along dim0
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_empty.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')

        # 6. Edge case: Full slice (float and double)
        if name in ["case1_float_small", "case3_double_large"]:
            result = original_data_3d[:, :, :]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_full.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%.10f' if np.issubdtype(dtype, np.floating) else '%d')


        # --- Integer specific tests ---
        if name == "case2_int_medium":
            # 1. Slice last dim from index
            result = original_data_3d[:, :, 1:]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_dim2_from1.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%d')

            # 2. Slice middle dim
            result = original_data_3d[:, 2:6, :]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_dim1_2to6.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%d')

            # 3. Slice last dim range
            result = original_data_3d[:, :, 3:5]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_dim2_3to5.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%d')

            # 4. Multi-dim slice
            result = original_data_3d[0:2, 1:4, 2:8]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_multidim_02_14_28.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%d')

            # 5. Edge case: Single element
            result = original_data_3d[1:2, 1:2, 1:2]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_single_element.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%d')

            # 6. Edge case: Negative indices
            result = original_data_3d[-2:-1, -3:-1, -4:-2]
            out_filename = os.path.join(OUTPUT_DIR, f"{name}_negative_indices.txt")
            np.savetxt(out_filename, result.flatten(), fmt='%d')


    print("Step 2 (Python) completed.\n")

if __name__ == "__main__":
    run_python_tests()