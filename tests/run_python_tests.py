# tests/run_python_tests.py
import numpy as np
import os

DATA_DIR = "./data"
OUTPUT_DIR = "./data/python_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Define Test Cases (MUST match generate_test_data.py) ---
test_cases = [
    ("case1_float_small", 5, 4, 6, np.float32),
    ("case2_int_medium", 10, 8, 12, np.int32),
    ("case3_double_large", 20, 15, 25, np.float64),
    ("case4_float_edge_small_dims", 1, 1, 5, np.float32),
    ("case5_int_edge_unit_dim1", 3, 1, 4, np.int32),
    ("case6_double_edge_unit_dim2", 2, 5, 1, np.float64),
    ("case7_int_edge_large_dim0", 50, 3, 2, np.int32),
    ("case8_float_mixed_types", 6, 7, 8, np.float32),
]

def load_flat_data(filename, shape, dtype):
    """Load flat data from txt and reshape."""
    try:
        flat_data = np.loadtxt(filename, dtype=dtype)
        return flat_data.reshape(shape)
    except ValueError as e:
        print(f"Error reshaping data from {filename} to {shape}: {e}")
        raise

def run_single_test_case(name, dim0, dim1, dim2, dtype):
    """Runs all defined slicing operations for a single test case."""
    print(f"  - Running Python tests for {name}...")
    data_filename = os.path.join(DATA_DIR, f"{name}_data.txt")
    try:
        data_3d = load_flat_data(data_filename, (dim0, dim1, dim2), dtype)
    except Exception as e:
        print(f"    ERROR: Failed to load data for {name}: {e}")
        return

    # --- Define All Test Operations ---
    # Each operation is a tuple: (description, sliced_data, output_filename_suffix)
    test_operations = [
        # Basic and Common Patterns
        ("[:, :, 2:]", data_3d[:, :, 2:], "dim2_from2"),
        ("[:, :, -3:]", data_3d[:, :, -3:], "dim2_last3"),
        ("[1:4, :, :]", data_3d[1:4, :, :], "dim0_1to4"),
        ("[1:3, 1:3, 1:4]", data_3d[1:3, 1:3, 1:4], "multidim_13_13_14"),
        ("[2:2, :, :]", data_3d[2:2, :, :], "empty"), # Empty slice
        ("[:, :, :]", data_3d[:, :, :], "full"),
        ("[:, :, 1:4]", data_3d[:, :, 1:4], "dim2_range_1to4"),
        
        # Comprehensive Patterns
        ("[3:1, :, :]", data_3d[3:1, :, :], "start_gt_stop_empty"), # start > stop
        ("[1000:1001, :, :]", data_3d[1000:1001, :, :], "large_start_clipped"), # Large index
        ("[:, 2:6, :]", data_3d[:, 2:6, :], "dim1_2to6"),
        ("[1:5, :, :]", data_3d[1:5, :, :], "dim0_1to5"), # Another dim0 range
        ("[0:min(3,dim0), 0:min(2,dim1), 0:min(3,dim2)]", data_3d[:min(3,dim0), :min(2,dim1), :min(3,dim2)], "all_dims_head"),
        ("[-1:, :, -1:]", data_3d[-1:, :, -1:], "negative_indices_last"), # Last elements
        ("[:, 1:, :]", data_3d[:, 1:, :], "stop_eq_dim"), # stop == dim

        # Negative Indices (Critical ones)
        ("[-2:-1, -3:-1, -4:-2]", data_3d[-2:-1, -3:-1, -4:-2], "negative_indices"), # The previously failing one
        ("[-2:, -1:, -3:]", data_3d[-2:, -1:, -3:], "negative_indices_tail"),

        # Complex / Specific
        ("[1:2, 1:2, 1:2]", data_3d[1:2, 1:2, 1:2], "single_element"),
        # Complex Tail Slice
        (
            "[max(0,dim0-3):, max(0,dim1-2):, max(0,dim2-4):]",
            data_3d[max(0,dim0-3):, max(0,dim1-2):, max(0,dim2-4):],
            "complex_multidim_tail"
        ),
    ]

    # --- Execute and Save Results ---
    for desc, sliced_data, suffix in test_operations:
        out_filename = os.path.join(OUTPUT_DIR, f"{name}_py_{suffix}.txt")
        try:
            fmt_str = '%.8f' if np.issubdtype(dtype, np.floating) else '%d'
            np.savetxt(out_filename, sliced_data.flatten(), fmt=fmt_str)
            # print(f"    Saved {desc} to {out_filename}") # Optional verbose output
        except Exception as e:
            print(f"    ERROR saving {desc} for {name}: {e}")


def run_python_tests():
    print("Step 2: Running Python slicing tests...")
    for name, dim0, dim1, dim2, dtype in test_cases:
        run_single_test_case(name, dim0, dim1, dim2, dtype)
    print("Step 2 (Python) completed.\n")

if __name__ == "__main__":
    run_python_tests()