# tests/compare_results.py
import numpy as np
import os
import sys

DATA_DIR = "./data"
CPP_OUTPUT_DIR = "./data/cpp_outputs"
PY_OUTPUT_DIR = "./data/python_outputs"

def load_and_compare(file1, file2, tolerance=1e-6):
    """Load two flat files and compare arrays."""
    try:
        # Load data
        # Assume float for simplicity in comparison, adjust dtype if needed
        # Or load without specifying dtype and let numpy infer, then compare
        # Let's load as float64 to handle both int and float comparisons reasonably
        arr1 = np.loadtxt(file1, dtype=np.float64)
        arr2 = np.loadtxt(file2, dtype=np.float64)
    except Exception as e:
        print(f"  ERROR loading files {file1} or {file2}: {e}")
        return False

    if arr1.shape != arr2.shape:
        print(f"  FAIL: Shape mismatch {arr1.shape} vs {arr2.shape}")
        return False

    # Use allclose for floating point comparison with tolerance
    # It also works for integers (tolerance will be effectively 0 for integer differences)
    if not np.allclose(arr1, arr2, atol=tolerance, rtol=tolerance):
        # Find the first few mismatched indices for debugging
        diff_indices = np.where(~np.isclose(arr1, arr2, atol=tolerance, rtol=tolerance))[0]
        print(f"  FAIL: Value mismatch found at {len(diff_indices)} indices.")
        for i in diff_indices[:5]: # Show first 5 mismatches
            print(f"    Index {i}: C++={arr1[i]}, Python={arr2[i]}")
        if len(diff_indices) > 5:
            print(f"    ... and {len(diff_indices) - 5} more.")
        return False
    return True

# --- Step 3: Compare results ---
def compare_outputs():
    print("Step 3: Comparing C++ and Python outputs...")
    all_passed = True

    # Get list of files from one directory
    try:
        cpp_files = set(os.listdir(CPP_OUTPUT_DIR))
        py_files = set(os.listdir(PY_OUTPUT_DIR))
    except FileNotFoundError as e:
        print(f"ERROR: Output directory not found: {e}")
        sys.exit(1)

    # Check if files match
    if cpp_files != py_files:
        print("FAIL: Set of output files differs between C++ and Python.")
        print(f"  C++ files not in Python: {cpp_files - py_files}")
        print(f"  Python files not in C++: {py_files - cpp_files}")
        return False

    print(f"Found {len(cpp_files)} output file pairs to compare.")

    for filename in cpp_files:
        cpp_file_path = os.path.join(CPP_OUTPUT_DIR, filename)
        py_file_path = os.path.join(PY_OUTPUT_DIR, filename)

        print(f"  - Comparing {filename}...")
        if load_and_compare(cpp_file_path, py_file_path):
            print(f"    PASS: {filename}")
        else:
            print(f"    FAIL: {filename}")
            all_passed = False

    if all_passed:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1) # Exit with error code if any test failed

    return all_passed

if __name__ == "__main__":
    compare_outputs()