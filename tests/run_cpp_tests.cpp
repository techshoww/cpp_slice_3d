// tests/run_cpp_tests.cpp
#include "../src/slice_3d.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm> // for std::min, std::max

namespace fs = std::filesystem;

const std::string DATA_DIR = "./data";
const std::string OUTPUT_DIR = "./data/cpp_outputs";

// --- Define Test Cases (MUST match Python) ---
struct TestCase {
    std::string name;
    size_t dim0, dim1, dim2;
    std::string dtype_str; // For identification, not used in logic
};

const std::vector<TestCase> test_cases = {
    {"case1_float_small", 5, 4, 6, "float32"},
    {"case2_int_medium", 10, 8, 12, "int32"},
    {"case3_double_large", 20, 15, 25, "float64"},
    {"case4_float_edge_small_dims", 1, 1, 5, "float32"},
    {"case5_int_edge_unit_dim1", 3, 1, 4, "int32"},
    {"case6_double_edge_unit_dim2", 2, 5, 1, "float64"},
    {"case7_int_edge_large_dim0", 50, 3, 2, "int32"},
    {"case8_float_mixed_types", 6, 7, 8, "float32"},
};

template<typename T>
void run_single_test_case(const TestCase& test_case) {
    std::cout << "  - Running C++ tests for " << test_case.name << "...\n";
    std::string data_filename = DATA_DIR + "/" + test_case.name + "_data.txt";
    
    try {
        auto data = load_vector_from_file<T>(data_filename);
        
        // --- Define All Test Operations (MUST match Python exactly) ---
        // The logic and file names must be identical to run_python_tests.py
        
        // Basic and Common Patterns
        {
            auto result = slice_3d_last_dim_from<T>(data, test_case.dim0, test_case.dim1, test_case.dim2, 2);
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_dim2_from2.txt");
        }
        {
            auto result = slice_3d_last_dim_last_n<T>(data, test_case.dim0, test_case.dim1, test_case.dim2, 3);
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_dim2_last3.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                1, 4, 0, static_cast<int>(test_case.dim1), 0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_dim0_1to4.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1, 3, 1, 3, 1, 4);
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_multidim_13_13_14.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                2, 2, 0, static_cast<int>(test_case.dim1), 0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_empty.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                0, static_cast<int>(test_case.dim0),
                                                0, static_cast<int>(test_case.dim1),
                                                0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_full.txt");
        }
        {
             auto result = slice_3d_last_dim_range<T>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1, 4);
             save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_dim2_range_1to4.txt");
        }

        // Comprehensive Patterns
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                3, 1, 0, static_cast<int>(test_case.dim1), 0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_start_gt_stop_empty.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                1000, 1001, 0, static_cast<int>(test_case.dim1), 0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_large_start_clipped.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                0, static_cast<int>(test_case.dim0), 2, 6, 0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_dim1_2to6.txt");
        }
        {
             auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                1, 5, 0, static_cast<int>(test_case.dim1), 0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_dim0_1to5.txt");
        }
        {
            int end0 = std::min(3, static_cast<int>(test_case.dim0));
            int end1 = std::min(2, static_cast<int>(test_case.dim1));
            int end2 = std::min(3, static_cast<int>(test_case.dim2));
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                0, end0, 0, end1, 0, end2);
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_all_dims_head.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                -1, static_cast<int>(test_case.dim0),
                                                0, static_cast<int>(test_case.dim1),
                                                -1, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_negative_indices_last.txt");
        }
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                0, static_cast<int>(test_case.dim0),
                                                1, static_cast<int>(test_case.dim1),
                                                0, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_stop_eq_dim.txt");
        }

        // Negative Indices (Critical ones)
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2, -2, -1, -3, -1, -4, -2);
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_negative_indices.txt");
        }
        {
             auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                -2, static_cast<int>(test_case.dim0),
                                                -1, static_cast<int>(test_case.dim1),
                                                -3, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_negative_indices_tail.txt");
        }

        // Complex / Specific
        {
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1, 2, 1, 2, 1, 2);
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_single_element.txt");
        }
        // Complex Tail Slice
        {
            int s0 = std::max(0, static_cast<int>(test_case.dim0) - 3);
            int s1 = std::max(0, static_cast<int>(test_case.dim1) - 2);
            int s2 = std::max(0, static_cast<int>(test_case.dim2) - 4);
            auto result = slice_3d_optimized<T>(data, test_case.dim0, test_case.dim1, test_case.dim2,
                                                s0, static_cast<int>(test_case.dim0),
                                                s1, static_cast<int>(test_case.dim1),
                                                s2, static_cast<int>(test_case.dim2));
            save_vector_to_file<T>(result, OUTPUT_DIR + "/" + test_case.name + "_py_complex_multidim_tail.txt");
        }

    } catch (const std::exception& e) {
        std::cerr << "    ERROR during C++ test for " << test_case.name << ": " << e.what() << "\n";
    }
}


int main() {
    std::cout << "Step 2: Running C++ slicing tests...\n";
    fs::create_directories(OUTPUT_DIR);

    for (const auto& test_case : test_cases) {
        if (test_case.name.find("float") != std::string::npos || test_case.name.find("double") != std::string::npos) {
            run_single_test_case<float>(test_case);
        } else if (test_case.name.find("int") != std::string::npos) {
            run_single_test_case<int>(test_case);
        } else {
            std::cerr << "  - WARNING: Unknown type for test case: " << test_case.name << "\n";
        }
    }

    std::cout << "Step 2 (C++) completed.\n\n";
    return 0;
}