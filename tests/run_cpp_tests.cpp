// tests/run_cpp_tests.cpp
#include "../src/slice_3d.h" // This includes slice_3d.tpp, which has save_vector_to_file
#include <iostream>
#include <vector>
#include <string>
#include <filesystem> // C++17

namespace fs = std::filesystem;

const std::string DATA_DIR = "./data";
const std::string OUTPUT_DIR = "./data/cpp_outputs";

// Define the same test cases as in Python
struct TestCase {
    std::string name;
    size_t dim0, dim1, dim2;
    std::string dtype_str; // Not used in logic now, but kept for info
};

std::vector<TestCase> test_cases = {
    {"case1_float_small", 5, 4, 6, "float32"},
    {"case2_int_medium", 10, 8, 12, "int32"},
    {"case3_double_large", 20, 15, 25, "float64"}
};

// --- Step 2: Run C++ slicing tests ---
int main() {
    std::cout << "Step 2: Running C++ slicing tests...\n";
    fs::create_directories(OUTPUT_DIR);

    for (const auto& test_case : test_cases) {
        std::cout << "  - Processing " << test_case.name << "...\n";
        std::string data_filename = DATA_DIR + "/" + test_case.name + "_data.txt";

        // --- Load and process data based on case name ---
        if (test_case.name == "case1_float_small" || test_case.name == "case3_double_large") {
            // Assuming template functions handle float/double correctly.
            // For file loading, using float is generally fine.
            using DataType = float;
            auto data = load_vector_from_file<DataType>(data_filename); // Use the one from slice_3d.tpp

            // --- Define slicing operations matching Python for float/double cases ---
            // 1. Slice last dim from index
            {
                auto result = slice_3d_last_dim_from<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 2);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_dim2_from2.txt";
                save_vector_to_file<DataType>(result, out_filename); // Use the one from slice_3d.tpp
            }
            // 2. Slice last dim last N
            {
                auto result = slice_3d_last_dim_last_n<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 3);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_dim2_last3.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 3. Slice first dim
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1, 4, 0, -1, 0, -1);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_dim0_1to4.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 4. Multi-dim slice
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1, 3, 1, 3, 1, 4);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_multidim_13_13_14.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 5. Edge case: Empty slice
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 2, 2, 0, -1, 0, -1);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_empty.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 6. Edge case: Full slice
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 0, -1, 0, -1, 0, -1);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_full.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }

        } else if (test_case.name == "case2_int_medium") {
            using DataType = int;
            auto data = load_vector_from_file<DataType>(data_filename); // Use the one from slice_3d.tpp

            // --- Define slicing operations matching Python for int case ---
            // 1. Slice last dim from index
            {
                auto result = slice_3d_last_dim_from<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_dim2_from1.txt";
                save_vector_to_file<DataType>(result, out_filename); // Use the one from slice_3d.tpp
            }
            // 2. Slice middle dim
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 0, -1, 2, 6, 0, -1);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_dim1_2to6.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 3. Slice last dim range
            {
                auto result = slice_3d_last_dim_range<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 3, 5);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_dim2_3to5.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 4. Multi-dim slice
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 0, 2, 1, 4, 2, 8);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_multidim_02_14_28.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 5. Edge case: Single element
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, 1, 2, 1, 2, 1, 2);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_single_element.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }
            // 6. Edge case: Negative indices
            {
                auto result = slice_3d_optimized<DataType>(data, test_case.dim0, test_case.dim1, test_case.dim2, -2, -1, -3, -1, -4, -2);
                std::string out_filename = OUTPUT_DIR + "/" + test_case.name + "_negative_indices.txt";
                save_vector_to_file<DataType>(result, out_filename);
            }

        } else {
            std::cerr << "  - WARNING: Unsupported test case name for C++ test: " << test_case.name << "\n";
        }
    }

    std::cout << "Step 2 (C++) completed.\n\n";
    return 0;
}