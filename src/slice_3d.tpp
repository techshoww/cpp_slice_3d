// src/slice_3d.tpp
#ifndef SLICE_3D_TPP
#define SLICE_3D_TPP

#include "slice_3d.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

// --- Helper for index normalization ---
template <typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, size_t>::type
normalize_slice_index(T index, size_t dim_size) {
    if (index < 0) {
        return static_cast<size_t>(std::max(static_cast<T>(0), static_cast<T>(dim_size) + index));
    } else {
        return static_cast<size_t>(std::min(static_cast<size_t>(index), dim_size));
    }
}

// --- Main optimized slicing function ---
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_optimized(const std::vector<T>& data_1d,
                   size_t dim0, size_t dim1, size_t dim2,
                   int start0, int stop0,
                   int start1, int stop1,
                   int start2, int stop2) {

    const size_t total_elements = dim0 * dim1 * dim2;
    if (data_1d.size() != total_elements) {
        throw std::invalid_argument("Data size does not match provided dimensions.");
    }

    const size_t norm_start0 = normalize_slice_index(start0, dim0);
    const size_t norm_stop0 = (stop0 == -1) ? dim0 : normalize_slice_index(stop0, dim0);
    const size_t norm_start1 = normalize_slice_index(start1, dim1);
    const size_t norm_stop1 = (stop1 == -1) ? dim1 : normalize_slice_index(stop1, dim1);
    const size_t norm_start2 = normalize_slice_index(start2, dim2);
    const size_t norm_stop2 = (stop2 == -1) ? dim2 : normalize_slice_index(stop2, dim2);

    const size_t final_start0 = std::min(norm_start0, norm_stop0);
    const size_t final_stop0 = norm_stop0;
    const size_t final_start1 = std::min(norm_start1, norm_stop1);
    const size_t final_stop1 = norm_stop1;
    const size_t final_start2 = std::min(norm_start2, norm_stop2);
    const size_t final_stop2 = norm_stop2;

    const size_t slice_len0 = final_stop0 - final_start0;
    const size_t slice_len1 = final_stop1 - final_start1;
    const size_t slice_len2 = final_stop2 - final_start2;

    if (slice_len0 == 0 || slice_len1 == 0 || slice_len2 == 0) {
        return std::vector<T>();
    }

    const size_t result_size = slice_len0 * slice_len1 * slice_len2;
    std::vector<T> result_1d(result_size);

    const size_t stride_dim2 = 1;
    const size_t stride_dim1 = dim2 * stride_dim2;
    const size_t stride_dim0 = dim1 * stride_dim1;

    // --- Optimization Strategy ---

    // 1. Slice only in the last dimension (dim2)
    if (slice_len0 == dim0 && slice_len1 == dim1) {
        const size_t elements_to_copy_per_row = slice_len2;
        size_t result_idx = 0;
        for (size_t n = 0; n < dim0; ++n) {
            for (size_t c = 0; c < dim1; ++c) {
                const size_t src_start_idx = n * stride_dim0 + c * stride_dim1 + final_start2;
                std::copy(data_1d.data() + src_start_idx,
                          data_1d.data() + src_start_idx + elements_to_copy_per_row,
                          result_1d.data() + result_idx);
                result_idx += elements_to_copy_per_row;
            }
        }
        return result_1d;
    }

    // 2. Slice only in the middle dimension (dim1) and full last dimension (dim2)
    if (slice_len0 == dim0 && slice_len2 == dim2) {
        if (final_start2 == 0 && final_stop2 == dim2) {
             const size_t elements_to_copy_per_n_block = slice_len1 * dim2;
             size_t result_idx = 0;
             for (size_t n = 0; n < dim0; ++n) {
                 const size_t src_start_idx = n * stride_dim0 + final_start1 * stride_dim1;
                 std::copy(data_1d.data() + src_start_idx,
                           data_1d.data() + src_start_idx + elements_to_copy_per_n_block,
                           result_1d.data() + result_idx);
                 result_idx += elements_to_copy_per_n_block;
             }
             return result_1d;
        }
    }

    // 3. Slice only in the first dimension (dim0) and full middle/last dimensions
    if (slice_len1 == dim1 && slice_len2 == dim2) {
        if (final_start1 == 0 && final_stop1 == dim1 &&
            final_start2 == 0 && final_stop2 == dim2) {
             const size_t elements_to_copy = slice_len0 * dim1 * dim2;
             const size_t src_start_idx = final_start0 * stride_dim0;
             std::copy(data_1d.data() + src_start_idx,
                       data_1d.data() + src_start_idx + elements_to_copy,
                       result_1d.data());
             return result_1d;
        }
    }

    // --- General Case (Fallback) ---
    size_t result_idx = 0;
    for (size_t n = final_start0; n < final_stop0; ++n) {
        const size_t base_offset_n = n * stride_dim0;
        for (size_t c = final_start1; c < final_stop1; ++c) {
            const size_t base_offset_nc = base_offset_n + c * stride_dim1;
            for (size_t l = final_start2; l < final_stop2; ++l) {
                const size_t original_index = base_offset_nc + l * stride_dim2;
                result_1d[result_idx++] = data_1d[original_index];
            }
        }
    }

    return result_1d;
}

// --- Convenience Wrappers ---
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_from(const std::vector<T>& data_1d,
                       size_t dim0, size_t dim1, size_t dim2,
                       int start2) {
    return slice_3d_optimized(data_1d, dim0, dim1, dim2,
                              0, -1, 0, -1, start2, -1);
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_last_n(const std::vector<T>& data_1d,
                         size_t dim0, size_t dim1, size_t dim2,
                         size_t n) {
    int start2 = (n >= dim2) ? 0 : static_cast<int>(dim2 - n);
    return slice_3d_last_dim_from(data_1d, dim0, dim1, dim2, start2);
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::vector<T>>::type
slice_3d_last_dim_range(const std::vector<T>& data_1d,
                        size_t dim0, size_t dim1, size_t dim2,
                        int start2, int stop2) {
    return slice_3d_optimized(data_1d, dim0, dim1, dim2,
                              0, -1, 0, -1, start2, stop2);
}

// --- Utility to save vector to file ---
template<typename T>
void save_vector_to_file(const std::vector<T>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    file << std::fixed << std::setprecision(10); // Set precision for floats
    for (size_t i = 0; i < vec.size(); ++i) {
        file << vec[i];
        if (i < vec.size() - 1) {
            file << "\n"; // Newline separated
        }
    }
    file.close();
}

// --- Utility to load vector from file ---
template<typename T>
std::vector<T> load_vector_from_file(const std::string& filename) {
     std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    std::vector<T> vec;
    std::string line;
    while(std::getline(file, line)) {
        if (!line.empty()) {
             std::stringstream ss(line);
             T value;
             ss >> value;
             vec.push_back(value);
        }
    }
    file.close();
    return vec;
}


#endif // SLICE_3D_TPP