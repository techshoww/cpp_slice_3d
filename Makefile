# Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -g
INCLUDES = -Isrc

# Use -lstdc++fs if filesystem is not part of standard library (older compilers)
# LIBS = -lstdc++fs

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
DATA_DIR = data

# Source files for the library (if compiled separately)
SLICE_SRC = $(SRC_DIR)/slice_3d.cpp # Can be empty if .tpp is included
SLICE_OBJ = $(BUILD_DIR)/slice_3d.o

# Test source
CPP_TEST_SRC = $(TEST_DIR)/run_cpp_tests.cpp
CPP_TEST_OBJ = $(BUILD_DIR)/run_cpp_tests.o
CPP_TEST_EXEC = $(BUILD_DIR)/run_cpp_tests

# Python scripts
PY_GEN_SCRIPT = $(TEST_DIR)/generate_test_data.py
PY_RUN_SCRIPT = $(TEST_DIR)/run_python_tests.py
PY_CMP_SCRIPT = $(TEST_DIR)/compare_results.py

# Ensure build and data directories exist
$(BUILD_DIR) $(DATA_DIR):
	mkdir -p $@

# Compile the slice library object (optional, if needed)
$(SLICE_OBJ): $(SLICE_SRC) $(SRC_DIR)/slice_3d.h $(SRC_DIR)/slice_3d.tpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile the C++ test runner
$(CPP_TEST_OBJ): $(CPP_TEST_SRC) $(SRC_DIR)/slice_3d.h $(SRC_DIR)/slice_3d.tpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link the C++ test runner
$(CPP_TEST_EXEC): $(CPP_TEST_OBJ) # $(SLICE_OBJ) # Uncomment if linking .o
	$(CXX) $(CXXFLAGS) $^ -o $@ # $(LIBS) # Add LIBS if needed

# --- Main Test Target ---
.PHONY: test
test: $(CPP_TEST_EXEC)
	@echo "=== Starting Full Test Suite ==="
	# Step 1: Generate data with Python
	python3 $(PY_GEN_SCRIPT)
	# Step 2: Run C++ tests
	./$(CPP_TEST_EXEC)
	# Step 2: Run Python tests
	python3 $(PY_RUN_SCRIPT)
	# Step 3: Compare results
	python3 $(PY_CMP_SCRIPT)
	@echo "=== Test Suite Completed ==="

# Convenience target for just running C++ part (useful for debugging C++)
.PHONY: cpp_test_only
cpp_test_only: $(CPP_TEST_EXEC)
	./$(CPP_TEST_EXEC)

# Clean build artifacts and generated data
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(DATA_DIR)

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make test     : Run the full test suite (generate data, run C++/Py tests, compare)"
	@echo "  make cpp_test_only : Compile and run only the C++ test part"
	@echo "  make clean    : Remove build artifacts and generated test data"