# cpp_slice_3d
对 1维 vector&lt;T> 数据做3d slice  
目的是用来复刻 python 中对 3d ndarray 的切片操作  
代码由 Qwen3 Coder 生成  

```
my_project/
├── src/
│   ├── slice_3d.h
│   ├── slice_3d.tpp
│   └── slice_3d.cpp (can be empty if using .tpp, or include .tpp)
├── tests/
│   ├── generate_test_data.py     # Step 1: Generate random data
│   ├── run_cpp_tests.cpp         # Step 2: Run C++ slicing and save results
│   ├── run_python_tests.py       # Step 2: Run Python slicing and save results
│   └── compare_results.py        # Step 3: Compare outputs
├── data/                         # Directory for test data (created by scripts)
└── Makefile
```

可以执行 `make test` 进行测试。  