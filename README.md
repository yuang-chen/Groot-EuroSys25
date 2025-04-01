# Groot
Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplication on Tensor Cores

## Prerequisites
- CMake (3.8 or higher)
- CUDA Toolkit
- KGraph (https://github.com/aaalgo/kgraph)
- Boost (required by KGraph)

## Build using CMake
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ..
```

## Data format
The supported format is `mtx` and binary `csr`. 

`csr` is encoded as `nrow nnz row_ptr[] col_idx[]` in binary.

## Running the example

```bash
./build/apps/groot -i /path/to/matrix/file -o /path/to/store/reordered/file
```