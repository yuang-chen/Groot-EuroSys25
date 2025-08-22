# Groot
Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplication on Tensor Cores

## Prerequisites
- CMake 3.28
- CUDA Toolkit 12.3
- KGraph (https://github.com/aaalgo/kgraph) 
- Boost 1.83 (required by KGraph)

It is advisable to use vcpkg for managing C++ libraries, such as Boost. However, KGraph requires manual installation.


## Build using CMake
Note: You may need to modify the CMakeLists.txt file to match your GPU architecture and library paths.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ..
```

## Dataset
The supported format is `mtx` and binary `csr`. 

`csr` is encoded as `nrow nnz row_ptr[] col_idx[]` in binary.

The `toydata` directory contains `cora.csr` and `cora_groot.csr`: the `Cora` dataset in CSR format, before and after reordering, respectively.

Loading the datasets to SpMM/SDDMM on Tensor Cores (TC) with different tile sizes, we can observe the impact of Groot reordering on the number of tiles generated:


| Tile Size | Before Reordering | After Reordering |
|-----------|-------------------|------------------|
| 16x16     | 718               | 437              |
| 8x8       | 1452              | 904              |
| 16x8      | 1360              | 791              |


Due to the inherent randomness in kNN, each reordering might result in a different number of tiles.

## Running the example

```bash
./build/apps/groot -i ./toydata/cora.csr -o cora_groot.csr
```

## Performance Tuning
For optimal K-Nearest Neighbors (KNN) performance, consider adjusting the parameters of KGraph. Detailed guidance can be found in the [WEAVESS documentation](https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters).

```c++
// Example configuration in build_KNN(), located in knn.h
unsigned i_k = std::min<unsigned>(nrow - 1, 20);   
unsigned i_l = std::min<unsigned>(i_k + 50, 30);   
unsigned s_k = std::min<unsigned>(nrow - 1, 25); 
```

## Citation
If you use Groot in your research, please cite our paper:
```
@inproceedings{chen2025groot,
  title={Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplications on Tensor Cores},
  author={Chen, Y. and Xie, J. and Teng, S. and Zeng, W. and Yu, J. X.},
  booktitle={Proceedings of the Twentieth European Conference on Computer Systems},
  pages={803-817},
  year={2025},
  month={March}
}
```