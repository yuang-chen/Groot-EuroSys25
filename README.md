# Groot
Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplication on Tensor Cores

## Python Implementation (Update)

**NEW**: A Python implementation (`groot.py`) is now available, providing a more flexible solution

```bash
# Install dependencies
pip install pynndescent 

# Basic usage
python groot.py --dataset cora

# Complete Usage
python groot.py -h
usage: groot.py [-h] [--dataset DATASET] [--knn KNN] [--similarity_metric {jaccard,hamming}] [--no_mst] [--traversal {dfs,bfs}]
                [--start_node {max_degree,random,first}] [--cache_dir CACHE_DIR] [--no_cache] [--force_rebuild]
                [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
```

Key features: k-NN graph construction (Jaccard or Hamming) → k-NN + BFS or k-NN + MST + DFS for row ordering.

**Note**: DFS requires MST extraction since k-NN graphs have cycles; BFS works directly on k-NN graphs. Jaccard seems to work better than Hamming in the python version.

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