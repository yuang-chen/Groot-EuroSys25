#pragma once

namespace groot {

// Functor for filling row_indices from csr_input.row_pointers
template<typename IndexType>
struct FillRowIndices {
    const IndexType* row_pointers;
    IndexType*       row_indices;

    explicit FillRowIndices(const IndexType* _row_pointers, IndexType* _row_indices):
        row_pointers(_row_pointers), row_indices(_row_indices)
    {
    }

    __host__ __device__ void operator()(const IndexType row) const
    {
        for (IndexType i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
            row_indices[i] = row;
        }
    }
};

template<typename IndexType, typename ValueType>
struct IsSelfLoop {
    __host__ __device__ bool operator()(const thrust::tuple<IndexType, IndexType, ValueType>& edge)
    {
        return thrust::get<0>(edge) == thrust::get<1>(edge);
    }
};



}  // namespace groot