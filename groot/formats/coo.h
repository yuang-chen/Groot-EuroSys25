#pragma once
#include <thrust/sort.h>

// #include <groot/matrices/traits.h>

namespace groot {

template<typename IndexType, typename ValueType, typename MemorySpace>
class CooMatrix {
public:
    using index_type   = IndexType;
    using value_type   = ValueType;
    using memory_space = MemorySpace;
    using IndexVector  = VectorType<IndexType, MemorySpace>;
    using ValueVector  = VectorType<ValueType, MemorySpace>;

    IndexType num_rows;
    IndexType num_cols;
    IndexType num_entries;

    IndexVector row_indices;
    IndexVector column_indices;
    ValueVector values;

    // Default constructor
    CooMatrix() = default;

    // Constructor with dimensions and default value
    CooMatrix(IndexType nrow, IndexType ncol, IndexType nnz, ValueType):
        num_rows(nrow), num_cols(ncol), num_entries(nnz), row_indices(nnz), column_indices(nnz), values(nnz)
    {
    }

    // Resize the matrix
    void resize(IndexType nrow, IndexType ncol, IndexType nnz)
    {
        num_rows    = nrow;
        num_cols    = ncol;
        num_entries = nnz;
        row_indices.resize(num_entries);
        column_indices.resize(num_entries);
        values.resize(num_entries);
    }

    void sort_columns_per_row()
    {
        thrust::sort_by_key(column_indices.begin(),
                            column_indices.end(),
                            thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), values.begin())));
        thrust::stable_sort_by_key(
            row_indices.begin(),
            row_indices.end(),
            thrust::make_zip_iterator(thrust::make_tuple(column_indices.begin(), values.begin())));
    }
};

}  // namespace groot