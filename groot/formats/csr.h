#pragma once
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

namespace groot {

template<typename IndexType, typename ValueType, typename MemorySpace>
class CsrMatrix {
public:
    using index_type   = IndexType;
    using value_type   = ValueType;
    using memory_space = MemorySpace;
    using IndexVector  = VectorType<IndexType, MemorySpace>;
    using ValueVector  = VectorType<ValueType, MemorySpace>;

    IndexType num_rows;
    IndexType num_cols;
    IndexType num_entries;

    IndexVector row_pointers;
    IndexVector column_indices;
    ValueVector values;

    // Default constructor
    CsrMatrix() = default;

    // Constructor with dimensions and default value
    CsrMatrix(IndexType nrow, IndexType ncol, IndexType nnz, ValueType default_value = ValueType()):
        num_rows(nrow),
        num_cols(ncol),
        num_entries(nnz),
        row_pointers(nrow + 1, 0),
        column_indices(nnz),
        values(nnz, default_value)
    {
    }

    // Resize the matrix
    void resize(IndexType nrow, IndexType ncol, IndexType nnz)
    {
        num_rows    = nrow;
        num_cols    = ncol;
        num_entries = nnz;
        row_pointers.resize(num_rows + 1);
        column_indices.resize(num_entries);
        values.resize(num_entries);
    }

    CsrMatrix& operator=(const CsrMatrix& other)
    {
        if (this != &other) {
            num_rows    = other.num_rows;
            num_cols    = other.num_cols;
            num_entries = other.num_entries;

            row_pointers   = other.row_pointers;
            column_indices = other.column_indices;
            values         = other.values;
        }
        return *this;
    }

    void free()
    {
        num_rows    = 0;
        num_cols    = 0;
        num_entries = 0;
        row_pointers.clear();
        row_pointers.shrink_to_fit();
        column_indices.clear();
        column_indices.shrink_to_fit();
        values.clear();
        values.shrink_to_fit();
    }
};

}  // namespace groot