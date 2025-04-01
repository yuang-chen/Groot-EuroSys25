#pragma once

namespace groot {

template<typename Vector>
void get_row_lengths_from_pointers(Vector& rowlen, const Vector& rowptr)
{
    using IndexType = typename Vector::value_type;
    thrust::transform(rowptr.begin() + 1, rowptr.end(), rowptr.begin(), rowlen.begin(), thrust::minus<IndexType>());
}


template<typename Vector>
void get_row_indices_from_pointers(const Vector& row_pointers, Vector& row_indices)
{
    using IndexType   = typename Vector::value_type;
    auto       policy = get_exec_policy<Vector>();
    const auto nrow   = row_pointers.size() - 1;

    thrust::for_each(policy,
                     thrust::counting_iterator<IndexType>(0),
                     thrust::counting_iterator<IndexType>(nrow),
                     FillRowIndices<IndexType>(thrust::raw_pointer_cast(row_pointers.data()),
                                               thrust::raw_pointer_cast(row_indices.data())));
}



template<typename IndexVector, typename ValueVector>
void sort_columns_per_row(IndexVector& row_indices, IndexVector& column_indices, ValueVector& values)
{
    // sort columns per row
    thrust::sort_by_key(column_indices.begin(),
                        column_indices.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), values.begin())));
    thrust::stable_sort_by_key(row_indices.begin(),
                               row_indices.end(),
                               thrust::make_zip_iterator(thrust::make_tuple(column_indices.begin(), values.begin())));
}


}  // namespace groot