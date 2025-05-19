
#pragma once

#include <omp.h>

namespace groot {

template<typename CsrMatrix, typename Vector>
void build_csr_cpu(CsrMatrix& mat, const Vector& new_id)
{
    using IndexType = typename CsrMatrix::index_type;
    using ValueType = typename CsrMatrix::value_type;
    ASSERT(mat.num_rows == new_id.size());

    thrust::host_vector<IndexType> rowptr = mat.row_pointers;
    thrust::host_vector<IndexType> colidx = mat.column_indices;
    thrust::host_vector<ValueType> values = mat.values;

    IndexType max_threads = omp_get_max_threads();

    thrust::host_vector<IndexType> new_degree(mat.num_rows, 0);
// Assign the outdegree to new id
#pragma omp parallel for schedule(static) num_threads(max_threads)
    for (IndexType i = 0; i < mat.num_rows; i++)
        new_degree[new_id[i]] = rowptr[i + 1] - rowptr[i];

    // Build new row_index array
    thrust::host_vector<IndexType> new_row(mat.num_rows + 1, 0);
    thrust::host_vector<IndexType> new_col(mat.num_entries, 0);
    thrust::host_vector<ValueType> new_val(mat.num_entries, 0);

    thrust::inclusive_scan(new_degree.begin(), new_degree.end(), new_row.begin() + 1);

    // Build new col_index array
#pragma omp parallel for schedule(static, 256) num_threads(max_threads)
    for (IndexType i = 0; i < mat.num_rows; i++) {
        IndexType count = 0;
        for (IndexType j = rowptr[i]; j < rowptr[i + 1]; j++) {
            new_col[new_row[new_id[i]] + count] = new_id[colidx[j]];
            new_val[new_row[new_id[i]] + count] = values[j];
            count++;
        }
    }
    thrust::copy(new_row.begin(), new_row.end(), mat.row_pointers.begin());
    thrust::copy(new_col.begin(), new_col.end(), mat.column_indices.begin());
    thrust::copy(new_val.begin(), new_val.end(), mat.values.begin());

}

template<typename CsrMatrix, typename Vector>
void build_csr_gpu(CsrMatrix& mat, const Vector& new_id)
{
    using IndexType = typename CsrMatrix::index_type;
    using ValueType = typename CsrMatrix::value_type;
    ASSERT(mat.num_rows == new_id.size());

    thrust::device_vector<IndexType> new_degree(mat.num_rows, 0);

    get_row_lengths_from_pointers(new_degree, mat.row_pointers);

    // Build new row_index array
    thrust::device_vector<IndexType> new_row(mat.num_rows + 1, 0);
    thrust::inclusive_scan(new_degree.begin(), new_degree.end(), new_row.begin() + 1);

    // Allocate memory for new column indices and values
    thrust::device_vector<IndexType> new_col(mat.num_entries);
    thrust::device_vector<ValueType> new_val(mat.num_entries);

    // Build new col_index array and values
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<IndexType>(0),
                     thrust::make_counting_iterator<IndexType>(mat.num_rows),
                     [row_ptr = thrust::raw_pointer_cast(mat.row_pointers.data()),
                      col_idx = thrust::raw_pointer_cast(mat.column_indices.data()),
                      values  = thrust::raw_pointer_cast(mat.values.data()),
                      new_row = thrust::raw_pointer_cast(new_row.data()),
                      new_col = thrust::raw_pointer_cast(new_col.data()),
                      new_val = thrust::raw_pointer_cast(new_val.data()),
                      new_id  = thrust::raw_pointer_cast(new_id.data())] __device__(IndexType i) {
                         IndexType start     = row_ptr[i];
                         IndexType end       = row_ptr[i + 1];
                         IndexType new_start = new_row[new_id[i]];
                         for (IndexType j = start; j < end; ++j) {
                             IndexType offset            = j - start;
                             new_col[new_start + offset] = new_id[col_idx[j]];
                             new_val[new_start + offset] = values[j];
                         }
                     });

    // Update the matrix
    mat.row_pointers   = std::move(new_row);
    mat.column_indices = std::move(new_col);
    mat.values         = std::move(new_val);
}

template<typename Config, typename CsrMatrix>
void reorder_graph(Config config, CsrMatrix& mat)
{
    if (config.reorder == ReorderAlgo::None) {
        return;
    }

    printf("\n\n----------------Reordering Graph----------------\n");
    thrust::device_vector<int> new_ids(mat.num_rows);

    // TODO: implement the knn_mst_dfs on GPU
    thrust::host_vector<int> new_ids_h(mat.num_rows);
    CPUTimer                 cpu_timer;
    cpu_timer.start();
    groot(mat, new_ids_h);  // on CPU
    cpu_timer.stop();
    printf("[KNN_MST_DFS] Reordering time (ms): %f \n", cpu_timer.elapsed());
    thrust::copy(new_ids_h.begin(), new_ids_h.end(), new_ids.begin());

    CUDATimer timer;
    timer.start();
    build_csr_gpu(mat, new_ids);
    timer.stop();
    printf("[Rebuilding] graph time (ms): %f \n", timer.elapsed());
}

}  // namespace groot