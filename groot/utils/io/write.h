#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace groot {

template<typename CsrMatrix>
bool write_into_csr(const CsrMatrix& mat, std::string output)
{
    using IndexType                        = CsrMatrix::index_type;
    auto                           nrow    = mat.num_rows;
    auto                           nnz     = mat.num_entries;
    thrust::host_vector<IndexType> row_ptr = mat.row_pointers;
    thrust::host_vector<IndexType> col_idx = mat.column_indices;
    FILE*                          fp      = fopen(output.c_str(), "wb");
    if (fp == NULL) {
        fputs("file error", stderr);
        return false;
    }
    std::cout << "writing to " << output << std::endl;
    ASSERT(row_ptr[nrow] == nnz && "row_ptr[nrow] != nnz");

    fwrite(&nrow, sizeof(IndexType), 1, fp);
    fwrite(&nnz, sizeof(IndexType), 1, fp);
    fwrite(row_ptr.data(), sizeof(IndexType), nrow + 1, fp);
    fwrite(col_idx.data(), sizeof(IndexType), nnz, fp);

    fclose(fp);

    return true;
};

template<typename CsrMatrix>
bool write_into_mtx(const CsrMatrix& mat, std::string out)
{
    std::ofstream output(out);
    if (!output.is_open()) {
        std::cout << "cannot open the output file!" << std::endl;
        return false;
    }
    auto                     nrow    = mat.num_rows;
    auto                     nnz     = mat.num_entries;
    thrust::host_vector<int> row_ptr = mat.row_pointers;
    thrust::host_vector<int> col_idx = mat.column_indices;
    ASSERT(row_ptr[nrow] == nnz && "row_ptr[nrow] != nnz");

    output << "%%MatrixMarket matrix coordinate pattern general\n";
    output << nrow << " " << nrow << " " << nnz << '\n';
    for (unsigned i = 0; i < nrow; i++) {
        for (unsigned j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            output << i + 1 << " " << col_idx[j] + 1 << '\n';
        }
    }
    output.close();

    return true;
}

template<typename CsrMatrix>
void write_matrix_file(CsrMatrix& d_csr_A, std::string output)
{
    if (output.empty()) {
        return;  // nothing happens
    }
    else if (string_end_with(output, ".csr")) {
        std::cout << "converting to CSR format" << std::endl;
        write_into_csr(d_csr_A, output);
    }
    else if (string_end_with(output, ".mtx")) {
        std::cout << "converting to MTX format" << std::endl;
        write_into_mtx(d_csr_A, output);
    }
    else {
        printf("file format is not supported\n");
        std::exit(1);
    }
}

}  // namespace groot