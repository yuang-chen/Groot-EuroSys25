#pragma once
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <type_traits>
#include <vector>

#include <thrust/adjacent_difference.h>

namespace groot {
// Custom exclusive scan function since C++ STL does not provide one out of the
// box
template<typename T>
void exclusive_scan(T* input, int length)
{
    if (length == 0 || length == 1)
        return;

    T old_val, new_val;

    old_val  = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++) {
        new_val  = input[i];
        input[i] = old_val + input[i - 1];
        old_val  = new_val;
    }
}

template<class CsrMatrix>
int read_from_mtx(CsrMatrix& matrix, std::string input)
{
    using IndexType = typename CsrMatrix::index_type;
    using ValueType = typename CsrMatrix::value_type;

    int       nrow, ncol;
    IndexType nnz_tmp;

    int         ret_code;
    MM_typecode matcode;
    FILE*       f;

    IndexType nnz_mtx_report;
    int       isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;
    // load matrix
    char* filename = const_cast<char*>(input.c_str());

    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode)) {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode)) {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode)) {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode)) {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    }
    else {
        // printf("input matrix is symmetric = false\n");
    }

    thrust::host_vector<IndexType> csrRowPtr_counter(nrow + 1);
    thrust::host_vector<IndexType> csrRowIdx_tmp(nnz_mtx_report);
    thrust::host_vector<IndexType> csrColIdx_tmp(nnz_mtx_report);
    thrust::host_vector<ValueType> csrVal_tmp(nnz_mtx_report);

    // int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    // int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    // ValueType *csrVal_tmp =
    //     (ValueType *)malloc(nnz_mtx_report * sizeof(ValueType));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (IndexType i = 0; i < nnz_mtx_report; i++) {
        int    idxi, idxj;
        double fval, fval_im;
        int    ival;

        if (isReal) {
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex) {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger) {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern) {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i]    = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp) {
        for (IndexType i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter.data(), nrow + 1);

    nnz_tmp = csrRowPtr_counter[nrow];

    thrust::host_vector<IndexType> csrRowPtr_alias = csrRowPtr_counter;
    thrust::host_vector<IndexType> csrColIdx_alias(nnz_tmp);
    thrust::host_vector<ValueType> csrVal_alias(nnz_tmp);

    std::fill(csrRowPtr_counter.begin(), csrRowPtr_counter.end(), 0);

    if (isSymmetric_tmp) {
        for (IndexType i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                IndexType offset        = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset                  = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else {
                IndexType offset        = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset]    = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else {
        for (IndexType i = 0; i < nnz_mtx_report; i++) {
            IndexType offset        = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset]    = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    matrix.num_rows    = nrow;
    matrix.num_cols    = ncol;
    matrix.num_entries = nnz_tmp;

    matrix.row_pointers   = csrRowPtr_alias;
    matrix.column_indices = csrColIdx_alias;
    matrix.values         = csrVal_alias;

    return 0;
}

bool inline string_end_with(std::string base, std::string postfix)
{
    // std::regex  string_end(".*" + postfix + "$");
    // std::smatch base_match;
    // return std::regex_match(base, base_match, string_end);
    return base.ends_with(postfix);
}

struct less_than_threshold {
    float threshold;

    less_than_threshold(float t): threshold(t) {}

    __host__ __device__ bool operator()(float x) const
    {
        return x < threshold;
    }
};

struct larger_than_threshold {
    float threshold;

    larger_than_threshold(float t): threshold(t) {}

    __host__ __device__ bool operator()(float x) const
    {
        return x >= threshold;
    }
};

template<class CsrMatrix>
void read_from_csr(CsrMatrix& matrix, const std::string& filename)
{
    using IndexType = typename CsrMatrix::index_type;
    std::ifstream csr_file;
    csr_file.open(filename, std::ios::binary);
    if (!csr_file.is_open()) {
        std::cout << "cannot open csr file!" << std::endl;
        std::exit(1);
    }
    IndexType nrow;
    IndexType nnz;
    csr_file.read(reinterpret_cast<char*>(&nrow), sizeof(IndexType));
    csr_file.read(reinterpret_cast<char*>(&nnz), sizeof(IndexType));

    thrust::host_vector<IndexType> row_ptr(nrow + 1);
    thrust::host_vector<IndexType> col_idx(nnz);

    csr_file.read(reinterpret_cast<char*>(row_ptr.data()), (nrow + 1) * sizeof(int));
    csr_file.read(reinterpret_cast<char*>(col_idx.data()), nnz * sizeof(IndexType));

    ASSERT(row_ptr[nrow] == nnz);

    csr_file.close();
    matrix.resize(nrow, nrow, nnz);
    matrix.row_pointers   = row_ptr;
    matrix.column_indices = col_idx;
    thrust::fill(matrix.values.begin(), matrix.values.end(), 1.0);
}

template<typename CsrMatrix>
bool read_from_edgelist(CsrMatrix& mat, const std::string& filename)
{
    std::ifstream input(filename);
    if (!input.is_open()) {
        std::cout << "Cannot open the input file!" << std::endl;
        return false;
    }

    // First pass: count number of rows and entries
    std::string                              line;
    int                                      max_row = -1;
    int                                      max_col = -1;
    thrust::host_vector<std::pair<int, int>> edges;

    int num_rows    = 0;
    int num_entries = 0;
    // First pass: read header and count nodes/edges
    while (std::getline(input, line)) {
        if (line.empty())
            continue;

        if (line[0] == '#') {
            // Check if this is the header line containing node/edge counts
            if (line.find("Nodes:") != std::string::npos && line.find("Edges:") != std::string::npos) {
                int                num_nodes, num_edges;
                std::istringstream iss(line.substr(2));  // Skip "# "
                std::string        dummy;
                if (iss >> dummy >> num_nodes >> dummy >> num_edges) {
                    edges.reserve(num_edges);
                }

                printf("num_nodes: %d, num_edges: %d\n", num_nodes, num_edges);

                num_rows    = num_nodes;
                num_entries = num_edges;
            }
            continue;
        }

        // Process edge line
        int                row, col;
        std::istringstream iss(line);
        if (!(iss >> row >> col)) {
            continue;  // Skip malformed lines
        }

        max_row = std::max(max_row, row);
        max_col = std::max(max_col, col);
        edges.push_back(std::make_pair(row, col));
    }
    printf("edges are paired\n");
    // int num_rows    = max_row + 1;
    // int num_entries = edges.size();

    // Initialize CSR arrays
    thrust::host_vector<int> row_ptr(num_rows + 1, 0);
    thrust::host_vector<int> col_idx(num_entries);

    printf("count entries per row\n");
    // Count entries per row
    for (const auto& edge : edges) {
        row_ptr[edge.first + 1]++;
    }

    printf("cumulative sum to get row pointers\n");
    // Cumulative sum to get row pointers
    for (int i = 1; i <= num_rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Second pass: fill column indices
    printf("fill column indices\n");
    thrust::host_vector<int> current_pos = row_ptr;  // Copy to track current position for each row
    for (const auto& edge : edges) {
        int row                     = edge.first;
        col_idx[current_pos[row]++] = edge.second;
    }

    // Sort column indices within each row
    // #pragma omp parallel for
    //     for (int i = 0; i < num_rows; i++) {
    //         std::sort(col_idx.begin() + row_ptr[i], col_idx.begin() + row_ptr[i + 1]);
    //     }

    // Update matrix properties
    mat.resize(num_rows, num_rows, num_entries);
    mat.row_pointers   = row_ptr;
    mat.column_indices = col_idx;

    return true;
}

template<class CsrMatrix>
void read_matrix_file(CsrMatrix& d_csr_A, std::string input)
{
    if (input.empty()) {
        printf("input file is NOT specified!\n");
        std::exit(1);
    }
    if (string_end_with(input, ".mtx")) {
        read_from_mtx(d_csr_A, input);
    }
    else if (string_end_with(input, ".csr")) {
        read_from_csr(d_csr_A, input);
    }
    else {
        printf("input file is NOT supported!\n");
        std::exit(1);
    }
}

}  // namespace groot