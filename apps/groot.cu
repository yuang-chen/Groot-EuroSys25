#include <groot.h>

using namespace groot;

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    CsrMatrix<int, float, device_memory> A_csr;

    Config config = program_options(argc, argv);

    read_matrix_file(A_csr, config.input_file);
    std::cout << "First 32 column indices:" << std::endl;
    for (int i = 0; i < std::min(32, (int)A_csr.column_indices.size()); i++) {
        std::cout << A_csr.column_indices[i] << " ";
    }
    std::cout << std::endl;
    reorder_graph(config, A_csr);

    write_matrix_file(A_csr, config.output_file);

    std::cout << "First 32 column indices after reordering:" << std::endl;
    for (int i = 0; i < std::min(32, (int)A_csr.column_indices.size()); i++) {
        std::cout << A_csr.column_indices[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}