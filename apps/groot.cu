#include <groot.h>

using namespace groot;

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    CsrMatrix<int, float, device_memory> A_csr;

    Config config = program_options(argc, argv);

    read_matrix_file(A_csr, config.input_file);

    reorder_graph(config, A_csr);

    write_matrix_file(A_csr, config.output_file);

    return 0;
}