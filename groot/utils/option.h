
#pragma once
#include <getopt.h>
#include <string>

namespace groot {

enum class ReorderAlgo { None = 0, Groot = 1};

const char* reorder_algo_to_string(ReorderAlgo algo)
{
    switch (algo) {
        case ReorderAlgo::None:
            return "None";
        case ReorderAlgo::Groot:
            return "Groot";
        default:
            return "Unknown";
    }
}

struct Config {
    std::string input_file;
    std::string output_file;
    ReorderAlgo reorder         = ReorderAlgo::Groot;
};

std::string option_hints =
    "              [-i input_file]\n"
    "              [-o output_file]\n"
    "              [-r reorder_algorithm (0: none, 1: groot)]\n";

auto program_options(int argc, char* argv[])
{
    Config config;
    int    opt;
    if (argc == 1) {
        printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
        std::exit(EXIT_FAILURE);
    }
    while ((opt = getopt(argc, argv, "e:r:i:c:o:s:b:v:")) != -1) {
        switch (opt) {
            case 'i':
                config.input_file = optarg;
                break;
            case 'o':
                config.output_file = optarg;
                break;
            case 'r':
                config.reorder = static_cast<ReorderAlgo>(std::stoi(optarg));
                break;
            default:
                printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
                exit(EXIT_FAILURE);
        }
    }

    printf("--------experimental setting--------\n");
    if (!config.input_file.empty()) {
        printf("input path: %s\n", config.input_file.c_str());
    }
    if (config.reorder != ReorderAlgo::None) {
        printf("reorder algorithm: %s\n", reorder_algo_to_string(config.reorder));
    }
    if (!config.output_file.empty()) {
        printf("output path: %s\n", config.output_file.c_str());
    }

    return config;
}
}  // namespace groot