#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace groot {

using uint8_t = unsigned char;
using groot64_t = unsigned long long int;
using half    = __half;
using half2   = __half2;

}  // namespace groot