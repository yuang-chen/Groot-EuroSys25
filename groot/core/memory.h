#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/omp/execution_policy.h>

namespace groot {

// Memory space tags
struct host_memory {};
struct device_memory {};

template<typename T, typename MemorySpace>
struct VectorTrait;

template<typename T>
struct VectorTrait<T, host_memory> {
    using MemoryVector = thrust::host_vector<T>;
    static constexpr auto execution_policy() -> decltype(thrust::host)
    {
        return thrust::host;
    }
};

template<typename T>
struct VectorTrait<T, device_memory> {
    using MemoryVector = thrust::device_vector<T>;
    static constexpr auto execution_policy() -> decltype(thrust::device)
    {
        return thrust::device;
    }
};

template<typename T, typename MemorySpace>
using VectorType = typename VectorTrait<T, MemorySpace>::MemoryVector;

// Helper type trait to check if T is a thrust::device_vector
template<typename T>
struct is_device_vector: std::false_type {};

template<typename T, typename Alloc>
struct is_device_vector<thrust::device_vector<T, Alloc>>: std::true_type {};

// Function to get the execution policy based on vector type
template<typename Vector>
auto get_exec_policy()
{
    if constexpr (is_device_vector<Vector>::value) {
        return thrust::device;
    }
    else {
        return thrust::omp::par;
    }
}


}  // namespace groot