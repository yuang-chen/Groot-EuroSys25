cmake_minimum_required(VERSION 3.22)
project(groot_spmm)

set(PROJECT_VERSION "0.1")

enable_language(CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP)
add_compile_definitions(KGRAPH)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)

set(KGRAPH_ROOT "/home/ychen/Software/kgraph")
set(BOOST_ROOT "/home/ychen/Software/vcpkg/installed/x64-linux/")

include_directories(${KGRAPH_ROOT}/include)
include_directories(${BOOST_ROOT}/include)

link_directories(${KGRAPH_ROOT}/lib)
link_directories(${BOOST_ROOT}/lib)

add_subdirectory(groot)
add_subdirectory(apps)
