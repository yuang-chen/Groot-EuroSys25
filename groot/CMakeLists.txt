add_library(grootlib INTERFACE)

# target_include_directories(grootlib INTERFACE .)
# Use this instead
target_include_directories(grootlib
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(grootlib INTERFACE
  OpenMP::OpenMP_CXX
)

target_link_libraries(grootlib INTERFACE
  kgraph
  Boost::timer
  Boost::chrono
  Boost::system
  Boost::program_options)

target_compile_options(grootlib INTERFACE
  $<$<CONFIG:Debug>:-G -g>
  $<$<COMPILE_LANGUAGE:CUDA>:
  -Xcompiler=-fopenmp # OpenMP flags for CUDA compiler
  --extended-lambda
  --expt-relaxed-constexpr
  -w # Suppress warning -- though warning is usually helpful
  >
)
