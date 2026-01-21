cmake_minimum_required(VERSION 3.18)
project(Phase5_Optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
endif()

# Include the bryan-library headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../bryan-library)

# Add the CPU library source files
add_library(bryan_library_cpu STATIC
    ../bryan-library/matrixMath.cpp
    ../bryan-library/machineLearning.cpp
    ../bryan-library/fileParsing.cpp
)

# Add the CUDA library source files
add_library(bryan_library_cuda STATIC
    ../bryan-library/cudaMatrixMath.cu
    ../bryan-library/cudaMachineLearning.cu
)

# Set CUDA properties for the CUDA library
set_target_properties(bryan_library_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create the executable for weight generator (CPU only)
add_executable(generate_weights generate_initial_weights.cpp)
target_link_libraries(generate_weights 
    PRIVATE
    bryan_library_cpu
)

# Create the executable for Phase 5 CUDA training
add_executable(phase5_neuron MNIST_Phase5.cu)

# Set CUDA properties for the executable
set_target_properties(phase5_neuron PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link the libraries
target_link_libraries(phase5_neuron 
    PRIVATE
    bryan_library_cpu
    bryan_library_cuda
)

# Find CUDA and link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(phase5_neuron PRIVATE CUDA::cudart)

# Create the executable for Phase 5.5 CUDA training (using cuBLAS)
add_executable(phase5_5_neuron MNIST_Phase5_5.cu)

# Set CUDA properties for the executable
set_target_properties(phase5_5_neuron PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link the libraries (only CPU library needed, no custom CUDA library)
target_link_libraries(phase5_5_neuron 
    PRIVATE
    bryan_library_cpu
)

# Link CUDA runtime and cuBLAS
target_link_libraries(phase5_5_neuron PRIVATE 
    CUDA::cudart
    CUDA::cublas
)
