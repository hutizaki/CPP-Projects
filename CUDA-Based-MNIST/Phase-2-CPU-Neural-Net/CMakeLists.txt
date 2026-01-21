cmake_minimum_required(VERSION 3.10)
project(MNIST_Phase2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Include the bryan-library headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../bryan-library)

# Add the bryan-library source files
add_library(bryan_library
    ../bryan-library/matrixMath.cpp
    ../bryan-library/machineLearning.cpp
    ../bryan-library/fileParsing.cpp
)

# Create the executable
add_executable(neuron MNIST_Phase2.cpp)

# Link the library
target_link_libraries(neuron bryan_library)
