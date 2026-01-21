#!/bin/bash

# Cross-platform build script for MNIST Phase 2

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
cmake --build . --config Release

# Go back to Phase-2 directory to run (for correct relative paths to data files)
cd ..

# Run the executable
echo "Running gpu_neuron..."
if [ -f "build/Release/gpu_neuron.exe" ]; then
    # Windows MSVC build
    ./build/Release/gpu_neuron.exe
elif [ -f "build/gpu_neuron.exe" ]; then
    # Windows MinGW build
    ./build/gpu_neuron.exe
elif [ -f "build/gpu_neuron" ]; then
    # Unix/Mac build
    ./build/gpu_neuron
else
    echo "Error: gpu_neuron executable not found!"
    exit 1
fi