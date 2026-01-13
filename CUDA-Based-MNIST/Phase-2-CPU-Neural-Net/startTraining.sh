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
echo "Running neuron..."
if [ -f "build/Release/neuron.exe" ]; then
    # Windows MSVC build
    ./build/Release/neuron.exe
elif [ -f "build/neuron.exe" ]; then
    # Windows MinGW build
    ./build/neuron.exe
elif [ -f "build/neuron" ]; then
    # Unix/Mac build
    ./build/neuron
else
    echo "Error: neuron executable not found!"
    exit 1
fi