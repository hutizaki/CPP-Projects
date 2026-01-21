#!/bin/bash
# Script to generate initial weights for fair comparison

cd "$(dirname "$0")"

echo "Building weight generator..."
mkdir -p build
cd build
cmake .. > /dev/null 2>&1
make > /dev/null 2>&1

if [ ! -f "generate_weights" ]; then
    echo "Error: Failed to build generate_weights"
    exit 1
fi

cd ..

echo "Generating initial weights..."
./build/generate_weights 128 784

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Initial weights generated successfully!"
    echo "  Files: initial_weights_W1.bin, initial_weights_W2.bin, initial_weights_b1.bin, initial_weights_b2.bin"
    echo ""
    echo "Now all implementations (PyTorch, TensorFlow, Phase 4) will use the same initial weights."
else
    echo "Error: Failed to generate weights"
    exit 1
fi
