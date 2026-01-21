#!/bin/bash

# Benchmark comparison script for Phase 5
# Runs PyTorch and Phase 5 CUDA implementations with fair comparison

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Phase 5: Benchmark Comparison"
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    echo "Then: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Build Phase 5 and Phase 5.5 CUDA executables if needed
if [ ! -f "build/phase5_neuron" ] || [ ! -f "build/phase5_5_neuron" ]; then
    echo ""
    echo "Building Phase 5 and Phase 5.5 CUDA executables..."
    mkdir -p build
    cd build
    cmake .. > /dev/null 2>&1
    make > /dev/null 2>&1
    cd ..
    
    if [ ! -f "build/phase5_neuron" ]; then
        echo "Error: Failed to build Phase 5 executable"
        exit 1
    fi
    if [ ! -f "build/phase5_5_neuron" ]; then
        echo "Error: Failed to build Phase 5.5 executable"
        exit 1
    fi
    echo "✓ Phase 5 and Phase 5.5 CUDA executables built successfully"
    echo ""
fi

# Check if initial weights exist, generate if not
if [ ! -f "initial_weights_W1.bin" ]; then
    echo ""
    echo "Initial weights not found. Generating them for fair comparison..."
    echo ""
    
    # Build weight generator if needed
    if [ ! -f "build/generate_weights" ]; then
        echo "Building weight generator..."
        mkdir -p build
        cd build
        cmake .. > /dev/null 2>&1
        make > /dev/null 2>&1
        cd ..
        
        if [ ! -f "build/generate_weights" ]; then
            echo "Error: Failed to build weight generator"
            exit 1
        fi
    fi
    
    # Generate weights
    ./build/generate_weights 128 784
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate initial weights"
        exit 1
    fi
    
    echo ""
    echo "✓ Initial weights generated successfully!"
    echo ""
else
    echo "✓ Initial weights found - all implementations will use the same starting weights"
    echo ""
fi

# Check if benchmark script exists
if [ ! -f "benchmark_comparison.py" ]; then
    echo "Error: benchmark_comparison.py not found!"
    exit 1
fi

# Run the benchmark
echo "============================================================"
echo "Running Benchmark Comparison"
echo "============================================================"
echo ""
echo "This will compare:"
echo "  1. PyTorch implementation"
echo "  2. Phase 5 CUDA implementation (custom kernels)"
echo "  3. Phase 5.5 CUDA implementation (NVIDIA cuBLAS)"
echo ""
echo "All implementations use the same:"
echo "  - Initial weights (for fair comparison)"
echo "  - Training data"
echo "  - Hyperparameters (10 epochs, lr=0.1, batch_size=32)"
echo ""
echo "Starting benchmark..."
echo ""

python benchmark_comparison.py

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "============================================================"