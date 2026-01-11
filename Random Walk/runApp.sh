#!/bin/bash
# Auto-rebuild and run your SFML project
# Just run: ./runApp.sh

cd "$(dirname "$0")"

# Rebuild if source files changed
echo "🔨 Building project..."
cd build
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4) > /dev/null 2>&1
cd ..

# Run the executable
echo "🚀 Running randomWalk..."
./build/randomWalk