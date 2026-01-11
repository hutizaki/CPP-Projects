#!/bin/bash
# Auto-rebuild and run your SFML project
# Just run: ./runApp.sh
# Works on Windows (Git Bash/WSL), macOS, and Linux

cd "$(dirname "$0")"

# Rebuild if source files changed
echo "🔨 Building project..."
cd build

EXE_PATH=""
IS_VS_GENERATOR=false

# Detect build system
if [ -f "CMakeCache.txt" ]; then
    # Check if using Visual Studio generator (Windows) - more robust detection
    if grep -q "CMAKE_GENERATOR:INTERNAL=Visual Studio" CMakeCache.txt 2>/dev/null || \
       grep -E "CMAKE_GENERATOR:INTERNAL=.*Visual Studio" CMakeCache.txt 2>/dev/null || \
       [ -d "Release" ] || [ -d "Debug" ]; then
        IS_VS_GENERATOR=true
    fi
fi

# Build the project
if [ "$IS_VS_GENERATOR" = true ]; then
    # Visual Studio generator - use cmake --build with config
    cmake --build . --config Release > /dev/null 2>&1
    # Check multiple possible locations
    if [ -f "Release/randomWalk.exe" ]; then
        EXE_PATH="Release/randomWalk.exe"
    elif [ -f "Debug/randomWalk.exe" ]; then
        EXE_PATH="Debug/randomWalk.exe"
    elif [ -f "randomWalk.exe" ]; then
        EXE_PATH="randomWalk.exe"
    fi
else
    # Unix Makefiles or other generators
    if command -v make >/dev/null 2>&1; then
        # Try to detect number of CPU cores
        if [ "$(uname)" = "Darwin" ]; then
            CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        elif [ "$(uname)" = "Linux" ]; then
            CORES=$(nproc 2>/dev/null || echo 4)
        else
            CORES=$NUMBER_OF_PROCESSORS
        fi
        cmake --build . -j$CORES > /dev/null 2>&1 || make -j$CORES > /dev/null 2>&1
    else
        cmake --build . > /dev/null 2>&1
    fi
    # Check for .exe extension (Windows) or no extension (Unix)
    if [ -f "randomWalk.exe" ]; then
        EXE_PATH="randomWalk.exe"
    elif [ -f "randomWalk" ]; then
        EXE_PATH="randomWalk"
    fi
fi

cd ..

# Run the executable - try multiple locations
echo "🚀 Running randomWalk..."
FOUND=false

# Try the detected path first
if [ -n "$EXE_PATH" ] && [ -f "build/$EXE_PATH" ]; then
    ./build/$EXE_PATH
    FOUND=true
# Try common Windows locations
elif [ -f "build/Release/randomWalk.exe" ]; then
    ./build/Release/randomWalk.exe
    FOUND=true
elif [ -f "build/Debug/randomWalk.exe" ]; then
    ./build/Debug/randomWalk.exe
    FOUND=true
elif [ -f "build/randomWalk.exe" ]; then
    ./build/randomWalk.exe
    FOUND=true
elif [ -f "build/randomWalk" ]; then
    ./build/randomWalk
    FOUND=true
fi

if [ "$FOUND" = false ]; then
    echo "Error: Executable not found!" >&2
    echo "Searched in:" >&2
    echo "  - build/$EXE_PATH" >&2
    echo "  - build/Release/randomWalk.exe" >&2
    echo "  - build/Debug/randomWalk.exe" >&2
    echo "  - build/randomWalk.exe" >&2
    echo "  - build/randomWalk" >&2
    echo "" >&2
    echo "Build directory contents:" >&2
    ls -la build/ >&2 || dir build\ >&2
    exit 1
fi