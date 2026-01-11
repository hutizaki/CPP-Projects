#!/bin/bash
# Auto-rebuild and run your SFML project
# Just run: ./runApp.sh
# Works on Windows (Git Bash/WSL), macOS, and Linux

# Get script directory (portable way)
SCRIPT_DIR="$( cd -- "$( pwd )" && pwd )"
cd "$SCRIPT_DIR"

# Get project name from directory
PROJECT_NAME="randomWalk"

# Check if build directory exists, if not configure
if [ ! -f "build/CMakeCache.txt" ]; then
    echo "🔧 First-time setup: Configuring CMake..."
    
    # Try to find SFML_DIR from repo
    SFML_DIR=""
    if [ -f "../_sfml/SFML_DIR.txt" ]; then
        SFML_DIR=$(head -n 1 "../_sfml/SFML_DIR.txt" 2>/dev/null || cat "../_sfml/SFML_DIR.txt" 2>/dev/null)
    elif [ -f "../../_sfml/SFML_DIR.txt" ]; then
        SFML_DIR=$(head -n 1 "../../_sfml/SFML_DIR.txt" 2>/dev/null || cat "../../_sfml/SFML_DIR.txt" 2>/dev/null)
    fi
    
    # Convert Unix path to Windows path if needed (pure bash, no sed)
    if [[ "$SFML_DIR" == /c/* ]]; then
        SFML_DIR="C:/${SFML_DIR#/c/}"
    elif [[ "$SFML_DIR" == /mnt/c/* ]]; then
        SFML_DIR="C:/${SFML_DIR#/mnt/c/}"
    fi
    
    # Configure with CMake (use -B to create build directory automatically)
    if [ -n "$SFML_DIR" ]; then
        cmake -B build -G "Visual Studio 17 2022" -DSFML_DIR="$SFML_DIR" || cmake -B build -DSFML_DIR="$SFML_DIR"
    else
        cmake -B build -G "Visual Studio 17 2022" || cmake -B build
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ CMake configuration failed!"
        echo "Make sure you ran ./setup.sh first"
        exit 1
    fi
fi

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
    echo "❌ Error: Executable not found!" >&2
    echo "Searched in:" >&2
    echo "  - build/$EXE_PATH" >&2
    echo "  - build/Release/randomWalk.exe" >&2
    echo "  - build/Debug/randomWalk.exe" >&2
    echo "  - build/randomWalk.exe" >&2
    echo "  - build/randomWalk" >&2
    echo "" >&2
    echo "💡 Try deleting the build/ folder and run ./runApp.sh again" >&2
    exit 1
fi