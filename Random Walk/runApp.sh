#!/bin/bash
# Auto-rebuild and run your SFML project
# Just run: ./runApp.sh
# Works on Windows (Git Bash/WSL), macOS, and Linux

# Get script directory (portable way)
SCRIPT_DIR="$( cd -- "$( pwd )" && pwd )"
cd "$SCRIPT_DIR"

# Get project name from directory
PROJECT_NAME="randomWalk"

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Linux"
fi

# Check if build directory exists, if not configure
if [ ! -f "build/CMakeCache.txt" ]; then
    echo "🔧 First-time setup: Configuring CMake..."
    
    # Try to find SFML_DIR from repo (use read builtin, it's always available)
    SFML_DIR=""
    if [ -f "../_sfml/SFML_DIR.txt" ]; then
        read -r SFML_DIR < "../_sfml/SFML_DIR.txt"
    elif [ -f "../../_sfml/SFML_DIR.txt" ]; then
        read -r SFML_DIR < "../../_sfml/SFML_DIR.txt"
    fi
    
    # Convert Unix path to Windows path if needed (pure bash, no external commands)
    if [[ "$SFML_DIR" == /c/* ]]; then
        SFML_DIR="C:/${SFML_DIR#/c/}"
    elif [[ "$SFML_DIR" == /mnt/c/* ]]; then
        SFML_DIR="C:/${SFML_DIR#/mnt/c/}"
    fi
    
    # Configure with CMake (use -B to create build directory automatically)
    if [ "$OS" = "Windows" ]; then
        # Windows: try Visual Studio generator
        if [ -n "$SFML_DIR" ]; then
            cmake -B build -G "Visual Studio 17 2022" -DSFML_DIR="$SFML_DIR" || cmake -B build -DSFML_DIR="$SFML_DIR"
        else
            cmake -B build -G "Visual Studio 17 2022" || cmake -B build
        fi
    else
        # macOS/Linux: use default generator
        if [ -n "$SFML_DIR" ]; then
            cmake -B build -DSFML_DIR="$SFML_DIR"
        else
            cmake -B build
        fi
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
    if [ -f "Release/${PROJECT_NAME}.exe" ]; then
        EXE_PATH="Release/${PROJECT_NAME}.exe"
    elif [ -f "Debug/${PROJECT_NAME}.exe" ]; then
        EXE_PATH="Debug/${PROJECT_NAME}.exe"
    elif [ -f "${PROJECT_NAME}.exe" ]; then
        EXE_PATH="${PROJECT_NAME}.exe"
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
    if [ -f "${PROJECT_NAME}.exe" ]; then
        EXE_PATH="${PROJECT_NAME}.exe"
    elif [ -f "${PROJECT_NAME}" ]; then
        EXE_PATH="${PROJECT_NAME}"
    fi
fi

cd ..

# Run the executable - try multiple locations
echo "🚀 Running ${PROJECT_NAME}..."
FOUND=false

# Try the detected path first
if [ -n "$EXE_PATH" ] && [ -f "build/$EXE_PATH" ]; then
    ./build/$EXE_PATH
    FOUND=true
# Try common Windows locations
elif [ -f "build/Release/${PROJECT_NAME}.exe" ]; then
    ./build/Release/${PROJECT_NAME}.exe
    FOUND=true
elif [ -f "build/Debug/${PROJECT_NAME}.exe" ]; then
    ./build/Debug/${PROJECT_NAME}.exe
    FOUND=true
elif [ -f "build/${PROJECT_NAME}.exe" ]; then
    ./build/${PROJECT_NAME}.exe
    FOUND=true
elif [ -f "build/${PROJECT_NAME}" ]; then
    ./build/${PROJECT_NAME}
    FOUND=true
fi

if [ "$FOUND" = false ]; then
    echo "❌ Error: Executable not found!" >&2
    echo "Searched in:" >&2
    echo "  - build/$EXE_PATH" >&2
    echo "  - build/Release/${PROJECT_NAME}.exe" >&2
    echo "  - build/Debug/${PROJECT_NAME}.exe" >&2
    echo "  - build/${PROJECT_NAME}.exe" >&2
    echo "  - build/${PROJECT_NAME}" >&2
    echo "" >&2
    echo "💡 Try deleting the build/ folder and run ./runApp.sh again" >&2
    exit 1
fi