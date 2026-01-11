#!/bin/bash
# Auto-rebuild and run your SFML project
# Just run: ./runApp.sh
# Works on Windows (Git Bash/WSL), macOS, and Linux

cd "$(dirname "$0")"

# Rebuild if source files changed
echo "🔨 Building project..."
cd build

# Detect build system and use appropriate command
if [ -f "CMakeCache.txt" ]; then
    # Check if using Visual Studio generator (Windows)
    if grep -q "CMAKE_GENERATOR:INTERNAL=Visual Studio" CMakeCache.txt 2>/dev/null || grep -q "CMAKE_GENERATOR:INTERNAL=.*Visual Studio" CMakeCache.txt 2>/dev/null; then
        # Visual Studio generator - use cmake --build with config
        cmake --build . --config Release > /dev/null 2>&1
        EXE_PATH="Release/codingInterviewDataAnnotation.exe"
    else
        # Unix Makefiles or other generators - use cmake --build or make
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
        if [ -f "codingInterviewDataAnnotation.exe" ]; then
            EXE_PATH="codingInterviewDataAnnotation.exe"
        else
            EXE_PATH="codingInterviewDataAnnotation"
        fi
    fi
else
    # First time build - configure first
    cmake .. > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Try Visual Studio generator first on Windows
        if [ -f "CMakeCache.txt" ] && grep -q "CMAKE_GENERATOR:INTERNAL=Visual Studio" CMakeCache.txt 2>/dev/null; then
            cmake --build . --config Release > /dev/null 2>&1
            EXE_PATH="Release/codingInterviewDataAnnotation.exe"
        else
            if command -v make >/dev/null 2>&1; then
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
            if [ -f "codingInterviewDataAnnotation.exe" ]; then
                EXE_PATH="codingInterviewDataAnnotation.exe"
            else
                EXE_PATH="codingInterviewDataAnnotation"
            fi
        fi
    fi
fi

cd ..

# Run the executable
echo "🚀 Running codingInterviewDataAnnotation..."
if [ -f "build/$EXE_PATH" ]; then
    ./build/$EXE_PATH
elif [ -f "build/Release/codingInterviewDataAnnotation.exe" ]; then
    ./build/Release/codingInterviewDataAnnotation.exe
elif [ -f "build/codingInterviewDataAnnotation.exe" ]; then
    ./build/codingInterviewDataAnnotation.exe
elif [ -f "build/codingInterviewDataAnnotation" ]; then
    ./build/codingInterviewDataAnnotation
else
    echo "Error: Executable not found!" >&2
    exit 1
fi
