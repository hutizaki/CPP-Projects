#!/bin/bash

# Check if a file was provided
if [ -z "$1" ]; then
    echo "Usage: ./runApp_wsl.sh <filename.cu>"
    echo "Example: ./runApp_wsl.sh vectorAdd.cu"
    exit 1
fi

# Get the filename without extension for output
SOURCE_FILE="$1"
OUTPUT_FILE="${SOURCE_FILE%.cu}"

# Check if file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: File '$SOURCE_FILE' not found!"
    exit 1
fi

echo "Compiling $SOURCE_FILE for WSL..."

# Check if the file includes bryan-library
BRYAN_LIB=""
if grep -q "bryan-library" "$SOURCE_FILE"; then
    BRYAN_LIB="../bryan-library/matrixMath.cpp"
    echo "Detected bryan-library dependency, linking matrixMath.cpp..."
fi

# Compile using nvcc in WSL (no Visual Studio needed)
if [ -n "$BRYAN_LIB" ]; then
    nvcc "$SOURCE_FILE" "$BRYAN_LIB" -o "$OUTPUT_FILE" -arch=sm_89 -std=c++11
else
    nvcc "$SOURCE_FILE" -o "$OUTPUT_FILE" -arch=sm_89
fi

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Compilation successful! Running..."
    echo ""
    ./"$OUTPUT_FILE"
else
    echo ""
    echo "Compilation failed"
    exit 1
fi
