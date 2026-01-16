#!/bin/bash

# Compile
cat > temp_compile.bat << 'EOF'
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
nvcc MNIST_Phase3.cu -o MNIST_Phase3.exe -arch=sm_89 --use-local-env 2>&1
EOF

cmd.exe /c temp_compile.bat < /dev/null
rm -f temp_compile.bat

# Run
if [ -f "MNIST_Phase3.exe" ]; then
    echo ""
    ./MNIST_Phase3.exe
else
    echo ""
    echo "Compilation failed"
fi
