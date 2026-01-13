@echo off
REM Windows batch script for MNIST Phase 2 training

REM Create build directory if it doesn't exist
if not exist "build" mkdir build

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release

if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Release

if errorlevel 1 (
    echo Build failed!
    cd ..
    exit /b 1
)

REM Go back to Phase-2 directory to run (for correct relative paths to data files)
cd ..

REM Run the executable
echo Running neuron...
if exist "build\Release\neuron.exe" (
    build\Release\neuron.exe
) else if exist "build\neuron.exe" (
    build\neuron.exe
) else (
    echo Error: neuron executable not found!
    exit /b 1
)
