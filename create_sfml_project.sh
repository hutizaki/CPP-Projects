#!/bin/bash

# Script to create a new SFML C++ project
# Usage: ./create_sfml_project.sh

echo "=========================================="
echo "  SFML Project Generator"
echo "=========================================="
echo ""

# Prompt for project name
read -p "What do you want to name the project? " folder_name

if [ -z "$folder_name" ]; then
    echo "Error: Project name cannot be empty!"
    exit 1
fi

# Check if project already exists
if [ -d "$folder_name" ]; then
    echo "Error: Project '$folder_name' already exists!"
    exit 1
fi


# Convert to camelCase for .cpp filename (pure bash, no external commands)
# Split by spaces, lowercase first word, capitalize first letter of subsequent words
project_name=""
first_word=true
for word in $folder_name; do
    # Convert to lowercase using bash parameter expansion
    word_lower="${word,,}"
    
    if [ "$first_word" = true ]; then
        project_name="$word_lower"
        first_word=false
    else
        # Capitalize first letter
        first_char="${word_lower:0:1}"
        first_char="${first_char^^}"
        rest="${word_lower:1}"
        project_name="${project_name}${first_char}${rest}"
    fi
done

echo ""
echo "Creating project: $folder_name"
echo "  Folder: $folder_name"
echo "  Source file: ${project_name}.cpp"
echo ""

# Store the workspace root path before changing directories
workspace_root="$(pwd)"

# Check if SFML exists at workspace root
if [ ! -d "$workspace_root/_sfml" ]; then
    echo "⚠️  SFML not found at workspace root!"
    echo ""
    echo "To set up SFML manually:"
    echo "  1. mkdir -p _sfml && cd _sfml"
    echo "  2. Create CMakeLists.txt with FetchContent to download SFML 3.0.2"
    echo "  3. mkdir build && cd build"
    echo "  4. cmake .. -DBUILD_SHARED_LIBS=OFF"
    echo "  5. cmake --build ."
    echo "  6. cd _deps/sfml-build && cmake --install . --prefix ../../../install"
    echo "  7. cd ../../../ && echo \"\$(cd install/lib/cmake/SFML && pwd)\" > SFML_DIR.txt"
    echo ""
    exit 1
fi

# Check if SFML_DIR.txt exists (saved from manual build)
if [ -f "$workspace_root/_sfml/SFML_DIR.txt" ]; then
    # Use read builtin instead of cat
    read -r SFML_DIR_PATH < "$workspace_root/_sfml/SFML_DIR.txt"
    if [ -d "$SFML_DIR_PATH" ] && [ -f "$SFML_DIR_PATH/SFMLConfig.cmake" ]; then
        echo "✓ Found SFML at workspace root: $SFML_DIR_PATH"
        echo ""
    else
        echo "⚠️  SFML_DIR.txt exists but SFML install directory not found."
        echo "   Please rebuild and install SFML manually."
        echo ""
        exit 1
    fi
else
    echo "⚠️  SFML directory exists but SFML_DIR.txt not found."
    echo "   SFML may not be fully installed. Please install SFML manually:"
    echo "   cd _sfml/build/_deps/sfml-build"
    echo "   cmake --install . --prefix ../../../install"
    echo "   cd ../../../ && echo \"\$(cd install/lib/cmake/SFML && pwd)\" > SFML_DIR.txt"
    echo ""
    exit 1
fi

# Create project directory (preserve exact name)
mkdir -p "$folder_name"
cd "$folder_name"

# Create CMakeLists.txt that references workspace-level SFML
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(PROJECT_NAME CXX)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard explicitly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find SFML from workspace root
get_filename_component(WORKSPACE_ROOT "${CMAKE_SOURCE_DIR}" DIRECTORY)

# Read SFML_DIR from the file saved during SFML build
if(EXISTS "${WORKSPACE_ROOT}/_sfml/SFML_DIR.txt")
    file(READ "${WORKSPACE_ROOT}/_sfml/SFML_DIR.txt" SFML_DIR_ABS)
    string(STRIP "${SFML_DIR_ABS}" SFML_DIR_ABS)
    # Convert Windows paths to CMake format (handles backslashes)
    file(TO_CMAKE_PATH "${SFML_DIR_ABS}" SFML_DIR_ABS)
    set(SFML_DIR "${SFML_DIR_ABS}" CACHE PATH "Path to SFML install directory")
else()
    # Fallback: try to find it in the install directory
    file(TO_CMAKE_PATH "${WORKSPACE_ROOT}/_sfml/install/lib/cmake/SFML" SFML_DIR_ABS)
    set(SFML_DIR "${SFML_DIR_ABS}" CACHE PATH "Path to SFML install directory")
endif()

# Set SFML to use static libraries (since FetchContent builds static by default)
set(SFML_STATIC_LIBRARIES ON CACHE BOOL "Use SFML static libraries" FORCE)

# Find SFML package
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

add_executable(PROJECT_NAME PROJECT_NAME.cpp)

# Set C++ standard on target as well
set_target_properties(PROJECT_NAME PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Link against SFML
target_link_libraries(PROJECT_NAME PRIVATE 
    SFML::Graphics 
    SFML::Window 
    SFML::System
)

# Set working directory to project root so your relative asset paths work
set_property(TARGET PROJECT_NAME PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
EOF

# Replace PROJECT_NAME placeholders in CMakeLists.txt
# Cross-platform sed: macOS requires extension, Linux/Windows doesn't
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/PROJECT_NAME/$project_name/g" CMakeLists.txt
else
    sed -i "s/PROJECT_NAME/$project_name/g" CMakeLists.txt
fi

# Create the main .cpp file with SFML basics (SFML 3 API - following official tutorial)
cat > "${project_name}.cpp" << 'EOF'
#include <iostream>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

int main()
{
  cout << "Hello World";
    
  return 0;
}
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Build directories
build/
cmake-build-*/

# Compiled files
*.o
*.a
*.so
*.dylib
*.exe

# CMake files
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile
*.cmake

# Workspace-level SFML (shared across projects)
_sfml/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# macOS
.DS_Store
EOF

# Create .clangd config for better IntelliSense
cat > .clangd << 'EOF'
CompileFlags:
  Add: [-std=c++17]
  
Diagnostics:
  UnusedIncludes: None
  MissingIncludes: None

Completion:
  AllScopes: true
  
Index:
  Background: Build

InlayHints:
  Enabled: true
  ParameterNames: true
  DeducedTypes: true
EOF

# Create .vscode/settings.json for clangd configuration
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "clangd.arguments": [
    "--background-index",
    "--clang-tidy",
    "--completion-style=detailed",
    "--header-insertion=iwyu",
    "--pch-storage=memory",
    "--function-arg-placeholders=true",
    "--header-insertion-decorators=true",
    "--hover"
  ],
  "clangd.fallbackFlags": [
    "-std=c++17"
  ],
  "editor.inlayHints.enabled": "on",
  "editor.suggest.showMethods": true,
  "editor.suggest.showFunctions": true,
  "editor.suggest.showConstructors": true,
  "editor.suggest.documentation": true,
  "editor.parameterHints.enabled": true,
  "editor.suggest.preview": true,
  "editor.hover.enabled": true,
  "editor.hover.delay": 100,
  "C_Cpp.intelliSenseEngine": "disabled"
}
EOF

# Copy runApp.sh from an existing project template (use Random Walk as reference)
if [ -f "$workspace_root/Random Walk/runApp.sh" ]; then
    cp "$workspace_root/Random Walk/runApp.sh" runApp.sh
    # Replace the project name in the script
    # Cross-platform sed: macOS requires extension, Linux/Windows doesn't
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i.bak "s/PROJECT_NAME=\"randomWalk\"/PROJECT_NAME=\"$project_name\"/g" runApp.sh
        rm -f runApp.sh.bak
    else
        sed -i "s/PROJECT_NAME=\"randomWalk\"/PROJECT_NAME=\"$project_name\"/g" runApp.sh
    fi
    chmod +x runApp.sh
else
    echo "⚠️  Warning: Could not find template runApp.sh. Creating basic version."
    # Fallback: create minimal runApp.sh
    cat > runApp.sh << 'EOF'
#!/bin/bash
cd "$(pwd)"
PROJECT_NAME="PROJECT_NAME_PLACEHOLDER"
echo "🔨 Building and running $PROJECT_NAME..."
cmake -B build && cmake --build build --config Release && (./build/Release/${PROJECT_NAME}.exe || ./build/${PROJECT_NAME}.exe || ./build/${PROJECT_NAME})
EOF
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i.bak "s/PROJECT_NAME_PLACEHOLDER/$project_name/g" runApp.sh
        rm -f runApp.sh.bak
    else
        sed -i "s/PROJECT_NAME_PLACEHOLDER/$project_name/g" runApp.sh
    fi
    chmod +x runApp.sh
fi

echo "✓ Project directory created: $folder_name/"
echo "✓ CMakeLists.txt created (references workspace-level SFML)"
echo "✓ ${project_name}.cpp created with SFML basics"
echo "✓ runApp.sh created (run with: ./runApp.sh)"
echo "✓ .gitignore created"
echo "✓ .clangd config created (for better IntelliSense)"
echo "✓ .vscode/settings.json created (for clangd configuration)"
echo ""
echo "Building project..."
echo ""

# Detect OS for proper CMake generator
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Linux"
fi

# Create build directory and build the project
mkdir -p build
cd build

# Convert SFML_DIR_PATH to Windows format if needed
SFML_CMAKE_DIR="$SFML_DIR_PATH"
if [[ "$SFML_CMAKE_DIR" == /c/* ]]; then
    SFML_CMAKE_DIR="C:/${SFML_CMAKE_DIR#/c/}"
elif [[ "$SFML_CMAKE_DIR" == /mnt/c/* ]]; then
    SFML_CMAKE_DIR="C:/${SFML_CMAKE_DIR#/mnt/c/}"
fi

# Get CMAKE_PREFIX_PATH from SFML_DIR
CMAKE_PREFIX_PATH_VAL="${SFML_CMAKE_DIR%/lib/cmake/SFML}"

# Configure and build with verbose output on error
# Use cmake --build for cross-platform compatibility (works with all generators)
if [ "$OS" = "Windows" ]; then
    CMAKE_RESULT=$(cmake .. -G "Visual Studio 17 2022" -DSFML_DIR="$SFML_CMAKE_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_VAL" 2>&1 || cmake .. -DSFML_DIR="$SFML_CMAKE_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_VAL" 2>&1)
else
    CMAKE_RESULT=$(cmake .. -DSFML_DIR="$SFML_CMAKE_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_VAL" 2>&1)
fi

echo "$CMAKE_RESULT"

if [ $? -eq 0 ] && cmake --build . 2>&1; then
    # Copy compile_commands.json to project root for IDE support
    if [ -f compile_commands.json ]; then
        cp compile_commands.json ..
    fi
    cd "$workspace_root"
    echo ""
    echo "=========================================="
    echo "  Project Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Project '$folder_name' is ready and built!"
    echo ""
    echo "To run your project:"
    echo "  cd $folder_name"
    echo "  ./runApp.sh"
    echo ""
else
    cd "$workspace_root"
    echo ""
    cd "$workspace_root"
    echo ""
    cd "$workspace_root"
    echo ""
    echo "⚠️  Build failed!"
    echo ""
    echo "Common issues:"
    echo "  1. SFML not built: Build SFML manually in _sfml directory"
    echo "  2. SFML_STATIC_LIBRARIES mismatch: Check SFML build configuration"
    echo "  3. CMake version too old: brew upgrade cmake"
    echo "  4. Compiler issues: Check Xcode Command Line Tools"
    echo ""
    echo "Check the error messages above for details."
    echo ""
    exit 1
fi
