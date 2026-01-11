#!/bin/bash
# Setup script for CPP Projects workspace
# Similar to 'npm install' - builds SFML and all projects
# Usage: ./setup.sh

# Don't exit on error for project builds (continue building other projects)
# But do exit on SFML build errors
set +e

echo "=========================================="
echo "  CPP Projects Workspace Setup"
echo "=========================================="
echo ""

# Store workspace root
workspace_root="$(pwd)"

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
    CMAKE_BUILD_TYPE="Release"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    CMAKE_BUILD_TYPE="Release"
else
    OS="Linux"
    CMAKE_BUILD_TYPE="Release"
fi

echo "Detected OS: $OS"
echo ""

# Function to get CPU cores
get_cores() {
    if [[ "$OS" == "macOS" ]]; then
        sysctl -n hw.ncpu 2>/dev/null || echo 4
    elif [[ "$OS" == "Linux" ]]; then
        nproc 2>/dev/null || echo 4
    else
        echo "$NUMBER_OF_PROCESSORS" || echo 4
    fi
}

CORES=$(get_cores)

# Step 1: Build SFML
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Building SFML (one-time setup)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -d "_sfml" ]; then
    echo "❌ Error: _sfml directory not found!"
    echo "   Make sure you're in the workspace root directory."
    exit 1
fi

cd "_sfml"

# Check if SFML is already built
if [ -f "SFML_DIR.txt" ] && [ -d "install/lib/cmake/SFML" ]; then
    SFML_DIR_PATH=$(cat "SFML_DIR.txt" 2>/dev/null || echo "")
    if [ -n "$SFML_DIR_PATH" ] && [ -d "$SFML_DIR_PATH" ] && [ -f "$SFML_DIR_PATH/SFMLConfig.cmake" ]; then
        echo "✓ SFML already built and installed"
        echo "  Location: $SFML_DIR_PATH"
        echo ""
        cd "$workspace_root"
    else
        echo "⚠️  SFML_DIR.txt exists but install directory not found. Rebuilding..."
        echo ""
        BUILD_SFML=true
    fi
else
    BUILD_SFML=true
fi

if [ "$BUILD_SFML" = true ]; then
    echo "Building SFML (this may take 5-10 minutes)..."
    echo ""
    
    # Configure (use -S/-B to create build directory automatically)
    echo "Configuring SFML..."
    if [[ "$OS" == "Windows" ]]; then
        cmake -S . -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=install
    else
        cmake -S . -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=install
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ CMake configuration failed!"
        exit 1
    fi
    
    # Build
    echo "Building SFML (using $CORES cores)..."
    if [[ "$OS" == "Windows" ]]; then
        cmake --build build --config Release -j$CORES
    else
        cmake --build build -j$CORES
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ SFML build failed!"
        exit 1
    fi
    
    # Install
    echo "Installing SFML..."
    if [[ "$OS" == "Windows" ]]; then
        cmake --install build --config Release
    else
        cmake --install build
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ SFML installation failed!"
        exit 1
    fi
    
    # Create SFML_DIR.txt
    echo "Creating SFML_DIR.txt..."
    if [[ "$OS" == "Windows" ]]; then
        # Windows path handling
        if command -v cygpath >/dev/null 2>&1; then
            # Git Bash
            SFML_DIR_ABS=$(cd install/lib/cmake/SFML && pwd -W)
        else
            # PowerShell or other
            SFML_DIR_ABS=$(cd install/lib/cmake/SFML && pwd)
        fi
    else
        # macOS/Linux
        SFML_DIR_ABS=$(cd install/lib/cmake/SFML && pwd)
    fi
    
    echo "$SFML_DIR_ABS" > SFML_DIR.txt
    echo "✓ SFML built and installed successfully"
    echo "  Location: $SFML_DIR_ABS"
    echo ""
fi

cd "$workspace_root"

# Step 2: Info about running projects
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Find all project directories (those with CMakeLists.txt) - Windows compatible
PROJECTS=()
for dir in */; do
    dir="${dir%/}"  # Remove trailing slash
    if [ "$dir" != "_sfml" ] && [ -f "$dir/CMakeLists.txt" ]; then
        PROJECTS+=("$dir")
    fi
done

if [ ${#PROJECTS[@]} -eq 0 ]; then
    echo "⚠️  No projects found."
    echo "   Create a project using: ./create_sfml_project.sh"
    echo ""
else
    echo "Found ${#PROJECTS[@]} project(s):"
    for project in "${PROJECTS[@]}"; do
        echo "  - $project"
    done
    echo ""
    echo "To build all projects at once:"
    echo "  ./build_all.sh"
    echo ""
    echo "To build and run individual projects:"
    for project in "${PROJECTS[@]}"; do
        echo "  cd \"$project\" && ./runApp.sh"
    done
    echo ""
fi

echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "SFML is ready! Next steps:"
echo "  - Build/run any project: cd \"Project Name\" && ./runApp.sh"
echo "  - Build all projects: ./build_all.sh"
echo "  - Create new project: ./create_sfml_project.sh"
echo ""

