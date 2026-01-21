#!/bin/bash
# Build all projects in the repository
# This is useful for testing that everything compiles after making changes

echo "=========================================="
echo "  Building All CPP Projects"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd -- "$( pwd )" && pwd )"
cd "$SCRIPT_DIR"

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Linux"
fi

# Check if SFML is built (optional - only needed for SFML projects)
SFML_DIR=""
CMAKE_PREFIX_PATH=""
if [ -f "_sfml/SFML_DIR.txt" ]; then
    # Read SFML_DIR using bash builtin (always available)
    read -r SFML_DIR < "_sfml/SFML_DIR.txt"
    # Get the install directory for CMAKE_PREFIX_PATH
    CMAKE_PREFIX_PATH="${SFML_DIR%/lib/cmake/SFML}"
    # Convert Unix path to Windows path if needed (pure bash, no external commands)
    if [[ "$SFML_DIR" == /c/* ]]; then
        SFML_DIR="C:/${SFML_DIR#/c/}"
        CMAKE_PREFIX_PATH="C:/${CMAKE_PREFIX_PATH#/c/}"
    elif [[ "$SFML_DIR" == /mnt/c/* ]]; then
        SFML_DIR="C:/${SFML_DIR#/mnt/c/}"
        CMAKE_PREFIX_PATH="C:/${CMAKE_PREFIX_PATH#/mnt/c/}"
    fi
    echo "✓ SFML found at: $SFML_DIR"
else
    echo "⚠️  SFML not found (optional - only needed for SFML projects)"
    echo "   Non-SFML projects (like CUDA) will still build"
fi

echo "OS: $OS"
echo ""

# Find all directories with CMakeLists.txt (except _sfml)
# Search in root directory and one level deep in subdirectories
PROJECTS=()
for dir in */; do
    dir="${dir%/}"  # Remove trailing slash
    if [ "$dir" != "_sfml" ] && [ -f "$dir/CMakeLists.txt" ]; then
        PROJECTS+=("$dir")
    fi
    # Also check subdirectories (one level deep)
    if [ -d "$dir" ]; then
        for subdir in "$dir"/*/; do
            subdir="${subdir%/}"  # Remove trailing slash
            if [ -f "$subdir/CMakeLists.txt" ]; then
                PROJECTS+=("$subdir")
            fi
        done
    fi
done

if [ ${#PROJECTS[@]} -eq 0 ]; then
    echo "⚠️  No projects found"
    exit 0
fi

echo "Found ${#PROJECTS[@]} project(s):"
for project in "${PROJECTS[@]}"; do
    echo "  - $project"
done
echo ""

# Build each project
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_PROJECTS=()

for project in "${PROJECTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Building: $project"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$SCRIPT_DIR/$project"
    
    # Configure if needed (use -B to create build directory automatically)
    if [ ! -f "build/CMakeCache.txt" ]; then
        echo "Configuring..."
        
        # Check if this project needs SFML (by checking CMakeLists.txt)
        NEEDS_SFML=false
        if [ -f "CMakeLists.txt" ] && grep -q "find_package.*SFML\|SFML_DIR" CMakeLists.txt; then
            NEEDS_SFML=true
        fi
        
        # Build cmake command
        CMAKE_ARGS=()
        if [ "$OS" = "Windows" ]; then
            CMAKE_ARGS+=(-G "Visual Studio 17 2022")
        fi
        
        # Add SFML args only if SFML is available and project needs it
        if [ "$NEEDS_SFML" = true ]; then
            if [ -n "$SFML_DIR" ]; then
                CMAKE_ARGS+=(-DSFML_DIR="$SFML_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH")
                echo "  (Using SFML for this project)"
            else
                echo "  ⚠️  Warning: Project requires SFML but SFML not found - build may fail"
            fi
        fi
        
        # Try configuration (with fallback for Windows)
        if [ "$OS" = "Windows" ]; then
            CONFIG_OUTPUT=$(cmake -B build "${CMAKE_ARGS[@]}" 2>&1 || cmake -B build "${CMAKE_ARGS[@]}" 2>&1)
        else
            CONFIG_OUTPUT=$(cmake -B build "${CMAKE_ARGS[@]}" 2>&1)
        fi
        
        if [ $? -ne 0 ]; then
            echo "❌ Configuration failed for $project"
            echo ""
            echo "Error output:"
            echo "$CONFIG_OUTPUT"
            echo ""
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_PROJECTS+=("$project (configure)")
            cd "$SCRIPT_DIR"
            continue
        fi
    fi
    
    # Check if build directory exists and is properly configured
    if [ ! -f "build/CMakeCache.txt" ]; then
        echo "⚠️  Skipping build - project not configured"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PROJECTS+=("$project (not configured)")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # Build
    echo "Building..."
    BUILD_OUTPUT=$(cmake --build build --config Release 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "✅ $project built successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Build failed for $project"
        echo ""
        echo "Error output:"
        echo "$BUILD_OUTPUT" | grep -i "error" || echo "$BUILD_OUTPUT"
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_PROJECTS+=("$project (build)")
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
done

# Summary
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo ""
echo "Total projects: ${#PROJECTS[@]}"
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed projects:"
    for failed in "${FAILED_PROJECTS[@]}"; do
        echo "  - $failed"
    done
    exit 1
fi

echo ""
echo "🎉 All projects built successfully!"
