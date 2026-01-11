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

# Check if SFML is built
if [ ! -f "_sfml/SFML_DIR.txt" ]; then
    echo "❌ SFML not found! Run ./setup.sh first"
    exit 1
fi

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

echo "Using SFML from: $SFML_DIR"
echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
echo ""

# Find all directories with CMakeLists.txt (except _sfml)
PROJECTS=()
for dir in */; do
    dir="${dir%/}"  # Remove trailing slash
    if [ "$dir" != "_sfml" ] && [ -f "$dir/CMakeLists.txt" ]; then
        PROJECTS+=("$dir")
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
        if [ -n "$SFML_DIR" ]; then
            cmake -B build -G "Visual Studio 17 2022" -DSFML_DIR="$SFML_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" > /dev/null 2>&1 || cmake -B build -DSFML_DIR="$SFML_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" > /dev/null 2>&1
        else
            cmake -B build -G "Visual Studio 17 2022" > /dev/null 2>&1 || cmake -B build > /dev/null 2>&1
        fi
        
        if [ $? -ne 0 ]; then
            echo "❌ Configuration failed for $project"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_PROJECTS+=("$project (configure)")
            cd "$SCRIPT_DIR"
            continue
        fi
    fi
    
    # Build
    echo "Building..."
    cmake --build build --config Release > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $project built successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Build failed for $project"
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
