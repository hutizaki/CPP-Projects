#!/bin/bash
# Generate .clangd config files for all projects
# Run this after setup.sh to enable IntelliSense

echo "=========================================="
echo "  Generating clangd Configs"
echo "=========================================="
echo ""

# Get workspace root
workspace_root="$(pwd)"

# Check if SFML_DIR.txt exists
if [ ! -f "_sfml/SFML_DIR.txt" ]; then
    echo "❌ Error: _sfml/SFML_DIR.txt not found!"
    echo "   Run ./setup.sh first to build SFML"
    exit 1
fi

# Read SFML_DIR
SFML_DIR=$(cat "_sfml/SFML_DIR.txt")
SFML_INCLUDE_PATH="${SFML_DIR%/lib/cmake/SFML}/include"

# Convert backslashes to forward slashes (Windows compatibility)
SFML_INCLUDE_PATH="${SFML_INCLUDE_PATH//\\//}"

echo "SFML include path: $SFML_INCLUDE_PATH"
echo ""

# Find all project directories
PROJECTS=()
for dir in */; do
    dir="${dir%/}"
    if [ "$dir" != "_sfml" ] && [ -f "$dir/CMakeLists.txt" ]; then
        PROJECTS+=("$dir")
    fi
done

if [ ${#PROJECTS[@]} -eq 0 ]; then
    echo "⚠️  No projects found."
    exit 0
fi

echo "Found ${#PROJECTS[@]} project(s):"
for project in "${PROJECTS[@]}"; do
    echo "  - $project"
done
echo ""

# Generate .clangd for each project
for project in "${PROJECTS[@]}"; do
    echo "Generating .clangd for: $project"
    
    cat > "$project/.clangd" << EOF
CompileFlags:
  Add: 
    - -std=c++17
    - -I${SFML_INCLUDE_PATH}
  
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
    
    echo "  ✓ Created $project/.clangd"
done

echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
echo ""
echo "All projects now have .clangd configs."
echo "Restart your IDE/editor to apply changes."
echo ""

