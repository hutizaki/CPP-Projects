#!/bin/bash
# Verification script to check if the workspace is set up correctly
# Run this after ./setup.sh to verify everything works

echo "=========================================="
echo "  CPP Projects Workspace Verification"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# Check 1: SFML CMakeLists.txt exists (tracked in git)
echo -n "✓ Checking _sfml/CMakeLists.txt... "
if [ -f "_sfml/CMakeLists.txt" ]; then
    echo "PASS"
    ((PASS++))
else
    echo "FAIL - Missing!"
    ((FAIL++))
fi

# Check 2: SFML is built
echo -n "✓ Checking _sfml/install/... "
if [ -d "_sfml/install/lib" ] && [ -f "_sfml/install/lib/libsfml-graphics.a" ]; then
    echo "PASS"
    ((PASS++))
else
    echo "FAIL - Run ./setup.sh first!"
    ((FAIL++))
fi

# Check 3: SFML_DIR.txt exists (generated locally)
echo -n "✓ Checking _sfml/SFML_DIR.txt... "
if [ -f "_sfml/SFML_DIR.txt" ]; then
    SFML_DIR=$(cat "_sfml/SFML_DIR.txt")
    if [ -d "$SFML_DIR" ] && [ -f "$SFML_DIR/SFMLConfig.cmake" ]; then
        echo "PASS"
        echo "  Path: $SFML_DIR"
        ((PASS++))
    else
        echo "FAIL - Path invalid: $SFML_DIR"
        ((FAIL++))
    fi
else
    echo "FAIL - Run ./setup.sh first!"
    ((FAIL++))
fi

# Check 4: SFML_DIR.txt is NOT tracked by git
echo -n "✓ Checking _sfml/SFML_DIR.txt not in git... "
if git ls-files --error-unmatch _sfml/SFML_DIR.txt >/dev/null 2>&1; then
    echo "FAIL - Should be in .gitignore!"
    ((FAIL++))
else
    echo "PASS"
    ((PASS++))
fi

# Check 5: Projects are built
echo ""
echo "Checking projects:"
PROJECTS=("Tetris" "Random Walk" "Coding Interview Data Annotation")
for project in "${PROJECTS[@]}"; do
    echo -n "  ✓ $project... "
    if [ -d "$project/build" ]; then
        # Check for executable (with or without .exe)
        project_name=$(echo "$project" | awk '{
            for(i=1; i<=NF; i++) {
                word = tolower($i)
                if(i == 1) {
                    result = word
                } else {
                    first = substr(word, 1, 1)
                    rest = substr(word, 2)
                    result = result toupper(first) rest
                }
            }
            print result
        }')
        
        if [ -f "$project/build/$project_name" ] || [ -f "$project/build/$project_name.exe" ] || \
           [ -f "$project/build/Release/$project_name.exe" ] || [ -f "$project/build/Debug/$project_name.exe" ]; then
            echo "PASS"
            ((PASS++))
        else
            echo "FAIL - Executable not found"
            ((FAIL++))
        fi
    else
        echo "FAIL - build/ folder missing"
        ((FAIL++))
    fi
done

# Check 6: Scripts are executable
echo ""
echo "Checking scripts:"
SCRIPTS=("setup.sh" "create_sfml_project.sh")
for script in "${SCRIPTS[@]}"; do
    echo -n "  ✓ $script executable... "
    if [ -x "$script" ]; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL - Run: chmod +x $script"
        ((FAIL++))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "  Verification Results"
echo "=========================================="
echo "  ✓ Passed: $PASS"
if [ $FAIL -gt 0 ]; then
    echo "  ❌ Failed: $FAIL"
    echo ""
    echo "Some checks failed. Please:"
    echo "  1. Run ./setup.sh if SFML is not built"
    echo "  2. Check error messages above"
    echo "  3. Make sure you're in the workspace root"
    exit 1
else
    echo ""
    echo "All checks passed! ✅"
    echo ""
    echo "Your workspace is ready:"
    echo "  - SFML is built and installed"
    echo "  - All projects are compiled"
    echo "  - Scripts are executable"
    echo ""
    echo "You can now:"
    echo "  - Run projects: cd \"Project Name\" && ./runApp.sh"
    echo "  - Create new projects: ./create_sfml_project.sh"
    echo ""
fi

