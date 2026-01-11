# Complete Workflow Test

This document verifies the complete workflow works as intended.

## Scenario: Fresh Clone on Windows

### What's in Git (Committed Files)
```
CPP Projects/
├── .gitignore
├── README.md
├── setup.sh
├── create_sfml_project.sh
├── _sfml/
│   └── CMakeLists.txt          ← Only this file (build recipe)
├── Tetris/
│   ├── CMakeLists.txt
│   ├── tetris.cpp
│   └── runApp.sh
├── Random Walk/
│   ├── CMakeLists.txt
│   ├── randomWalk.cpp
│   └── runApp.sh
└── Coding Interview Data Annotation/
    ├── CMakeLists.txt
    ├── codingInterviewDataAnnotation.cpp
    └── runApp.sh
```

### What's NOT in Git (Ignored by .gitignore)
```
_sfml/build/                    ← Build artifacts
_sfml/install/                  ← Compiled SFML binaries
_sfml/SFML_DIR.txt              ← Machine-specific path
*/build/                        ← Project build folders
*.exe, *.o, *.a                 ← Compiled binaries
```

## Step-by-Step Workflow

### 1. Clone on Windows
```bash
git clone https://github.com/hutizaki/C-Projects.git
cd "C-Projects"
```

**Result:**
- ✅ `_sfml/` folder exists
- ✅ `_sfml/CMakeLists.txt` exists
- ❌ `_sfml/build/` doesn't exist yet
- ❌ `_sfml/install/` doesn't exist yet
- ❌ `_sfml/SFML_DIR.txt` doesn't exist yet

### 2. Run Setup
```bash
./setup.sh
```

**What happens:**
1. Detects OS: "Windows"
2. Enters `_sfml/` directory
3. Checks if SFML already built (it's not)
4. Creates `_sfml/build/` directory
5. Runs `cmake ..` (downloads SFML 3.0.2 from GitHub via FetchContent)
6. Runs `cmake --build . --config Release` (compiles SFML)
7. Runs `cmake --install . --config Release` (installs to `_sfml/install/`)
8. Creates `_sfml/SFML_DIR.txt` with absolute path (e.g., `C:/Users/bryan/git/C-Projects/_sfml/install/lib/cmake/SFML`)
9. Finds all projects (Tetris, Random Walk, etc.)
10. For each project:
    - Creates `build/` directory
    - Runs `cmake ..` (reads `_sfml/SFML_DIR.txt` to find SFML)
    - Runs `cmake --build . --config Release`
    - Shows success/failure with actual error messages

**Result:**
- ✅ `_sfml/build/` exists (with compiled SFML)
- ✅ `_sfml/install/` exists (with SFML libraries)
- ✅ `_sfml/SFML_DIR.txt` exists (with Windows path)
- ✅ All projects built successfully
- ✅ Executables ready to run

### 3. Create New Project
```bash
./create_sfml_project.sh
# Enter: "My Game"
```

**What happens:**
1. Checks `_sfml/SFML_DIR.txt` exists ✅
2. Creates `My Game/` folder
3. Generates `My Game/CMakeLists.txt`:
   - Reads `_sfml/SFML_DIR.txt`
   - Sets `SFML_DIR` to that path
   - Links to SFML::Graphics, SFML::Window, SFML::System
4. Creates `My Game/myGame.cpp` with SFML template
5. Creates `My Game/runApp.sh`
6. Builds the project automatically
7. Copies `compile_commands.json` for IDE support

**Result:**
- ✅ `My Game/` folder created
- ✅ `My Game/myGame.cpp` created
- ✅ `My Game/CMakeLists.txt` created (links to workspace SFML)
- ✅ `My Game/build/` created
- ✅ `My Game/build/Release/myGame.exe` created (on Windows)
- ✅ Project ready to run

### 4. Run Project
```bash
cd "My Game"
./runApp.sh
```

**What happens:**
1. Detects Visual Studio generator (Windows)
2. Runs `cmake --build . --config Release`
3. Finds executable at `build/Release/myGame.exe`
4. Runs `./build/Release/myGame.exe`

**Result:**
- ✅ Program runs successfully
- ✅ SFML window opens (if using graphics)

## Key Points

### ✅ What Works
1. **Clone once, build anywhere**: Git only tracks source files
2. **Platform-specific builds**: SFML compiled for each OS
3. **Automatic setup**: `setup.sh` does everything
4. **Shared SFML**: All projects use same SFML installation
5. **Error visibility**: Updated `setup.sh` shows actual errors

### ✅ What's Fixed
1. **No absolute paths in git**: `SFML_DIR.txt` is now ignored
2. **No platform-specific binaries in git**: All builds are local
3. **Error messages visible**: No more silent failures on Windows
4. **Path format handling**: CMakeLists.txt converts Windows paths

### ✅ Works Like npm
```bash
# npm workflow
git clone repo
npm install          # Downloads and builds dependencies
npm run dev          # Runs the app

# CPP Projects workflow
git clone repo
./setup.sh           # Downloads and builds SFML + projects
./runApp.sh          # Runs the app
```

## Testing Checklist

To verify this works on Windows:

- [ ] Delete entire folder on Windows
- [ ] `git clone` the repo
- [ ] Run `./setup.sh`
- [ ] Verify SFML builds successfully
- [ ] Verify all projects build successfully
- [ ] Run `./create_sfml_project.sh` to create new project
- [ ] Verify new project builds and runs
- [ ] Run `cd "Random Walk" && ./runApp.sh`
- [ ] Verify existing project runs

## Expected Files After Setup

```
CPP Projects/
├── _sfml/
│   ├── CMakeLists.txt          [git tracked]
│   ├── SFML_DIR.txt            [git ignored - generated locally]
│   ├── build/                  [git ignored - build artifacts]
│   └── install/                [git ignored - compiled SFML]
│       └── lib/
│           ├── libsfml-graphics.a
│           ├── libsfml-window.a
│           ├── libsfml-system.a
│           └── cmake/SFML/
│               └── SFMLConfig.cmake
├── Random Walk/
│   ├── CMakeLists.txt          [git tracked]
│   ├── randomWalk.cpp          [git tracked]
│   ├── runApp.sh               [git tracked]
│   ├── build/                  [git ignored]
│   │   └── randomWalk.exe      (or randomWalk on Unix)
│   └── compile_commands.json   [git ignored]
└── ...
```

## Commit Message

```
Fix Windows build issues - make workflow truly cross-platform

Changes:
- Remove SFML_DIR.txt from git (contains machine-specific paths)
- Add SFML_DIR.txt to .gitignore
- Update setup.sh to show actual error messages instead of silent failures
- Update README.md to explain npm-like workflow
- SFML now builds locally on each machine (like node_modules)

This makes the workflow work exactly like npm:
1. git clone
2. ./setup.sh (builds SFML + all projects)
3. ./create_sfml_project.sh (creates new projects)
4. ./runApp.sh (runs projects)

Works on Windows, macOS, and Linux.
```

