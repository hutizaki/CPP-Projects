# CPP Projects Workspace

A cross-platform workspace for SFML C++ projects. Works on **Windows**, **macOS**, and **Linux**.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/hutizaki/C-Projects.git
cd C-Projects
```

### 2. Run Setup (Like `npm install`)

**This is all you need!** The setup script builds SFML and all projects automatically:

```bash
./setup.sh
```

This will:
- ✅ Build SFML (shared across all projects) - takes 5-10 minutes first time
- ✅ Build all existing projects automatically
- ✅ Set up everything needed to run projects

**Works on Windows (Git Bash/WSL), macOS, and Linux!**

### 3. Create a New Project

Run the project creation script:

```bash
./create_sfml_project.sh
```

Enter your project name when prompted. The script will:
- Create a new folder with your exact project name
- Generate a `CMakeLists.txt` configured for SFML
- Create a main `.cpp` file (camelCase naming)
- Generate a `runApp.sh` script for easy building and running
- Build the project automatically

### 4. Run Your Project

Navigate to your project folder and run:

```bash
cd "Your Project Name"
./runApp.sh
```

The `runApp.sh` script automatically:
- Rebuilds your project if source files changed
- Runs the executable

**On Windows:** The script works in Git Bash or WSL. It automatically detects Visual Studio generators and handles `.exe` executables.

## Requirements

### Windows (Fresh PC? Read This First!)

**🚀 Brand new Windows PC with nothing installed?**

See **[WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md)** for automatic installation!

Just download the repo, right-click `WINDOWS_INSTALL.bat`, select "Run as Administrator", and you're done!

---

**Already have tools installed?** You need:
- **CMake** (3.20 or newer) - [Download](https://cmake.org/download/)
- **C++ Compiler:**
  - Visual Studio 2019 or newer (with C++ workload), OR
  - MinGW-w64, OR
  - Clang
- **Git Bash** (comes with Git for Windows) or **WSL** for running `.sh` scripts

### macOS
- **CMake**: `brew install cmake`
- **Xcode Command Line Tools**: `xcode-select --install`

### Linux
- **CMake**: `sudo apt install cmake` (Ubuntu/Debian) or equivalent
- **Build tools**: `sudo apt install build-essential`

## Project Structure

```
CPP Projects/
├── setup.sh            # Setup script (builds SFML + all projects)
├── _sfml/              # SFML build (shared across all projects)
│   ├── build/          # SFML build directory
│   └── install/        # SFML installation
├── create_sfml_project.sh  # Project generator script
├── Your Project Name/   # Your project folder
│   ├── yourProjectName.cpp
│   ├── CMakeLists.txt
│   ├── runApp.sh       # Build and run script
│   └── build/          # Project build directory
└── ...
```

## How It Works

- **Workspace-level SFML**: SFML is built once at `_sfml/` and shared by all projects
- **Platform-specific builds**: SFML is compiled on each machine (Windows/macOS/Linux)
  - `_sfml/CMakeLists.txt` downloads SFML 3.0.2 from GitHub
  - `setup.sh` builds and installs it to `_sfml/install/`
  - Each project links to the local SFML installation
- **Cross-platform scripts**: `runApp.sh` automatically detects your build system:
  - Visual Studio generators (Windows) → uses `cmake --build . --config Release`
  - Unix Makefiles → uses `cmake --build` or `make`
- **Path handling**: CMakeLists.txt files handle Windows backslashes and Unix forward slashes automatically

### Why SFML Isn't in Git

SFML binaries are **platform-specific** and **large** (~100MB compiled). Instead:
- ✅ Git tracks `_sfml/CMakeLists.txt` (the build recipe)
- ✅ `setup.sh` builds SFML locally on first run
- ✅ Works the same on Windows, macOS, and Linux
- ✅ Just like `npm install` rebuilds `node_modules/` on each machine

## Manual Setup (Alternative)

If you prefer to build manually instead of using `setup.sh`:

### Build SFML

#### On Windows (Git Bash or PowerShell):

```bash
cd _sfml
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --config Release
cmake --install . --config Release
cd ..
echo "$(cd install/lib/cmake/SFML && pwd -W)" > SFML_DIR.txt
cd ..
```

**Note for Windows:** If using PowerShell, use `(Resolve-Path install\lib\cmake\SFML).Path` instead of the `pwd -W` command.

#### On macOS/Linux:

```bash
cd _sfml
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=../install
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --install .
cd ..
echo "$(cd install/lib/cmake/SFML && pwd)" > SFML_DIR.txt
cd ..
```

## Troubleshooting

### "SFML not found" error
- Run `./setup.sh` to build SFML automatically
- Or manually build SFML following the instructions above
- Check that `_sfml/SFML_DIR.txt` exists and contains the correct path

### Build errors on Windows
- Ensure you have Visual Studio installed with the C++ workload, OR
- Install MinGW-w64 and make sure it's in your PATH
- Try running `cmake ..` manually in the project's `build/` directory to see detailed errors

### "Permission denied" when running runApp.sh
- On Unix/macOS: `chmod +x runApp.sh`
- On Windows: Make sure you're using Git Bash or WSL

### Executable not found
- Check the `build/` directory for your executable
- On Windows with Visual Studio, executables are in `build/Release/`
- On Unix/macOS, executables are directly in `build/`

## Notes

- Project folder names preserve your exact input (spacing, capitalization)
- Source files use camelCase (e.g., "Random Walk" → `randomWalk.cpp`)
- All projects use C++17 standard
- SFML is statically linked (no DLL dependencies)

