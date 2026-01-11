# Windows Quick Start Guide

## For a Brand New Windows PC (No Tools Installed)

This guide will get you from **zero to coding** in under 30 minutes.

---

## Option 1: Automatic Installation (Recommended)

### Step 1: Download This Repository

**If you don't have Git yet:**
1. Go to: https://github.com/hutizaki/CPP-Projects
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to a folder (e.g., `C:\Users\YourName\CPP-Projects`)

### Step 2: Run the Installer

1. **Navigate to the extracted folder**
2. **Right-click** on `WINDOWS_INSTALL.bat`
3. **Select "Run as Administrator"**
4. Follow the prompts:
   - Choose MinGW (option 1) for faster install
   - Or Visual Studio (option 2) for full IDE support

**What it installs:**
- ✅ Chocolatey (Windows package manager)
- ✅ Git (version control)
- ✅ CMake (build system)
- ✅ C++ Compiler (MinGW or Visual Studio)
- ✅ Ninja (fast build tool)

### Step 3: Build the Projects

After installation completes:

1. **Open Git Bash** (Search for "Git Bash" in Start Menu)
2. Navigate to the folder:
   ```bash
   cd /c/Users/YourName/CPP-Projects
   ```
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
   This will:
   - Download and build SFML (5-10 minutes)
   - Build all existing projects
   - Set everything up

### Step 4: Create Your First Project

```bash
./create_sfml_project.sh
# Enter your project name when prompted
```

### Step 5: Run Your Project

```bash
cd "Your Project Name"
./runApp.sh
```

**Done! 🎉**

---

## Option 2: Manual Installation

If you prefer to install tools manually:

### 1. Install Chocolatey (Package Manager)

Open **PowerShell as Administrator** and run:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Close and reopen PowerShell as Administrator.

### 2. Install Required Tools

```powershell
choco install git cmake ninja mingw -y
```

**Or for Visual Studio instead of MinGW:**

```powershell
choco install git cmake ninja visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive" -y
```

### 3. Verify Installation

Close PowerShell and open **Git Bash**, then run:

```bash
git --version
cmake --version
g++ --version  # or cl.exe --version for Visual Studio
ninja --version
```

All commands should show version numbers.

### 4. Clone and Build

```bash
git clone https://github.com/hutizaki/CPP-Projects.git
cd CPP-Projects
./setup.sh
```

---

## Troubleshooting

### "Command not found" after installation

**Solution:** Close and reopen Git Bash. Windows needs to refresh environment variables.

### "Permission denied" when running scripts

**Solution:** Make scripts executable:
```bash
chmod +x setup.sh create_sfml_project.sh
```

### SFML build fails

**Solution:** Make sure you have a C++ compiler installed:
```bash
g++ --version  # Should show MinGW version
# OR
cl  # Should show Visual Studio version
```

If neither works, reinstall the compiler:
```powershell
# In PowerShell as Administrator
choco install mingw -y --force
```

### CMake can't find SFML

**Solution:** Delete the build folder and rebuild:
```bash
rm -rf _sfml/build _sfml/install _sfml/SFML_DIR.txt
./setup.sh
```

### "vcvarsall.bat" not found (Visual Studio)

**Solution:** Either:
1. Use MinGW instead (easier): `choco install mingw -y`
2. Or run from "Developer Command Prompt for VS" instead of Git Bash

---

## What Gets Installed

### Disk Space Required

- **MinGW (Lightweight):** ~500 MB
- **Visual Studio Build Tools (Full):** ~7 GB
- **SFML (built locally):** ~100 MB
- **Each project:** ~5-10 MB

### Installation Locations

- **Chocolatey:** `C:\ProgramData\chocolatey\`
- **Git:** `C:\Program Files\Git\`
- **CMake:** `C:\Program Files\CMake\`
- **MinGW:** `C:\ProgramData\chocolatey\lib\mingw\tools\install\mingw64\`
- **Visual Studio:** `C:\Program Files\Microsoft Visual Studio\`
- **SFML:** `CPP-Projects\_sfml\install\`

---

## Uninstalling

To remove all installed tools:

```powershell
# In PowerShell as Administrator
choco uninstall git cmake ninja mingw -y
# Or if you installed Visual Studio:
choco uninstall visualstudio2022buildtools -y
```

To remove Chocolatey itself:
```powershell
Remove-Item C:\ProgramData\chocolatey -Recurse -Force
```

---

## Alternative: Use WSL (Windows Subsystem for Linux)

If you prefer a Linux-like environment:

1. **Install WSL:**
   ```powershell
   wsl --install
   ```
2. **Restart your PC**
3. **Open Ubuntu (from Start Menu)**
4. **Install tools:**
   ```bash
   sudo apt update
   sudo apt install git cmake build-essential -y
   ```
5. **Clone and build:**
   ```bash
   git clone https://github.com/hutizaki/CPP-Projects.git
   cd CPP-Projects
   ./setup.sh
   ```

---

## Next Steps

Once everything is installed:

1. **Learn the workflow:**
   - Read `README.md` for full documentation
   - Read `WORKFLOW_TEST.md` for detailed testing guide

2. **Create projects:**
   ```bash
   ./create_sfml_project.sh
   ```

3. **Run projects:**
   ```bash
   cd "Project Name"
   ./runApp.sh
   ```

4. **Verify everything works:**
   ```bash
   ./verify_setup.sh
   ```

---

## System Requirements

- **OS:** Windows 10 or Windows 11
- **RAM:** 4 GB minimum (8 GB recommended)
- **Disk Space:** 1-8 GB (depending on compiler choice)
- **Internet:** Required for downloading tools and SFML

---

## Support

If you encounter issues:

1. **Check the error message** - The updated `setup.sh` now shows actual errors
2. **Run verification:** `./verify_setup.sh`
3. **Check documentation:** `README.md`, `WORKFLOW_TEST.md`, `CHANGES_SUMMARY.md`
4. **Try manual installation** (Option 2 above)

---

## Summary

**Absolute beginner on Windows?**
1. Download ZIP from GitHub
2. Right-click `WINDOWS_INSTALL.bat` → Run as Administrator
3. Choose MinGW (option 1)
4. Open Git Bash
5. Run `./setup.sh`
6. Start coding! 🚀

**That's it!** No prior knowledge needed.

