# Windows Setup Script for CPP Projects
# Run this FIRST on a fresh Windows PC
# This installs all required tools: Git, CMake, C++ compiler, and sets up the workspace
#
# Usage (Run PowerShell as Administrator):
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\windows_setup.ps1

Write-Host "=========================================="
Write-Host "  CPP Projects - Windows Setup"
Write-Host "=========================================="
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Yellow
    Write-Host "  1. Right-click PowerShell" -ForegroundColor Yellow
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "  3. Run: Set-ExecutionPolicy Bypass -Scope Process -Force" -ForegroundColor Yellow
    Write-Host "  4. Run: .\windows_setup.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "Running as Administrator... OK" -ForegroundColor Green
Write-Host ""

# Function to check if a command exists
function Test-Command {
    param($Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

# Step 1: Install Chocolatey (Windows package manager)
Write-Host "=========================================="
Write-Host "Step 1: Installing Chocolatey (Package Manager)"
Write-Host "=========================================="
Write-Host ""

if (Test-Command choco) {
    Write-Host "[OK] Chocolatey already installed" -ForegroundColor Green
} else {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Chocolatey installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "[ERROR] Chocolatey installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 2: Install Git
Write-Host "=========================================="
Write-Host "Step 2: Installing Git"
Write-Host "=========================================="
Write-Host ""

if (Test-Command git) {
    $gitVersion = git --version
    Write-Host "[OK] Git already installed: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "Installing Git..." -ForegroundColor Yellow
    choco install git -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Git installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "[ERROR] Git installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 3: Install CMake
Write-Host "=========================================="
Write-Host "Step 3: Installing CMake"
Write-Host "=========================================="
Write-Host ""

if (Test-Command cmake) {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "[OK] CMake already installed: $cmakeVersion" -ForegroundColor Green
} else {
    Write-Host "Installing CMake..." -ForegroundColor Yellow
    choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] CMake installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "[ERROR] CMake installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 4: Install C++ Compiler (Visual Studio Build Tools or MinGW)
Write-Host "=========================================="
Write-Host "Step 4: Installing C++ Compiler"
Write-Host "=========================================="
Write-Host ""

# Check if Visual Studio or MinGW is already installed
$hasVS = Test-Path "C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
$hasMinGW = Test-Command g++

if ($hasVS) {
    Write-Host "[OK] Visual Studio C++ compiler already installed" -ForegroundColor Green
} elseif ($hasMinGW) {
    $gccVersion = g++ --version | Select-Object -First 1
    Write-Host "[OK] MinGW C++ compiler already installed: $gccVersion" -ForegroundColor Green
} else {
    Write-Host "No C++ compiler found. Choose installation method:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. MinGW (Lightweight, 500MB, faster install)" -ForegroundColor Cyan
    Write-Host "2. Visual Studio Build Tools (Full-featured, 7GB, slower install)" -ForegroundColor Cyan
    Write-Host ""
    
    $choice = Read-Host "Enter choice (1 or 2)"
    
    if ($choice -eq "1") {
        Write-Host ""
        Write-Host "Installing MinGW..." -ForegroundColor Yellow
        choco install mingw -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] MinGW installed successfully" -ForegroundColor Green
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        } else {
            Write-Host "[ERROR] MinGW installation failed!" -ForegroundColor Red
            exit 1
        }
    } elseif ($choice -eq "2") {
        Write-Host ""
        Write-Host "Installing Visual Studio Build Tools (this will take 10-30 minutes)..." -ForegroundColor Yellow
        choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --includeOptional --passive" -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Visual Studio Build Tools installed successfully" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Visual Studio Build Tools installation failed!" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[ERROR] Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 5: Install Ninja (faster build system)
Write-Host "=========================================="
Write-Host "Step 5: Installing Ninja Build System"
Write-Host "=========================================="
Write-Host ""

if (Test-Command ninja) {
    $ninjaVersion = ninja --version
    Write-Host "[OK] Ninja already installed: $ninjaVersion" -ForegroundColor Green
} else {
    Write-Host "Installing Ninja..." -ForegroundColor Yellow
    choco install ninja -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Ninja installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "[ERROR] Ninja installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 6: Verify installations
Write-Host "=========================================="
Write-Host "Step 6: Verifying Installations"
Write-Host "=========================================="
Write-Host ""

# Refresh PATH one more time
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$allGood = $true

# Check Git
if (Test-Command git) {
    $gitVersion = git --version
    Write-Host "[OK] Git: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Git: Not found" -ForegroundColor Red
    $allGood = $false
}

# Check CMake
if (Test-Command cmake) {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "[OK] CMake: $cmakeVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] CMake: Not found" -ForegroundColor Red
    $allGood = $false
}

# Check C++ Compiler
if (Test-Command g++) {
    $gccVersion = g++ --version | Select-Object -First 1
    Write-Host "[OK] C++ Compiler: $gccVersion" -ForegroundColor Green
} elseif (Test-Command cl) {
    Write-Host "[OK] C++ Compiler: Visual Studio (cl.exe)" -ForegroundColor Green
} else {
    # Check if Visual Studio is installed (even if not in PATH)
    $vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
    $vsPath2 = "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
    if ((Test-Path $vsPath) -or (Test-Path $vsPath2)) {
        Write-Host "[OK] C++ Compiler: Visual Studio Build Tools (installed, not in PATH)" -ForegroundColor Green
        Write-Host "     Note: Use 'Developer Command Prompt' or CMake will configure it automatically" -ForegroundColor Cyan
    } else {
        Write-Host "[ERROR] C++ Compiler: Not found" -ForegroundColor Red
        $allGood = $false
    }
}

# Check Ninja
if (Test-Command ninja) {
    $ninjaVersion = ninja --version
    Write-Host "[OK] Ninja: $ninjaVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Ninja: Not found" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

if (-not $allGood) {
    Write-Host "[WARNING] Some required tools are missing!" -ForegroundColor Yellow
    Write-Host "  Please install missing tools and run this script again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Step 7: Instructions for next steps
Write-Host "=========================================="
Write-Host "  Setup Complete!"
Write-Host "=========================================="
Write-Host ""
Write-Host "All required tools are installed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open Git Bash in this directory:" -ForegroundColor Yellow
Write-Host "   - Search for 'Git Bash' in Start Menu" -ForegroundColor White
Write-Host "   - Navigate to: $PWD" -ForegroundColor White
Write-Host "   OR press Y below to open Git Bash now" -ForegroundColor White
Write-Host ""
Write-Host "2. Run the project setup script:" -ForegroundColor Yellow
Write-Host "   ./setup.sh" -ForegroundColor White
Write-Host ""
Write-Host "3. Create your first SFML project:" -ForegroundColor Yellow
Write-Host "   ./create_sfml_project.sh" -ForegroundColor White
Write-Host ""
Write-Host "4. Build and run your project:" -ForegroundColor Yellow
Write-Host "   cd ""Your Project Name""" -ForegroundColor White
Write-Host "   ./runApp.sh" -ForegroundColor White
Write-Host ""
Write-Host "=========================================="
Write-Host ""

# Offer to open Git Bash
$openGitBash = Read-Host "Open Git Bash in this directory now? (y/n)"
if ($openGitBash -eq "y" -or $openGitBash -eq "Y") {
    if (Test-Path "C:\Program Files\Git\git-bash.exe") {
        Start-Process "C:\Program Files\Git\git-bash.exe" -WorkingDirectory $PWD
    } else {
        Write-Host "Git Bash not found at default location. Please open it manually." -ForegroundColor Yellow
    }
}

