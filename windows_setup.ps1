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
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "Step 1: Installing Chocolatey (Package Manager)"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

if (Test-Command choco) {
    Write-Host "✓ Chocolatey already installed" -ForegroundColor Green
} else {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Chocolatey installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "✗ Chocolatey installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 2: Install Git
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "Step 2: Installing Git"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

if (Test-Command git) {
    $gitVersion = git --version
    Write-Host "✓ Git already installed: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "Installing Git..." -ForegroundColor Yellow
    choco install git -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Git installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "✗ Git installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 3: Install CMake
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "Step 3: Installing CMake"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

if (Test-Command cmake) {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "✓ CMake already installed: $cmakeVersion" -ForegroundColor Green
} else {
    Write-Host "Installing CMake..." -ForegroundColor Yellow
    choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ CMake installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "✗ CMake installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 4: Install C++ Compiler (Visual Studio Build Tools or MinGW)
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "Step 4: Installing C++ Compiler"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

# Check if Visual Studio or MinGW is already installed
$hasVS = Test-Path "C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
$hasMinGW = Test-Command g++

if ($hasVS) {
    Write-Host "✓ Visual Studio C++ compiler already installed" -ForegroundColor Green
} elseif ($hasMinGW) {
    $gccVersion = g++ --version | Select-Object -First 1
    Write-Host "✓ MinGW C++ compiler already installed: $gccVersion" -ForegroundColor Green
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
            Write-Host "✓ MinGW installed successfully" -ForegroundColor Green
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        } else {
            Write-Host "✗ MinGW installation failed!" -ForegroundColor Red
            exit 1
        }
    } elseif ($choice -eq "2") {
        Write-Host ""
        Write-Host "Installing Visual Studio Build Tools (this will take 10-30 minutes)..." -ForegroundColor Yellow
        choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --includeOptional --passive" -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Visual Studio Build Tools installed successfully" -ForegroundColor Green
        } else {
            Write-Host "✗ Visual Studio Build Tools installation failed!" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "✗ Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 5: Install Ninja (faster build system)
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "Step 5: Installing Ninja Build System"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

if (Test-Command ninja) {
    $ninjaVersion = ninja --version
    Write-Host "✓ Ninja already installed: $ninjaVersion" -ForegroundColor Green
} else {
    Write-Host "Installing Ninja..." -ForegroundColor Yellow
    choco install ninja -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Ninja installed successfully" -ForegroundColor Green
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "✗ Ninja installation failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 6: Verify installations
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host "Step 6: Verifying Installations"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

# Refresh PATH one more time
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$allGood = $true

# Check Git
if (Test-Command git) {
    $gitVersion = git --version
    Write-Host "✓ Git: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Git: Not found" -ForegroundColor Red
    $allGood = $false
}

# Check CMake
if (Test-Command cmake) {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "✓ CMake: $cmakeVersion" -ForegroundColor Green
} else {
    Write-Host "✗ CMake: Not found" -ForegroundColor Red
    $allGood = $false
}

# Check C++ Compiler
if (Test-Command g++) {
    $gccVersion = g++ --version | Select-Object -First 1
    Write-Host "✓ C++ Compiler: $gccVersion" -ForegroundColor Green
} elseif (Test-Command cl) {
    Write-Host "✓ C++ Compiler: Visual Studio (cl.exe)" -ForegroundColor Green
} else {
    Write-Host "✗ C++ Compiler: Not found" -ForegroundColor Red
    $allGood = $false
}

# Check Ninja
if (Test-Command ninja) {
    $ninjaVersion = ninja --version
    Write-Host "✓ Ninja: $ninjaVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Ninja: Not found" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

if (-not $allGood) {
    Write-Host "⚠ Some tools are not in PATH. Please:" -ForegroundColor Yellow
    Write-Host "  1. Close this PowerShell window" -ForegroundColor Yellow
    Write-Host "  2. Open a NEW PowerShell window" -ForegroundColor Yellow
    Write-Host "  3. Run: .\windows_setup.ps1 again to verify" -ForegroundColor Yellow
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
Write-Host "1. Close this PowerShell window and open Git Bash" -ForegroundColor Yellow
Write-Host "   (Search for 'Git Bash' in Start Menu)" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Clone the repository:" -ForegroundColor Yellow
Write-Host "   git clone https://github.com/hutizaki/CPP-Projects.git" -ForegroundColor White
Write-Host "   cd CPP-Projects" -ForegroundColor White
Write-Host ""
Write-Host "3. Run the setup script:" -ForegroundColor Yellow
Write-Host "   ./setup.sh" -ForegroundColor White
Write-Host ""
Write-Host "4. Create a new project:" -ForegroundColor Yellow
Write-Host "   ./create_sfml_project.sh" -ForegroundColor White
Write-Host ""
Write-Host "5. Run your project:" -ForegroundColor Yellow
Write-Host "   cd `"Your Project Name`"" -ForegroundColor White
Write-Host "   ./runApp.sh" -ForegroundColor White
Write-Host ""
Write-Host "=========================================="
Write-Host ""

# Offer to open Git Bash
$openGitBash = Read-Host "Open Git Bash now? (y/n)"
if ($openGitBash -eq "y" -or $openGitBash -eq "Y") {
    Start-Process "C:\Program Files\Git\git-bash.exe"
}

