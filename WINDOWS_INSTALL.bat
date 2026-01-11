@echo off
REM Windows Installation Script for CPP Projects
REM Right-click this file and select "Run as Administrator"

echo ==========================================
echo   CPP Projects - Windows Installer
echo ==========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo.
    echo To run as Administrator:
    echo   1. Right-click this file: WINDOWS_INSTALL.bat
    echo   2. Select "Run as Administrator"
    echo.
    pause
    exit /b 1
)

echo Running as Administrator... OK
echo.
echo This will install:
echo   - Chocolatey (Package Manager)
echo   - Git (Version Control)
echo   - CMake (Build System)
echo   - C++ Compiler (MinGW or Visual Studio)
echo   - Ninja (Fast Build Tool)
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0windows_setup.ps1"

if %errorLevel% neq 0 (
    echo.
    echo Installation failed! Check the errors above.
    pause
    exit /b 1
)

echo.
echo Installation complete!
pause

