# Find Visual Studio
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath

if (-not $vsPath) {
    Write-Host "ERROR: Visual Studio 2022 not found!" -ForegroundColor Red
    exit 1
}

# Import VS environment
$vsDevCmd = Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Import-Module $vsDevCmd
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64" | Out-Null

# Compile
nvcc MNIST_Phase3.cu -o MNIST_Phase3.exe -arch=sm_89 --use-local-env

# Run
if (Test-Path "MNIST_Phase3.exe") {
    Write-Host ""
    & .\MNIST_Phase3.exe
} else {
    Write-Host ""
    Write-Host "Compilation failed" -ForegroundColor Red
}
