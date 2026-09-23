# Build REGEN: the .exe (PyInstaller), then the Setup.exe installer (Inno Setup, if installed)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}

Write-Host "Creating icon..."
& $py tools\make_icon.py

Write-Host "Running PyInstaller..."
& $py -m PyInstaller --noconfirm --clean --windowed `
    --name "REGEN" `
    --icon assets\icon.ico `
    --add-data "assets;assets" `
    --exclude-module tkinter `
    --exclude-module PySide6.QtWebEngineCore `
    --exclude-module PySide6.QtWebEngineWidgets `
    --exclude-module PySide6.Qt3DCore `
    --exclude-module PySide6.QtQuick `
    --exclude-module PySide6.QtQml `
    --exclude-module PySide6.QtMultimedia `
    main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
# build\ only holds temporary PyInstaller files; the .exe there cannot be run
Remove-Item -Recurse -Force build, "REGEN.spec" -ErrorAction SilentlyContinue
Write-Host "Application ready: dist\REGEN\REGEN.exe"

$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($iscc) {
    Write-Host "Building the installer with Inno Setup..."
    & $iscc installer.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
    Write-Host "Installer ready in installer_output\"
} else {
    Write-Host "Inno Setup 6 not found; skipping the installer."
    Write-Host "Download it from https://jrsoftware.org/isdl.php and run build.ps1 again."
}
