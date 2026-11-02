# One-step setup and launch (Windows PowerShell):   .\run.ps1
# Creates a virtual environment, installs dependencies, downloads the TinyCD weights from the
# authors' repository (checksum verified), runs the sample pairs, then opens the live demo.
param([switch]$NoServe)   # -NoServe: set up and run the samples, but do not start the web demo
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment (is Python 3.9+ installed?)" }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installing the requirements failed" }

& $py -m tinycd                 # downloads weights on first run, writes outputs\samples.png
if ($LASTEXITCODE -ne 0) { throw "Running TinyCD failed (a weight download error usually means no internet access to github.com)" }
Write-Host "`nResults image: outputs\samples.png"
if (-not $NoServe) {
    Write-Host "Opening the live demo (Ctrl+C to stop)..."
    & $py scripts\serve.py
}
