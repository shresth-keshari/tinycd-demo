# One-step setup and launch (Windows PowerShell):   .\run.ps1
# Creates a virtual environment, installs dependencies, downloads the TinyCD weights from the
# authors' repository (checksum verified), runs the sample pairs, then opens the live demo.
param([switch]$NoServe)   # -NoServe: set up and run the samples, but do not start the web demo
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -r requirements.txt

& $py -m tinycd                 # downloads weights on first run, writes outputs\samples.png
Write-Host "`nResults image: outputs\samples.png"
if (-not $NoServe) {
    Write-Host "Opening the live demo (Ctrl+C to stop)..."
    & $py scripts\serve.py
}
