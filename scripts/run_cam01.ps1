# Offline cam-01 spike helper (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = $null
foreach ($candidate in @("python", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$ver" -match "Python 3") {
            $Python = $candidate
            break
        }
    } catch { }
}

if (-not $Python) {
    Write-Error "Python 3 not found. Install from https://www.python.org/downloads/ and re-open the terminal."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m idle_agent --source file --camera-id cam-01 @args
