# FastAPI Server Startup Script for PowerShell
Write-Host "Starting FPL Insights FastAPI Server..." -ForegroundColor Green

# Set paths
$projectDir = "C:\Users\joebr\FPL-Elo-Insights\backend"
$venvActivate = "$projectDir\venv\Scripts\Activate.ps1"

# Change to project directory
Set-Location $projectDir

# Set PYTHONPATH
$env:PYTHONPATH = $projectDir

# Activate virtual environment
& $venvActivate

# Start the server
Write-Host "Starting uvicorn server..." -ForegroundColor Yellow
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload