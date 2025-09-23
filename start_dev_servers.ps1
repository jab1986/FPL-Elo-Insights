# Master startup script for FPL-Elo-Insights
# Starts both backend and frontend servers in separate windows

Write-Host "FPL-Elo-Insights Development Server Startup" -ForegroundColor Magenta
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host ""

# Get current directory
$rootDir = Get-Location
Write-Host "Project root: $rootDir" -ForegroundColor Green

# Check if we're in the correct directory
if (-not (Test-Path "backend\main.py") -or -not (Test-Path "frontend\package.json")) {
    Write-Error "Please run this script from the FPL-Elo-Insights root directory"
    Write-Host "Expected structure:"
    Write-Host "  - backend/main.py"
    Write-Host "  - frontend/package.json"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting backend server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$rootDir\backend\start_backend.ps1'"

Write-Host "Waiting 3 seconds for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Starting frontend server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$rootDir\frontend\start_frontend.ps1'"

Write-Host ""
Write-Host "Both servers are starting in separate windows..." -ForegroundColor Green
Write-Host ""
Write-Host "Backend API will be available at: http://localhost:8001" -ForegroundColor Cyan
Write-Host "Frontend will be available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "Health check command: curl http://localhost:8001/health" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Enter to exit this script (servers will continue running)"
Read-Host
