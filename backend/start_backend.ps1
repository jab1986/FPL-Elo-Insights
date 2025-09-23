# PowerShell script to start the backend server
# This ensures we're in the correct directory for imports to work

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to the backend directory (where this script is located)
Set-Location -Path $scriptDir
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green

# Verify main.py exists
if (Test-Path "main.py") {
    Write-Host "Found main.py, starting server..." -ForegroundColor Yellow
    Write-Host "Backend server will be available at: http://localhost:8001" -ForegroundColor Cyan
    Write-Host "API endpoints available at: http://localhost:8001/api" -ForegroundColor Cyan
    Write-Host "Health check: http://localhost:8001/health" -ForegroundColor Cyan
    Write-Host ""
    
    # Start the uvicorn server
    python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
} else {
    Write-Error "main.py not found in current directory: $(Get-Location)"
    Write-Host "Available files:" -ForegroundColor Yellow
    Get-ChildItem -Name "*.py"
    Write-Host ""
    Write-Host "Please ensure you're running this script from the backend directory."
    Read-Host "Press Enter to exit"
}
