# PowerShell script to start the frontend development server
# This ensures we're in the correct directory and provides helpful information

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to the frontend directory (where this script is located)
Set-Location -Path $scriptDir
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green

# Verify package.json exists
if (Test-Path "package.json") {
    Write-Host "Found package.json, starting development server..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Frontend development server starting..." -ForegroundColor Cyan
    Write-Host "Server will be available at: http://localhost:5173" -ForegroundColor Cyan
    Write-Host "Note: If you experience connection issues on Windows, the DNS fix is already applied in vite.config.ts" -ForegroundColor Green
    Write-Host ""
    
    # Start the development server
    npm run dev
} else {
    Write-Error "package.json not found in current directory: $(Get-Location)"
    Write-Host "Available files:" -ForegroundColor Yellow
    Get-ChildItem -Name "*.json"
    Write-Host ""
    Write-Host "Please ensure you're running this script from the frontend directory."
    Read-Host "Press Enter to exit"
}
