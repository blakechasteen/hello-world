# PowerShell script to add Claude to PATH permanently

$claudePath = "C:\Users\blake\.local\bin"

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if already in PATH
if ($currentPath -notlike "*$claudePath*") {
    Write-Host "Adding $claudePath to user PATH..." -ForegroundColor Green

    # Add to PATH
    $newPath = $currentPath + ";" + $claudePath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")

    Write-Host "✓ PATH updated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: Close and reopen PowerShell for changes to take effect." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To test in this session, run:" -ForegroundColor Cyan
    Write-Host '  $env:Path = [Environment]::GetEnvironmentVariable("Path", "User")' -ForegroundColor White
    Write-Host "  claude --version" -ForegroundColor White
} else {
    Write-Host "✓ $claudePath is already in PATH" -ForegroundColor Green
    Write-Host ""
    Write-Host "If 'claude' still doesn't work, try:" -ForegroundColor Yellow
    Write-Host "1. Close and reopen PowerShell" -ForegroundColor White
    Write-Host "2. Or reload PATH in current session:" -ForegroundColor White
    Write-Host '   $env:Path = [Environment]::GetEnvironmentVariable("Path", "User")' -ForegroundColor White
}
