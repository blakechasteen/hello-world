# COZ Daily Brief Automation - Windows Deployment Script
# PowerShell script to deploy COZ automation using Windows Task Scheduler

param(
    [string]$ScheduleTime = "09:00",
    [switch]$Uninstall
)

$ScriptPath = "c:\Users\blake\OneDrive\Documents\mythRL"
$PythonExe = "$ScriptPath\.venv\Scripts\python.exe"
$AutomationScript = "$ScriptPath\elle\coz\daily_brief_automation.py"
$TaskName = "COZ-Daily-Brief-Automation"

if ($Uninstall) {
    Write-Host "Uninstalling COZ Daily Brief automation..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "✓ Uninstalled successfully"
    exit 0
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "COZ Daily Brief Automation - Windows Deployment" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Verify files exist
Write-Host "[1/5] Verifying files..." -ForegroundColor Yellow
if (!(Test-Path $PythonExe)) {
    Write-Host "✗ Python not found: $PythonExe" -ForegroundColor Red
    exit 1
}
if (!(Test-Path $AutomationScript)) {
    Write-Host "✗ Automation script not found: $AutomationScript" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Files verified" -ForegroundColor Green

# Check dependencies
Write-Host "[2/5] Checking dependencies..." -ForegroundColor Yellow
$scheduleInstalled = & $PythonExe -c "import schedule; print('installed')" 2>$null
if ($scheduleInstalled -ne "installed") {
    Write-Host "✗ 'schedule' module not installed" -ForegroundColor Red
    Write-Host "  Run: .venv\Scripts\pip.exe install schedule" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Create output directory
Write-Host "[3/5] Creating output directory..." -ForegroundColor Yellow
$OutputDir = "$ScriptPath\daily_briefs"
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
Write-Host "✓ Output directory ready: $OutputDir" -ForegroundColor Green

# Create Task Scheduler task
Write-Host "[4/5] Creating scheduled task..." -ForegroundColor Yellow

# Remove existing task if it exists
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create action
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "elle\coz\daily_brief_automation.py --schedule `"$ScheduleTime`"" `
    -WorkingDirectory $ScriptPath

# Create trigger (daily at startup, script will handle scheduling)
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Register task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "COZ Daily Brief Automation - generates daily briefs at $ScheduleTime" `
    -User $env:USERNAME `
    -RunLevel Highest

Write-Host "✓ Scheduled task created: $TaskName" -ForegroundColor Green

# Start task
Write-Host "[5/5] Starting task..." -ForegroundColor Yellow
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2

# Verify task is running
$Task = Get-ScheduledTask -TaskName $TaskName
if ($Task.State -eq "Running" -or $Task.State -eq "Ready") {
    Write-Host "✓ Task started successfully" -ForegroundColor Green
} else {
    Write-Host "⚠ Task state: $($Task.State)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Task Name: $TaskName"
Write-Host "  Schedule: Daily at $ScheduleTime"
Write-Host "  Output Directory: $OutputDir"
Write-Host "  Working Directory: $ScriptPath"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Verify task is running: Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  2. Check logs: Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host "  3. Run health check: python elle\coz\health_check.py"
Write-Host "  4. Wait for first brief at $ScheduleTime tomorrow"
Write-Host ""
Write-Host "Management Commands:" -ForegroundColor Cyan
Write-Host "  Start:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Stop:   Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Remove: .\deployment\windows\deploy_windows.ps1 -Uninstall"
Write-Host ""
