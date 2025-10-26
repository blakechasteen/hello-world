# Promptly PowerShell Wrapper
# Simplified command interface for Promptly + UltraPrompt

param(
    [Parameter(Position=0)]
    [string]$Command,
    
    [Parameter(Position=1)]
    [string]$Backend,
    
    [Parameter(Position=2)]
    [string]$Request,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Ultraprompt {
    param(
        [string]$Backend,
        [string]$Request,
        [string[]]$Args
    )
    
    $pythonArgs = @("$ScriptDir\promptly_cli.py", "ultraprompt", $Backend, $Request)
    
    if ($Args) {
        $pythonArgs += $Args
    }
    
    & python @pythonArgs
}

# Main logic
if ($Command -eq "ultraprompt") {
    if (-not $Backend -or -not $Request) {
        Write-Host "Usage: .\Promptly.ps1 ultraprompt [ollama|llm] `"your request`" [--execute]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host '  .\Promptly.ps1 ultraprompt ollama "write a fibonacci function"'
        Write-Host '  .\Promptly.ps1 ultraprompt ollama "explain quantum computing" --execute'
        Write-Host ""
        exit 1
    }
    
    Invoke-Ultraprompt -Backend $Backend -Request $Request -Args $RemainingArgs
}
else {
    # Pass through to regular promptly commands
    $allArgs = @($Command, $Backend, $Request) + $RemainingArgs
    $allArgs = $allArgs | Where-Object { $_ -ne "" -and $null -ne $_ }
    
    if ($allArgs.Count -eq 0) {
        & python "$ScriptDir\promptly.py" --help
    }
    else {
        & python "$ScriptDir\promptly.py" @allArgs
    }
}
