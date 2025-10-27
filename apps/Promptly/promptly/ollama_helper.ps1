# Ollama Helper Script
# Use this until you restart PowerShell and Ollama is in your PATH

$OllamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

function Test-OllamaInstalled {
    Test-Path $OllamaPath
}

function Invoke-Ollama {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Arguments
    )
    
    if (Test-OllamaInstalled) {
        & $OllamaPath @Arguments
    } else {
        Write-Error "Ollama not found at $OllamaPath"
    }
}

# Create alias
Set-Alias -Name ollama -Value Invoke-Ollama -Scope Global

Write-Host "✅ Ollama helper loaded!" -ForegroundColor Green
Write-Host "You can now use 'ollama' commands in this session." -ForegroundColor Cyan
Write-Host ""
Write-Host "Examples:" -ForegroundColor Yellow
Write-Host "  ollama --version"
Write-Host "  ollama list"
Write-Host "  ollama run llama3.2:3b 'Hello, world!'"
Write-Host ""
Write-Host "💡 To make this permanent, restart your terminal." -ForegroundColor Gray
