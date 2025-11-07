#!/usr/bin/env pwsh
# Launch Consciousness Stack Web UI

Write-Host "`n🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠`n" -ForegroundColor Cyan
Write-Host "  CONSCIOUSNESS STACK WEB UI" -ForegroundColor Green
Write-Host "`n🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠`n" -ForegroundColor Cyan

Write-Host "Starting Gradio server..." -ForegroundColor Yellow
Write-Host "UI will be available at: http://localhost:7860`n" -ForegroundColor White

$env:PYTHONPATH = "."
python ui/consciousness_ui.py
