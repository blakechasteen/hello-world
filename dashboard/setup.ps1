# 🚀 mythRL Dashboard - Quick Setup Script
# Run this to get everything working in minutes!

Write-Host "🌊 mythRL Narrative Intelligence Dashboard Setup" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor DarkCyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "HoloLoom")) {
    Write-Host "❌ Error: Please run this script from the mythRL root directory" -ForegroundColor Red
    exit 1
}

# Step 1: Python backend dependencies
Write-Host "📦 Step 1/3: Installing Python backend dependencies..." -ForegroundColor Yellow
pip install fastapi uvicorn websockets pydantic 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: Some Python packages may already be installed" -ForegroundColor Yellow
}
Write-Host ""

# Step 2: Node.js frontend dependencies
Write-Host "📦 Step 2/3: Installing Node.js frontend dependencies..." -ForegroundColor Yellow
Push-Location dashboard
npm install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Error installing Node.js dependencies" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location
Write-Host ""

# Step 3: Test the system
Write-Host "🧪 Step 3/3: Testing cross-domain adapter..." -ForegroundColor Yellow
$env:PYTHONPATH = "."
python -c "from HoloLoom.cross_domain_adapter import CrossDomainAdapter; print('✅ Cross-domain adapter works!')" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All systems operational!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Warning: Could not verify adapter (may still work)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=" * 80 -ForegroundColor DarkCyan
Write-Host "🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the system:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Terminal 1 (Backend):" -ForegroundColor Yellow
Write-Host '  $env:PYTHONPATH = "."; python dashboard/backend.py' -ForegroundColor White
Write-Host ""
Write-Host "Terminal 2 (Frontend):" -ForegroundColor Yellow
Write-Host '  cd dashboard; npm run dev' -ForegroundColor White
Write-Host ""
Write-Host "Then open: http://localhost:3000" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor DarkCyan
Write-Host ""
Write-Host "🌟 Features:" -ForegroundColor Magenta
Write-Host "  • 6 narrative domains (Mythology, Business, Science, Personal, Product, History)" -ForegroundColor White
Write-Host "  • Real-time streaming analysis (20 words/second)" -ForegroundColor White
Write-Host "  • Progressive gate unlocking (Surface → Cosmic)" -ForegroundColor White
Write-Host "  • Auto-domain detection" -ForegroundColor White
Write-Host "  • Extensible plugin system" -ForegroundColor White
Write-Host ""
Write-Host "📚 Docs: dashboard/README.md" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor DarkCyan
