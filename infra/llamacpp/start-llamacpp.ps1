# start-llamacpp.ps1 — llama-server on dev box (GTX 1080 Ti, 11GB VRAM)
#
# Usage:
#   .\start-llamacpp.ps1                    # Start with defaults
#   .\start-llamacpp.ps1 -Model "path.gguf" # Custom model
#   .\start-llamacpp.ps1 -Stop              # Stop running server
#   .\start-llamacpp.ps1 -Status            # Check health
#
# Default port: 8080 (matches LLAMACPP_URL default in HoloLoom + Femtoclaw)

param(
    [string]$Model = "$env:USERPROFILE\models\qwen3.5-9b-q4_k_m.gguf",
    [string]$LlamaCppDir = "$env:USERPROFILE\llama.cpp",
    [int]$Port = 8080,
    [int]$CtxSize = 16384,
    [int]$Parallel = 2,
    [switch]$Stop,
    [switch]$Status
)

$ServerExe = Join-Path $LlamaCppDir "build\bin\Release\llama-server.exe"
if (-not (Test-Path $ServerExe)) {
    $ServerExe = Join-Path $LlamaCppDir "build\bin\llama-server.exe"
}
$PidFile = Join-Path $LlamaCppDir ".llamacpp-devbox.pid"
$LogFile = Join-Path $LlamaCppDir "llamacpp-devbox.log"

function Get-ServerProcess {
    if (Test-Path $PidFile) {
        $pid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($pid) {
            return Get-Process -Id $pid -ErrorAction SilentlyContinue
        }
    }
    return $null
}

function Stop-Server {
    $proc = Get-ServerProcess
    if ($proc) {
        Write-Host "Stopping llama-server (PID $($proc.Id))..."
        Stop-Process -Id $proc.Id -Force
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        Write-Host "Stopped."
    } else {
        Write-Host "No running llama-server found."
    }
}

function Get-Health {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:${Port}/health" -TimeoutSec 5
        Write-Host "Status: $($resp.status)"
        if ($resp.slots_idle) { Write-Host "Slots idle: $($resp.slots_idle)" }
        if ($resp.slots_processing) { Write-Host "Slots processing: $($resp.slots_processing)" }
    } catch {
        Write-Host "llama-server unreachable on port $Port"
    }
}

if ($Stop) { Stop-Server; return }
if ($Status) { Get-Health; return }

# Pre-flight checks
if (-not (Test-Path $ServerExe)) {
    Write-Host "ERROR: llama-server not found at $ServerExe"
    Write-Host "Build it: cd $LlamaCppDir && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"
    return
}
if (-not (Test-Path $Model)) {
    Write-Host "ERROR: Model not found at $Model"
    Write-Host "Download a GGUF model and place it there, or pass -Model path"
    return
}

# Stop existing if running
$existing = Get-ServerProcess
if ($existing) {
    Write-Host "Stopping existing server (PID $($existing.Id))..."
    Stop-Process -Id $existing.Id -Force
    Start-Sleep -Seconds 2
}

# Launch
Write-Host "Starting llama-server..."
Write-Host "  Model:    $Model"
Write-Host "  Port:     $Port"
Write-Host "  Context:  $CtxSize"
Write-Host "  Parallel: $Parallel"
Write-Host "  Log:      $LogFile"
Write-Host ""

$args = @(
    "--host", "0.0.0.0",
    "--port", $Port,
    "--model", $Model,
    "--ctx-size", $CtxSize,
    "--n-gpu-layers", "99",
    "--flash-attn",
    "--parallel", $Parallel,
    "--cont-batching",
    "--metrics"
)

$proc = Start-Process -FilePath $ServerExe -ArgumentList $args `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError "$LogFile.err" `
    -PassThru -WindowStyle Hidden

$proc.Id | Out-File $PidFile -Force
Write-Host "Started (PID $($proc.Id))"
Write-Host ""

# Wait for health
Write-Host "Waiting for server to be ready..."
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:${Port}/health" -TimeoutSec 3
        if ($resp.status -eq "ok") {
            Write-Host "llama-server ready on port $Port"
            return
        }
    } catch { }
    Write-Host "  ..." -NoNewline
}
Write-Host ""
Write-Host "WARNING: Server started but health check didn't pass within 60s. Check $LogFile"
